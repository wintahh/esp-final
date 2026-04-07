import pymunk
import pymunk.pygame_util
import math
import json
import pygame
from hashlib import sha256
import random

BACKGROUND  = (180, 180, 180)
TARGET_FPS  = 60
MAX_SPEED   = 21
EVAL_STEPS  = 500
VIEW_STEPS  = TARGET_FPS * 9 # seconds of rendered simulation per generation
 
def generate_name(s): # stole this from https://github.com/carykh/jes/blob/main/utils.py
    salted = str(s) + str(random.randint(0,999999))
    _hex = sha256(salted.encode('utf-8')).hexdigest()
    result = int(_hex, 16)
    length_choices = [5,5,6,6,7]
    length_choice = result%5
    result = result//5
    
    letters = ["bcdfghjklmnprstvwxz","aeiouy"]
    name_len = length_choices[length_choice]
    name = ""
    for n in range(name_len):
        letter_type = n%len(letters)
        option_count = len(letters[letter_type])
        choice = result%option_count
        letter = letters[letter_type][choice]
        if n >= 2 and letter == "g" and name[n-2].lower() == "n":
            letter = "m"
        if n == 0:
            letter = letter.upper()
        name += letter
        result = result//option_count
    return name

class Creature:
    TORSO_MASS       = 2
    LEG_MASS         = 1
    MUSCLE_STIFFNESS = 120000
    MUSCLE_DAMPING   = 12000

    def __init__(self, space, position):
        self.space    = space
        self.position = list(position)
        self.filter   = pymunk.ShapeFilter(group=1)
        self.bodies   = {}
        self.shapes   = {}
        self.springs  = []

        with open("creatures/creature3.json") as f:
            data = json.load(f)

        self.create_bodies(data)
        self.create_joints(data)

    def create_bodies(self, data):
        for body_data in data["bodies"]:
            body_type = body_data["type"]
            if body_type == "torso":
                mass = self.TORSO_MASS
            elif body_type == "leg":
                mass = self.LEG_MASS
            else:
                print(f"Unknown body type '{body_type}' for '{body_data['name']}', defaulting to leg")
                mass = self.LEG_MASS

            size = body_data["size"]
            pos  = [a+b for a, b in zip(body_data["position"], self.position)]

            body = pymunk.Body(mass, pymunk.moment_for_box(mass, size))
            body.position = pymunk.Vec2d(*pos)
            body.angular_damping = 0.4

            shape = pymunk.Poly.create_box(body, size)
            shape.friction = 0.8
            shape.collision_type = 2
            shape.filter = self.filter

            self.space.add(body, shape)
            self.bodies[body_data["name"]] = body
            self.shapes[body_data["name"]] = shape

    def create_joints(self, data):
        for jd in data["joints"]:
            body1 = self.bodies[jd["body_a"]]
            body2 = self.bodies[jd["body_b"]]

            if jd["type"] == "pivot":
                anchor = [a + b for a, b in zip(jd["anchor"], self.position)]
                joint  = pymunk.PivotJoint(body1, body2, anchor)
                self.space.add(joint)

                if jd.get("actuated", False):
                    spring = pymunk.DampedRotarySpring(
                        body1, body2, 0,
                        self.MUSCLE_STIFFNESS,
                        self.MUSCLE_DAMPING
                    )
                    self.springs.append(spring)
                    self.space.add(spring)

            elif jd["type"] == "rotary_limit":
                joint = pymunk.RotaryLimitJoint(
                    body1, body2,
                    math.radians(jd["min_angle"]),
                    math.radians(jd["max_angle"])
                )
                self.space.add(joint)

            else:
                print(f"Unknown joint type '{jd['type']}' for '{jd['name']}'")

    def center_x(self):
        return sum(b.position.x for b in self.bodies.values()) / len(self.bodies)


class Game:
    def __init__(self, render=True):
        self.render = render
        self.sim_speed = 1
        self.width = 1000
        self.height = 600
        self.show_sliders = False
        self.settings_button_size = 48
        self.creatures = []
        self.species_names = {}
        if 'focused_index' not in locals():
            self.focused_index = 0
        if 'follow_best' not in locals():
            self.follow_best = False

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width, self.height), pygame.RESIZABLE
            )
            pygame.display.set_caption("Evolution Simulator")
            self.clock         = pygame.time.Clock()
            self.settings_icon = pygame.transform.scale(
                pygame.image.load("settings.png").convert_alpha(),
                (self.settings_button_size, self.settings_button_size)
            )
        else:
            self.screen = self.clock = self.settings_icon = None

        self.reset()

    # setup

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)
        self.draw_options  = pymunk.pygame_util.DrawOptions(self.screen) if self.render else None
        self.create_ground()

    def create_ground(self):
        ground = pymunk.Segment(self.space.static_body, (-1000, 550), (30000, 550), 5)
        ground.friction = 1.0
        ground.collision_type = 1
        self.space.add(ground)

    def spawn_creatures(self, count):
        self.creatures = [Creature(self.space, (0, 400)) for _ in range(count)]

    # phyics helpers

    def touches_ground(self, shape):
        for contact in self.space.shape_query(shape):
            if isinstance(contact.shape, pymunk.Segment) and contact.shape.body == self.space.static_body:
                return True
        return False

    def get_state(self, creature):
        vel_x = vel_y = 0.0
        per_body = []

        for name, body in creature.bodies.items():
            vel_x += body.velocity.x
            vel_y += body.velocity.y
            per_body += [
                math.sin(body.angle),
                body.angular_velocity / (2 * math.pi),
                float(self.touches_ground(creature.shapes[name])),
            ]

        n = len(creature.bodies)
        return [vel_x / n, vel_y / n] + per_body

    def apply_outputs(self, creature, outputs):
        for spring, value in zip(creature.springs, outputs):
            spring.rest_angle = value * math.pi * 3 / 4

    # drawing

    def set_camera(self, target_x):
        self.draw_options.transform = pymunk.Transform(
            tx=self.width  // 2 - target_x,
            ty=self.height - 600,
        )

    def draw_ground(self):
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Segment):
                a = self.draw_options.transform @ shape.a
                b = self.draw_options.transform @ shape.b
                pygame.draw.line(self.screen, (200, 200, 200), (a.x, a.y), (b.x, b.y), 3)

    def draw_creatures(self, focused_index, display_fitness):
        best_x = self.creatures[focused_index].center_x()
        max_dist = self.width / 2

        draw_order = sorted(range(len(self.creatures)), key=lambda i: -abs(self.creatures[i].center_x() - self.creatures[focused_index].center_x())) # distance from focused creature
        # i hate this sm ^
        for i in draw_order:
            creature = self.creatures[i]
            dist = abs(creature.center_x() - best_x)
            nd = min(1.0, dist / max_dist)
            shade = int(60 + 120 * nd)
            color = (30, 30, 30) if i == focused_index else (shade, shade, shade)

            for shape in creature.shapes.values():
                if isinstance(shape, pymunk.Poly):
                    body  = shape.body
                    verts = [body.position + v.rotated(body.angle) for v in shape.get_vertices()]
                    verts = [(v.x, v.y) for v in (self.draw_options.transform @ v for v in verts)]
                    pygame.draw.polygon(self.screen, color, verts)

    def draw_fps(self):
        font = pygame.font.SysFont("Courier New", 24)
        self.screen.blit(font.render(f"FPS: {int(self.clock.get_fps())}", True, (255, 255, 255)), (10, 10))

    def draw_settings_button(self):
        x = self.width - self.settings_button_size - 10
        self.screen.blit(self.settings_icon, (x, 10))
        return pygame.Rect(x, 10, self.settings_button_size, self.settings_button_size)

    def draw_speed_slider(self):
        sw, sh = 100, 8
        sx = self.width - sw - 20
        sy = self.settings_button_size + 40

        pygame.draw.rect(self.screen, (100, 100, 100), (sx, sy, sw, sh))
        knob_x = sx + (self.sim_speed / MAX_SPEED) * sw
        pygame.draw.circle(self.screen, (255, 255, 255), (int(knob_x), sy + sh // 2), 8)

        font = pygame.font.SysFont(None, 24)
        self.screen.blit(font.render(f"Speed: {self.sim_speed}x", True, (255, 255, 255)), (sx, sy - 20))

        return pygame.Rect(sx, sy - sh, sw, sh * 3)   # generous hitbox
    
    def draw_distance_markers(self):
        interval = 250  # pixels per meter
        font = pygame.font.SysFont("Courier New", 16)

        # figure out which markers are visible on screen
        tx = self.draw_options.transform.tx
        left_world  = -tx
        right_world = -tx + self.width

        first = int(left_world // interval)
        last  = int(right_world // interval) + 1

        for i in range(first, last):
            if i <= 0:
                continue
            world_x = i * interval
            screen_x = world_x + tx  # apply transform manually (no y offset needed here)

            ground_screen_y = 550 + (self.height - 600)
            pygame.draw.line(self.screen, (90, 90, 90),
                            (screen_x, ground_screen_y - 40),
                            (screen_x, ground_screen_y), 1)

            label = font.render(f"{i}m", True, (90, 90, 90))
            self.screen.blit(label, (screen_x + 3, ground_screen_y - 38))

    def draw_species_hud(self, focused_index, display_fitness, follow_best, names):
        n       = len(self.creatures)
        slot_w  = min(80, self.width // max(n + 1, 1))
        slot_h  = 36
        pad     = 4
        total_w = (n + 1) * slot_w
        start_x = (self.width - total_w) // 2
        font    = pygame.font.SysFont("Courier New", 12, bold=True)
        slot_rects = []

        ranked = sorted(range(len(self.creatures)), key=lambda x: display_fitness[x], reverse=True)
        podium = {}
        if len(self.creatures) >= 1:
            podium[ranked[0]] = (220, 180, 40)   # gold
        if len(self.creatures) >= 2:
            podium[ranked[1]] = (240, 240, 240)  # silver
        if len(self.creatures) >= 3:
            podium[ranked[2]] = (180, 100, 40)   # bronze

        for i in range(n):
            x       = start_x + i * slot_w
            rect    = pygame.Rect(x + pad, pad, slot_w - pad*2, slot_h)
            focused = i == focused_index and not follow_best

            pod_color  = podium.get(i)

            bg_col     = (50, 50, 55)
            border_col = pod_color if pod_color else (80, 80, 90)

            pygame.draw.rect(self.screen, bg_col,     rect, border_radius=4)
            pygame.draw.rect(self.screen, border_col, rect, width=2, border_radius=4)

            label = font.render(names[i], True, (230, 230, 230))
            self.screen.blit(label, label.get_rect(center=rect.center))
            slot_rects.append(rect)

            if focused or (follow_best and i == ranked[0]):
                pygame.draw.line(self.screen, bg_col,
                     (rect.left + 4,  rect.bottom + 3),
                     (rect.right - 4, rect.bottom + 3), 2)

        # best button
        bx = start_x + n * slot_w
        best_rect = pygame.Rect(bx + pad, pad, slot_w - pad*2, slot_h)
        bg_col     = (50, 50, 55)
        border_col = (80, 80, 90)

        pygame.draw.rect(self.screen, bg_col,     best_rect, border_radius=4)
        pygame.draw.rect(self.screen, border_col, best_rect, width=2, border_radius=4)

        lbl = font.render("BEST", True, (230, 230, 230))
        self.screen.blit(lbl, lbl.get_rect(center=best_rect.center))

        if follow_best:
            pygame.draw.line(self.screen, border_col,
                     (best_rect.left + 4,  best_rect.bottom + 2),
                     (best_rect.right - 4, best_rect.bottom + 2), 2)
        return slot_rects, best_rect
    
    # actual cool stuff

    def run_multiple_genomes(self, nets, sids): # render one gen of creatures based on best genome of each species
        for sid in sids:
            if sid not in self.species_names:
                self.species_names[sid] = generate_name(sid)
        names = [self.species_names[sid] for sid in sids]
    
        self.reset()
        self.spawn_creatures(len(nets))

        start_x = [c.center_x() for c in self.creatures]
        display_fitness = [0.0] * len(self.creatures)
        step_counter = 0

        best_btn_rect = None # prevent crash if u click between generations
        slot_rects = []    # this too

        while step_counter <= VIEW_STEPS:
            self.clock.tick(TARGET_FPS)
            step_counter += self.sim_speed

            # physics
            for _ in range(self.sim_speed):
                self.space.step(1 / TARGET_FPS)
                for i, (creature, net) in enumerate(zip(self.creatures, nets)):
                    self.apply_outputs(creature, net.activate(self.get_state(creature)))
                    display_fitness[i] = creature.center_x() - start_x[i]

            # rendering + input + sliders
            settings_rect = self.draw_settings_button() if self.render else None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if settings_rect and settings_rect.collidepoint(mx, my):
                        self.show_sliders = not self.show_sliders
                    if best_btn_rect and best_btn_rect.collidepoint(mx, my):
                        self.follow_best = True
                    for i, rect in enumerate(slot_rects):
                        if rect.collidepoint(mx, my):
                            self.follow_best = False
                            self.focused_index = i
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.follow_best = False
                        self.focused_index = (self.focused_index - 1) % len(self.creatures)
                    if event.key == pygame.K_RIGHT:
                        self.follow_best = False
                        self.focused_index = (self.focused_index + 1) % len(self.creatures)

            if self.follow_best:
                self.focused_index = display_fitness.index(max(display_fitness))
            
            self.set_camera(self.creatures[self.focused_index].center_x())

            self.screen.fill(BACKGROUND)
            self.draw_ground()
            self.draw_distance_markers()
            self.draw_creatures(self.focused_index, display_fitness)
            self.draw_fps()
            self.draw_settings_button()
            slot_rects, best_btn_rect = self.draw_species_hud(self.focused_index, display_fitness, self.follow_best, names)

            if self.show_sliders:
                slider_rect = self.draw_speed_slider()
                if pygame.mouse.get_pressed()[0]:
                    mx, my = pygame.mouse.get_pos()
                    if slider_rect.collidepoint(mx, my):
                        value = (mx - slider_rect.x) / slider_rect.width
                        self.sim_speed = max(1, int(value * MAX_SPEED))

            pygame.display.flip()

    def run_genome(self, net):
        self.reset()
        self.spawn_creatures(1)
        creature = self.creatures[0]
        start_x  = creature.center_x()

        ground_penalty  = 0
        rolling_penalty = 0

        for _ in range(EVAL_STEPS):
            self.space.step(1 / TARGET_FPS)
            self.apply_outputs(creature, net.activate(self.get_state(creature)))

            for name, shape in creature.shapes.items():
                if "torso" in name:
                    if self.touches_ground(shape):
                        ground_penalty += 100
                    if abs(creature.bodies[name].angle) > math.pi / 2: # angle accumulates past 2pi so a single barrel roll keeps punishing!
                        rolling_penalty += 10

        fitness = creature.center_x() - start_x - ground_penalty - rolling_penalty
        return fitness / 100
