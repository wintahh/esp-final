import pymunk
import pymunk.pygame_util
import sys
import math
import json
import pygame

BACKGROUND = (180,180,180)
target_fps = 60

class Creature:
    def __init__(self, space, position):
        self.space = space
        self.torsomass = 2
        self.legmass = 1
        self.group = 1
        self.musclestiffness = 120000
        self.muscledamping = 12000
        self.filter = pymunk.ShapeFilter(group=self.group)
        self.position = list(position)

        with open("creatures/creature3.json") as f:
            data = json.load(f)

        self.create_bodies(data)
        self.create_joints_springs(data)

    def create_bodies(self, data):
        self.bodies = {}
        self.body_types = {}
        self.shapes = {}

        for body_data in data["bodies"]:
            t = body_data["type"]
            if t == "torso":
                mass = self.torsomass
            elif t == "leg":
                mass = self.legmass
            else:
                print(f"Couldn't find {body_data["name"]}'s type, going with leg")
            size = body_data["size"]
            pos = [a+b for a,b in zip(body_data["position"], self.position)]

            body = pymunk.Body(mass, pymunk.moment_for_box(mass, size))
            body.position = pymunk.Vec2d(*pos)
            body.angular_damping = 0.4

            shape = pymunk.Poly.create_box(body,size)
            shape.friction = 0.8
            shape.collision_type = 2
            shape.filter = self.filter

            self.space.add(body,shape)

            self.bodies[body_data["name"]] = body
            self.body_types[body_data["name"]] = type
            self.shapes[body_data["name"]] = shape

    def create_joints_springs(self, data):
        self.springs = []
        self.rotlimits = []
        for joint in data["joints"]:
            body1 = self.bodies[joint["body_a"]]
            body2 = self.bodies[joint["body_b"]]
            s = None
            j = None

            if joint["type"] == "pivot":
                anchor = [a+b for a,b in zip(joint["anchor"], self.position)]
                j = pymunk.PivotJoint(
                    body1,
                    body2,
                    anchor
                )
                
                if joint.get("actuated", False):
                    s = pymunk.DampedRotarySpring(
                        body1,
                        body2,
                        0,
                        self.musclestiffness,
                        self.muscledamping
                    )
            elif joint["type"] == "rotary_limit":
                j = pymunk.RotaryLimitJoint(
                    body1,
                    body2,
                    math.radians(joint["min_angle"]),
                    math.radians(joint["max_angle"])
                )
            else:
                print(f"Couldn't find {joint["name"]}'s type, report plz :)")

            self.space.add(j)
            if s:
                self.springs.append(s)
                self.space.add(s)

    def get_center_x(self):
        com_x = 0
        for name in self.bodies:
            com_x += self.bodies[name].position.x
        return com_x/len(self.bodies)

class Game:
    def __init__(self, render=True):
        self.render = render
        self.sim_speed = 1
        self.width = getattr(self, "width", 1000)
        self.height = getattr(self, "height", 600)

        self.show_sliders = getattr(self, "show_sliders", False)
        self.settings_button_size = 48

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.RESIZABLE
            )
            pygame.display.set_caption("Evolution Simulator")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        if self.render:
            self.settings_icon = pygame.image.load("settings.png").convert_alpha()
            self.settings_icon = pygame.transform.scale(self.settings_icon, (self.settings_button_size, self.settings_button_size))
        self.reset()

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)

        if self.render:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        else:
            self.draw_options = None

        self.create_ground()

        self.running = True
    
    def create_ground(self):
        
        ground = pymunk.Segment(self.space.static_body,
                                (-1000, 550),
                                (30000, 550),
                                5)
        ground.friction = 1.0
        ground.collision_type = 1
        self.space.add(ground)

    def draw_ground(self):

        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Segment):
                a = shape.a
                b = shape.b

                a = self.draw_options.transform @ a
                b = self.draw_options.transform @ b

                pygame.draw.line(
                    self.screen,
                    (200,200,200),
                    (a.x, a.y),
                    (b.x, b.y),
                    3
                )

    def draw_creatures(self, best_index, fitness):

        best_creature = self.creatures[best_index]
        best_x = self.creature_center_x(best_creature)

        # draw worst first, best last
        order = sorted(range(len(self.creatures)), key=lambda i: fitness[i])

        for i in order:
            creature = self.creatures[i]

            cx = self.creature_center_x(creature)
            max_dist = self.width/2
            dist = abs(cx - best_x)

            # fade based on distance
            nd = min(1, dist / max_dist) #normalized distance
            fade = int(60 + (180-60)*nd) # mehhhhh i dislike this code

            if i == best_index:
                color = (30, 30, 30)
            else:
                color = (fade, fade, fade)

            for shape in creature.shapes.values():
                if isinstance(shape, pymunk.Poly):

                    body = shape.body

                    verts = [
                        body.position + v.rotated(body.angle)
                        for v in shape.get_vertices()
                    ]

                    verts = [self.draw_options.transform @ v for v in verts]
                    verts = [(v.x, v.y) for v in verts]

                    # fill
                    pygame.draw.polygon(self.screen, color, verts)

                    # outline
                    #pygame.draw.polygon(self.screen, (20,20,20), verts, 2)

    def spawn_creatures(self, count):
        self.creatures = []

        for i in range(count):
            creature = Creature(self.space, (0, 400))
            self.creatures.append(creature)

    def creature_center_x(self, creature):
        com_x = 0
        for name in creature.bodies:
            com_x += creature.bodies[name].position.x
        return com_x/len(creature.bodies)

    def check_ground_contact(self, body_shape):
        contacts = self.space.shape_query(body_shape)
        for c in contacts:
            if isinstance(c.shape, pymunk.Segment) and c.shape.body == self.space.static_body:
                return 1
        return 0

    def get_state(self, creature):
        state = [
            0,                                               # vel_x
            0                                                # vel_y
        ]

        for name in creature.bodies:                         # for each body in creature
            body = creature.bodies[name]
            shape = creature.shapes[name]

            state[0] += body.velocity.x                      # vel_x
            state[1] += body.velocity.y                      # vel_y
            state.append(math.sin(body.angle))               # normalized angle
            state.append(body.angular_velocity/(2*math.pi))  # tour par seconde
            state.append(self.check_ground_contact(shape))   # touche le sol

        state[0] = state[0]/len(creature.bodies)             #vel_x
        state[1] = state[1]/len(creature.bodies)             #vel_y
        return state
    
    def draw_speed_slider(self):
        slider_width = 100
        slider_height = 8
        slider_x = self.width - slider_width - 20
        slider_y = self.settings_button_size + 40
        num_pos = 21

        # draw slider bar
        pygame.draw.rect(self.screen, (100,100,100),
                        (slider_x, slider_y, slider_width, slider_height))

        # knob position
        knob_x = slider_x + (self.sim_speed / num_pos) * slider_width

        pygame.draw.circle(
            self.screen,
            (255,255,255),
            (int(knob_x), slider_y + slider_height//2),
            8
        )

        # text
        font = pygame.font.SysFont(None, 24)
        txt = font.render(f"Speed: {self.sim_speed}x", True, (255,255,255))
        self.screen.blit(txt, (slider_x, slider_y - 20))

        return slider_x, slider_y, slider_width, slider_height

    def draw_fps(self):
        fps = int(self.clock.get_fps())
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"FPS: {fps}", True, (255,255,255))
        self.screen.blit(text, (10,10))

    def draw_settings_button(self):
        x = self.width - self.settings_button_size - 10
        y = 10

        rect = pygame.Rect(x, y, self.settings_button_size, self.settings_button_size)
        
        self.screen.blit(self.settings_icon, (x, y))

        return rect

    def run_multiple_genomes(self, nets): # rendering
        self.render = True
        self.reset()

        steps = target_fps*9

        self.spawn_creatures(len(nets))
        start_x = [self.creature_center_x(c) for c in self.creatures]
        fitness = [0 for _ in self.creatures]

        step_counter = 0
        while step_counter <= steps:
            step_counter += self.sim_speed
            self.clock.tick(target_fps)

            for _ in range(self.sim_speed): 
                self.space.step(1/target_fps)

                for i, (creature, net) in enumerate(zip(self.creatures, nets)):

                    inputs = self.get_state(creature)
                    outputs = net.activate(inputs)

                    for j, s in enumerate(creature.springs):
                        s.rest_angle = outputs[j] * math.pi*3/4 # to change

                    fitness[i] = self.creature_center_x(creature) - start_x[i] # this is not the creatures actual fitness its only for the camera ok?
                
                settings_rect = pygame.Rect(
                    self.width - self.settings_button_size - 10,
                    10,
                    self.settings_button_size,
                    self.settings_button_size
                )


            # rendering &
            # user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.VIDEORESIZE:
                    NOTHING = None
                    self.width, self.height = event.w, event.h
                    #self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()

                    if settings_rect.collidepoint(mx, my):
                        self.show_sliders = not self.show_sliders

            self.screen.fill(BACKGROUND)

            # camera centered on best creature  
            best_index = fitness.index(max(fitness))
            best_creature = self.creatures[best_index]

            cam_x = self.creature_center_x(best_creature)
            offset_x = self.width//2 - cam_x

            offset_y = self.height-600

            self.draw_options.transform = pymunk.Transform(
                tx=offset_x,
                ty=offset_y
            )

            self.draw_ground()
            self.draw_creatures(best_index, fitness)
            if self.show_sliders:
                speed_slider_x, speed_slider_y, speed_slider_width, speed_slider_height = self.draw_speed_slider()
            self.draw_fps()
            self.draw_settings_button()

            mouse = pygame.mouse.get_pressed()
            mx, my = pygame.mouse.get_pos()

            if mouse[0]:  # left mouse held
                if speed_slider_y-2*speed_slider_height < my < speed_slider_y+2*speed_slider_height and speed_slider_x < mx < speed_slider_x+speed_slider_width:
                    value = (mx - speed_slider_x) / speed_slider_width
                    self.sim_speed = max(1, int(value * 21))

            pygame.display.flip()

    def run_genome(self, net): # evaluating
        self.reset()

        self.spawn_creatures(1)
        creature = self.creatures[0]

        start_x = self.creature_center_x(creature)

        max_steps = 500

        ground_punishment = 0
        rolling_punishment = 0

        for _ in range(max_steps):
            self.space.step(1/60)
            inputs = self.get_state(creature)  # get current state first

            outputs = net.activate(inputs)
            
            for name in creature.shapes:
                if "torso" in name:
                    if self.check_ground_contact(creature.shapes[name]):
                        ground_punishment += 100
                    if abs(creature.bodies[name].angle) > math.pi/2: # goes from 0 to infinity (angles dont reset at 2pi for some reason) so if a creature does one barrel roll then itll continue being punished
                        rolling_punishment += 10                     # trains better for some reason, and we want to WALK not roll

            for i, s in enumerate(creature.springs):
                s.rest_angle = outputs[i] * math.pi*3/4

        position_x_for_debug = creature.get_center_x()
        fitness = position_x_for_debug - start_x - ground_punishment - rolling_punishment
        return fitness/100