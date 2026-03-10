import pygame
import pymunk
import pymunk.pygame_util
import sys
import math
import json

BACKGROUND = (35,35,35)


class Environment:
    def __init__(self, space):
        self.space = space
        self.create_ground()

    def create_ground(self):
        ground = pymunk.Segment(self.space.static_body,
                                (-1000, 550),
                                (75110, 550),
                                5)
        ground.friction = 1.0
        ground.collision_type = 1
        self.space.add(ground)







class Creature:
    def __init__(self, space, position):
        self.space = space
        self.torsomass = 2
        self.legmass = 1
        self.group = 1
        self.musclestiffness = 120000
        self.muscledamping = 12000
        self.filter = pymunk.ShapeFilter(group=self.group)

        with open("creatures/creature3.json") as f:
            data = json.load(f)

        self.create_bodies(data)
        self.create_joints_springs(data)

    def create_bodies(self, data):
        self.bodies = {}
        self.body_types = {}
        self.shapes = {}

        for body_data in data["bodies"]:
            type = body_data["type"]
            if type == "torso":
                mass = self.torsomass
            elif type == "leg":
                mass = self.legmass
            else:
                print(f"Couldn't find {body_data["name"]}'s type, going with leg")
            size = body_data["size"]
            pos = body_data["position"]

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
                anchor = joint["anchor"]
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


    def update_controls(self, keys): # wtv just debug stuff
        for s in self.springs:
            if self.springs.index(s) % 2:
                if keys[pygame.K_LEFT]:
                    s.rest_angle = math.pi * 3/4
                elif keys[pygame.K_RIGHT]:
                    s.rest_angle = -1 * math.pi * 3/4
            else:
                if keys[pygame.K_a]:
                    s.rest_angle = math.pi * 3/4
                elif keys[pygame.K_d]:
                    s.rest_angle = -1 * math.pi * 3/4

    def get_center_x(self):
        com_x = 0
        for name in self.bodies:
            com_x += self.bodies[name].position.x
        return com_x/len(self.bodies)
    
    def get_center_y(self):
        com_y = 0
        for name in self.bodies :
            com_y += self.bodies[name].position.y
        return com_y/len(self.bodies)

class Game:
    def __init__(self, render=True):
        self.width = 800
        self.height = 600
        self.render = render

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.RESIZABLE
        )
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

        self.reset()

    def reset(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 981)

        if self.render:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        else:
            self.draw_options = None

        self.environment = Environment(self.space)
        self.creature = Creature(self.space, (400, 300))

        self.running = True

    def check_ground_contact(self, body_shape):
        contacts = self.space.shape_query(body_shape)
        for c in contacts:
            if isinstance(c.shape, pymunk.Segment) and c.shape.body == self.space.static_body:
                return 1
        return 0

    def get_state(self):
        state = [
            0,                                               # vel_x
            0                                                # vel_y
        ]

        for name in self.creature.bodies:   # for each body in creature
            body = self.creature.bodies[name]
            shape = self.creature.shapes[name]

            state[0] += body.velocity.x                      # vel_x
            state[1] += body.velocity.y                      # vel_y
            state.append(math.sin(body.angle))               # normalized angle
            state.append(body.angular_velocity/(2*math.pi))  # tour par seconde
            state.append(self.check_ground_contact(shape))   # touche le sol

        state[0] = state[0]/len(self.creature.bodies)        #vel_x
        state[1] = state[1]/len(self.creature.bodies)       #vel_y
        return state

    def step(self):
        self.space.step(1 / 60)

        if self.render:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill(BACKGROUND)

            creature_x = self.creature.get_center_x()
            offset_x = self.width / 2 - creature_x

            self.draw_options.transform = pymunk.Transform(
                tx=offset_x,
                ty=0
            )

            self.space.debug_draw(self.draw_options)

            pygame.display.flip()
            self.clock.tick(60)

        return self.get_state()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
        pygame.quit
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        keys = pygame.key.get_pressed()
        self.creature.update_controls(keys)

        self.space.step(1 / 60)

    def draw(self):
        self.screen.fill(BACKGROUND)

        creature_x = self.creature.get_center_x()
        offset_x = self.width / 2 - creature_x

        self.draw_options.transform = pymunk.Transform(
            tx=offset_x,
            ty=0
        )

        self.space.debug_draw(self.draw_options)

        pygame.display.flip()
        self.clock.tick(60)

    def run_genome(self, net, render=False):

        self.render = render
        self.reset()

        start_x = self.creature.get_center_x()
        if render:
            max_steps = 10000
        else:
            max_steps = 500

        ground_punishment = 0
        rolling_punishment = 0

        for _ in range(max_steps):
            inputs = self.step()  # get current state first

            outputs = net.activate(inputs)
            
            for name in self.creature.shapes:
                if "torso" in name:
                    if self.check_ground_contact(self.creature.shapes[name]):
                        ground_punishment += 100
                    if abs(self.creature.bodies[name].angle) > math.pi/4: # va donc de 0 a infini, si creature fait un tour complet bin elle continura a avoir le punishment
                        rolling_punishment += 10

            for s in self.creature.springs:
                s.rest_angle = outputs[self.creature.springs.index(s)] * math.pi*3/4 # this is to change, access rotation limit and change it to radians
                if render:
                    print(f"{self.creature.springs.index(s)}: {round(outputs[self.creature.springs.index(s)],3)}")

        position_x_for_debug = self.creature.get_center_x()
        fitness = position_x_for_debug - start_x - ground_punishment - rolling_punishment
        if render:
            print(position_x_for_debug, ground_punishment, rolling_punishment)
        return fitness/100

        


# ==============================
# Run Game
# ==============================
if __name__ == "__main__":
    Game().run()
