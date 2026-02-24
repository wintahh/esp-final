import pygame
import pymunk
import pymunk.pygame_util
import sys
import math




class Environment:
    def __init__(self, space):
        self.space = space
        self.create_ground()

    def create_ground(self):
        ground = pymunk.Segment(self.space.static_body,
                                (50, 550),
                                (75110, 550),
                                5)
        ground.friction = 1.0
        ground.collision_type = 1
        self.space.add(ground)







class Creature:
    def __init__(self, space, position):
        self.space = space
        self.mass = 2
        self.size = (120, 30)

        self.group = 1
        self.filter = pymunk.ShapeFilter(group=self.group)

        self.create_bodies(position)
        self.create_joints()

    def create_bodies(self, position):
        pos = pymunk.Vec2d(*position)

        # body1
        self.body1 = pymunk.Body(
            self.mass,
            pymunk.moment_for_box(self.mass, self.size)
        )
        self.body1.position = pos
        self.body1.angular_damping = 0.9

        self.shape1 = pymunk.Poly.create_box(self.body1, self.size)
        self.shape1.friction = 0.8
        self.shape1.collision_type = 2
        self.shape1.filter = self.filter

        # b2
        self.body2 = pymunk.Body(
            self.mass,
            pymunk.moment_for_box(self.mass, self.size)
        )
        self.body2.position = pos + (120,0)
        self.body2.angular_damping = 0.9

        self.shape2 = pymunk.Poly.create_box(self.body2, self.size)
        self.shape2.friction = 0.8
        self.shape2.collision_type = 2
        self.shape2.filter = self.filter

        self.space.add(
            self.body1, self.shape1,
            self.body2, self.shape2
        )

    def create_joints(self):
        hinge_position = self.body1.position + (60, 0)

        self.pivot = pymunk.PivotJoint(self.body1, self.body2, hinge_position)
        
        self.limit = pymunk.RotaryLimitJoint(
            self.body1, self.body2,
            math.radians(-90),
            math.radians(90)
        )

        self.motor1 = pymunk.SimpleMotor(self.space.static_body, self.body1, 0)
        self.motor1.max_force = 1000000

        self.motor2 = pymunk.SimpleMotor(self.body1, self.body2, 0)
        self.motor2.max_force = 1000000

        self.space.add(self.pivot, self.limit, self.motor1, self.motor2)

    def update_controls(self, keys):
        rate = 5
        if keys[pygame.K_LEFT]:
            self.motor1.rate = -rate
        elif keys[pygame.K_RIGHT]:
            self.motor1.rate = rate
        else:
            self.motor1.rate = 0

        if keys[pygame.K_a]:
            self.motor2.rate = -rate
        elif keys[pygame.K_d]:
            self.motor2.rate = rate
        else:
            self.motor2.rate = 0

    def get_center_x(self):
        return (self.body1.position.x +
                self.body2.position.x) / 2
    
    def get_center_y(self):
        return (self.body1.position.y +
                self.body2.position.y) / 2









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
        b1 = self.creature.body1
        b2 = self.creature.body2

        bodies = [(b1, self.creature.shape1),(b2,self.creature.shape2)]
        state = [
            0,                                               # vel_x
            0,                                               # vel_y
            self.creature.get_center_x(),
            self.creature.get_center_y()
        ]

        for body,shape in bodies:                                  # for each body in creature
            state[0] += body.velocity.x                      # vel_x
            state[1] += body.velocity.y                      # vel_y
            state.append(body.angle/(2*math.pi))             # normalized angle
            state.append(body.angular_velocity/(2*math.pi))  # tour par seconde
            state.append(body.position.x - state[2])         # position relative to center of mass
            state.append(body.position.y - state[3])         # ditto
            state.append(self.check_ground_contact(shape))

        state[0] = state[0]/len(bodies)                      #vel_x
        state[1] = state[1]/len(bodies)                      #vel_y
        return state

    def step(self):
        self.space.step(1 / 60)

        if self.render:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill((30, 30, 30))

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

        pygame.quit()
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
        self.screen.fill((30, 30, 30))

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
        max_steps = 500

        for _ in range(max_steps):
            inputs = self.step()  # get current state first

            outputs = net.activate(inputs)

            if self.render:
                print(outputs)

            self.creature.motor1.rate = outputs[0] * 2  # body1 rotation
            self.creature.motor2.rate = outputs[1] * 2  # body2 rotation

            state = self.step()

        fitness = state[2] - start_x
        return fitness/100

        


# ==============================
# Run Game
# ==============================
if __name__ == "__main__":
    Game().run()
