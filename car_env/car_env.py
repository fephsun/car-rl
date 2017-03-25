import logging
import math

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import pyglet
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)

class CarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.dt = 0.1
        # Make the map.  1 = drivable
        self.world_size = 500
        self.world = np.zeros((self.world_size, self.world_size))
        for i in range(5):
            self.world[100*i + 40 : 100*i + 60, 40:460] = 1
            self.world[40:460, 100*i + 40 : 100*i + 60] = 1

        self.render_size = 200
        self.window_size = 80

        # All units in meters, kilograms, seconds
        self.car_length = 4.5
        self.car_width = 1.8

        self.car_max_steering_turn = 1.0  # radian / sec
        self.car_max_accel = 3.8    # assuming 0-60 in 7 s

        self.n_steering_options = 3
        self.n_accel_options = 0
        self.action_space = spaces.Discrete((2 * self.n_steering_options + 1) *
            (2 * self.n_accel_options + 1))
        self.observation_space = spaces.Box(0, 1,
            shape=(self.window_size, self.window_size, 2))

        self._seed()
        self.viewer = None
        self.done = False

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # Decode the action choice.
        steering_option = action % (2 * self.n_steering_options + 1)
        steering_vel = (steering_option - self.n_steering_options) \
            * self.car_max_steering_turn / self.n_steering_options
        if self.n_accel_options == 0:
            accel = 0
        else:
            accel_option = action / (2 * self.n_steering_options + 1)
            assert accel_option < 2 * self.n_accel_options + 1
            accel = (accel_option - self.n_accel_options) \
                * self.car_max_accel / self.n_accel_options

        # Get the new theta
        self.theta += steering_vel * self.dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        # Get the new velocity
        self.velocity += accel * self.dt
        # Update position
        self.x += self.velocity * self.dt * np.cos(self.theta)
        self.y += self.velocity * self.dt * np.sin(self.theta)
        # Check for collisions
        colliding = self.check_collisions()
        # Update deadline
        self.deadline -= self.dt

        if colliding or self.deadline < 0:
            self.done = True
            reward = 0
        else:
            reward = self.update_goal()

        return self.make_obs(self.window_size), reward, self.done, {}

    def make_obs(self, window_size):
        half_width = window_size / 2
        shifted_map = ndimage.shift(self.world, (self.x - self.world.shape[0] / 2,
            self.y - self.world.shape[1] / 2), mode='constant', cval=0, order=0)
        rotated_map = ndimage.rotate(shifted_map, -self.theta * 180 / np.pi,
            reshape=False, order=0)
        out_map = np.zeros((window_size, window_size, 2))
        out_map[:, :, 0] = rotated_map[rotated_map.shape[0] / 2 - half_width
                : rotated_map.shape[0] / 2 + half_width,
            rotated_map.shape[1] / 2 - half_width
                : rotated_map.shape[1] / 2 + half_width]
        # Draw the car.
        out_map[int(half_width - self.car_length / 2) 
                : int(half_width + self.car_length / 2),
            int(half_width - self.car_width / 2)
                : int(half_width + self.car_width / 2), 0] = 0

        # Draw the goal point on channel 2.
        goal_x_1 = self.goal_x - self.x
        goal_y_1 = self.goal_y - self.y
        goal_r = (goal_x_1 ** 2 + goal_y_1 ** 2) ** 0.5
        goal_theta = np.arctan2(goal_y_1, goal_x_1) - self.theta + np.pi
        goal_x_2 = goal_r * np.cos(goal_theta)
        if abs(goal_x_2) > half_width - 2:
            goal_r = abs((half_width - 2) / np.cos(goal_theta))
        goal_y_2 = goal_r * np.sin(goal_theta)
        if abs(goal_y_2) > half_width - 2:
            goal_r = abs((half_width - 2) / np.sin(goal_theta))
        goal_x_2 = goal_r * np.cos(goal_theta)
        goal_y_2 = goal_r * np.sin(goal_theta)
        left_edge = max(int(half_width + goal_x_2) - 2, 0)
        right_edge = min(int(half_width + goal_x_2) + 2, window_size - 1)
        bottom_edge = max(int(half_width + goal_y_2) - 2, 0)
        top_edge = min(int(half_width + goal_y_2) + 2, window_size - 1)
        out_map[left_edge:right_edge, bottom_edge:top_edge, 1] = 1
        return out_map

    def check_collisions(self):
        for delta_x in [-self.car_width / 2, 0, self.car_width / 2]:
            for delta_y in [-self.car_length / 2, 0, self.car_length / 2]:
                x_idx = int(self.x + delta_x)
                y_idx = int(self.y + delta_y)
                if (x_idx < 0 or x_idx >= self.world.shape[0]
                    or y_idx < 0 or y_idx > self.world.shape[1]):
                    return True
                if self.world[x_idx, y_idx] == 0:
                    return True
        return False

    def update_goal(self):
        dist = ((self.x - self.goal_x) ** 2 + (self.y - self.goal_y) ** 2) ** 0.5
        if dist < 12:
            self.make_new_goal()
            return 100
        return max(0, 0.03 * (200 - dist))

    def make_new_goal(self):
        done = False
        while not done:
            self.goal_x = self.np_random.randint(150, 350)
            self.goal_y = self.np_random.randint(0, 200)
            done = (self.world[self.goal_x, self.goal_y] == 1)
        # Update the deadline.
        l1_dist = abs(self.goal_x - self.x) + abs(self.goal_y - self.y)
        kMinSpeed = 5
        self.deadline = l1_dist / kMinSpeed

    def _reset(self):
        # The following variables represent the state of the system.
        self.x = self.np_random.uniform(50, 200)
        self.y = self.np_random.uniform(48, 52)
        self.theta = self.np_random.uniform(-0.2, 0.2)
        self.velocity = 10
        self.done = False

        self.make_new_goal()
        return self.make_obs(self.window_size)

    def _render(self, mode='human', close=False):
        out_img = self.make_img(self.make_obs(self.render_size))
        if mode == 'human':
            if self.viewer is None:
                self.viewer = pyglet.window.Window(width=self.render_size,
                    height=self.render_size)
            pyglet_img = pyglet.image.ImageData(self.render_size, self.render_size,
                'RGB', out_img.astype('uint8').data.__str__())
            @self.viewer.event
            def on_draw():
                self.viewer.clear()
                pyglet_img.blit(0, 0)
            self.viewer.dispatch_events()
            self.viewer.dispatch_event('on_draw')
            self.viewer.flip()

    def make_img(self, obs_img):
        out_img = np.zeros((obs_img.shape[0], obs_img.shape[1], 3))
        out_img[:, :, 0] = obs_img[:, :, 0] * 255
        out_img[:, :, 1] = obs_img[:, :, 1] * 255
        return out_img


        return out_img
