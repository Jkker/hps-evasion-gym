import gym
import numpy as np
import pygame
from gym import spaces

from .Game import EvasionGame


class HunterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 render_mode=None,
                 size=300,
                 max_walls: int = 10,
                 wall_placement_delay: int = 10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self._game = EvasionGame(max_walls, wall_placement_delay)

        self.observation_space = spaces.Dict({
            "hunter_x_pos":
            spaces.Box(low=0, high=size - 1, shape=(), dtype=np.uint16),
            "hunter_y_pos":
            spaces.Box(low=0, high=size - 1, shape=(), dtype=np.uint16),
            "hunter_x_vel":
            spaces.Box(low=0, high=size - 1, shape=(), dtype=np.uint8),
            "hunter_y_vel":
            spaces.Box(low=0, high=size - 1, shape=(), dtype=np.uint8),
            "prey_x":
            spaces.Box(low=0, high=size - 1, shape=(), dtype=np.uint16),
            "prey_y":
            spaces.Box(low=0, high=size - 1, shape=(), dtype=np.uint16),
            "wall_num":
            spaces.Box(low=0, high=max_walls, shape=(), dtype=np.uint8),
            "current_wall_timer":
            spaces.Box(low=0,
                       high=wall_placement_delay,
                       shape=(),
                       dtype=np.uint16),
            "player_time_left":
            spaces.Box(low=0, high=120, shape=(), dtype=np.float32),
            "wall_list":
            spaces.Sequence(
                spaces.Dict({
                    "l":
                    spaces.Box(low=0, high=size - 1, shape=(),
                               dtype=np.uint16),
                    "r":
                    spaces.Box(low=0, high=size - 1, shape=(),
                               dtype=np.uint16),
                    "t":
                    spaces.Box(low=0, high=size - 1, shape=(),
                               dtype=np.uint16),
                    "b":
                    spaces.Box(low=0, high=size - 1, shape=(),
                               dtype=np.uint16),
                })),
            "tick_num":
            spaces.Box(low=0, high=100000, shape=(), dtype=np.uint64),
        })
        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Dict({
            "create_action":
            spaces.Discrete(3),  # noop, vertical wall, horizontal wall
        })
        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Get the observation, i.e. the agent's and the target's location."""
        return {
            "hunter_x_pos": self._game.state.hunter_x_pos,
            "hunter_y_pos": self._game.state.hunter_y_pos,
            "hunter_x_vel": self._game.state.hunter_x_vel,
            "hunter_y_vel": self._game.state.hunter_y_vel,
            "prey_x": self._game.state.prey_x,
            "prey_y": self._game.state.prey_y,
            "wall_num": self._game.state.wall_num,
            "current_wall_timer": self._game.state.current_wall_timer,
            "player_time_left": self._game.state.player_time_left,
            "wall_list": self._game.state.wall_dict_list,
            "tick_num": self._game.state.tick_num,
        }

    def _get_info(self):
        return {
            "distance":
            np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0,
                                                       self.size,
                                                       size=2,
                                                       dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0,
                                                            self.size,
                                                            size=2,
                                                            dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0,
                                       self.size - 1)
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location,
                                    self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
