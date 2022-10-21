from collections import OrderedDict

import gym
import numpy as np
import pygame
from gym import spaces

from .Game import EvasionGame, Point, VerticalWall, HorizontalWall, WALL_ACTION, pointsBetween


class HunterEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
        "render_size": 1200
    }

    def __init__(self, env_config: dict = dict()):

        size = env_config.get("size", 300)
        self.size = size  # The size of the square grid
        self.debug = env_config.get("debug", False)
        max_walls = env_config.get("max_walls", 10)
        wall_placement_delay = env_config.get("wall_placement_delay", 25)

        self._game = EvasionGame(max_walls,
                                 wall_placement_delay,
                                 debug=self.debug)
        self.state = self._game.state
        self.prey_movement = Point(0, 0)

        bound_high = size - 1
        self.observation_space = spaces.Dict({
            "hunter_x_pos":
            spaces.Box(low=0, high=bound_high, shape=(), dtype=int),
            "hunter_y_pos":
            spaces.Box(low=0, high=bound_high, shape=(), dtype=int),
            "hunter_x_vel":
            spaces.Box(low=-1, high=2, shape=(), dtype=int),
            "hunter_y_vel":
            spaces.Box(low=-1, high=2, shape=(), dtype=int),
            "prey_x":
            spaces.Box(low=0, high=bound_high, shape=(), dtype=int),
            "prey_y":
            spaces.Box(low=0, high=bound_high, shape=(), dtype=int),
            "current_wall_timer":
            spaces.Box(low=0, high=wall_placement_delay, shape=(), dtype=int),
            # "player_time_left":
            # spaces.Box(low=0, high=120, shape=(), dtype=float),
            # "board": spaces.Discrete(size**2),
            # "wall_num":           spaces.Box(low=0, high=max_walls, shape=(), dtype=np.uint8),
            "board":
            spaces.Box(0, 1, shape=(size, size), dtype=np.uint8),
            # "tick_num": spaces.Box(low=0, high=100000, shape=(), dtype=np.uint64),
            # "wall_list":
            # Repeated(
            #     spaces.Dict({
            #         "l":
            #         spaces.Box(low=0, high=size - 1, shape=(), dtype=int),
            #         "r":
            #         spaces.Box(low=0, high=size - 1, shape=(), dtype=int),
            #         "t":
            #         spaces.Box(low=0, high=size - 1, shape=(), dtype=int),
            #         "b":
            #         spaces.Box(low=0, high=size - 1, shape=(), dtype=int),
            #     }), max_walls)
        })
        # self.action_space = spaces.Dict({
        #     "wall_create":
        #     spaces.Discrete(3),  # noop, vertical wall, horizontal wall
        #     "wall_delete":
        #     spaces.MultiBinary(self.state.MAX_WALLS)
        # })
        self.action_space = spaces.Tuple(
            (spaces.Discrete(3), spaces.MultiDiscrete([self.state.MAX_WALLS])))

        # assert render_mode is None or render_mode in self.metadata[
        #     "render_modes"]
        self.render_mode = env_config.get("render_mode", None)
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.canvas = None

    def _get_obs(self):
        """Get the observation, i.e. the agent's and the target's location."""
        return OrderedDict({
            "hunter_x_pos": self.state.hunter_x_pos,
            "hunter_y_pos": self.state.hunter_y_pos,
            "hunter_x_vel": self.state.hunter_x_vel,
            "hunter_y_vel": self.state.hunter_y_vel,
            "prey_x": self.state.prey_x,
            "prey_y": self.state.prey_y,
            "current_wall_timer": self.state.current_wall_timer,
            # "player_time_left": self.state.player_time_left,-
            "board":
            self._game.board.copy()  # "wall_num": self.state.wall_num,
            # "wall_list": self.state.wall_dict_list,
            # "tick_num": self.state.tick_num,
        })

    def _get_info(self):
        return OrderedDict({
            "distance":
            # np.linalg.norm(self._agent_location - self._target_location, ord=2)
            self.state.hunter_prey_distance()
        })

    def reset(self):
        # Choose the agent's location uniformly at random
        self._game = EvasionGame(self.state.MAX_WALLS,
                                 self.state.WALL_PLACEMENT_DELAY,
                                 debug=self.debug,
                                 size=self.size)

        observation = self._get_obs()
        info = self._get_info()

        return observation

    def step(self, action: tuple[WALL_ACTION, list[int]], prey_movement: tuple[int, int] | None = None):

        # prev = self.prey_movement
        if prey_movement is not None:
            self.prey_movement = Point(*prey_movement)

        else:
            if np.random.rand() < 0.1:  # 20% chance of random movement
                self.prey_movement = Point(np.random.randint(-1, 2),
                                        np.random.randint(-1, 2))

        # prev_dst  = self.state.hunter_prey_distance()
        terminated = self._game.tick(
            action[0], [w for w in action[1] if w < len(self.state.wall_list)],
            self.prey_movement)

        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        truncated = False
        # return observation, reward, terminated, truncated, info
        return observation, reward, terminated, info

    def render(self):
        if not self.render_mode:
            return

        return self._render_frame()

    def _render_frame(self):
        # Init
        render_size = (self.metadata['render_size'],
                       self.metadata['render_size'])

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(render_size)
            pygame.display.set_caption('HPS Evasion Game Hunter Gym Env')

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.canvas is None:
            self.canvas = pygame.Surface((self.size, self.size))
        self.canvas.fill((34, 31, 34))

        # Draw Prey
        pygame.draw.rect(
            self.canvas,
            (120, 220, 232),
            pygame.Rect(
                (
                    self.state.prey_x,
                    self.state.prey_y,
                ),
                (1, 1),
            ),
        )
        # Draw Hunter
        pygame.draw.rect(
            self.canvas,
            (255, 97, 136),
            pygame.Rect(
                (self.state.hunter_x_pos, self.state.hunter_y_pos),
                (1, 1),
            ),
        )

        # Draw Walls
        for w in self.state.wall_list:
            if isinstance(w, VerticalWall):

                pygame.draw.rect(
                    self.canvas,
                    (138, 136, 134),
                    pygame.Rect(
                        (w.x, w.t),
                        (1, w.b - w.t + 1),
                    ),
                )
            elif isinstance(w, HorizontalWall):
                pygame.draw.rect(
                    self.canvas,
                    (138, 136, 134),
                    pygame.Rect(
                        (w.l, w.y),
                        (w.r - w.l + 1, 1),
                    ),
                )

        scaled_canvas = pygame.transform.scale(self.canvas, render_size)
        if self.render_mode == "human" and self.window is not None:
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(scaled_canvas, self.window.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(
                pygame.surfarray.pixels3d(scaled_canvas)),
                                axes=(1, 0, 2))

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
