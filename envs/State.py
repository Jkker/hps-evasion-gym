from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
from copy import deepcopy, copy
from typing import List


class Point:
    x: int
    y: int
    __slots__ = ["x", "y"]

    def __init__(self, x, y) -> None:
        self.x, self.y = x, y

    def clone(self):
        return copy(self)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Point):
            return self.x == __o.x and self.y == __o.y
        return False


class PositionAndVelocity:
    pos: Point
    vel: Point

    def __init__(self, pos, vel) -> None:
        self.pos, self.vel = pos, vel

    def clone(self):
        return PositionAndVelocity(self.pos.clone(), self.vel.clone())


class Wall:
    l: int
    r: int
    t: int
    b: int

    # def occupies(self, point: Point):
    #     raise NotImplementedError


@dataclass
class HorizontalWall(Wall):
    """Horizontal wall class
    """
    # i: int  # index
    y: int
    l: int  # left
    r: int  # right

    # def occupies(self, point: Point):
    #     return point.y == self.y and point.x >= self.l and point.x <= self.r

    @property
    def t(self):
        return self.y

    @property
    def b(self):
        return self.y

    def __str__(self):
        return f"HWall(y={self.y}, l={self.l}, r={self.r})"

    __repr__ = __str__


@dataclass
class VerticalWall(Wall):
    """Vertical wall class
    """
    # i: int  # index
    x: int
    t: int  # top
    b: int  # bottom

    # def occupies(self, point: Point):
    #     return point.x == self.x and point.y >= self.t and point.y <= self.b

    @property
    def l(self):
        return self.x

    @property
    def r(self):
        return self.x

    def __str__(self):
        return f"VWall(x={self.x}, t={self.t}, b={self.b})"

    __repr__ = __str__


def euclidean_distance(x, y) -> float:
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


@dataclass
class State:
    """Game state representation
    """
    player_time_left: float = 120
    game_num: int = 0
    tick_num: int = 0

    MAX_WALLS: int = 10
    WALL_PLACEMENT_DELAY: int = 20
    BOARD_SIZE_X: int = 300
    BOARD_SIZE_Y: int = 300

    current_wall_timer: int = 0

    hunter_x_pos: int = 0
    hunter_y_pos: int = 0
    hunter_x_vel: int = 1
    hunter_y_vel: int = 1

    prey_x: int = 230
    prey_y: int = 200
    wall_num: int = 0

    wall_list: List[Wall] = field(default_factory=list, init=False)

    @property
    def can_prey_move(self):
        return self.tick_num % 2 != 0

    @property
    def hunterPos(self):
        return Point(self.hunter_x_pos, self.hunter_y_pos)

    def setHunterPosAndVel(self, pv: PositionAndVelocity):
        self.hunter_x_pos = pv.pos.x
        self.hunter_y_pos = pv.pos.y
        self.hunter_x_vel = pv.vel.x
        self.hunter_y_vel = pv.vel.y

    @property
    def hunterPosAndVel(self):
        return PositionAndVelocity(Point(self.hunter_x_pos, self.hunter_y_pos),
                                   Point(self.hunter_x_vel, self.hunter_y_vel))

    @property
    def preyPos(self):
        return Point(self.prey_x, self.prey_y)

    def setPreyPos(self, pos: Point):
        self.prey_x = pos.x
        self.prey_y = pos.y

    @property
    def wall_dict_list(self):
        return ([[w.l, w.r, w.t, w.b] for w in self.wall_list])

    def hunter_prey_distance(self):
        return euclidean_distance((self.hunter_x_pos, self.hunter_y_pos),
                                  (self.prey_x, self.prey_y))