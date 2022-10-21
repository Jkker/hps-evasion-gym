from enum import IntEnum
from typing import List

import numpy as np
from .BresenhamsAlgorithm import pointsBetween
from .State import (HorizontalWall, Point, PositionAndVelocity, State,
                    VerticalWall, Wall, euclidean_distance)


class WALL_ACTION(IntEnum):
    NONE = 0
    HORIZONTAL = 1
    VERTICAL = 2


class EvasionGame:
    state: State

    def __init__(self,
                 max_walls: int,
                 wall_placement_delay: int,
                 size=300,
                 debug: bool = False):
        self.state = State(MAX_WALLS=max_walls,
                           WALL_PLACEMENT_DELAY=wall_placement_delay,
                           BOARD_SIZE_X=size,
                           BOARD_SIZE_Y=size)
        self.board = np.zeros((size, size), dtype=np.int8)

        self.debug = debug

    def tick(self, hunterWallAction: WALL_ACTION,
             hunterWallsToDelete: List[int], preyMovement: Point):
        self.removeWall(hunterWallsToDelete)
        prevHunterPos = self.state.hunterPos.clone()

        self.state.setHunterPosAndVel(self.move(self.state.hunterPosAndVel))
        self.doBuildAction(prevHunterPos, hunterWallAction)

        if self.canPreyMove():
            self.state.setPreyPos(
                self.move(PositionAndVelocity(self.state.preyPos,
                                              preyMovement)).pos)

        self.state.tick_num += 1
        if self.state.current_wall_timer > 0:
            self.state.current_wall_timer -= 1

        return self.captured()

    def isOccupied(self, p: Point) -> bool:
        if p.x < 0 or p.x >= self.state.BOARD_SIZE_X or p.y < 0 or p.y >= self.state.BOARD_SIZE_Y:
            return True

        # for w in self.state.wall_list:
        # if w.occupies(p):
        # return True

        # return False
        return self.board[p.x, p.y]

    def addWall(self, wall: Wall) -> bool:
        if len(self.state.wall_list
               ) < self.state.MAX_WALLS and self.state.current_wall_timer <= 0:
            self.state.wall_list.append(wall)
            self.state.current_wall_timer = self.state.WALL_PLACEMENT_DELAY
            if isinstance(wall, HorizontalWall):
                self.board[wall.l:wall.r + 1, wall.y] = 1
            elif isinstance(wall, VerticalWall):
                self.board[wall.x, wall.t:wall.b + 1] = 1
            return True

        return False

    def removeWall(self, removeList: List[int]):
        if self.debug:
            for i in removeList:
                print("🧱 Removed: ", self.state.wall_list[i], "\tTick:",
                      self.state.tick_num)

        new_list = []
        for i, w in enumerate(self.state.wall_list):
            if i in removeList:
                if isinstance(w, HorizontalWall):
                    self.board[w.l:w.r + 1, w.y] = 0
                elif isinstance(w, VerticalWall):
                    self.board[w.x, w.t:w.b + 1] = 0
            else:
                new_list.append(w)

        self.state.wall_list = new_list

        return

    def captured(self) -> bool:
        if euclidean_distance(
            (self.state.hunter_x_pos, self.state.hunter_y_pos),
            (self.state.prey_x, self.state.prey_y)) <= 4.0:
            pts = pointsBetween(self.state.hunter_x_pos,
                                self.state.hunter_y_pos, self.state.prey_x,
                                self.state.prey_y)

            for p in pts:
                if self.isOccupied(p):
                    return False

            return True
        return False

    def canPreyMove(self) -> bool:
        return self.state.can_prey_move

    def doBuildAction(self, pos: Point, action: WALL_ACTION) -> bool:
        if action == WALL_ACTION.HORIZONTAL:
            greater = pos.clone()
            lesser = pos.clone()

            while not self.isOccupied(greater):
                if greater == self.state.hunterPos or greater == self.state.preyPos:
                    return False
                greater.x += 1

            while not self.isOccupied(lesser):
                if lesser == self.state.hunterPos or lesser == self.state.preyPos:
                    return False
                lesser.x -= 1

            wall = HorizontalWall(pos.y, lesser.x + 1, greater.x - 1)

            if self.debug:
                print("🧱 Built: ", wall, "\tBudget:",
                      self.state.MAX_WALLS - len(self.state.wall_list),
                      "\tTick:", self.state.tick_num)

            # self.board[wall.l:wall.r + 1, wall.y] = True
            return self.addWall(wall)

        elif action == WALL_ACTION.VERTICAL:
            greater = pos.clone()
            lesser = pos.clone()

            while not self.isOccupied(greater):
                if greater == self.state.hunterPos or greater == self.state.preyPos:
                    return False
                greater.y += 1

            while not self.isOccupied(lesser):
                if lesser == self.state.hunterPos or lesser == self.state.preyPos:
                    return False
                lesser.y -= 1

            wall = VerticalWall(pos.x, lesser.y + 1, greater.y - 1)
            if self.debug:
                print("🧱 Built: ", wall, "\tBudget:",
                      self.state.MAX_WALLS - len(self.state.wall_list),
                      "\tTick:", self.state.tick_num)
            # self.board[wall.x, wall.t:wall.b + 1] = True
            return self.addWall(wall)

        return False

    def move(self, posAndVel: PositionAndVelocity) -> PositionAndVelocity:
        newPosAndVel = posAndVel.clone()

        newPosAndVel.vel.x = min(max(newPosAndVel.vel.x, -1), 1)
        newPosAndVel.vel.y = min(max(newPosAndVel.vel.y, -1), 1)
        target = self.add(newPosAndVel.pos, newPosAndVel.vel)
        if (not self.isOccupied(target)):
            newPosAndVel.pos = target
        else:
            if (newPosAndVel.vel.x == 0 or newPosAndVel.vel.y == 0):
                if (newPosAndVel.vel.x != 0):
                    newPosAndVel.vel.x = -newPosAndVel.vel.x
                else:
                    newPosAndVel.vel.y = -newPosAndVel.vel.y
            else:
                oneRight = self.isOccupied(
                    self.add(newPosAndVel.pos, Point(newPosAndVel.vel.x, 0)))
                oneUp = self.isOccupied(
                    self.add(newPosAndVel.pos, Point(0, newPosAndVel.vel.y)))
                if (oneRight and oneUp):
                    newPosAndVel.vel.x = -newPosAndVel.vel.x
                    newPosAndVel.vel.y = -newPosAndVel.vel.y
                elif (oneRight):
                    newPosAndVel.vel.x = -newPosAndVel.vel.x
                    newPosAndVel.pos.y = target.y
                elif (oneUp):
                    newPosAndVel.vel.y = -newPosAndVel.vel.y
                    newPosAndVel.pos.x = target.x
                else:
                    twoUpOneRight = self.isOccupied(
                        self.add(
                            newPosAndVel.pos,
                            Point(newPosAndVel.vel.x, newPosAndVel.vel.y * 2)))
                    oneUpTwoRight = self.isOccupied(
                        self.add(
                            newPosAndVel.pos,
                            Point(newPosAndVel.vel.x * 2, newPosAndVel.vel.y)))
                    if ((twoUpOneRight and oneUpTwoRight)
                            or (not twoUpOneRight and not oneUpTwoRight)):
                        newPosAndVel.vel.x = -newPosAndVel.vel.x
                        newPosAndVel.vel.y = -newPosAndVel.vel.y
                    elif (twoUpOneRight):
                        newPosAndVel.vel.x = -newPosAndVel.vel.x
                        newPosAndVel.pos.y = target.y
                    else:
                        newPosAndVel.vel.y = -newPosAndVel.vel.y
                        newPosAndVel.pos.x = target.x
        return newPosAndVel

    def add(self, a: Point, b: Point) -> Point:
        return Point(a.x + b.x, a.y + b.y)

    def getState(self) -> State:
        return self.state
