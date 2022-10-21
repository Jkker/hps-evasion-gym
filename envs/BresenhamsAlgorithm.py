from .State import Point
from typing import List


def pointsBetween(x0: int, y0: int, x1: int, y1: int) -> List[Point]:
    points = []
    steep = abs(y1 - y0) > abs(x1 - x0)
    if (steep):
        tx0 = x0
        x0 = y0
        y0 = tx0
        tx1 = x1
        x1 = y1
        y1 = tx1
    if (x0 > x1):
        tx0 = x0
        x0 = x1
        x1 = tx0
        ty0 = y0
        y0 = y1
        y1 = ty0
    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = int(deltax / 2)
    y = y0
    ystep = 0
    if (y0 < y1):
        ystep = 1
    else:
        ystep = -1

    for x in range(x0, x1 + 1):
        if (steep):
            points.append(Point(y, x))
        else:
            points.append(Point(x, y))
        error -= deltay
        if (error < 0):
            y += ystep
            error += deltax
    return points