# -*- coding: utf-8 -*- 
# @Time : 2020/9/10 14:41 
# @Author : CaiXin
# @File : OccupancyGridMapping.py
from math import log
import numpy as np


def isInclude(grid, zt):
    pass


def inverseSensorModel(i, j, grid_length, xt, zt):
    grid_center_x = float(i + 0.5) * grid_length
    grid_center_y = float(j + 0.5) * grid_length
    r = np.sqrt((xt[0] - grid_center_x) ** 2 + (xt[1] - grid_center_y) ** 2)



def occupancyGridMapping(map, xt, zt):
    l0 = log(1, 10)
    num_rows = map.num_rows
    num_cols = map.num_cols
    grid_length = map.grid_length
    for i in range(num_rows):
        for j in range(num_cols):
            map[i, j] = map[i, j] + inverseSensorModel(i, j, grid_length, xt, zt) - l0
    return map


class map:
    def __init__(self):
        pass

    def init(self, grid_length, num_rows=100, num_cols=100):
        self.grid_length = grid_length
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.graph = np.zeros((self.num_rows, self.num_cols))
