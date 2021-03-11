# -*- coding: utf-8 -*- 
# @Time : 2021/3/11 15:29 
# @Author : CaiXin
# @File : MappingTest.py

import numpy as np
from modules.PointCloudMapping import MappingManager
from modules.PointCloudMapping import showPointcloudFromFile

curr_se3 = np.identity(4)
curr_se3[:3, -1] = [1, 1, 0]
map = MappingManager()
map.curr_se3 = curr_se3
points = np.random.random((1000, 3))
i = 0
while (i < 100):
    i += 1
    points -= 0.001
    map.curr_local_ptcloud = points
    map.updateMap()
    # map.vizMapWithOpen3D()
map.saveMap2File("map.pcd")
# showPointcloudFromFile("map_20201220_846.pcd")