# -*- coding: utf-8 -*- 
# @Time : 2020/9/7 19:17 
# @Author : CaiXin
# @File : VerticalScanContextManager.py
import numpy as np

np.set_printoptions(precision=4)

import time
from scipy import spatial


def xy2theta(x, y):
    if (x >= 0 and y >= 0):
        theta = 180 / np.pi * np.arctan(y / x);
    if (x < 0 and y >= 0):
        theta = 180 - ((180 / np.pi) * np.arctan(y / (-x)));
    if (x < 0 and y < 0):
        theta = 180 + ((180 / np.pi) * np.arctan(y / x));
    if (x >= 0 and y < 0):
        theta = 360 - ((180 / np.pi) * np.arctan((-y) / x));

    return theta


def pt2rs(point, gap_ring, gap_sector, num_ring, num_sector):
    x = point[0]
    y = point[1]
    # z = point[2]

    if (x == 0.0):
        x = 0.001
    if (y == 0.0):
        y = 0.001

    theta = xy2theta(x, y)
    faraway = np.sqrt(x * x + y * y)

    idx_ring = np.divmod(faraway, gap_ring)[0]
    idx_sector = np.divmod(theta, gap_sector)[0]

    if (idx_ring >= num_ring):
        idx_ring = num_ring - 1  # python starts with 0 and ends with N-1

    return int(idx_ring), int(idx_sector)


def pt2ls(point, gap_level, gap_sector, num_level, min_height):
    x = point[0]
    y = point[1]
    z = point[2]

    if (x == 0.0):
        x = 0.001
    if (y == 0.0):
        y = 0.001

    theta = xy2theta(x, y)
    height = z - min_height

    idx_level = np.divmod(height, gap_level)[0]
    idx_sector = np.divmod(theta, gap_sector)[0]

    if (idx_level >= num_level):
        idx_level = num_level - 1  # python starts with 0 and ends with N-1
    if (idx_level < 0):
        idx_level = 0

    return int(idx_level), int(idx_sector)


def ptcloud2vsc(ptcloud, expnd_shape, max_height, min_height, max_dist):
    num_level = expnd_shape[0]
    num_sector = expnd_shape[1]

    gap_level = (max_height - min_height) / num_level
    gap_sector = 360 / num_sector

    enough_large = 500
    vsc_storage = np.zeros([enough_large, num_level, num_sector])
    vsc_counter = np.zeros([num_level, num_sector])

    num_points = ptcloud.shape[0]

    for pt_idx in range(num_points):
        point = ptcloud[pt_idx, :]
        point_dist = np.sqrt(point[0] ** 2 + point[1] ** 2)
        if point_dist>max_dist:
            point_dist=max_dist

        idx_level, idx_sector = pt2ls(point, gap_level, gap_sector, num_level, min_height)

        if vsc_counter[idx_level, idx_sector] >= enough_large:
            continue
        vsc_storage[int(vsc_counter[idx_level, idx_sector]), idx_level, idx_sector] = point_dist
        vsc_counter[idx_level, idx_sector] = vsc_counter[idx_level, idx_sector] + 1
    vsc = np.amax(vsc_storage, axis=0)
    return vsc


def vsc2key(desp):
    return np.mean(desp, axis=0)


def distance_sc(sc1, sc2):
    num_sectors = sc1.shape[1]

    # repeate to move 1 columns
    _one_step = 1  # const
    sim_for_each_cols = np.zeros(num_sectors)
    for i in range(num_sectors):
        # Shift
        sc1 = np.roll(sc1, _one_step, axis=1)  # columne shift

        # compare
        sum_of_cossim = 0
        num_col_engaged = 0
        for j in range(num_sectors):
            col_j_1 = sc1[:, j]
            col_j_2 = sc2[:, j]
            if (~np.any(col_j_1) or ~np.any(col_j_2)):
                # to avoid being divided by zero when calculating cosine similarity
                # - but this part is quite slow in python, you can omit it.
                continue

            cossim = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
            sum_of_cossim = sum_of_cossim + cossim

            num_col_engaged = num_col_engaged + 1

        # save
        sim_for_each_cols[i] = sum_of_cossim / num_col_engaged

    yaw_diff = np.argmax(sim_for_each_cols) + 1  # because python starts with 0
    sim = np.max(sim_for_each_cols)
    dist = 1 - sim

    return dist, yaw_diff


class ScanContextManager:
    def __init__(self, shape=[20, 60], num_candidates=10,
                 threshold=0.010):  # defualt configs are same as the original paper
        self.shape = shape
        self.num_candidates = num_candidates
        self.threshold = threshold

        self.max_height = 3
        self.min_height = -3
        self.max_dist = 80


        self.ENOUGH_LARGE = 15000  # capable of up to ENOUGH_LARGE number of nodes
        self.ptclouds = [None] * self.ENOUGH_LARGE
        self.vertical_scancontexts = [None] * self.ENOUGH_LARGE
        self.keys = [None] * self.ENOUGH_LARGE

        self.curr_node_idx = 0

    def addNode(self, node_idx, ptcloud):
        vertical_sc = ptcloud2vsc(ptcloud, self.shape, self.max_height, self.min_height, self.max_dist)
        # 归一化,sc存的是高度值，exp存的是深度值,需要归一化处理
        #vertical_sc = 4 * vertical_sc / np.median(vertical_sc)
        # from matplotlib import pyplot as plt
        # plt.imshow(vertical_sc)
        key = vsc2key(vertical_sc)

        self.curr_node_idx = node_idx
        self.ptclouds[node_idx] = ptcloud
        self.vertical_scancontexts[node_idx] = vertical_sc
        self.keys[node_idx] = key

    def getPtcloud(self, node_idx):
        return self.ptclouds[node_idx]

    def detectLoop(self):
        exclude_recent_nodes = 30
        valid_recent_node_idx = self.curr_node_idx - exclude_recent_nodes

        if (valid_recent_node_idx < 1):
            return None, None, None
        else:
            # step 1
            key_history = np.array(self.keys[:valid_recent_node_idx])
            key_tree = spatial.KDTree(key_history)

            key_query = self.keys[self.curr_node_idx]
            _, nncandidates_idx = key_tree.query(key_query, k=self.num_candidates)

            # step 2
            query_vsc = self.vertical_scancontexts[self.curr_node_idx]

            nn_dist = 1.0  # initialize with the largest value of distance
            nn_idx = None
            nn_yawdiff = None
            for ith in range(self.num_candidates):
                candidate_idx = nncandidates_idx[ith]
                candidate_vsc = self.vertical_scancontexts[candidate_idx]
                dist, yaw_diff = distance_sc(candidate_vsc, query_vsc)
                if (dist < nn_dist):
                    nn_dist = dist
                    nn_yawdiff = yaw_diff
                    nn_idx = candidate_idx

            if (nn_dist < self.threshold):

                '''打印回环信息的，需要删除'''
                print("发现回环：匹配帧索引： ",self.curr_node_idx)
                print("对应回环索引：",nn_idx)
                print("Desp距离：", nn_dist)

                nn_yawdiff_deg = nn_yawdiff * (360 / self.shape[1])
                return nn_idx, nn_dist, nn_yawdiff_deg  # loop detected!
            else:
                nn_yawdiff_deg = nn_yawdiff * (360 / self.shape[1])
                return None, nn_dist, nn_yawdiff_deg
