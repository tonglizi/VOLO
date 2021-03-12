# -*- coding: utf-8 -*- 
# @Time : 2020/9/10 16:37 
# @Author : CaiXin
# @File : PointCloudMapping.py
import numpy as np
import open3d as o3d
from queue import Queue

from utils.UtilsMisc import getGraphNodePose
from utils.UtilsPointcloud import random_sampling


def showPointcloudFromFile(filename=None):
    if filename is None:
        print("No file input...")
    else:
        pointcloud = o3d.io.read_point_cloud(filename)
        o3d.visualization.draw_geometries([pointcloud], window_name='Map_' + filename)


def integrateSubmap(sub_ptcloud_list, se3_to_curr_ptcloud_list):
    assert sub_ptcloud_list.qsize() == len(se3_to_curr_ptcloud_list)
    size = sub_ptcloud_list.qsize()
    sub_ptcloud = None
    if size > 0:
        for i in range(size):
            ptcloud_item = sub_ptcloud_list.__getitem__(i)
            ptcloud_to_curr_ptcloud_frame = se3_to_curr_ptcloud_list[i] @ ptcloud_item.T
            if sub_ptcloud is None:
                sub_ptcloud = ptcloud_to_curr_ptcloud_frame.T[:, :3]
            else:
                sub_ptcloud = np.concatenate([sub_ptcloud, ptcloud_to_curr_ptcloud_frame.T[:, :3]])
    return sub_ptcloud


def transform_to_curr_ptclound(se3_list):
    se3_to_curr_ptcloud_list = []
    size = se3_list.qsize()
    curr_se3 = se3_list.__getitem__(size - 1)
    inv_curr_se3 = np.linalg.inv(curr_se3)
    for i in range(size):
        se3_to_curr_ptcloud = np.matmul(inv_curr_se3, se3_list[i])
        se3_to_curr_ptcloud_list.append(se3_to_curr_ptcloud)
    return se3_to_curr_ptcloud_list


class MappingManager:
    def __init__(self, k=5):
        # internal vars
        self.global_ptcloud = None  # numpy type
        self.curr_local_ptcloud = None
        self.ptcloud_list = []
        self.curr_se3 = None

        self.k = k  # 创建子图的点云数
        self.sub_ptcloud = None  # numpy type 维护最近k帧点云形成的子图
        self.se3_list = IndexableQueue(maxsize=self.k)  # 维护最近k帧点云的全局位姿
        self.sub_ptcloud_list = IndexableQueue(maxsize=self.k)  # 维护最近k帧点云
        self.se3_to_curr_ptcloud_list = []  # 维护最近k帧点云相对于当前点云的位姿

        self.pointcloud = o3d.geometry.PointCloud()  # open3d type
        # visualizaton
        self.viz = None

    def updateMap(self, curr_se3, curr_local_ptcloud, down_points=100,submap_points=10000):
        '''
        创建并更新地图，正常情况下每一个位姿点都需要进行一次
        :param curr_se3:
        :param curr_local_ptcloud:
        :param down_points:
        :return:
        '''
        # 更新字段
        self.curr_se3 = curr_se3
        self.curr_local_ptcloud = curr_local_ptcloud
        self.submap_points=submap_points
        # 将点云坐标转化为齐次坐标（x,y,z）->(x,y,z,1)
        self.curr_local_ptcloud = random_sampling(self.curr_local_ptcloud, down_points)
        tail = np.zeros((self.curr_local_ptcloud.shape[0], 1))
        tail.fill(1)
        self.curr_local_ptcloud = np.concatenate([self.curr_local_ptcloud, tail], axis=1)
        self.ptcloud_list.append(self.curr_local_ptcloud)
        curr_local_ptcloud_into_global_map = self.curr_se3 @ (self.curr_local_ptcloud.T)

        # concatenate the latest local pointcloud into global pointcloud
        if (self.global_ptcloud is None):
            self.global_ptcloud = curr_local_ptcloud_into_global_map.T[:, :3]
        else:
            self.global_ptcloud = np.concatenate([self.global_ptcloud, curr_local_ptcloud_into_global_map.T[:, :3]])
        # updata open3d_pointscloud
        self.pointcloud.points = o3d.utility.Vector3dVector(self.global_ptcloud)

        # 子图相对位姿和点云维护
        if self.sub_ptcloud_list.qsize() == self.k and (not self.sub_ptcloud_list.empty()):
            self.sub_ptcloud_list.get()
            self.se3_list.get()
        downsample_ptcloud=random_sampling(curr_local_ptcloud,self.submap_points)
        # 将点云坐标转化为齐次坐标（x,y,z）->(x,y,z,1)
        tail = np.zeros((downsample_ptcloud.shape[0], 1))
        tail.fill(1)
        downsample_ptcloud = np.concatenate([downsample_ptcloud, tail], axis=1)
        self.sub_ptcloud_list.put(downsample_ptcloud)
        self.se3_list.put(curr_se3)

    def optimizeGlobalMap(self, graph_optimized, curr_node_idx=None):
        global_ptcloud = None
        se3 = np.identity(4)
        for i in range(curr_node_idx+1):
            pose_trans, pose_rot = getGraphNodePose(graph_optimized, i)
            se3[:3, :3] = pose_rot
            se3[:3, 3] = pose_trans
            ptcloud_to_global_map = se3 @ (self.ptcloud_list[i].T)
            if (global_ptcloud is None):
                global_ptcloud = ptcloud_to_global_map.T[:, :3]
            else:
                global_ptcloud = np.concatenate([global_ptcloud, ptcloud_to_global_map.T[:, :3]])

        self.global_ptcloud = global_ptcloud
        self.pointcloud.points = o3d.utility.Vector3dVector(self.global_ptcloud)
        # correct current pose
        pose_trans, pose_rot = getGraphNodePose(graph_optimized, curr_node_idx)
        self.curr_se3[:3, :3] = pose_rot
        self.curr_se3[:3, 3] = pose_trans

        # 子图相对位姿维护
        size=self.se3_list.qsize()
        if size>0:
            self.se3_list = IndexableQueue(maxsize=self.k)
            for i in range(size):
                pose_trans, pose_rot = getGraphNodePose(graph_optimized, curr_node_idx-(size-1)+i)
                se3[:3, :3] = pose_rot
                se3[:3, 3] = pose_trans
                self.se3_list.put(se3)

    def vizMapWithOpen3D(self):
        if self.viz == None:
            self.viz = o3d.visualization.Visualizer()
            self.viz.create_window()
            # self.viz.get_render_option().point_size = 2.0
            # self.viz.get_render_option().point_color_option = o3d.visualization.PointColorOption.XCoordinate
            self.viz.add_geometry(self.pointcloud)
            # self.viz.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=400, origin=[0., 0., 0.]))
        if self.global_ptcloud is not None:
            self.pointcloud.points = o3d.utility.Vector3dVector(self.global_ptcloud)
            self.viz.update_geometry()
            self.viz.poll_events()
            self.viz.update_renderer()

    def saveMap2File(self, filename=None):
        if self.global_ptcloud is not None:
            o3d.io.write_point_cloud(filename, self.pointcloud)

    def getSubMap(self):
        self.se3_to_curr_ptcloud_list = transform_to_curr_ptclound(self.se3_list)
        self.sub_ptcloud = integrateSubmap(self.sub_ptcloud_list, self.se3_to_curr_ptcloud_list)
        self.sub_ptcloud=random_sampling(self.sub_ptcloud,self.submap_points)
        return self.sub_ptcloud


class IndexableQueue(Queue):
    def __getitem__(self, index):
        with self.mutex:
            return self.queue[index]
