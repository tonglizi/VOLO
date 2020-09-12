# -*- coding: utf-8 -*- 
# @Time : 2020/9/10 16:37 
# @Author : CaiXin
# @File : PointCloudMapping.py
import random
import numpy as np
import open3d as o3d

from utils.UtilsMisc import getGraphNodePose


def random_sampling(orig_points, num_points):
    if orig_points.shape[0] > num_points:
        points_down_idx = random.sample(range(orig_points.shape[0]), num_points)
        down_points = orig_points[points_down_idx, :]
        return down_points
    else:
        return orig_points


def showPointcloudFromFile(filename=None):
    if filename is None:
        print("No file input...")
    else:
        pointcloud = o3d.io.read_point_cloud(filename)
        o3d.draw_geometries([pointcloud], window_name='Map_' + filename)


class MappingManager:
    def __init__(self):
        # internal vars
        self.global_ptcloud = None
        self.curr_local_ptcloud = None
        self.ptcloud_list = []
        self.curr_se3 = np.identity(4)
        # visualizaton
        self.pointcloud = o3d.geometry.PointCloud()
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        self.viz.get_render_option().point_size = 2.0
        self.viz.get_render_option().point_color_option = o3d.PointColorOption.XCoordinate
        self.viz.add_geometry(self.pointcloud)
        self.viz.add_geometry(o3d.create_mesh_coordinate_frame(size=400, origin=[0., 0., 0.]))

    def updateMap(self, down_points=100):
        # 将点云坐标转化为齐次坐标（x,y,z）->(x,y,z,1)
        self.curr_local_ptcloud = random_sampling(self.curr_local_ptcloud, down_points)
        tail = np.zeros((self.curr_local_ptcloud.shape[0], 1))
        tail.fill(1)
        self.curr_local_ptcloud = np.concatenate([self.curr_local_ptcloud, tail], axis=1)
        self.ptcloud_list.append(self.curr_local_ptcloud)
        sub_ptcloud_map = self.curr_se3 @ (self.curr_local_ptcloud.T)

        # concatenate the latest local pointcloud into global pointcloud
        if (self.global_ptcloud is None):
            self.global_ptcloud = sub_ptcloud_map.T[:, :3]
        else:
            self.global_ptcloud = np.concatenate([self.global_ptcloud, sub_ptcloud_map.T[:, :3]])

    def optimizeGlobalMap(self, graph_optimized, curr_node_idx=None):
        global_ptcloud = None
        se3 = np.identity(4)
        for i in range(curr_node_idx):
            pose_trans, pose_rot = getGraphNodePose(graph_optimized, i)
            se3[:3, :3] = pose_rot
            se3[:3, 3] = pose_trans
            sub_ptcloud_map = se3 @ (self.ptcloud_list[i].T)
            if (global_ptcloud is None):
                global_ptcloud = sub_ptcloud_map.T[:, :3]
            else:
                global_ptcloud = np.concatenate([global_ptcloud, sub_ptcloud_map.T[:, :3]])

        self.global_ptcloud = global_ptcloud
        # correct current pose
        pose_trans, pose_rot = getGraphNodePose(graph_optimized, curr_node_idx)
        self.curr_se3[:3, :3] = pose_rot
        self.curr_se3[:3, 3] = pose_trans

    def vizMapWithOpen3D(self):
        if self.global_ptcloud is not None:
            self.pointcloud.points = o3d.utility.Vector3dVector(self.global_ptcloud)
            self.viz.update_geometry()
            self.viz.poll_events()
            self.viz.update_renderer()

    def saveMap2File(self, filename=None):
        if self.global_ptcloud is not None:
            o3d.io.write_point_cloud(filename, self.pointcloud)

    # use matplotlib to display 3d ptcloud, low efficiency
    # def vizMap(self, fig_idx=None):
    #     import matplotlib.pyplot as plt
    #     x = self.global_ptcloud[:, 0]
    #     y = self.global_ptcloud[:, 1]
    #     z = self.global_ptcloud[:, 2]
    #     fig = plt.figure(fig_idx)
    #     plt.clf()
    #     ax = fig.add_subplot(111, projection='3d')
    #     plt.title("CurrentMap")
    #     ax.scatter(-y, x, z, c=y, marker='.', s=2, linewidth=0, alpha=1, cmap='summer')
    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')
    #     plt.draw()
    #     plt.pause(0.01)


# # ******test script*******
# curr_se3 = np.identity(4)
# curr_se3[:3, -1] = [1, 1, 0]
# map = MappingManager()
# map.curr_se3 = curr_se3
# points = np.random.random((1000, 3))
# i = 0
# while (i < 100):
#     i += 1
#     points -= 0.001
#     map.curr_local_ptcloud = points
#     map.updateMap()
#     # map.vizMap(1)
#     map.vizMapWithOpen3D()
# map.saveMap2File("map.pcd")
# showPointcloudFromFile("map.pcd")
