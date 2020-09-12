# -*- coding: utf-8 -*- 
# @Time : 2020/9/10 16:37 
# @Author : CaiXin
# @File : PointCloudMapping.py
import random
import numpy as np
import open3d as o3d


def random_sampling(orig_points, num_points):
    if orig_points.shape[0] > num_points:
        points_down_idx = random.sample(range(orig_points.shape[0]), num_points)
        down_points = orig_points[points_down_idx, :]
        return down_points
    else:
        return orig_points


class MapManager:
    def __init__(self):
        self.curr_ptcloud_map = None
        self.curr_ptcloud = None
        self.curr_se3 = np.identity(4)
        self.pointcloud = o3d.geometry.PointCloud()
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        self.viz.get_render_option().point_size = 2.0
        self.viz.get_render_option().point_color_option = o3d.PointColorOption.XCoordinate
        self.viz.add_geometry(self.pointcloud)
        self.viz.add_geometry(o3d.create_mesh_coordinate_frame(size=10, origin=[0., 0., 0.]))

    def updateMap(self, down_points=100):
        # 将点云坐标转化为齐次坐标（x,y,z）->(x,y,z,1)
        self.curr_ptcloud = random_sampling(self.curr_ptcloud, down_points)
        count = self.curr_ptcloud.shape[0]
        tail = np.zeros((count, 1))
        tail.fill(1)
        self.curr_ptcloud = np.concatenate([self.curr_ptcloud, tail], axis=1)
        sub_ptcloud_map = self.curr_se3 @ (self.curr_ptcloud.T)
        if (self.curr_ptcloud_map is None):
            self.curr_ptcloud_map = sub_ptcloud_map.T[:, :3]
        else:
            self.curr_ptcloud_map = np.concatenate([self.curr_ptcloud_map, sub_ptcloud_map.T[:, :3]])

    def vizMapWithOpen3D(self):
        self.pointcloud.points = o3d.utility.Vector3dVector(self.curr_ptcloud_map)
        self.viz.update_geometry()
        self.viz.poll_events()
        self.viz.update_renderer()

    # use matplotlib to display 3d ptcloud, low efficiency
    def vizMap(self, fig_idx=None):
        import matplotlib.pyplot as plt
        x = self.curr_ptcloud_map[:, 0]
        y = self.curr_ptcloud_map[:, 1]
        z = self.curr_ptcloud_map[:, 2]
        fig = plt.figure(fig_idx)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        plt.title("CurrentMap")
        color = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        ax.scatter(-y, x, z, c=y, marker='.', s=2, linewidth=0, alpha=1, cmap='summer')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.draw()
        plt.pause(0.01)

# # ******test script*******
# curr_se3 = np.identity(4)
# curr_se3[:3, -1] = [1, 1, 0]
# map = MapManager()
# map.curr_se3 = curr_se3
# points = np.random.random((1000,3))
# while True:
#     points -= 0.001
#     map.curr_ptcloud = points
#     map.updateMap()
#     # map.vizMap(1)
#     map.vizMapWithOpen3D()
# print(map.curr_ptcloud_map)
