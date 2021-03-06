# -*- coding: utf-8 -*- 
# @Time : 2021/3/16 20:07 
# @Author : CaiXin
# @File : TriangulationTest.py

import open3d as o3d
import numpy as np

from utils.UtilsPointcloud import readBinScan


def array_to_o3d_pointcloud(pointcloud_arr):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pointcloud_arr)
    return pointcloud

def main():
    pcd = o3d.io.read_point_cloud('C:/Users/93121/Desktop/5.pcd')
    # pcd = readBinScan('E:/mydataset/sequences/20210302_164405/velodyne/0000.bin')
    # pcd = array_to_o3d_pointcloud(pcd)
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2]))
    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    main()
