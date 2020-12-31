# -*- coding: utf-8 -*-
# @Time : 2020/9/10 16:37
# @Author : CaiXin
# @File : PointCloudMapping.py

import open3d as o3d
import argparse

args=argparse.ArgumentParser(description="Input map filename (.pcd)",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args.add_argument("filename",type=str,default=".")
args=args.parse_args()

def showPointcloudFromFile(filename=None):
    if filename is None:
        print("No file input...")
    else:
        pointcloud = o3d.io.read_point_cloud(filename)
        o3d.visualization.draw_geometries([pointcloud], window_name='Map_' + filename)

showPointcloudFromFile(args.filename)
