# -*- coding: utf-8 -*- 
# @Time : 2021/3/12 10:14 
# @Author : CaiXin
# @File : ICPRegistration.py

'''
The ICP module is based on Open3d API
'''
import copy
import open3d as o3d
import numpy as np


def icp(source, target, trans_init, threshold=0.05, downsample_voxpel_size=0.2, coff=1.5, indoor=True):
    '''
    重要参数配置：（按照重要程度排序）
    1. downsample_voxpel_size: 0.2；非常重要的参数！过小时，降采样不足，长廊效应影响明显；过大时，降采样过度，信息损失严重
    2. estimate_normals()中的radius,决定了能够较好寻找到点的法向量，对于稀疏点区域，一定要将半径放得越大越好，才能找的比较好，这里选了20倍的voxel_size
    3. coff:计算FHFP特征的范围半径系数，直接影响粗匹配；经过循环求值，发现1.5比较合适
    4.threshold：指定global和refine的配齐阈值标准
    :param source:
    :param target:
    :param trans_init:
    :param threshold:
    :param downsample_voxpel_size:
    :return:
    '''
    source = array_to_o3d_pointcloud(source,isIndoor=indoor)
    target = array_to_o3d_pointcloud(target,isIndoor=indoor)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size=downsample_voxpel_size, coff=coff)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=downsample_voxpel_size, coff=coff)

    # Global registration: coarse matching
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                threshold)
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    # ICP point to Plane refinement registration: fine mayching
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20 * downsample_voxpel_size, max_nn=30))
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20 * downsample_voxpel_size, max_nn=30))
    result_icp = refine_registration(source, target,
                                     threshold, result_ransac.transformation)
    draw_registration_result(source, target, result_icp.transformation)
    return result_icp.transformation, result_icp.fitness, result_icp.inlier_rmse, result_icp.correspondence_set


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def array_to_o3d_pointcloud(pointcloud_arr, robot_height=0.64,isIndoor=False, room_height=2.85):
    # points removal: remove ground and top
    # pointcloud_filtered = []
    # for i in range(len(pointcloud_arr)):
    #     if pointcloud_arr[i, 2] > 0.1 - robot_height :
    #             if isIndoor:
    #                 if pointcloud_arr[i, 2] < room_height - robot_height - 0.35:
    #                     pointcloud_filtered.append(pointcloud_arr[i])
    #             else:
    #                 pointcloud_filtered.append(pointcloud_arr[i])
    # pointcloud_filtered = np.asarray(pointcloud_filtered)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pointcloud_arr)
    return pointcloud


def preprocess_point_cloud(pcd, voxel_size, coff):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 20
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * coff
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, threshold):
    distance_threshold = threshold * 0.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, threshold, trans_init):
    distance_threshold = threshold * 0.5
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    loss = o3d.pipelines.registration.TukeyLoss(k=0.02)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        p2l,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    return result
