# -*- coding: utf-8 -*- 
# @Time : 2021/3/12 10:26 
# @Author : CaiXin
# @File : ICPRegistrationTest.py
'''
测试得到的规律性结果：
1. open3d的p2l ICP方法对预估非常敏感，较好的预估能够得到较好的结果；较差的预估会致使其非常容易陷入局部极值点
2. myownicp对预估的敏感性相比之下不够高，可能会出现好的预估也不能导向非常好的jing 匹配结果，但是也有可能较差的预估却会得到一个好的结果
3. 在转弯幅度较大的阶段，myownicp的相对鲁棒想要好，但是小转弯时，open3d icp鲁棒性更好
'''

import time

from modules.ICPRegistration import *
from utils.UtilsPointcloud import readBinScan, random_sampling
import numpy as np
from modules.ICP import icp


def array_to_o3d_pointcloud(pointcloud_arr, robot_height=0.64, isIndoor=False, room_height=2.85):
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
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 20
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * coff
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
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
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, voxel_size, trans_init):
    distance_threshold = voxel_size * 0.5
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    loss = o3d.pipelines.registration.TukeyLoss(k=1)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        p2l,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return result


def refine_registration_myownicp(source, target, trans_init, tolerance, max_iteration=50, num_icp_points=10000):
    source_down = random_sampling(source, num_icp_points)
    tarhet_down = random_sampling(target, num_icp_points)
    rel_LO_pose, distacnces, iterations = icp(source_down, tarhet_down, init_pose=trans_init,
                                              tolerance=tolerance,
                                              max_iterations=max_iteration)
    return rel_LO_pose


def main():
    source_array = readBinScan('E:/data_odometry/dataset/sequences/09/velodyne/000230.bin')
    target_array = readBinScan('E:/data_odometry/dataset/sequences/09/velodyne/000229.bin')

    # tf,_,_,_=icp(source,target,None)
    # print(tf)

    source = array_to_o3d_pointcloud(source_array)
    target = array_to_o3d_pointcloud(target_array)

    # draw_registration_result(source, target, trans_init)
    # print("Initial alignment")
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #     source, target, threshold, trans_init)
    # print(evaluation)

    voxel_size = 0.05
    # 全局匹配
    # source_down, source_fpfh = preprocess_point_cloud(source, voxel_size=0.2, coff=1.5)
    # target_down, target_fpfh = preprocess_point_cloud(target, voxel_size=0.2, coff=1.5)
    # start = time.time()
    # result_ransac = execute_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)
    # print("Global registration took %.3f sec.\n" % (time.time() - start))
    # print(result_ransac)
    # print(result_ransac.transformation)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)

    # start = time.time()
    # result_fast = execute_fast_global_registration(source_down, target_down,
    #                                                source_fpfh, target_fpfh,
    #                                                voxel_size)
    # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    # print(result_fast)
    # print(result_fast.transformation)
    # draw_registration_result(source_down, target_down, result_fast.transformation)

    # ICP 精细匹配 point to Plane
    trans_init = np.identity(4)
    trans_init[0, 3] = 0.5


    start = time.time()
    # source=source.voxel_down_sample(0.25)
    # target=target.voxel_down_sample(0.25)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
    result_icp = refine_registration(source, target,
                                     voxel_size, trans_init)
    print("refine registration took %.3f sec.\n" % (time.time() - start))
    print(result_icp)
    print(result_icp.transformation)
    draw_registration_result(source, target, result_icp.transformation)


    start = time.time()
    odo_result=refine_registration_myownicp(source_array,target_array,trans_init,tolerance=0.0005,max_iteration=50,num_icp_points=50000)
    print("refine registration took %.3f sec.\n" % (time.time() - start))
    print(odo_result)
    draw_registration_result(source, target, odo_result)

    # start = time.time()
    # odo_result=refine_registration_myownicp(source_array,target_array,result_ransac.transformation,tolerance=0.0005,max_iteration=50,num_icp_points=10000)
    # print("refine registration took %.3f sec.\n" % (time.time() - start))
    # print(odo_result)
    # draw_registration_result(source, target, odo_result)

    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)
    #
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)

    # print("Apply point-to-plane ICP")
    # source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    # # o3d.visualization.draw_geometries([source],
    # #                                   zoom=0.3412,
    # #                                   front=[0.4257, -0.2125, -0.8795],
    # #                                   lookat=[2.6172, 2.0475, 1.532],
    # #                                   up=[-0.0694, -0.9768, 0.2024],
    # #                                   point_show_normal=True)
    # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    # reg_p2l = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # draw_registration_result(source, target, reg_p2l.transformation)


if __name__ == '__main__':
    main()
