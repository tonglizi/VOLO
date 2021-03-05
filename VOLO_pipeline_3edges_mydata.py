# -*- coding: utf-8 -*-
# @Time : 2020/8/27 23:13
# @Author : CaiXin
# @File : VOLO_pipeline_kitti.py
import random
import torch
from path import Path
import argparse
import numpy as np
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from modules.PointCloudMapping import MappingManager
from models import PoseExpNet
from utils.InverseWarp import pose_vec2mat
from modules.PoseGraphManager import *
from utils.UtilsMisc import *
from sympy import *

np.set_printoptions(precision=4)

parser = argparse.ArgumentParser(
    description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=341, type=int, help="Image height")
parser.add_argument("--img-width", default=427, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default=None, type=str,
                    help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument('--num_icp_points', type=int, default=5000)  # 5000 is enough for real time
parser.add_argument('--proposal', type=int, default=2)
parser.add_argument('--tolerance', type=float, default=0.001)
parser.add_argument('--scm_type', type=str, default='ring')
parser.add_argument('--num_rings', type=int, default=20)  # same as the original paper
parser.add_argument('--num_sectors', type=int, default=60)  # same as the original paper
parser.add_argument('--num_candidates', type=int, default=10)  # must be int
parser.add_argument('--try_gap_loop_detection', type=int, default=10)  # same as the original paper
parser.add_argument('--loop_threshold', type=float, default=0.11)  # 0.11 for sc, 0.015 for expand
parser.add_argument('--data_base_dir', type=str,
                    default='/your/path/.../data_odometry_velodyne/dataset/sequences')
parser.add_argument('--sequence_idx', type=str, default='09')
parser.add_argument('--save_gap', type=int, default=300)
parser.add_argument('--mapping', type=bool, default=True, help="build real-time map")
parser.add_argument('--vizmapping', type=bool, default=False, help="display the real-time map")
parser.add_argument('--optimization-period', type=int, default=30, help="Pose graph optimization period")

# CPU or GPU computing
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

args = parser.parse_args()
if args.scm_type == "ring":
    from modules.RingScanContextManager import *
elif args.scm_type == "vertical":
    from modules.VerticalScanContextManager import *
elif args.scm_type == "combined":
    from modules.CombinedScanContextManager import *


@torch.no_grad()
def main():
    from utils.UtilsMyData import test_framework_MyData as test_framework

    weights = torch.load(args.pretrained_posenet)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    sequences = [args.sequence_idx]
    framework = test_framework(dataset_dir, sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    optimized_errors = np.zeros((len(framework), 2), np.float32)
    ICP_iterations = np.zeros(len(framework))
    VO_processing_time = np.zeros(len(framework))
    ICP_iteration_time = np.zeros(len(framework))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

        '''绝对位姿列表初始化'''
        # 对齐到雷达坐标系，VO模型输出的带有尺度的绝对位姿
        abs_VO_poses = np.zeros((len(framework), 12))
        # 对齐到雷达坐标系，LO模型输出的带有尺度的绝对位姿
        abs_LO_poses = np.zeros((len(framework), 12))
        # 位姿估计值，对齐到相机坐标系下，和真值直接比较（仅适用于有相机坐标系下的真值）
        est_poses = np.zeros((len(framework), 12))
        est_poses[0] = np.identity(4)[:3, :].reshape(-1, 12)[0]

        '''帧间位姿列表初始化'''
        # 对齐到相机坐标系，VO模型输出的带有尺度的帧间位姿
        cur_VO_poses_C = np.zeros((len(framework), 12))
        # 对齐到雷达坐标系，VO模型输出的带有尺度的帧间位姿
        cur_VO_poses = np.zeros((len(framework), 6))
        # 对齐到雷达坐标系，VO模型输出的带有尺度的帧间位姿
        cur_LO_poses = np.zeros((len(framework), 12))
        '''尺度因子'''
        scale_factors = np.zeros((len(framework), 1))

    abs_VO_pose = np.identity(4)[:3, :]

    '''循环周期所需变量初始化'''
    # 用来估计尺度因子
    last_pose = np.identity(4)
    last_gap2_pose = np.identity(4)
    last_VO_pose = np.identity(4)
    # 周期中涉及到的点云，当前，前一个，前前一个
    curr_pts = None
    last_pts = None
    sec_last_pts = None
    # 当前节点和前一个，前前一个节点的帧间预估
    gap1_VO_pose = np.identity(4)
    gap2_VO_pose = np.identity(4)
    last_gap2_VO_pose = np.identity(4)

    # L和C的转换矩阵,对齐输入位姿到雷达坐标系
    Transform_matrix_L2C = np.identity(4)
    Transform_matrix_L2C[:3, :3] = np.array([[-1.51482698e-02, -9.99886648e-01, 5.36310553e-03],
                                             [-4.65337018e-03, -5.36307196e-03, -9.99969412e-01],
                                             [9.99870070e-01, -1.56647995e-02, -4.48880010e-03]])
    Transform_matrix_L2C[:3, -1:] = np.array([4.29029924e-03, -6.08539196e-02, -9.20346161e-02]).reshape(3, 1)
    Transform_matrix_L2C = GramSchmidtHelper(Transform_matrix_L2C)
    Transform_matrix_C2L = np.linalg.inv(Transform_matrix_L2C)

    pointClouds = loadPointCloud(args.dataset_dir + "/sequences/" + args.sequence_idx + "/velodyne")

    '''Pose Graph Manager (for back-end optimization) initialization'''
    PGM = PoseGraphManager()
    PGM.addPriorFactor()
    num_frames = len(tqdm(framework))
    save_dir = "result/" + args.sequence_idx
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3,
                                       save_gap=args.save_gap,
                                       num_frames=num_frames,
                                       seq_idx=args.sequence_idx,
                                       save_dir=save_dir)
    '''Scan Context Manager (for loop detection) initialization'''
    SCM = ScanContextManager(shape=[args.num_rings, args.num_sectors],
                             num_candidates=args.num_candidates,
                             threshold=args.loop_threshold)
    '''Mapping initialzation'''
    if args.mapping is True:
        Map = MappingManager()

    # for save the results as a video
    fig_idx = 1
    fig = plt.figure(fig_idx)
    writer = FFMpegWriter(fps=15)
    video_name = args.sequence_idx + "_" + str(args.num_icp_points) + "_prop@" + str(
        args.proposal) + "_tolerance@" + str(
        args.tolerance) + "_scm@" + str(args.scm_type) + "_thresh@" + str(args.loop_threshold) + ".mp4"
    num_frames_to_skip_to_show = 5
    num_frames_to_save = np.floor(num_frames / num_frames_to_skip_to_show)
    with writer.saving(fig, video_name, num_frames_to_save):  # this video saving part is optional

        for j, sample in enumerate(tqdm(framework)):
            '''
            ***************************************VO部分*******************************************
            '''
            imgs = sample['imgs']
            w, h = imgs[0].size
            if (not args.no_resize) and (h != args.img_height or w != args.img_width):
                imgs = [(np.array(img.resize((args.img_width, args.img_height)))).astype(np.float32) for img in imgs]
            imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

            ref_imgs = []
            for i, img in enumerate(imgs):
                img = torch.from_numpy(img).unsqueeze(0)
                img = ((img / 255 - 0.5) / 0.5).to(device)
                if i == len(imgs) // 2:
                    tgt_img = img
                else:
                    ref_imgs.append(img)

            startTimeVO = time.time()
            _, poses = pose_net(tgt_img, ref_imgs)
            VO_processing_time[j] = time.time() - startTimeVO

            poses = poses.cpu()[0]
            poses = torch.cat([poses[:len(imgs) // 2], torch.zeros(1, 6).float(), poses[len(imgs) // 2:]])

            inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

            rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
            tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

            transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)
            # print('**********DeepVO result: time_cost {:.3} s'.format(timeCostVO / (len(imgs) - 1)))
            # 将对[0 1 2]中间1的转换矩阵变成对0的位姿转换
            first_inv_transform = inv_transform_matrices[0]
            final_poses = first_inv_transform[:, :3] @ transform_matrices
            final_poses[:, :, -1:] += first_inv_transform[:, -1:]

            # gap2_VO_pose取final poses的第3项（0-2）
            gap2_VO_pose[:3, :] = final_poses[2]
            # cur_VO_pose取final poses的第2项（0-1）
            gap1_VO_pose[:3, :] = final_poses[1]

            # 尺度因子的确定：采用上一帧的LO输出位姿和VO输出位姿的尺度比值作为当前帧的尺度因子，初始尺度为1
            if j == 0:
                scale_factor = 7
            else:
                scale_factor = math.sqrt(np.sum(last_pose[:3, -1] ** 2) / np.sum(last_VO_pose[:3, -1] ** 2))
            scale_factors[j] = scale_factor
            last_VO_pose = copy.deepcopy(gap1_VO_pose)  # 注意深拷贝
            # 先尺度修正，再对齐坐标系，施密特正交化避免病态矩阵
            gap2_VO_pose[:3, -1:] = gap2_VO_pose[:3, -1:] * scale_factor
            gap2_VO_pose = Transform_matrix_C2L @ gap2_VO_pose @ np.linalg.inv(Transform_matrix_C2L)
            gap2_VO_pose = GramSchmidtHelper(gap2_VO_pose)

            gap1_VO_pose[:3, -1:] = gap1_VO_pose[:3, -1:] * scale_factor
            cur_VO_poses_C[j] = gap1_VO_pose[:3, :].reshape(-1, 12)[0]
            gap1_VO_pose = Transform_matrix_C2L @ gap1_VO_pose @ np.linalg.inv(Transform_matrix_C2L)
            gap1_VO_pose = GramSchmidtHelper(gap1_VO_pose)

            '''*************************LO部分******************************************'''
            # 初始化
            if j == 0:
                last_pts = random_sampling(pointClouds[j], args.num_icp_points)
                sec_last_pts = last_pts
                SCM.addNode(j, last_pts)
                if args.mapping is True:
                    Map.curr_se3 = PGM.curr_se3
                    Map.curr_local_ptcloud = last_pts
                    Map.updateMap()

            curr_pts = random_sampling(pointClouds[j + 1], args.num_icp_points)

            from modules.ICP import icp
            # 选择LO的初值预估，分别是无预估，上一帧位姿，VO位姿
            if args.proposal == 0:
                init_pose_1 = None
                init_pose_2 = None
            elif args.proposal == 1:
                init_pose_1 = last_pose
                init_pose_2 = last_gap2_pose
            elif args.proposal == 2:
                init_pose_1 = gap1_VO_pose
                init_pose_2 = last_gap2_VO_pose

            startTime = time.time()
            icp_odom_transform_1, distacnces, iterations = icp(curr_pts, last_pts, init_pose=init_pose_1,
                                                               tolerance=args.tolerance,
                                                               max_iterations=50)
            if j > 0:
                icp_odom_transform_2, distacnces_2, iterations_2 = icp(last_pts, sec_last_pts, init_pose=init_pose_2,
                                                                       tolerance=args.tolerance,
                                                                       max_iterations=50)
            else:
                icp_odom_transform_2 = icp_odom_transform_1
                distacnces_2 = distacnces
                iterations_2 - iterations

            ICP_iteration_time[j] = time.time() - startTime
            ICP_iterations[j] = (iterations + iterations_2) / 2

            '''更新指针'''
            # 点云
            sec_last_pts = last_pts
            last_pts = curr_pts
            # gap2预估位姿
            last_gap2_VO_pose = gap2_VO_pose
            # icp 输出位姿
            last_pose = icp_odom_transform_1
            last_gap2_pose = icp_odom_transform_2

            print("LO优化后的位姿,mean_dis: ", np.asarray(distacnces).mean())
            print(icp_odom_transform_1)

            '''Update loop detection nodes'''
            SCM.addNode(j + 1, curr_pts)
            '''Update the edges and nodes of pose graph'''
            PGM.curr_node_idx = j + 1
            PGM.curr_se3 = np.matmul(PGM.curr_se3, icp_odom_transform_1)
            PGM.addOdometryFactor(icp_odom_transform_1, -1)
            PGM.addOdometryFactor(icp_odom_transform_2, -2)
            PGM.sec_prev_node_idx = PGM.prev_node_idx
            PGM.prev_node_idx = PGM.curr_node_idx

            # 建图更新
            if args.mapping is True:
                Map.curr_local_ptcloud = curr_pts
                Map.curr_se3 = PGM.curr_se3
                Map.updateMap()

            # loop detection and optimize the graph
            if (PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.try_gap_loop_detection == 0):
                # 1/ loop detection
                loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
                if (loop_idx == None):  # NOT FOUND
                    pass
                else:
                    print("Loop event detected: ", PGM.curr_node_idx, loop_idx, loop_dist)
                    # 2-1/ add the loop factor
                    loop_scan_down_pts = SCM.getPtcloud(loop_idx)
                    loop_transform, _, _ = icp(curr_pts, loop_scan_down_pts,
                                               init_pose=yawdeg2se3(yaw_diff_deg), max_iterations=20)
                    PGM.addLoopFactor(loop_transform, loop_idx)

                    # 2-2/ graph optimization
                    PGM.optimizePoseGraph()

                    # 2-2/ save optimized poses
                    ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)

                    # 2-3/ updateMap
                    if args.mapping is True:
                        Map.optimizeGlobalMap(PGM.graph_optimized, PGM.curr_node_idx)

            # 定时进行位姿图优化
            if (PGM.curr_node_idx > 1 and PGM.curr_node_idx % args.optimization_period == 0):
                # 2-2/ graph optimization
                PGM.optimizePoseGraph()

                # 2-2/ save optimized poses
                ResultSaver.saveOptimizedPoseGraphResult(PGM.curr_node_idx, PGM.graph_optimized)

                # 2-3/ updateMap
                if args.mapping is True:
                    Map.optimizeGlobalMap(PGM.graph_optimized, PGM.curr_node_idx)

            # save the ICP odometry pose result (no loop closure)
            ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx)
            if (j % num_frames_to_skip_to_show == 0):
                ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
                writer.grab_frame()
            if args.vizmapping is True:
                Map.vizMapWithOpen3D()

            # 对齐到雷达坐标系下，VO输出的绝对位姿
            abs_VO_pose[:, :3] = gap1_VO_pose[:3, :3] @ abs_VO_pose[:, :3]
            abs_VO_pose[:, -1:] += gap1_VO_pose[:3, -1:]

            if args.output_dir is not None:
                predictions_array[j] = final_poses
                # cur_VO_poses[j]=cur_VO_pose[:3, :].reshape(-1, 12)[0]
                cur_LO_poses[j] = icp_odom_transform_1[:3, :].reshape(-1, 12)[0]
                abs_VO_poses[j] = abs_VO_pose[:3, :].reshape(-1, 12)[0]
                abs_LO_poses[j] = PGM.curr_se3[:3, :].reshape(-1, 12)[0]
                est_pose = Transform_matrix_L2C @ PGM.curr_se3 @ np.linalg.inv(Transform_matrix_L2C)
                est_poses[j + 1] = est_pose[:3, :].reshape(-1, 12)[0]

        if args.mapping is True:
            Map.saveMap2File('map_' + args.sequence_idx + "_" + str(args.num_icp_points) + "_prop@" + str(
                args.proposal) + "_tolerance@" + str(args.tolerance) + "_scm@" + str(args.scm_type) +
                             "_thresh@" + str(args.loop_threshold) + '.pcd')
        if args.output_dir is not None:
            # np.save(output_dir / 'predictions.npy', predictions_array)
            np.savetxt(output_dir / 'scale_factors.txt', scale_factors)
            np.savetxt(output_dir / 'cur_VO_poses_C.txt', cur_VO_poses_C)
            np.savetxt(output_dir / 'cur_VO_poses.txt', cur_VO_poses)
            np.savetxt(output_dir / 'abs_VO_poses.txt', abs_VO_poses)
            np.savetxt(output_dir / 'cur_LO_poses.txt', cur_LO_poses)
            np.savetxt(output_dir / 'abs_LO_poses.txt', abs_LO_poses)
            np.savetxt(output_dir / 'iterations.txt', ICP_iterations)
            np.savetxt(output_dir / 'est_kitti_{0}_poses.txt'.format(args.sequence_idx), est_poses)

        # VO输出位姿的精度指标
        mean_errors = errors.mean(0)
        std_errors = errors.std(0)
        error_names = ['ATE', 'RE']
        print('')
        print("VO_Results")
        print("\t {:>10}, {:>10}".format(*error_names))
        print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
        print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

        # LO二次优化后的精度指标
        optimized_mean_errors = optimized_errors.mean(0)
        optimized_std_errors = optimized_errors.std(0)
        optimized_error_names = ['optimized_ATE', 'optimized_RE']
        print('')
        print("LO_optimized_Results")
        print("\t {:>10}, {:>10}".format(*optimized_error_names))
        print("mean \t {:10.4f}, {:10.4f}".format(*optimized_mean_errors))
        print("std \t {:10.4f}, {:10.4f}".format(*optimized_std_errors))

        # 迭代次数
        mean_iterations = ICP_iterations.mean()
        std_iterations = ICP_iterations.std()
        _names = ['iteration']
        print('')
        print("LO迭代次数")
        print("\t {:>10}".format(*_names))
        print("mean \t {:10.4f}".format(mean_iterations))
        print("std \t {:10.4f}".format(std_iterations))

        # 迭代时间
        mean_iter_time = ICP_iteration_time.mean()
        std_iter_time = ICP_iteration_time.std()
        _names = ['iter_time']
        print('')
        print("LO迭代时间：单位/s")
        print("\t {:>10}".format(*_names))
        print("mean \t {:10.4f}".format(mean_iter_time))
        print("std \t {:10.4f}".format(std_iter_time))


def compute_LO_pose_error(gt, odom_transform, Transform_matrix_L2C=None):
    gt_pose = gt[1]
    odom_transform_L2C = Transform_matrix_L2C @ odom_transform @ np.linalg.inv(Transform_matrix_L2C)

    ATE = np.linalg.norm((gt_pose[:3, -1] - odom_transform_L2C[:3, -1]).reshape(-1))

    # Residual matrix to which we compute angle's sin and cos
    R = gt_pose[:3, :3] @ np.linalg.inv(odom_transform_L2C[:3, :3])
    s = np.linalg.norm([R[0, 1] - R[1, 0],
                        R[1, 2] - R[2, 1],
                        R[0, 2] - R[2, 0]])
    c = np.trace(R) - 1
    # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
    RE = np.arctan2(s, c)
    return ATE, RE


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    # print("scale_factor: %s", scale_factor)
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE / snippet_length, RE / snippet_length


def random_sampling(orig_points, num_points):
    assert orig_points.shape[0] > num_points
    points_down_idx = random.sample(range(orig_points.shape[0]), num_points)
    down_points = orig_points[points_down_idx, :]
    return down_points


def loadPointCloud(rootdir):
    files = os.listdir(rootdir)
    files.sort()
    pointclouds = []
    for file in files:
        if not os.path.isdir(file):
            scan = np.fromfile(rootdir + "/" + file, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            ptcloud_xyz = scan[:, :-1]
            print(ptcloud_xyz.shape)
            pointclouds.append(ptcloud_xyz)
    return pointclouds


def GramSchmidtHelper(transformation):
    a1 = Matrix(transformation[0, :3])
    a2 = Matrix(transformation[1, :3])
    a3 = Matrix(transformation[2, :3])
    so3 = [a1, a2, a3]
    O = GramSchmidt(so3, True)
    O = np.array(O)
    transformation[:3, :3] = O[:3, :3].reshape(3, 3)
    return transformation


if __name__ == '__main__':
    main()
