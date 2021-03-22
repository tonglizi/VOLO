# -*- coding: utf-8 -*-
# @Time : 2020/8/27 23:13
# @Author : CaiXin
# @File : VOLO_pipeline.py
'''
用来测试VOLO
有位姿图优化PGM模块，做位姿记录，同时如果开启回环检测时会进行图优化
开关介绍：
--isKitti:适用于带有位姿真值的kitti测试集；能够额外输出和真值比较得到误差
'''
import hashlib
import torch
from path import Path
import argparse
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from modules.ICP import icp
from modules.ICPRegistration import p2l_icp
from modules.PointCloudMapping import MappingManager
from models import PoseExpNet
from utils.InverseWarp import pose_vec2mat
from modules.PoseGraphManager import *
from utils.UtilsMisc import *
from sympy import *

from utils.UtilsPointcloud import loadPointCloud, random_sampling

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
parser.add_argument("--output-dir", default='./result/', type=str,
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
parser.add_argument('--map-down-points', type=int, default=200, help="mapping density")
parser.add_argument('--isKitti', type=bool, default=False,
                    help="Only for KITTI dataset test, if not, then for mydataset")
parser.add_argument('--scan2submap', type=bool, default=False,
                    help="ICP matching method: scan to scan (off); scan to sub map (on) ")
parser.add_argument('--icp-version', type=int, default=1,
                    help="options for ICP implementations: 0 is my own, 1 is from open3d ")
# CPU or GPU computing
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

args = parser.parse_args()

def MD5_ID(pretrained_posenet):
    md = hashlib.md5()  # 创建md5对象
    md.update(str(pretrained_posenet).encode(encoding='utf-8'))
    return md.hexdigest()


@torch.no_grad()
def main():
    '''加载训练后的模型'''
    weights = torch.load(args.pretrained_posenet)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    # 网络模型的MD5 ID
    net_ID = MD5_ID(args.pretrained_posenet)
    # L和C的转换矩阵,对齐输入位姿到雷达坐标系
    Transform_matrix_L2C = np.identity(4)
    '''Kitti switch'''
    if args.isKitti:
        from utils.UtilsKitti import test_framework_KITTI as test_framework
        save_dir = os.path.join(args.output_dir, "kitti", args.sequence_idx, 'net_' + net_ID)
        downsample_img_height = 128
        downsample_img_width = 416
        Transform_matrix_L2C[:3, :3] = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
                                                 [1.480249e-02, 7.280733e-04, -9.998902e-01],
                                                 [9.998621e-01, 7.523790e-03, 1.480755e-02]])
        Transform_matrix_L2C[:3, -1:] = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    else:
        from utils.UtilsMyData import test_framework_MyData as test_framework
        save_dir = os.path.join(args.output_dir, "mydataset", args.sequence_idx, 'net_' + net_ID)
        downsample_img_height = args.img_height
        downsample_img_width = args.img_width
        Transform_matrix_L2C[:3, :3] = np.array([[-1.51482698e-02, -9.99886648e-01, 5.36310553e-03],
                                                 [-4.65337018e-03, -5.36307196e-03, -9.99969412e-01],
                                                 [9.99870070e-01, -1.56647995e-02, -4.48880010e-03]])
        Transform_matrix_L2C[:3, -1:] = np.array([4.29029924e-03, -6.08539196e-02, -9.20346161e-02]).reshape(3, 1)
    Transform_matrix_L2C = GramSchmidtHelper(Transform_matrix_L2C)
    Transform_matrix_C2L = np.linalg.inv(Transform_matrix_L2C)
    '''载入测试数据:图像和点云'''
    dataset_dir = Path(args.dataset_dir)
    sequences = [args.sequence_idx]
    framework = test_framework(dataset_dir, sequences, seq_length)
    pointcloud_dir = os.path.join(args.dataset_dir, 'sequences', args.sequence_idx, 'velodyne')
    pointClouds = loadPointCloud(pointcloud_dir)
    print('{} snippets to test'.format(len(framework)))
    '''误差初始化'''
    errors = np.zeros((len(framework), 2), np.float32)
    optimized_errors = np.zeros((len(framework), 2), np.float32)
    ##############################################################
    '''输出到文件中的数据初始化'''
    num_poses = len(framework) - (seq_length - 2)
    '''效率'''
    VO_processing_time = np.zeros(num_poses - 1)
    ICP_iterations = np.zeros(num_poses - 1)
    ICP_fitness = np.zeros(num_poses - 1)
    ICP_iteration_time = np.zeros(num_poses - 1)
    '''绝对位姿列表初始化'''
    # 对齐到雷达坐标系，VO模型输出的带有尺度的绝对位姿
    abs_VO_poses = np.zeros((num_poses, 12))
    abs_VO_poses[0] = np.identity(4)[:3, :].reshape(12)
    abs_VO_pose = np.identity(4)
    # 位姿估计值，对齐到相机坐标系下，和真值直接比较（仅适用于有相机坐标系下的真值）
    est_poses = np.zeros((num_poses, 12))
    est_poses[0] = np.identity(4)[:3, :].reshape(12)
    '''帧间位姿列表初始化'''
    # 对齐到雷达坐标系，VO模型输出的带有尺度的帧间位姿
    rel_VO_poses = np.zeros((num_poses - 1, 12))
    '''尺度因子'''
    scale_factors = np.zeros((num_poses - 1, 1))
    # 用来估计尺度因子
    last_rel_LO_pose = np.identity(4)
    last_rel_VO_pose = np.identity(4)
    ##############################################################
    '''创建输出文件夹位置及文件后缀'''
    save_dir = Path(save_dir)
    print('Output files wiil be saved in： ' + save_dir)
    if not os.path.exists(save_dir): save_dir.makedirs_p()
    suffix = "pts@" + str(args.num_icp_points) + \
             "_prop@" + str(args.proposal) + \
             "_tolerance@" + str(args.tolerance) + \
             "_scm@" + str(args.scm_type) + \
             "_thresh@" + str(args.loop_threshold)
    if args.scan2submap:
        suffix = suffix + '_scan2map@True'

    '''Pose Graph Manager (for back-end optimization) initialization'''
    PGM = PoseGraphManager()
    PGM.addPriorFactor()
    '''Saver初始化'''
    num_frames = len(tqdm(framework))
    ResultSaver = PoseGraphResultSaver(init_pose=PGM.curr_se3,
                                       save_gap=args.save_gap,
                                       num_frames=num_frames,
                                       seq_idx=args.sequence_idx,
                                       save_dir=save_dir)

    '''Mapping initialzation'''
    if args.mapping is True:
        Map = MappingManager()

    # for save the result as a video
    fig_idx = 1
    fig = plt.figure(fig_idx)
    writer = FFMpegWriter(fps=15)
    vedio_name = suffix + ".mp4"
    vedio_path = os.path.join(save_dir, vedio_name)
    num_frames_to_skip_to_show = 5
    num_frames_to_save = np.floor(num_frames / num_frames_to_skip_to_show)
    with writer.saving(fig, vedio_path, num_frames_to_save):  # this video saving part is optional

        for j, sample in enumerate(tqdm(framework)):
            '''
            ***********************************VO部分*********************************
            '''
            # 图像降采样
            imgs = sample['imgs']
            w, h = imgs[0].size
            if (not args.no_resize) and (h != downsample_img_height or w != downsample_img_width):
                imgs = [(np.array(img.resize((downsample_img_width, downsample_img_height)))).astype(np.float32) for img
                        in imgs]
            imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]
            # numpy array 转troch tensor
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

            final_poses = pose2tf_mat(args.rotation_mode, imgs, poses)
            # cur_VO_pose取final poses的第2项，则是取T10,T21,T32。。。
            rel_VO_pose = np.identity(4)
            rel_VO_pose[:3, :] = final_poses[1]
            # 尺度因子的确定：采用上一帧的LO输出位姿和VO输出位姿的尺度比值作为当前帧的尺度因子，初始尺度为1
            # version1.0
            # if j == 0:
            #     scale_factor = 7
            # else:
            #     scale_factor = math.sqrt(np.sum(last_rel_LO_pose[:3, -1] ** 2) / np.sum(last_rel_VO_pose[:3, -1] ** 2))

            # version2.0 固定模型的尺度因子
            scale_factor = 7

            scale_factors[j] = scale_factor
            last_rel_VO_pose = copy.deepcopy(rel_VO_pose)  # 注意深拷贝
            # 先尺度修正，再对齐坐标系，施密特正交化避免病态矩阵
            rel_VO_pose[:3, -1:] = rel_VO_pose[:3, -1:] * scale_factor
            rel_VO_pose = Transform_matrix_C2L @ rel_VO_pose @ np.linalg.inv(Transform_matrix_C2L)
            rel_VO_pose = GramSchmidtHelper(rel_VO_pose)
            rel_VO_poses[j] = rel_VO_pose[:3, :].reshape(12)
            abs_VO_pose = np.matmul(abs_VO_pose, rel_VO_pose)
            abs_VO_poses[j + 1] = abs_VO_pose[:3, :].reshape(12)

            '''*************************LO部分******************************************'''
            # 初始化
            if j == 0:
                last_pts = pointClouds[j]
                if args.mapping is True:
                    Map.updateMap(curr_se3=PGM.curr_se3, curr_local_ptcloud=last_pts, down_points=args.map_down_points,
                                  submap_points=args.num_icp_points)

            curr_pts = pointClouds[j + 1]

            # 选择LO的初值预估，分别是无预估，上一帧位姿，VO位姿
            if args.proposal == 0:
                init_pose = np.identity(4)
            elif args.proposal == 1:
                init_pose = last_rel_LO_pose
            elif args.proposal == 2:
                init_pose = rel_VO_pose

            print('init_pose')
            print(init_pose)
            '''icp 类型选择2*2=4'''
            startTime = time.time()
            if args.scan2submap:
                submap = Map.getSubMap()
                if args.icp_version == 0:
                    curr_pts = random_sampling(curr_pts, args.num_icp_points)
                    rel_LO_pose, distacnces, iterations = icp(curr_pts, submap, init_pose=init_pose,
                                                              tolerance=args.tolerance,
                                                              max_iterations=50)
                elif args.icp_version == 1:
                    if args.isKitti:
                        curr_pts = random_sampling(curr_pts, args.num_icp_points)
                    rel_LO_pose, fitness, inlier_rmse = p2l_icp(curr_pts, submap, trans_init=init_pose, threshold=0.05)
            else:
                if args.icp_version == 0:
                    curr_pts = random_sampling(curr_pts, args.num_icp_points)
                    last_pts = random_sampling(last_pts, args.num_icp_points)
                    rel_LO_pose, distacnces, iterations = icp(curr_pts, last_pts, init_pose=init_pose,
                                                              tolerance=args.tolerance,
                                                              max_iterations=50)
                elif args.icp_version == 1:
                    if args.isKitti:
                        curr_pts = random_sampling(curr_pts, args.num_icp_points)
                        last_pts = random_sampling(last_pts, args.num_icp_points)
                    rel_LO_pose, fitness, inlier_rmse = p2l_icp(curr_pts, last_pts, trans_init=init_pose,
                                                                threshold=0.05)

            ICP_iteration_time[j] = time.time() - startTime

            print('rel_LO_pose')
            print(rel_LO_pose)
            if args.icp_version == 0:
                ICP_iterations[j] = iterations
            elif args.icp_version == 1:
                ICP_fitness[j] = fitness
            ResultSaver.saveRelativePose(rel_LO_pose)
            '''更新变量'''
            last_pts = curr_pts
            last_rel_LO_pose = rel_LO_pose
            '''Update the edges and nodes of pose graph'''
            PGM.curr_node_idx = j + 1
            PGM.curr_se3 = np.matmul(PGM.curr_se3, rel_LO_pose)
            PGM.addOdometryFactor(rel_LO_pose)
            PGM.prev_node_idx = PGM.curr_node_idx

            est_pose = Transform_matrix_L2C @ PGM.curr_se3 @ np.linalg.inv(Transform_matrix_L2C)
            est_poses[j + 1] = est_pose[:3, :].reshape(12)

            # 建图更新
            if args.mapping is True:
                Map.updateMap(curr_se3=PGM.curr_se3, curr_local_ptcloud=curr_pts, down_points=args.map_down_points,
                              submap_points=args.num_icp_points)

            # save the ICP odometry pose result (no loop closure)
            ResultSaver.saveUnoptimizedPoseGraphResult(PGM.curr_se3, PGM.curr_node_idx)
            # if (j % num_frames_to_skip_to_show == 0):
            #     ResultSaver.vizCurrentTrajectory(fig_idx=fig_idx)
            #     writer.grab_frame()
            if args.vizmapping is True:
                Map.vizMapWithOpen3D()

        if args.mapping is True:
            map_name = suffix + '.pcd'
            map_path = os.path.join(save_dir, map_name)
            Map.saveMap2File(map_path)

        if args.output_dir is not None:
            # np.save(output_dir / 'predictions.npy', predictions_array)
            np.savetxt(save_dir / 'scale_factors_' + suffix + '.txt', scale_factors)
            np.savetxt(save_dir / 'rel_VO_poses.txt', rel_VO_poses)
            np.savetxt(save_dir / 'abs_VO_poses.txt', abs_VO_poses)
            rel_LO_poses_file = 'rel_LO_poses_' + suffix + '.txt'
            abs_LO_poses_file = 'abs_LO_poses_' + suffix + '.txt'
            ResultSaver.saveRelativePosesResult(rel_LO_poses_file)
            ResultSaver.saveFinalPoseGraphResult(abs_LO_poses_file)
            if args.icp_version == 0:
                np.savetxt(save_dir / 'iterations_' + suffix + '.txt', ICP_iterations)
            elif args.icp_version == 1:
                np.savetxt(save_dir / 'fitness_' + suffix + '.txt', ICP_fitness)
            np.savetxt(save_dir / 'iteration_time_' + suffix + '.txt', ICP_iteration_time)
            if args.isKitti:
                np.savetxt(save_dir / 'est_poses_' + suffix + '.txt'.format(args.sequence_idx), est_poses)

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

        # 存储优化前后误差精度指标
        if args.isKitti:
            err_statics = np.array([mean_errors, std_errors, optimized_mean_errors, optimized_std_errors])
            np.savetxt(save_dir / 'err_statics_' + suffix + '.txt'.format(args.sequence_idx), err_statics)

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


def pose2tf_mat(rotation_mode, imgs, poses):
    poses = poses.cpu()[0]
    poses = torch.cat([poses[:len(imgs) // 2], torch.zeros(1, 6).float(), poses[len(imgs) // 2:]])
    inv_transform_matrices = pose_vec2mat(poses, rotation_mode=rotation_mode).numpy().astype(np.float64)
    rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
    tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]
    transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)
    # 将对[0 1 2]中间1的转换矩阵变成对0的位姿转换:T(0->0),T(1->0),T(2->0)
    first_inv_transform = inv_transform_matrices[0]
    final_poses = first_inv_transform[:, :3] @ transform_matrices
    final_poses[:, :, -1:] += first_inv_transform[:, -1:]
    return final_poses


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
