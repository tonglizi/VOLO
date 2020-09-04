# -*- coding: utf-8 -*- 
# @Time : 2020/9/4 15:33 
# @Author : CaiXin
# @File : SCMTest.py
import argparse
import os
from tqdm import tqdm

from modules.ScanContextManager import *
import utils.UtilsPointcloud as Ptutils

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='E:/data_odometry/dataset/sequences')
parser.add_argument("--sequence_idx", type=str, default='00')
parser.add_argument("--num_rings", type=int, default=20)
parser.add_argument("--num_sectors", type=int, default=60)
parser.add_argument("--num_candidates", type=int, default=10)
parser.add_argument("--loop_threshold", type=float, default=0.11)
args = parser.parse_args()


def SCM_test():
    # SCM parms
    num_rings = args.num_rings
    num_sectors = args.num_sectors
    num_candidates = args.num_candidates
    loop_threshold = args.loop_threshold

    # data
    sequence_dir = os.path.join(args.data_dir, args.sequence_idx, 'velodyne')
    sequence_manager = Ptutils.KittiScanDirManager(sequence_dir)
    scan_paths = sequence_manager.scan_fullpaths

    SCM = ScanContextManager([num_rings, num_sectors], num_candidates, loop_threshold)
    num_downsample_points = 10000
    for i, scan_path in tqdm(enumerate(scan_paths), total=len(scan_paths)):
        print('Running:{}...'.format(i))
        curr_scan_pts = Ptutils.readScan(scan_path)
        curr_scan_down_pts = Ptutils.random_sampling(curr_scan_pts, num_downsample_points)
        # add current node
        SCM.addNode(i, curr_scan_down_pts)

        if i > 1:
            loop_idx, loop_dist, yaw_diff_deg = SCM.detectLoop()
            if loop_idx == None:
                pass
            else:
                print("Loop detected at frame : ", loop_idx)


if __name__ == '__main__':
    SCM_test()
