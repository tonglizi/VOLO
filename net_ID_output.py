# -*- coding: utf-8 -*- 
# @Time : 2021/3/14 21:31 
# @Author : CaiXin
# @File : net_ID_output.py.py
import argparse
import hashlib

parser = argparse.ArgumentParser(
    description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
args = parser.parse_args()

def MD5_ID(pretrained_posenet):
    md = hashlib.md5()  # 创建md5对象
    md.update(str(pretrained_posenet).encode(encoding='utf-8'))
    return md.hexdigest()


def main():
    net_ID = MD5_ID(args.pretrained_posenet)
    print('*********************************')
    print(args.pretrained_posenet)
    print(net_ID)
    print('*********************************')

if __name__=="__main__":
    main()