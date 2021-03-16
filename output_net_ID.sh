#!/bin/bash
#主要控制参数是：点云降采样数目(影响速度)，以及是否采用scan to map的配齐方法（同样影响速度）
# 对于跑完一个period,需要（77+22+11）*2=220个测试，每个测试10分钟左右；一共需要36h

model_dir=/home/cx/SLAM/SfmLearner-Pytorch-master_new/checkpoints/
pretrained_dir=/home/cx/SLAM/SfmLearner-Pytorch-master_new/SfmLearner_Models/

# 包括原生模型*1和新模型*10在训练集上训练的model
vo_models_for_mydataset=(
  "data,b16,lr0.0004,m0.2/01-18-16:09/"
  "data,b16,lr0.0004/03-04-19:34/"
  "data,b16,lr0.0004/03-05-09:54/"
  "data,b16,lr0.0001/03-06-18:58/"
  "data,b16,lr0.0004/03-08-11:21/"
  "data,b16,lr0.0004/03-09-15:31/"
  "data,b16,lr0.0004/03-10-09:50/"
  "data,seq5,b8,lr0.0004/03-11-14:57/"
  "data,seq5,b8,lr0.0004/03-12-11:39/")

# 包括原生模型*1和新模型*1在KITTI训练集上训练的model
vo_models_for_kitti=(
  "data,500epochs,epoch_size3000,b32,m0.2/06-17-04_17/"
)


# 自建数据集上的模型
# shellcheck disable=SC2068
for model in ${vo_models_for_mydataset[@]}; do
  python3.6 net_ID_output.py $model_dir$model'exp_pose_model_best.pth.tar'
done

# kitti上的模型
# shellcheck disable=SC2068
for model in ${vo_models_for_kitti[@]}; do
  python3.6 net_ID_output.py $model_dir$model'exp_pose_model_best.pth.tar'
done

# 预训练模型
python3.6 net_ID_output.py $pretrained_dir'exp_pose_model_best.pth.tar'
