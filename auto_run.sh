#!/bin/bash
interpreter="python3.6"
script="VOLO_pipeline.py"

model_dir=/home/cx/SLAM/SfmLearner-Pytorch-master_new/checkpoints/
pretrained_dir=/home/cx/SLAM/SfmLearner-Pytorch-master_new/SfmLearner_Models/

mydataset_dir=/home/sda/mydataset/
kittidata_dir=/home/sda/dataset

num_icp_points=(20000 15000)
scm_type=(ring vertical combined)

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
mydataset=(
  20210308_184153
  20210308_184607
  20210308_184904
  20210308_185145
  20210308_185628
  20210308_190006
  20210308_190432)
kittidataset=(00 01 02 03 04 05 06 07 08 09 10)

# 自建数据集测试:(10+1)*7*2=154组
# shellcheck disable=SC2068
for model in ${vo_models_for_mydataset[@]}; do
  for data_seq in ${mydataset[@]}; do
    for num_points in ${num_icp_points[@]}; do
      # no vo proposal
      $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $mydataset_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap False

      # with vo proposal
      $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $mydataset_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap False

      # no vo proposal with scan2submap
      $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $mydataset_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True

      # with vo proposal with scan2submap
      $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $mydataset_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True
    done
  done
done
# kitti数据集:(1+1)*11*2=44组
# shellcheck disable=SC2068
for model in ${vo_models_for_kitti[@]}; do
  for data_seq in ${kittidataset[@]}; do
    for num_points in ${num_icp_points[@]}; do
      # no vo proposal
      $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap False --isKitti True

      # with vo proposal
      $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap False --isKitti True

      # no vo proposal with scan2submap
      $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --isKitti True

      # with vo proposal with scan2submap
      $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --isKitti True
    done
  done
done

# 预训练模型在Kitti上的表现：1*11*2=22组
  # shellcheck disable=SC2068
  for data_seq in ${kittidataset[@]}; do
    # shellcheck disable=SC2068
    for num_points in ${num_icp_points[@]}; do
      # no vo proposal
      $interpreter $script $pretrained_dir'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap False --isKitti True

      # with vo proposal
      $interpreter $script $pretrained_dir'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap False --isKitti True

      # no vo proposal with scan2submap
      $interpreter $script $pretrained_dir'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --isKitti True

      # with vo proposal with scan2submap
      $interpreter $script $pretrained_dir'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --isKitti True
    done
  done