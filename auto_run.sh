#!/bin/bash
#主要控制参数是：点云降采样数目(影响速度)，以及是否采用scan to map的配齐方法（同样影响速度）
# 对于跑完一个period,需要（77+22+11）*2=220个测试，每个测试10分钟左右；一共需要36h
interpreter="python3.6"
script="VOLO_pipeline.py"
useScanToMap=0
icp_version=1

model_dir=/home/cx/SLAM/SfmLearner-Pytorch-master_new/checkpoints/
pretrained_dir=/home/cx/SLAM/SfmLearner-Pytorch-master_new/SfmLearner_Models/

mydataset_dir=/home/sda/mydataset/
kittidata_dir=/home/sda/dataset

num_icp_points=(10000) #(10000 20000)
scm_type=(ring vertical combined)

# 包括原生模型*1和新模型*10在训练集上训练的model
#vo_models_for_mydataset=(
#  "data,b16,lr0.0004,m0.2/01-18-16:09/"
#  "data,b16,lr0.0004/03-04-19:34/"
#  "data,b16,lr0.0004/03-05-09:54/"
#  "data,b16,lr0.0001/03-06-18:58/"
#  "data,b16,lr0.0004/03-08-11:21/"
#  "data,b16,lr0.0004/03-09-15:31/"
#  "data,b16,lr0.0004/03-10-09:50/"
#  "data,seq5,b8,lr0.0004/03-11-14:57/"
#  "data,seq5,b8,lr0.0004/03-12-11:39/")
#  vo_models_for_mydataset=(
#  "data,b16,lr0.0004,m0.2/01-18-16:09/"
#  "data,b16,lr0.0004/03-05-09:54/"
#  "data,b16,lr0.0001/03-06-18:58/")
#  vo_models_for_mydataset=(
#  "data,b16,lr0.0004,m0.2/01-18-16:09/")
vo_models_for_mydataset=(
  "data,b16,lr0.0004/03-05-09:54/"
  "data,b16,lr0.0001/03-06-18:58/")

# 包括原生模型*1和新模型*1在KITTI训练集上训练的model
vo_models_for_kitti=(
  "data,500epochs,epoch_size3000,b32,m0.2/06-17-04_17/"
  "data,b16,lr0.0004/03-09-11:53"
)
#mydataset=(
#  20210308_184153
#  20210308_184607
#  20210308_184904
#  20210308_185145
#  20210308_185628
#  20210308_190006
#  20210308_190432)
#mydataset=(
#  20210308_184153
#  20210308_184607
#  20210308_184904)

#mydataset=(
#  20210315_163543
#  20210315_164054
#  20210315_164326
#  20210315_164900
#  20210315_165031
#  20210315_165219
#  20210315_165342
#  20210315_165519
#  20210315_165644
#  20210315_165815
#  20210315_165951
#  20210315_170122
#  20210315_170253
#  20210315_170434
#  20210315_170938
#  20210315_171127
#  20210315_171302
#  20210315_171424
#  20210315_171600
#  20210315_171724
#  20210315_171849
#  20210315_172023
#  20210315_172155
#  20210315_172318
#  20210315_172450
#  20210315_172614
#  20210315_172806
#  20210315_172925
#  20210315_173052
#  20210315_173218
#  20210315_173403
#  20210315_173520
#  20210315_173656
#  20210315_173824
#  20210315_173951
#  20210315_174119
#  20210315_174240
#  20210315_174502
#  20210315_174658
#  20210315_174900
#  20210315_175029
#  20210315_175151
#  20210315_175525
#  20210315_175643
#  20210315_175803
#  20210315_175941
#  20210315_180145
#  20210315_180323
#  20210315_180443
#  20210315_180608)

#kittidataset=(00 01 02 03 04 05 06 07 08 09 10)
kittidataset=(00 02 05 08 09)

# 自建数据集测试:(10+1)*7*1=77组
# shellcheck disable=SC2068
for model in ${vo_models_for_mydataset[@]}; do
  for data_seq in ${mydataset[@]}; do
    for num_points in ${num_icp_points[@]}; do
      if useScanToMap==0; then
        # no vo proposal
        $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $mydataset_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --icp-version $icp_version

        # with vo proposal
        $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $mydataset_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --icp-version $icp_version
      else
        # no vo proposal with scan2submap
        $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $mydataset_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --icp-version $icp_version

        # with vo proposal with scan2submap
        $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $mydataset_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --icp-version $icp_version
      fi
    done
  done
done
# kitti数据集:(1+1)*11*1=22组
# shellcheck disable=SC2068
for model in ${vo_models_for_kitti[@]}; do
  for data_seq in ${kittidataset[@]}; do
    for num_points in ${num_icp_points[@]}; do
      if useScanToMap==0; then
        # no vo proposal
        $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --isKitti True --icp-version $icp_version

        # with vo proposal
        $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --isKitti True --icp-version $icp_version
      else
        # no vo proposal with scan2submap
        $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --isKitti True --icp-version $icp_version

        # with vo proposal with scan2submap
        $interpreter $script $model_dir$model'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --isKitti True --icp-version $icp_version
      fi
    done
  done
done

# 预训练模型在Kitti上的表现：1*11*1=11组
# shellcheck disable=SC2068
#for data_seq in ${kittidataset[@]}; do
#  # shellcheck disable=SC2068
#  for num_points in ${num_icp_points[@]}; do
#    if useScanToMap==0; then
#      # no vo proposal
#      $interpreter $script $pretrained_dir'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --isKitti True --icp-version $icp_version
#
#      # with vo proposal
#      $interpreter $script $pretrained_dir'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --isKitti True --icp-version $icp_version
#    else
#      # no vo proposal with scan2submap
#      $interpreter $script $pretrained_dir'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 0 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --isKitti True --icp-version $icp_version
#
#      # with vo proposal with scan2submap
#      $interpreter $script $pretrained_dir'exp_pose_model_best.pth.tar' --dataset-dir $kittidata_dir --sequence_idx $data_seq --proposal 2 --tolerance 0.0005 --loop_threshold 0 --num_icp_points $num_points --scm_type ${scm_type[0]} --scan2submap True --isKitti True --icp-version $icp_version
#    fi
#  done
#done
