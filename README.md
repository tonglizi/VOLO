# VOLO
VOLO is a SLAM sysytem which integrate a LiDAR odometry and a visual odometry. The VO is an invariant of SFMLearner, and the LO is ICP model which is widely used to match 2D/3D pointclouds.
VOLO also contains a loop closure detection module to imporve the shift. The loop detection is scan context.
## 1. Prerequisite
* Python(>=3.6): VOLO takes python as code environment.
* PyTorch(>=1.6.0): VO is a DL model trained by torch.
* Open3d: A pointcloud library which is friendly in python. VOLO uses Open3d to handle the pointcloud processing.
* miniSAM: Graph optimization library for python.
## 2. How to use
### 2.1 Dataset
Due to the integration of VO and LO, the dataset should contain the image and the pointcloud synchronized in time. 
* KITTI odometry dataset
* Self-built dataset  
We provide data reading scripts for the above two datasets in 'utils' folder.
### 2.2 Run
`VOLO_pipeline.py` is the pipeline of VOLO processing.

    python3 VOLO_pipeline.py /home/cx/SLAM/SfmLearner-Pytorch-master_new/SfmLearner_Models/exp_pose_model_best.pth.tar --dataset-dir /home/sda/dataset --sequence_idx 09 --output-dir /home/cx/SLAM --proposal 2 --tolerance 0.001 --loop_threshold 0.075 --num_icp_points 10000 --scm_type ring --mapping True
* --sequence_idx: data sequence index
* --mapping: map real-time visualization switch(default off)
* --proposal: options for the LO proposal, 0--no proposal, 1--last LO pose as proposal, 2--VO pose as proposal (default)
* --fine-matching: enable fine matching (scan2map after scan2scan)
* --isKitti: Only for KITTI dataset test, if not, then for mydataset
* --scan2submap: ICP matching method: scan to scan (off); scan to sub map (on)
* --icp-version: options for ICP implementations: 0 is my own, 1 is from open3d
* --loop: enable loop closure detection or not
### 2.3 AutoRun
`auto_run.sh` is a shell script for multiple data sequences. 