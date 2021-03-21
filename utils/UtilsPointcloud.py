import os 
import random
import numpy as np

def random_sampling(orig_points, num_points):
    if orig_points.shape[0] > num_points:
        points_down_idx = random.sample(range(orig_points.shape[0]), num_points)
        down_points = orig_points[points_down_idx, :]
        return down_points
    else:
        return orig_points

def readScan(bin_path, dataset='KITTI'):
    if(dataset == 'KITTI'):
        return readBinScan(bin_path)


def readBinScan(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    ptcloud_xyz = scan[:, :-1]
    return ptcloud_xyz

def loadPointCloud(rootdir):
    files = os.listdir(rootdir)
    files.sort()
    pointclouds = []
    for file in files:
        if not os.path.isdir(file):
            scan = np.fromfile(rootdir + "/" + file, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            ptcloud_xyz = scan[:, :-1]
            pointclouds.append(ptcloud_xyz)
    return pointclouds
    

class KittiScanDirManager:
    def __init__(self, scan_dir):
        self.scan_dir = scan_dir
        
        self.scan_names = os.listdir(scan_dir)
        self.scan_names.sort()    
        
        self.scan_fullpaths = [os.path.join(self.scan_dir, name) for name in self.scan_names]
  
        self.num_scans = len(self.scan_names)

    def __repr__(self):
        return ' ' + str(self.num_scans) + ' scans in the sequence (' + self.scan_dir + '/)'

    def getScanNames(self):
        return self.scan_names
    def getScanFullPaths(self):
        return self.scan_fullpaths
    def printScanFullPaths(self):
        return print("\n".join(self.scan_fullpaths))

