from get_path import project_dir
import os
import csv
import numpy as np
# get camera intrinsic matrix K
file_name = '{}/datasets/kitti/training/calib/{}'.format(project_dir,'000000.txt')
with open(file_name, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=' ')
    for line, row in enumerate(reader):
        print('line={},row={}'.format(line,row))
        if row[0] == 'P2:':
            K = row[1:]
            K = [float(i) for i in K]
            K = np.array(K, dtype=np.float32).reshape(3, 4)  #3x4矩阵,旋转+平移
            print(K)
            K = K[:3, :3] #旋转
            break