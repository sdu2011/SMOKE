from get_path import project_dir
import os
import csv
import numpy as np
# get camera intrinsic matrix K
# file_name = '{}/datasets/kitti/training/calib/{}'.format(project_dir,'000000.txt')
# with open(file_name, 'r') as csv_file:
#     reader = csv.reader(csv_file, delimiter=' ')
#     for line, row in enumerate(reader):
#         print('line={},row={}'.format(line,row))
#         if row[0] == 'P2:':
#             K = row[1:]
#             K = [float(i) for i in K]
#             K = np.array(K, dtype=np.float32).reshape(3, 4)  #3x4矩阵,旋转+平移
#             print(K)
#             K = K[:3, :3] #旋转

#             print(K)
#             break



#测试网络结构
from smoke.modeling.heads.smoke_head.smoke_predictor import SMOKEPredictor
from smoke.config import cfg
from smoke.data import make_data_loader
from smoke.solver.build import (
    make_optimizer,
    make_lr_scheduler,
)
from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from smoke.utils import comm
from smoke.engine.trainer import do_train
from smoke.modeling.detector import build_detection_model
from smoke.engine.test_net import run_test

# python scripts/some_test.py --config-file "configs/smoke_gn_vector.yaml"
def setup(args):
    print('setup: args={}'.format(args))
    cfg.merge_from_file(args.config_file)
    print('setup1: cfg={}'.format(cfg))

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    print('setup2: cfg={}'.format(cfg))

    return cfg


def main(args):
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args) #生成配置
    s = SMOKEPredictor(cfg,in_channels=64)
    print(s)
    return

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
