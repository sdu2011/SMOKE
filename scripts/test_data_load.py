from get_path import project_dir
import os
import csv
import numpy as np

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

from smoke.data.datasets.kitti import KITTIDataset
import torch


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
    # s = SMOKEPredictor(cfg,in_channels=64)
    # print(s)
    
    # dataset = KITTIDataset(cfg,"/home/sc/keepgoing/SMOKE/datasets/kitti/training")
    # data_loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    data_loader = make_data_loader(  #数据加载 
        cfg,
        is_train=True,
    )
    
    for data in data_loader:
        #data={'images': <smoke.structures.image_list.ImageList object at 0x7fdcc7aa21d0>, 'targets': (ParamsList(regress_number=1, image_width=1280, image_height=384), ParamsList(regress_number=1, image_width=1280, image_height=384)), 'img_ids': ('000000', '000000')}
        print('data={}'.format(data))
        return

    return

# # python scripts/test_data_load.py --config-file "configs/smoke_gn_vector.yaml"
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
