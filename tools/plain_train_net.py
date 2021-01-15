import torch

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


def train(cfg, model, device, distributed):
    optimizer = make_optimizer(cfg, model) #优化器
    scheduler = make_lr_scheduler(cfg, optimizer) #学习率调整策略

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = comm.get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    torch.multiprocessing.set_sharing_strategy('file_system')
    data_loader = make_data_loader(  #数据加载 
        cfg,
        is_train=True,
    )
    print('data_loader type is:{}'.format(type(data_loader)))

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    #训练入口函数
    do_train(
        cfg,
        distributed,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments
    )


#smoke/config/defaults.py里有很多默认配置选项
def setup(args):
    print('setup: args={}'.format(args))
    cfg.merge_from_file(args.config_file)
    print('setup1: cfg={}'.format(cfg))

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    print('setup2: cfg={}'.format(cfg))

    return cfg


def inference_on_one_img(model,img):
    # 准备输入

    # 
    # output = model(images, targets)
    pass
    
def main(args):
    cfg = setup(args) #生成配置

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    print('args.eval_img={}'.format(args.eval_img))

    if args.eval_img:
        print(args.eval_img)

        # 载入模型
        checkpointer = DetectronCheckpointer(
            cfg, model, save_dir=cfg.OUTPUT_DIR
        )
        ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
        _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
        
        # 数据准备
        img_path = 'datasets/kitti/training/image_2/{}'.format(args.eval_img)
        label_path = 'datasets/kitti/training/image_2/{}.txt'.format(args.eval_img)

        return

    if args.eval_only:
        checkpointer = DetectronCheckpointer(
            cfg, model, save_dir=cfg.OUTPUT_DIR
        )
        ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
        _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
        return run_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True,
        )

    train(cfg, model, device, distributed)

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
