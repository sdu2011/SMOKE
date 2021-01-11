import torch
from torch import nn

from .smoke_predictor import make_smoke_predictor
from .loss import make_smoke_loss_evaluator
from .inference import make_smoke_post_processor

import numpy as np

class SMOKEHead(nn.Module):
    def __init__(self, cfg, in_channels):
        print('in_channels={}'.format(in_channels))
        super(SMOKEHead, self).__init__()

        self.cfg = cfg.clone()
        self.predictor = make_smoke_predictor(cfg, in_channels)
        self.loss_evaluator = make_smoke_loss_evaluator(cfg) #损失计算类
        self.post_processor = make_smoke_post_processor(cfg)

    def forward(self, features, targets=None):
        x = self.predictor(features) # 卷积得到[head_class, head_regression]  前者用于分类 后者用于判断3d box的位置,大小,朝向
    
        if self.training:
            loss_heatmap, loss_regression = self.loss_evaluator(x, targets) #调用loss_evaluator.__call__()计算损失
            
            # print('features.shape={},x.shape={},targets.shape={}'.format(features.shape,x.shape,targets.shape))
            print('features.type={},features.shape={}'.format(type(features),features.shape))
            print('targets.type={},targets.shape={}'.format(type(targets),np.array(targets[0]).shape))


            return {}, dict(hm_loss=loss_heatmap,
                            reg_loss=loss_regression, )
        if not self.training:
            result = self.post_processor(x, targets)

            return result, {}


def build_smoke_head(cfg, in_channels):
    return SMOKEHead(cfg, in_channels)
