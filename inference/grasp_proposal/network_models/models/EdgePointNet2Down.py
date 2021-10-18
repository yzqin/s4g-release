import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils.modules import EdgeSAModule, PointnetFPModule
from .PointNet2 import PointNet2, PointNet2Loss, PointNet2Metric


class EdgePointNet2Down(PointNet2):
    _SA_MODULE = EdgeSAModule
    _FP_MODULE = PointnetFPModule


def build_edgepointnet2down(cfg):
    net = EdgePointNet2Down(
        score_classes=cfg.DATA.SCORE_CLASSES,
        num_centroids=cfg.MODEL.EDGEPN2D.NUM_CENTROIDS,
        radius=cfg.MODEL.EDGEPN2D.RADIUS,
        num_neighbours=cfg.MODEL.EDGEPN2D.NUM_NEIGHBOURS,
        sa_channels=cfg.MODEL.EDGEPN2D.SA_CHANNELS,
        fp_channels=cfg.MODEL.EDGEPN2D.FP_CHANNELS,
        num_fp_neighbours=cfg.MODEL.EDGEPN2D.NUM_FP_NEIGHBOURS,
        seg_channels=cfg.MODEL.EDGEPN2D.SEG_CHANNELS,
        dropout_prob=cfg.MODEL.EDGEPN2D.DROPOUT_PROB,
    )

    loss_func = PointNet2Loss(
        label_smoothing=cfg.MODEL.EDGEPN2D.LABEL_SMOOTHING,
        neg_weight=cfg.MODEL.EDGEPN2D.NEG_WEIGHT,
    )
    metric = PointNet2Metric()

    return net, loss_func, metric
