import torch
import torch.nn as nn

from .pointnet2_utils.modules import EdgeSAModule, PointnetFPModule, EdgeFPModule
from .PointNet2 import PointNet2, PointNet2Loss, PointNet2Metric


class EdgePointNet2DownUp(PointNet2):
    _SA_MODULE = EdgeSAModule
    _FP_MODULE = EdgeFPModule

    def __init__(self,
                 score_classes,
                 num_centroids=(10240, 1024, 128, 0),
                 radius=(0.2, 0.3, 0.4, -1.0),
                 num_neighbours=(64, 64, 64, -1),
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 512, 1024)),
                 fp_channels=((256, 256), (256, 128), (128, 128), (64, 64, 64)),
                 num_fp_neighbours=(0, 3, 3, 3),
                 seg_channels=(128,),
                 dropout_prob=0.5):
        super(PointNet2, self).__init__()

        # Sanity check
        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)
        assert len(radius) == num_sa_layers
        assert len(num_neighbours) == num_sa_layers
        assert len(sa_channels) == num_sa_layers
        assert num_sa_layers == num_fp_layers
        assert len(num_fp_neighbours) == num_fp_layers

        # Set Abstraction Layers
        feature_channels = 0
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = self._SA_MODULE(in_channels=feature_channels,
                                        mlp_channels=sa_channels[ind],
                                        num_centroids=num_centroids[ind],
                                        radius=radius[ind],
                                        num_neighbours=num_neighbours[ind],
                                        use_xyz=True)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]

        inter_channels = [3]
        inter_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            if num_fp_neighbours[ind] == 0:
                fp_module = self._FP_MODULE(in_channels=feature_channels + inter_channels[-2 - ind],
                                            mlp_channels=fp_channels[ind],
                                            num_neighbors=num_fp_neighbours[ind])
            else:
                fp_module = self._FP_MODULE(in_channels=feature_channels * 2 + inter_channels[-2 - ind],
                                            mlp_channels=fp_channels[ind],
                                            num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
        self.mlp_seg = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        self.seg_logit = nn.Conv1d(seg_channels[-1], score_classes, 1, bias=True)

        self.mlp_frame = SharedMLP(feature_channels, seg_channels, ndim=1)
        self.frame_logit = nn.Conv1d(seg_channels[-1], 9, 1, bias=True)


def build_edgepointnet2downup(cfg):
    net = EdgePointNet2DownUp(
        score_classes=cfg.DATA.SCORE_CLASSES,
        num_centroids=cfg.MODEL.EDGEPN2DU.NUM_CENTROIDS,
        radius=cfg.MODEL.EDGEPN2DU.RADIUS,
        num_neighbours=cfg.MODEL.EDGEPN2DU.NUM_NEIGHBOURS,
        sa_channels=cfg.MODEL.EDGEPN2DU.SA_CHANNELS,
        fp_channels=cfg.MODEL.EDGEPN2DU.FP_CHANNELS,
        num_fp_neighbours=cfg.MODEL.EDGEPN2DU.NUM_FP_NEIGHBOURS,
        seg_channels=cfg.MODEL.EDGEPN2DU.SEG_CHANNELS,
        dropout_prob=cfg.MODEL.EDGEPN2DU.DROPOUT_PROB,
    )

    loss_func = PointNet2Loss(
        label_smoothing=cfg.MODEL.EDGEPN2DU.LABEL_SMOOTHING,
        neg_weight=cfg.MODEL.EDGEPN2DU.NEG_WEIGHT,
    )
    metric = PointNet2Metric()

    return net, loss_func, metric
