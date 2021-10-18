import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn_utils.mlp import SharedMLP
from .pointnet2_utils.modules import PointNetSAModule, PointnetFPModule, PointNetSAAvgModule
from ..nn_utils.functional import smooth_cross_entropy


class PointNet2(nn.Module):
    """PointNet++ part segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        score_classes (int): the number of grasp score classes
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features

    References:
        https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py

    """
    _SA_MODULE = PointNetSAModule
    _FP_MODULE = PointnetFPModule

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

        inter_channels = [0]
        inter_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            fp_module = self._FP_MODULE(in_channels=feature_channels + inter_channels[-2 - ind],
                                        mlp_channels=fp_channels[ind],
                                        num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
        # self.mlp_seg = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        # self.seg_logit = nn_utils.Conv1d(seg_channels[-1], score_classes, 1, bias=True)

        self.mlp_grasp_eval = SharedMLP(feature_channels + 48, seg_channels, ndim=2, dropout_prob=dropout_prob)
        self.grasp_eval_logit = nn.Conv2d(seg_channels[-1], score_classes, 1, bias=True)

        self.mlp_R = SharedMLP(feature_channels, seg_channels, ndim=1)
        self.R_logit = nn.Conv1d(seg_channels[-1], 9, 1, bias=True)

        self.mlp_t = SharedMLP(feature_channels, seg_channels, ndim=1)
        self.t_logit = nn.Conv1d(seg_channels[-1], 3, 1, bias=True)

        self.mlp_movable = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        self.movable_logit = nn.Conv1d(seg_channels[-1], 2, 1, bias=True)

        self.init_weights()

    def forward(self, data_batch):
        points = data_batch['scene_points']

        xyz = points
        feature = None

        # save intermediate results
        inter_xyz = [xyz]
        inter_feature = [None]

        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)

        # Feature Propagation Layers
        sparse_xyz = xyz
        sparse_feature = feature
        for fp_ind, fp_module in enumerate(self.fp_modules):
            dense_xyz = inter_xyz[-2 - fp_ind]
            dense_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            sparse_xyz = dense_xyz
            sparse_feature = fp_feature

        R = self.mlp_R(sparse_feature)
        R = self.R_logit(R)

        t = self.mlp_t(sparse_feature)
        t = self.t_logit(t)

        mov = self.mlp_movable(sparse_feature)
        mov = self.movable_logit(mov)

        # grasp eval
        if "local_search_frame" in data_batch.keys():
            local_search_frame = data_batch["local_search_frame"]
            valid_frame_num, local_search_num = local_search_frame.shape[2:]
            points_tmp = points[:, :, :valid_frame_num].unsqueeze(-1).expand(-1, -1, -1, local_search_num)
            local_search_frame[:, 9:, :, :] = local_search_frame[:, 9:, :, :] - points_tmp
            valid_feature = sparse_feature[:, :, :valid_frame_num].unsqueeze(-1).expand(-1, -1, -1, local_search_num)
            local_search_frame = local_search_frame.repeat(1, 4, 1, 1)
            valid_feature = torch.cat([valid_feature, local_search_frame], dim=1)
            local_search_logit = self.grasp_eval_logit(self.mlp_grasp_eval(valid_feature))

        else:  # real experiments
            local_search_frame = torch.cat([R, t], dim=1).unsqueeze(-1)
            local_search_frame = local_search_frame.repeat(1, 4, 1, 1)
            sparse_feature = sparse_feature.unsqueeze(-1)
            valid_feature = torch.cat([sparse_feature, local_search_frame], dim=1)
            local_search_logit = self.grasp_eval_logit(self.mlp_grasp_eval(valid_feature))

        t = points + t

        preds = {"local_search_logits": local_search_logit,
                 "frame_R": R,
                 "frame_t": t,
                 "movable_logits": mov,
                 }

        return preds

    def init_weights(self):
        nn.init.zeros_(self.t_logit.weight)
        nn.init.zeros_(self.t_logit.bias)


class PointNet2Loss(nn.Module):
    def __init__(self, label_smoothing=0, neg_weight=0.1):
        super(PointNet2Loss, self).__init__()
        self.label_smoothing = label_smoothing
        self.neg_weight = neg_weight

    def forward(self, preds, labels):
        scene_score_logits = preds["local_search_logits"]  # (B, C, N2, LOCAL_SEARCH)
        score_classes = scene_score_logits.shape[1]
        weight = torch.ones(score_classes, device=scene_score_logits.device)
        weight[0] = self.neg_weight

        movable_logits = preds["movable_logits"]
        movable_labels = labels["scene_movable_labels"]
        mov_weight = torch.ones(2, device=movable_logits.device)
        mov_weight[0] = 0.4

        scene_score_labels = labels["scored_grasp_labels"]  # (B, N)

        if self.label_smoothing > 0:
            selected_logits = scene_score_logits.permute(0, 2, 3, 1).contiguous().view(-1, score_classes)
            scene_score_labels = scene_score_labels.view(-1)
            cls_loss = smooth_cross_entropy(selected_logits, scene_score_labels, self.label_smoothing,
                                            weight=weight)

            movable_logits = movable_logits.transpose(1, 2).contiguous().view(-1, 2)
            movable_labels = movable_labels.view(-1)

            mov_loss = smooth_cross_entropy(movable_logits, movable_labels, self.label_smoothing, weight=mov_weight)
        else:
            cls_loss = F.cross_entropy(scene_score_logits, scene_score_labels, weight)
            mov_loss = F.cross_entropy(movable_logits, movable_labels, mov_weight)

        gt_frame_R = labels["best_frame_R"]
        num_frame_points = gt_frame_R.shape[2]
        pred_frame_R = preds["frame_R"][:, :, :num_frame_points]
        R_loss_1 = ((pred_frame_R - gt_frame_R) ** 2).mean(1, True)
        gt_frame_R_inv = gt_frame_R.clone()
        gt_frame_R_inv[:, 1:3, :] = - gt_frame_R_inv[:, 1:3, :]
        gt_frame_R_inv[:, 4:6, :] = - gt_frame_R_inv[:, 4:6, :]
        gt_frame_R_inv[:, 7:9, :] = - gt_frame_R_inv[:, 7:9, :]
        R_loss_2 = ((pred_frame_R - gt_frame_R_inv) ** 2).mean(1, True)
        R_loss, _ = torch.min(torch.cat([R_loss_1, R_loss_2], dim=1), dim=1)
        R_loss = R_loss.mean() * 4.0
        gt_norm = torch.stack([gt_frame_R[:, 0, :], gt_frame_R[:, 3, :], gt_frame_R[:, 6, :]], dim=1)
        pred_norm = torch.stack([pred_frame_R[:, 0, :], pred_frame_R[:, 3, :], pred_frame_R[:, 6, :]], dim=1)
        norm_loss = torch.mean((pred_norm - gt_norm) ** 2)

        gt_frame_t = labels["best_frame_t"]
        pred_frame_t = preds["frame_t"][:, :, :num_frame_points]
        t_loss = torch.mean((pred_frame_t - gt_frame_t) ** 2) * 20.0

        loss_dict = {"cls_loss": cls_loss,
                     "R_loss": R_loss,
                     # "norm_loss": norm_loss,
                     "t_loss": t_loss,
                     "mov_loss": mov_loss,
                     }

        return loss_dict


class PointNet2Metric(nn.Module):
    def forward(self, preds, labels):
        scene_score_logits = preds["local_search_logits"]  # (B, C, N2)
        score_classes = scene_score_logits.shape[1]

        scene_score_labels = labels["scored_grasp_labels"]  # (B, N)

        selected_preds = scene_score_logits.argmax(1).view(-1)
        scene_score_labels = scene_score_labels.view(-1)

        cls_acc = selected_preds.eq(scene_score_labels).float()

        movable_logits = preds["movable_logits"]
        movable_labels = labels["scene_movable_labels"]
        movable_preds = movable_logits.argmax(1).view(-1)
        movable_labels = movable_labels.view(-1)
        mov_acc = movable_preds.eq(movable_labels).float()

        gt_frame_R = labels["best_frame_R"]
        batch_size, _, num_frame_points = gt_frame_R.shape
        pred_frame_R = preds["frame_R"][:, :, :num_frame_points]
        gt_frame_R = gt_frame_R.transpose(1, 2).contiguous().view(batch_size * num_frame_points, 3, 3)
        gt_frame_R_inv = gt_frame_R.clone()
        gt_frame_R_inv[:, :, 1:] = -gt_frame_R_inv[:, :, 1:]
        pred_frame_R = pred_frame_R.transpose(1, 2).contiguous().view(batch_size * num_frame_points, 3, 3)
        M = torch.bmm(gt_frame_R, pred_frame_R.transpose(1, 2))
        angle = torch.acos(torch.clamp((M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0))
        M_inv = torch.bmm(gt_frame_R_inv, pred_frame_R.transpose(1, 2))
        angle_inv = torch.acos(torch.clamp((M_inv[:, 0, 0] + M_inv[:, 1, 1] + M_inv[:, 2, 2] - 1.0) / 2.0, -1.0, 1.0))

        angle_min = torch.stack([angle, angle_inv], dim=1).min(1)[0].mean()

        gt_frame_t = labels["best_frame_t"]
        pred_frame_t = preds["frame_t"][:, :, :num_frame_points]

        t_err = torch.mean(torch.sqrt(((gt_frame_t - pred_frame_t) ** 2).sum(1)))

        return {"cls_acc": cls_acc,
                "mov_acc": mov_acc,
                "R_err": angle_min,
                "t_err": t_err,
                }


def build_pointnet2_local(cfg):
    net = PointNet2(
        score_classes=cfg.DATA.SCORE_CLASSES,
        num_centroids=cfg.MODEL.PN2.NUM_CENTROIDS,
        radius=cfg.MODEL.PN2.RADIUS,
        num_neighbours=cfg.MODEL.PN2.NUM_NEIGHBOURS,
        sa_channels=cfg.MODEL.PN2.SA_CHANNELS,
        fp_channels=cfg.MODEL.PN2.FP_CHANNELS,
        num_fp_neighbours=cfg.MODEL.PN2.NUM_FP_NEIGHBOURS,
        seg_channels=cfg.MODEL.PN2.SEG_CHANNELS,
        dropout_prob=cfg.MODEL.PN2.DROPOUT_PROB,
    )

    loss_func = PointNet2Loss(
        label_smoothing=cfg.MODEL.PN2.LABEL_SMOOTHING,
        neg_weight=cfg.MODEL.PN2.NEG_WEIGHT,
    )
    metric = PointNet2Metric()

    return net, loss_func, metric


if __name__ == "__main__":
    batch_size = 2
    num_points = 102400
    score_classes = 3
    num_frame = 4000

    import numpy as np

    frame_index = np.stack(
        [np.random.choice(np.arange(num_points), num_frame, replace=False) for _ in range(batch_size)])

    scene_points = np.random.randn(batch_size, 3, num_points)
    scene_score_labels = np.random.randint(0, score_classes - 1, (batch_size, num_frame))

    data_batch = {
        "scene_points": torch.tensor(scene_points).float().cuda(),
        "scene_score_labels": torch.tensor(scene_score_labels).long().cuda(),
        "frame_index": torch.tensor(frame_index).long().cuda()
    }

    pn2 = PointNet2(score_classes=score_classes).cuda()
    pn2_loss_fn = PointNet2Loss().cuda()
    pn2_metric = PointNet2Metric().cuda()

    preds = pn2(data_batch)
    print("PointNet2: ")
    for k, v in preds.items():
        print(k, v.shape)

    loss = pn2_loss_fn(preds, data_batch)
    print(loss)
    metric = pn2_metric(preds, data_batch)
    print(metric)

    sum(loss.values()).backward()
