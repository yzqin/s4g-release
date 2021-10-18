import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, input_chann=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, 2)
        x = x.view(batchsize, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, 3)

        I = torch.eye(3, dtype=x.dtype, device=x.device)
        x = x.add(I)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, input_chann=3, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(input_chann=input_chann)
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(batchsize, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


class PointNetClassifier(nn.Module):
    def __init__(self, input_chann, score_classes):
        super(PointNetClassifier, self).__init__()
        self.out_channels = score_classes
        self.feat = PointNetfeat(input_chann=input_chann, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, score_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, data_batch):
        close_region_points = data_batch["close_region_points"]
        if len(close_region_points.shape) == 3:
            batch_size, _, num_points = close_region_points.shape
        else:
            assert len(close_region_points.shape) == 4
            batch_size, num_grasp, _, num_points = close_region_points.shape
            close_region_points = close_region_points.view(batch_size*num_grasp, -1, num_points)

        x, trans = self.feat(close_region_points)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return {
            "grasp_logits": x,
        }


class PointNetLoss(nn.Module):
    def forward(self, preds, labels):
        grasp_logits = preds["grasp_logits"]
        grasp_score_labels = labels["grasp_score_labels"]

        cls_loss = F.cross_entropy(grasp_logits, grasp_score_labels)
        loss_dict = {"cls_loss": cls_loss}

        return loss_dict


class PointNetMetric(nn.Module):
    def forward(self, preds, labels):
        grasp_logits = preds["grasp_logits"]
        score_classes = grasp_logits.shape[-1]
        grasp_score_labels = labels["grasp_score_labels"]

        grasp_preds = grasp_logits.argmax(1)
        cls_acc = grasp_preds.eq(grasp_score_labels).float()

        gt_success_grasp = grasp_score_labels == (score_classes - 1)
        pred_success_grasp = grasp_preds == (score_classes - 1)
        true_pos = (gt_success_grasp & pred_success_grasp).float()
        precision = torch.sum(true_pos) / torch.clamp(torch.sum(pred_success_grasp.float()), 1e-6)
        recall = torch.sum(true_pos) / torch.clamp(torch.sum(gt_success_grasp.float()), 1e-6)

        return {"cls_acc": cls_acc,
                "prec": precision,
                "recall": recall}


def build_pointnetgpd(cfg):
    net = PointNetClassifier(
        input_chann=3,
        score_classes=cfg.DATA.SCORE_CLASSES,
    )

    loss_func = PointNetLoss()
    metric = PointNetMetric()

    return net, loss_func, metric


if __name__ == "__main__":
    pointnet = PointNetClassifier(3, 4)

    preds = pointnet({"close_region_points": torch.rand(2, 10, 3, 128)})
    for k, v in preds.items():
        print(k, v.shape)
