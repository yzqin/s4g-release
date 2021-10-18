import torch
import torch.nn as nn
import torch.nn.functional as F


class GPDClassifier(nn.Module):
    """
    Input: (batch_size, input_chann, 60, 60)
    """

    def __init__(self, in_channels, score_clases, dropout=False):
        super(GPDClassifier, self).__init__()
        self.out_channels = score_clases
        self.conv1 = nn.Conv2d(in_channels, 20, 5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(12 * 12 * 50, 500)
        self.dp = nn.Dropout2d(p=0.5, inplace=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, score_clases)
        self.if_dropout = dropout

    def forward(self, data_batch):
        close_region_projection_maps = data_batch["close_region_projection_maps"]
        if len(close_region_projection_maps.shape) == 4:
            batch_size, channels, height, width = close_region_projection_maps.shape
        else:
            assert len(close_region_projection_maps.shape) == 5
            batch_size, num_grasp, channels, height, width = close_region_projection_maps.shape
            close_region_projection_maps = close_region_projection_maps.view(batch_size*num_grasp,
                                                                             channels, height, width)

        x = self.pool1(self.conv1(close_region_projection_maps))
        x = self.pool2(self.conv2(x))
        x = x.view(-1, 7200)
        x = self.relu(self.fc1(x))
        if self.if_dropout:
            x = self.dp(x)
        x = self.fc2(x)

        return {
            "grasp_logits": x
        }


class GPDLoss(nn.Module):
    def forward(self, preds, labels):
        grasp_logits = preds["grasp_logits"]
        grasp_score_labels = labels["grasp_score_labels"]

        cls_loss = F.cross_entropy(grasp_logits, grasp_score_labels)
        loss_dict = {"cls_loss": cls_loss}

        return loss_dict


class GPDMetric(nn.Module):
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
                "recall": recall
                }


def build_gpd(cfg):
    net = GPDClassifier(
        in_channels=cfg.DATA.GPD_IN_CHANNELS,
        score_clases=cfg.DATA.SCORE_CLASSES,
        dropout=cfg.MODEL.GPD.DROPOUT
    )

    loss_func = GPDLoss()
    metric = GPDMetric()

    return net, loss_func, metric


if __name__ == "__main__":
    gpdnet = GPDClassifier(3, 4)

    preds = gpdnet({"close_region_projection_maps": torch.rand(2, 10, 3, 60, 60)})
    for k, v in preds.items():
        print(k, v.shape)
