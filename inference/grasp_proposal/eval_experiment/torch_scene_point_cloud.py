import numpy as np
import open3d
import torch

from .pointcloud import PointCloudProcessor


class TorchScenePointCloud(PointCloudProcessor):
    def __init__(self, cloud: open3d.geometry.PointCloud, label_array, visualization=False):
        PointCloudProcessor.__init__(self, cloud, visualization=visualization)

        cloud_array = np.asarray(self.cloud.points)
        normal_array = np.asarray(self.cloud.normals)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cloud_array_homo = torch.cat(
            [torch.tensor(cloud_array.T).float(), torch.ones(1, cloud_array.shape[0])], dim=0).float().to(device)
        self.normal_array = torch.tensor(normal_array.T).float().to(device)
        self.label_array = torch.tensor(label_array).int().to(device)
        assert self.cloud_array_homo.shape[1] == self.normal_array.shape[1]

        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)
        self.normals = normal_array
