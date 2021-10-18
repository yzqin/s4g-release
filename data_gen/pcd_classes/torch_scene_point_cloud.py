import open3d
import numpy as np
from configs import config
from .pointcloud import PointCloud
import torch
from configs.dataset_config import color_array_to_label


class TorchScenePointCloud(PointCloud):
    def __init__(self, cloud: open3d.geometry.PointCloud, voxelize=True, visualization=False, filter_work_space=True):
        PointCloud.__init__(self, cloud, visualization=visualization)
        if voxelize:
            self.voxelize(voxel_size=config.VOXEL_SIZE / config.SCENE_MULTIPLE)
        if filter_work_space:
            work_space = config.WORKSPACE.copy()
            work_space[4] += 0.005
            self.filter_work_space(workspace=config.WORKSPACE)

        cloud_array = np.asarray(self.cloud.points)
        normal_array = np.asarray(self.cloud.normals)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cloud_array_homo = torch.cat(
            [torch.tensor(cloud_array.T).float(), torch.ones(1, cloud_array.shape[0])], dim=0).float().to(device)
        self.normal_array = torch.tensor(normal_array.T).float().to(device)
        self.label_array = torch.tensor(self.get_label_array()).int().to(device)
        assert self.cloud_array_homo.shape[1] == self.normal_array.shape[1], 'shape1: {}, shape2:{}'.format(
            self.cloud_array_homo.shape, self.normal_array.shape)

        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)
        self.normals = normal_array

    def get_label_array(self):
        """
        Get the object label for each points based on color (Details in generate_scene.py)
        The table will get label zeros
        Is there any rounding error?
        :return: Label array (n,)
        """
        colors = np.asarray(self.cloud.colors)
        label = color_array_to_label(colors)

        return label


if __name__ == '__main__':
    cloud = open3d.read_point_cloud('../scene/2.ply')
    pc = TorchScenePointCloud(cloud)
