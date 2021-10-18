import open3d
import torch

from configs.path import get_resource_dir_path
from .pointcloud import PointCloud

data_scene_path = get_resource_dir_path('data_scene')


class TorchContactScenePointCloud(PointCloud):
    def __init__(self, data, visualization=False):
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(data['cloud'])
        PointCloud.__init__(self, cloud, visualization=visualization)

        cloud_array = data['cloud']
        normal_array = data['normal']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cloud_array_homo = torch.cat(
            [torch.tensor(cloud_array.T).float(), torch.ones(1, cloud_array.shape[0])], dim=0).float().to(device)
        self.normal_array = torch.tensor(normal_array.T).float().to(device)
        self.label_array = torch.tensor(data['label']).int().to(device)
        assert self.cloud_array_homo.shape[1] == self.normal_array.shape[1], 'shape1: {}, shape2:{}'.format(
            self.cloud_array_homo.shape, self.normal_array.shape)

        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)
        self.normals = normal_array
        self.global_to_local = data['global_to_local']
        self.search_score = data['search_score']
        self.antipodal_score = data['antipodal_score']
        self.frame_point_index = data['frame_point_index']
