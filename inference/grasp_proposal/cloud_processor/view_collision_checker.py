import open3d
import numpy as np
import torch
from ..configs import gripper_config as config
from .cloud_processor import CloudPreProcessor
from ..configs.processing_config import BACK_COLLISION_MARGIN, BACK_COLLISION_THRESHOLD, FINGER_COLLISION_THRESHOLD


class CloudCollisionChecker(CloudPreProcessor):
    def __init__(self, cloud: open3d.geometry.PointCloud, visualization=False):
        """
        Point Cloud Class for single view points, which is rendered before training or captured by rgbd camera
        Note that we maintain a self.frame_indices to keep a subset of all points which will calculate frame and grading
        However, all of the points should have score, and of course those do not grading will have a 0 grade
        These setting will benefit the training since we want the network to learn to distinguish table and objects

        :param cloud: Base Open3D PointCloud, no preprocess is needed ofr this input, just read from file
        :param visualization: Whether to visualize
        """

        CloudPreProcessor.__init__(self, cloud, visualization)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cloud_array = np.asarray(self.pcd.points).astype(np.float32)
        self.cloud_array = torch.tensor(cloud_array).float().to(self.device)
        self.cloud_array_homo = torch.cat(
            [self.cloud_array.transpose(0, 1), torch.ones(1, self.cloud_array.shape[0], device=self.device)],
            dim=0).float().to(self.device)

        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

    @property
    def points_num(self):
        return self.cloud_array.shape[0]

    def view_non_collision(self, global2local):
        local_cloud = torch.matmul(global2local, self.cloud_array_homo)
        close_plane_bool = (local_cloud[0, :] < config.FINGER_LENGTH) & (
                local_cloud[0, :] > -config.BOTTOM_LENGTH)

        local_search_close_plane_points = local_cloud[:, close_plane_bool][0:3, :]  # only filter along x axis

        z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                           (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)

        back_collision_bool = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                              (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                              (local_search_close_plane_points[0, :] < -BACK_COLLISION_MARGIN) & z_collision_bool

        if torch.sum(back_collision_bool) > BACK_COLLISION_THRESHOLD:
            return False

        y_finger_region_bool_left = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                    (local_search_close_plane_points[1, :] > config.HALF_BOTTOM_SPACE)
        y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                     (local_search_close_plane_points[1, :] < -config.HALF_BOTTOM_SPACE)

        y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
        collision_region_bool = (z_collision_bool & y_finger_region_bool)

        if torch.sum(collision_region_bool) > FINGER_COLLISION_THRESHOLD:
            return False
        else:
            return True
