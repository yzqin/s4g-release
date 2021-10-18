import numpy as np
import open3d
import torch

from . import config
from .pointcloud import PointCloudProcessor
from .torch_scene_point_cloud import TorchScenePointCloud


class EvalExpCloud(PointCloudProcessor):
    def __init__(self, cloud: open3d.geometry.PointCloud,
                 visualization=False):
        """
        Point Cloud Class for single view points, which is rendered before training or captured by rgbd camera
        Note that we maintain a self.frame_indices to keep a subset of all points which will calculate frame and grading
        However, all of the points should have score, and of course those do not grading will have a 0 grade
        These setting will benefit the training since we want the network to learn to distinguish table and objects

        :param cloud: Base Open3D PointCloud, no preprocess is needed ofr this input, just read from file
        :param visualization: Whether to visualize
        """

        PointCloudProcessor.__init__(self, cloud, visualization)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cloud_array = np.asarray(self.cloud.points).astype(np.float32)
        self.cloud_array = torch.tensor(cloud_array).float().to(self.device)
        self.cloud_array_homo = torch.cat(
            [self.cloud_array.transpose(0, 1), torch.ones(1, self.cloud_array.shape[0], device=self.device)],
            dim=0).float().to(self.device)

        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

    @property
    def points_num(self):
        return self.cloud_array.shape[0]

    def _antipodal_score(self, close_region_cloud, close_region_cloud_normal):
        """
        Estimate the antipodal score of a single grasp using scene point cloud
        Antipodal score is proportional to the reciprocal of friction angle
        Antipodal score is also divided by the square of objects in the closing region
        :param close_region_cloud: The point cloud in the gripper closing region, torch.tensor (3, n)
        :param close_region_cloud_normal: The point cloud normal in the gripper closing region, torch.tensor (3, n)
        :return close_region_cloud_normal for calculating projection
        """

        assert close_region_cloud.shape == close_region_cloud_normal.shape, \
            "Points and corresponding normals should have same shape"

        left_y = torch.max(close_region_cloud[1, :])
        right_y = torch.min(close_region_cloud[1, :])
        normal_search_depth = torch.min((left_y - right_y) / 3, config.NEIGHBOR_DEPTH)

        left_region_bool = close_region_cloud[1, :] > left_y - normal_search_depth
        right_region_bool = close_region_cloud[1, :] < right_y + normal_search_depth
        left_normal_theta = torch.abs(torch.matmul(self.left_normal, close_region_cloud_normal[:, left_region_bool]))
        right_normal_theta = torch.abs(torch.matmul(self.right_normal, close_region_cloud_normal[:, right_region_bool]))

        geometry_average_theta = torch.mean(left_normal_theta) * torch.mean(right_normal_theta)
        return geometry_average_theta

    def eval_frame(self, T_global_to_local, scene: TorchScenePointCloud):
        result = {'antipodal_score': 0, 'collision': False, 'multi_objects': False}

        local_cloud = torch.matmul(T_global_to_local, scene.cloud_array_homo)

        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3], scene.normal_array)
        close_plane_bool = (local_cloud[0, :] < config.FINGER_LENGTH) & (
                local_cloud[0, :] > -config.BOTTOM_LENGTH)

        local_search_close_plane_points = local_cloud[:, close_plane_bool][0:3, :]  # only filter along x axis

        z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                           (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)

        back_collision_bool = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                              (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                              (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN) & \
                              z_collision_bool

        if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
            result['collision'] = True

        y_finger_region_bool_left = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                    (local_search_close_plane_points[1, :] > config.HALF_BOTTOM_SPACE)
        y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                     (local_search_close_plane_points[1, :] < -config.HALF_BOTTOM_SPACE)

        y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
        collision_region_bool = (z_collision_bool & y_finger_region_bool)
        if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
            result['collision'] = True
        close_region_bool = z_collision_bool & \
                            (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_SPACE) & \
                            (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_SPACE)

        close_region_label = torch.unique(scene.label_array[close_plane_bool][close_region_bool],
                                          sorted=False)
        if close_region_label.shape[0] > 1:
            result['multi_objects'] = True

        close_region_normals = local_cloud_normal[:, close_plane_bool][:, close_region_bool]
        close_region_cloud = local_search_close_plane_points[:, close_region_bool]

        if close_region_cloud.shape[1] < config.CLOSE_REGION_MIN_POINTS:
            return result

        if result['collision'] or result['multi_objects']:
            return result
        result['antipodal_score'] = self._antipodal_score(close_region_cloud, close_region_normals).cpu().numpy()
        return result

    def view_non_collision(self, T_global_to_local):
        local_cloud = torch.matmul(T_global_to_local, self.cloud_array_homo)
        close_plane_bool = (local_cloud[0, :] < config.FINGER_LENGTH) & (
                local_cloud[0, :] > -config.BOTTOM_LENGTH)

        local_search_close_plane_points = local_cloud[:, close_plane_bool][0:3, :]  # only filter along x axis

        z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                           (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)

        back_collision_bool = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                              (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                              (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN) & \
                              z_collision_bool

        if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
            return False

        y_finger_region_bool_left = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                    (local_search_close_plane_points[1, :] > config.HALF_BOTTOM_SPACE)
        y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                     (local_search_close_plane_points[1, :] < -config.HALF_BOTTOM_SPACE)

        y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
        collision_region_bool = (z_collision_bool & y_finger_region_bool)

        if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
            return False
        else:
            return True
