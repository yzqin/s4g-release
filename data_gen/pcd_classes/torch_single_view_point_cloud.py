import open3d
from configs.dataset_config import NAME_LIST
import numpy as np
from configs import config
from .pointcloud import PointCloud
from .torch_scene_point_cloud import TorchScenePointCloud
from tqdm import tqdm, trange

import torch

DEBUG = False


class TorchSingleViewPointCloud(PointCloud):
    def __init__(self, cloud: open3d.geometry.PointCloud,
                 camera_pose,
                 noise=True,
                 remove_outliers=True,
                 voxelize=True,
                 filter_work_space=True,
                 estimate_normals=False,
                 visualization=False):
        """
        Point Cloud Class for single view points, which is rendered before training or captured by rgbd camera
        Note that we maintain a self.frame_indices to keep a subset of all points which will calculate frame and grading
        However, all of the points should have score, and of course those do not grading will have a 0 grade
        These setting will benefit the training since we want the network to learn to distinguish table and objects

        :param cloud: Base Open3D PointCloud, no pre-process is needed ofr this input, just read from file
        :param remove_outliers: Whether to remove outlier points
        :param voxelize: Whether to voxelize point cloud for a uniform sampling
        :param filter_work_space: Whether to remove all points outside the workspace
        :param estimate_normals: If estimate normal, it will not match normal with scene but calculated explicitly
        :param visualization: Whether to visualize
        """

        PointCloud.__init__(self, cloud, visualization)
        work_space = config.WORKSPACE
        if filter_work_space:
            self.filter_work_space(workspace=work_space)
        if voxelize:
            self.voxelize()
        if remove_outliers:
            self.remove_outliers()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise = noise
        self.camera_pos = camera_pose
        self.camera_pos_torch_inv = torch.tensor(np.linalg.inv(camera_pose), device=self.device).float()

        self.cloud_array = np.asarray(self.cloud.points).astype(np.float32)
        self.normals = np.zeros(self.cloud_array.shape, dtype=self.cloud_array.dtype)
        self.frame_indices = np.arange(self.points_num)[self.cloud_array[:, 2] > config.SAMPLE_REGION]
        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)
        self.frame = np.tile(np.eye(3), [self.frame_indices.shape[0], 1, 1])
        self.global_to_local = torch.eye(4).unsqueeze(0).expand(self.frame_indices.shape[0], 4, 4).to(self.device)
        self.local_to_global = None
        if estimate_normals:
            self.estimate_normals(camera_pose[0:3, 3])
        if cloud.has_normals():
            assert not estimate_normals
            self.normals = np.asarray(self.cloud.normals)

        length_num = len(config.LENGTH_SEARCH)
        self.frame_search_score = torch.zeros(self.frame_num, length_num, config.GRASP_PER_LENGTH, dtype=torch.int,
                                              device=self.device)
        self.frame_antipodal_score = torch.zeros(self.frame_num, length_num, config.GRASP_PER_LENGTH,
                                                 dtype=torch.float32,
                                                 device=self.device)
        self.valid_frame = torch.zeros(self.frame_indices.shape[0], length_num, config.GRASP_PER_LENGTH, 4, 4,
                                       device=self.device).float()
        self.valid_grasp = int(0)
        self.valid_index = torch.zeros(self.frame_indices.shape[0], device=self.device).int()
        self.frame_objects_label = torch.ones(self.frame_num, length_num, config.GRASP_PER_LENGTH, dtype=torch.int16,
                                              device=self.device) * len(NAME_LIST)

        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

    @property
    def points_num(self):
        return self.cloud_array.shape[0]

    @property
    def frame_num(self):
        return self.frame_indices.shape[0]

    def to_torch(self):
        self.frame = torch.tensor(self.frame).float().to(self.device).contiguous()
        self.cloud_array = torch.tensor(self.cloud_array).float().to(self.device)
        self.global_to_local = self.global_to_local.contiguous()
        self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        self.global_to_local[:, 0:3, 3:4] = -torch.bmm(self.frame.transpose(1, 2),
                                                       self.cloud_array[self.frame_indices, :].unsqueeze(2))
        self.local_to_global = torch.inverse(self.global_to_local).contiguous()
        self.frame_indices = torch.tensor(self.frame_indices, device=self.device).int()

    def estimate_frames(self):
        """
        Estimate Darboux frame for all points in the self.frame_indices
        """

        for frame_index in range(len(self.frame_indices)):
            index = self.frame_indices[frame_index]
            self._estimate_frame(index, frame_index)

    def _estimate_frame(self, index: int, frame_index: int):
        """
        Estimate the Darboux frame of single point
        In self.frame, each column of one point frame is a vec3, with the order of x, y, z axis
        Note there is a minus sign of the whole frame, which means that x is the negative direction of normal
        :param index: The index of point in all single view cloud
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        """

        [k, idx, _] = self.kd_tree.search_radius_vector_3d(self.cloud_array[index, :], config.CURVATURE_RADIUS)
        normal = self.normals[index:index + 1, :]
        if k < 5:
            # self.frame[frame_index, :, :] = np.zeros([3, 3])
            return

        M = np.eye(3) - normal.T @ normal
        xyz_centroid = np.mean(M @ self.normals[idx, :].T, axis=1, keepdims=True)
        normal_diff = self.normals[idx, :].T - xyz_centroid
        cov = normal_diff @ normal_diff.T
        eig_value, eig_vec = np.linalg.eigh(cov)

        minor_curvature = eig_vec[:, 0] - eig_vec[:, 0] @ normal.T * np.squeeze(normal)
        minor_curvature /= np.linalg.norm(minor_curvature)
        principal_curvature = np.cross(minor_curvature, np.squeeze(normal))

        self.frame[frame_index, :, :] = np.stack([-self.normals[index, :], -principal_curvature, minor_curvature],
                                                 axis=1)

    def _find_normal(self, scene: TorchScenePointCloud):
        kd_tree = scene.kd_tree
        for index in range(self.cloud_array.shape[0]):
            [k, idx, _] = kd_tree.search_hybrid_vector_3d(self.cloud_array[index, :], radius=config.CURVATURE_RADIUS,
                                                          max_nn=config.NORMAL_MAX_NN)
            average_normal = np.mean(scene.normals[idx, :], axis=0, keepdims=False)
            self.normals[index, :] = average_normal

        self.cloud.normals = open3d.utility.Vector3dVector(self.normals)
        self.cloud.normalize_normals()
        if True:
            self.cloud.orient_normals_towards_camera_location(camera_location=self.camera_pos[0:3, 3])
        if self.visualization:
            open3d.visualization.draw_geometries([self.cloud])

        self.normals = np.asarray(self.cloud.normals)

    def _antipodal_score(self, close_region_cloud, close_region_cloud_normal, index: int, frame_num: int):
        """
        Estimate the antipodal score of a single grasp using scene point cloud
        Antipodal score is proportional to the reciprocal of friction angle
        Antipodal score is also divided by the square of objects in the closing region
        :param close_region_cloud: The point cloud in the gripper closing region, torch.tensor (3, n)
        :param close_region_cloud_normal: The point cloud normal in the gripper closing region, torch.tensor (3, n)
        :param index: The index of graded frame in self.cloud_array
        :param frame_num: The frame num of the point index above, to grade self.frame_antipodal_score[index, frame_num]
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

        # Label based grading
        self.frame_antipodal_score[
            self.valid_grasp, frame_num // config.GRASP_PER_LENGTH, frame_num % config.GRASP_PER_LENGTH] = geometry_average_theta

    def run_score(self, scene: TorchScenePointCloud, match_normal: bool):
        """
        Running the main loop for all points in the class: including frame estimation and grading
        If the mode is eval, the scene point cloud is the same as itself
        :param scene: The scene point cloud used for grading
        :param match_normal: Whether to match normal with the scene point cloud
        """
        if match_normal:
            self._find_normal(scene)
        # open3d.visualization.draw_geometries([self.cloud])
        self.estimate_frames()
        self.to_torch()
        if self.frame_indices.shape[0] == 0:
            import ipdb
            ipdb.set_trace()

        for frame_index in trange(self.frame_indices.shape[0]):
            index = self.frame_indices[frame_index]
            if self.finger_hand(index, frame_index, scene):
                self.valid_grasp += 1

    def dump(self):
        """
        Damp the data inside this object to a python dictionary
        May need to store color later
        :return: Python dictionary store useful data
        """
        cloud_array = torch.cat(
            [self.cloud_array, torch.ones([self.cloud_array.shape[0], 1], dtype=torch.float, device=self.device), ],
            dim=1)
        cloud_array = torch.matmul(self.camera_pos_torch_inv, cloud_array.transpose(0, 1))
        valid_frame = torch.matmul(self.camera_pos_torch_inv.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                   self.valid_frame[0:self.valid_grasp, :, :, :, :])
        result_dict = {'search_score': self.frame_search_score[0:self.valid_grasp, :, :].cpu().numpy(),
                       'antipodal_score': self.frame_antipodal_score[0:self.valid_grasp, :, :].cpu().numpy(),
                       'objects_label': self.frame_objects_label[0:self.valid_grasp, :, :].cpu().numpy(),
                       'point_cloud': cloud_array[0:3, ].cpu().numpy(),
                       'valid_index': self.valid_index[0:self.valid_grasp].cpu().numpy(),
                       'valid_frame': valid_frame.cpu().numpy()}

        return result_dict

    def _table_collision_check(self, point, frame):
        """
        Check whether the gripper collide with the table top with offset
        :param point: torch.tensor(3)
        :param frame: torch.tensor(3, 3)
        :return: a torch boolean tensor with shape (len(config.INDEX_TO_ARRAY))
        """

        T_local_to_global = torch.eye(4, device=self.device).float()
        T_local_to_global[0:3, 0:3] = frame
        T_local_to_global[0:3, 3] = point
        T_local_search_to_global_all = torch.bmm(
            T_local_to_global.squeeze(0).expand(config.LOCAL_SEARCH_TO_LOCAL.shape[0], 4, 4).contiguous(),
            config.LOCAL_SEARCH_TO_LOCAL)
        boundary_global = torch.bmm(T_local_search_to_global_all, config.TORCH_GRIPPER_BOUND.squeeze(0).expand(
            T_local_search_to_global_all.shape[0], -1, -1).contiguous())
        table_collision_bool_all = boundary_global[:, 2, :] < config.TABLE_HEIGHT + config.TABLE_COLLISION_OFFSET
        return table_collision_bool_all.any(dim=1, keepdim=False)

    def finger_hand(self, index, frame_index, scene: TorchScenePointCloud):
        """
        Local search one point and store the closing region point num of each configurations
        Search height first, then width, finally theta
        Save the number of points in the close region if the grasp do not fail in local search
        Save the score of antipodal_grasp, note that multi-objects heuristic is also stored here
        :param index: The index of point in all single view cloud
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        :param scene: The complete scene point cloud generated to give a score
        """

        frame = self.frame[frame_index, :, :]
        point = self.cloud_array[index, :]

        if torch.mean(torch.abs(frame)) < 1e-6:
            return False
        if point[2] + frame[2, 0] * config.FINGER_LENGTH < config.TABLE_HEIGHT:
            return False

        table_collision_bool = self._table_collision_check(point, frame)

        T_global_to_local = self.global_to_local[frame_index, :, :]
        local_cloud = torch.matmul(T_global_to_local, scene.cloud_array_homo)

        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3], scene.normal_array)
        i = 0
        for dl_num, dl in enumerate(config.LENGTH_SEARCH):
            close_plane_bool = (local_cloud[0, :] < dl + config.FINGER_LENGTH) & (
                    local_cloud[0, :] > dl - config.BOTTOM_LENGTH)

            if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
                i += config.GRASP_PER_LENGTH
                continue

            close_plane_points = local_cloud[:, close_plane_bool]  # only filter along x axis

            T_local_to_local_search_all = config.LOCAL_TO_LOCAL_SEARCH[
                                          dl_num * config.GRASP_PER_LENGTH:(dl_num + 1) * config.GRASP_PER_LENGTH, :, :]

            local_search_close_plane_points_all = torch.matmul(T_local_to_local_search_all.contiguous().view(-1, 4),
                                                               close_plane_points).contiguous().view(
                config.GRASP_PER_LENGTH, 4, -1)[:, 0:3, :]

            for _ in range(config.GRASP_PER_LENGTH):

                if table_collision_bool[i]:
                    i += 1
                    continue

                local_search_close_plane_points = local_search_close_plane_points_all[i % config.GRASP_PER_LENGTH, :, :]

                z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                                   (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)

                back_collision_bool = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                      (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                      (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN) & \
                                      z_collision_bool

                if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
                    i += 1
                    continue

                y_finger_region_bool_left = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                            (local_search_close_plane_points[1, :] > config.HALF_BOTTOM_SPACE)
                y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                             (local_search_close_plane_points[1, :] < -config.HALF_BOTTOM_SPACE)

                y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
                collision_region_bool = (z_collision_bool & y_finger_region_bool)
                if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
                    i += 1
                    continue
                else:
                    close_region_bool = z_collision_bool & \
                                        (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_SPACE) & \
                                        (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_SPACE)

                    close_region_point_num = torch.sum(close_region_bool)

                    if close_region_point_num < config.CLOSE_REGION_MIN_POINTS > 1:
                        i += 1
                        continue
                    close_region_label = torch.unique(scene.label_array[close_plane_bool][close_region_bool],
                                                      sorted=False)
                    if close_region_label.shape[0] > 1:
                        i += 1
                        continue

                    self.frame_search_score[
                        self.valid_grasp, i // config.GRASP_PER_LENGTH, i % config.GRASP_PER_LENGTH] = close_region_point_num
                    self.frame_objects_label[
                        self.valid_grasp, i // config.GRASP_PER_LENGTH, i % config.GRASP_PER_LENGTH] = \
                        close_region_label[0]

                    xyz_plane_normals = local_cloud_normal[:, close_plane_bool][:, close_region_bool]
                    close_region_cloud_normal = torch.matmul(
                        T_local_to_local_search_all[i % config.GRASP_PER_LENGTH, 0:3, 0:3],
                        xyz_plane_normals)
                    close_region_cloud = local_search_close_plane_points[:, close_region_bool]

                    self._antipodal_score(close_region_cloud, close_region_cloud_normal, index, i)

                    i += 1

        if torch.max(self.frame_antipodal_score[self.valid_grasp]) < 1e-4:
            return False
        else:
            local_to_global = self.local_to_global[frame_index, :, :]

            frame_all = torch.bmm(local_to_global.unsqueeze(0).expand(len(config.INDEX_TO_ARRAY), 4, 4).contiguous(),
                                  config.TORCH_LOCAL_SEARCH_TO_LOCAL)
            self.valid_frame[self.valid_grasp, :, :, :] = frame_all.contiguous().view(len(config.LENGTH_SEARCH),
                                                                                      config.GRASP_PER_LENGTH, 4, 4)
            self.valid_index[self.valid_grasp] = index
            return True
