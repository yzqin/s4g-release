import open3d
import numpy as np
from configs import config
from .pointcloud import PointCloud
from .torch_scene_point_cloud import TorchScenePointCloud
from tqdm import tqdm

import torch

DEBUG = False
x_dim = config.FINGER_LENGTH
y_dim = config.HALF_BOTTOM_SPACE * 2
z_dim = config.HALF_HAND_THICKNESS * 2
projection_resolution = config.PROJECTION_RESOLUTION
projection_margin = config.PROJECTION_MARGIN
x_unit = x_dim / (projection_resolution - projection_margin)
y_unit = y_dim / (projection_resolution - projection_margin)
z_unit = z_dim / (projection_resolution - projection_margin)
vox_unit = [x_unit, y_unit, z_unit]
gripper_dimensions = [x_dim, y_dim, z_dim]


class TorchBaseLineSingleViewPointCloud(PointCloud):
    def __init__(self, cloud: open3d.geometry.PointCloud,
                 camera_pose,
                 noise=True,
                 grasp_num=300,
                 remove_outliers=True,
                 voxelize=True,
                 filter_work_space=True,
                 visualization=False):
        """
        Point Cloud Class for single view points, which is rendered before training or captured by rgbd camera
        Note that we maintain a self.frame_indices to keep a subset of all points which will calculate frame and grading
        However, all of the points should have score, and of course those do not grading will have a 0 grade
        These setting will benefit the training since we want the network to learn to distinguish table and objects

        :param cloud: Base Open3D PointCloud, no preprocess is needed ofr this input, just read from file
        :param remove_outliers: Whether to remove outlier points
        :param voxelize: Whether to voxelize point cloud for a uniform sampling
        :param filter_work_space: Whether to remove all points outside the workspace
        :param visualization: Whether to visualize
        """

        PointCloud.__init__(self, cloud, visualization)
        self.grasp_num = grasp_num
        self.valid_grasp = 0
        if filter_work_space:
            self.filter_work_space()
        if voxelize:
            self.voxelize()
        if remove_outliers:
            self.remove_outliers()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise = noise
        self.camera_pos = camera_pose
        self.camera_pos_torch_inv = torch.tensor(np.linalg.inv(camera_pose), device=self.device).float()



        self.cloud_array = np.asarray(self.cloud.points).astype(np.float32)
        frame_indices = np.arange(self.points_num)[self.cloud_array[:, 2] > config.SAMPLE_REGION]
        np.random.shuffle(frame_indices)
        self.frame_indices = frame_indices
        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)
        self.frame = np.zeros((self.frame_indices.shape[0], 3, 3))
        self.global_to_local = torch.eye(4).unsqueeze(0).expand(self.frame_indices.shape[0], 4, 4).to(self.device)
        if not cloud.has_normals():
            self.estimate_normals(camera_pose)
        else:
            self.normals = np.asarray(self.cloud.normals)

        self.frame_antipodal_score = torch.zeros(self.grasp_num, dtype=torch.float32, device=self.device)
        self.baseline_frame = torch.zeros(self.grasp_num, 4, 4, dtype=torch.float32, device=self.device)
        self.valid_index = torch.zeros(grasp_num, dtype=torch.int, device=self.device)
        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

        self.close_region_points_set = []
        self.close_region_normals_set = []
        self.close_region_projection_map = []

    @property
    def points_num(self):
        return self.cloud_array.shape[0]

    def to_torch(self):
        self.frame = torch.FloatTensor(self.frame).to(self.device).contiguous()
        self.cloud_array = torch.FloatTensor(self.cloud_array).to(self.device)
        self.global_to_local = self.global_to_local.contiguous()

        self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        self.global_to_local[:, 0:3, 3:4] = -torch.bmm(self.frame.transpose(1, 2),
                                                       self.cloud_array[self.frame_indices, :].unsqueeze(2))
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
            return None

        M = np.eye(3) - normal.T @ normal
        xyz_centroid = np.mean(M @ self.normals[idx, :].T, axis=1, keepdims=True)
        normal_diff = self.normals[idx, :].T - xyz_centroid
        cov = normal_diff @ normal_diff.T
        eig_value, eig_vec = np.linalg.eigh(cov)

        minor_curvature = eig_vec[:, 0] - eig_vec[:, 0] @ normal.T * np.squeeze(normal)
        minor_curvature /= np.linalg.norm(minor_curvature)
        principal_curvature = np.cross(minor_curvature, np.squeeze(normal))

        self.frame[frame_index, :, :] = -np.stack([self.normals[index, :], principal_curvature, minor_curvature],
                                                  axis=1)

    def _antipodal_score(self, close_region_cloud, close_region_cloud_normal):
        """
        Estimate the antipodal score of a single grasp using scene point cloud
        Antipodal score is proportional to the reciprocal of friction angle
        Antipodal score is also divided by the square of objects in the closing region
        :param close_region_cloud: The point cloud in the gripper closing region, torch.tensor (3, n)
        :param close_region_cloud_normal: The point cloud normal in the gripper closing region, torch.tensor (3, n)
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

    def run_score(self, scene: TorchScenePointCloud = None):
        """
        Running the main loop for all points in the class: including frame estimation and grading
        If the mode is eval, the scene point cloud is the same as itself
        :param scene: The scene point cloud used for grading
        """

        self.estimate_frames()
        self.to_torch()

        frame_index = 0
        with tqdm(total=self.grasp_num, position=0) as pbar:
            while self.valid_grasp < self.grasp_num and frame_index < self.frame_indices.shape[0]:
                index = self.frame_indices[frame_index]
                if self.finger_hand(index, frame_index, scene):
                    self.valid_grasp += 1
                    pbar.update(1)
                pbar.set_postfix(search_num=frame_index)
                frame_index += 1

        if self.valid_grasp < self.grasp_num:
            return False
        else:
            return True

    def dump(self):
        """
        Damp the data inside this object to a python dictionary
        May need to store color later
        :return: Python dictionary store useful data
        """
        result_dict = {'antipodal_score': self.frame_antipodal_score[0:self.valid_grasp].cpu().numpy(),
                       'point_cloud': self.cloud_array.cpu().numpy().T,
                       # 'frame': self.frame,
                       # 'frame_index': self.frame_indices.cpu().numpy(),
                       'valid_index': self.valid_index[0:self.valid_grasp],
                       'baseline_frame': self.baseline_frame[0:self.valid_grasp, :, :].cpu().numpy(),
                       'close_region_points_set': [x.cpu().numpy() for x in self.close_region_points_set],
                       'close_region_normals_set': [x.cpu().numpy() for x in self.close_region_normals_set],
                       'close_region_projection_map_set': [x.cpu().numpy() for x in self.close_region_projection_map]
                       }
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
            return
        if point[2] + frame[2, 0] * config.FINGER_LENGTH < config.TABLE_HEIGHT:
            return

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

                    if torch.sum(close_region_bool.int()) < config.CLOSE_REGION_MIN_POINTS:
                        i += 1
                        continue

                    xyz_plane_normals = local_cloud_normal[:, close_plane_bool][:, close_region_bool]
                    close_region_cloud_normal = torch.matmul(
                        T_local_to_local_search_all[i % config.GRASP_PER_LENGTH, 0:3, 0:3],
                        xyz_plane_normals)
                    close_region_cloud = local_search_close_plane_points[:, close_region_bool]
                    antipodal_score = self._antipodal_score(close_region_cloud, close_region_cloud_normal)
                    if antipodal_score > self.frame_antipodal_score[self.valid_grasp]:
                        self.frame_antipodal_score[self.valid_grasp] = antipodal_score
                    else:
                        i+=1
                        continue

                    close_region_cloud[1, :] += config.HALF_BOTTOM_SPACE
                    close_region_cloud[2, :] += config.HALF_HAND_THICKNESS
                    close_region_points = close_region_cloud
                    close_region_normals = close_region_cloud_normal
                    close_region_projections = close_region_projection(close_region_cloud, close_region_cloud_normal)

                    best_frame = torch.matmul(
                        T_local_to_local_search_all[i % config.GRASP_PER_LENGTH],
                        self.global_to_local[frame_index])
        if self.frame_antipodal_score[self.valid_grasp] < 1e-4:
            return False
        else:
            self.valid_index[self.valid_grasp] = index
            self.close_region_points_set.append(close_region_points)
            self.close_region_normals_set.append(close_region_normals)
            self.close_region_projection_map.append(close_region_projections)
            self.baseline_frame[self.valid_grasp] = best_frame
            return True


def close_region_projection(points, normals):
    """
        Calculate the 12 channels projection map of the original GPD paper
        :param points: torch.tensor (3, N)
        :param normals: torch.tensor (3, N)
        :return:
            projection_map: (12, projection_resolution, projection_resolution)
        """
    projection_map = torch.zeros((12, config.PROJECTION_RESOLUTION, config.PROJECTION_RESOLUTION), device=points.device)
    x_cor = points[0, :] / x_unit
    y_cor = points[1, :] / y_unit
    z_cor = points[2, :] / z_unit

    x_cor = torch.floor(x_cor).long()
    y_cor = torch.floor(y_cor).long()
    z_cor = torch.floor(z_cor).long()

    valid = (x_cor >= 0) & (x_cor < projection_resolution) & (y_cor >= 0) & (y_cor < projection_resolution) \
            & (z_cor >= 0) & (z_cor < projection_resolution)

    x_cor = x_cor[valid]
    y_cor = y_cor[valid]
    z_cor = z_cor[valid]

    normals = normals[:, valid].squeeze(-1)

    flat_cor = x_cor * projection_resolution * projection_resolution + y_cor * projection_resolution + z_cor
    flat_cor = flat_cor.view(-1)
    expand_flat_cor = flat_cor.view(1, -1).expand(3, -1)

    norm_map = torch.zeros((3, projection_resolution, projection_resolution, projection_resolution)).to(
        points.device).view(3, -1)
    occupancy_map = torch.zeros((1, projection_resolution, projection_resolution, projection_resolution)).to(
        points.device).view(-1)
    occupancy_map.scatter_add_(0, flat_cor, torch.ones_like(flat_cor, dtype=torch.float, device=points.device))
    norm_map.scatter_add_(1, expand_flat_cor, normals)

    occupancy_map = occupancy_map.view(1, projection_resolution, projection_resolution, projection_resolution)
    norm_map = norm_map.view(3, projection_resolution, projection_resolution, projection_resolution)

    norm_map = norm_map / torch.clamp(occupancy_map, 1e-4)
    occupancy_map = (occupancy_map > 0).float()

    order_list = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    for i, order in enumerate(order_list):
        height_map = torch.linspace(0.5 * vox_unit[order[-1]],
                                    gripper_dimensions[order[-1]] - 0.5 * vox_unit[order[-1]],
                                    projection_resolution).view(1, 1, 1, projection_resolution).to(points.device)

        curr_occupancy_map = occupancy_map.contiguous().permute(0, order[0] + 1, order[1] + 1, order[2] + 1)
        curr_norm_map = norm_map.contiguous().permute(0, order[0] + 1, order[1] + 1, order[2] + 1)

        projection_occupancy_map = curr_occupancy_map.sum(3)
        projection_norm_map = curr_norm_map.sum(3) / torch.clamp(projection_occupancy_map, 1e-4)
        projection_height_map = (curr_occupancy_map * height_map).sum(3) / torch.clamp(projection_occupancy_map, 1e-4)

        projection_map[4 * i:4 * i + 1, :, :] = projection_height_map
        projection_map[4 * i + 1: 4 * i + 4, :, :] = projection_norm_map

    return projection_map
