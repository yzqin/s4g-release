import numpy as np
import open3d
import torch
from tqdm import tqdm

from configs import config
from pcd_classes.torch_data_scene_point_cloud import TorchDataScenePointCloud
from .pointcloud import PointCloud

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


class TorchPrecomputedBaselinePointCloud(PointCloud):
    def __init__(self, cloud: open3d.geometry.PointCloud,
                 noise_cloud: open3d.geometry.PointCloud,
                 camera_pose,
                 grasp_num=500,
                 visualization=False):
        """
        Point Cloud Class for single view points, which is rendered before training or captured by rgbd camera
        Note that we maintain a self.frame_indices to keep a subset of all points which will calculate frame and grading
        However, all of the points should have score, and of course those do not grading will have a 0 grade
        These setting will benefit the training since we want the network to learn to distinguish table and objects

        :param cloud: Base Open3D PointCloud, no pre-process is needed ofr this input, just read from file
        :param visualization: Whether to visualize
        """

        assert np.asarray(cloud.points).shape == np.asarray(noise_cloud.points).shape
        PointCloud.__init__(self, noise_cloud, visualization)
        self.reference_cloud = np.asarray(cloud.points)
        self.grasp_num = grasp_num

        self.index_in_ref = self.processing_and_trace()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.camera_pos = camera_pose
        self.camera_pos_torch_inv = torch.tensor(np.linalg.inv(camera_pose), device=self.device).float()

        self.cloud_array = np.asarray(self.cloud.points).astype(np.float32)
        self.normals = np.zeros(self.cloud_array.shape, dtype=self.cloud_array.dtype)
        self.valid_grasp = int(0)

        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

        self.frame_indices = None
        self.global_to_local = None
        self.valid_frame = None
        self.valid_index = None
        self.frame = None
        self.local_to_global = None
        self.frame_antipodal_score = None
        self.pre_antipodal_score = None
        self.cloud_array_homo = None
        self.all_frame_bool = None
        self.normal_view = None

        self.close_region_points_set = []
        self.close_region_normals_set = []
        self.close_region_projection_map = []

    @property
    def points_num(self):
        return self.cloud_array.shape[0]

    def processing_and_trace(self):
        # Filter workspace
        workspace = config.WORKSPACE
        has_normal = self.cloud.has_normals()
        has_color = self.cloud.has_colors()
        min_bound = np.array([[workspace[0]], [workspace[2]], [workspace[4]]])
        max_bound = np.array([[workspace[1]], [workspace[3]], [workspace[5]]])
        points = np.asarray(self.cloud.points)
        points_num = points.shape[0]
        valid_index = np.logical_and.reduce((
            points[:, 0] > min_bound[0, 0], points[:, 1] > min_bound[1, 0], points[:, 2] > min_bound[2, 0],
            points[:, 0] < max_bound[0, 0], points[:, 1] < max_bound[1, 0], points[:, 2] < max_bound[2, 0],
        ))
        points = points[valid_index, :]
        self.cloud.points = open3d.utility.Vector3dVector(points)
        if has_color:
            color = np.asarray(self.cloud.colors)[valid_index, :]
            self.cloud.colors = open3d.utility.Vector3dVector(color)

        if has_normal:
            normals = np.asarray(self.cloud.normals)[valid_index, :]
            self.cloud.normals = open3d.utility.Vector3dVector(normals)

        # Voxelize
        self.cloud, trace_index_voxel = self.cloud.voxel_down_sample_and_trace(config.VOXEL_SIZE, [-0.5, -0.5, 0.7],
                                                                               [0.5, 0.5, 1.5])
        valid_index2 = np.max(trace_index_voxel, axis=1)

        # Remove outlier
        self.cloud, valid_index3 = self.cloud.remove_radius_outlier(config.NUM_POINTS_THRESHOLD,
                                                                    config.RADIUS_THRESHOLD)
        index_in_ref = np.arange(points_num)[valid_index][valid_index2][valid_index3]

        return index_in_ref

    def to_torch(self):
        possible_frame_num = len(self.frame_indices)
        length_num = len(config.LENGTH_SEARCH)

        self.global_to_local = torch.eye(4).unsqueeze(0).expand(possible_frame_num, 4, 4).to(self.device)
        self.valid_frame = torch.zeros(possible_frame_num, 4, 4, device=self.device).float()
        self.valid_index = torch.zeros(possible_frame_num, device=self.device).int()

        self.frame_antipodal_score = torch.zeros(possible_frame_num, dtype=torch.float32, device=self.device)
        self.frame = torch.tensor(self.frame).float().to(self.device).contiguous()
        self.cloud_array = torch.tensor(self.cloud_array).float().to(self.device)
        self.normal_view = torch.tensor(self.normals.T, dtype=torch.float, device=self.device)
        self.cloud_array_homo = torch.cat([self.cloud_array.transpose(1, 0),
                                           torch.ones(1, self.points_num, dtype=torch.float, device=self.device)])
        self.global_to_local = self.global_to_local.contiguous()
        self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        self.global_to_local[:, 0:3, 3:4] = -torch.bmm(self.frame.transpose(1, 2),
                                                       self.cloud_array[self.frame_indices, :].unsqueeze(2))
        self.local_to_global = torch.inverse(self.global_to_local).contiguous()
        self.frame_indices = torch.tensor(self.frame_indices, dtype=torch.int, device=self.device)

        self.pre_antipodal_score = torch.tensor(self.pre_antipodal_score, dtype=torch.float, device=self.device)

    def _find_match(self, scene: TorchDataScenePointCloud):
        single_frame_bool = np.zeros([self.cloud_array.shape[0]], dtype=np.bool)
        kd_tree = scene.kd_tree
        frame = np.tile(np.eye(3), [self.cloud_array.shape[0], 1, 1])
        antipodal_score = np.zeros([self.cloud_array.shape[0], len(config.LENGTH_SEARCH), config.GRASP_PER_LENGTH])
        inv_antipodal_score = np.zeros(antipodal_score.shape)
        for index in range(self.cloud_array.shape[0]):
            [k, idx, _] = kd_tree.search_hybrid_vector_3d(self.reference_cloud[self.index_in_ref[index], :],
                                                          radius=config.CURVATURE_RADIUS,
                                                          max_nn=1)
            if k < 1:
                self.normals[index, :] = [0, 0, 1]
                single_frame_bool[index] = False
                continue

            i = idx[0]

            frame[index, :, :] = scene.frame[i, :, :]
            self.normals[index, :] = scene.normals[i, :]
            single_frame_bool[index] = True
            antipodal_score[index, :, :] = scene.antipodal_score[i, :, :]
            inv_antipodal_score[index, :, :] = scene.inv_antipodal_score[i, :, :]

        self.cloud.normals = open3d.utility.Vector3dVector(self.normals)
        self.cloud.normalize_normals()
        self.cloud.orient_normals_towards_camera_location(camera_location=self.camera_pos[0:3, 3])
        self.normals = np.asarray(self.cloud.normals)

        self.frame = frame
        bool_frame = np.sum(self.normals * self.frame[:, :, 0], axis=1) > 0
        print('Wrong frame direction correction: {}/{}'.format(np.sum(bool_frame), bool_frame.shape[0]))

        self.frame[bool_frame, :, 0:2] = -self.frame[bool_frame, :, 0:2]
        self.pre_antipodal_score = antipodal_score
        self.pre_antipodal_score[bool_frame, :, :] = inv_antipodal_score[bool_frame, :, :]

        possible_frame_indices, self.all_frame_bool = self.magic_formula(single_frame_bool)
        sample_frame_indices = np.arange(self.points_num)[self.cloud_array[:, 2] > config.SAMPLE_REGION]
        self.frame_indices = np.intersect1d(possible_frame_indices, sample_frame_indices)
        self.frame = frame[self.frame_indices, :, :]
        self.pre_antipodal_score = self.pre_antipodal_score[self.frame_indices, :, :]
        self.all_frame_bool = self.all_frame_bool[self.frame_indices, :, :]

    def magic_formula(self, single_frame_bool):
        antipodal_bool = self.pre_antipodal_score > 0.3
        frame_bool = np.logical_and(antipodal_bool.max(2).max(1), single_frame_bool)
        return np.nonzero(frame_bool), antipodal_bool

    def run_score(self, scene: TorchDataScenePointCloud):
        """
        Running the main loop for all points in the class: including frame estimation and grading
        If the mode is eval, the scene point cloud is the same as itself
        :param scene: The scene point cloud used for grading
        """
        self._find_match(scene)
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
        valid_frame_global = torch.inverse(self.valid_frame[0:self.valid_grasp])
        cloud_array = torch.matmul(self.camera_pos_torch_inv, self.cloud_array_homo)
        valid_frame = torch.matmul(self.camera_pos_torch_inv.unsqueeze(0),
                                   valid_frame_global)
        result_dict = {
            'antipodal_score': self.frame_antipodal_score[0:self.valid_grasp].cpu().numpy(),
            'point_cloud': cloud_array[0:3, ].cpu().numpy(),
            'valid_index': self.valid_index[0:self.valid_grasp].cpu().numpy(),
            'close_region_points_set': [x.cpu().numpy() for x in self.close_region_points_set],
            'close_region_normals_set': [x.cpu().numpy() for x in self.close_region_normals_set],
            'close_region_projection_map_set': [x.cpu().numpy() for x in self.close_region_projection_map],
            'baseline_frame': valid_frame.cpu().numpy()}

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
            T_local_to_global.unsqueeze(0).expand(config.LOCAL_SEARCH_TO_LOCAL.shape[0], 4, 4).contiguous(),
            config.LOCAL_SEARCH_TO_LOCAL)
        boundary_global = torch.bmm(T_local_search_to_global_all, config.TORCH_GRIPPER_BOUND.squeeze(0).expand(
            T_local_search_to_global_all.shape[0], -1, -1).contiguous())
        table_collision_bool_all = boundary_global[:, 2, :] < config.TABLE_HEIGHT + config.TABLE_COLLISION_OFFSET
        return table_collision_bool_all.any(dim=1, keepdim=False)

    def finger_hand(self, index, frame_index, scene: TorchDataScenePointCloud):
        """
        Local search one point and store the closing region point num of each configurations
        Search height first, then width, finally theta
        Save the number of points in the close region if the grasp do not fail in local search
        Save the score of antipodal_grasp, note that multi-objects heuristic is also stored here
        :param index: The index of point in all single view cloud
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        :param scene: The complete scene point cloud generated to give a score
        """
        # if index == 8325:
        #     print('debug')
        #     a = 1
        frame_bool = self.all_frame_bool[frame_index, :, :]

        frame = self.frame[frame_index, :, :]
        point = self.cloud_array[index, :]

        if torch.mean(torch.abs(frame)) < 1e-6:
            return False

        table_collision_bool = self._table_collision_check(point, frame)

        T_global_to_local = self.global_to_local[frame_index, :, :]
        local_cloud = torch.matmul(T_global_to_local, scene.cloud_array_homo)
        best_local_index = int(-1)

        i = 0
        for dl_num, dl in enumerate(config.LENGTH_SEARCH):
            if not np.any(frame_bool[dl_num, :]):
                i += config.GRASP_PER_LENGTH
                continue

            close_plane_bool = (local_cloud[0, :] < dl + config.FINGER_LENGTH) & (
                    local_cloud[0, :] > dl - config.BOTTOM_LENGTH)

            if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
                i += config.GRASP_PER_LENGTH
                continue

            close_plane_points = local_cloud[:, close_plane_bool]  # only filter along x axis

            T_local_to_local_search_all = config.LOCAL_TO_LOCAL_SEARCH[
                                          dl_num * config.GRASP_PER_LENGTH:(dl_num + 1) * config.GRASP_PER_LENGTH,
                                          :, :]

            local_search_close_plane_points_all = torch.matmul(T_local_to_local_search_all.contiguous().view(-1, 4),
                                                               close_plane_points).contiguous().view(
                config.GRASP_PER_LENGTH, 4, -1)[:, 0:3, :]

            for _ in range(config.GRASP_PER_LENGTH):
                if table_collision_bool[i] or not frame_bool[dl_num, i % config.GRASP_PER_LENGTH]:
                    i += 1
                    continue

                local_search_close_plane_points = local_search_close_plane_points_all[i % config.GRASP_PER_LENGTH,
                                                  :, :]

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

                    if close_region_point_num < config.CLOSE_REGION_MIN_POINTS:
                        i += 1
                        continue

                    antipodal = self.pre_antipodal_score[
                        frame_index, i // config.GRASP_PER_LENGTH, i % config.GRASP_PER_LENGTH]
                    if antipodal > self.frame_antipodal_score[self.valid_grasp]:
                        self.frame_antipodal_score[self.valid_grasp] = antipodal
                    else:
                        i += 1
                        continue

                    # close_region_normals = local_search_normals[:, close_plane_bool][:, close_region_bool]
                    # close_region_projections = close_region_projection(close_region_view_cloud, close_region_normals)
                    best_local_index = i
                    i += 1

        if best_local_index == -1:
            return False
        else:
            local_view_cloud = torch.matmul(T_global_to_local, self.cloud_array_homo)
            local_view_normal = torch.matmul(T_global_to_local[0:3, 0:3], self.normal_view)
            local_search_view_cloud = torch.matmul(config.LOCAL_TO_LOCAL_SEARCH[best_local_index, :, :],
                                                   local_view_cloud)
            local_search_normals = torch.matmul(config.LOCAL_TO_LOCAL_SEARCH[best_local_index, 0:3, 0:3],
                                                local_view_normal)

            view_z_bool = (local_search_view_cloud[2, :] < config.HALF_HAND_THICKNESS) & \
                          (local_search_view_cloud[2, :] > -config.HALF_HAND_THICKNESS)
            view_y_bool = (local_search_view_cloud[1, :] < config.HALF_BOTTOM_SPACE) & \
                          (local_search_view_cloud[1, :] > -config.HALF_BOTTOM_SPACE)
            view_x_bool = (local_search_view_cloud[0, :] < config.FINGER_LENGTH) & \
                          (local_search_view_cloud[0, :] > 0)
            view_close_region_bool = view_z_bool & view_x_bool & view_y_bool

            close_region_view_cloud = local_search_view_cloud[0:3, view_close_region_bool]
            close_region_normals = local_search_normals[0:3, view_close_region_bool]
            close_region_view_cloud[1, :] += config.HALF_BOTTOM_SPACE
            close_region_view_cloud[2, :] += config.HALF_HAND_THICKNESS
            if self.valid_grasp == 317:
                print('ss')
                pass
            close_region_projections = close_region_projection(close_region_view_cloud, close_region_normals)

            self.valid_index[self.valid_grasp] = index
            self.close_region_points_set.append(close_region_view_cloud)
            self.close_region_normals_set.append(close_region_normals)
            self.close_region_projection_map.append(close_region_projections)
            self.valid_frame[self.valid_grasp] = torch.matmul(config.LOCAL_TO_LOCAL_SEARCH[best_local_index, :, :],
                                                              T_global_to_local)
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
