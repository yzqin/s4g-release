import numpy as np
import open3d
import torch
from tqdm import trange

from configs import config
from configs.dataset_config import NAME_LIST
from pcd_classes.torch_contact_scene_point_cloud import TorchContactScenePointCloud
from .pointcloud import PointCloud

WIDTH_SEARCH = [-0.005, 0.005, 0]
HEIGHT_SEARCH = [-0.005, 0.005, 0]
# HEIGHT_SEARCH = [0]
LENGTH_SEARCH = [0]
# LENGTH_SEARCH = [0]

NUM_OF_LOCAL_SEARCH = len(WIDTH_SEARCH) * len(HEIGHT_SEARCH) * len(LENGTH_SEARCH)
LOCAL_SEARCH_TO_LOCAL = torch.eye(4).unsqueeze(0).expand(NUM_OF_LOCAL_SEARCH, 4, 4)
ii = 0
for dx in LENGTH_SEARCH:
    for dy in WIDTH_SEARCH:
        for dz in HEIGHT_SEARCH:
            LOCAL_SEARCH_TO_LOCAL[ii, 0, 3] = dx
            LOCAL_SEARCH_TO_LOCAL[ii, 1, 3] = dy
            LOCAL_SEARCH_TO_LOCAL[ii, 2, 3] = dz
            ii += 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOCAL_SEARCH_TO_LOCAL = LOCAL_SEARCH_TO_LOCAL.to(device)


class TorchPrecomputedSingleViewPointCloud(PointCloud):
    def __init__(self, cloud: open3d.geometry.PointCloud,
                 noise_cloud: open3d.geometry.PointCloud,
                 camera_pose,
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

        self.index_in_ref = self.processing_and_trace()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.camera_pos = camera_pose
        self.camera_pos_torch_inv = torch.tensor(np.linalg.inv(camera_pose), device=self.device).float()

        self.cloud_array = np.asarray(self.cloud.points).astype(np.float32)
        self.normals = np.zeros(self.cloud_array.shape, dtype=self.cloud_array.dtype)

        # self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        # self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

        self.pre_antipodal_score = None
        self.pre_search_score = None
        self.global_to_local = None
        self.point_frame_index = None

        self.frame_valid_bool = None
        self.local_to_global = None
        self.frame_objects_label = None
        self.frame_point_index = None
        self.frame_indices = None

        self.output = {}

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
        frame_num = self.global_to_local.shape[0]

        self.global_to_local = torch.tensor(self.global_to_local, dtype=torch.float, device=self.device)

        self.cloud_array = torch.tensor(self.cloud_array).float().to(self.device)
        self.local_to_global = torch.inverse(self.global_to_local).contiguous()
        self.frame_point_index = torch.tensor(self.frame_point_index, dtype=torch.long, device=self.device)

        self.pre_antipodal_score = torch.tensor(self.pre_antipodal_score, dtype=torch.float, device=self.device)
        self.pre_search_score = torch.tensor(self.pre_search_score, dtype=torch.float, device=self.device)
        self.frame_objects_label = torch.ones(frame_num, dtype=torch.int, device=self.device) * len(NAME_LIST)
        self.frame_valid_bool = torch.zeros(frame_num, dtype=torch.uint8, device=self.device)
        assert self.frame_objects_label.shape[0] == self.pre_search_score.shape[0] == self.pre_antipodal_score.shape[0]

    def _find_match(self, scene: TorchContactScenePointCloud):
        # TODO: hard code max frame here
        point_frame_index = np.ones([self.points_num, 10], dtype=np.int) * -1
        global_to_local = []
        frame_point_index = []
        kd_tree = scene.kd_tree
        possible_frame = 0
        normals = np.zeros([self.points_num, 3])

        antipodal_score = []
        search_score = []
        point_do_not_has_frame = 0

        for index in range(self.cloud_array.shape[0]):
            [k, idx, _] = kd_tree.search_hybrid_vector_3d(self.reference_cloud[self.index_in_ref[index], :],
                                                          radius=config.CURVATURE_RADIUS,
                                                          max_nn=1)
            if k < 1:
                normals[index, :] = [0, 0, 1]  # If neighbor not found, it must be table
                continue

            i = idx[0]
            normals[index, :] = scene.normals[i, :]
            neighbor_frame_index = np.nonzero(scene.frame_point_index == i)[0]
            num_of_frame = neighbor_frame_index.shape[0]
            if num_of_frame == 0:
                point_do_not_has_frame += 1
                continue
            global_to_local.append(scene.global_to_local[neighbor_frame_index])
            point_frame_index[index, 0:num_of_frame] = np.arange(possible_frame, possible_frame + num_of_frame)
            antipodal_score.append(scene.antipodal_score[neighbor_frame_index])
            search_score.append(scene.search_score[neighbor_frame_index])
            frame_point_index.append([index] * num_of_frame)
            possible_frame += num_of_frame

        print("Points do not have frames : {}/{}".format(point_do_not_has_frame, self.points_num))
        self.global_to_local = np.concatenate(global_to_local, axis=0)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = normals
        self.cloud.normals = open3d.utility.Vector3dVector(self.normals)
        self.cloud.orient_normals_towards_camera_location(camera_location=self.camera_pos[0:3, 3])
        self.pre_search_score = np.concatenate(search_score, axis=0)
        self.pre_antipodal_score = np.concatenate(antipodal_score, axis=0)
        self.point_frame_index = point_frame_index
        self.frame_point_index = np.concatenate(frame_point_index, axis=0).astype(np.int)

    def run_score(self, scene: TorchContactScenePointCloud):
        self._find_match(scene)
        # open3d.visualization.draw_geometries([self.cloud])
        self.to_torch()
        if self.global_to_local.shape[0] == 0:
            import ipdb
            ipdb.set_trace()

        for frame_index in trange(self.global_to_local.shape[0]):
            if self.finger_hand(frame_index, scene):
                self.frame_valid_bool[frame_index] = 1

        print('Frames that will not collide {}/{}'.format(self.frame_valid_bool.sum(), self.frame_valid_bool.shape[0]))

        log_search = torch.log(self.pre_search_score) / 6.5
        log_score = \
            torch.min(torch.stack([log_search, torch.ones(log_search.shape[0]).float().to(self.device)], dim=0), dim=0)[
                0]
        score_all = log_score * self.pre_antipodal_score

        point_score = torch.zeros(self.points_num, device=self.device, dtype=torch.float)
        point_frame_index = torch.zeros(self.points_num, device=self.device, dtype=torch.long)

        valid_frame_index = torch.nonzero(self.frame_valid_bool).long()

        for frame_index in valid_frame_index:
            point_index = self.frame_point_index[frame_index]
            if point_score[point_index] > score_all[frame_index]:
                continue
            else:
                point_score[point_index] = score_all[frame_index]
                point_frame_index[point_index] = frame_index

        valid_index = torch.nonzero(point_score > 0).squeeze(dim=1)
        final_point_frame_index = point_frame_index[valid_index]

        final_frame = self.local_to_global[final_point_frame_index]
        final_search_score = self.pre_search_score[final_point_frame_index].cpu().numpy()
        final_antipodal_score = self.pre_antipodal_score[final_point_frame_index].cpu().numpy()
        final_object_label = self.frame_objects_label[final_point_frame_index].cpu().numpy()
        valid_index = valid_index.cpu().numpy()

        cloud_array = torch.cat(
            [self.cloud_array, torch.ones([self.cloud_array.shape[0], 1], dtype=torch.float, device=self.device), ],
            dim=1)
        cloud_array = torch.matmul(self.camera_pos_torch_inv, cloud_array.transpose(0, 1)).cpu().numpy()
        final_frame = torch.matmul(self.camera_pos_torch_inv.unsqueeze(0), final_frame).cpu().numpy()

        self.output.update(
            {"search_score": final_search_score, 'antipodal_score': final_antipodal_score, 'valid_frame': final_frame,
             'valid_index': valid_index})
        self.output.update({"point_cloud": cloud_array[0:3, :], "objects_label": final_object_label})

    def dump(self):
        """
        Damp the data inside this object to a python dictionary
        May need to store color later
        :return: Python dictionary store useful data
        """
        return self.output

    def _table_collision_check(self, T_local_to_global):
        """
        Check whether the gripper collide with the table top with offset
        :param point: torch.tensor(3)
        :param frame: torch.tensor(3, 3)
        :return: a torch boolean tensor with shape (len(config.INDEX_TO_ARRAY))
        """

        T_local_search_to_global_all = torch.bmm(
            T_local_to_global.unsqueeze(0).expand(NUM_OF_LOCAL_SEARCH, 4, 4).contiguous(), LOCAL_SEARCH_TO_LOCAL)
        boundary_global = torch.bmm(T_local_search_to_global_all, config.TORCH_GRIPPER_BOUND.squeeze(0).expand(
            T_local_search_to_global_all.shape[0], -1, -1).contiguous())
        table_collision_bool_all = boundary_global[:, 2, :] < config.TABLE_HEIGHT + config.TABLE_COLLISION_OFFSET
        return table_collision_bool_all.any(dim=1, keepdim=False).any(dim=0, keepdim=False)

    def finger_hand(self, frame_index, scene: TorchContactScenePointCloud):
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
        T_global_to_local = self.global_to_local[frame_index, :, :]
        T_local_to_global = self.local_to_global[frame_index, :, :]
        if self._table_collision_check(T_local_to_global):
            return False
        local_cloud = torch.matmul(T_global_to_local, scene.cloud_array_homo)

        for dz in HEIGHT_SEARCH:
            z_bool = (local_cloud[2, :] < config.HALF_HAND_THICKNESS + dz) & (local_cloud[2,
                                                                              :] > -config.HALF_HAND_THICKNESS + dz)
            for dy in WIDTH_SEARCH:
                y_bool = (local_cloud[1, :] < config.HALF_BOTTOM_SPACE + dy) & (
                        local_cloud[1, :] > -config.HALF_BOTTOM_SPACE + dy)
                abs_y = torch.abs(local_cloud[1, :] + dy)
                y_collision_bool = (abs_y > config.HALF_BOTTOM_SPACE) & (abs_y < config.HALF_BOTTOM_WIDTH)

                for dx in LENGTH_SEARCH:
                    x_bool = (local_cloud[0] > -config.BOTTOM_LENGTH + dx) & (
                            local_cloud[0] < config.FINGER_LENGTH + dx)
                    collision_bool = z_bool & x_bool & y_collision_bool
                    if collision_bool.sum() > 0:
                        return False
                    close_region_bool = x_bool & z_bool & y_bool
                    if local_cloud[0, close_region_bool].min() < config.BACK_COLLISION_MARGIN:
                        return False
                    close_region_label = torch.unique(scene.label_array[close_region_bool],
                                                      sorted=False)
                    if close_region_label.shape[0] > 1:
                        return False

        self.frame_objects_label[frame_index] = close_region_label[0]
        return True
