import numpy as np
import pickle
import open3d
from time import time
import os
from configs.dataset_config import NAME_LIST, NAME_TO_COLOR
from configs.path import get_resource_dir_path
from configs import config
import torch
from tqdm import trange

single_object_data_path = get_resource_dir_path('single_object_data')
ply_dir = get_resource_dir_path('bad_ply')
npy_dir = get_resource_dir_path('npy')


class GenerateDarbouxObjectData:
    def __init__(self):
        self.ply_dir = ply_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

    def dump(self, data, name):
        with open(os.path.join(single_object_data_path, '{}.p'.format(name)), 'wb') as f:
            pickle.dump(data, f)
        print('Dump to file of {} with keys: {}'.format(name, data.keys()))

    def run_loop(self, start, end=None):
        if end:
            todo = list(range(start, end))
        else:
            todo = list(range(start, len(NAME_LIST)))

        for _, name in enumerate(NAME_LIST[todo]):
            tic = time()
            data_dict = {}
            ply_path = os.path.join(self.ply_dir, "{}.ply".format(name))
            mesh = open3d.io.read_triangle_mesh(ply_path)
            mesh.compute_vertex_normals()
            pc = mesh.sample_points_uniformly(np.asarray(mesh.vertices).shape[0] * 10)
            pc = pc.voxel_down_sample(0.0025)

            color = NAME_TO_COLOR[name]
            pc.paint_uniform_color(color)
            pc.orient_normals_towards_camera_location(camera_location=np.mean(np.asarray(pc.points), axis=0))
            normals = np.asarray(pc.normals)
            normals /= -np.linalg.norm(normals, axis=1, keepdims=True)
            kd_tree = open3d.geometry.KDTreeFlann(pc)
            data_dict.update({'cloud': np.asarray(pc.points)})

            points = np.asarray(pc.points)
            frames, inv_frames = self._estimate_frame(points, normals, kd_tree)
            data_dict.update({'normal': normals, 'frame': frames, 'inv_frame': inv_frames})

            grasp_data = self._estimate_grasp_quality(points, frames, inv_frames, normals)
            data_dict.update(grasp_data)

            print("Finish {} with time: {}s".format(name, time() - tic))
            self.dump(data_dict, name)

    def _estimate_frame(self, points, normals, kd_tree):
        """
        Estimate the Darboux frame of single point
        In self.frame, each column of one point frame is a vec3, with the order of x, y, z axis
        Note there is a minus sign of the whole frame, which means that x is the negative direction of normal
        """

        frames = np.tile(np.eye(3), [points.shape[0], 1, 1])
        inv_frames = np.tile(np.eye(3), [points.shape[0], 1, 1])
        for i in range(frames.shape[0]):
            [k, idx, _] = kd_tree.search_radius_vector_3d(points[i, :], config.CURVATURE_RADIUS)
            normal = np.mean(normals[idx, :], axis=0)
            normal /= np.linalg.norm(normal)
            if k < 5:
                frames[i, :, :] = np.zeros([3, 3])
                inv_frames[i, :, :] = np.zeros([3, 3])
                continue

            M = np.eye(3) - normal.T @ normal
            xyz_centroid = np.mean(M @ normals[idx, :].T, axis=1, keepdims=True)
            normal_diff = normals[idx, :].T - xyz_centroid
            cov = normal_diff @ normal_diff.T
            eig_value, eig_vec = np.linalg.eigh(cov)

            minor_curvature = eig_vec[:, 0] - eig_vec[:, 0] @ normal.T * np.squeeze(normal)
            minor_curvature /= np.linalg.norm(minor_curvature)
            principal_curvature = np.cross(minor_curvature, np.squeeze(normal))

            frames[i, :, :] = np.stack([-normal, -principal_curvature, minor_curvature], axis=1)
            inv_frames[i, :, :] = np.stack([normal, principal_curvature, minor_curvature], axis=1)
        return frames, inv_frames

    def _estimate_grasp_quality(self, points: np.ndarray, frames: np.ndarray, inv_frames: np.ndarray,
                                normals: np.ndarray):
        torch_points = torch.tensor(points, device=self.device).float().transpose(1, 0)
        torch_points_homo = torch.cat([torch_points,
                                       torch.ones(1, torch_points.shape[1], dtype=torch.float, device=self.device)],
                                      dim=0)
        torch_normals = torch.tensor(normals, device=self.device).float().transpose(1, 0)

        torch_frames = torch.tensor(frames, device=self.device).float()
        torch_inv_frames = torch.tensor(inv_frames, device=self.device).float()
        frame_search_scores = torch.zeros([points.shape[0], len(config.LENGTH_SEARCH), config.GRASP_PER_LENGTH],
                                          dtype=torch.int, device=self.device)
        inv_frame_search_scores = frame_search_scores.new_zeros(size=frame_search_scores.shape)
        frame_antipodal_score = torch.zeros(frame_search_scores.shape, dtype=torch.float, device=self.device)
        inv_frame_antipodal_score = frame_antipodal_score.new_zeros(size=frame_antipodal_score.shape)

        for index in trange(torch_frames.shape[0]):
            search_scores, antipodal_score = self.finger_hand_view(index, torch_points_homo, torch_frames[index, :, :],
                                                                   torch_normals)

            inv_search_scores, inv_antipodal_score = self.finger_hand_view(index, torch_points_homo,
                                                                           torch_inv_frames[index, :, :],
                                                                           torch_normals)

            frame_search_scores[index, :, :] = search_scores
            inv_frame_search_scores[index, :, :] = inv_search_scores
            frame_antipodal_score[index, :, :] = antipodal_score
            inv_frame_antipodal_score[index, :, :] = inv_antipodal_score

        # All return variable should be numpy on the end of this function
        grasp_data = {}
        grasp_data.update({"search_score": frame_search_scores.cpu().numpy()})
        grasp_data.update({"inv_search_score": inv_frame_search_scores.cpu().numpy()})
        grasp_data.update({"antipodal_score": frame_antipodal_score.cpu().numpy()})
        grasp_data.update({"inv_antipodal_score": inv_frame_antipodal_score.cpu().numpy()})
        return grasp_data

    def finger_hand_view(self, index, point_homo: torch.Tensor, single_frame: torch.Tensor, normal: torch.Tensor):
        all_frame_search_score = torch.zeros([len(config.LENGTH_SEARCH), config.GRASP_PER_LENGTH], dtype=torch.float,
                                             device=self.device)
        all_frame_antipodal = torch.zeros(all_frame_search_score.shape, dtype=torch.float, device=self.device)

        if torch.mean(torch.abs(single_frame)) < 1e-6:
            return all_frame_search_score

        point = point_homo[0:3, index:index + 1]  # (3,1)
        T_global_to_local = torch.eye(4, device=self.device).float()
        T_global_to_local[0:3, 0:3] = single_frame
        T_global_to_local[0:3, 3:4] = point
        T_global_to_local = torch.inverse(T_global_to_local)

        local_cloud = torch.matmul(T_global_to_local, point_homo)
        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3], normal)

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
                local_search_close_plane_points = local_search_close_plane_points_all[i % config.GRASP_PER_LENGTH, :, :]

                back_collision_bool_xy = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                         (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                         (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN)

                y_finger_region_bool_left = (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                            (local_search_close_plane_points[1, :] > config.HALF_BOTTOM_SPACE)
                y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                             (local_search_close_plane_points[1, :] < -config.HALF_BOTTOM_SPACE)

                # Here we use the average of dz local search to compensate for the error in end-effector
                temp_search, temp_antipodal = torch.zeros(1, dtype=torch.float, device=self.device)
                close_region_point_num, single_antipodal = torch.zeros(1, dtype=torch.float, device=self.device)
                for dz_num, dz in enumerate([-0.02, 0.02, 0]):

                    z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS + dz) & \
                                       (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS + dz)

                    back_collision_bool = back_collision_bool_xy & z_collision_bool

                    if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
                        continue

                    y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
                    collision_region_bool = (z_collision_bool & y_finger_region_bool)
                    if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
                        continue

                    close_region_bool = z_collision_bool & \
                                        (local_search_close_plane_points[1, :] < config.HALF_BOTTOM_SPACE) & \
                                        (local_search_close_plane_points[1, :] > -config.HALF_BOTTOM_SPACE)

                    close_region_point_num = torch.sum(close_region_bool, dtype=torch.float)
                    if close_region_point_num < config.CLOSE_REGION_MIN_POINTS:
                        continue

                    xyz_plane_normals = local_cloud_normal[:, close_plane_bool][:, close_region_bool]  # (3,n)
                    close_region_cloud_normal = torch.matmul(
                        T_local_to_local_search_all[i % config.GRASP_PER_LENGTH, 0:3, 0:3],
                        xyz_plane_normals)

                    close_region_cloud = local_search_close_plane_points[:, close_region_bool]

                    single_antipodal = self._antipodal_score(close_region_cloud, close_region_cloud_normal)
                    temp_antipodal += single_antipodal / 3
                    temp_search += close_region_point_num / 3

                all_frame_search_score[dl_num, i % config.GRASP_PER_LENGTH] = torch.min(temp_search,
                                                                                        close_region_point_num)

                all_frame_antipodal[dl_num, i % config.GRASP_PER_LENGTH] += torch.min(temp_antipodal, single_antipodal)
                i += 1

        return all_frame_search_score, all_frame_antipodal

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
        left_normal_theta = torch.abs(
            torch.matmul(self.left_normal, close_region_cloud_normal[:, left_region_bool]))
        right_normal_theta = torch.abs(
            torch.matmul(self.right_normal, close_region_cloud_normal[:, right_region_bool]))

        geometry_average_theta = torch.mean(left_normal_theta) * torch.mean(right_normal_theta)
        return geometry_average_theta
