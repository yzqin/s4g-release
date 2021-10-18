import copy
import os
import time

import numpy as np
import open3d
import transforms3d

from configs.dataset_config import NAME_LIST, NAME_TO_COLOR, TABLE_COLOR
from configs.path import get_resource_dir_path
from matplotlib import cm
color_map = cm.get_cmap('jet',256)

dirname = os.path.dirname(__file__)
ply_dir = get_resource_dir_path('ply')
npy_dir = get_resource_dir_path('npy')
table_path = os.path.abspath(os.path.join(dirname, '../assets/table.ply'))
scene_path = get_resource_dir_path('scene')


class GenerateSceneServer:
    """
    Pre load all ply file and generate multiple scene without reloading
    """

    def __init__(self):
        self.ply_dir = ply_dir
        self.cloud_dict = {}
        self.load_ply()

        # Load table
        self.table = open3d.io.read_point_cloud(table_path)
        self.table.estimate_normals(open3d.geometry.KDTreeSearchParamKNN(knn=30))
        self.table.paint_uniform_color(TABLE_COLOR)

    def smooth_normal(self, normals, points, kd_tree):
        smooth_normals = np.copy(normals)
        for i in range(points.shape[0]):
            single_normal = normals[i, :]
            [k, idx, _] = kd_tree.search_radius_vector_3d(points[i], radius=0.012)
            if k < 3:
                continue
            neighbors_distance = points[idx, :] - points[i:i + 1, :]
            normal_distance = neighbors_distance @ single_normal.T
            plane_neighbor_bool = normal_distance < 0.001
            average_normal = np.mean(normals[idx, :][plane_neighbor_bool], axis=0)
            smooth_normals[i, :] = average_normal
        smooth_normals /= np.linalg.norm(smooth_normals, axis=1, keepdims=True)
        return smooth_normals

    def load_ply(self):
        for i, name in enumerate(NAME_LIST):
            ply_path = os.path.join(self.ply_dir, "{}.ply".format(name))
            pc = open3d.io.read_point_cloud(ply_path)
            pc = pc.voxel_down_sample(0.002)
            num = int(name[0:3]) * 3
            color = color_map(num)[0:3]
            pc.paint_uniform_color(color)

            points = np.asarray(pc.points)
            normals = np.asarray(pc.normals)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            kd_tree = open3d.geometry.KDTreeFlann(pc)
            # normals = self.smooth_normal(normals, points, kd_tree)
            pc.normals = open3d.utility.Vector3dVector(normals)
            self.cloud_dict.update({name: pc})

    def generate_single_scene(self, npy_path):
        obj_dict = np.load(npy_path, allow_pickle=True)[()]
        name = os.path.splitext(os.path.basename(npy_path))[0]
        cloud_list = [np.asarray(self.table.points)]
        normal_list = [np.asarray(self.table.normals)]
        color_list = [np.asarray(self.table.colors)]
        for obj, pose in obj_dict.items():
            if pose[0] > 0.35 or pose[1] > 0.35:
                continue
            obj_name = obj.split('@')[0]
            pc = copy.deepcopy(self.cloud_dict[obj_name])
            rotation = transforms3d.quaternions.quat2mat(pose[3:7])
            mat = np.eye(4)
            mat[0:3, 0:3] = rotation
            mat[0:3, 3] = pose[0:3]
            pc.transform(mat)
            cloud_list.append(np.asarray(pc.points))
            normal_list.append(np.asarray(pc.normals))
            color_list.append(np.asarray(pc.colors))

        scene_cloud = open3d.geometry.PointCloud()
        scene_cloud.points = open3d.utility.Vector3dVector(np.vstack(cloud_list))
        scene_cloud.normals = open3d.utility.Vector3dVector(np.vstack(normal_list))
        scene_cloud.colors = open3d.utility.Vector3dVector(np.vstack(color_list))
        open3d.io.write_point_cloud(os.path.join(scene_path, '{}.ply'.format(name)), scene_cloud)

    def run_loop(self, start, end):
        num_of_scene = end - start
        missing_data = []
        for i in range(start, end):
            current_time = time.time()
            path = os.path.join(npy_dir, '{}.npy'.format(i))
            if not os.path.exists(path):
                missing_data.append(i)
                continue
            self.generate_single_scene(path)
            print('Generate {}/{} scene with {} seconds'.format(i + 1, num_of_scene, time.time() - current_time))
        print('Missing data : \n {}'.format(missing_data))
        print('Missing percentage: {} / {}'.format(len(missing_data), num_of_scene))
