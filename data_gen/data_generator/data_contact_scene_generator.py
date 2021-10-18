import os
import pickle
from time import time

import numpy as np
import transforms3d
from tqdm import trange

from configs.dataset_config import NAME_LIST, NAME_TO_COLOR, NAME_TO_INDEX
from configs.path import get_resource_dir_path
from pcd_classes.torch_contact_scene_point_cloud import TorchContactScenePointCloud

data_scene_path = get_resource_dir_path('contact_data_scene')
single_object_data_path = get_resource_dir_path('reduced_contact')
dirname = os.path.dirname(__file__)
ply_dir = get_resource_dir_path('ply')
npy_dir = get_resource_dir_path('npy')
table_path = os.path.abspath(os.path.join(dirname, '../assets/table.ply'))


class GenerateContactScene:
    def __init__(self):
        self.cloud = {}
        self.global_to_local = {}  # (n,4,4)
        self.search_score = {}
        self.antipodal_score = {}
        self.normal = {}
        self.frame_point_index = {}
        self.load_data()

    def load_data(self):
        for name in NAME_LIST:
            data_path = os.path.join(single_object_data_path, '{}.p'.format(name))
            data = np.load(data_path, allow_pickle=True)
            self.cloud.update({name: data['cloud']})
            self.normal.update({name: data['normal']})
            self.search_score.update({name: data['search_score']})
            self.antipodal_score.update({name: data['antipodal_score']})
            self.global_to_local.update({name: data['global_to_local']})
            self.frame_point_index.update({name: data['frame_point_index']})

    def generate_single_scene(self, npy_path=None, dump_to_file=False):
        filename = os.path.splitext(os.path.basename(npy_path))[0]
        tic = time()
        pose_dict = np.load(npy_path, allow_pickle=True)[()]
        final_cloud = []
        final_global_to_local = []
        final_label = []
        final_color = []
        final_normal = []
        final_search_score = []
        final_antipodal_score = []
        final_frame_point_index = []

        frame_point_index_start_num = 0

        for name, pose in pose_dict.items():
            rotation = transforms3d.quaternions.quat2mat(pose[3:7])
            translation = pose[0:3]
            mat = np.eye(4)
            mat[0:3, 0:3] = rotation
            mat[0:3, 3] = translation
            pc = self.cloud[name]
            pc_homo = np.concatenate([pc, np.ones([pc.shape[0], 1])], axis=1).T  # (4,n)
            pc_after_move = np.dot(mat, pc_homo)
            final_cloud.append(pc_after_move[0:3, :].T)

            normal = self.normal[name]
            normal_after_move = np.dot(rotation, normal.T).T
            final_normal.append(normal_after_move)

            label = NAME_TO_INDEX[name]
            label_array = np.ones(pc_homo.shape[1]) * label
            final_label.append(label_array)

            global_to_local = self.global_to_local[name]
            global_to_local_after_move = np.matmul(global_to_local, np.linalg.inv(mat))
            final_global_to_local.append(global_to_local_after_move)
            frame_point_index = self.frame_point_index[name] + frame_point_index_start_num
            final_frame_point_index.append(frame_point_index)
            frame_point_index_start_num += pc_homo.shape[1]

            color = NAME_TO_COLOR[name]
            color_array = np.tile(color, [pc_homo.shape[1], 1])
            final_color.append(color_array)
            final_search_score.append(self.search_score[name])
            final_antipodal_score.append(self.antipodal_score[name])

        clouds = np.concatenate(final_cloud, axis=0)
        colors = np.concatenate(final_color, axis=0)
        labels = np.concatenate(final_label, axis=0)
        normals = np.concatenate(final_normal, axis=0)
        search_scores = np.concatenate(final_search_score, axis=0)
        antipodal_scores = np.concatenate(final_antipodal_score, axis=0)
        global_to_locals = np.concatenate(final_global_to_local)
        frame_point_indices = np.concatenate(final_frame_point_index)

        data = {'cloud': clouds, 'global_to_local': global_to_locals, 'label': labels, 'color': colors,
                'normal': normals, 'frame_point_index': frame_point_indices,
                'search_score': search_scores, 'antipodal_score': antipodal_scores}

        if dump_to_file:
            file_path = os.path.join(data_scene_path, "{}.p".format(filename))
            print("Finish data scene {} with time {}s".format(filename, time() - tic))
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return file_path
        else:
            scene = TorchContactScenePointCloud(data=data)
            return scene

    def run_loop(self, start, end):
        num_of_scene = end - start
        missing_data = []
        for i in trange(start, end):
            path = os.path.join(npy_dir, '{}.npy'.format(i))
            if not os.path.exists(path):
                missing_data.append(i)
                continue
            self.generate_single_scene(path, dump_to_file=True)
        print('Missing data : \n {}'.format(missing_data))
        print('Missing percentage: {} / {}'.format(len(missing_data), num_of_scene))
