import numpy as np
import pickle
from time import time
import os
import transforms3d
from configs.dataset_config import NAME_LIST, NAME_TO_COLOR, NAME_TO_INDEX
from configs.path import get_resource_dir_path
from tqdm import trange

data_scene_path = get_resource_dir_path('data_scene')
single_object_data_path = get_resource_dir_path('single_object_data')
dirname = os.path.dirname(__file__)
ply_dir = get_resource_dir_path('ply')
npy_dir = get_resource_dir_path('npy')
table_path = os.path.abspath(os.path.join(dirname, '../assets/table.ply'))


class GenerateDarbouxScene:
    def __init__(self):
        self.cloud = {}
        self.frame = {}
        self.normal = {}
        self.inv_frame = {}
        self.search_score = {}
        self.inv_search_score = {}
        self.antipodal_score = {}
        self.inv_antipodal_score = {}
        self.load_data()

    def load_data(self):
        for name in NAME_LIST:
            data_path = os.path.join(single_object_data_path, '{}.p'.format(name))
            data = np.load(data_path, allow_pickle=True)
            self.cloud.update({name: data['cloud']})
            self.frame.update({name: data['frame']})
            self.normal.update({name: data['normal']})
            self.inv_frame.update({name: data['inv_frame']})
            self.search_score.update({name: data['search_score']})
            self.inv_search_score.update({name: data['inv_search_score']})
            self.antipodal_score.update({name: data['antipodal_score']})
            self.inv_antipodal_score.update({name: data['inv_antipodal_score']})

    def generate_single_scene(self, npy_path):
        filename = os.path.splitext(os.path.basename(npy_path))[0]
        tic = time()
        pose_dict = np.load(npy_path, allow_pickle=True)[()]
        final_cloud = []
        final_frame = []
        final_inv_frame = []
        final_label = []
        final_color = []
        final_normal = []

        final_search_score = []
        final_inv_search_score = []
        final_antipodal_score = []
        final_inv_antipodal_score = []

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

            frame = self.frame[name]
            inv_frame = self.inv_frame[name]
            frame_after_move = np.matmul(rotation, frame)
            final_frame.append(frame_after_move)
            inv_frame_after_move = np.matmul(rotation, inv_frame)
            final_inv_frame.append(inv_frame_after_move)

            color = NAME_TO_COLOR[name]
            color_array = np.tile(color, [pc_homo.shape[1], 1])
            final_color.append(color_array)

            final_search_score.append(self.search_score[name])
            final_inv_search_score.append(self.inv_search_score[name])
            final_antipodal_score.append(self.antipodal_score[name])
            final_inv_antipodal_score.append(self.inv_antipodal_score[name])

        clouds = np.concatenate(final_cloud, axis=0)
        frames = np.concatenate(final_frame, axis=0)
        inv_frames = np.concatenate(final_inv_frame, axis=0)
        colors = np.concatenate(final_color, axis=0)
        labels = np.concatenate(final_label, axis=0)
        normals = np.concatenate(final_normal, axis=0)
        search_scores = np.concatenate(final_search_score, axis=0)
        antipodal_scores = np.concatenate(final_antipodal_score, axis=0)
        inv_search_scores = np.concatenate(final_inv_search_score, axis=0)
        inv_antipodal_scores = np.concatenate(final_inv_antipodal_score, axis=0)

        data = {'cloud': clouds, 'frame': frames, 'label': labels, 'color': colors, 'normal': normals,
                'inv_frame': inv_frames, 'search_score': search_scores, 'antipodal_score': antipodal_scores,
                'inv_search_score': inv_search_scores, 'inv_antipodal_score': inv_antipodal_scores}

        file_path = os.path.join(data_scene_path, "{}.p".format(filename))
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print("Finish data {} with time {}s".format(filename, time() - tic))

    def run_loop(self, start, end):
        num_of_scene = end - start
        missing_data = []
        for i in trange(start, end):
            path = os.path.join(npy_dir, '{}.npy'.format(i))
            if not os.path.exists(path):
                missing_data.append(i)
                continue
            self.generate_single_scene(path)
        print('Missing data : \n {}'.format(missing_data))
        print('Missing percentage: {} / {}'.format(len(missing_data), num_of_scene))
