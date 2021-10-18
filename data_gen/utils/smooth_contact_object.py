import os
import pickle

import numpy as np
import open3d
from matplotlib import cm

from configs.dataset_config import NAME_LIST
from configs.path import get_resource_dir_path

color_map = cm.get_cmap('jet', 256)
output_dir = get_resource_dir_path('reduced_contact')

FRAME_PER_POINT = 5
MAX_NEIGHBOR_FRAME = 4
MIN_SEARCH_SCORE = 50

def main():
    for object_name in NAME_LIST:
        # object_name = '029_plate#1'
        print(object_name)

        data_dir = get_resource_dir_path('refined_contact_object')
        data_path = os.path.join(data_dir, '{}.p'.format(object_name))

        data = np.load(data_path, allow_pickle=True)
        cloud = data['cloud']
        frame = data['global_to_local']
        search_score = data['search_score']
        antipodal_score = data['antipodal_score']
        normal = data['normal']
        frame_point_index = data['frame_point_index']
        print("{} has {} frames".format(object_name, frame_point_index.shape[0]))
        point_num = cloud.shape[0]

        # Min search reduce
        valid_index = np.nonzero(search_score > MIN_SEARCH_SCORE)[0]
        frame = frame[valid_index]
        search_score = search_score[valid_index]
        antipodal_score = antipodal_score[valid_index]
        frame_point_index = frame_point_index[valid_index]

        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(cloud)
        pc.normals = open3d.utility.Vector3dVector(normal)
        kd_tree = open3d.geometry.KDTreeFlann(pc)

        point_score = np.minimum(np.log(search_score + 1) / 4, np.ones(1)) * antipodal_score

        point_score = (point_score - np.min(point_score)) / (np.max(point_score) - np.min(point_score))
        assert len(point_score.shape) == 1

        output_data = {}
        output_path = os.path.join(output_dir, "{}.p".format(object_name))
        frame_list = []
        frame_point_index_list = []
        search_score_list = []
        antipodal_score_list = []
        point_frame_num = np.zeros(point_num, dtype=np.int)

        for i in range(point_num):
            point_frame_index = np.nonzero((frame_point_index == i))[0]
            if len(point_frame_index) > FRAME_PER_POINT:
                new_point_index = point_frame_index[0:FRAME_PER_POINT - point_frame_num[i]]
                frame_list.append(frame[new_point_index, :, :])
                search_score_list.append(search_score[new_point_index])
                antipodal_score_list.append(antipodal_score[new_point_index])
                frame_point_index_list.append([i] * new_point_index.shape[0])
                point_frame_num[i] += new_point_index.shape[0]
                rest_index = point_frame_index[FRAME_PER_POINT:]
                if rest_index.shape[0] > 5:
                    [k, idx, _] = kd_tree.search_hybrid_vector_3d(cloud[i, :], radius=0.01, max_nn=5)
                    for nn_num in range(len(idx)):
                        neighbor = idx[nn_num]
                        if (neighbor < i and point_frame_num[neighbor] < FRAME_PER_POINT) or (
                                neighbor > i and point_frame_num[neighbor] < MAX_NEIGHBOR_FRAME):
                            point_frame_num[neighbor] += 1
                            search_score_list.append(search_score[rest_index[nn_num:nn_num + 1]])
                            antipodal_score_list.append(antipodal_score[rest_index[nn_num:nn_num + 1]])
                            frame_list.append(frame[rest_index[nn_num:nn_num + 1], :, :])
                            frame_point_index_list.append([neighbor])

            elif len(point_frame_index) > 0:
                added_frame_num = np.min([FRAME_PER_POINT - point_frame_num[i], point_frame_index.shape[0]])
                frame_list.append(frame[point_frame_index[:added_frame_num]])
                search_score_list.append(search_score[point_frame_index[:added_frame_num]])
                antipodal_score_list.append(antipodal_score[point_frame_index[:added_frame_num]])
                frame_point_index_list.append([i] * added_frame_num)
                point_frame_num[i] += added_frame_num

        final_frame = np.concatenate(frame_list, axis=0)
        final_index = np.concatenate(frame_point_index_list, axis=0)
        final_search_score = np.concatenate(search_score_list, axis=0)
        final_antipodal_score = np.concatenate(antipodal_score_list,axis=0)
        output_data.update(
            {"global_to_local": final_frame, 'frame_point_index': final_index, 'cloud': cloud, 'normal': normal,
             'search_score': final_search_score, 'antipodal_score': final_antipodal_score})

        print('After filtering: it has {} frames'.format(final_index.shape[0]))

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)

        # score = np.zeros([cloud.shape[0]])
        # point_frame = np.zeros([cloud.shape[0], 4, 4])
        # color = np.zeros([cloud.shape[0], 3])
        # point_frame_index = np.ones(cloud.shape[0], dtype=np.int) * -1
        # frame_point_index = frame_point_index.astype(np.int)
        # for i in range(frame.shape[0]):
        #     i = int(i)
        #     if score[frame_point_index[i]] < point_score[i]:
        #         score[frame_point_index[i]] = point_score[i]
        #         point_frame[frame_point_index[i]] = frame[i]
        #         point_frame_index[frame_point_index[i]] = i
        #     else:
        #         continue
        #
        # for i in range(cloud.shape[0]):
        #     color[i, :] = color_map(score[i])[0:3]
        #
        # pc.colors = open3d.utility.Vector3dVector(color)
        # open3d.visualization.draw_geometries([pc])

        # while True:
        #     from utils.visualization_utils import get_hand_geometry
        #     vis = open3d.visualization.VisualizerWithEditing()
        #     vis.create_window()
        #     vis.add_geometry(pc)
        #     vis.run()
        #     vis.destroy_window()
        #     pick_inds = vis.get_picked_points()
        #     vis_list = [pc]
        #     for ind in pick_inds:
        #         if ind in frame_point_index:
        #             print(antipodal_score[point_frame_index[ind]])
        #             print(search_score[point_frame_index[ind]])
        #             hand = get_hand_geometry(point_frame[ind])
        #             ball = open3d.geometry.TriangleMesh.create_sphere(0.0015)
        #             ball.translate(cloud[ind, :])
        #             vis_list.extend(hand)
        #             vis_list.append(ball)
        #     open3d.visualization.draw_geometries(vis_list)
        #


if __name__ == '__main__':
    main()
    # test2()
