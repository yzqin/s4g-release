import os
import pickle

import numpy as np
import open3d
from matplotlib import cm
from tqdm import trange

from configs import config
from configs.dataset_config import NAME_LIST
from configs.path import get_resource_dir_path

color_map = cm.get_cmap('jet', 256)
output_dir = get_resource_dir_path('refined_contact_object')

FRAME_PER_POINT = 16
MAX_NEIGHBOR_FRAME = 5
MIN_SEARCH_SCORE = 100

# WIDTH_SEARCH = [-0.005, 0.005, 0]
WIDTH_SEARCH = [0]
HEIGHT_SEARCH = [-0.01, 0.01, 0]
LENGTH_SEARCH = [-0.01, 0.01, 0]


def main():
    for object_name in NAME_LIST[105:]:
        # object_name = '063-a_marbles#2'
        print(object_name)

        data_dir = get_resource_dir_path('contact_single_object_data')
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
        homo_cloud = np.concatenate([cloud.T, np.ones([1, point_num])])

        # Min search reduce
        valid_index = np.nonzero(search_score > MIN_SEARCH_SCORE)[0]
        frame = frame[valid_index]
        search_score = search_score[valid_index]
        antipodal_score = antipodal_score[valid_index]
        frame_point_index = frame_point_index[valid_index].astype(np.int)
        frame_num = frame.shape[0]

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

        def check_single_collision(j):
            result = 9999
            local_cloud = frame[i, :, :] @ homo_cloud
            for dz in HEIGHT_SEARCH:
                z_bool = (local_cloud[2, :] < config.HALF_HAND_THICKNESS + dz) & (local_cloud[2,
                                                                                  :] > -config.HALF_HAND_THICKNESS + dz)
                for dy in WIDTH_SEARCH:
                    y_bool = (local_cloud[1, :] < config.HALF_BOTTOM_SPACE + dy) & (
                            local_cloud[1, :] > -config.HALF_BOTTOM_SPACE + dy)
                    abs_y = np.abs(local_cloud[1, :] + dy)
                    y_collision_bool = (abs_y > config.HALF_BOTTOM_SPACE) & (abs_y < config.HALF_BOTTOM_WIDTH)

                    for dx in LENGTH_SEARCH:
                        x_bool = (local_cloud[0] > -config.BOTTOM_LENGTH + dx) & (
                                local_cloud[0] < config.FINGER_LENGTH + dx)
                        collision_bool = z_bool & x_bool & y_collision_bool
                        if collision_bool.sum() > 0:
                            return None
                        close_region_bool = x_bool & z_bool & y_bool
                        close_region_num = close_region_bool.sum()
                        if close_region_num < MIN_SEARCH_SCORE:
                            return None
                        if local_cloud[0, close_region_bool].min() < 0:
                            return None
                        result = np.minimum(result, close_region_num)

            return result

        for i in trange(frame_num):
            search_result = check_single_collision(i)
            if search_result:
                frame_list.append(i)
                search_score_list.append(search_result)
                antipodal_score_list.append(antipodal_score[i])
                frame_point_index_list.append(frame_point_index[i])

        final_frame = frame[np.array(frame_list), :, :]
        final_index = np.array(frame_point_index_list)
        final_search_score = np.array(search_score_list)
        final_antipodal_score = np.array(antipodal_score_list)

        output_data.update(
            {"global_to_local": final_frame, 'frame_point_index': final_index, 'cloud': cloud, 'normal': normal,
             'search_score': final_search_score, 'antipodal_score': final_antipodal_score})

        print('After refinement: it has {} frames'.format(final_index.shape[0]))

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)

        break

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
