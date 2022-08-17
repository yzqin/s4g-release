import os
import pickle

import numpy as np
import open3d
from matplotlib import cm

from configs import config

color_map = cm.get_cmap('jet', 256)

FRAME_PER_POINT = 1
MAX_NEIGHBOR_FRAME = 1
MIN_SEARCH_SCORE = 40

WIDTH_SEARCH = [0]
HEIGHT_SEARCH = [0]
LENGTH_SEARCH = [0]


def inverse_batch_pose(poses):
    result = np.zeros_like(poses)
    result[:, :3, :3] = np.transpose(poses[:, :3, :3], [0, 2, 1])
    result[:, 3, 3] = 1
    result[:, :3, 3:4] = - np.matmul(result[:, :3, :3], poses[:, :3, 3:4])
    return result


# Should tune: coke

def main():
    object_names = ["camera"]
    data_dir = "../objects/single_object_grasp"
    output_dir = "../objects/processed_single_object_grasp"
    for object_name in object_names:
        os.makedirs(output_dir, exist_ok=True)
        data_path = os.path.join(data_dir, '{}.pkl'.format(object_name))

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
        frame_point_index = frame_point_index[valid_index]

        # Build point cloud data structure
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(cloud)
        pc.normals = open3d.utility.Vector3dVector(normal)
        kd_tree = open3d.geometry.KDTreeFlann(pc)

        # Compute normalized point score
        point_score = np.minimum(np.log(search_score + 1) / 3, np.ones(1)) * antipodal_score
        point_score = (point_score - np.min(point_score)) / (np.max(point_score) - np.min(point_score))
        assert len(point_score.shape) == 1

        output_data = {}
        output_path = os.path.join(output_dir, "{}.pkl".format(object_name))
        frame_list = []
        frame_point_index_list = []
        point_frame_num = np.zeros(point_num, dtype=int)

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
                        if collision_bool.sum() > 3:
                            return None
                        close_region_bool = x_bool & z_bool & y_bool
                        close_region_num = close_region_bool.sum()
                        if close_region_num < MIN_SEARCH_SCORE:
                            return None
                        if local_cloud[0, close_region_bool].min() < 0:
                            return None
                        result = np.minimum(result, close_region_num)

            return result

        for i in range(point_num):
            if not check_single_collision(i):
                continue

            point_frame_index = np.nonzero((frame_point_index == i))[0]
            if len(point_frame_index) > FRAME_PER_POINT:
                new_point_index = point_frame_index[0:FRAME_PER_POINT - point_frame_num[i]]
                frame_list.append(frame[new_point_index, :, :])
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
                            frame_list.append(frame[rest_index[nn_num:nn_num + 1], :, :])
                            frame_point_index_list.append([neighbor])

            elif len(point_frame_index) > 0:
                added_frame_num = np.min([FRAME_PER_POINT - point_frame_num[i], point_frame_index.shape[0]])
                frame_list.append(frame[point_frame_index[:added_frame_num]])
                frame_point_index_list.append([i] * added_frame_num)
                point_frame_num[i] += added_frame_num

        final_frame = np.concatenate(frame_list, axis=0)
        final_index = np.concatenate(frame_point_index_list, axis=0)
        output_data.update(
            {"grasp_pose": inverse_batch_pose(final_frame), 'grasp_point_index': final_index, 'cloud': cloud,
             'normal': normal})

        print('After filtering: it has {} frames'.format(final_index.shape[0]))

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)


if __name__ == '__main__':
    main()
