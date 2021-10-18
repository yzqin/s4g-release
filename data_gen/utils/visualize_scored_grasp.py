import numpy as np
import open3d
import argparse
from utils.transformation_utils import global_to_local_transformation
from utils.visualization_utils import get_hand_geometry
from utils.path_utils import get_scene_and_training_data_path
import seaborn as sns
import matplotlib.pyplot as plt
from configs import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_index",
        "-s",
        type=int
    )
    parser.add_argument(
        "--view_index",
        "-v",
        type=int
    )

    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=5
    )

    args = parser.parse_args()
    return args


def score_function(search_score, antipodal_score, heuristic_score, object_num):
    object_num = object_num + (object_num < 0.5).astype(np.float) * 10000
    scored_grasp = np.minimum(np.log(search_score + 1) / 4, np.ones([1, 1])) * antipodal_score / np.power(object_num, 2)
    score = np.zeros((scored_grasp.shape[0], len(config.LENGTH_SEARCH)))
    for i in range(len(config.LENGTH_SEARCH)):
        score[:, i] = np.mean(scored_grasp[:, i * config.GRASP_PER_LENGTH:(i + 1) * config.GRASP_PER_LENGTH], axis=1)
    return score


def vis_scored_grasp(scene_path, training_data_path, topk, score_function):
    data = np.load(training_data_path, allow_pickle=True)
    scene_point_score = score_function(data["search_score"], data["antipodal_score"],
                                       data["heuristic_score"], data["objects_num"])
    # scene_point_score = (scene_point_score - np.min(scene_point_score)) / (
    #         np.max(scene_point_score) - np.min(scene_point_score) + 1e-4)
    flat_score = scene_point_score.flatten()
    sns.distplot(flat_score[flat_score > 0], kde=False)
    plt.show()

    topk_inds = np.argsort(-flat_score)[:topk]
    frame_inds = list(data["frame_index"])

    view_point_cloud = open3d.PointCloud()
    view_point_cloud.points = open3d.Vector3dVector(data["point_cloud"].T)
    color = np.zeros(data["point_cloud"].T.shape)
    scene_point_cloud = open3d.io.read_point_cloud(scene_path)
    color[:, 0] = np.mean(scene_point_score, axis=1)
    color[:, 1] = 0
    color[:, 2] = 1.0 - np.mean(scene_point_score, axis=1)
    view_point_cloud.colors = open3d.Vector3dVector(color)

    vis_list = [view_point_cloud, scene_point_cloud]
    for ind in topk_inds:
        ind = ind // len(config.LENGTH_SEARCH)
        length_ind = ind % len(config.LENGTH_SEARCH)
        frame_ind = frame_inds.index(ind)
        frame = data["frame"][frame_ind]
        point = data["point_cloud"][:, ind]
        T_global_to_local = global_to_local_transformation(frame, point)
        hand = get_hand_geometry(T_global_to_local)
        vis_list.extend(hand)
        print("Point {} score: {}".format(ind, scene_point_score[ind]))

    open3d.draw_geometries(vis_list)

    while True:
        vis = open3d.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(view_point_cloud)
        vis.run_loop()
        vis.destroy_window()
        pick_inds = vis.get_picked_points()
        vis_list = [view_point_cloud, scene_point_cloud]
        for ind in pick_inds:
            if ind in frame_inds:
                print(scene_point_score[ind])
                frame_ind = frame_inds.index(ind)
                frame = data["frame"][frame_ind]
                point = data["point_cloud"][:, ind]
                T_global_to_local = global_to_local_transformation(frame, point)
                hand = get_hand_geometry(T_global_to_local)
                vis_list.extend(hand)
            else:
                print("{} is not valid.".format(ind))
        open3d.draw_geometries(vis_list)


def main():
    args = parse_args()
    # scene_path, training_data_path = get_scene_and_training_data_path(args.scene_index, args.view_index)
    scene_path, training_data_path = get_scene_and_training_data_path(10, 0)
    vis_scored_grasp(scene_path, training_data_path, args.topk, score_function)


if __name__ == "__main__":
    main()
