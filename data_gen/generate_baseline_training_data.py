import open3d
import os
from pcd_classes.torch_precomputed_single_view_point_cloud import TorchDataScenePointCloud, \
    TorchPrecomputedSingleViewPointCloud
from configs import config, path
from utils.transformation_utils import local_to_global_transformation_quat
import pickle
from time import time
import numpy as np

camera_pose_quat = [local_to_global_transformation_quat(pose[3:7], pose[0: 3]) for pose in config.CAMERA_POSE]


def generate_single_scene(scene_name, num_of_view=4, visualize=False):
    current = time()

    if type(scene_name) == int:
        scene_name = str(scene_name)
    assert type(scene_name) == str

    scene_path, single_view_list, noise_single_view_list = path.get_data_scene_and_view_path(scene_name,
                                                                                             num_of_view=num_of_view,
                                                                                             noise=True)
    if not (os.path.exists(scene_path) and os.path.exists(single_view_list[0])):
        print("Scene name or view name do not exist for {}".format(scene_name))
        print(scene_path)
        return

    scene_cloud = TorchDataScenePointCloud(scene_path)

    for i in range(num_of_view):
        if not os.path.exists(single_view_list[i]):
            continue
        view_cloud = TorchPrecomputedSingleViewPointCloud(open3d.io.read_point_cloud(single_view_list[i]),
                                                          open3d.io.read_point_cloud(noise_single_view_list[i]),
                                                          camera_pose=camera_pose_quat[i])
        view_cloud = TorchBaseLineSingleViewPointCloud(
            open3d.io.read_point_cloud(single_view_list[i]),
            camera_pose=camera_pose_quat[i][0:3, 3])
        if view_cloud.run_score(scene_cloud):
            if visualize:
                visualizer.add_single_view(view_cloud)
                visualizer.visualize_view_score(50)

            output_path = os.path.join(path.TRAINING_DATA_PATH + "_baseline",
                                       "baseline_{}_view_{}.p".format(scene_name, i))
            if not os.path.exists(path.TRAINING_DATA_PATH + "_baseline"):
                os.mkdir(path.TRAINING_DATA_PATH + "_baseline")
            with open(output_path, 'wb') as file:
                pickle.dump(view_cloud.dump(), file)
                print('Save data in {}'.format(output_path))

    print("It takes {} to finish scene {}".format(time() - current, scene_name))


def generate(start: int, end: int, process=8):
    if process == 0:
        for i in range(start, end):
            generate_single_scene(i)
    else:
        assert False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)
    parser.add_argument('--end', '-e', type=int)
    parser.add_argument('--process', '-p', type=int, default=0)
    args = parser.parse_args()

    current_time = time()
    if not os.path.exists(path.TRAINING_DATA_PATH):
        os.mkdir(path.TRAINING_DATA_PATH)
    generate(int(args.start), int(args.end), int(args.process))
    # generate(0, 8, 0)
    print("Finished with {}s".format(-current_time + time()))
