import os
import pickle
from time import time

import open3d

from configs import config, path
from pcd_classes.torch_contact_single_view_point_cloud import TorchContactScenePointCloud, \
    TorchPrecomputedSingleViewPointCloud
from utils.transformation_utils import local_to_global_transformation_quat

camera_pose_quat = [local_to_global_transformation_quat(pose[3:7], pose[0: 3]) for pose in config.CAMERA_POSE]

output_dir = path.get_resource_dir_path('contact_training_data')
npy_dir = path.get_resource_dir_path('npy')


def generate_single_scene(scene_name, mode, geneartor=None, num_of_view=4, visualize=False):
    current = time()

    if type(scene_name) == int:
        scene_name = str(scene_name)
    assert type(scene_name) == str

    scene_path, single_view_list, noise_single_view_list = path.get_contact_data_scene_and_view_path(scene_name,
                                                                                             num_of_view=num_of_view,
                                                                                             noise=True)

    if mode == 'file':
        if not (os.path.exists(scene_path) and os.path.exists(single_view_list[0])):
            print("Scene name or view name do not exist for {}".format(scene_name))
            print(scene_path)
            return
        import numpy as np
        data = np.load(scene_path, allow_pickle=True)
        scene_cloud = TorchContactScenePointCloud(data)
    elif mode == 'online':
        if not (os.path.exists(single_view_list[0])):
            print("View name do not exist for {}".format(scene_name))
            print(single_view_list[0])
            return
        scene_cloud = geneartor.generate_single_scene(npy_path=os.path.join(npy_dir, "{}.npy".format(scene_name)))

    for i in range(num_of_view):
        if not os.path.exists(single_view_list[i]):
            continue
        view_cloud = TorchPrecomputedSingleViewPointCloud(open3d.io.read_point_cloud(single_view_list[i]),
                                                          open3d.io.read_point_cloud(noise_single_view_list[i]),
                                                          camera_pose=camera_pose_quat[i])

        view_cloud.run_score(scene_cloud)

        output_path = os.path.join(output_dir, "{}_view_{}.p".format(scene_name, i))
        with open(output_path, 'wb') as file:
            pickle.dump(view_cloud.dump(), file)
            print('Save data in {}'.format(output_path))

    print("It takes {} to finish scene {}".format(time() - current, scene_name))


def generate(start: int, end: int, mode: str):
    if mode == 'file':
        for i in range(start, end):
            generate_single_scene(i, mode, None)
    elif mode == 'online':
        from data_generator.data_contact_scene_generator import GenerateContactScene
        generator = GenerateContactScene()
        for i in range(start, end):
            generate_single_scene(i, mode, generator)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)
    parser.add_argument('--end', '-e', type=int)
    parser.add_argument('--mode', '-m', type=str, choices=['file', 'online'], default='online')
    args = parser.parse_args()

    current_time = time()
    if not os.path.exists(path.TRAINING_DATA_PATH):
        os.mkdir(path.TRAINING_DATA_PATH)
    generate(int(args.start), int(args.end), args.mode)
    print("Finished with {}s".format(-current_time + time()))
