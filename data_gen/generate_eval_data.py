from configs.path import get_resource_dir_path, get_scene_and_view_path
from eval.torch_scene_point_cloud import TorchScenePointCloud
from eval.evaluation_data_generator import EvalDataGenerator
import os
from time import time
import open3d
import pickle

eval_data_dir = get_resource_dir_path('eval_data')


def generate_eval(scene_name, num_of_view=1, visualize=False):
    current = time()

    if type(scene_name) == int:
        scene_name = str(scene_name)
    assert type(scene_name) == str
    scene_path, single_view_list = get_scene_and_view_path(scene_name, num_of_view=num_of_view)
    if not (os.path.exists(scene_path) and os.path.exists(single_view_list[0])):
        print("Scene name or view name do not exist for {}".format(scene_name))
        print(scene_path)
        return
    scene_cloud = TorchScenePointCloud(open3d.io.read_point_cloud(scene_path))

    for i in range(num_of_view):
        if not os.path.exists(single_view_list[i]):
            continue
        view_cloud = EvalDataGenerator(open3d.io.read_point_cloud(single_view_list[i]), view_num=i)
        # open3d.draw_geometries([scene_cloud.cloud, view_cloud.cloud])
        dict = view_cloud.run_collision(scene_cloud)

        output_path = os.path.join(eval_data_dir, "{}_view_{}_noise.p".format(scene_name, i))
        with open(output_path, 'wb') as file:
            pickle.dump(dict, file)
            print('Save data in {}'.format(output_path))

    print("It takes {} to finish scene {}".format(time() - current, scene_name))


def generate(start, end):
    for i in range(start, end):
        generate_eval(i)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)
    parser.add_argument('--end', '-e', type=int)
    args = parser.parse_args()

    current_time = time()
    generate(int(args.start), int(args.end))
    print("Finished with {}s".format(-current_time + time()))
