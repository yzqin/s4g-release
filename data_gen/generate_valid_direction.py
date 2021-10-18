from mujoco.grasp_direction_generator import DirectionGenerator
from time import time
from configs.path import get_resource_dir_path
import os
import multiprocessing as mp


def handle_single_direction(scene_name: int):
    generator = DirectionGenerator(scene_num=scene_name, visualize=False)
    generator.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)
    parser.add_argument('--end', '-e', type=int)
    parser.add_argument('--process', '-p', type=int)
    args = parser.parse_args()

    start = args.start
    end = args.end

    current_time = time()
    if not os.path.exists(get_resource_dir_path('direction')):
        os.mkdir(get_resource_dir_path('direction'))

    npy_list = os.listdir(get_resource_dir_path('npy'))
    direction_list = os.listdir(get_resource_dir_path('direction'))
    todo_list = [n for n in range(start, end) if
                 "{}.npy".format(n) in npy_list and "{}.p".format(n) not in direction_list]
    print(todo_list)
    print("The num of scene valid direction to be generated: {}".format(len(todo_list)))

    if args.process == 1:
        for i in todo_list:
            handle_single_direction(i)
    else:
        with mp.Pool(processes=args.process) as pool:
            results = pool.map(handle_single_direction, todo_list)

    print("Generate {} direction with {}s".format(len(todo_list), time() - current_time))
