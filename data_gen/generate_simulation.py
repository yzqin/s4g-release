import numpy as np
import time
import multiprocessing as mp
import argparse
import os
from configs.path import get_resource_dir_path

parser = argparse.ArgumentParser()
parser.add_argument('--start', '-s', type=int)
parser.add_argument('--end', '-e', type=int)
parser.add_argument('--process', '-p', type=int)
parser.add_argument('--fixed', '-f', type=int, default=0)
args = parser.parse_args()
xml_path = get_resource_dir_path('xml')


def handle_single_env(name: int):
    from mujoco import TableEnv
    print('Begin simulate scene {}'.format(name))
    try:
        tic = time.time()
        env = TableEnv(visualize=True, percentage=0.30, random_seed=name)
        pose = env.run()
        if not pose:
            return False
        np.save(os.path.join(get_resource_dir_path('npy'), "{}.npy".format(name)), pose)
        with open(os.path.join(xml_path, "{}.xml".format(name)), 'w') as f:
            env.sim.save(f, keep_inertials=True)
        print("Num {}-th simulation takes {} s".format(name, time.time() - tic))
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == '__main__':
    # TODO: the simulation results of same random seed on server and laptop diverge, do not know why
    start = args.start
    end = args.end

    npy_list = os.listdir(get_resource_dir_path('npy'))
    npy_name = [int(n[:-4]) for n in npy_list if not n.startswith('.')]
    todo_list = [n for n in range(start, end) if n not in npy_name]
    print(todo_list)
    print("The num of simulation to be generated: {}".format(len(todo_list)))

    if args.fixed == 0:
        if args.process == 1:
            for i in todo_list:
                result = handle_single_env(i)
        else:
            with mp.Pool(processes=args.process) as pool:
                results = pool.map(handle_single_env, todo_list)
