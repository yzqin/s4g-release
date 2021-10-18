import numpy as np
from mujoco.table_arena import TableArena, SingleObjectTableArena
import mujoco_py
import os
from mujoco_py.builder import MujocoException
from configs import dataset_config
from configs.path import get_resource_dir_path

dirname = os.path.dirname(__file__)


class TableEnv:
    def __init__(self, convex=True, percentage=0.5,
                 visualize=False, random_seed=None):
        self.arena = TableArena(convex=convex)
        self.arena.option.set('gravity', '0 0 -9.8')
        self.percentage = percentage
        self.visualize = visualize
        self.tolerance = 2e-3
        self.sim = None
        self.model = None
        self.viewer = None
        self.obj = []
        self.convex = convex
        if random_seed:
            np.random.seed(random_seed)
        stl_dir = 'convex_stl' if convex else 'stl'
        self.stl_dir = get_resource_dir_path(stl_dir)

    def add_object(self):
        for name in dataset_config.NAME_LIST:
            if np.random.rand() > self.percentage:
                continue
            xyz, quat = self.arena.generate_random_pose()
            self.arena.add_free_object(os.path.join(os.path.abspath(self.stl_dir), name + '.stl'), xyz, quat)
            self.obj.append(name)

    def add_convex_object(self):
        obj_names = []
        for name in dataset_config.NAME_LIST:
            if np.random.rand() > self.percentage:
                continue
            else:
                obj_names.append(name)

        np.random.shuffle(obj_names)
        for i, name in enumerate(obj_names):
            xyz, quat = self.arena.generate_random_pose(height_percentage=(i + 1) / len(obj_names))
            self.arena.add_free_convex_object(os.path.join((os.path.abspath(self.stl_dir)), name), xyz, quat)
            self.obj.append(name)

    def simulate(self):
        xml = self.arena.get_xml()
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        if self.visualize:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.render()
        else:
            # OpenAI add a sim.forward() in MjViewer. To ensure reproducibility, add a explicit forward step
            self.sim.forward()

        obj_on_table = np.arange(len(self.obj))  # Maintain an list for velocity detection as as termination condition
        for _ in range(1000):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        # Reset simulation and disable collision of the wall
        for wall_num in range(4):
            wall_id = self.model.geom_name2id('wall_{}'.format(wall_num))
            self.model.geom_pos[wall_id][2] = -10

        for _ in range(500):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        mean_vel = 100.0
        while mean_vel > self.tolerance:
            obj_on_table, mean_vel = self.update_velocity(obj_on_table)
            for _ in range(50):
                self.sim.step()
                if self.visualize:
                    self.viewer.render()

        final_pos = {}
        for name in self.obj:
            pos = self.sim.data.get_body_xpos(name).astype(np.float32)
            if pos[2] < self.arena.table_top_height - 0.4 or abs(pos[0]) > self.arena.table_half_size[0] + 0.4:
                continue
            quat = self.sim.data.get_body_xquat(name).astype(np.float32)
            final_pos.update({name: np.append(pos, quat)})

        print(self.sim.data.time)
        return final_pos

    def update_velocity(self, obj_on_table):
        on_table_bool = self.sim.data.qpos[obj_on_table * 7 + 2] > self.arena.table_top_height - 0.4
        x_table_bool = np.abs(self.sim.data.qpos[obj_on_table * 7]) < self.arena.table_half_size[0] + 0.4
        valid = np.logical_and(on_table_bool, x_table_bool)
        new_obj_on_table = obj_on_table[valid]
        vel = np.max(
            np.abs(self.sim.data.qvel[
                       np.append(new_obj_on_table * 6, [new_obj_on_table * 6 + 1, new_obj_on_table * 6 + 2])]))
        return new_obj_on_table, vel

    def run(self):
        if self.convex:
            self.add_convex_object()
        else:
            self.add_object()
        try:
            final_pos = self.simulate()
        except MujocoException:
            return None
        print('Num of objects on the table {} / {}'.format(len(final_pos), len(self.obj)))
        return final_pos


class SingleObjectTableEnv:
    def __init__(self, obj_name='004_sugar_box', visualize=False, random_seed=None):
        self.arena = SingleObjectTableArena(obj=obj_name)
        self.arena.option.set('gravity', '0 0 -9.8')
        self.visualize = visualize
        self.tolerance = 2e-3
        self.sim = None
        self.model = None
        self.viewer = None
        self.obj = []
        if random_seed:
            np.random.seed(random_seed)

    def simulate(self):
        xml = self.arena.get_xml()
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        if self.visualize:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.render()
        else:
            # OpenAI add a sim.forward() in MjViewer. To ensure reproducibility, add a explicit forward step
            self.sim.forward()

        for _ in range(1000):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        # Reset simulation and disable collision of the wall
        for wall_num in range(4):
            wall_id = self.model.geom_name2id('wall_{}'.format(wall_num))
            self.model.geom_pos[wall_id][2] = -10

        for _ in range(5000):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        final_pos = {}
        for name in self.obj:
            pos = self.sim.data.get_body_xpos(name).astype(np.float32)
            if pos[2] < self.arena.table_top_height - 0.4 or abs(pos[0]) > self.arena.table_half_size[0] + 0.4:
                continue
            quat = self.sim.data.get_body_xquat(name).astype(np.float32)
            final_pos.update({name: np.append(pos, quat)})

        print(self.sim.data.time)
        return final_pos

    def run(self, num):
        name_list = self.arena.add_multiple_free_convex_object(num)
        self.obj.extend(name_list)
        # self.arena.save_model('single_temp.xml')
        try:
            final_pos = self.simulate()
        except MujocoException:
            return None
        print('Num of objects on the table {} / {}'.format(len(final_pos), len(self.obj)))
        return final_pos


if __name__ == '__main__':
    directory = get_resource_dir_path('npy')
    over_all = 0
    for i in range(10, 15):
        obj_name = '004_sugar_box'
        env = SingleObjectTableEnv(obj_name=obj_name, visualize=False, random_seed=i)
        pose = env.run(i)
        path_name = os.path.join(directory, "{}.npy".format(over_all))
        np.save(path_name, pose)
        over_all += 1

    # for i in range(1, 5):
    #     obj_name = '056_tennis_ball'
    #     env = SingleObjectTableEnv(obj_name=obj_name, visualize=False, random_seed=i)
    #     pose = env.run(i)
    #     path_name = os.path.join(directory, "{}.npy".format(over_all))
    #     np.save(path_name, pose)
    #     over_all += 1
    #
    # for i in range(1, 5):
    #     obj_name = '002_master_chef_can'
    #     env = SingleObjectTableEnv(obj_name=obj_name, visualize=False, random_seed=i)
    #     pose = env.run(i)
    #     path_name = os.path.join(directory, "{}.npy".format(over_all))
    #     np.save(path_name, pose)
    #     over_all += 1
    #
    # for i in range(1, 5):
    #     obj_name = '021_bleach_cleanser'
    #     env = SingleObjectTableEnv(obj_name=obj_name, visualize=False, random_seed=i)
    #     pose = env.run(i)
    #     path_name = os.path.join(directory, "{}.npy".format(over_all))
    #     np.save(path_name, pose)
    #     over_all += 1
    #
    # for i in range(1, 5):
    #     obj_name = '065-c_cups'
    #     env = SingleObjectTableEnv(obj_name=obj_name, visualize=False, random_seed=i)
    #     pose = env.run(i)
    #     path_name = os.path.join(directory, "{}.npy".format(over_all))
    #     np.save(path_name, pose)
    #     over_all += 1
    #
    # for i in range(1, 5):
    #     obj_name = '048_hammer'
    #     env = SingleObjectTableEnv(obj_name=obj_name, visualize=False, random_seed=i)
    #     pose = env.run(i)
    #     path_name = os.path.join(directory, "{}.npy".format(over_all))
    #     np.save(path_name, pose)
    #     over_all += 1
    #
    # for i in range(1, 5):
    #     obj_name = '035_power_drill'
    #     env = SingleObjectTableEnv(obj_name=obj_name, visualize=False, random_seed=i)
    #     pose = env.run(i)
    #     path_name = os.path.join(directory, "{}.npy".format(over_all))
    #     np.save(path_name, pose)
    #     over_all += 1
    #
    # for i in range(1, 5):
    #     obj_name = '072-c_toy_airplane'
    #     env = SingleObjectTableEnv(obj_name=obj_name, visualize=False, random_seed=i)
    #     pose = env.run(i)
    #     path_name = os.path.join(directory, "{}.npy".format(over_all))
    #     np.save(path_name, pose)
    #     over_all += 1
