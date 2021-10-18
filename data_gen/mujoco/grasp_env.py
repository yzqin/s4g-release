import numpy as np
from mujoco.grasp_arena import GraspingArena
import mujoco_py
import os
import time
from mujoco_py.builder import MujocoException
from configs import dataset_config
import transforms3d

dirname = os.path.dirname(__file__)


class GraspEnv:
    def __init__(self, stl_directory=os.path.join(dirname, '../../stl'), percentage=0.5, visualize=False):
        self.arena = GraspingArena()
        self.percentage = percentage
        self.visualize = visualize
        self.tolerance = 1e-3
        self.sim = None
        self.model = None
        self.viewer = None
        self.obj = []
        self.stl_dir = stl_directory

        self.gripper_id = None
        self.gripper_actuator_id = None
        self.free_qvel_id = None

    def add_object(self, npy_path=None):
        if npy_path:
            obj_dict = np.load(npy_path, allow_pickle=True)[()]
            for name, pose in obj_dict.items():
                self.arena.add_free_object(os.path.join(os.path.abspath(self.stl_dir), name + '.stl'), pose=pose)
        else:
            for name in dataset_config.NAME_LIST:
                if np.random.rand() > self.percentage:
                    continue
                self.arena.add_free_object(os.path.join(os.path.abspath(self.stl_dir), name + '.stl'))
                self.obj.append(name)

    def wait_for_static(self):
        xml = self.arena.get_xml()
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        if self.visualize:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer.render()

        for _ in range(1000):
            try:
                self.sim.step()
                if self.visualize:
                    self.viewer.render()
            except MujocoException:
                return None

        # Reset simulation and disable collision of the wall
        for wall_num in range(4):
            wall_id = self.model.geom_name2id('wall_{}'.format(wall_num))
            self.model.geom_pos[wall_id][2] = -10

        for _ in range(500):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        self.gripper_id = self.model.get_joint_qpos_addr('gripper')[0]
        self.gripper_actuator_id = [self.model.actuator_name2id('left_finger_motor'),
                                    self.model.actuator_name2id('right_finger_motor')]
        self.free_qvel_id = self.model.get_joint_qvel_addr('gripper')[0]

    def get_static_model(self):
        self.add_object(npy_path='../npy/32.npy')
        self.arena.save_model('./temp/temp.xml', pretty=True)
        self.wait_for_static()
        return self.sim.get_state()

    def evaluate_single_grasp(self, T_local_to_global):
        # TODO: init actuator
        quat = transforms3d.quaternions.mat2quat(T_local_to_global[0:3, 0:3])
        self.sim.data.qpos[self.gripper_id:self.gripper_id + 3] = [0, 0, 1]
        self.sim.data.qpos[self.gripper_id + 3:self.gripper_id + 7] = quat
        if self.visualize:
            self.viewer.render()
        self.sim.step()
        if self.visualize:
            self.viewer.render()

        self.sim.data.ctrl[self.gripper_actuator_id] = 1
        for i in range(200):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        for i in range(2000):
            self.sim.data.qvel[self.free_qvel_id + 2] = 1
            self.sim.data.qvel[self.free_qvel_id:self.free_qvel_id + 2] = 0
            self.sim.step()
            if self.visualize:
                self.viewer.render()

    def evaluate(self):
        init_state = self.get_static_model()
        for i in range(1):
            self.sim.set_state(init_state)
            self.evaluate_single_grasp(np.array([[-0.44603145, 0.4873747, -0.7506809, 0.9779138],
                                                 [-0.8455296, 0.04557911, 0.53197956, -0.3950432],
                                                 [-0.2934887, -0.8720025, -0.39176008, 0.20937684],
                                                 [0., 0., 0., 1.]]))


if __name__ == '__main__':
    env = GraspEnv(visualize=True)
    env.evaluate()
