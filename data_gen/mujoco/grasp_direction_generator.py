import os
import mujoco_py
import numpy as np
from configs.path import get_resource_dir_path
from configs.dataset_config import DIR_LIST
import xml.etree.ElementTree as ET
import io
import pickle
import time

dir_list = np.array(DIR_LIST)
dir_list = dir_list / np.linalg.norm(dir_list, axis=1, keepdims=True)
print(dir_list)
xml_dir = get_resource_dir_path('xml')
mesh_dir = get_resource_dir_path('convex_stl')
npy_dir = get_resource_dir_path('npy')
direction_dir = get_resource_dir_path('direction')
GRAVITY = np.array([0, 0, 9.8])


class DirectionGenerator:
    def __init__(self, scene_num: int, time_step=0.002, visualize=False):
        xml_path = os.path.join(xml_dir, "{}.xml".format(scene_num))
        self.xml = ET.parse(xml_path)
        root = self.xml.getroot()
        root.find('compiler').set('meshdir', mesh_dir)
        root.find('option').set('timestep', str(time_step))
        contact = ET.Element('contact')
        exclude = ET.Element('exclude', name='table_exclude', body1='table', body2='world')
        contact.append(exclude)
        root.append(contact)

        with io.StringIO() as string:
            string.write(ET.tostring(root, encoding="unicode"))
            xml_string = string.getvalue()
        self.model = mujoco_py.load_model_from_xml(xml_string)
        self.sim = mujoco_py.MjSim(self.model)

        if visualize:
            self.viewer = mujoco_py.MjViewer(self.sim)
        else:
            self.sim.forward()
        self.visualize = visualize

        self.obj = None
        self.obj_qvel_id = None
        self.obj_qpos_id = None
        self.obj_body_id = None
        self.body_center = None
        self.body_geom = []
        npy = np.load(os.path.join(npy_dir, "{}.npy".format(scene_num)), allow_pickle=True)[()]
        self.get_objects(npy)
        self.table_body_id = self.model.body_name2id('table')

        self.displacement = 0.2
        self.tolerance = 2e-1
        self.time_step = time_step
        self.scene_num = scene_num
        self.quat_threshold = 0.9

    def get_objects(self, npy):
        mujoco_py.functions.mj_resetDataKeyframe(self.model, self.sim.data, 0)
        self.sim.step()
        self.obj = sorted(list(npy.keys()))
        obj_joint_id = []
        obj_body_id = []
        obj_qpos_id = []
        body_center = np.zeros((len(self.obj), 3))
        for i, obj in enumerate(self.obj):
            obj_joint_id.append(self.model.get_joint_qvel_addr("{}_joint".format(obj))[0])
            obj_qpos_id.append(self.model.get_joint_qpos_addr("{}_joint".format(obj))[0])
            obj_body_id.append(self.model.body_name2id(obj))
            body_center[i, :] = self.sim.data.subtree_com[obj_body_id[i]]
        self.obj_body_id = np.array(obj_body_id)
        self.obj_qvel_id = np.array(obj_joint_id)
        self.obj_qpos_id = np.array(obj_qpos_id)
        self.body_center = body_center

        if self.visualize:
            geom_bodyid = self.model.geom_bodyid[:]
            for i, body_id in enumerate(self.obj_body_id):
                index = np.where(geom_bodyid == body_id)
                self.body_geom.append(index[0])

    def get_all_valid_direction(self):
        velocity = 1
        max_step = int(self.displacement / velocity / self.time_step)
        move_steps = np.ones((len(self.obj), dir_list.shape[0]), dtype=np.float) * max_step
        for i, obj in enumerate(self.obj):
            joint_id = self.obj_qvel_id[i]
            body_id = self.obj_body_id[i]
            qpos_id = self.obj_qpos_id[i]
            others = np.repeat(np.append(self.obj_qvel_id[0:i], self.obj_qvel_id[i + 1:]), 3)
            others[1::3] += 1
            others[2::3] += 2
            self.model.exclude_signature[0] = ((self.table_body_id + 1) << 16) + self.obj_body_id[i] + 1
            balance_gravity = GRAVITY * self.model.body_subtreemass[body_id]
            balance_torque = np.cross(self.sim.data.body_xpos[body_id, :] - self.sim.data.subtree_com[body_id, :],
                                      balance_gravity) * 0
            balance = np.append(balance_gravity, [balance_torque])

            if self.visualize:
                last_color = self.model.geom_rgba[self.body_geom[i], :]
                self.model.geom_rgba[self.body_geom[i], :] = np.array([[1, 0, 0, 0.3]])

            for dir_i, direction in enumerate(dir_list):
                mujoco_py.functions.mj_resetDataKeyframe(self.model, self.sim.data, 0)
                init_quat = np.copy(self.sim.data.qpos[qpos_id + 3: qpos_id + 7])
                self.sim.data.xfrc_applied[body_id, :] = 1 * balance
                qv = np.append(direction * velocity, [0, 0, 0])
                for step in range(max_step):
                    self.sim.data.qvel[joint_id: joint_id + 6] = qv
                    self.sim.step()
                    if self.visualize:
                        self.viewer.render()
                    max_velocity = np.max(np.abs(self.sim.data.qvel[others]))
                    if max_velocity > self.tolerance or np.inner(
                            self.sim.data.qpos[qpos_id + 3: qpos_id + 7], init_quat) < self.quat_threshold:
                        move_steps[i, dir_i] = step

                        if self.visualize:
                            if max_velocity > self.tolerance:
                                max_others_id = \
                                    np.where(np.abs((np.abs(self.sim.data.qvel[others])) - max_velocity) < 1e-6)[0]
                                max_id = int(np.where(self.obj_qvel_id == others[max_others_id] // 6 * 6)[0][0])
                                temp_color = self.model.geom_rgba[self.body_geom[max_id], :]
                                self.model.geom_rgba[self.body_geom[max_id], :] = np.array([[0.1, 0.1, 0.1, 0.3]])
                                for _ in range(100):
                                    self.sim.data.qvel[joint_id: joint_id + 6] = qv
                                    self.sim.step()
                                    self.viewer.render()
                                self.model.geom_rgba[self.body_geom[max_id], :] = temp_color
                            else:
                                self.model.geom_rgba[self.body_geom[i], :] = np.array([[0, 0, 0, 0.9]])
                                for _ in range(100):
                                    self.sim.data.qvel[joint_id: joint_id + 6] = [0, 0, 0, 0, 0, 0]
                                    self.sim.step()
                                    self.viewer.render()
                        break

            if self.visualize:
                self.model.geom_rgba[self.body_geom[i], :] = last_color
        return move_steps / max_step * self.displacement

    def run(self):
        tic = time.time()
        move_distance = self.get_all_valid_direction()
        save_path = os.path.join(direction_dir, '{}.p'.format(self.scene_num))
        result = {'move_distance': move_distance, 'obj_list': self.obj, 'mesh_center': self.body_center}
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)
            print('It takes {}s for {}'.format(time.time() - tic, save_path))


if __name__ == '__main__':
    a = DirectionGenerator(56, visualize=True)
    a.run()
