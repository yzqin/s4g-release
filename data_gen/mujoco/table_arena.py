from mujoco.base.arena_base import ArenaBase
import os
import numpy as np
from mujoco.base.mjcf_utils import array_to_string, new_geom, new_body, new_joint
import xml.etree.ElementTree as ET
from matplotlib import cm
from configs.path import get_resource_dir_path

dirname = os.path.dirname(__file__)
color_map = cm.get_cmap('jet', 256)


class TableArena(ArenaBase):
    def __init__(self, table_full_size=(0.76, 0.69, 0.55), table_friction=(5, 0.005, 0.0001), convex=True):
        xml_file = os.path.join(dirname, 'assets/table_arena.xml')
        ArenaBase.__init__(self, xml_file=xml_file)

        self.table_thickness = 0.4
        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_top_height = table_full_size[2] + self.table_thickness / 2

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_visual = self.table_body.find("./geom[@name='table_geom']")
        self.table_top = self.table_body.find("./body[@name='table_top']")

        self.configure_location()
        self.configure_wall(visible=True)
        size = self.create_default_element('size')
        self.convex = convex
        if convex:
            self.compiler.set('meshdir', get_resource_dir_path('convex_stl'))
            size.set('nconmax', '3000')
            size.set('njmax', '3000')
        else:
            self.compiler.set('meshdir', get_resource_dir_path('stl'))
            size.set('nconmax', '500')
            size.set('njmax', '500')

    def configure_location(self):
        center_pos = np.array([0, 0, self.table_full_size[2]])
        self.table_body.set("pos", array_to_string(center_pos))
        self.table_visual.set("friction", array_to_string(self.table_friction))
        table_plane_size = self.table_half_size
        table_plane_size[2] = self.table_thickness / 2
        self.table_visual.set("size", array_to_string(table_plane_size))

        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, table_plane_size[2]]))
        )

    def configure_wall(self, wall_half_thickness=0.05, wall_height=5.00, visible=True):
        setting_table = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for num in range(len(setting_table)):
            i = setting_table[num]
            pos = np.array([i[0] * self.table_half_size[0], i[1] * self.table_half_size[1], wall_height / 2 - 0.5])
            if i[0] != 0:
                size = np.array([wall_half_thickness, self.table_half_size[1], wall_height / 2])
            else:
                size = np.array([self.table_half_size[0], wall_half_thickness, wall_height / 2])

            if visible:
                rgba = [0.9, 0, 0, 0.2]
            else:
                rgba = [0, 0, 0, 0]

            wall = new_geom(geom_type='box', size=size, pos=pos, rgba=rgba)
            wall.set('name', 'wall_{}'.format(num))
            wall.set('contype', '1')
            wall.set('conaffinity', '20')
            self.table_top.append(wall)

    def add_free_object(self, stl_path: str, xyz, quat):
        name = os.path.splitext(os.path.basename(stl_path))[0]
        rgba = list(color_map(int(name[0:3]) * 3))
        body = new_body(name=name, pos=xyz, quat=quat)
        geom = new_geom(geom_type='mesh', mesh=name + '_mesh', size='0', density='1000', rgba=rgba,
                        solref="-6000 -300", solimp="0.98 0.999 0.001 0.1 6", friction="10 0.01 0.0001")
        joint = new_joint(type='free', damping='0.0001', name=name + '_joint')
        body.append(geom)
        body.append(joint)

        asset_mesh = ET.Element('mesh', {'file': "{}.stl".format(name), 'name': name + '_mesh'})
        self.asset.append(asset_mesh)
        self.worldbody.append(body)

    def add_free_convex_object(self, stl_folder: str, xyz, quat):
        name = os.path.basename(stl_folder)
        rgba = list(color_map(int(name[0:3]) * 3))
        body = new_body(name=name, pos=xyz, quat=quat)
        stl_list = [stl for stl in os.listdir(stl_folder) if stl.startswith("{}_".format(name))]
        for i, stl_name in enumerate(stl_list):
            stl_path = os.path.join(name, stl_name)
            geom = new_geom(geom_type='mesh', mesh=stl_name + '_mesh', size='0', density='1000', rgba=rgba,
                            solref="-20000 -300", solimp="0.999 0.9999 0.001 0.1 6", friction="10 0.01 0.0001")
            asset_mesh = ET.Element('mesh', {'file': stl_path, 'name': stl_name + '_mesh'})
            body.append(geom)
            self.asset.append(asset_mesh)
        joint = new_joint(type='free', damping='0.001', name=name + '_joint')
        body.append(joint)
        self.worldbody.append(body)

    def generate_random_pose(self, wall_height=5.00, height_percentage=None):
        rand_xy = np.random.uniform(-1, 1, 2)
        area_size = self.table_half_size[:2] - 0.15
        xy = rand_xy * area_size
        if height_percentage:
            z = self.table_top_height + height_percentage * (wall_height - 0.5)
        else:
            z = self.table_top_height + np.random.uniform(0.05, 1, 1) * (wall_height - 1)
        xyz = np.append(xy, z)
        quat = np.random.uniform(-1, 1, 4)
        return xyz, array_to_string(quat)

    def add_fixed_scene(self, scene_name: int):
        scene_path = os.path.join(os.path.join(dirname, 'assets/scene_{}.npy'.format(scene_name)))
        assert os.path.exists(scene_path), "Fixed scene test: scene name below do not exist: \n {}".format(scene_path)
        saved_pose = np.load(scene_path, allow_pickle=True)[()]
        for name, pose in saved_pose.items():
            if self.convex:
                stl_dir = os.path.join(get_resource_dir_path('convex_stl'), name)
                self.add_free_convex_object(stl_folder=stl_dir, xyz=pose[0:3], quat=pose[3:7])
            else:
                stl_path = os.path.join(get_resource_dir_path('stl'), "{}.stl".format(name))
                self.add_free_object(stl_path=stl_path, xyz=pose[0:3], quat=pose[3:7])
        return saved_pose.keys()


class SingleObjectTableArena(TableArena):
    def __init__(self, table_full_size=(0.76, 0.69, 0.55), table_friction=(5, 0.005, 0.0001), obj='004_sugar_box'):
        TableArena.__init__(self, table_full_size, table_friction, convex=True)
        self.object_name = obj
        stl_home_dir = get_resource_dir_path('convex_stl')
        self.stl_dir = os.path.join(os.path.abspath(stl_home_dir), self.object_name)

    def add_multiple_free_convex_object(self, num=5):
        stl_folder = self.stl_dir
        add_asset = False
        name_list = []
        for i in range(num):
            name = "{}@{}".format(self.object_name, i)
            name_list.append(name)
            rgba = list(color_map(int(name[0:3]) * 3))
            xyz, quat = self.generate_random_pose()
            body = new_body(name=name, pos=xyz, quat=quat)
            stl_list = [stl for stl in os.listdir(stl_folder) if stl.startswith("{}_".format(self.object_name))]
            for i, stl_name in enumerate(stl_list):
                stl_path = os.path.join(self.object_name, stl_name)
                geom = new_geom(geom_type='mesh', mesh=stl_name + '_mesh', size='0', density='3000', rgba=rgba,
                                solref="-20000 -300", solimp="0.999 0.9999 0.001 0.1 6", friction="10 0.01 0.0001")
                if add_asset:
                    body.append(geom)
                    continue
                else:
                    asset_mesh = ET.Element('mesh', {'file': stl_path, 'name': stl_name + '_mesh'})
                    body.append(geom)
                    self.asset.append(asset_mesh)
            add_asset = True
            joint = new_joint(type='free', damping='0.001', name="{}_joint_{}".format(name, i))
            body.append(joint)
            self.worldbody.append(body)

        return name_list
