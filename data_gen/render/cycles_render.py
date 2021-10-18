import bpy
import numpy as np
import os
import sys
import open3d
import transforms3d
import time

# Blender python internally do not find modules in the current workspace, we need to add it explicitly
sys.path.append(os.getcwd())
from configs.dataset_config import NAME_LIST
from configs.path import get_resource_dir_path

CAMERA_POSE = [
    [0.8, 0, 1.7, 0.948, 0, 0.317, 0],
    [-0.8, 0, 1.6, -0.94, 0, 0.342, 0],
    [0.0, 0.75, 1.7, 0.671, -0.224, 0.224, 0.671],
    [0.0, -0.75, 1.6, -0.658, -0.259, -0.259, 0.658]
]

# Convention: pos + quaternion, where quaternion is [w,x,y,z]
SAVE_DIR = get_resource_dir_path('rendered')


def get_cycles_mapping_matrix(camera_intrinsics=np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])):
    x = np.linspace(0.5, 639.5, 640)
    y = np.linspace(0.5, 479.5, 480)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    cor = np.stack([x, y, np.ones_like(x)], axis=0)
    cor = np.dot(np.linalg.inv(camera_intrinsics), cor)
    nor = np.linalg.norm(cor, axis=0, keepdims=True)
    cor = cor / nor
    return cor


def local_to_global_transformation_quat(quat, point):
    T_local_to_global = np.eye(4)
    frame = transforms3d.quaternions.quat2mat(quat)
    T_local_to_global[0:3, 0:3] = frame
    T_local_to_global[0:3, 3] = point

    return T_local_to_global


def build_nodes():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    print(bpy.data.scenes['Scene'].render.resolution_x)
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')

    # create output node
    v = tree.nodes.new('CompositorNodeViewer')
    v.use_alpha = False

    # Links
    links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input
    print(tree.nodes.keys())


class BlensorSceneServer:
    def __init__(self, start_num: int, end_num: int, preload=False):
        self.obj_dir = get_resource_dir_path('obj')
        npy_dir = get_resource_dir_path('npy')
        self.npy_path = [os.path.join(npy_dir, "{}.npy".format(p)) for p in range(start_num, end_num) if
                         os.path.exists(os.path.join(npy_dir, "{}.npy".format(p)))]
        self.name_mapping = {}
        self.scanner = bpy.data.objects["Camera"]
        self.scanner.rotation_mode = 'QUATERNION'
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not preload:
            self.import_objects()
        self.mapping_matrix = get_cycles_mapping_matrix()
        self.camera_to_world = [local_to_global_transformation_quat(pose[3:7], pose[0: 3]) for pose in
                                CAMERA_POSE]

    def import_objects(self):
        for name in NAME_LIST:
            find = False
            obj_path = os.path.join(self.obj_dir, name + '.obj')
            bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='X', axis_up='Z')
            all_name = list(bpy.data.objects.keys())
            not_allocated_name = [n for n in all_name if n not in list(self.name_mapping.values())]
            for bpy_name in not_allocated_name:
                if bpy_name[0: 7] == name[0:7]:
                    obj = bpy.data.objects[bpy_name]
                    obj.location[2] = -10
                    obj.hide_render = True
                    find = True
                    self.name_mapping.update({name: bpy_name})
                    break
            assert find, 'Do not find objects in blender'

    def move_objects(self, pose_dict: dict):
        for key, value in pose_dict.items():
            obj = bpy.data.objects[self.name_mapping[key]]
            obj.rotation_mode = 'QUATERNION'
            obj.location[0:3] = value[0:3]
            obj.rotation_quaternion = value[3:7]
            obj.hide_render = False

    def move_back(self):
        for bpy_name in self.name_mapping.values():
            obj = bpy.data.objects[bpy_name]
            obj.location[2] = -10
            obj.hide_render = True

    def render(self, name):
        for i in range(len(CAMERA_POSE)):
            pose = CAMERA_POSE[i]
            self.scanner.location[0:3] = pose[0:3]
            self.scanner.rotation_quaternion[0:4] = pose[3:7]
            bpy.ops.render.render()
            depth_array = np.asarray(bpy.data.images['Viewer Node'].pixels)
            depth_array = depth_array[::4]
            noise_percentage = np.random.randn(depth_array.shape[0]) * 0.005 + 1.0
            depth_array_noise = depth_array * noise_percentage
            if depth_array.shape[0] != 640 * 480:  # TODO:hardcode here
                raise RuntimeError

            pc = depth_array[np.newaxis, :] * self.mapping_matrix
            pc_noise = depth_array_noise[np.newaxis, :] * self.mapping_matrix

            bool_index = pc[2, :] < 5.0
            index = np.nonzero(bool_index)[0]

            pc = pc[:, index]
            pc[2, :] *= -1
            pc_noise = pc_noise[:, index]
            pc_noise[2, :] *= -1

            cloud = open3d.PointCloud()
            cloud.points = open3d.Vector3dVector(pc.T)
            cloud.transform(self.camera_to_world[i])

            cloud_noise = open3d.PointCloud()
            cloud_noise.points = open3d.Vector3dVector(pc_noise.T)
            cloud_noise.transform(self.camera_to_world[i])

            filename = os.path.join(SAVE_DIR, "scene_{}_view_{}.pcd".format(name, i))
            filename_noise = os.path.join(SAVE_DIR, "scene_{}_view_{}_noise.pcd".format(name, i))
            open3d.write_point_cloud(filename, cloud)
            open3d.write_point_cloud(filename_noise, cloud_noise)

    def run_single_scene(self, npy_path):
        name = os.path.splitext(os.path.basename(npy_path))[0]
        pose_dict = np.load(npy_path, allow_pickle=True)[()]
        self.move_back()
        self.move_objects(pose_dict)
        self.render(name)

    def run(self):
        for npy in self.npy_path:
            tic = time.time()
            self.run_single_scene(npy)
            print("Finish {} with {}s".format(os.path.basename(npy), time.time() - tic))


class BlensorSceneDuplicate:
    def __init__(self, n_path):
        self.obj_dir = get_resource_dir_path('obj')
        self.pose_dict = np.load(n_path)[()]
        self.name_mapping = {}
        self.scanner = bpy.data.objects["Camera"]
        self.scanner.rotation_mode = 'QUATERNION'
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        self.mapping_matrix = get_cycles_mapping_matrix()
        self.camera_to_world = [local_to_global_transformation_quat(pose[3:7], pose[0: 3]) for pose in
                                CAMERA_POSE]

    def import_objects(self, obj_list):
        for name in obj_list:
            find = False
            obj_name = name.split('@')[0]
            obj_path = os.path.join(self.obj_dir, obj_name + '.obj')
            bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='X', axis_up='Z')
            all_name = list(bpy.data.objects.keys())
            not_allocated_name = [n for n in all_name if n not in list(self.name_mapping.values())]
            for bpy_name in not_allocated_name:
                if bpy_name[0: 7] == obj_name[0:7]:
                    obj = bpy.data.objects[bpy_name]
                    obj.location[2] = -10
                    obj.hide_render = True
                    find = True
                    self.name_mapping.update({name: bpy_name})
                    break
            assert find, 'Do not find objects in blender'

    def move_objects(self, pose_dict: dict):
        for key, value in pose_dict.items():
            obj = bpy.data.objects[self.name_mapping[key]]
            obj.rotation_mode = 'QUATERNION'
            obj.location[0:3] = value[0:3]
            obj.rotation_quaternion = value[3:7]
            obj.hide_render = False

    def move_back(self):
        for bpy_name in self.name_mapping.values():
            obj = bpy.data.objects[bpy_name]
            obj.location[2] = -10
            obj.hide_render = True

    def render(self, name):
        for i in range(len(CAMERA_POSE)):
            pose = CAMERA_POSE[i]
            self.scanner.location[0:3] = pose[0:3]
            self.scanner.rotation_quaternion[0:4] = pose[3:7]
            bpy.ops.render.render()
            depth_array = np.asarray(bpy.data.images['Viewer Node'].pixels)
            depth_array = depth_array[::4]
            noise_percentage = np.random.randn(depth_array.shape[0]) * 0.003 + 1.0
            depth_array_noise = depth_array * noise_percentage
            if depth_array.shape[0] != 640 * 480:  # TODO:hardcode here
                raise RuntimeError

            pc = depth_array[np.newaxis, :] * self.mapping_matrix
            pc_noise = depth_array_noise[np.newaxis, :] * self.mapping_matrix

            bool_index = pc[2, :] < 5.0
            index = np.nonzero(bool_index)[0]

            pc = pc[:, index]
            pc[2, :] *= -1
            pc_noise = pc_noise[:, index]
            pc_noise[2, :] *= -1

            cloud = open3d.PointCloud()
            cloud.points = open3d.Vector3dVector(pc.T)
            cloud.transform(self.camera_to_world[i])

            cloud_noise = open3d.PointCloud()
            cloud_noise.points = open3d.Vector3dVector(pc_noise.T)
            cloud_noise.transform(self.camera_to_world[i])

            filename = os.path.join(SAVE_DIR, "scene_{}_view_{}.pcd".format(name, i))
            filename_noise = os.path.join(SAVE_DIR, "scene_{}_view_{}_noise.pcd".format(name, i))
            open3d.write_point_cloud(filename, cloud)
            open3d.write_point_cloud(filename_noise, cloud_noise)

    def run_single_scene(self, npy_path):
        pose_dict = self.pose_dict
        self.import_objects(list(pose_dict.keys()))
        self.move_objects(pose_dict)
        name = os.path.splitext(os.path.basename(npy_path))[0]
        self.render(name)
        print(self.name_mapping)
        for _, bpy_name in self.name_mapping.items():
            print(list(bpy.data.objects.keys()))
            bpy.data.objects[bpy_name].select = True
            bpy.ops.object.delete()
            if len(bpy.data.objects.keys()) == 3:
                return


if __name__ == '__main__':
    # blensor assets/table_cycles.blend --python render/cycles_render.py
    # If you use xvfb-run, you should avoid use alias like blensor, specify the path otherwise
    # e.g.
    # xvfb-run -a -s "-screen 0 640x480x24"
    # /root/software/blensor/blender assets/table_cycles.blend --python render/cycles_render.py
    server = BlensorSceneServer(1, 3)
    server.run()

    # obj_dir = get_resource_dir_path('obj')
    # npy_dir = get_resource_dir_path('npy')
    # npy_path = [os.path.join(npy_dir, "{}.npy".format(p)) for p in range(0, 200) if
    #             os.path.exists(os.path.join(npy_dir, "{}.npy".format(p)))]
    # for n_path in npy_path:
    #     server = BlensorSceneDuplicate(n_path)
    #     server.run_single_scene(n_path)
