import os
import socket

hostname = socket.gethostname()


def get_resource_dir_path(resource_type: str):
    dirname = os.path.dirname(__file__)
    if hostname.startswith('grasp') or hostname.startswith('py'):
        source_dir_name = '/cephfs/dataset/ycb_data'
    else:
        source_dir_name = os.path.join(dirname, '../../')
    path = os.path.abspath(os.path.join(source_dir_name, resource_type))
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_scene_and_view_path(scene_name: str, num_of_view, noise=True):
    scene_path = os.path.join(SCENE_PATH, "{}.ply".format(scene_name))
    view_path_list = [os.path.join(SINGLE_VIEW_PATH, "scene_{}_view_{}.pcd".format(scene_name, view_index)) for
                      view_index in range(num_of_view)]
    return scene_path, view_path_list


def get_data_scene_and_view_path(scene_name: str, num_of_view, noise: bool):
    scene_path = os.path.join(DATA_SCENE_PATH, "{}.p".format(scene_name))
    view_path_list = [os.path.join(SINGLE_VIEW_PATH, "scene_{}_view_{}.pcd".format(scene_name, view_index)) for
                      view_index in range(num_of_view)]
    if noise:
        noise_view_path_list = [
            os.path.join(SINGLE_VIEW_PATH, "scene_{}_view_{}_noise.pcd".format(scene_name, view_index)) for
            view_index in range(num_of_view)]
    else:
        noise_view_path_list = None

    return scene_path, view_path_list, noise_view_path_list


def get_contact_data_scene_and_view_path(scene_name: str, num_of_view, noise: bool):
    scene_path = os.path.join(CONTACT_DATA_SCENE_PATH, "{}.p".format(scene_name))
    view_path_list = [os.path.join(SINGLE_VIEW_PATH, "scene_{}_view_{}.pcd".format(scene_name, view_index)) for
                      view_index in range(num_of_view)]
    if noise:
        noise_view_path_list = [
            os.path.join(SINGLE_VIEW_PATH, "scene_{}_view_{}_noise.pcd".format(scene_name, view_index)) for
            view_index in range(num_of_view)]
    else:
        noise_view_path_list = None

    return scene_path, view_path_list, noise_view_path_list


def get_npy_and_training_data_path(scene_name, num_of_view):
    npy_path = os.path.join(NPY_PATH, "{}.npy".format(scene_name))
    training_data_list = [os.path.join(TRAINING_DATA_PATH, "{}_view_{}.p".format(scene_name, view_index)) for
                          view_index in range(num_of_view)]
    return npy_path, training_data_list


SCENE_PATH = get_resource_dir_path('scene')
DATA_SCENE_PATH = get_resource_dir_path('data_scene')
CONTACT_DATA_SCENE_PATH = get_resource_dir_path('contact_data_scene')
NPY_PATH = get_resource_dir_path('npy')
SINGLE_VIEW_PATH = get_resource_dir_path('rendered')
TRAINING_DATA_PATH = get_resource_dir_path('training_data')
CONVEX_STL_PATH = get_resource_dir_path("convex_stl")
STL_PATH = get_resource_dir_path('stl')
