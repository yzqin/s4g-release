import numpy as np
from configs.path import get_resource_dir_path
import os
import pickle
from configs.dataset_config import NAME_LIST, NAME_TO_INDEX
import tqdm

NUM_OF_VIEW = 4

DIRECTION_CHANGE_LIST = np.array([[1, 0, 3, 2, 4], [0, 1, 2, 3, 4], [3, 2, 0, 1, 4], [2, 3, 1, 0, 4]])


def check_training_data(scene_num, data_list):
    exist = True
    for i in range(NUM_OF_VIEW):
        exist = exist and "{}_view_{}.p".format(scene_num, i) in data_list
        if not exist:
            return False

    return True


if __name__ == '__main__':
    data_list = os.listdir(get_resource_dir_path('contact_training_data'))
    direction_list = os.listdir(get_resource_dir_path('direction'))
    todo_list = []
    missing_data = 0
    for direction_path in direction_list:
        if direction_path.startswith('.'):
            continue
        name = os.path.splitext(os.path.basename(direction_path))[0]
        if check_training_data(name, data_list):
            todo_list.append(name)
        else:
            missing_data += 1
    print("Found {} file in direction folder".format(len(direction_list)))
    print("Will merge {} data".format(len(todo_list)))
    print("Missing {} training data".format(missing_data))

    data_dir = get_resource_dir_path('contact_training_data')
    direction_dir = get_resource_dir_path('direction')
    merge_dir = get_resource_dir_path('merged_data')
    for scene in tqdm.tqdm(todo_list):
        direction_path = os.path.join(direction_dir, "{}.p".format(scene))
        direction_data = np.load(direction_path, allow_pickle=True)
        move_distance = direction_data['move_distance']
        new_direction = np.ones([len(NAME_LIST) + 1, move_distance.shape[1]]) * -1
        new_direction[-1] = 0  # Label of table
        for obj_i, obj in enumerate(direction_data['obj_list']):
            obj_index = NAME_TO_INDEX[obj]
            new_direction[obj_index, :] = move_distance[obj_i, :]

        for i in range(NUM_OF_VIEW):
            direction_change = DIRECTION_CHANGE_LIST[i]
            data = np.load(os.path.join(data_dir, "{}_view_{}.p".format(scene, i)), allow_pickle=True)
            transformed_direction = new_direction[:, direction_change]
            data.update({'direction': transformed_direction})
            merge_path = os.path.join(merge_dir, '{}_view_{}.p'.format(scene, i))

            # grasp_label = data['objects_num']
            # data.update({'objects_label': grasp_label})
            # del data['objects_num']

            with open(merge_path, 'wb') as file:
                pickle.dump(data, file)
