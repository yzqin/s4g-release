import os
from shutil import copy2

INPUT = '/Users/yuzheqin/Downloads'
OUTPUT = '/Users/yuzheqin/ycb_data'
SUB_FOLDER_NAME = 'google_16k'

if __name__ == '__main__':
    objects_list = []
    # folders = {'obj': 'textured.obj', 'stl': 'nontextured.stl', 'ply': 'nontextured.ply', 'dae': 'textured.dae'}
    folders = {'ply': 'nontextured.ply'}
    for folder, file in folders.items():
        path = os.path.join(OUTPUT, folder)
        if not os.path.exists(path):
            os.mkdir(path)

        for data_folder in os.listdir(INPUT):
            if os.path.basename(data_folder)[0] != '0':
                continue
            name = os.path.basename(data_folder)
            objects_list.append(name)
            data_folder = os.path.join(INPUT, data_folder)
            ycb_data_folder = os.path.join(data_folder, SUB_FOLDER_NAME)

            input_path = os.path.join(ycb_data_folder, file)
            output_path = os.path.join(path, '{}.{}'.format(name, folder))
            copy2(input_path, output_path)
        print(objects_list)
