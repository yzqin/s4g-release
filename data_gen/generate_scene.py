from configs.path import get_resource_dir_path
from data_generator.data_contact_scene_generator import GenerateContactScene
from data_generator.data_object_contact_point_generator import GenerateContactObjectData
from data_generator.data_object_darboux_generator import GenerateDarbouxObjectData
from data_generator.data_scene_generator import GenerateDarbouxScene
from data_generator.point_cloud_scene_generator import GenerateSceneServer

npy_dir = get_resource_dir_path('npy')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)
    parser.add_argument('--end', '-e', type=int)
    parser.add_argument('--mode', '-p', type=str,
                        choices=['point_cloud', 'darboux_data_object', 'darboux_data_scene', 'contact_pair_object',
                                 'contact_pair_scene'])
    args = parser.parse_args()
    mode = args.mode
    start = args.start
    end = args.end

    if mode == 'point_cloud':
        generator = GenerateSceneServer()
    elif mode == 'darboux_data_object':
        generator = GenerateDarbouxObjectData()
    elif mode == 'darboux_data_scene':
        generator = GenerateDarbouxScene()
    elif mode == 'contact_pair_object':
        generator = GenerateContactObjectData()
    elif mode == 'contact_pair_scene':
        generator = GenerateContactScene()
    else:
        raise NotImplementedError

    generator.run_loop(start, end)
