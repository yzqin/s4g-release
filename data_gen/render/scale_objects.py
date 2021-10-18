import bpy
import os
import sys

# Blender python internally do not find modules in the current workspace, we need to add it explicitly
sys.path.append(os.getcwd())
from configs.dataset_config import NAME_SCALE
from configs.path import get_resource_dir_path

args = sys.argv
args = args[args.index("--") + 1:]  # get all args after "--"
print(args)  # --> ['example', 'args', '123']

OBJ_OUTPUT_DIR = get_resource_dir_path('scale_obj')
PLY_OUTPUT_DIR = get_resource_dir_path('scale_ply')


def handle_single_scale(obj_path, scale, num_order):
    bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='X', axis_up='Z')
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    output_path_obj = os.path.join(OBJ_OUTPUT_DIR, "{}#{}.obj".format(obj_name, num_order))
    output_path_ply = os.path.join(PLY_OUTPUT_DIR, "{}#{}.ply".format(obj_name, num_order))

    object_name = [obj for obj in bpy.data.objects.keys() if obj.startswith('0')][0]
    bpy.data.objects[object_name].select = True
    bpy.data.objects[object_name].scale = (scale, scale, scale)
    bpy.ops.export_scene.obj(filepath=output_path_obj, use_selection=True, axis_forward='X', axis_up='Z')
    bpy.context.scene.objects.active = bpy.data.objects[object_name]
    bpy.ops.export_mesh.ply(filepath=output_path_ply, axis_forward='X', axis_up='Z')
    bpy.data.objects.remove(bpy.data.objects[object_name])


# example: blender --python render/scale_objects.py -- /Users/yuzheqin/ycb_data/obj
if __name__ == '__main__':
    input_dir = args[0]
    files = [file for file in os.listdir(input_dir) if not file.startswith('.')]
    print("The list of files :{}".format(files))
    for obj in bpy.data.objects.keys():
        bpy.data.objects.remove(bpy.data.objects[obj])
    assert len(bpy.data.objects.keys()) == 0

    name_scale = sorted(list(NAME_SCALE.keys()))
    print(name_scale)

    for name in name_scale:
        scales = NAME_SCALE[name]
        file_name = "{}.obj".format(name)
        if file_name in files:
            input_file = os.path.join(input_dir, file_name)
        else:
            raise RuntimeError
        for i, scale in enumerate(scales):
            handle_single_scale(input_file, scale, i)
