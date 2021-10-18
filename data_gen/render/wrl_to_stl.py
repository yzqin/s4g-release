import bpy
import os
import sys

args = sys.argv
args = args[args.index("--") + 1:]  # get all args after "--"
print(args)  # --> ['example', 'args', '123']


def check_exist_obj():
    for obj in bpy.data.objects.keys():
        if obj.startswith('Shape_'):
            return True
    return False


def handle_single_wrl(wrl_path: str, output: str, name: str):
    assert os.path.exists(output) and os.path.exists(wrl_path)
    assert not check_exist_obj()

    bpy.ops.import_scene.x3d(filepath=wrl_path, axis_forward='X', axis_up='Z')
    objects = [obj for obj in bpy.data.objects.keys() if obj.startswith('Shape_')]
    bpy.ops.export_mesh.stl(filepath=os.path.join(output, "{}.stl".format(name)), axis_forward='X', axis_up='Z')
    for i, obj_name in enumerate(objects):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.data.objects[obj_name].select = True
        bpy.ops.export_mesh.stl(filepath=os.path.join(output, "{}_{}.stl".format(name, i)), use_selection=True,
                                axis_forward='X', axis_up='Z')
        bpy.data.objects.remove(bpy.data.objects[obj_name])


# example: blender --python wrl_to_stl.py -- /Users/yuzheqin/wrl /Users/yuzheqin/convex_stl
if __name__ == '__main__':
    input_dir = args[0]
    output_dir = args[1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    files = [file for file in os.listdir(input_dir) if not file.startswith('.')]
    print("The list of files :{}".format(files))
    bpy.data.objects.remove(bpy.data.objects['Cube'])
    bpy.data.objects.remove(bpy.data.objects['Camera'])
    bpy.data.objects.remove(bpy.data.objects['Lamp'])

    for file in files:
        input_file = os.path.join(input_dir, file)
        name = os.path.splitext(file)[0]
        output_name = os.path.join(output_dir, name)
        os.mkdir(output_name)
        handle_single_wrl(wrl_path=input_file, output=output_name, name=name)
