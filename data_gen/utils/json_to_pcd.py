import os
import argparse
import json
import open3d
from configs.not_used_config import NAME_SCALE
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', '-j', type=str)
parser.add_argument('--pcd_dir', '-p', type=str)
args = parser.parse_args()
input_dir = args.json_dir
output_dir = args.pcd_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

valid_object = list(NAME_SCALE.keys())

if __name__ == '__main__':
    for filename in os.listdir(input_dir):
        name = os.path.basename(filename)[:-5]
        if name.startswith('.'):
            continue
        if name not in valid_object:
            continue
        json_path = os.path.join(input_dir, filename)
        with open(json_path, "r") as FILE:
            data = json.loads(FILE.read())
        points = list()
        normals = list()
        pairs = list()

        for i in range(len(data)):
            points.append(data[i]['v'])
            normals.append(data[i]['n'])
        for num, scale in enumerate(NAME_SCALE[name]):
            output_path = os.path.join(output_dir, "{}#{}.ply".format(name, num))
            print("{} {}".format(name, scale))
            cloud = np.array(points)
            cloud *= scale
            pc = open3d.geometry.PointCloud()
            pc.points = open3d.utility.Vector3dVector(cloud)
            pc.normals = open3d.utility.Vector3dVector(normals)
            # open3d.visualization.draw_geometries([pc])
            open3d.io.write_point_cloud(output_path,pc)
