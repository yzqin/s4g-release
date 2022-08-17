import os

import numpy as np
import open3d
from matplotlib import cm

color_map = cm.get_cmap('jet', 256)


def main():
    data_dir = "../objects/processed_single_object_grasp"
    object_names = ["camera"]
    for object_name in object_names:
        data_path = os.path.join(data_dir, '{}.pkl'.format(object_name))

        data = np.load(data_path, allow_pickle=True)
        cloud = data['cloud']
        grasp_pose = data['grasp_pose']
        grasp_point_index = data['grasp_point_index'].astype(int)
        grasp_point_index = grasp_point_index.tolist()
        print("{} has {} frames".format(object_name, len(grasp_point_index)))

        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(cloud)

        score = np.zeros([cloud.shape[0]])
        color = np.zeros([cloud.shape[0], 3])

        for i in grasp_point_index:
            score[i] = 0.99

        for i in range(cloud.shape[0]):
            color[i, :] = color_map(score[i])[0:3]

        pc.colors = open3d.utility.Vector3dVector(color)

        while True:
            from utils.visualization_utils import get_hand_geometry
            vis = open3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(pc)
            vis.run()
            vis.destroy_window()
            pick_inds = vis.get_picked_points()
            vis_list = [pc]
            for ind in pick_inds:
                if ind in grasp_point_index:
                    frame_index = grasp_point_index.index(ind)
                    grasp = np.linalg.inv(grasp_pose[frame_index])
                    print(grasp_pose[frame_index])
                    hand = get_hand_geometry(grasp)
                    ball = open3d.geometry.TriangleMesh.create_sphere(0.0015)
                    ball.translate(cloud[ind, :])
                    vis_list.extend(hand)
                    vis_list.append(ball)
            open3d.visualization.draw_geometries(vis_list)
        #


if __name__ == '__main__':
    main()
