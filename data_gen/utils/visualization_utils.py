import open3d
import numpy as np
import copy
from configs import config
import socket

hostname = socket.gethostname()

create_box = open3d.geometry.TriangleMesh.create_box


def get_hand_geometry(T_global_to_local, color=(0.1, 0.6, 0.3)):
    back_hand = create_box(height=2 * config.HALF_BOTTOM_WIDTH,
                           depth=config.HALF_HAND_THICKNESS * 2,
                           width=config.BOTTOM_LENGTH - config.BACK_COLLISION_MARGIN)
    # back_hand = open3d.geometry.TriangleMesh.create_cylinder(height=0.1, radius=0.02)

    temp_trans = np.eye(4)
    temp_trans[0, 3] = -config.BOTTOM_LENGTH
    temp_trans[1, 3] = -config.HALF_BOTTOM_WIDTH
    temp_trans[2, 3] = -config.HALF_HAND_THICKNESS
    back_hand.transform(temp_trans)

    finger = create_box((config.FINGER_LENGTH + config.BACK_COLLISION_MARGIN),
                        config.FINGER_WIDTH,
                        config.HALF_HAND_THICKNESS * 2)
    finger.paint_uniform_color(color)
    back_hand.paint_uniform_color(color)
    left_finger = copy.deepcopy(finger)

    temp_trans = np.eye(4)
    temp_trans[1, 3] = config.HALF_BOTTOM_SPACE
    temp_trans[2, 3] = -config.HALF_HAND_THICKNESS
    temp_trans[0, 3] = -config.BACK_COLLISION_MARGIN
    left_finger.transform(temp_trans)
    temp_trans[1, 3] = -config.HALF_BOTTOM_WIDTH
    finger.transform(temp_trans)

    # Global transformation
    T_local_to_global = np.linalg.inv(T_global_to_local)

    back_hand.transform(T_local_to_global)
    finger.transform(T_local_to_global)
    left_finger.transform(T_local_to_global)

    vis_list = [back_hand, left_finger, finger]
    for vis in vis_list:
        vis.compute_vertex_normals()
    return vis_list


def create_point_sphere(pos):
    sphere = open3d.geometry.create_mesh_sphere(0.08)
    trans = np.eye(4)
    trans[0:3, 3] = pos
    sphere.transform(trans)
    sphere.paint_uniform_color([0, 1, 0])
    return sphere


def create_coordinate_marker(frame, point):
    p = [point, point + frame[:, 0], point + frame[:, 1], point + frame[:, 2]]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.Vector3dVector(p)
    line_set.lines = open3d.Vector2iVector(lines)
    line_set.colors = open3d.Vector3dVector(colors)
    return line_set


def create_local_points_with_normals(points, normals):
    """
    Draw points and corresponding normals in local frame with gripper in (0, 0, 0)
    :param points: (3, n) np.array
    :param normals: (3, n) np.array
    :return: Open3d geometry list
    """
    hand = get_hand_geometry(np.eye(4))
    assert points.shape == normals.shape
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.Vector3dVector(points.T)
    pc.normals = open3d.Vector3dVector(normals.T)
    return [pc] + hand
