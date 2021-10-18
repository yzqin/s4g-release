import open3d
import numpy as np
from grasp_proposal.eval_experiment import config
import copy
from .math_utils import transformation_inv


class GraspVisualizer:
    def __init__(self, pcd: open3d.geometry.PointCloud, on_screen=True):
        self._on_screen = on_screen
        self._cloud = pcd
        self._visualization_group = [self._cloud]

    def add_single_pose(self, pose, real_mesh=False):
        if real_mesh:
            pass
        else:
            self._visualization_group.extend(self.get_hand_geometry(pose))

    def add_multiple_poses(self, poses: np.ndarray, real_mesh=False):
        assert poses.ndim == 3, "Poses should have dimension (n, 4, 4)"
        assert poses.shape[1:3] == (4, 4), "Poses should have dimension (n, 4, 4)"
        for i in range(poses.shape[0]):
            self.add_single_pose(poses[i], real_mesh)

    def visualize(self):
        frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        open3d.visualization.draw_geometries(self._visualization_group + [frame])

    @staticmethod
    def get_hand_geometry(local2global, color=(0.1, 0.6, 0.3)):
        back_hand = open3d.geometry.TriangleMesh.create_box(height=2 * config.HALF_BOTTOM_WIDTH,
                                                            depth=config.HALF_HAND_THICKNESS * 2,
                                                            width=config.BOTTOM_LENGTH - config.BACK_COLLISION_MARGIN)

        temp_trans = np.eye(4)
        temp_trans[0, 3] = -config.BOTTOM_LENGTH
        temp_trans[1, 3] = -config.HALF_BOTTOM_WIDTH
        temp_trans[2, 3] = -config.HALF_HAND_THICKNESS
        back_hand.transform(temp_trans)

        finger = open3d.geometry.TriangleMesh.create_box((config.FINGER_LENGTH + config.BACK_COLLISION_MARGIN),
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
        back_hand.transform(local2global)
        finger.transform(local2global)
        left_finger.transform(local2global)

        vis_list = [back_hand, left_finger, finger]
        for vis in vis_list:
            vis.compute_vertex_normals()
        return vis_list
