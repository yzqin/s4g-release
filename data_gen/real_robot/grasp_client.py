import os.path as osp
import sys

sys.path.insert(0, osp.dirname(__file__) + '/..')
import roslibpy
import open3d
from robot.ros import remote
from robot.vision_client import VisionClient
import numpy as np
import transforms3d
from utils.visualization_utils import get_hand_geometry

# This matrix is used for the transformation from real ee_link to our assumed griper origin (bottom)
# compensate = np.array([[0.9989, 0.0423, -0.0203, 0.0128],
#                        [-0.0446, 0.9907, -0.1287, -0.0026],
#                        [0.0146, 0.1295, 0.9915, -0.0025],
#                        [0, 0, 0, 1.0000]])
# compensate = np.array([[0.99474372, -0.10057405, 0.01922993, -0.05097888],
#                        [-0.09979456, -0.99428509, -0.03792365, -0.04553803],
#                        [0.02293417, 0.03580528, -0.99909559, 0.01220511],
#                        [0., 0., 0., 1.]])

hand_to_ee = np.array([[1., 0., 0., -0.03607],
                       [0., 0.956206, 0.292695, -0.002978],
                       [0., -0.292695, 0.956206, -0.01328],
                       [0., 0., 0., 1.]])
ee_to_hand = np.linalg.inv(hand_to_ee)


class GraspClient:

    def __init__(self, table_to_eye: np.ndarray):
        self.grasp_service = roslibpy.core.Service(remote.ros, '/web_server/mat_grasp_server',
                                                   'web_server/MatGraspService')
        self.trans = roslibpy.Param(remote.ros, table_to_eye)
        self.camera_frame = 'kinect2_rgb_optical_frame'
        self.table_to_eye = table_to_eye

    def call_grasp(self, grasps, order=0, service_type='grasp', return_type='init'):
        req_dict = {'grasp': grasps, 'order': order, 'type': service_type, 'return_type': return_type}
        req = roslibpy.core.ServiceRequest(req_dict)
        print('Calling service...')
        res = self.grasp_service.call(req)
        print('Success: {} \n response: {} \n new_quat: {}'.format(res['success'], res['response'], res['new_pos']))

    def add_table_collision_pose(self, table_to_eye):
        # Call service to send the pose of table top center to the server
        table_to_eye_pose_stamped = self.mat_pose_2_pose_stamp(table_to_eye, self.camera_frame)
        grasp = {'pose_stamped': table_to_eye_pose_stamped}
        grasps = [grasp]
        self.call_grasp(grasps, service_type='table')

    @staticmethod
    def quat_pose_2_pose_stamp(pos, quat, frame_id):
        # quaternion should be in (w,x,y,z) convention
        position = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
        orientation = {'x': quat[1], 'y': quat[2], 'z': quat[3], 'w': quat[0]}
        pose = {'position': position, 'orientation': orientation}
        header = {'frame_id': frame_id}
        pose_stamped = {'header': header, 'pose': pose}
        return pose_stamped

    def mat_pose_2_pose_stamp(self, mat_pos, frame_id):
        rotation = mat_pos[0:3, 0:3]
        pos = mat_pos[0:3, 3]
        quaternion = transforms3d.quaternions.mat2quat(rotation)
        pose_stamped = self.quat_pose_2_pose_stamp(pos, quaternion, frame_id)
        return pose_stamped

    def table_mats_2_pose_stamp(self, mats: np.ndarray, from_table=True):
        assert mats.shape[1:3] == (4, 4) and len(mats.shape) == 3, 'Matrix should be in shape (4,4), not {}'.format(
            mats.shape)
        mats = np.matmul(mats, ee_to_hand)
        if from_table:
            mats = np.matmul(self.table_to_eye, mats)
        pose_stampeds = []
        for j in range(mats.shape[0]):
            pose_stamped = self.mat_pose_2_pose_stamp(mats[j], frame_id=self.camera_frame)
            pose_stampeds.append(pose_stamped)
        return pose_stampeds

    def generate_grasps(self, transformation_matrices: np.ndarray, scores: np.ndarray, return_directions: list,
                        reach_directions: list, reach_offsets: np.ndarray):
        pose_stampeds = self.table_mats_2_pose_stamp(transformation_matrices)
        grasps = []
        for i in range(transformation_matrices.shape[0]):
            grasp = {'pose_stamped': pose_stampeds[i], 'score': scores[i], 'return_direction': return_directions[i],
                     'reach_direction': reach_directions[i], 'reach_offset': reach_offsets[i]}
            grasps.append(grasp)
        return grasps

    def run(self, transformation_matrices: np.ndarray, scores: np.ndarray, return_directions: list,
            reach_directions: list, reach_offsets: np.ndarray, order_info=0, service_type="grasp"):
        grasps = self.generate_grasps(transformation_matrices, scores, return_directions, reach_directions,
                                      reach_offsets)
        self.call_grasp(grasps, order_info, service_type, return_type='init')


if __name__ == '__main__':
    table_frame = np.loadtxt("/home/rayc/Projects/3DGPD/tools/table_frame.txt")
    client = GraspClient(table_to_eye=table_frame)

    transformation_matrices = np.load("/home/rayc/Projects/3DGPD/top_frames.npy")


    point_cloud = open3d.io.read_point_cloud(
        "/home/rayc/Projects/3DGPD/outputs/pn2_negweight1.0_deeper_test/test_step00000/pred_pts.ply")

    for i in range(1):
        t = transformation_matrices[i]
        # if t[2, 2] < 0:
        #     transformation_matrices[i, :3, 1:3] = -transformation_matrices[i, :3, 1:3]
        # t[1, 3] -= 0.04
        hand = get_hand_geometry(np.linalg.inv(t))
        vis_list = [point_cloud]
        vis_list.extend(hand)
        open3d.visualization.draw_geometries(vis_list)

    transformation_matrices[:, 1, 3] -= 0.01
    client.run(transformation_matrices, np.array([0.8] * transformation_matrices.shape[0]),
               return_directions=['up'] * transformation_matrices.shape[0],
               reach_directions=['up'] * transformation_matrices.shape[0],
               reach_offsets=np.array([0.2] * transformation_matrices.shape[0]))
    remote.ros.terminate()
