import os
import time
from typing import Tuple, Optional

import numpy as np
import open3d
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs.yacs_config import load_cfg_from_file
from .utils.checkpoint import CheckPointer
from .utils.logger import setup_logger, MetricLogger
from .utils.math_utils import torch_batch_transformation_inv
from .cloud_processor.cloud_processor import CloudPreProcessor
from .utils.math_utils import transform_numpy_points
from .cloud_processor.view_collision_checker import CloudCollisionChecker
from .network_models.models.build_model import build_model
from .configs import real_world_config as realworld


class GraspDetector:
    __SUPPORTED_MODEL = ["curvature_model", "contact_model"]

    # TODO: delete temp variable
    _REAL2TRAIN = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    _TRAIN2REAL = np.linalg.inv(_REAL2TRAIN)

    def __init__(self, model: str = "curvature_model", training: bool = False, output_dir: str = "output",
                 logger_name: str = "S4G"):
        """
        Detect grasp pose in camera frame given point cloud in camera frame
        :param training:
        :param output_dir:
        :param logger_name:
        :param model: Currently, only "contact_model" and "curvature_model" is supported
        """
        # Load config file based on the given model name
        current_path = os.path.dirname(__file__)
        cfg_path = ""
        if model in self.__SUPPORTED_MODEL:
            cfg_path = os.path.join(current_path, "configs", "{}.yaml".format(model))
        else:
            print("Model {} is not supported, options are {}".format(model, self.__SUPPORTED_MODEL))
            exit(0)

        self.cfg = load_cfg_from_file(cfg_path)
        self.cfg.freeze()
        assert self.cfg.TEST.BATCH_SIZE == 1

        # Setup logger and corresponding logging directory
        self._output_path = os.path.join(current_path, "../", output_dir)
        os.makedirs(self._output_path, exist_ok=True)
        self.logger = setup_logger(logger_name, self._output_path, "unit_test")
        self.logger.info("Using {} of GPUs".format(torch.cuda.device_count()))
        self.logger.info("Load config file from {}".format(cfg_path))
        self.logger.debug("Running with config \n {}".format(self.cfg))

        # Load network parameters
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _, _ = build_model(self.cfg)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model).to(self._device, non_blocking=True)
        else:
            self.model = model.to(self._device, non_blocking=True)

        self.check_pointer = CheckPointer(self.model, save_dir=self._output_path, logger=self.logger)
        if self.cfg.TEST.WEIGHT:
            weight_path = self.cfg.TEST.WEIGHT.replace("${PROJECT_HOME}", os.path.join(current_path, "../"))
            self.check_pointer.load(weight_path, resume=False)
        else:
            self.check_pointer.load(None, resume=True)

        # Preparer metric logger and switch mode between training and testing
        self._training = training
        self.model.train(training)
        self.meters = MetricLogger(delimiter="  ")

        # Global variable
        self.vertical_direction = np.array([[0, 0, 1]], dtype=np.float32)

    def sample_single_cloud(self, points: np.ndarray) -> np.ndarray:
        # Randomly sample
        # You can use some other sampling strategy here, e.g. furthest point sampling
        input_point_size = self.cfg.MODEL.PN2.NUM_INPUT
        if points.shape[1] > input_point_size:
            random_index = np.random.choice(np.arange(points.shape[1]), input_point_size, replace=False)
        else:
            random_index = np.random.choice(np.arange(points.shape[1]), input_point_size, replace=True)

        points = points[:, random_index]
        return points

    def _pre_processing(self, cloud_array: np.ndarray):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud_array.T)
        cloud = CloudPreProcessor(pcd, False)
        cloud.voxelize()
        cloud.remove_outliers()

        # After sampling, points should have size (3, n)
        points = transform_numpy_points(np.asarray(cloud.pcd.points).T, self._REAL2TRAIN)
        points = self.sample_single_cloud(points)

        return points, cloud.pcd

    def eval(self, cloud: np.ndarray) -> dict:
        tic = time.time()
        points, open3d_cloud = self._pre_processing(cloud)

        # Data batch should have size (1, 3, n)
        data_batch = {"scene_points": torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(self._device)}
        tac = time.time()
        self.logger.info("Pre-processing finish, cost {}s".format(tac - tic))

        with torch.no_grad():
            predictions = self.model(data_batch)
            tic = time.time()
            self.logger.info("Prediction finish, cost {}s".format(tic - tac))

        return predictions

    @staticmethod
    def orthogonalization(batch_rotation: np.ndarray, batch_translation) -> np.ndarray:
        x = batch_rotation[:, :, 0]
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        y = batch_rotation[:, :, 1]
        y = y - np.sum(x * y, axis=1, keepdims=True) * x
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        z = np.cross(x, y)

        mat44 = np.tile(np.eye(4), [batch_rotation.shape[0], 1, 1])  # (n, 4, 4)
        mat44[:, :3, :3] = np.stack([x, y, z], axis=2)
        mat44[:, :3, 3] = batch_translation
        return mat44

    def post_processing(self, points_array: np.ndarray, predictions: dict,
                        score_threshold: float, vertical_degree_threshold: float, debug: bool):
        if debug:
            os.makedirs(os.path.join(self._output_path, "debug"), exist_ok=True)

        # Process prediction scores
        all_scores = F.softmax(predictions["score"][0], dim=0).detach().cpu().numpy()
        score_classes = all_scores.shape[0]
        score_value = np.linspace(0, 1, score_classes + 1)[1:][:, np.newaxis]
        all_scores = np.sum(score_value * all_scores, axis=0)

        # Remove low scores
        high_score_index = np.nonzero(all_scores > score_threshold)[0]
        index_high2low = np.argsort(all_scores[high_score_index])[::-1]

        # Remove invalid direction
        rotation = predictions["frame_R"][0].detach().cpu().numpy()[:, index_high2low]
        rotation = rotation.transpose(0, 1).reshape([-1, 3, 3])
        x_direction = -realworld.camera2base[:3, :3] @ self._TRAIN2REAL[:3, :3] @ rotation[:, :, 0].T  # (3, n)
        vertical_degree = np.sum(x_direction.T * self.vertical_direction, axis=1, keepdims=False)
        index_good_direction = np.nonzero(vertical_degree > vertical_degree_threshold)[0]

        # Get all valid index
        valid_index = high_score_index[index_good_direction]
        if points_array.shape[0] == 3:
            points_array = points_array.T  # (n, 3)
        points = points_array[valid_index, :]
        rotation = rotation[index_good_direction, :, :]
        translation = F.softmax(predictions["frame_t"][0][:, valid_index], dim=0)
        translation = translation.transpose(0, 1).detach().cpu().numpy()
        scores = all_scores[valid_index]
        if debug:
            np.savetxt(os.path.join(self._output_path, "debug", "selected_index.txt"), valid_index, fmt="%.4f")
            np.savetxt(os.path.join(self._output_path, "debug", "all_scores.txt"), all_scores, fmt="%.4f")
            np.savetxt(os.path.join(self._output_path, "debug", "top_scores.txt"), scores, fmt="%.4f")
            np.savetxt(os.path.join(self._output_path, "debug", "top_rotation.txt"), rotation.reshape([-1, 9]),
                       fmt="%.4f")
            np.savetxt(os.path.join(self._output_path, "debug", "top_translation.txt"), translation, fmt="%.4f")

        # Process the rotation and translation to camera frame
        t_score = np.array([0.08, 0.06, 0.04, 0.02])[np.newaxis, :]
        global_translation = - (translation * t_score).sum(1, keepdims=True) * rotation[:, :, 0] + points
        global_mat44 = self.orthogonalization(rotation, global_translation)
        global_mat44 = np.matmul(self._TRAIN2REAL[np.newaxis, :, :], global_mat44)
        if debug:
            np.savetxt(os.path.join(self._output_path, "debug", "processed_mat44.txt"), global_mat44.reshape([-1, 16]),
                       fmt="%.4f")

        return global_mat44, scores

    def detect(self, cloud_array: np.ndarray, cloud_mask=Optional[np.ndarray], num_selected=5, score_threshold=0.7,
               verticalness_threshold=0.2, collision_check=True, debug=True):
        start = time.time()
        assert cloud_array.ndim == 2, "Evaluation mode do not support batch, input should have shape (n, 3) or (3, n)."
        assert (cloud_array.shape[0] == 3 or cloud_array.shape[
            1] == 3), "input should have shape (n, 3) or (3, n), but given {}".format(cloud_array.shape)
        if cloud_array.shape[1] == 3:
            cloud_array = cloud_array.T  # (3, n)

        if isinstance(cloud_mask, np.ndarray):
            target_cloud = cloud_array[:, cloud_mask]
        else:
            target_cloud = cloud_array.copy()

        points, pcd = self._pre_processing(target_cloud)

        # Data batch should have size (1, 3, n)
        data_batch = {"scene_points": torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(self._device)}
        tac = time.time()
        self.logger.info("Pre-processing finish, cost ***{0:.4f}s***".format(tac - start))

        with torch.no_grad():
            predictions = self.model(data_batch)
            tic = time.time()
            self.logger.info("Prediction finish, cost ***{0:.4f}s***".format(tic - tac))

        poses, scores = self.post_processing(points, predictions, score_threshold, verticalness_threshold, debug)

        # Collision check
        if collision_check:
            valid_indices = []
            torch_poses = torch_batch_transformation_inv(torch.tensor(poses, device=self._device, dtype=torch.float32))
            reference_cloud = open3d.geometry.PointCloud()
            reference_cloud.points = open3d.utility.Vector3dVector(cloud_array.T)
            evaluator = CloudCollisionChecker(reference_cloud)
            for i in range(torch_poses.shape[0]):
                if evaluator.view_non_collision(torch_poses[i, :, :]):
                    valid_indices.append(i)

            num_detected_grasp = poses.shape[0]
            if not valid_indices:
                self.logger.info("No valid grasp found after collision checking")
            valid_indices = np.array(valid_indices)
            poses = poses[valid_indices, :, :]
            scores = scores[valid_indices]
            self.logger.info("{}/{} grasp poses is removed during view collision checking".format(
                num_detected_grasp - len(valid_indices), num_detected_grasp))
            self.logger.info("Collision check finish, cost ***{0:.4f}s***".format(time.time() - tic))

        # Importance Sampling
        if poses.shape[0] > num_selected:
            tic = time.time()
            scores_cum = np.cumsum(np.exp(5 * scores))
            random_score = np.sort(np.random.rand(num_selected)) * scores_cum[-1]
            sampling_indices = []
            index = 0
            for i in range(num_selected):
                score_target = random_score[i]
                while scores_cum[index] < score_target:
                    index += 1
                sampling_indices.append(index)
            sampling_indices = np.array(sampling_indices)
            poses = poses[sampling_indices, :, :]
            scores = scores[sampling_indices]
            self.logger.info("Importance sampling finish, cost ***{0:.4f}s***".format(time.time() - tic))

        self.logger.info("Overall time cost of grasp detection: ***{0:.4f}s***".format(time.time() - start))
        return poses, scores
