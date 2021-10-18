import os

import numpy as np
import open3d
import time
import torch
import torch.nn as nn
from grasp_proposal.cloud_processor.cloud_processor import CloudPreProcessor
from grasp_proposal.configs.yacs_config import load_cfg_from_file
from grasp_proposal.network_models.models.build_model import build_model
from grasp_proposal.utils.checkpoint import CheckPointer
from grasp_proposal.utils.file_logger_cls import loggin_to_file
from grasp_proposal.utils.grasp_visualizer import GraspVisualizer
from grasp_proposal.utils.logger import setup_logger, MetricLogger


def load_static_data_batch():
    single_training_data = np.load("/home/sim/project/s4g/2638_view_0.p", allow_pickle=True)
    cloud_array = single_training_data["point_cloud"]
    cloud = CloudPreProcessor(open3d.geometry.PointCloud(open3d.utility.Vector3dVector(cloud_array.T)), False)

    # do not filter workspace here since training data
    cloud.voxelize()
    cloud.remove_outliers()
    points = np.asarray(cloud.pcd.points)
    if points.shape[0] > 25600:
        random_index = np.random.choice(np.arange(points.shape[0]), 25600, replace=False)
    else:
        random_index = np.random.choice(np.arange(points.shape[0]), 25600, replace=True)

    points = points[random_index, :]
    data_batch = {"scene_points": torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(1, 2)}
    return data_batch, cloud.pcd


def main():
    cfg_path = "./configs/curvature_model.yaml"
    cfg = load_cfg_from_file(cfg_path)
    cfg.TEST.WEIGHT = cfg.TEST.WEIGHT.replace("${PROJECT_HOME}", os.path.join(os.path.dirname(__file__), "../"))
    cfg.freeze()
    assert cfg.TEST.BATCH_SIZE == 1

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("S4G", output_dir, "unit_test")
    logger.info("Using {} of GPUs".format(torch.cuda.device_count()))
    logger.info("Load config file from {}".format(cfg_path))
    logger.debug("Running with config \n {}".format(cfg))

    model, _, _ = build_model(cfg)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.device_count() == 1:
        model = model.cuda()

    trained_model_path = output_dir
    check_pointer = CheckPointer(model, save_dir=trained_model_path, logger=logger)
    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        check_pointer.load(weight_path, resume=False)
    else:
        check_pointer.load(None, resume=True)

    # Test
    model.eval()
    meters = MetricLogger(delimiter="  ")
    data_batch, pcd = load_static_data_batch()
    tic = time.time()
    with torch.no_grad():
        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        tac = time.time()
        data_time = tac - tic
        predictions = model(data_batch)
        tic = time.time()
        batch_time = tic - tac
        with open("inference_time_{}.txt".format("ours"), "a+") as f:
            f.write("{:.4f}\n".format(batch_time * 1000.0))
        meters.update(time=batch_time, data=data_time)

        logger.info(meters.delimiter.join(["{meters}", ]).format(meters=str(meters), ))

        top_poses, score = loggin_to_file(data_batch, predictions, 0, output_dir, prefix="test", with_label=False)
        visualizer = GraspVisualizer(pcd)
        visualizer.add_multiple_poses(top_poses)
        visualizer.visualize()


if __name__ == '__main__':
    main()
    print("Finish")
