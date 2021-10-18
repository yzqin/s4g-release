import numpy as np
import torch
from .gripper_config import *


def _get_local_search_to_local(LOCAL_TO_LOCAL_SEARCH):
    NUMPY_LOCAL_TO_LOCAL_SEARCH = LOCAL_TO_LOCAL_SEARCH.cpu().numpy()
    NUMPY_LOCAL_SEARCH_TO_LOCAL = np.zeros([LOCAL_TO_LOCAL_SEARCH.shape[0], 4, 4])
    for j in range(LOCAL_TO_LOCAL_SEARCH.shape[0]):
        NUMPY_LOCAL_SEARCH_TO_LOCAL[j, :, :] = np.linalg.inv(NUMPY_LOCAL_TO_LOCAL_SEARCH[j, :, :])
    return NUMPY_LOCAL_SEARCH_TO_LOCAL


# Point Cloud Pre-processing
TABLE_HEIGHT = 0.75
SAMPLE_REGION = TABLE_HEIGHT + 0.015
# Workspace should be (6,): low_x, high_x, low_y, high_y, low_z, high_z
WORKSPACE = [-0.40, 0.40, -0.4, 0.4, TABLE_HEIGHT - 0.001, TABLE_HEIGHT + 0.45]
WORKSPACE_SCENE = [-0.40, 0.40, -0.35, 0.35, TABLE_HEIGHT - 0.001, TABLE_HEIGHT + 0.45]
VOXEL_SIZE = 0.005
NUM_POINTS_THRESHOLD = 32
RADIUS_THRESHOLD = 0.02

# Scene Point Cloud
SCENE_MULTIPLE = 8  # The density of points in scene over view cloud

# Normal Estimation
NORMAL_RADIUS = 0.01
NORMAL_MAX_NN = 30

# Local Frame Search
# LENGTH_SEARCH = [-0.08, -0.06, -0.04, -0.02]
LENGTH_SEARCH = [-0.55, -0.35, -0.15]
THICKNESS_SEARCH = [0]
THETA_SEARCH = list(range(-90, 90, 15))
CURVATURE_RADIUS = 0.01
BACK_COLLISION_THRESHOLD = 10 * np.sqrt(
    SCENE_MULTIPLE)  # if more than this number of points exist behind the back of hand, grasp fail
BACK_COLLISION_MARGIN = 0.0  # points that collide with back hand within this range will not be detected
FINGER_COLLISION_THRESHOLD = 10
CLOSE_REGION_MIN_POINTS = 50
for i in range(len(THETA_SEARCH)):
    THETA_SEARCH[i] /= 57.29578

# Antipodal Grasp
NEIGHBOR_DEPTH = torch.tensor(0.005).to(device)

# GPD Projection Configuration
GRASP_NUM = 600
PROJECTION_RESOLUTION = 60
PROJECTION_MARGIN = 1

INDEX_TO_ARRAY = []
GRASP_PER_LENGTH = len(THETA_SEARCH) * len(THICKNESS_SEARCH)
for length in LENGTH_SEARCH:
    for theta in THETA_SEARCH:
        for height in THICKNESS_SEARCH:
            INDEX_TO_ARRAY.append((length, theta, height))

_INDEX_TO_ARRAY_TORCH = torch.tensor(INDEX_TO_ARRAY).to(device)
LOCAL_TO_LOCAL_SEARCH = torch.eye(4).unsqueeze(0).expand(size=(len(INDEX_TO_ARRAY), 4, 4))
LOCAL_TO_LOCAL_SEARCH = LOCAL_TO_LOCAL_SEARCH.contiguous()
LOCAL_TO_LOCAL_SEARCH[:, 0, 3] = -_INDEX_TO_ARRAY_TORCH[:, 0]
LOCAL_TO_LOCAL_SEARCH[:, 2, 3] = -_INDEX_TO_ARRAY_TORCH[:, 2]
LOCAL_TO_LOCAL_SEARCH[:, 1, 1] = torch.cos(_INDEX_TO_ARRAY_TORCH[:, 1])
LOCAL_TO_LOCAL_SEARCH[:, 2, 2] = torch.cos(_INDEX_TO_ARRAY_TORCH[:, 1])
LOCAL_TO_LOCAL_SEARCH[:, 1, 2] = torch.sin(_INDEX_TO_ARRAY_TORCH[:, 1])
LOCAL_TO_LOCAL_SEARCH[:, 2, 1] = -torch.sin(_INDEX_TO_ARRAY_TORCH[:, 1])
LOCAL_TO_LOCAL_SEARCH = LOCAL_TO_LOCAL_SEARCH.to(device)
NUMPY_LOCAL_SEARCH_TO_LOCAL = _get_local_search_to_local(LOCAL_TO_LOCAL_SEARCH)
TORCH_LOCAL_SEARCH_TO_LOCAL = torch.inverse(LOCAL_TO_LOCAL_SEARCH)

# Table collision check
LOCAL_SEARCH_TO_LOCAL = torch.inverse(LOCAL_TO_LOCAL_SEARCH).contiguous()
TABLE_COLLISION_OFFSET = 0.005
