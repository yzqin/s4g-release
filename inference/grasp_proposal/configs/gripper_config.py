import torch
import numpy as np

"""
**********************************************************************************************
Gripper parameters and configuration, specify below based on the size of your parallel gripper
**********************************************************************************************
"""
HALF_BOTTOM_WIDTH = 0.057
BOTTOM_LENGTH = 0.16
FINGER_WIDTH = 0.023
HALF_HAND_THICKNESS = 0.012
FINGER_LENGTH = 0.09

"""
**********************************************
Derived parameters below, please do not modify
**********************************************
"""
HAND_LENGTH = BOTTOM_LENGTH + BOTTOM_LENGTH
HALF_BOTTOM_SPACE = HALF_BOTTOM_WIDTH - FINGER_WIDTH

GRIPPER_BOUND = np.ones([4, 8])
i = 0
for x in [FINGER_LENGTH, -BOTTOM_LENGTH]:
    for y in [HALF_BOTTOM_WIDTH, -HALF_BOTTOM_WIDTH]:
        for z in [HALF_HAND_THICKNESS, -HALF_HAND_THICKNESS]:
            GRIPPER_BOUND[0:3, i] = [x, y, z]
            i += 1
del i

"""
***********************************************
Pre-computed torch variable cache for later use
***********************************************
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_GRIPPER_BOUND = torch.tensor(GRIPPER_BOUND, device=device).float()
