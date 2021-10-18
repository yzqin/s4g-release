import numpy as np

"""
********************************************************************
Modify the parameters in this block according to your hardware setup
********************************************************************
"""
# (length, width), the definition of length and width should be consistent with the camera2table matrix
table_size = ()

# The maximum height you think you cluttered scene can achieve above table surface plane
max_height = 0.4

# Transformation matrix from camera frame to table center frame. This can be a estimated value, precise value not needed
# After transformation, x-axis should be the direction of length of table, y-axis should be width direction
camera2table = np.array([[-0.00377177, 0.54720216, -0.83699198, 0.766],
                         [0.99981506, -0.01372054, -0.01347562, -0.276],
                         [-0.01885787, -0.83688801, -0.54704921, 0.62],
                         [0., 0., 0., 1.]])

camera2base = np.array([[-0.00377177, 0.54720216, -0.83699198, 0.766],
                        [0.99981506, -0.01372054, -0.01347562, -0.276],
                        [-0.01885787, -0.83688801, -0.54704921, 0.62],
                        [0., 0., 0., 1.]])
"""
**********************************************
Derived parameters below, please do not modify.
**********************************************
"""
workspace = [-0.4, 0.4, -0.6, 0.1, -0.08, 0.5]
target_space = [-0.4, 0.4, -0.6, -0.15, -0.06, 0.4]
table2camera = np.linalg.inv(camera2table)
base2camera = np.linalg.inv(camera2base)
