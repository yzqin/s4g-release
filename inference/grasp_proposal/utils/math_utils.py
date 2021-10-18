import numpy as np
import torch


def transformation_inv(transformation: np.array):
    """
    This function is used to substitute np.linalg.inv when input strictly belongs to SE(3)
    :param transformation:
    """
    assert transformation.shape == (4, 4), "Transformation matrix should have dimension (4, 4), get {} instead".format(
        transformation.shape)

    result = np.eye(4)
    rotation_inv = transformation[:3, :3].T
    result[:3, :3] = rotation_inv
    result[:3, 3:4] = -rotation_inv @ transformation[:3, 3:4]
    return result


def transform_numpy_points(cloud_array: np.ndarray, transformation_matrix):
    assert (cloud_array.shape[0] == 3 and cloud_array.ndim == 2)
    homo_array = np.concatenate([cloud_array, np.ones([1, cloud_array.shape[1]])], axis=0)
    cloud_array = transformation_matrix @ homo_array
    return cloud_array[:3, :]


def torch_batch_transformation_inv(transformation: torch.Tensor) -> torch.Tensor:
    """
    This function is used to substitute the torch.inv when input strictly belongs to batch SE(3)
    :rtype: torch.Tensor
    :param transformation:
    """
    assert transformation.ndim == 3, "Batched transformation should have dimension (n ,4, 4), get {} instead".format(
        transformation.shape)
    device = transformation.device
    dtype = transformation.dtype
    result = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(transformation.shape[0], 4, 4).contiguous()
    result[:, :3, :3] = transformation[:, :3, :3].transpose(1, 2).clone()
    result[:, :3, 3:] = torch.bmm(-transformation[:, :3, :3].transpose(1, 2), transformation[:, :3, 3:])
    return result
