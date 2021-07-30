import numpy as np
from numba import njit

from jmodt.utils.pcd_utils import PointCloud


@njit
def euler_angle_to_rotate_matrix(theta, t=None):
    # Calculate rotation about x axis
    R_x = np.array([
        [1, 0., 0.],
        [0., np.cos(theta[0]), -np.sin(theta[0])],
        [0., np.sin(theta[0]), np.cos(theta[0])]
    ])
    # Calculate rotation about y axis
    R_y = np.array([
        [np.cos(theta[1]), 0., np.sin(theta[1])],
        [0., 1., 0.],
        [-np.sin(theta[1]), 0., np.cos(theta[1])]
    ])
    # Calculate rotation about z axis
    R_z = np.array([
        [np.cos(theta[2]), -np.sin(theta[2]), 0.],
        [np.sin(theta[2]), np.cos(theta[2]), 0.],
        [0., 0., 1.]])
    R = np.dot(R_x, np.dot(R_y, R_z))

    # Add translation if available
    if t is not None:
        t = t.reshape((-1, 1))
        R = np.concatenate((R, t), axis=-1)
        R = np.concatenate((R, np.array([0, 0, 0, 1]).reshape((1, -1))), axis=0)
    return R


@njit
def psr_to_corners(p, s, r):
    """
    :param p: position
    :param s: scale
    :param r: rotation
    :return: (8, 3)
    """
    trans_matrix = euler_angle_to_rotate_matrix(r, p)
    x = s[0] / 2
    y = s[1] / 2
    z = s[2] / 2
    local_coord = np.array([
        x, y, -z, 1, x, -y, -z, 1,  # front-left-bottom, front-right-bottom
        x, -y, z, 1, x, y, z, 1,  # front-right-top,   front-left-top
        -x, y, -z, 1, -x, -y, -z, 1,  # rear-left-bottom, rear-right-bottom
        -x, -y, z, 1, -x, y, z, 1,  # rear-right-top,   rear-left-top
    ]).reshape((-1, 4))
    world_coord = np.dot(trans_matrix, np.transpose(local_coord))
    return world_coord


@njit
def proj_lidar_to_img(pts, extrinsic, intrinsic, img_shape):
    """
    Project lidar points to image, the points outside the image are discarded.
    :param pts: (N, 3) xyz data
    :param extrinsic: (3, 4)
    :param intrinsic: (3, 3)
    :param img_shape: H, W
    :return: pts_2d: (M, 2), valid_mask: (N,)
    """
    assert pts.shape[1] == 3
    pts = np.hstack((pts, np.ones((len(pts), 1), dtype=np.float32)))
    pts = np.transpose(pts)  # (4, N)
    pts_2d = np.dot(extrinsic, pts)  # (3, N)
    # rect matrix shall be applied here, for kitti
    pts_2d = np.dot(intrinsic, pts_2d)  # (3, N)
    depth_mask = pts_2d[2, :] >= 0
    pts_2d = np.transpose(pts_2d[0:2, :] / pts_2d[2, :])  # (N, 2), W, H
    row_mask = np.logical_and(pts_2d[:, 0] >= 0, pts_2d[:, 0] < img_shape[1])  # 0 <= x < W
    col_mask = np.logical_and(pts_2d[:, 1] >= 0, pts_2d[:, 1] < img_shape[0])  # 0 <= y < H
    xy_mask = np.logical_and(row_mask, col_mask)
    valid_mask = np.logical_and(xy_mask, depth_mask)  # depth >= 0
    pts_2d = pts_2d[valid_mask]

    return pts_2d, valid_mask


def pcd_to_bin(in_path, out_path):
    pc = PointCloud(in_path)
    pts = pc.data
    pts.tofile(out_path)
