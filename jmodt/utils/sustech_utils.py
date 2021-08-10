import numpy as np
from numba import njit

from jmodt.utils.pcd_utils import PointCloud


@njit
def euler_angle_to_rotate_matrix(theta):
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
    return R


def psr_to_corners(psr: dict):
    """
    :param psr: a dict with keys: position, scale and rotation
    :return: (8, 3)
    """
    p = np.array((psr['position']['x'], psr['position']['y'], psr['position']['z']))
    s = np.array((psr['scale']['x'], psr['scale']['y'], psr['scale']['z']))
    r = np.array((psr['rotation']['x'], psr['rotation']['y'], psr['rotation']['z']))
    return psr_to_corners_jit(p, s, r)


@njit
def psr_to_corners_jit(p, s, r):
    rotate_matrix = euler_angle_to_rotate_matrix(r)
    p = p.reshape((-1, 1))
    trans_matrix = np.concatenate((rotate_matrix, p), axis=-1)
    x = s[0] / 2
    y = s[1] / 2
    z = s[2] / 2
    local_coord = np.array([
        x, y, -z, 1, x, -y, -z, 1,  # front-left-bottom, front-right-bottom
        x, -y, z, 1, x, y, z, 1,  # front-right-top,   front-left-top
        -x, y, -z, 1, -x, -y, -z, 1,  # rear-left-bottom, rear-right-bottom
        -x, -y, z, 1, -x, y, z, 1,  # rear-right-top,   rear-left-top
    ]).reshape((-1, 4))
    world_coord = np.transpose(np.dot(trans_matrix, np.transpose(local_coord)))
    return world_coord


def psr_to_kitti_box(psr):
    """
    :param psr: a dict with keys: position, scale and rotation
    :return: (7,) [x,y,z,h,w,l,ry]
    """
    box = np.array((
        -psr['position']['x'],
        psr['scale']['z'] / 2 - psr['position']['z'],
        -psr['position']['y'],
        psr['scale']['z'],  # h
        psr['scale']['y'],  # w
        psr['scale']['x'],  # l
        -psr['rotation']['z'] + (np.pi if psr['rotation']['z'] > 0 else - np.pi)
    ))
    return box


def psr_to_kitti_alpha(psr):
    x = -psr['position']['x']
    z = -psr['position']['y']
    ry = -psr['rotation']['z'] + (np.pi if psr['rotation']['z'] > 0 else - np.pi)
    beta = np.arctan2(z, x)
    alpha = -np.sign(beta) * np.pi / 2 + beta + ry
    return alpha


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


# @njit (matmul not supported)
def proj_box_to_img(corners_3d, extrinsic, intrinsic, img_shape):
    assert corners_3d.shape[1] == 8 and corners_3d.shape[2] == 3
    corners_3d = np.concatenate((
        corners_3d, np.ones((len(corners_3d), 8, 1), dtype=np.float32)
    ), axis=-1)  # (N, 8, 4)
    corners_3d = np.transpose(corners_3d, (0, 2, 1))  # (N, 4, 8)
    corners_2d = np.matmul(extrinsic, corners_3d)  # (N, 3, 8)
    corners_2d = np.matmul(intrinsic, corners_2d)  # (N, 3, 8)
    depth_mask = corners_2d[:, 2, :] >= 0  # (N, 8)
    depth_mask = np.all(depth_mask, axis=1)  # (N) all depth should be >= 0
    corners_2d = np.transpose(corners_2d, (1, 2, 0))  # (3, 8, N)
    corners_2d = corners_2d[:2] / corners_2d[2]  # (2, 8, N)
    corners_2d = np.transpose(corners_2d, (2, 1, 0))  # (N, 8, 2) [W, H]
    row_mask = np.logical_and(corners_2d[:, :, 0] >= 0, corners_2d[:, :, 0] < img_shape[1])  # 0 <= x < W
    col_mask = np.logical_and(corners_2d[:, :, 1] >= 0, corners_2d[:, :, 1] < img_shape[0])  # 0 <= y < H
    xy_mask = np.logical_and(row_mask, col_mask)  # (N, 8)
    xy_mask = np.any(xy_mask, axis=1)  # (N) at least one corner should be in image

    valid_mask = np.logical_and(depth_mask, xy_mask)  # (N)
    corners_2d = corners_2d[valid_mask]  # (M, 8, 2)

    min_xy = np.min(corners_2d, axis=1)  # (M, 2)
    max_xy = np.max(corners_2d, axis=1)  # (M, 2)

    boxes = np.hstack((min_xy, max_xy))  # (M, 4) [x1, y1, x2, y2]

    img_boxes = np.hstack(
        (np.clip(boxes[:, 0], 0, img_shape[1] - 1).reshape(-1, 1),
         np.clip(boxes[:, 1], 0, img_shape[0] - 1).reshape(-1, 1),
         np.clip(boxes[:, 2], 0, img_shape[1] - 1).reshape(-1, 1),
         np.clip(boxes[:, 3], 0, img_shape[0] - 1).reshape(-1, 1))
    )  # (M, 4) [x1, y1, x2, y2]

    area1 = (img_boxes[:, 2] - img_boxes[:, 0]) * (img_boxes[:, 3] - img_boxes[:, 1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    truncation = (area2 - area1) / area2

    return img_boxes, valid_mask, truncation


def pcd_to_bin(in_path, out_path):
    pc = PointCloud(in_path)
    pts = pc.data
    pts.tofile(out_path)
