import json
import os

import numpy as np
import open3d
from PIL import Image
from matplotlib import cm

from jmodt.utils.sustech_utils import proj_lidar_to_img
from line_mesh import LineMesh
from kitti_viewer import create_video


def get_lidar(file_path):
    assert os.path.exists(file_path)
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def get_image(file_path):
    assert os.path.exists(file_path)
    img = Image.open(file_path).convert('RGB')
    img = np.array(img).astype(np.float32)
    return img


def get_calib(file_path):
    with open(file_path, 'r') as f:
        calib = json.load(f)
    extrinsic = np.array(calib['extrinsic'], dtype=np.float32).reshape(4, 4)[:3, :]
    intrinsic = np.array(calib['intrinsic'], dtype=np.float32).reshape(3, 3)
    return extrinsic, intrinsic


def get_colors_from_pc(pc, img, extrinsic, intrinsic):
    pts_xy, valid_mask = proj_lidar_to_img(pc, extrinsic, intrinsic, (1536, 2048))
    pts_xy = np.trunc(pts_xy).astype(int)
    colors = np.array([img[h, w] for w, h in pts_xy], dtype=np.float32)
    colors /= 255
    return colors, valid_mask


def sample_points(pc):
    npoints = 16384
    # generate inputs
    if npoints < len(pc):
        # keep far points, sample near points
        pts_depth = pc[:, 0]
        pts_near_flag = pts_depth < 40.0
        far_indices_choice = np.argwhere(pts_near_flag == 0).flatten()
        near_indices = np.argwhere(pts_near_flag == 1).flatten()
        near_indices_choice = np.random.choice(near_indices, npoints - len(far_indices_choice), replace=False)
        choice = np.concatenate((near_indices_choice, far_indices_choice), axis=0) \
            if len(far_indices_choice) > 0 else near_indices_choice
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(pc), dtype=np.int32)
        while npoints > len(choice):
            extra_choice = np.random.choice(choice, npoints % len(choice), replace=False)
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
    return choice


def get_pc_in_range(pc, xy_dist=70, z_range=(-3, 1)):
    horizontal_distances = np.sqrt(np.square(pc[:, 0]) + np.square(pc[:, 1]))
    xy_mask = horizontal_distances <= xy_dist
    z_mask = np.logical_and(pc[:, 2] >= z_range[0], pc[:, 2] <= z_range[1])
    pc = pc[np.logical_and(xy_mask, z_mask)]
    return pc


def visualize(root_dir, viewpoint_file, output_dir, label_dir, save_screenshot=False):
    frames = sorted([f[:-4] for f in os.listdir(os.path.join(root_dir, 'lidar_bin'))])

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.set_full_screen(True)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.poll_events()
    vis.update_renderer()

    if not save_screenshot:
        frame = frames[0]
        lidar_path = os.path.join(root_dir, 'lidar_bin', f'{frame}.bin')
        lidar_data = get_lidar(lidar_path)[:, :3]
        lidar_data = get_pc_in_range(lidar_data)

        all_points = []
        all_colors = []
        for camera in ['rear', 'rear_left', 'rear_right', 'front', 'front_left', 'front_right']:
            image_path = os.path.join(root_dir, f'camera/{camera}/{frame}.jpg')
            calib_path = os.path.join(root_dir, f'calib/camera/{camera}.json')
            extrinsic, intrinsic = get_calib(calib_path)
            image_data = get_image(image_path)
            colors, valid_mask = get_colors_from_pc(lidar_data, image_data, extrinsic, intrinsic)
            all_points.append(lidar_data[valid_mask])
            all_colors.append(colors)
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)

        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(all_points)
        pc.colors = open3d.utility.Vector3dVector(all_colors)

        vis.add_geometry(pc)
        vis.run()  # user changes the view and press "q" to terminate
        viewpoint_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(viewpoint_file, viewpoint_param)

    else:
        os.makedirs(output_dir, exist_ok=True)

        assert os.path.exists(viewpoint_file)
        viewpoint_param = open3d.io.read_pinhole_camera_parameters(viewpoint_file)

        color_map = cm.get_cmap('gist_rainbow', 2000)
        color_indices = np.arange(2000)
        np.random.shuffle(color_indices)

        for i, frame in enumerate(frames):
            lidar_path = os.path.join(root_dir, 'lidar_bin', f'{frame}.bin')
            lidar_data = get_lidar(lidar_path)[:, :3]
            lidar_data = get_pc_in_range(lidar_data)

            try:
                all_points = []
                all_colors = []
                for camera in ['rear', 'rear_left', 'rear_right', 'front', 'front_left', 'front_right']:
                    image_path = os.path.join(root_dir, f'camera/{camera}/{frame}.jpg')
                    calib_path = os.path.join(root_dir, f'calib/camera/{camera}.json')
                    extrinsic, intrinsic = get_calib(calib_path)
                    image_data = get_image(image_path)
                    colors, valid_mask = get_colors_from_pc(lidar_data, image_data, extrinsic, intrinsic)
                    all_points.append(lidar_data[valid_mask])
                    all_colors.append(colors)
                all_points = np.vstack(all_points)
                all_colors = np.vstack(all_colors)
            except:
                continue

            pc = open3d.geometry.PointCloud()
            pc.points = open3d.utility.Vector3dVector(all_points)
            pc.colors = open3d.utility.Vector3dVector(all_colors)
            vis.add_geometry(pc)

            # points[0] = center_ - x_axis - y_axis - z_axis;
            # points[1] = center_ + x_axis - y_axis - z_axis;
            # points[2] = center_ - x_axis + y_axis - z_axis;
            # points[3] = center_ - x_axis - y_axis + z_axis;
            # points[4] = center_ + x_axis + y_axis + z_axis;
            # points[5] = center_ - x_axis + y_axis + z_axis;
            # points[6] = center_ + x_axis - y_axis + z_axis;
            # points[7] = center_ + x_axis + y_axis - z_axis;
            box_lines = [[0, 1], [0, 3], [3, 6], [1, 6], [0, 2], [2, 5], [3, 5], [1, 7],
                         [7, 4], [4, 6], [4, 5], [2, 7],
                         [1, 4], [6, 7]  # front
                         ]
            label_path = os.path.join(label_dir, f'{frame}.json')

            if not os.path.exists(label_path):
                data = []
            else:
                with open(label_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.decoder.JSONDecodeError:
                        data = []
            for label in data:
                psr = label['psr']
                center = np.array(
                    [float(psr['position']['x']), float(psr['position']['y']), float(psr['position']['z'])])
                R = open3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array(
                        [float(psr['rotation']['x']), float(psr['rotation']['y']), float(psr['rotation']['z'])]
                    ))
                extent = np.array([float(psr['scale']['x']), float(psr['scale']['y']), float(psr['scale']['z'])])
                box = open3d.geometry.OrientedBoundingBox(center, R, extent)
                points = box.get_box_points()
                color = color_map(color_indices[int(label['obj_id'])])[:3]
                line_mesh1 = LineMesh(points, box_lines, color, 0.1)
                box = line_mesh1.cylinder_segments
                for line in box:
                    vis.add_geometry(line)

            vis.get_view_control().convert_from_pinhole_camera_parameters(viewpoint_param)
            vis.poll_events()
            vis.update_renderer()

            vis.capture_screen_image(os.path.join(output_dir, f'{frame}.png'))
            vis.clear_geometries()

        vis.destroy_window()


if __name__ == '__main__':
    visualize(root_dir='/media/kemo/Kemo/sustech-data/2021-06-25-06-56-55/dataset_10hz/',
              viewpoint_file='viewpoint_all.json',
              output_dir=f'/media/kemo/Kemo/sustech-data/2021-06-25-06-56-55/dataset_10hz/output/all',
              label_dir='/home/kemo/Github/JMODT/output/trk/2021-06-25/all/',
              # label_dir = '/media/kemo/Kemo/sustech-data/2021-06-25-06-56-55/dataset/label'
              save_screenshot=True)
    create_video(f'/media/kemo/Kemo/sustech-data/2021-06-25-06-56-55/dataset_10hz/output/all',
                 '2021-06-25_10hz.avi',
                 (1920, 1080),
                 10)
