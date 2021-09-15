import os

import cv2
import numpy as np
import open3d
import tqdm
from PIL import Image
from matplotlib import cm

from jmodt.detection.datasets.kitti_dataset import KittiDataset
from jmodt.utils.calibration import Calibration
from line_mesh import LineMesh


class KittiSequenceViewer:
    def __init__(self, root_dir, seq, output_dir, viewpoint_file, label_dir,
                 classes=('Car', 'Van'), radius=0.1, trajectory_len=15):
        self.root_dir = root_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.viewpoint_file = viewpoint_file
        self.radius = radius
        self.trajectory_length = trajectory_len
        self.box_lines = [[0, 1], [0, 3], [3, 6], [1, 6], [0, 2], [2, 5], [3, 5], [1, 7],
                          [7, 4], [4, 6], [4, 5], [2, 7]]

        # load data
        self.seq = str(int(seq)).zfill(4)
        self.calib = Calibration(os.path.join(self.root_dir, 'calib', f'{self.seq}.txt'))
        label_file = os.path.join(label_dir, f'{self.seq}.txt')
        self.box_data, self.seq_id_dict, all_id_list = self.get_labels(label_file, classes)

        # get all the frames of the sequence
        lidar_dir = os.path.join(self.root_dir, 'velodyne', self.seq)
        lidar_files = os.listdir(lidar_dir)
        self.frames = [lidar_file[:-4] for lidar_file in lidar_files]
        self.frames.sort()

        # create color map
        self.id_color_dict = dict.fromkeys(set(all_id_list))
        color_indices = np.arange(len(self.id_color_dict))
        np.random.shuffle(color_indices)
        for i, tracking_id in enumerate(self.id_color_dict):
            self.id_color_dict[tracking_id] = color_indices[i]
        self.color_map = cm.get_cmap('gist_rainbow', len(self.id_color_dict))

        # init visualizer
        self._vis = open3d.visualization.Visualizer()
        self._vis.create_window()
        self._vis.set_full_screen(True)
        self._vis.get_render_option().background_color = np.asarray([0, 0, 0])

    @staticmethod
    def get_image_rgb_with_normal(img_file):
        assert os.path.exists(img_file)
        im = Image.open(img_file).convert('RGB')
        im = np.array(im).astype(np.float)
        im /= 255.0

        return im  # (H,W,3) RGB mode

    @staticmethod
    def get_lidar(file_path):
        assert os.path.exists(file_path)
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def get_labels(file_path, classes):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        box_data = {}
        seq_id_dict = {}
        all_id_list = []
        for line in lines:
            label = line.split()
            if label[2] in classes:
                center = np.array([float(label[13]), float(label[14]) - float(label[11]) / 2, float(label[15])])
                R = open3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0.0, float(label[16]), 0.0]))
                extent = np.array([float(label[12]), float(label[10]), float(label[11])])  # l, h, w
                frame = int(label[0])
                if frame in box_data:
                    box_data[frame].append((center, R, extent))
                else:
                    box_data[frame] = [(center, R, extent)]
                if frame in seq_id_dict:
                    seq_id_dict[frame].append(int(label[1]))
                else:
                    seq_id_dict[frame] = [int(label[1])]
                all_id_list.append(int(label[1]))
        return box_data, seq_id_dict, all_id_list

    def get_painted_point_cloud(self, frame):
        img_path = os.path.join(self.root_dir, 'image_02', self.seq, f'{frame}.png')
        lidar_path = os.path.join(self.root_dir, 'velodyne', self.seq, f'{frame}.bin')
        img = self.get_image_rgb_with_normal(img_path)
        pts_lidar = self.get_lidar(lidar_path)

        # get valid point (projected points should be in image)
        pts_rect = self.calib.lidar_to_rect(pts_lidar[:, 0:3])

        pts_img, pts_rect_depth = self.calib.rect_to_img(pts_rect)
        pts_valid_flag = KittiDataset.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img.shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_xy = pts_img[pts_valid_flag]  # (N, W, H)
        colors = np.array([img[round(h) - 1, round(w) - 1] for w, h in pts_xy])
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pts_rect)
        pc.colors = open3d.utility.Vector3dVector(colors)
        return pc

    def create_boxes_and_trajectories(self, frame):
        frame = int(frame)
        boxes = []
        trajectories = []
        if frame in self.box_data:
            box_frame_data = self.box_data[frame]
            tracking_frame_ids = self.seq_id_dict[frame]
            track_dict = {}
            for i, label in enumerate(box_frame_data):
                box = open3d.geometry.OrientedBoundingBox(center=label[0], R=label[1], extent=label[2])
                points = box.get_box_points()
                tracking_id = tracking_frame_ids[i]
                color = self.color_map(self.id_color_dict[tracking_id])[:3]
                line_mesh1 = LineMesh(points, self.box_lines, color, self.radius)
                boxes.append(line_mesh1.cylinder_segments)
                track_dict[tracking_id] = [label[0]]
            # get box centers from previous frames
            min_frame = max(0, frame - self.trajectory_length)
            for cur_frame in range(frame - 1, min_frame - 1, -1):
                if cur_frame in self.box_data:
                    box_frame_data = self.box_data[cur_frame]
                    tracking_frame_ids = self.seq_id_dict[cur_frame]
                    for i, label in enumerate(box_frame_data):
                        tracking_id = tracking_frame_ids[i]
                        if tracking_id in track_dict:
                            track_dict[tracking_id].append(label[0])
            for tracking_id, centers in track_dict.items():
                if len(centers) < 2:
                    continue
                centers = np.vstack(centers)
                lines = [[q, q + 1] for q in range(len(centers) - 1)]
                color = self.color_map(self.id_color_dict[tracking_id])[:3]
                line_mesh1 = LineMesh(centers, lines, color, self.radius)
                trajectories.append(line_mesh1.cylinder_segments)

        for box in boxes:
            for line in box:
                self._vis.add_geometry(line)
        for track in trajectories:
            for line in track:
                self._vis.add_geometry(line)

    def save_viewpoint_file(self, frame, show_labels=True):
        frame = str(int(frame)).zfill(6)
        pc = self.get_painted_point_cloud(frame)
        self._vis.add_geometry(pc)
        if show_labels:
            self.create_boxes_and_trajectories(frame)
        self._vis.run()  # user changes the view and press "q" to terminate
        viewpoint_param = self._vis.get_view_control().convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(self.viewpoint_file, viewpoint_param)
        self._vis.clear_geometries()

    def visualize(self, frames=None, show_labels=True, screenshot=True):
        frames = self.frames if frames is None else [str(int(frame)).zfill(6) for frame in frames]
        self._vis.poll_events()
        self._vis.update_renderer()
        viewpoint_param = open3d.io.read_pinhole_camera_parameters(self.viewpoint_file)
        self._vis.get_view_control().convert_from_pinhole_camera_parameters(viewpoint_param)
        for frame in frames:
            pc = self.get_painted_point_cloud(frame)
            self._vis.add_geometry(pc)
            if show_labels:
                self.create_boxes_and_trajectories(frame)
            self._vis.poll_events()
            self._vis.update_renderer()
            if screenshot:
                self._vis.capture_screen_image(os.path.join(self.output_dir, f'{frame}.png'))
            self._vis.clear_geometries()
        self._vis.destroy_window()


def create_video(img_dir, video_name, size, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
    images = sorted(os.listdir(img_dir))
    for filename in tqdm.tqdm(images):
        video_writer.write(cv2.imread(os.path.join(img_dir, filename)))
    video_writer.release()


if __name__ == '__main__':
    viewer = KittiSequenceViewer(
        root_dir='/media/kemo/Kemo/Kitti/tracking/training',
        seq=0,
        output_dir='jmodt',
        viewpoint_file='viewpoint.json',
        label_dir='/media/kemo/Kemo/Kitti/tracking/training/label_02 (jmodt)',
        radius=0.1,
        trajectory_len=20
    )
    viewer.save_viewpoint_file(frame=1)
    viewer.visualize()
    create_video('jmodt', 'jmodt.avi', (1920, 1080))
