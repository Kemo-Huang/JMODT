import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from jmodt.utils.sustech_utils import proj_lidar_to_img, psr_to_corners


class SUSTechDataset(Dataset):
    def __init__(self, root_dir, camera):
        self.root_dir = root_dir
        self.camera = camera

        self.npoints = 16384
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.num_class = 2  # only cars
        self.img_shape = (1536, 2048, 3)

        lidar_sample_list = sorted(
            [f[:-4] for f in os.listdir(os.path.join(root_dir, 'lidar_bin'))]
        )
        camera_sample_list = sorted(
            [f[:-4] for f in os.listdir(os.path.join(root_dir, 'camera', camera))]
        )
        self.sample_id_list = [s for s in lidar_sample_list if s in camera_sample_list]

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        sample_id = self.sample_id_list[idx]
        return self.get_sample_dict(sample_id)

    def get_lidar(self, frame):
        file_path = os.path.join(self.root_dir, 'lidar_bin', f'{frame}.bin')
        assert os.path.exists(file_path), file_path
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

    def get_image(self, frame):
        file_path = os.path.join(self.root_dir, 'camera', self.camera, f'{frame}.jpg')
        assert os.path.exists(file_path), file_path
        img = Image.open(file_path).convert('RGB')
        img = np.array(img).astype(np.float32)
        assert img.shape == self.img_shape
        img /= 255.0
        img -= self.mean
        img /= self.std
        return img

    def get_calib(self):
        file_path = os.path.join(self.root_dir, 'calib', 'camera', f'{self.camera}.json')
        assert os.path.exists(file_path), file_path
        with open(file_path, 'r') as f:
            calib = json.load(f)
        extrinsic = np.array(calib['extrinsic'], dtype=np.float32).reshape(4, 4)[:3, :]
        intrinsic = np.array(calib['intrinsic'], dtype=np.float32).reshape(3, 3)
        return extrinsic, intrinsic

    def sample_points(self, pc):
        # generate inputs
        if self.npoints < len(pc):
            # keep far points, sample near points
            pts_depth = pc[:, 0]
            pts_near_flag = pts_depth < 40.0
            far_indices_choice = np.argwhere(pts_near_flag == 0).flatten()
            near_indices = np.argwhere(pts_near_flag == 1).flatten()
            near_indices_choice = np.random.choice(near_indices, self.npoints - len(far_indices_choice), replace=False)
            choice = np.concatenate((near_indices_choice, far_indices_choice), axis=0) \
                if len(far_indices_choice) > 0 else near_indices_choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(pc), dtype=np.int32)
            while self.npoints > len(choice):
                extra_choice = np.random.choice(choice, self.npoints % len(choice), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        return choice

    @staticmethod
    def get_pc_in_range(pc, xy_dist=70, z_range=(-3, 1)):
        horizontal_distances = np.sqrt(np.square(pc[:, 0]) + np.square(pc[:, 1]))
        xy_mask = horizontal_distances <= xy_dist
        z_mask = np.logical_and(pc[:, 2] >= z_range[0], pc[:, 2] <= z_range[1])
        return pc[np.logical_and(xy_mask, z_mask)]

    def get_sample_dict(self, frame):
        """
        :return:
            sample_id
            img
            pts_xy
            pts_input
            gt_boxes3d
            gt_tid
        """
        pc = self.get_lidar(frame)
        pts_xyz = pc[:, :3]
        # intensity = pc[:, 3]
        img = self.get_image(frame)
        img_shape = img.shape[:2]
        extrinsic, intrinsic = self.get_calib()

        pts_xyz = self.get_pc_in_range(pts_xyz)
        pts_xy, valid_mask = proj_lidar_to_img(pts_xyz, extrinsic, intrinsic, img_shape)
        pts_xyz = pts_xyz[valid_mask]

        choice = self.sample_points(pts_xyz)
        pts_xyz = pts_xyz[choice]
        pts_xy = pts_xy[choice]

        # normalize xy to [-1,1]
        pts_xy[:, 0] = pts_xy[:, 0] / (self.img_shape[1] - 1.0) * 2.0 - 1.0
        pts_xy[:, 1] = pts_xy[:, 1] / (self.img_shape[0] - 1.0) * 2.0 - 1.0

        # change view
        if 'rear' in self.camera:
            pts_xyz[:, 1] *= -1

        # change coordinates
        # sustech: x left, y rear, z up
        # kitti: x right, y down, z front
        pts_xyz = pts_xyz[:, [0, 2, 1]]
        pts_xyz *= -1

        sample_dict = {
            'sample_id': frame,
            'img': img,
            'pts_xy': pts_xy,
            'pts_input': pts_xyz
        }

        return sample_dict

    @staticmethod
    def collate_batch(batch):
        batch_size = batch.__len__()
        ans_dict = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)],
                                                   axis=0)
            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)
        return ans_dict


def save_sustech_format(sample_id, bbox3d, score, txt_output_dir,
                        feat=None, feat_output_dir=None):
    bbox3d = bbox3d.astype(np.float64)
    score = score.astype(np.float64).flatten()
    output_file = os.path.join(txt_output_dir, f'{sample_id}.json')
    output_list = [{
        'psr': {
            'position': {
                'x': -bbox3d[k, 0],
                'y': -bbox3d[k, 2],
                'z': -bbox3d[k, 1] + bbox3d[k, 4] / 2
            },
            'scale': {
                'x': bbox3d[k, 5],  # l
                'y': bbox3d[k, 4],  # w
                'z': bbox3d[k, 3]  # h
            },
            'rotation': {
                'x': 0,
                'y': 0,
                'z': np.pi - bbox3d[k, 6]
            }
        },
        'obj_type': 'Car',
        'score': score[k]
    } for k in range(bbox3d.shape[0])]
    with open(output_file, 'w') as f:
        json.dump(output_list, f)
    if feat is not None:
        output_file = os.path.join(feat_output_dir, f'{sample_id}.npy')
        np.save(output_file, feat.astype(np.float32))


def convert_sustech_label_to_kitti(label_dir, output_dir, calib_dir):
    labels = os.listdir(label_dir)
    for label in labels:
        with open(os.path.join(label_dir, label), 'r') as f:
            objects = json.load(f)
            psr = objects['psr']
            corners_3d = psr_to_corners(psr['position'], psr['scale'], psr['rotation'])

            for camera in ['rear', 'rear_left', 'rear_right', 'front', 'front_left', 'front_right']:
                file_path = os.path.join(calib_dir, 'camera', f'{camera}.json')
                with open(file_path, 'r') as calib_f:
                    calib = json.load(calib_f)
                extrinsic = np.array(calib['extrinsic'], dtype=np.float32).reshape(4, 4)[:3, :]
                intrinsic = np.array(calib['intrinsic'], dtype=np.float32).reshape(3, 3)

                corners_2d, valid_mask = proj_lidar_to_img(corners_3d, extrinsic, intrinsic, (1536, 2048))
                if np.sum(valid_mask) == 8:
                    break
    # TODO detection and tracking formats
