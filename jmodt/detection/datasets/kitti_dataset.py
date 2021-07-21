import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from jmodt.config import cfg
from jmodt.utils import calibration, kitti_utils


class KittiDataset(Dataset):
    def __init__(self, root_dir, npoints=16384, split='train', classes='Car', mode='TRAIN', logger=None,
                 challenge='detection', fixed_img_size=(384, 1280)):
        self.split = split
        is_test = self.split == 'test'
        self.challenge = challenge
        self.fixed_img_size = fixed_img_size
        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode
        if challenge == 'detection':
            data_dir = os.path.join(root_dir, 'object', 'testing' if is_test else 'training')
            split_file = os.path.join(root_dir, 'object', 'ImageSets', split + '.txt')
            self.image_idx_list = [x.strip() for x in open(split_file).readlines()]
        else:
            data_dir = os.path.join(root_dir, 'tracking_object',
                                    'testing' if is_test else 'training')
            split_file = os.path.join(root_dir, 'tracking_object',
                                      'ImageSets', split + '.txt')
            if self.mode == 'TRAIN':
                # two consecutive frames
                self.sample_pair_list = [x.split() for x in open(split_file).readlines()]
            else:
                self.image_idx_list = [x.strip() for x in open(split_file).readlines()]

        self.image_dir = os.path.join(data_dir, 'image_2')
        self.lidar_dir = os.path.join(data_dir, 'velodyne')
        self.calib_dir = os.path.join(data_dir, 'calib')
        self.label_dir = os.path.join(data_dir, 'label_2')

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.npoints = npoints
        self.logger = logger

        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes
        self.num_class = self.classes.__len__()

        if cfg.RPN.ENABLED:
            if self.challenge == 'detection':
                if self.mode == 'TRAIN':
                    self.logger.info('Loading %s samples from %s ...' % (self.mode, self.label_dir))
                    self.sample_id_list = []
                    for idx in range(len(self.image_idx_list)):
                        sample_id = int(self.image_idx_list[idx])
                        obj_list = self.filtrate_objects(self.get_label(sample_id))
                        if len(obj_list) != 0:
                            self.sample_id_list.append(sample_id)
                    self.logger.info(
                        f'Done filtering: {len(self.sample_id_list)} / {len(self.image_idx_list)}\n')
                else:
                    self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            else:
                if self.mode == 'TRAIN':
                    self.logger.info('Loading %s samples from %s ...' % (self.mode, self.label_dir))
                    self.sample_pair_id_list = []
                    for idx in range(len(self.sample_pair_list)):
                        prev_sample_id, next_sample_id = self.sample_pair_list[idx]
                        prev_sample_id = int(prev_sample_id)
                        next_sample_id = int(next_sample_id)
                        prev_obj_list = self.filtrate_objects(self.get_label(prev_sample_id))
                        next_obj_list = self.filtrate_objects(self.get_label(next_sample_id))
                        if len(prev_obj_list) > 0 and len(next_obj_list) > 0:
                            self.sample_pair_id_list.append((prev_sample_id, next_sample_id))
                    self.logger.info(
                        f'Done filtering: {len(self.sample_pair_id_list)} / {len(self.sample_pair_list)}\n')
                else:
                    self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        elif cfg.RCNN.ENABLED:
            if self.challenge == 'detection' or self.mode != 'TRAIN':
                self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            else:
                self.sample_pair_id_list = [(int(x1), int(x2)) for x1, x2 in self.sample_pair_list]

    def get_normalized_image(self, idx):
        """
        :return: fixed_size_img (H,W,3)
        """
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        img = Image.open(img_file).convert('RGB')
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        fixed_size_img = np.zeros([self.fixed_img_size[0], self.fixed_img_size[1], 3], dtype=np.float32)
        fixed_size_img[:img.shape[0], :img.shape[1], :] = img

        return fixed_size_img

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file), f'Path {img_file} does not exist'
        img = Image.open(img_file)
        width, height = img.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file), f'Path {lidar_file} does not exist'
        points = np.fromfile(lidar_file, dtype=np.float32)
        points = points.reshape(-1, 4)
        return points

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file), f'Path {calib_file} does not exist'
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file), f'Path {label_file} does not exist'
        return kitti_utils.get_objects_from_label(label_file)

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue
            if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    def __len__(self):
        if self.challenge == 'detection' or self.mode != 'TRAIN':
            return len(self.sample_id_list)
        else:
            return len(self.sample_pair_id_list)

    def __getitem__(self, index):
        if self.challenge == 'detection' or self.mode != 'TRAIN':
            sample_id = self.sample_id_list[index]
            return self.get_sample_dict(sample_id)
        else:
            prev_sample_id, next_sample_id = self.sample_pair_id_list[index]
            return self.get_sample_dict(prev_sample_id), self.get_sample_dict(next_sample_id)

    def get_sample_dict(self, sample_id):
        """
        :return:
            sample_id
            img
            pts_xy
            pts_input
            gt_boxes3d
            gt_tid
        """
        calib = self.get_calib(sample_id)
        img = self.get_normalized_image(sample_id)
        img_shape = self.get_image_shape(sample_id)
        pts_lidar = self.get_lidar(sample_id)

        # get valid points (projected points should be in image)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
        pts_rect = pts_rect[pts_valid_flag]
        pts_intensity = pts_lidar[pts_valid_flag, 3]
        pts_xy = pts_img[pts_valid_flag]

        # generate inputs
        if self.npoints < len(pts_rect):
            # keep far points, sample near points
            pts_depth = pts_rect[:, 2]
            pts_near_flag = pts_depth < 40.0
            far_indices_choice = np.where(pts_near_flag == 0)[0]
            near_indices = np.where(pts_near_flag == 1)[0]
            near_indices_choice = np.random.choice(near_indices, self.npoints - len(far_indices_choice), replace=False)

            choice = np.concatenate((near_indices_choice, far_indices_choice), axis=0) \
                if len(far_indices_choice) > 0 else near_indices_choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(pts_rect), dtype=np.int32)
            while self.npoints > len(choice):
                extra_choice = np.random.choice(choice, self.npoints % len(choice), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)

        ret_pts_rect = pts_rect[choice, :]
        ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
        ret_pts_xy = pts_xy[choice, :]

        # normalize xy to [-1,1]
        ret_pts_xy[:, 0] = ret_pts_xy[:, 0] / (self.fixed_img_size[1] - 1.0) * 2.0 - 1.0
        ret_pts_xy[:, 1] = ret_pts_xy[:, 1] / (self.fixed_img_size[0] - 1.0) * 2.0 - 1.0

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        sample_info = {'sample_id': sample_id, 'img': img, 'pts_xy': ret_pts_xy}

        if self.mode == 'TEST':  # no label
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect
            sample_info['pts_input'] = pts_input
        else:
            gt_obj_list = self.filtrate_objects(self.get_label(sample_id))

            gt_boxes3d = np.zeros((gt_obj_list.__len__(), 7), dtype=np.float32)
            gt_alpha = np.zeros((gt_obj_list.__len__()), dtype=np.float32)
            gt_tids = np.zeros((gt_obj_list.__len__()), dtype=np.float32)

            for k, obj in enumerate(gt_obj_list):
                gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, obj.ry
                gt_alpha[k] = obj.alpha
                gt_tids[k] = obj.score

            # data augmentation
            aug_pts_rect = ret_pts_rect.copy()
            aug_gt_boxes3d = gt_boxes3d.copy()
            if cfg.AUG_DATA and self.mode == 'TRAIN':
                aug_pts_rect, aug_gt_boxes3d = self.data_augmentation(aug_pts_rect, aug_gt_boxes3d, gt_alpha)

            # prepare input
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = aug_pts_rect

            sample_info['pts_input'] = pts_input
            sample_info['gt_boxes3d'] = aug_gt_boxes3d
            sample_info['gt_tids'] = gt_tids
            if not cfg.RPN.FIXED:
                # generate training labels
                rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(aug_pts_rect, aug_gt_boxes3d)
                sample_info['rpn_cls_label'] = rpn_cls_label
                sample_info['rpn_reg_label'] = rpn_reg_label

        return sample_info

    @staticmethod
    def generate_rpn_training_labels(pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # ry

        return cls_label, reg_label

    @staticmethod
    def rotate_box3d_along_y(box3d, rot_angle):
        old_x, old_z, ry = box3d[0], box3d[2], box3d[6]
        old_beta = np.arctan2(old_z, old_x)
        alpha = -np.sign(old_beta) * np.pi / 2 + old_beta + ry

        box3d = kitti_utils.rotate_pc_along_y(box3d.reshape(1, 7), rot_angle=rot_angle)[0]
        new_x, new_z = box3d[0], box3d[2]
        new_beta = np.arctan2(new_z, new_x)
        box3d[6] = np.sign(new_beta) * np.pi / 2 + alpha - new_beta

        return box3d

    @staticmethod
    def data_augmentation(aug_pts_rect, aug_gt_boxes3d, gt_alpha):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            aug_pts_rect = kitti_utils.rotate_pc_along_y(aug_pts_rect, rot_angle=angle)
            # xyz change, hwl unchanged
            aug_gt_boxes3d = kitti_utils.rotate_pc_along_y(aug_gt_boxes3d, rot_angle=angle)

            # calculate the ry after rotation
            x, z = aug_gt_boxes3d[:, 0], aug_gt_boxes3d[:, 2]
            beta = np.arctan2(z, x)
            new_ry = np.sign(beta) * np.pi / 2 + gt_alpha - beta
            aug_gt_boxes3d[:, 6] = new_ry  # TODO: not in [-np.pi / 2, np.pi / 2]

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect = aug_pts_rect * scale
            aug_gt_boxes3d[:, 0:6] = aug_gt_boxes3d[:, 0:6] * scale

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            aug_gt_boxes3d[:, 6] = np.sign(aug_gt_boxes3d[:, 6]) * np.pi - aug_gt_boxes3d[:, 6]

        return aug_pts_rect, aug_gt_boxes3d

    def collate_batch(self, batch):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        if self.challenge == 'detection' or self.mode != 'TRAIN':
            for key in batch[0].keys():
                if cfg.RPN.ENABLED and key == 'gt_boxes3d' or (
                        cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                    max_gt = 0
                    for k in range(batch_size):
                        max_gt = max(max_gt, batch[k][key].__len__())
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                    for i in range(batch_size):
                        batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                    ans_dict[key] = batch_gt_boxes3d

                elif isinstance(batch[0][key], np.ndarray):
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

        else:
            for key in batch[0][0].keys():
                if cfg.RPN.ENABLED and key == 'gt_boxes3d' or (
                        cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d',
                                                                                 'gt_tids']):
                    max_gt = 0
                    for k in range(batch_size):
                        max_gt = max(max_gt, batch[k][0][key].__len__(), batch[k][1][key].__len__())
                    if key == 'gt_tids':
                        batch_gt_tids = np.zeros((batch_size * 2, max_gt), dtype=np.int32)
                        for i in range(batch_size):
                            batch_gt_tids[2 * i, :batch[i][0][key].__len__()] = batch[i][0][key]
                            batch_gt_tids[2 * i + 1, :batch[i][1][key].__len__()] = batch[i][1][key]
                        ans_dict[key] = batch_gt_tids
                    else:
                        batch_gt_boxes3d = np.zeros((batch_size * 2, max_gt, 7), dtype=np.float32)
                        for i in range(batch_size):
                            batch_gt_boxes3d[2 * i, :batch[i][0][key].__len__(), :] = batch[i][0][key]
                            batch_gt_boxes3d[2 * i + 1, :batch[i][1][key].__len__(), :] = batch[i][1][key]
                        ans_dict[key] = batch_gt_boxes3d

                elif isinstance(batch[0][0][key], np.ndarray):
                    value_list = []
                    for k in range(batch_size):
                        value_list.append(batch[k][0][key][np.newaxis, ...])
                        value_list.append(batch[k][1][key][np.newaxis, ...])
                    ans_dict[key] = np.concatenate(value_list)

                else:
                    value_list = []
                    for k in range(batch_size):
                        value_list.append(batch[k][0][key])
                        value_list.append(batch[k][1][key])
                    if isinstance(batch[0][0][key], int):
                        ans_dict[key] = np.array(value_list, dtype=np.int32)
                    elif isinstance(batch[0][0][key], float):
                        ans_dict[key] = np.array(value_list, dtype=np.float32)

        return ans_dict
