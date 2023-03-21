import argparse
import logging
import os
import re
import time
from datetime import datetime

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from jmodt.config import VALID_SEQ_ID, cfg
from jmodt.detection.datasets.kitti_dataset import KittiDataset
from jmodt.detection.evaluation.evaluate import evaluate as evaluate_detection
from jmodt.detection.modeling.point_rcnn import PointRCNN
from jmodt.ops.iou3d import iou3d_utils
from jmodt.tracking import tracker
from jmodt.tracking.kitti_evaluate import evaluate as evaluate_tracking
from jmodt.utils import kitti_utils, train_utils
from jmodt.utils.bbox_transform import decode_bbox_target
from jmodt.utils.object3d import Object3d

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--data_root', type=str, default='data/KITTI', help='the ground truth data root')
parser.add_argument('--det_output', type=str, default='output/det', help='the detection output directory')
parser.add_argument('--output_dir', type=str, default='output', help='the tracking output directory')
parser.add_argument('--ckpt', type=str, default='checkpoints/jmodt.pth', help='the pretrained model path')
parser.add_argument('--tag', type=str, default='mot_data', help='the tag for tracking results')
parser.add_argument('--hungarian', action='store_true', help='whether to use hungarian algorithm')
parser.add_argument('--only_tracking', action='store_true', help='whether to evaluate tracking without detection')
parser.add_argument('--test', action='store_true', help='whether to use the test split data')
args = parser.parse_args()

# global random seed can be specified here
np.random.seed(2333)


@torch.no_grad()
def eval_joint_detection(logger):
    det_output = args.det_output
    # set epoch_id and output dir
    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'

    logger.info('**********************Start evaluate detection**********************')

    # create dataloader & network
    mode = 'TEST' if args.test else 'EVAL'
    split = cfg.TEST.SPLIT if args.test else cfg.EVAL.SPLIT
    # create dataloader
    dataset = KittiDataset(root_dir=args.data_root, npoints=cfg.RPN.NUM_POINTS, split=split, mode=mode,
                           classes=cfg.CLASSES, challenge='tracking', logger=logger)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                            num_workers=4, collate_fn=dataset.collate_batch)
    model = PointRCNN(num_classes=dataset.num_class, use_xyz=True, mode=mode)
    model.eval()
    model.cuda()

    # load checkpoint
    train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)

    detection_res_txt_dir = os.path.join(det_output, 'txt')
    os.makedirs(detection_res_txt_dir, exist_ok=True)
    detection_res_feat_dir = os.path.join(det_output, 'feat')
    os.makedirs(detection_res_feat_dir, exist_ok=True)

    logger.info('==> Detection output dir: %s' % det_output)

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0

    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='test' if args.test else 'eval')
    for data in dataloader:
        cnt += 1
        sample_id = data['sample_id']
        batch_size = len(sample_id)

        inputs = torch.from_numpy(data['pts_input']).cuda(non_blocking=True).float()
        input_data = {'pts_input': inputs}
        # img feature
        if cfg.LI_FUSION.ENABLED:
            pts_xy, img = data['pts_xy'], data['img']
            pts_xy = torch.from_numpy(pts_xy).cuda(non_blocking=True).float()
            img = torch.from_numpy(img).cuda(non_blocking=True).float().permute((0, 3, 1, 2))
            input_data['pts_xy'] = pts_xy
            input_data['img'] = img

        # model inference
        ret_dict = model(input_data)

        roi_boxes3d = ret_dict['rois']  # (B, M, 7)
        seg_result = ret_dict['seg_result'].long()  # (B, N)

        rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
        rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)
        rcnn_feat = ret_dict['rcnn_feat'].view(batch_size, -1, ret_dict['rcnn_feat'].shape[1])

        if cfg.USE_IOU_BRANCH:
            rcnn_iou_branch = ret_dict['rcnn_iou_branch'].view(batch_size, -1, ret_dict['rcnn_iou_branch'].shape[1])
            rcnn_iou_branch = torch.max(rcnn_iou_branch,
                                        rcnn_iou_branch.new().resize_(rcnn_iou_branch.shape).fill_(1e-4))
            rcnn_cls = rcnn_iou_branch * rcnn_cls

        # bounding box regression
        pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size=torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda(),
                                          loc_scope=cfg.RCNN.LOC_SCOPE,
                                          loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                          num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                          get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                          loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                          get_ry_fine=True).view(batch_size, -1, 7)

        # scoring
        if rcnn_cls.shape[2] == 1:
            raw_scores = rcnn_cls  # (B, M, 1)
            norm_scores = torch.sigmoid(raw_scores)
        else:
            pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
            cls_norm_scores = torch.softmax(rcnn_cls, dim=1)
            raw_scores = rcnn_cls[:, pred_classes]
            norm_scores = cls_norm_scores[:, pred_classes]

        # evaluation
        if not args.test:
            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).long()

            gt_boxes3d = data['gt_boxes3d']

            for k in range(batch_size):
                # calculate recall
                cur_gt_boxes3d = gt_boxes3d[k]
                tmp_idx = cur_gt_boxes3d.__len__() - 1

                while tmp_idx >= 0 and cur_gt_boxes3d[tmp_idx].sum() == 0:
                    tmp_idx -= 1

                if tmp_idx >= 0:
                    cur_gt_boxes3d = cur_gt_boxes3d[:tmp_idx + 1]

                    cur_gt_boxes3d = torch.from_numpy(cur_gt_boxes3d).cuda(non_blocking=True).float()
                    iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou, _ = iou3d.max(dim=0)
                    refined_iou, _ = iou3d.max(dim=1)

                    for idx, thresh in enumerate(thresh_list):
                        total_recalled_bbox_list[idx] += (gt_max_iou > thresh).sum().item()
                    total_gt_bbox += cur_gt_boxes3d.shape[0]

                    # original recall
                    iou3d_in = iou3d_utils.boxes_iou3d_gpu(roi_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou_in, _ = iou3d_in.max(dim=0)

                    for idx, thresh in enumerate(thresh_list):
                        total_roi_recalled_bbox_list[idx] += (gt_max_iou_in > thresh).sum().item()

                if not cfg.RPN.FIXED:
                    fg_mask = rpn_cls_label > 0
                    correct = ((seg_result == rpn_cls_label) & fg_mask).sum().float()
                    union = fg_mask.sum().float() + (seg_result > 0).sum().float() - correct
                    rpn_iou = correct / torch.clamp(union, min=1.0)
                    total_rpn_iou += rpn_iou.item()

        disp_dict = {'mode': mode, 'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)}
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

        # scores thresh
        inds = norm_scores > cfg.RCNN.SCORE_THRESH

        for k in range(batch_size):
            cur_inds = inds[k].view(-1)
            if cur_inds.sum() == 0:
                continue

            pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
            raw_scores_selected = raw_scores[k, cur_inds]
            feat_selected = rcnn_feat[k, cur_inds]
            norm_scores_selected = norm_scores[k, cur_inds]

            # NMS thresh
            # rotated nms
            boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
            keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
            pred_boxes3d_selected = pred_boxes3d_selected[keep_idx].cpu().numpy()
            scores_selected = norm_scores_selected[keep_idx].cpu().numpy()
            feat_selected = feat_selected[keep_idx].cpu().numpy()

            cur_sample_id = sample_id[k]
            calib = dataset.get_calib(cur_sample_id)
            final_total += pred_boxes3d_selected.shape[0]
            image_shape = dataset.get_image_shape(cur_sample_id)
            save_kitti_detection_format(cur_sample_id, calib, pred_boxes3d_selected, detection_res_txt_dir,
                                        scores_selected, image_shape, feat_selected, detection_res_feat_dir)

    progress_bar.close()
    # dump empty files
    image_idx_list = dataset.image_idx_list
    empty_cnt = 0
    for k in range(image_idx_list.__len__()):
        cur_file = os.path.join(detection_res_txt_dir, '%06d.txt' % int(image_idx_list[k]))
        if not os.path.exists(cur_file):
            with open(cur_file, 'w'):
                pass
            empty_cnt += 1
            logger.info('empty_cnt=%d: dump empty file %s' % (empty_cnt, cur_file))

    if not args.test:
        logger.info('-------------------performance of epoch %s---------------------' % epoch_id)
        avg_rpn_iou = (total_rpn_iou / max(cnt, 1))
        avg_cls_acc = (total_cls_acc / max(cnt, 1))
        avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1))
        avg_det_num = (final_total / max(len(dataset), 1))
        logger.info('final average detections: %.3f' % avg_det_num)
        logger.info('final average rpn_iou refined: %.3f' % avg_rpn_iou)
        logger.info('final average cls acc: %.3f' % avg_cls_acc)
        logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)

        for idx, thresh in enumerate(thresh_list):
            cur_roi_recall = total_roi_recalled_bbox_list[idx] / max(total_gt_bbox, 1)
            logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_roi_recalled_bbox_list[idx],
                                                                              total_gt_bbox, cur_roi_recall))

        for idx, thresh in enumerate(thresh_list):
            cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1)
            logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_recalled_bbox_list[idx],
                                                                          total_gt_bbox, cur_recall))

        logger.info('Averate Precision:')
        name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        ap_result_str, ap_dict = evaluate_detection(dataset.label_dir, detection_res_txt_dir,
                                                    image_idx_list=image_idx_list,
                                                    current_class=name_to_class[cfg.CLASSES])
        logger.info(ap_result_str)

    logger.info('result is saved to: %s\n' % det_output)


def save_kitti_detection_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape,
                                feat=None, feat_output_dir=None):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if not box_valid_mask[k]:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f)
    if feat is not None:
        output_file = os.path.join(feat_output_dir, '%06d.npy' % sample_id)
        np.save(output_file, feat[box_valid_mask].astype(np.float32))


def convert_det_sample_to_seq_frame(seq2sample_path, sample2frame_path):
    seq2sample_dict = {}
    sample2frame_dict = {}
    with open(seq2sample_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split()
        seq_id = split_line[0]
        seq2sample_dict[seq_id] = split_line[1:]
    with open(sample2frame_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split()
        sample2frame_dict[split_line[0]] = split_line[2]
    return seq2sample_dict, sample2frame_dict


def eval_tracking(logger):
    part = 'test' if args.test else 'val'
    tracking_res_dir = os.path.join(args.output_dir, args.tag, part)
    os.makedirs(tracking_res_dir, exist_ok=True)
    det_res_dir = args.det_output

    # MOT hyper-parameters for MIP
    t_miss = 2
    t_hit = 0
    w_cls = 100
    w_app = 2
    w_iou = 10
    w_dis = 10
    w_se = 1
    cls_thresh = 0.85

    # MOT hyper-parameters for HA
    if args.hungarian:
        t_miss = 2
        t_hit = 0
        w_app = 2
        w_iou = 10
        w_dis = 10
        cls_thresh = 0.85
        score_thresh = 0
        match_thresh = 0


    logger.info("**********************Start evaluate tracking**********************")
    logger.info(f't_miss={t_miss}, t_hit={t_hit}, '
                f'w_cls={w_cls}, w_app={w_app}, w_iou={w_iou}, w_dis={w_dis}, w_se={w_se}')

    model = PointRCNN(num_classes=2, mode='TEST' if args.test else 'EVAL')
    model.eval()
    model.cuda()

    # load checkpoint
    train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)

    car_tracker = tracker.Tracker(
        link_model=model.rcnn_net.link_layer,
        se_model=model.rcnn_net.se_layer,
        t_miss=t_miss, t_hit=t_hit,
        w_cls=w_cls, w_app=w_app, w_iou=w_iou, w_dis=w_dis, w_se=w_se,
        hungarian=args.hungarian,
        score_thresh=score_thresh,
        match_thresh=match_thresh)

    total_time = 0
    total_frames = 0

    seq2sample_dict, sample2frame_dict = convert_det_sample_to_seq_frame(
        os.path.join(args.data_root, 'tracking_object', 'testing' if args.test else 'training', 'seq2sample.txt'),
        os.path.join(args.data_root, 'tracking_object', 'testing' if args.test else 'training', 'sample2frame.txt')
    )

    seq_list = [str(i).zfill(4) for i in range(29)] if args.test else VALID_SEQ_ID
    for seq_id in seq_list:
        sample_ids = seq2sample_dict[seq_id]
        out_file = open(os.path.join(tracking_res_dir, f'{seq_id}.txt'), 'w+')
        tbar = tqdm.tqdm(total=len(sample_ids), desc=seq_id, dynamic_ncols=True, leave=True)
        car_tracker.reset()
        with torch.no_grad():
            for sample_id in sample_ids:
                with open(os.path.join(det_res_dir, 'txt', f'{sample_id}.txt'), 'r') as f:
                    object_lines = f.readlines()
                if len(object_lines) == 0:
                    tbar.update()
                    continue
                car_objects = np.array([Object3d(line) for line in object_lines])
                car_features = np.load(os.path.join(det_res_dir, 'feat', f'{sample_id}.npy'))
                assert len(car_features) == len(car_objects), f"obj {len(car_objects)} != feat {len(car_features)}"

                car_features = torch.from_numpy(car_features).cuda(non_blocking=True)
                boxes_3d = np.empty((len(car_objects), 7), dtype=np.float32)
                for d in range(len(car_objects)):
                    boxes_3d[d, :3] = car_objects[d].pos
                    boxes_3d[d, 3] = car_objects[d].h
                    boxes_3d[d, 4] = car_objects[d].w
                    boxes_3d[d, 5] = car_objects[d].l
                    boxes_3d[d, 6] = car_objects[d].ry
                scores = np.array([obj.score for obj in car_objects], dtype=np.float32)

                mask = scores > cls_thresh

                boxes_3d = boxes_3d[mask]
                scores = scores[mask]
                car_features = car_features[mask]
                car_objects = car_objects[mask]

                frame_id = sample2frame_dict[sample_id]
                frame_id = int(frame_id)

                start_time = time.time()
                car_results = car_tracker.update(frame_id, boxes_3d, scores, car_features, car_objects)
                frame_time = time.time() - start_time

                total_time += frame_time
                total_frames += 1
                tbar.set_postfix({'time': frame_time})
                tbar.update()

                save_kitti_tracking_format(car_results, frame_id, out_file)
        out_file.close()
        tbar.close()
    logger.info(
        f'total frames: {total_frames}, total time: {total_time}, frames per second: {total_frames / total_time}')

    if not args.test:
        gt_path = os.path.join(args.data_root, 'tracking', 'training')
        evaluate_tracking(
            result_sha=args.tag, result_root=args.output_dir, part=part, gt_path=gt_path, logger=logger)


def save_kitti_tracking_format(results, frame_id, out_file):
    for tid, info, score in results:
        out_file.write(
            '%d %d %s %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (
                frame_id, tid, info.cls_type, int(info.truncation), int(info.occlusion), info.alpha,
                info.box2d[0], info.box2d[1], info.box2d[2], info.box2d[3],
                info.h, info.w, info.l, info.pos[0], info.pos[1], info.pos[2],
                info.ry, info.score
            )
        )


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level=logging.INFO)
    logger.addHandler(ch)

    if not args.only_tracking:
        os.makedirs(args.det_output, exist_ok=True)
        det_log_path = os.path.join(args.det_output, f'{datetime.now().strftime("%Y-%m-%d-%S-%M-%H")}.log')
        det_fh = logging.FileHandler(det_log_path)
        det_fh.setFormatter(formatter)
        det_fh.setLevel(level=logging.INFO)
        logger.addHandler(det_fh)

        # start eval detection
        eval_joint_detection(logger)

        logger.removeHandler(det_fh)

    os.makedirs(os.path.join(args.output_dir, args.tag), exist_ok=True)
    trk_log_path = os.path.join(args.output_dir, args.tag, f'{datetime.now().strftime("%Y-%m-%d-%S-%M-%H")}.log')
    trk_fh = logging.FileHandler(trk_log_path)
    trk_fh.setFormatter(formatter)
    trk_fh.setLevel(level=logging.INFO)
    logger.addHandler(trk_fh)

    # start eval tracking
    eval_tracking(logger)


if __name__ == '__main__':
    main()
