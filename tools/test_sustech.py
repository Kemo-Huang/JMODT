import argparse
import json
import logging
import os
import time
from multiprocessing.pool import Pool

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from jmodt.config import cfg
from jmodt.detection.datasets.sustech_dataset import SUSTechDataset, save_sustech_format
from jmodt.detection.modeling.point_rcnn import PointRCNN
from jmodt.ops.iou3d import iou3d_utils
from jmodt.tracking import tracker
from jmodt.utils import train_utils, kitti_utils, sustech_utils
from jmodt.utils.bbox_transform import decode_bbox_target

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--data_root', type=str, default='E://sustech-data/2021-07-07/dataset_2hz',
                    help='the ground truth data root')
parser.add_argument('--det_output', type=str, default='output/det/2021-07-07/',
                    help='the detection output directory')
parser.add_argument('--trk_output', type=str, default='output/trk/2021-07-07/', help='the tracking output directory')
parser.add_argument('--ckpt', type=str, default='checkpoints/jmodt.pth', help='the pretrained model path')
parser.add_argument('--tag', type=str, default='all', help='the tag for tracking results')
parser.add_argument('--hungarian', action='store_true', help='whether to use hungarian algorithm')
args = parser.parse_args()

# global random seed can be specified here
np.random.seed(2333)


@torch.no_grad()
def joint_detection_and_tracking(logger, detection=True, tracking=True, fps=10):
    """
    detection from six camera views, nms results, joint tracking 
    """
    det_output = args.det_output
    detection_res_txt_dir = os.path.join(det_output, 'label')
    detection_res_feat_dir = os.path.join(det_output, 'feat')
    os.makedirs(detection_res_txt_dir, exist_ok=True)
    os.makedirs(detection_res_feat_dir, exist_ok=True)

    tracking_res_dir = os.path.join(args.trk_output, args.tag)
    os.makedirs(tracking_res_dir, exist_ok=True)

    mode = 'TEST'
    model = PointRCNN(num_classes=2, use_xyz=True, mode=mode)
    model.eval()
    model.cuda()
    # load checkpoint
    train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)

    if detection:
        logger.info('********************** Start detection and tracking **********************')
        logger.info('==> Detection output dir: %s' % det_output)

        camera_list = ['rear', 'rear_left', 'rear_right', 'front', 'front_left', 'front_right']
        sample_det_results = {}

        for camera in camera_list:
            # create dataloader
            dataset = SUSTechDataset(args.data_root, camera=camera)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                                    num_workers=4, collate_fn=dataset.collate_batch)

            for data in tqdm.tqdm(dataloader):
                sample_id = data['sample_id']
                batch_size = len(sample_id)

                inputs = torch.from_numpy(data['pts_input']).cuda(non_blocking=True).float()
                input_data = {'pts_input': inputs}
                # img feature
                if cfg.LI_FUSION.ENABLED:
                    pts_xy, img = data['pts_xy'], data['img']
                    pts_xy = torch.from_numpy(pts_xy).cuda(non_blocking=True).float()
                    img = torch.from_numpy(img).permute((0, 3, 1, 2)).cuda(non_blocking=True).float()
                    input_data['pts_xy'] = pts_xy
                    input_data['img'] = img

                # model inference
                ret_dict = model(input_data)

                roi_boxes3d = ret_dict['rois']  # (B, M, 7)

                rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
                rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)
                rcnn_feat = ret_dict['rcnn_feat'].view(batch_size, -1, ret_dict['rcnn_feat'].shape[1])

                # bounding box regression
                pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                                  anchor_size=torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda(),
                                                  loc_scope=cfg.RCNN.LOC_SCOPE,
                                                  loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                                  num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                                  get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                                  loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
                                                  loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
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

                    # change view
                    if 'rear' in camera:
                        pred_boxes3d_selected[:, 2] *= -1
                        pred_boxes3d_selected[:, -1] *= -1

                    cur_sample_id = sample_id[k]
                    if cur_sample_id not in sample_det_results:
                        sample_det_results[cur_sample_id] = {
                            'boxes3d': pred_boxes3d_selected,
                            'raw_scores': raw_scores_selected,
                            'features': feat_selected,
                            'norm_scores': norm_scores_selected
                        }

                    else:
                        sample_det_results[cur_sample_id]['boxes3d'] = torch.cat(
                            (sample_det_results[cur_sample_id]['boxes3d'], pred_boxes3d_selected))
                        sample_det_results[cur_sample_id]['raw_scores'] = torch.cat(
                            (sample_det_results[cur_sample_id]['raw_scores'], raw_scores_selected))
                        sample_det_results[cur_sample_id]['features'] = torch.cat(
                            (sample_det_results[cur_sample_id]['features'], feat_selected))
                        sample_det_results[cur_sample_id]['norm_scores'] = torch.cat(
                            (sample_det_results[cur_sample_id]['norm_scores'], norm_scores_selected))

        # NMS thresh
        for cur_sample_id, det_results in sample_det_results.items():
            pred_boxes3d_selected = det_results['boxes3d']
            raw_scores_selected = det_results['raw_scores']
            feat_selected = det_results['features']
            norm_scores_selected = det_results['norm_scores']

            boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
            keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
            pred_boxes3d_selected = pred_boxes3d_selected[keep_idx].cpu().numpy()
            feat_selected = feat_selected[keep_idx].cpu().numpy()
            scores_selected = norm_scores_selected[keep_idx].cpu().numpy()

            save_sustech_format(cur_sample_id, bbox3d=pred_boxes3d_selected, score=scores_selected,
                                txt_output_dir=detection_res_txt_dir,
                                feat=feat_selected, feat_output_dir=detection_res_feat_dir)

    if tracking:
        # MOT hyper-parameters
        t_miss = fps
        t_hit = 0
        w_cls = 100
        w_app = 2
        w_iou = 10
        w_dis = 10
        w_se = 1
        cls_threshold = 0.85

        logger.info("**********************Start evaluate tracking**********************")
        logger.info(f't_miss={t_miss}, t_hit={t_hit}, '
                    f'w_cls={w_cls}, w_app={w_app}, w_iou={w_iou}, w_dis={w_dis}, w_se={w_se}')

        car_tracker = tracker.Tracker(
            link_model=model.rcnn_net.link_layer,
            se_model=model.rcnn_net.se_layer,
            t_miss=t_miss, t_hit=t_hit,
            w_cls=w_cls, w_app=w_app, w_iou=w_iou, w_dis=w_dis, w_se=w_se,
            hungarian=args.hungarian)

        total_time = 0
        total_frames = 0

        sample_ids = sorted([f[:-5] for f in os.listdir(os.path.join(det_output, 'label'))])
        tbar = tqdm.tqdm(total=len(sample_ids), dynamic_ncols=True, leave=True)
        car_tracker.reset()
        with torch.no_grad():
            for sample_id in sample_ids:
                with open(os.path.join(det_output, 'label', f'{sample_id}.json'), 'r') as f:
                    try:
                        car_objects = json.load(f)
                    except json.decoder.JSONDecodeError:
                        car_objects = []
                if len(car_objects) == 0:
                    tbar.update()
                    continue
                car_objects = np.array(car_objects)
                car_features = np.load(os.path.join(det_output, 'feat', f'{sample_id}.npy'))
                assert len(car_features) == len(car_objects), f"obj {len(car_objects)} != feat {len(car_features)}"

                car_features = torch.from_numpy(car_features).cuda(non_blocking=True)
                boxes_3d = np.empty((len(car_objects), 7), dtype=np.float32)
                for d in range(len(car_objects)):
                    psr = car_objects[d]['psr']
                    boxes_3d[d, 0] = -psr['position']['x']
                    boxes_3d[d, 1] = psr['scale']['y'] / 2 - psr['position']['z']
                    boxes_3d[d, 2] = -psr['position']['y']
                    boxes_3d[d, 3] = psr['scale']['z']
                    boxes_3d[d, 4] = psr['scale']['y']
                    boxes_3d[d, 5] = psr['scale']['x']
                    boxes_3d[d, 6] = np.pi - psr['rotation']['z']
                scores = np.array([obj['score'] for obj in car_objects], dtype=np.float32)

                mask = scores > cls_threshold

                boxes_3d = boxes_3d[mask]
                scores = scores[mask]
                car_features = car_features[mask]
                car_objects = car_objects[mask]

                frame_id = float(sample_id)
                frame_id = int(frame_id * fps)

                start_time = time.time()
                car_results = car_tracker.update(frame_id, boxes_3d, scores, car_features, car_objects)
                frame_time = time.time() - start_time

                total_time += frame_time
                total_frames += 1
                tbar.set_postfix({'time': frame_time})
                tbar.update()

                sample_tracks = []
                for tid, info, score in car_results:
                    info.update({"obj_id": str(tid)})
                    sample_tracks.append(info)

                with open(os.path.join(tracking_res_dir, f'{sample_id}.json'), 'w') as f:
                    json.dump(sample_tracks, f)

            tbar.close()
        logger.info(
            f'total frames: {total_frames}, total time: {total_time}, frames per second: {total_frames / total_time}')


def convert_pcd_to_bin(logger):
    data_root = 'E://sustech-data/2021-07-07/dataset_2hz'
    assert os.path.exists(data_root)
    os.makedirs(os.path.join(data_root, 'lidar_bin'), exist_ok=True)

    logger.info(f'==> Converting pcd to bin from {data_root}')

    arguments = []
    for sustech_lidar in os.listdir(os.path.join(data_root, 'lidar')):
        frame = sustech_lidar[:-4]
        in_path = os.path.join(data_root, 'lidar', sustech_lidar)
        out_path = os.path.join(data_root, 'lidar_bin', f'{frame}.bin')
        arguments.append((in_path, out_path))

    with Pool(8) as pool:
        pool.starmap(sustech_utils.pcd_to_bin, arguments)

    logger.info('==> Done')


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level=logging.INFO)
    logger.addHandler(ch)

    convert_pcd_to_bin(logger)

    # start
    joint_detection_and_tracking(logger, detection=True, tracking=True, fps=2)


if __name__ == '__main__':
    main()
