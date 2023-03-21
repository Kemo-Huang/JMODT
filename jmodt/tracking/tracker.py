import numpy as np
import torch

from jmodt.tracking.data_association import hungarian_match, ortools_solve
from jmodt.tracking.track import Track


class Tracker:
    def __init__(self, link_model, se_model, t_miss, t_hit,
                 w_cls, w_app, w_iou, w_dis, w_se, hungarian=False,
                 score_thresh=0, match_thresh=0):
        self.link_model = link_model
        self.se_model = se_model
        self.t_miss = t_miss
        self.t_hit = t_hit
        self.w_cls = w_cls
        self.w_app = w_app
        self.w_iou = w_iou
        self.w_dis = w_dis
        self.w_se = w_se
        self.hungarian = hungarian
        self.score_thresh = score_thresh
        self.match_thresh = match_thresh

        self.tracks = []
        self.frame_count = 0
        self.last_frame_idx = 0

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        self.last_frame_idx = 0
        Track.new_id = 1

    def track_management(self):
        idx = len(self.tracks)
        results = []
        for trk in reversed(self.tracks):
            if trk.hits >= self.t_hit or self.frame_count <= self.t_hit:
                if trk.misses == 0:
                    results.append(trk.get_data())
            idx -= 1
            # remove dead tracks
            if trk.misses >= self.t_miss:
                self.tracks.pop(idx)
        return results

    def update(self, frame_id, boxes_3d, det_scores, det_features, frame_detections):
        num_det = len(det_scores)
        num_pred = len(self.tracks)

        if num_det == 0:
            return []

        passed_frames = frame_id - self.last_frame_idx
        self.frame_count += passed_frames
        self.last_frame_idx = frame_id

        # for the first frame
        if num_pred == 0:
            # add in tracks
            for d in range(num_det):
                self.tracks.append(Track(bbox=boxes_3d[d], score=det_scores[d], feature=det_features[d],
                                         info=frame_detections[d]))
            return self.track_management()

        # get predictions of the current frame.
        pred_boxes = []
        pred_scores = []
        pred_features = []
        for trk in self.tracks:
            box, score, feature = trk.predict(passed_frames)
            pred_boxes.append(box)
            pred_scores.append(score)
            pred_features.append(feature.view(1, -1))

        pred_boxes = np.vstack(pred_boxes)
        pred_scores = np.array(pred_scores)
        pred_features = torch.cat(pred_features)

        cor_feat = torch.abs(
            pred_features.unsqueeze(1).repeat(1, num_det, 1)
            - det_features.unsqueeze(0).repeat(num_pred, 1, 1)
        )

        link_scores = self.link_model(cor_feat.view(num_pred * num_det, -1, 1)).view(num_pred, num_det)
        link_score_pred = torch.softmax(link_scores, dim=1)
        link_score_det = torch.softmax(link_scores, dim=0)
        link_scores = (link_score_pred + link_score_det) / 2

        if self.hungarian:
            matched, unmatched_dets, tentative_dets = hungarian_match(
                torch.from_numpy(boxes_3d).cuda(non_blocking=True),
                torch.from_numpy(pred_boxes).cuda(non_blocking=True),
                det_scores,
                link_scores,
                w_app=self.w_app,
                w_iou=self.w_iou,
                w_dis=self.w_dis,
                score_threshold=self.score_thresh,
                match_threshold=self.match_thresh
            )
        else:
            cls_scores = self.w_cls * (np.concatenate([pred_scores, det_scores]) - 1)
            start_scores = self.w_se * torch.sigmoid(
                self.se_model(cor_feat.mean(dim=0).unsqueeze(-1))
            ).cpu().numpy().flatten()
            end_scores = self.w_se * torch.sigmoid(
                self.se_model(cor_feat.mean(dim=1).unsqueeze(-1))
            ).cpu().numpy().flatten()
            start_scores = np.concatenate([np.zeros(num_pred), start_scores])
            end_scores = np.concatenate([end_scores, np.zeros(num_det)])

            matched, unmatched_dets, tentative_dets = ortools_solve(
                torch.from_numpy(boxes_3d).cuda(non_blocking=True),
                torch.from_numpy(pred_boxes).cuda(non_blocking=True),
                cls_scores,
                link_scores,
                start_scores,
                end_scores,
                w_app=self.w_app,
                w_iou=self.w_iou,
                w_dis=self.w_dis
            )
        # update matched tracks
        for t, d in matched:
            self.tracks[t].update_with_feature(boxes_3d[d],
                                               det_features[d],
                                               det_scores[d],
                                               info=frame_detections[d])
        # init new tracks for unmatched detections
        for i in unmatched_dets:
            trk = Track(bbox=boxes_3d[i], feature=det_features[i],
                        score=det_scores[i], info=frame_detections[i])
            self.tracks.append(trk)

        for i in tentative_dets:
            # print('tentative')
            trk = Track(bbox=boxes_3d[i], feature=det_features[i],
                        score=det_scores[i], info=frame_detections[i])
            trk.misses += 1
            self.tracks.append(trk)
        return self.track_management()
