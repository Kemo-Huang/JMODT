import numpy as np
import torch
from ortools.linear_solver import pywraplp
from scipy.optimize import linear_sum_assignment

from jmodt.ops.iou3d.iou3d_utils import boxes_iou3d_gpu
from jmodt.utils import kitti_utils


def boxes_dist_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        distance: (M, N)
    """
    m = len(boxes_a)
    n = len(boxes_b)
    boxes_a_corners = kitti_utils.boxes3d_to_corners3d_torch(boxes_a)
    boxes_b_corners = kitti_utils.boxes3d_to_corners3d_torch(boxes_b)
    center_distance = torch.linalg.norm(
        boxes_a[:, :3].unsqueeze(1).repeat(1, n, 1) - boxes_b[:, :3].unsqueeze(0).repeat(m, 1, 1),
        ord=2, dim=-1)
    corner_distance, _ = torch.max(torch.linalg.norm(
        boxes_a_corners.view(m, 1, 8, 1, 3).repeat(1, n, 1, 8, 1)
        - boxes_b_corners.view(1, n, 1, 8, 3).repeat(m, 1, 8, 1, 1),
        ord=2, dim=-1).view(m, n, 64), dim=-1)
    return 1. - center_distance / corner_distance


def ortools_solve(det_boxes,
                  pred_boxes,
                  cls_score,
                  link_score,
                  new_score,
                  end_score,
                  w_app,
                  w_iou,
                  w_dis):
    num_det = len(det_boxes)
    num_pred = len(pred_boxes)
    iou_matrix = boxes_iou3d_gpu(pred_boxes, det_boxes)
    dis_matrix = boxes_dist_gpu(pred_boxes, det_boxes)
    link_matrix = link_score * w_app + iou_matrix * w_iou + dis_matrix * w_dis
    link_matrix = link_matrix.cpu().numpy()
    solver = pywraplp.Solver('SolveAssignmentProblemMIP',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    y_det = {}
    y_new = {}
    y_end = {}
    y_link = {}
    for i in range(len(cls_score)):
        y_det[i] = solver.BoolVar('y_det[%i]' % i)
        y_new[i] = solver.BoolVar('y_new[%i]' % i)
        y_end[i] = solver.BoolVar('y_end[%i]' % i)
    w_link_y = []
    for j in range(num_pred):
        y_link[j] = {}
        for k in range(num_det):
            y_link[j][k] = solver.BoolVar(f'y_link[{j}, {k}]')
            w_link_y.append(y_link[j][k] * link_matrix[j, k])
    w_det_y = [y_det[i] * cls_score[i] for i in range(len(cls_score))]
    w_new_y = [y_new[i] * new_score[i] for i in range(len(cls_score))]
    w_end_y = [y_end[i] * end_score[i] for i in range(len(cls_score))]

    # Objective
    solver.Maximize(solver.Sum(w_det_y + w_new_y + w_end_y + w_link_y))

    # Constraints
    for j in range(num_pred):
        det_idx = j
        # pred = link + end
        solver.Add(
            solver.Sum([y_end[det_idx], (-1) * y_det[det_idx]] +
                       [y_link[j][k] for k in range(num_det)]) == 0)

    for k in range(num_det):
        det_idx = num_pred + k
        # det = link + start
        solver.Add(
            solver.Sum([y_new[det_idx], (-1) * y_det[det_idx]] +
                       [y_link[j][k] for j in range(num_pred)]) == 0)

    solver.Solve()

    assign_det = torch.zeros(len(cls_score))
    assign_new = torch.zeros(len(cls_score))
    assign_end = torch.zeros(len(cls_score))
    assign_link = torch.zeros(num_pred, num_det)

    for j in range(num_pred):
        for k in range(num_det):
            assign_link[j][k] = y_link[j][k].solution_value()

    for i in range(len(cls_score)):
        assign_det[i] = y_det[i].solution_value()
        assign_new[i] = y_new[i].solution_value()
        assign_end[i] = y_end[i].solution_value()

    matched = torch.nonzero(assign_link, as_tuple=False).tolist()
    unmatched_detections = torch.flatten(torch.nonzero(assign_new[num_pred:], as_tuple=False)).tolist()
    tentative_detections = torch.flatten(torch.nonzero((assign_det[num_pred:] == 0), as_tuple=False)).tolist()

    return matched, unmatched_detections, tentative_detections


def hungarian_match(det_boxes,
                    pred_boxes,
                    det_scores,
                    link_scores,
                    w_app,
                    w_iou,
                    w_dis,
                    score_threshold=0,
                    match_threshold=0
    ):
    iou_matrix = boxes_iou3d_gpu(pred_boxes, det_boxes)
    dis_matrix = boxes_dist_gpu(pred_boxes, det_boxes)
    link_matrix = link_scores * w_app + iou_matrix * w_iou + dis_matrix * w_dis
    link_matrix = link_matrix.cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(link_matrix, maximize=True)
    valid_mask = link_matrix[row_ind, col_ind] > match_threshold
    row_ind = row_ind[valid_mask]
    col_ind = col_ind[valid_mask]

    unmatched_detections = []
    tentative_detections = []
    for d in range(len(det_scores)):
        if d not in row_ind:
            if det_scores[d] > score_threshold:
                unmatched_detections.append(d)
            else:
                tentative_detections.append(d)

    matched = np.vstack((row_ind, col_ind)).T.tolist()

    return matched, unmatched_detections, tentative_detections
