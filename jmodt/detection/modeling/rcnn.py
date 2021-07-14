import torch
import torch.nn as nn
import torch.nn.functional as F

import jmodt.ops.pointnet2.pytorch_utils as pt_utils
from jmodt.config import cfg
from jmodt.detection.layers.proposal_target_layer import ProposalTargetLayer
from jmodt.ops.pointnet2.pointnet2_modules import PointnetSAModule
from jmodt.utils import loss_utils


class RCNN(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True, mode='TRAIN'):
        super().__init__()
        self.mode = mode

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]

            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=cfg.RCNN.SA_CONFIG.RADIUS[k],
                    nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RCNN.USE_BN
                )
            )
            channel_in = mlps[-1]

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        if cfg.USE_IOU_BRANCH:
            iou_branch = []
            iou_branch.append(pt_utils.Conv1d(channel_in, cfg.RCNN.REG_FC[0], bn=cfg.RCNN.USE_BN))
            iou_branch.append(pt_utils.Conv1d(cfg.RCNN.REG_FC[0], cfg.RCNN.REG_FC[1], bn=cfg.RCNN.USE_BN))
            iou_branch.append(pt_utils.Conv1d(cfg.RCNN.REG_FC[1], 1, activation=None))
            if cfg.RCNN.DP_RATIO >= 0:
                iou_branch.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
            self.iou_branch = nn.Sequential(*iou_branch)

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        # link layer
        link_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.REID.LINK_FC.__len__()):
            link_layers.append(pt_utils.Conv1d(pre_channel, cfg.REID.LINK_FC[k], bn=cfg.REID.USE_BN))
            pre_channel = cfg.REID.LINK_FC[k]
        link_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        if cfg.REID.DP_RATIO >= 0:
            link_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.link_layer = nn.Sequential(*link_layers)

        # start-end layer
        se_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.REID.SE_FC.__len__()):
            se_layers.append(pt_utils.Conv1d(pre_channel, cfg.REID.SE_FC[k], bn=cfg.REID.USE_BN))
            pre_channel = cfg.REID.SE_FC[k]
        se_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        if cfg.REID.DP_RATIO >= 0:
            se_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.se_layer = nn.Sequential(*se_layers)

        self.proposal_target_layer = ProposalTargetLayer(mode=self.mode)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    @staticmethod
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    @staticmethod
    def get_unique_tid_feature(prev_fg_tid: torch.Tensor, prev_fg_feat: torch.Tensor):
        prev_tid_diff = torch.min(prev_fg_tid)
        prev_fg_tid_clip = (prev_fg_tid - prev_tid_diff).long()
        m = prev_fg_tid.new_zeros(torch.max(prev_fg_tid_clip) + 1, len(prev_fg_tid))
        m[prev_fg_tid_clip, torch.arange(len(prev_fg_tid))] = 1
        m = F.normalize(m, p=1, dim=1)
        prev_tid_feat_mean = torch.mm(m, prev_fg_feat)
        prev_fg_tid_clip_unique = torch.unique(prev_fg_tid_clip)
        prev_unique_feat = prev_tid_feat_mean[prev_fg_tid_clip_unique]
        prev_fg_tid_unique = prev_fg_tid_clip_unique + prev_tid_diff
        return prev_fg_tid_unique, prev_unique_feat

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            with torch.no_grad():
                pts_input, target_dict = self.proposal_target_layer(input_data)  # generate labels
        else:
            pts_input = input_data['pts_input']
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input']
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d']
            if self.training:
                target_dict['cls_label'] = input_data['cls_label']
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask']
                target_dict['gt_of_rois'] = input_data['gt_boxes3d']

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).contiguous().unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).contiguous().unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, _ = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        rcnn_cls = self.cls_layer(l_features[-1]).squeeze(-1)  # (B, 1)
        rcnn_reg = self.reg_layer(l_features[-1]).squeeze(-1)  # (B, C)

        if cfg.USE_IOU_BRANCH:
            rcnn_iou_branch = self.iou_branch(l_features[-1]).squeeze(-1)  # (B,1)
            ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg, 'rcnn_iou_branch': rcnn_iou_branch}
        else:
            ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.mode == 'TRAIN':
            ret_dict.update(target_dict)
            # reid
            gt_tids = target_dict['gt_tids']
            num_frames = gt_tids.shape[0]
            # assert num_frames == 2, str(num_frames)
            input_features = l_features[-1]
            # split rois into prev and next
            prev_tids = gt_tids[range(0, num_frames, 2)]
            next_tids = gt_tids[range(1, num_frames, 2)]
            feat_len = input_features.shape[-2]
            input_features = input_features.view(num_frames, -1, feat_len)
            prev_features = input_features[range(0, num_frames, 2)]
            next_features = input_features[range(1, num_frames, 2)]
            prev_fg_mask = prev_tids > 0
            next_fg_mask = next_tids > 0
            rcnn_link = []
            start_features = []
            end_features = []
            gt_links = []
            gt_starts = []
            gt_ends = []
            for i in range(num_frames // 2):
                prev_fg_tid = prev_tids[i][prev_fg_mask[i]]
                next_fg_tid = next_tids[i][next_fg_mask[i]]
                prev_fg_feat = prev_features[i][prev_fg_mask[i]]
                next_fg_feat = next_features[i][next_fg_mask[i]]
                n_prev = len(prev_fg_feat)
                n_next = len(next_fg_feat)
                if n_prev > 0 and n_next > 0:
                    # link
                    prev_tid_unique, prev_feat_unique = self.get_unique_tid_feature(prev_fg_tid, prev_fg_feat)
                    next_tid_unique, next_feat_unique = self.get_unique_tid_feature(next_fg_tid, next_fg_feat)
                    unique_link = (prev_tid_unique.unsqueeze(1) == next_tid_unique).float()
                    gt_links.append(unique_link.view(-1))
                    cor_feat = torch.abs(
                        prev_feat_unique.unsqueeze(1).repeat(1, len(next_tid_unique), 1)
                        - next_feat_unique.unsqueeze(0).repeat(len(prev_tid_unique), 1, 1)
                    )
                    # link + softmax
                    link_feat = cor_feat.view(len(prev_tid_unique) * len(next_tid_unique), feat_len, 1)
                    link_scores = self.link_layer(link_feat).view(len(prev_tid_unique), len(next_tid_unique))
                    link_prev = torch.softmax(link_scores, dim=1)
                    link_next = torch.softmax(link_scores, dim=0)
                    link_scores = (link_prev + link_next) / 2
                    rcnn_link.append(link_scores.view(len(prev_tid_unique) * len(next_tid_unique), 1))
                    # start end
                    gt_start = 1 - unique_link.sum(0)
                    gt_end = 1 - unique_link.sum(1)
                    gt_starts.append(gt_start)
                    gt_ends.append(gt_end)
                    start_feat = cor_feat.mean(dim=0)
                    end_feat = cor_feat.mean(dim=1)
                    start_features.append(start_feat)
                    end_features.append(end_feat)

            if len(gt_links) > 0:
                gt_links = torch.cat(gt_links)
                rcnn_link = torch.cat(rcnn_link)
                ret_dict['gt_links'] = gt_links
                ret_dict['rcnn_link'] = rcnn_link
            else:
                ret_dict['gt_links'] = gt_tids.new(0)
                ret_dict['rcnn_link'] = gt_tids.new(0, 1)

            if len(gt_starts) > 0:
                gt_starts = torch.cat(gt_starts)
                start_features = torch.cat(start_features).unsqueeze(-1)
                rcnn_start = self.se_layer(start_features).squeeze(-1)
                ret_dict['gt_starts'] = gt_starts
                ret_dict['rcnn_start'] = rcnn_start
            else:
                ret_dict['gt_starts'] = gt_tids.new(0)
                ret_dict['rcnn_start'] = gt_tids.new(0, 1)

            if len(gt_ends) > 0:
                gt_ends = torch.cat(gt_ends)
                end_features = torch.cat(end_features).unsqueeze(-1)
                rcnn_end = self.se_layer(end_features).squeeze(-1)
                ret_dict['gt_ends'] = gt_ends
                ret_dict['rcnn_end'] = rcnn_end
            else:
                ret_dict['gt_ends'] = gt_tids.new(0)
                ret_dict['rcnn_end'] = gt_tids.new(0, 1)
        else:
            ret_dict['rcnn_feat'] = l_features[-1]
        return ret_dict
