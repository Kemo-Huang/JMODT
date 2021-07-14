import argparse
import logging
import os

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from jmodt.config import cfg, print_config_to_log, cfg_from_list
from jmodt.detection.datasets.kitti_dataset import KittiDataset
from jmodt.detection.modeling import train_functions
from jmodt.detection.modeling.point_rcnn import PointRCNN
from jmodt.utils import train_utils

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--data_root', type=str, default='data/KITTI', help='specify the data root')
parser.add_argument('--challenge', type=str, default='tracking', help='specify the KITTI benchmark')
parser.add_argument('--finetune', action='store_false', help='whether to finetune the pretrained model')
parser.add_argument("--batch_size", type=int, default=12, required=True, help="the batch size for training")
parser.add_argument('--output_dir', type=str, default='output', help='specify an output directory if needed')
parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument('--mgpus', action='store_true', help='whether to use multiple gpu')
parser.add_argument('--train_with_eval', action='store_true', help='whether to train with evaluation')
parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='set extra config keys if needed')
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_dataloader(logger, split):
    data_set = KittiDataset(root_dir=args.data_root, npoints=cfg.RPN.NUM_POINTS, split=split,
                            mode='TRAIN', logger=logger, classes=cfg.CLASSES, challenge=args.challenge)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, pin_memory=True, shuffle=True,
                             num_workers=4, collate_fn=data_set.collate_batch,
                             drop_last=True)
    return data_set, data_loader


def main():
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if args.finetune:
        cfg.RPN.FIXED = True
        cfg.TRAIN.FINETUNE = True
    else:
        cfg.RPN.FIXED = False
        cfg.TRAIN.FINETUNE = False

    root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_train.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    print_config_to_log(cfg, logger=logger)

    # tensorboard log
    tb_log = SummaryWriter(logdir=os.path.join(root_result_dir, 'tensorboard'))

    # create dataloader & network & optimizer
    train_set, train_loader = create_dataloader(logger, split=cfg.TRAIN.SPLIT)
    val_set, val_loader = create_dataloader(logger, split=cfg.TRAIN.VAL_SPLIT) if args.train_with_eval else None, None

    fn_decorator = train_functions.model_joint_fn_decorator()

    model = PointRCNN(num_classes=train_set.num_class, use_xyz=True, mode='TRAIN')
    if args.mgpus:
        model = nn.DataParallel(model)
    model.cuda()

    params_to_update = model.parameters()

    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        pure_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        if cfg.TRAIN.FINETUNE:
            for param in pure_model.parameters():
                param.requires_grad = False
            params_to_update = \
                list(pure_model.rcnn_net.link_layer.parameters()) + \
                list(pure_model.rcnn_net.se_layer.parameters())
            for param in params_to_update:
                param.requires_grad = True
            optimizer = optim.AdamW([
                {'params': pure_model.rcnn_net.link_layer.parameters()},
                {'params': pure_model.rcnn_net.se_layer.parameters()},
            ], lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            optimizer = optim.AdamW(params_to_update, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.RELOAD_OPTIMIZER:
            it, start_epoch = train_utils.load_checkpoint(pure_model, optimizer, filename=args.ckpt, logger=logger)
            last_epoch = start_epoch + 1
        else:
            train_utils.load_checkpoint(pure_model, optimizer=None, filename=args.ckpt, logger=logger)
    else:
        optimizer = optim.AdamW(params_to_update, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.TMAX,
                                                        eta_min=cfg.TRAIN.ETA_MIN, last_epoch=last_epoch)

    # start training
    logger.info('**********************Start training**********************')
    ckpt_dir = os.path.join(root_result_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer = train_utils.Trainer(
        model=model,
        params_to_update=params_to_update,
        model_fn_train=fn_decorator,
        optimizer=optimizer,
        ckpt_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        model_fn_val=fn_decorator,
        tb_log=tb_log,
        eval_frequency=1,
        grad_norm_clip=cfg.TRAIN.GRAD_NORM_CLIP
    )

    trainer.train(
        it,
        start_epoch,
        cfg.TRAIN.EPOCHS,
        train_loader,
        val_loader
    )

    logger.info('**********************End training**********************')


if __name__ == "__main__":
    main()
