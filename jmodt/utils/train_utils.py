import logging
import math
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

logging.getLogger(__name__).addHandler(logging.StreamHandler())
cur_logger = logging.getLogger(__name__)


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model=None, optimizer=None, filename='checkpoint', logger=cur_logger):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            keys = model.load_state_dict(checkpoint['model_state'], strict=False)
            logger.info(f"missing keys: {keys.missing_keys}\n")
            logger.info(f"unexpected keys: {keys.unexpected_keys}\n")
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return it, epoch


def load_part_ckpt(model, filename, logger=cur_logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


class Trainer:
    def __init__(self, model, params_to_update, model_fn_train, optimizer, ckpt_dir, lr_scheduler,
                 model_fn_val, tb_log, eval_frequency=1, grad_norm_clip=1.0):
        self.model = model
        self.params_to_update = params_to_update
        self.model_fn = model_fn_train
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_fn_eval = model_fn_val
        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency
        self.tb_log = tb_log
        self.grad_norm_clip = grad_norm_clip

    def eval_epoch(self, val_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = []
        nan_dict = {}

        # eval one epoch
        with tqdm.tqdm(val_loader, leave=False, desc='val') as tbar:
            for data in tbar:
                loss, tb_dict, disp_dict = self.model_fn_eval(self.model, data)
                if loss > 0:
                    total_loss.append(loss.item())
                tbar.set_postfix(disp_dict)
                for k, v in tb_dict.items():
                    if not math.isnan(v):
                        eval_dict[k] = eval_dict.get(k, 0) + v
                    else:
                        nan_dict[k] = nan_dict.get(k, 0) + 1

        # statistics this epoch
        for k, v in eval_dict.items():
            eval_dict[k] = eval_dict[k] / (len(val_loader) - nan_dict.get(k, 0))

        cur_performance = 0
        if 'recalled_cnt' in eval_dict:
            eval_dict['recall'] = eval_dict['recalled_cnt'] / max(eval_dict['gt_cnt'], 1)
            cur_performance = eval_dict['recall']
        elif 'iou' in eval_dict:
            cur_performance = eval_dict['iou']

        return sum(total_loss) / len(total_loss) if len(total_loss) > 0 else 0, eval_dict, cur_performance

    def train(self, start_it, start_epoch, n_epochs, train_loader, val_loader=None, stop_thres=5):
        eval_frequency = self.eval_frequency if self.eval_frequency > 0 else 1

        it = start_it
        min_val_loss = float('inf')
        prev_train_loss = -1
        prev_val_loss = -1
        counter = 0
        scaler = torch.cuda.amp.GradScaler()

        with tqdm.trange(start_epoch, n_epochs, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:

            for epoch in tbar:
                # train one epoch
                self.model.train()
                train_loss_epoch = []
                for batch in train_loader:
                    it += 1
                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        train_loss, tb_dict, disp_dict = self.model_fn(self.model, batch)
                    if train_loss > 0:
                        scaler.scale(train_loss).backward()
                        clip_grad_norm_(self.params_to_update, self.grad_norm_clip)
                        train_loss_epoch.append(train_loss.item())
                        scaler.step(self.optimizer)
                        scaler.update()
                        if self.tb_log is not None:
                            self.tb_log.add_scalar('train_loss', train_loss.item(), it)
                            for key, val in tb_dict.items():
                                self.tb_log.add_scalar('train_' + key, val, it)
                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.set_postfix(disp_dict)
                    tbar.refresh()

                trained_epoch = epoch + 1
                train_loss_epoch = sum(train_loss_epoch) / len(train_loss_epoch) if len(train_loss_epoch) > 0 else 0

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    if self.tb_log is not None:
                        cur_lr = self.lr_scheduler.get_last_lr()
                        if type(cur_lr) == list:
                            self.tb_log.add_scalar('learning_rate_1', cur_lr[0], trained_epoch)
                            self.tb_log.add_scalar('learning_rate_2', cur_lr[-1], trained_epoch)
                        else:
                            self.tb_log.add_scalar('learning_rate', cur_lr, trained_epoch)

                if self.tb_log is not None:
                    self.tb_log.add_scalar('train_loss_epoch', train_loss_epoch, trained_epoch)

                pbar.close()
                # save trained model
                ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(self.model, self.optimizer, trained_epoch, it), filename=ckpt_name,
                )

                # eval one epoch
                if (epoch % eval_frequency) == 0:
                    if val_loader is not None:
                        with torch.set_grad_enabled(False):
                            val_loss_epoch, eval_dict, cur_performance = self.eval_epoch(val_loader)

                        if self.tb_log is not None:
                            self.tb_log.add_scalar('val_loss_epoch', val_loss_epoch, trained_epoch)
                            for key, val in eval_dict.items():
                                self.tb_log.add_scalar('val_' + key, val, trained_epoch)

                if prev_train_loss != -1 and prev_val_loss != -1:
                    if train_loss_epoch < prev_train_loss and val_loss_epoch > prev_val_loss:
                        counter += 1
                        cur_logger.info("Bad train")
                        if counter > stop_thres:
                            cur_logger.info("Early stopping")
                            break
                    else:
                        counter = 0
                else:
                    prev_train_loss = train_loss_epoch
                    prev_val_loss = val_loss_epoch

                if val_loss_epoch < min_val_loss:
                    min_val_loss = val_loss_epoch
                    ckpt_name = os.path.join(self.ckpt_dir, 'best_model')
                    if isinstance(self.model, torch.nn.DataParallel):
                        model_state = self.model.module.state_dict()
                    else:
                        model_state = self.model.state_dict()
                    save_checkpoint({'model_state': model_state}, filename=ckpt_name)

                pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train')
                pbar.set_postfix(dict(total_it=it))
