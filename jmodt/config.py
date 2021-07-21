from ast import literal_eval

import numpy as np
import yaml
from easydict import EasyDict

# data split
TRAIN_SEQ_ID = ['0001', '0003', '0004', '0006', '0013', '0008', '0009', '0012', '0015', '0020']
VALID_SEQ_ID = ['0000', '0002', '0005', '0007', '0010', '0011', '0014', '0016', '0018', '0019']
TEST_SEQ_ID = ['%04d' % seq for seq in range(29)]
SMALL_VAL_SEQ_ID = ['0019']

# config
cfg = EasyDict()

# 0. basic config
cfg.TAG = 'default'
cfg.CLASSES = 'Car'
cfg.INCLUDE_SIMILAR_TYPE = True

# config of augmentation
cfg.AUG_DATA = False
cfg.AUG_METHOD_LIST = ['rotation', 'scaling', 'flip']
cfg.AUG_METHOD_PROB = [1.0, 1.0, 0.5]
cfg.AUG_ROT_RANGE = 18

cfg.GT_AUG_ENABLED = False
cfg.GT_EXTRA_NUM = 15
cfg.GT_AUG_RAND_NUM = True
cfg.GT_AUG_APPLY_PROB = 1.0
cfg.GT_AUG_HARD_RATIO = 0.6

cfg.PC_REDUCE_BY_RANGE = True
cfg.PC_AREA_SCOPE = np.array([[-40, 40],
                              [-1, 3],
                              [0, 70.4]])  # x, y, z scope in rect camera coords

cfg.CLS_MEAN_SIZE = np.array([[1.52563191462, 1.62856739989, 3.88311640418]], dtype=np.float32)

# 0.1 config of use img
cfg.USE_IOU_BRANCH = False

# config of LI-Fusion
cfg.LI_FUSION = EasyDict()
cfg.LI_FUSION.ENABLED = True
cfg.LI_FUSION.IMG_FEATURES_CHANNEL = 128
cfg.LI_FUSION.IMG_CHANNELS = [3, 64, 128, 256, 512]
cfg.LI_FUSION.POINT_CHANNELS = [96, 256, 512, 1024]

cfg.LI_FUSION.DeConv_Reduce = [16, 16, 16, 16]
cfg.LI_FUSION.DeConv_Kernels = [2, 4, 8, 16]
cfg.LI_FUSION.DeConv_Strides = [2, 4, 8, 16]

# 1. config of rpn
cfg.RPN = EasyDict()
cfg.RPN.ENABLED = True
cfg.RPN.FIXED = True

cfg.RPN.USE_INTENSITY = False

# 1.1 config of use img_rgb input (x)
cfg.RPN.USE_RGB = False

# config of bin-based loss
cfg.RPN.LOC_XZ_FINE = True
cfg.RPN.LOC_SCOPE = 3.0
cfg.RPN.LOC_BIN_SIZE = 0.5
cfg.RPN.NUM_HEAD_BIN = 12

# config of network structure
cfg.RPN.USE_BN = True
cfg.RPN.NUM_POINTS = 16384

cfg.RPN.SA_CONFIG = EasyDict()
cfg.RPN.SA_CONFIG.NPOINTS = [4096, 1024, 256, 64]
cfg.RPN.SA_CONFIG.RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
cfg.RPN.SA_CONFIG.NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
cfg.RPN.SA_CONFIG.MLPS = [[[16, 16, 32], [32, 32, 64]],
                          [[64, 64, 128], [64, 96, 128]],
                          [[128, 196, 256], [128, 196, 256]],
                          [[256, 256, 512], [256, 384, 512]]]
cfg.RPN.FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
cfg.RPN.CLS_FC = [128]
cfg.RPN.REG_FC = [128]
cfg.RPN.DP_RATIO = 0.5

# config of training
cfg.RPN.LOSS_CLS = 'SigmoidFocalLoss'
cfg.RPN.FG_WEIGHT = 15
cfg.RPN.FOCAL_ALPHA = [0.25, 0.75]
cfg.RPN.FOCAL_GAMMA = 2.0
cfg.RPN.REG_LOSS_WEIGHT = [1.0, 1.0, 1.0, 1.0]
cfg.RPN.LOSS_WEIGHT = [1.0, 1.0]
cfg.RPN.NMS_TYPE = 'normal'  # normal, rotate

# config of testing
cfg.RPN.SCORE_THRESH = 0.2

# 2. config of rcnn
cfg.RCNN = EasyDict()
cfg.RCNN.ENABLED = True

# config of input
cfg.RCNN.ROI_SAMPLE_JIT = True
cfg.RCNN.REG_AUG_METHOD = 'multiple'  # multiple, single, normal
cfg.RCNN.ROI_FG_AUG_TIMES = 0

cfg.RCNN.USE_RPN_FEATURES = True
cfg.RCNN.USE_MASK = True
cfg.RCNN.MASK_TYPE = 'seg'
cfg.RCNN.USE_INTENSITY = False
cfg.RCNN.USE_DEPTH = True
cfg.RCNN.USE_SEG_SCORE = False

cfg.RCNN.POOL_EXTRA_WIDTH = 0.2

cfg.RCNN.USE_RGB = False
# config of bin-based loss
cfg.RCNN.LOC_SCOPE = 1.5
cfg.RCNN.LOC_BIN_SIZE = 0.5
cfg.RCNN.NUM_HEAD_BIN = 9
cfg.RCNN.LOC_Y_BY_BIN = False
cfg.RCNN.LOC_Y_SCOPE = 0.5
cfg.RCNN.LOC_Y_BIN_SIZE = 0.25
cfg.RCNN.SIZE_RES_ON_ROI = False

# config of network structure
cfg.RCNN.USE_BN = False
cfg.RCNN.DP_RATIO = 0.0
cfg.RCNN.XYZ_UP_LAYER = [128, 128]

cfg.RCNN.NUM_POINTS = 512
cfg.RCNN.SA_CONFIG = EasyDict()
cfg.RCNN.SA_CONFIG.NPOINTS = [128, 32, -1]
cfg.RCNN.SA_CONFIG.RADIUS = [0.2, 0.4, 100]
cfg.RCNN.SA_CONFIG.NSAMPLE = [64, 64, 64]
cfg.RCNN.SA_CONFIG.MLPS = [[128, 128, 128],
                           [128, 128, 256],
                           [256, 256, 512]]
cfg.RCNN.CLS_FC = [512, 512]
cfg.RCNN.REG_FC = [512, 512]

# config of training
cfg.RCNN.LOSS_CLS = 'BinaryCrossEntropy'
cfg.RCNN.FOCAL_ALPHA = [0.25, 0.75]
cfg.RCNN.FOCAL_GAMMA = 2.0
cfg.RCNN.CLS_WEIGHT = np.array([1.0, 1.0, 1.0], dtype=np.float32)
cfg.RCNN.CLS_FG_THRESH = 0.6
cfg.RCNN.CLS_BG_THRESH = 0.45
cfg.RCNN.CLS_BG_THRESH_LO = 0.05
cfg.RCNN.REG_FG_THRESH = 0.55
cfg.RCNN.FG_RATIO = 0.5
cfg.RCNN.ROI_PER_IMAGE = 64
cfg.RCNN.HARD_BG_RATIO = 0.8
cfg.RCNN.IOU_LOSS_TYPE = 'raw'
cfg.RCNN.IOU_ANGLE_POWER = 1

# config of testing
cfg.RCNN.SCORE_THRESH = 0.2
cfg.RCNN.NMS_THRESH = 0.1

# 3. config of reid branches
cfg.REID = EasyDict()
cfg.REID.ENABLED = True
cfg.REID.FG_THRESH = 0.85
cfg.REID.LINK_FC = [512, 512]
cfg.REID.SE_FC = [512, 512]
cfg.REID.USE_BN = False
cfg.REID.DP_RATIO = 0.0
cfg.REID.LOSS_LINK = 'L1'
cfg.REID.LOSS_SE = 'L1'

# general training config
cfg.TRAIN = EasyDict()
cfg.TRAIN.SPLIT = 'train'
cfg.TRAIN.VAL_SPLIT = 'small_val'

cfg.TRAIN.FINETUNE = True
cfg.TRAIN.RELOAD_OPTIMIZER = False
cfg.TRAIN.EPOCHS = 50
cfg.TRAIN.LR = 2e-4
cfg.TRAIN.TMAX = 50
cfg.TRAIN.ETA_MIN = 0
cfg.TRAIN.WEIGHT_DECAY = 1e-2
cfg.TRAIN.GRAD_NORM_CLIP = 1.0

cfg.TRAIN.RPN_PRE_NMS_TOP_N = 9000
cfg.TRAIN.RPN_POST_NMS_TOP_N = 512
cfg.TRAIN.RPN_NMS_THRESH = 0.85
cfg.TRAIN.RPN_DISTANCE_BASED_PROPOSE = True
cfg.TRAIN.RPN_TRAIN_WEIGHT = 1.0
cfg.TRAIN.RCNN_TRAIN_WEIGHT = 1.0
cfg.TRAIN.LINK_TRAIN_WEIGHT = 1.0
cfg.TRAIN.SE_TRAIN_WEIGHT = 1.0
cfg.TRAIN.CE_WEIGHT = 5.0
cfg.TRAIN.IOU_LOSS_TYPE = 'cls_mask_with_bin'
cfg.TRAIN.BBOX_AVG_BY_BIN = True
cfg.TRAIN.RY_WITH_BIN = False

# config of testing
cfg.EVAL = EasyDict()
cfg.EVAL.SPLIT = 'val'
cfg.EVAL.RPN_PRE_NMS_TOP_N = 9000
cfg.EVAL.RPN_POST_NMS_TOP_N = 100
cfg.EVAL.RPN_NMS_THRESH = 0.8
cfg.EVAL.RPN_DISTANCE_BASED_PROPOSE = True
cfg.EVAL.BBOX_AVG_BY_BIN = True
cfg.EVAL.RY_WITH_BIN = False

cfg.TEST = EasyDict()
cfg.TEST.SPLIT = 'test'
cfg.TEST.RPN_PRE_NMS_TOP_N = 9000
cfg.TEST.RPN_POST_NMS_TOP_N = 100
cfg.TEST.RPN_NMS_THRESH = 0.8
cfg.TEST.RPN_DISTANCE_BASED_PROPOSE = True
cfg.TEST.BBOX_AVG_BY_BIN = True
cfg.TEST.RY_WITH_BIN = False


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, cfg)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = cfg
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert isinstance(value, type(d[subkey])), \
            'type {} does not match original type {}'.format(type(value), type(d[subkey]))
        d[subkey] = value


def print_config_to_log(config, pre='cfg', logger=None):
    for key, val in config.items():
        if isinstance(config[key], EasyDict):
            if logger is not None:
                logger.info('\n%s.%s = edict()' % (pre, key))
            else:
                print('\n%s.%s = edict()' % (pre, key))
            print_config_to_log(config[key], pre=pre + '.' + key, logger=logger)
            continue

        if logger is not None:
            logger.info('%s.%s: %s' % (pre, key, val))
        else:
            print('%s.%s: %s' % (pre, key, val))
