from yacs.config import CfgNode as CN
from yacs.config import load_cfg

_C = CN()

# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = "@"
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means not to set explicitly.
_C.RNG_SEED = 1

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

_C.DATA = CN()

_C.DATA.NUM_WORKERS = 1

_C.DATA.SCORE_CLASSES = 3
_C.DATA.NUM_REMOVAL_DIRECTIONS = 5

_C.DATA.TYPE = "Scene"

_C.DATA.STD_R = 0.1
_C.DATA.STD_T = 0.02

_C.DATA.NUM_POINTS = 25600
_C.DATA.NUM_CLOSE_REGION_POINTS = 1024
_C.DATA.GPD_IN_CHANNELS = 3

_C.DATA.TRAIN = CN()
_C.DATA.TRAIN.ROOT_DIR = ""

_C.DATA.VAL = CN()
_C.DATA.VAL.ROOT_DIR = ""
_C.DATA.VAL.NUM_GRASP = 300

_C.DATA.TEST = CN()
_C.DATA.TEST.ROOT_DIR = ""
_C.DATA.TEST.NUM_GRASP = 300

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.WEIGHT = ""

_C.MODEL.TYPE = ""

_C.MODEL.GPD = CN()
_C.MODEL.GPD.DROPOUT = False

_C.MODEL.EDGEPN2D = CN()
_C.MODEL.EDGEPN2D.NUM_CENTROIDS = (10240, 1024, 128, 0)
_C.MODEL.EDGEPN2D.RADIUS = (0.2, 0.3, 0.4, -1.0)
_C.MODEL.EDGEPN2D.NUM_NEIGHBOURS = (64, 64, 64, -1)
_C.MODEL.EDGEPN2D.SA_CHANNELS = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 512, 1024))
_C.MODEL.EDGEPN2D.FP_CHANNELS = ((256, 256), (256, 128), (128, 128), (64, 64, 64))
_C.MODEL.EDGEPN2D.NUM_FP_NEIGHBOURS = (0, 3, 3, 3)
_C.MODEL.EDGEPN2D.SEG_CHANNELS = (128, )
_C.MODEL.EDGEPN2D.DROPOUT_PROB = 0.5
_C.MODEL.EDGEPN2D.LABEL_SMOOTHING = 0.0
_C.MODEL.EDGEPN2D.NEG_WEIGHT = 1.0

_C.MODEL.EDGEPN2DU = CN()
_C.MODEL.EDGEPN2DU.NUM_CENTROIDS = (10240, 1024, 128, 0)
_C.MODEL.EDGEPN2DU.RADIUS = (0.2, 0.3, 0.4, -1.0)
_C.MODEL.EDGEPN2DU.NUM_NEIGHBOURS = (64, 64, 64, -1)
_C.MODEL.EDGEPN2DU.SA_CHANNELS = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 512, 1024))
_C.MODEL.EDGEPN2DU.FP_CHANNELS = ((256, 256), (256, 128), (128, 128), (64, 64, 64))
_C.MODEL.EDGEPN2DU.NUM_FP_NEIGHBOURS = (0, 3, 3, 3)
_C.MODEL.EDGEPN2DU.SEG_CHANNELS = (128, )
_C.MODEL.EDGEPN2DU.DROPOUT_PROB = 0.5
_C.MODEL.EDGEPN2DU.LABEL_SMOOTHING = 0.0
_C.MODEL.EDGEPN2DU.NEG_WEIGHT = 1.0

_C.MODEL.PN2 = CN()
_C.MODEL.PN2.NUM_INPUT = 25600
_C.MODEL.PN2.NUM_CENTROIDS = (10240, 1024, 128, 0)
_C.MODEL.PN2.RADIUS = (0.2, 0.3, 0.4, -1.0)
_C.MODEL.PN2.NUM_NEIGHBOURS = (64, 64, 64, -1)
_C.MODEL.PN2.SA_CHANNELS = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 512, 1024))
_C.MODEL.PN2.FP_CHANNELS = ((256, 256), (256, 128), (128, 128), (64, 64, 64))
_C.MODEL.PN2.NUM_FP_NEIGHBOURS = (0, 3, 3, 3)
_C.MODEL.PN2.SEG_CHANNELS = (128,)
_C.MODEL.PN2.DROPOUT_PROB = 0.5
_C.MODEL.PN2.LABEL_SMOOTHING = 0.0
_C.MODEL.PN2.NEG_WEIGHT = 1.0

# ---------------------------------------------------------------------------- #
# Solver (optimizer)
# ---------------------------------------------------------------------------- #

_C.SOLVER = CN()

# Type of optimizer
_C.SOLVER.TYPE = "Adam"

# Basic parameters of solvers
# Notice to change learning rate according to batch size
_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.WEIGHT_DECAY = 0.0

# Specific parameters of solvers
_C.SOLVER.RMSprop = CN()
_C.SOLVER.RMSprop.alpha = 0.9

_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.momentum = 0.9

_C.SOLVER.Adam = CN()
_C.SOLVER.Adam.betas = (0.9, 0.999)

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.MAX_EPOCH = 2

_C.SCHEDULER.TYPE = ""

_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 0
_C.SCHEDULER.StepLR.gamma = 0.1

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 1

# The period to save a checkpoint
_C.TRAIN.CHECKPOINT_PERIOD = 1
_C.TRAIN.LOG_PERIOD = 10
_C.TRAIN.FILE_LOG_PERIOD = 1000
# The period to validate
_C.TRAIN.VAL_PERIOD = 1
# Data augmentation. The format is "method" or ("method", *args)
# For example, ("PointCloudRotate", ("PointCloudRotatePerturbation",0.1, 0.2))
_C.TRAIN.AUGMENTATION = ()

_C.TRAIN.VAL_METRIC = "cls_acc"

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1

# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ""

# Data augmentation.
_C.TEST.AUGMENTATION = ()

_C.TEST.LOG_PERIOD = 10
_C.TEST.FILE_LOG_PERIOD = 1000

_C.TEST.TOPK = 10


def load_cfg_from_file(cfg_filename):
    """Load config from a file

    Args:
        cfg_filename (str):

    Returns:
        CfgNode: loaded configuration

    """
    with open(cfg_filename, "r") as f:
        cfg = load_cfg(f)

    cfg_template = _C
    cfg_template.merge_from_other_cfg(cfg)
    return cfg_template
