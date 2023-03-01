# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# This file is part of YOLinO.
#
# YOLinO is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# YOLinO is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# YOLinO. If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------- #
# ----------------------------- COPYRIGHT ------------------------------------ #
# ---------------------------------------------------------------------------- #
from enum import Enum


class BaseEnum(Enum):
    def __lt__(self, other):
        return self.value.__lt__(other.value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


class Network(BaseEnum):
    YOLO_CLASS = "yolo_class"


class AnchorVariables(BaseEnum):
    DIRECTION = "direction"
    MIDPOINT = "midpoint"
    POINTS = "points"


class AnchorDistribution(BaseEnum):
    NONE = "none"
    EQUAL = "equal"
    KMEANS = "k-means"


class Distance(BaseEnum):
    HAUSDORFF = "hausdorff"
    SQUARED = "squared"
    COSINE = "cosine"
    POINT = "point"
    EUCLIDEAN = "euclidean"


class Variables(BaseEnum):
    CLASS = "class"
    GEOMETRY = "geom"
    INSTANCE = "instance"
    FOLLOWER = "follower"
    CONF = "confidence"
    POSITION_IN_LINE = "position"  # is start/middle/end
    SAMPLE_ANGLE = "sample_angle"


class Dataset(BaseEnum):
    # ARGOVERSE = "argo"
    ARGOVERSE2 = "argo2"
    TUSIMPLE = "tusimple"
    CULANE = "culane"
    CIFAR = "cifar"
    CALTECH = "caltech"


class CoordinateSystem(BaseEnum):
    UV_CONTINUOUS = "(batch, instances, control points, ?)"
    UV_SPLIT = "(batch, line_segments, 2 * 2 + ?)"
    CELL_SPLIT = "(batch, cells, <=predictors, 2 * 2 + ?)"
    EMPTY = ""

    def __str__(self):
        return "The lines should have shape %s" % self.value


class Augmentation(BaseEnum):
    ROTATION = "rotation"
    CROP = "crop"
    NORM = "normalize"
    JITTER = "jitter"
    ERASING = "erasing"


class LOSS(BaseEnum):
    CROSS_ENTROPY_SUM = "ce_sum"
    CROSS_ENTROPY_MEAN = "ce"
    BINARY_CROSS_ENTROPY_SUM = "bce_sum"
    BINARY_CROSS_ENTROPY_MEAN = "bce_mean"
    MSE_MEAN = "mse_mean"
    MSE_SUM = "mse_sum"
    NORM_MEAN = "norm_mean"
    NORM_SUM = "norm_sum"


class LossWeighting(BaseEnum):
    FIXED = "fixed"
    LEARN_LOG = "learn_log"
    LEARN_LOG_NORM = "learn_log_norm"
    LEARN = "learn"
    FIXED_NORM = "fixed_norm"


class ACTIVATION(BaseEnum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    LINEAR = "linear"


class LINE(BaseEnum):
    EULER = "euler"
    MILEAN = "milean"
    ONE_D = "oned"
    POINTS = "points"
    POLAR = "polar"
    CLASS = "class"
    MID_LEN_DIR = "mld"  # midpoint, direction, length
    MID_DIR = "md"  # midpoint, direction


class Metric(BaseEnum):
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    # PR_CURVE = "pr"
    ACCURACY = "accuracy"
    CONFUSION = "confusion"
    TP = "tp"
    FN = "fn"
    TN = "tn"
    FP = "fp"
    RMSE = "rmse"
    MAE = "mae"


class Scheduler(BaseEnum):
    NONE = "none"


class Optimizer(BaseEnum):
    ADAM = "adam"
    SGD = "sgd"
    RMS_PROP = "rms_prop"
    ADA_DELTA = "ada_delta"


class Level(BaseEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class TaskType(BaseEnum):
    TRAIN = "train"
    PRED = "predict"  # with eval
    TEST = "test"
    PARAM_OPTIMIZATION = "params"


class Logger(BaseEnum):
    TENSORBOARD = "tb"
    CLEARML = "clearml"
    WEIGHTSBIASES = "wb"
    FILE = "file"


class ColorStyle(BaseEnum):
    RANDOM = "random"
    ID = "id"
    CLASS = "class"
    CONFIDENCE = "confidence"
    CONFIDENCE_BW = "confidence_bw"
    ORIENTATION = "orientation"
    UNIFORM = "uniform"
    ANCHOR = "anchor"


class ImageIdx(BaseEnum):
    DEFAULT = ""
    IMAGE = "0_image"
    LABEL = "1_label"
    ANCHOR = "1_anchor"
    CLASS = "1_class_label"
    GRID = "1_grid_label"
    PRED = "3_pred"
    LOSS = "3_loss"
    MATCH = "3_match"
    NMS = "4_nms"
    RESULT = "5_result"
