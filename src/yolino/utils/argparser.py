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
import argparse
import os
import pwd

from yolino.utils.enums import Dataset, Network, Level, Logger, Optimizer, LINE, LOSS, ACTIVATION, Scheduler, Metric, \
    Augmentation, Variables, Distance, LossWeighting, AnchorDistribution, AnchorVariables
from yolino.utils.logger import Log


class AbstractParser(argparse.Action):
    def parse_fct(self, namespace, values, variable_type):  #
        if values == "":
            values = list()
        elif "," in values:
            values = list(map(variable_type, values.replace(" ", "").split(",")))
        else:
            values = list(map(variable_type, values.replace(",", "").split(" ")))
        setattr(namespace, self.dest, values)


class ParseActivation(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, ACTIVATION)


class ParseLoss(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, LOSS)


class ParseAnchorVariables(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, AnchorVariables)


class ParseBool(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):

        if values == "":
            values = True  # use as a flag
        elif values.lower() == "false":
            values = False
        elif values.lower() == "true":
            values = True
        else:
            raise ValueError(values)

        setattr(namespace, self.dest, values)


class ParseVariables(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, Variables)


class ParseAugmentation(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, Augmentation)


class ParseWeight(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        if "calculate" in values:
            self.parse_fct(namespace, values, str)
        else:
            self.parse_fct(namespace, values, float)


class ParseFloat(AbstractParser):
    def __call__(self, parser, namespace, values, option_string=None):
        self.parse_fct(namespace, values, float)


def generate_argparse(name, config_file="params.yaml", default_config="default_params.yaml",
                      ignore_cmd_args=False, alternative_args=[], preloaded_argparse=None):
    CONFIG_AVAILABLE, parser = define_argparse(config_file, default_config, name, preloaded_argparse)

    if not CONFIG_AVAILABLE:
        import yaml
        with open(config_file) as f:
            data = yaml.safe_load(f)
            str_data = []
            for k, v in data.items():

                if v == False:  # remove flag
                    continue
                elif v == True:
                    str_data.append("--" + k)  # set flag
                    continue

                str_data.append("--" + k)
                str_data.append(str(v))
            args = parser.parse_args(str_data)
    elif ignore_cmd_args:  # inside a unittest we do not read the cmd args
        args, _ = parser.parse_known_args(alternative_args)
    else:
        args = parser.parse_args()

    args.debug_tool_name = "_".join(name.lower().split(" "))
    return args


def define_argparse(config_file="params.yaml", default_config="default_params.yaml", name="no_name",
                    preloaded_argparse=None):
    CONFIG_AVAILABLE, parser = setup_argparse(config_file, default_config, name, preloaded_argparse)
    # ------ Invidivual Run Params ---------
    run_group = parser.add_argument_group("Invidiual Run Parameters")
    # parser.add_argument("--input", type=str,
    #                     help="Folder containing the input data. Can be empty if the default dataset path should be used. If it is presplit into train, val, test folder "
    #                          "add --presplit otherwise we split 80:20. You should provide both an 'images' "
    #                          "and 'labels' folder below the root or the split.")
    add_dataset(run_group)
    # parser.add_argument("--specs", type=str,
    #                     help="Provide the path to the specs file e.g. specs/culane_train_specs.yaml. "
    #                          "If relative it will be expected to lie within --dvc.")
    add_root(run_group)
    add_dvc(run_group)
    run_group.add_argument("--gpu", action="store_true", help="Enable GPU usage")  # TODO: merge with self.cuda
    run_group.add_argument("--gpu_id", type=int, help="Provide GPU ID for CUDA_VISIBLE_DEVICES.", default=-1)
    run_group.add_argument("--nondeterministic", action="store_true",
                           help="Do not set a deterministic seed; ATTENTION: bad reproducibility")
    add_ignore_missing(run_group)
    add_loading_workers(run_group)
    run_group.add_argument("--show_params", action="store_true",
                           help="Flag to show the chosen parameters of the script.")
    # Evaluation / Prediction
    file_group = parser.add_argument_group("File Handling")
    # parser.add_argument('--every_n', type=int, default=50, help='plots or saves every n-th sample')
    add_max_n(file_group)
    # parser.add_argument("--enhance", action="store_true", help="Only add new files, nothing will be deleted.")
    add_plot(file_group)
    add_explicit(file_group)
    # parser.add_argument('--out', type=str, default="metrics.json",
    #                     help='Path to eval file inside the dvc folder. (dvc will be prepended)')
    # Logging
    log_group = parser.add_argument_group("Logging")
    # parser.add_argument("--logging_frequency", type=int, default=1000,
    #                     help="Iterations to wait before logging to logger")
    add_level(log_group)
    add_loggers(log_group)
    add_tags(log_group)
    log_group.add_argument("--resume_log", action="store_true", help="Resume logging jobs if available.\n"
                                                                     "Wandb: https://docs.wandb.ai/ref/python/init resume='auto'.\n"
                                                                     "ClearML: https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit continue_last_task=True.")
    # ------ Experiment Params ---------
    file_group.add_argument("--log_dir", type=str, required=True,
                            help="Name of the experiment e.g. tus_po_8p_dn19_up. Should also be the branch name and the folder name.")
    # Dataset
    dataset_group = parser.add_argument_group("Dataset")
    # parser.add_argument("--fixed_crop_range", default=[-1, -1, -1, -1], nargs=4, type=int,
    #                     help="Crop the images by a defined window: [left, upper, right, lower]")
    # parser.add_argument("--presplit", action="store_true",
    #                     help="Add if your dataset is split into train, val and test set")
    # parser.add_argument("--classes", action="store_true", help="Dataset contains classes")
    add_img_height(dataset_group)
    add_split(dataset_group)
    add_subsample(dataset_group)
    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=Network, choices=list(Network), default=Network.YOLO_CLASS,
                             help="Provide network model name")
    model_group.add_argument("--darknet_cfg", type=str, default="model/cfg/darknet19_448_d2.cfg",
                             help="Path to the darknet config. Will be appended to the root path.")
    model_group.add_argument("--darknet_weights", type=str, default="model/cfg/darknet19_448.weights",
                             help="Path to the darknet weights. Will be appended to the root path. Use e.g. model/cfg/darknet19_448.weights.")
    model_group.add_argument("--linerep", type=LINE, choices=list(LINE), required=True,
                             help="Provide the line representation.")
    add_num_predictors(model_group)
    model_group.add_argument("--activations", type=str, required=True,
                             help="Provide the activation for each block in the training variables. Choose from %s" % [
                                 a.value for a in ACTIVATION],
                             action=ParseActivation)
    model_group.add_argument("--training_variables", type=str, required=True,
                             help="Provide variables to be predicted by the network. Remaining will only be used for visualization, but not learned"
                                  "Choose from %s" % [a.value for a in Variables], action=ParseVariables)
    add_scale(model_group)
    # parser.add_argument("--grid_shape", type=int, nargs=2, help="The number of cells per image [rows,cols]")
    # parser.add_argument("--cell_size", type=int, nargs=2, help="The number of pixels per cell [rows,cols]. Will overwrite --grid_shape.")
    # Augmentation
    augment_group = parser.add_argument_group("Augmentation")
    augment_group.add_argument("--crop_range", type=float, required=True,
                               help="Range to sample crop portion from for the augmentation during training")
    augment_group.add_argument("--rotation_range", type=float, required=True,
                               help="Range of radians to sample the rotation from for the augmentation during training")
    augment_group.add_argument("--augment", type=str, action=ParseAugmentation, required=True,
                               help="Provide list of all augmentations to apply. The methods will be applied in order."
                                    "Choose from %s" % [a.value for a in Augmentation])
    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--retrain", action="store_true", help="Does not load weights even if available")
    add_batch_size(train_group)
    train_group.add_argument("--decay_rate", type=float, default=0.0001,
                             help="Decay rate for Adam Optimizer")
    train_group.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    train_group.add_argument("--epoch", type=int, required=True, help="Maximum epoch")
    train_group.add_argument("--checkpoint_iteration", type=int, default=10,
                             help="Save separate .pth file after several epochs")
    add_keep(train_group)
    train_group.add_argument("--eval_iteration", type=int, default=3,
                             help="Run model on full validation set after several epochs")
    train_group.add_argument("--full_eval", action="store_true", help="Apply UV evaluation also during training")
    train_group.add_argument("--optimizer", type=Optimizer, choices=list(Optimizer), default=Optimizer.ADAM,
                             help="Speciy Optimizer")
    train_group.add_argument("--scheduler", type=Scheduler, choices=list(Scheduler), default=Scheduler.NONE,
                             help="Specify Scheduler for the learning rate")
    train_group.add_argument("--patience", type=int, default=5,
                             help="Number of validation epochs to wait for early convergence. "
                                  "If the last p epochs are worse than a previous one, we stop.")
    train_group.add_argument("--best_mean_loss", action=ParseBool, required=True,
                             help="If patience is set, activate this in order to regard the mean of the losses"
                                  " for convergence instead of the actual sum loss used for backpropagation.")
    train_group.add_argument("--earliest_stop", type=int, default=5,
                             help="Minimum number of epochs to run before appyl early stopping criteria.")
    train_group.add_argument("--learning_rate", required=True, type=float, help="The learning rate for the training")
    # Loss
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument("--loss", type=str, required=True,
                            help="Specify loss functions. For each entry in --training_variables a loss has to be specified"
                                 "Choose from %s" % [a.value for a in LOSS], action=ParseLoss)
    loss_group.add_argument("--weights", type=str, action=ParseWeight,
                            help="Specify the initial weights of the loss function. By default each weight is set to 1. "
                                 "If specified, for each entry in --training_variables a loss weight has to be given."
                                 "The weights will be adapted acording the loss' variance (Kendall et al.) or stay fixed "
                                 "depending on your --loss_weight_strategy. Fixed weights are recommended to be 4,1 from "
                                 "geom:conf. Pass `calculate` to let us calculate the weights.")
    loss_group.add_argument("--loss_weight_strategy", type=LossWeighting, required=True, choices=list(LossWeighting),
                            help="As weight for the multi-task loss terms we either learn log(sigma^2) or sigma of the "
                                 "data distribution for each task or set the provided values fixed."
                                 "The log version should be more stable compared to pure sigma according to Kendall et al. ")
    loss_group.add_argument("--conf_match_weight", type=str, default=[5, 1], action=ParseWeight, required=True,
                            help="Specify the weight for all matches on the conf loss and the unmatched. "
                                 "Helps with the imbalance between number of matched predictors and unmatched predictors."
                                 "Pass `calculate` to let us calculate the weights.")
    loss_group.add_argument("--match_by_conf_first", action=ParseBool,
                            help="Apply two-stage matching. 1. Only match predictions with confidence > --confidence. "
                                 "2. Match all remaining. This only affects the loss matching, not the evaluation.")
    loss_group.add_argument("--use_conf_in_loss_matching", action=ParseBool,
                            help="Use the confidence variable in matching line segments for the loss. "
                                 "Only for --anchors=none.")
    loss_group.add_argument("--association_metric", type=Distance, choices=list(Distance), default=Distance.EUCLIDEAN,
                            help="Specify metric to associate two line segments geometrically. "
                                 "Will determine the responsibility in the loss function.", required=True)
    # Anchors
    anchor_group = parser.add_argument_group("Anchors")
    add_anchors(anchor_group)
    anchor_group.add_argument("--offset", action=ParseBool, required=True,
                              help="Predict offset values to an anchor instead of absolute values. "
                                   "--anchors has to be set to something other than 'none'.")
    anchor_group.add_argument("--anchor_vars", type=str, action=ParseAnchorVariables, required=True,
                              help="Specify which aspect of a line will be used to define an anchor. You can en-/disable anchors "
                                   "and define the distribution with --anchors. "
                                   "Choose from %s" % [a.value for a in AnchorVariables])
    # # NMS
    nms_group = parser.add_argument_group("Non-Maximum-Suppression")
    nms_group.add_argument("--nms", action="store_true", help="Apply non maximum suppression")
    nms_group.add_argument("--eps", type=float, required=True,  # 0.02,
                           help="NMS: Epsilon for DBSCAN")
    nms_group.add_argument("--min_samples", type=int, required=True,
                           help="NMS: Minimum number of samples required for main points in DBSCAN Cluster")
    nms_group.add_argument("--nxw", type=float, required=True,  # 0.05,
                           help="NMS: Weight for the normed x-widths in the DBSCAN Clustering")  # TODO: WHAT?!
    nms_group.add_argument("--confidence", type=float, required=True,  # 0.9,
                           help="Confidence threshold", )
    nms_group.add_argument("--lw", type=float, required=True,  # 0.05,
                           help="NMS: Weight for the length in the DBSCAN Clustering")
    nms_group.add_argument("--mpxw", type=float, required=True,  # 0.016,
                           help="NMS: Weight for the midpoint in the DBSCAN Clustering")
    # Eval
    eval_group = parser.add_argument_group("Evaluation")
    # parser.add_argument("--sample_distance", type=int, default=1,
    #                     help="Distance to sample points on the line segments for point based evaluation")
    eval_group.add_argument("--metrics", type=Metric, nargs="*", default=list(Metric),
                            choices=list(Metric), help="Select metrics that should be calculated")
    eval_group.add_argument("--matching_gate", type=float, required=True,
                            help="Provide ratio of cell_sizes to be included in the matching of start-/endpoint in the "
                                 "evaluation. E.g. in order to match with the squared association_metric a specific number "
                                 "of cells 'no_cells' each with a 'cell_size' choose matching_gate=cell_size "
                                 "and (no_cells * cell_size)^2 will be the radius in px.")
    eval_group.add_argument("--explicit_model", type=str,  # default="log/checkpoints/model.pth",
                            help="Provide a path to an alternative model.pth file. By default we use "
                                 "log/checkpoints/model.path (training continued) or log/checkpoints/best_model.pth (prediction)"
                                 "in the --dvc folder.")
    # Tusimple Benchmark / Connection
    postproc_group = parser.add_argument_group("Postprocessing")
    postproc_group.add_argument("--min_segments_for_polyline", type=int, required=True,
                                help="Minimum number of segments to build a valid polyline in the line fitting for tusimple")
    postproc_group.add_argument("--adjacency_threshold", required=True, type=float)  # , default=0.75 * 32 * 32)
    # parser.add_argument("--label_idx", type=int)
    # parser.add_argument("--nms_distance_threshold")
    # parser.add_argument("--only_class", type=int)
    # parser.add_argument("--y_weighted")
    # parser.add_argument("--num_labellines", type=int)
    return CONFIG_AVAILABLE, parser


def add_batch_size(parser):
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")


def add_keep(parser):
    parser.add_argument("--keep", action="store_true",
                        help="Set true if you would like to keep the checkpoint in log/checkpoints/<ID>.pt")


def add_loading_workers(run_group):
    run_group.add_argument("--loading_workers", type=int, required=True,
                           help="How many cpu cores (?) / threads (?) will be used by torch dataloader")


def add_anchors(parser):
    parser.add_argument("--anchors", type=AnchorDistribution, choices=list(AnchorDistribution),
                        help="Anchors fix the GT to specific positions based on th anchors position (--anchors_vars). "
                             "Set to 'none' to allow the network to learn specific predictors on random positions. "
                             "ATTENTION: this heavily increases computation time, as each loss calculation needs a "
                             "matching. Other choices define the distribution of the anchors. '%s' loads the values "
                             "from the anchors file." % AnchorDistribution.KMEANS)  # TODO where is the anchors file


def add_scale(parser):
    parser.add_argument("-up", "--scale", type=int, choices=[32, 16, 8], required=True,
                        help="Provide the downsampling factor we should use to add upsampling layer. "
                             "With -up=32 the output will have 32x32 px per cell.")


def add_img_height(parser):
    parser.add_argument("--img_height", type=int, required=False,
                        help="Expected image height for the training, width will be calculated. "
                             "Input will be cropped/scaled to that if not valid.")


def setup_argparse(config_file="params.yaml", default_config="default_params.yaml", name="no_name",
                   preloaded_argparse=None):
    CONFIG_AVAILABLE = True
    print("Config file: file://%s" % os.path.abspath(config_file))
    print("Default config file: file://%s" % os.path.abspath(default_config))
    import argparse
    if preloaded_argparse is None or not isinstance(preloaded_argparse, argparse.ArgumentParser):
        try:
            # Test if configargparse is available (not avail on unittests in CI)
            import configargparse
            parser = configargparse.ArgumentParser(name, default_config_files=[default_config])
        except ModuleNotFoundError as ex:
            import socket
            host = socket.gethostname()
            user = pwd.getpwuid(os.getuid())[0]

            if "mrtbuild" != user:  # this should only happen in CI
                Log.error("%s with %s: No configargparse available" % (host, user))
                raise ex

            Log.warning("%s with %s: No configargparse available" % (host, user))
            CONFIG_AVAILABLE = False

            parser = argparse.ArgumentParser(name)
    else:
        parser = preloaded_argparse
    parser.add_argument("-c", "--config", required=False, is_config_file=True, default=config_file,
                        help="Config file path e.g. params.yaml")
    return CONFIG_AVAILABLE, parser


def add_plot(parser):
    parser.add_argument("--plot", action="store_true", help="Plot a lot of debug images")


def add_loggers(parser):
    parser.add_argument("--loggers", type=Logger, choices=list(Logger), nargs="*",
                        help="Specify logging services")


def add_tags(parser):
    parser.add_argument("--tags", type=str, nargs="*", help="Add tags for the experiments")


def add_ignore_missing(run_group):
    run_group.add_argument("--ignore_missing", action="store_true",
                           help="Do not abort on missing files. Use this for test environments with only subset of the dataset.")


def add_root(parser):
    parser.add_argument("--root", default="../yolino", help="Folder containing source code e.g. src")


def add_subsample(parser):
    parser.add_argument("-sdr", "--subsample_dataset_rhythm", type=int, default=-1,
                        help="Especially training with sequence based datasets might benefit from using only a subset of the images. The subsample rhythm will be used to select those.")


def add_max_n(parser):
    parser.add_argument('--max_n', type=int, default=-1, help='Runs for only max_n images')


def add_explicit(parser):
    parser.add_argument("--explicit", type=str, nargs="+",
                        help="Provide explicit filenames from the set that is chosen with --split, e.g. " \
                             "'driver_23_30frame/05161540_0603.MP4/05275.jpg' to process. This will ignore --max_n.")


def add_dvc(parser):
    parser.add_argument("--dvc", type=str, default=".",
                        help="DVC folder where the log, checkpoints and eval data will be stored.")


def add_level(parser):
    parser.add_argument("--level", type=Level, choices=list(Level), default=Level.INFO,
                        help="Choosing logging level. Only the chose and more severe levels are vizualized.")


def add_split(parser):
    parser.add_argument("--split", required=True, help="Provide split folder name (train, val, test)",
                        choices=["train", "val", "test"])


def add_dataset(parser):
    parser.add_argument("--dataset", type=Dataset, choices=list(Dataset), required=True,
                        help="Specify the dataset here")


def add_num_predictors(parser):
    parser.add_argument("--num_predictors", type=int, required=True,
                        help="The number of allowed predictors for each cell")
