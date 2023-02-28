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
import matplotlib
import yaml

from yolino.dataset.dataset_factory import DatasetFactory

matplotlib.use('Agg')

import hashlib
import os
import time
from collections import namedtuple

import numpy as np
import random
import torch

from yolino.utils.argparser import generate_argparse
from yolino.viz.translation import experiment_param_keys
from yolino.utils.enums import TaskType, Logger, AnchorDistribution, LINE, AnchorVariables, ACTIVATION
from yolino.utils.gpu import getCuda
from yolino.utils.logger import Log
from yolino.utils.paths import Paths


def __push_commit__(root, ignore_dirty=False):
    import git
    repo = git.Repo(root, search_parent_directories=True)

    try:
        branch = repo.active_branch
    except TypeError as te:
        try:
            Log.debug("Are we in CI? %s" % te)
            branch = os.environ["CI_COMMIT_REF_NAME"]
            Log.debug("We will use $CI_COMMIT_REF_NAME=%s instead" % branch)
        except KeyError as ke:
            Log.error(te)
            Log.error(ke)
            branch = "none"

    params = {"yolino": {}}
    try:
        params["yolino"]["commit"] = repo.head.object.hexsha
        params["yolino"]["short_commit"] = repo.head.object.hexsha[0:8]
    except ValueError as e:
        Log.error(f"Git Repo read hex SHA: {e}")
        params["yolino"]["commit"] = "??"
        params["yolino"]["short_commit"] = "??"

    params["yolino"]["has_uncommitted"] = repo.is_dirty()
    params["yolino"]["branch"] = str(branch)

    if repo.is_dirty() and not ignore_dirty:
        import socket
        Log.warning(socket.gethostname())
        if not "hkn" in socket.gethostname():
            input("Your repo has uncommited changes, please commit first or ignore this warning and press any key.")
        else:
            raise ValueError("Git repo is dirty!")
    # Log.upload_params(params)
    # Log.debug(params)
    return params


def validate_args(args):
    n = len(args.training_variables)
    nw = n if args.weights is None else len(args.weights)
    nl = len(args.loss)
    na = len(args.activations)
    if not (nw == n and nl == n and na == n):
        Log.malconfig("We expect --weights (%d), --loss (%d) and --activations (%d) to have %d values, "
                      "exactly one value per training variable (%s). Weights=%s, Loss=%s, Activations=%s"
                      % (nw, nl, na, n, args.training_variables, args.weights, args.loss, args.activations))

    if not len(args.conf_match_weight) == 2:
        Log.malconfig("We expect exactly two --conf_match_weight, but got %s" % str(args.conf_match_weight))

    if args.anchors != AnchorDistribution.NONE:
        if args.linerep == LINE.POINTS:
            if AnchorVariables.POINTS not in args.anchor_vars:
                Log.malconfig("We can only use %s "
                              "anchors for linerep %s." % (AnchorVariables.POINTS, LINE.POINTS) +
                              "You chose %s and %s" % (args.anchor_vars, args.linerep))
        elif args.linerep == LINE.MID_LEN_DIR:
            Log.malconfig("We do not support length and angle definition anymore")
        elif args.linerep == LINE.MID_DIR:
            if AnchorVariables.POINTS in args.anchor_vars:
                Log.malconfig("We can only use %s, %s anchors for linerep %s. You chose %s"
                              % (AnchorVariables.DIRECTION, AnchorVariables.MIDPOINT, LINE.MID_DIR, args.anchor_vars))
        else:
            Log.malconfig("We cannot use anchors with %s, choose points or mld." % args.linerep)


def __set_imgsize__(args):
    max_size = DatasetFactory.get_max_image_size(dataset=args.dataset)
    if args.img_height is not None:
        Log.error(
            f"{args.dataset} has --img_size {max_size}. You chose --img_height {args.img_height}. Do not change the image size if not necessary!")
        Log.tag("wrong_img")
        args.img_size = DatasetFactory.get_img_size(dataset=args.dataset, img_height=args.img_height)
    else:
        args.img_size = max_size


def __set_matching_gate__(args):
    args.matching_gate_px = args.matching_gate * args.cell_size[0]
    Log.debug(f"Matching Gate: {args.matching_gate_px}")


def general_setup(name, config_file="params.yaml", ignore_cmd_args=False, alternative_args=None, setup_logging=True,
                  task_type: TaskType = TaskType.TRAIN, default_config="default_params.yaml",
                  show_params=False, project_suffix="", preloaded_argparse=None):
    # ignore_cmd_args: bool, set to True if you want to e.g. run a test and want the command line args to be set to an empty list.
    #    Params will still be retrieved from params.yaml and default config. You can also provde alternative_args as a list of strings e.g. ["--model", "yolo"].
    # setup_logging: bool, set to False if you would like to setup clearml or tensorboard with a specific ID e.g. retrieved from 
    #    the checkpoint; to do that call setup_progress_logger() 
    # task_type: TaskType enum, required for setting up the logger (setup_logging=True). This will be mapped to the clearml TaskTypes

    import sys

    print("\n\n----- START %s -----" % name)
    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print("-----------------")

    Log.debug("----- ARGPARSE -----")
    if preloaded_argparse is None:
        args = generate_argparse(name, config_file=config_file, default_config=default_config,
                                 ignore_cmd_args=ignore_cmd_args, alternative_args=alternative_args)
    else:
        args = preloaded_argparse

    if "SLURM_JOB_ID" in os.environ:
        args.slurm_job = os.environ["SLURM_JOB_ID"]

    params = __push_commit__(args.root)
    args.yolino = params["yolino"]
    if params["yolino"]["branch"] != "master":
        if args.tags is None:
            args.tags = []
        args.tags.append(params["yolino"]["branch"][0:10])

    Log.debug("----- LOGGER -----")
    __set_hash__(args)
    __set_cmd_logger__(args, name)
    if setup_logging:
        set_progress_logger(args, task_type, project_suffix)

    validate_args(args)

    Log.debug("----- MAKE PATHS ABS -----")
    args = __abs_paths__(args)
    args = __set_paths__(args)

    Log.debug("----- REPRODUCIBLE -----")
    __set_seed__(args)

    Log.debug("----- GPU -----")
    args.cuda = getCuda(args)

    Log.debug("----- IMAGE -----")
    __set_imgsize__(args)

    Log.debug("----- GRID -----")
    __set_cellsize__(args)
    __set_matching_gate__(args)

    if args.show_params or show_params:
        fnc = print
    else:
        fnc = Log.debug
    fnc("----- PARAMETERS -----")

    args.paths.create_dir(args.paths.experiment_dir)
    yaml_path = os.path.join(args.paths.experiment_dir, "latest_params.yaml")
    with open(yaml_path, "w") as f:
        Log.warning(f"Dump params to file://{yaml_path}")
        dirname = os.path.dirname(yaml_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for k, v in sorted(args.__dict__.items()):
            yaml.safe_dump({k: str(v)}, f)

    yaml_path = os.path.join(args.paths.experiment_dir, "latest_interesting_params.yaml")
    with open(yaml_path, "w") as f:
        Log.warning(f"Dump params to file://{yaml_path}")
        for k, v in sorted(args.__dict__.items()):
            if k in experiment_param_keys.keys():
                fnc("%s: %s" % (k, v))
                yaml.safe_dump({k: str(v)}, f)

    Log.info("----- SETUP DONE -----\n")
    Log.upload_params(args)
    return args


def __set_hash__(args):
    if True:
        from datetime import datetime
        now = datetime.now()
        args.id = now.strftime("%y-%b-%d_%H-%M-%S-%f")
    else:
        hash = hashlib.sha1()
        hash.update(str(time.time()).encode('utf-8'))
        args.id = hash.hexdigest()[:10]


def __set_paths__(args):
    args.paths = Paths(dvc=args.dvc, split=args.split, explicit_model=args.explicit_model,
                       debug_tool_name=args.debug_tool_name, id=args.id, keep_checkpoints=args.keep)
    return args


def __set_cellsize__(args):
    args.cell_size = [args.scale, args.scale]  # the models only support squared cells so far

    args.grid_shape = np.ceil(np.asarray(args.img_size) / np.asarray(args.cell_size)).astype(int)
    if np.any(np.asarray(args.img_size) % np.asarray(args.cell_size) != 0):
        Log.warning("Image of %s is not easily separable in the given cell size of %s."
                    % (args.img_size, args.cell_size))

    Log.debug("Grid: %sx%s, Cell: %sx%s" % (args.grid_shape[0], args.grid_shape[1],
                                            args.cell_size[0], args.cell_size[1]))


def __set_seed__(args):
    if not args.nondeterministic:
        Log.debug("[x] torch, [x] random, [x] numpy")

        torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

        random.seed(hash("setting random seeds") % 2 ** 32 - 1)

        np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)


def getExperimentDefinition(branch):
    Params = namedtuple('Params', ['dataset', 'line', 'predictors', 'upsampling'])

    splits = branch.split("_")
    if len(splits) < 4 or len(splits) > 5:
        Log.error("Invalid experiment code %s" % splits)
        return None

    params = Params(splits[0], splits[1], splits[2], "_".join(splits[3:]))
    return params


def __abs_paths__(args):
    Log.debug("Root: '%s' -> '%s'" % (args.root, os.path.abspath(args.root)))
    args.root = os.path.abspath(args.root)

    Log.debug("DVC: '%s' -> '%s'" % (args.dvc, os.path.abspath(args.dvc)))
    args.dvc = os.path.abspath(args.dvc)

    if args.explicit_model is not None:
        Log.debug("Model: '%s' -> '%s'" % (args.explicit_model, os.path.join(args.dvc, args.explicit_model)))
        args.explicit_model = os.path.join(args.dvc, args.explicit_model)

    # if not args.specs:
    #     args.specs = args.paths.generate_specs_file_name()
    # Log.debug("Specs: '%s' -> '%s'" % (args.specs, os.path.join(args.dvc, args.specs)))
    # args.specs = os.path.join(args.dvc, args.specs)

    if not os.path.isfile(args.darknet_cfg):
        Log.debug("We did not find the darknet config at %s. We try to find it in --root." % args.darknet_cfg)
        args.darknet_cfg = os.path.join(args.root, "src", "yolino", args.darknet_cfg)
        if not os.path.isfile(args.darknet_cfg):
            Log.error("We did not find the darknet config at the root path %s. Is this on purpose?" % args.darknet_cfg)
    if not os.path.isfile(args.darknet_weights):
        Log.debug("We did not find the darknet config at %s. We try to find it in --root." % args.darknet_weights)
        args.darknet_weights = os.path.join(args.root, "src", "yolino", args.darknet_weights)
        if not os.path.isfile(args.darknet_weights):
            Log.error(
                "We did not find the darknet weights at the root path %s. Is this on purpose?" % args.darknet_weights)
    return args


def __set_cmd_logger__(args, name):
    if args.loggers is None:
        args.loggers = []
    Log.setup_cmd(viz_level=args.level, setup_file_log=Logger.FILE in args.loggers,
                  log_file=os.path.join(args.dvc, "cmd_logs", args.id + ".log"),
                  name=name if "_" in args.id else str(args.id)[-10:0])


def set_progress_logger(args, task_type, project_suffix=""):
    from yolino.utils.logger import Log

    if args.loggers is None:
        args.loggers = []

    # TODO: get is_train from somewhere better 
    Log.setup(args, task_type, project_suffix)
