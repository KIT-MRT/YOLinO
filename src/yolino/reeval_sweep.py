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
import os
from argparse import ArgumentParser

import numpy as np
import wandb
import yaml
from tqdm import tqdm
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.runner.evaluator import Evaluator
from yolino.utils.argparser import define_argparse
from yolino.utils.enums import TaskType, LINE, AnchorDistribution, AnchorVariables, ACTIVATION
from yolino.utils.general_setup import __set_cellsize__, __set_imgsize__, __set_seed__, __set_paths__, \
    __abs_paths__, set_progress_logger, __set_cmd_logger__, __set_hash__, __set_matching_gate__
from yolino.utils.gpu import getCuda
from yolino.utils.logger import Log
from yolino.utils.paths import Paths
from yolino.viz.translation import experiment_param_keys

if __name__ == "__main__":
    # Setup
    _, parser = define_argparse(name="reeval")
    parser: ArgumentParser
    parser.add_argument("--run", type=str, help="Provide wandb run id incl project name etc.")
    args = parser.parse_args()

    args.debug_tool_name = "reeval"

    __set_hash__(args)
    __set_cmd_logger__(args, args.debug_tool_name)
    set_progress_logger(args, task_type=TaskType.TEST, project_suffix="")

    args = __abs_paths__(args)
    args = __set_paths__(args)
    __set_seed__(args)
    args.cuda = getCuda(args)
    __set_imgsize__(args)
    __set_cellsize__(args)
    __set_matching_gate__(args)

    # if args.ignore_missing:
    #     raise NotImplementedError("Please use the full dataset and not --ignore_missing")
    if args.split == "train":
        ok = input("You evaluate the train set. Is this on purpose?")

    # Prepare yaml
    file = os.path.join(args.dvc, "manual_eval", f"{args.run.split('/')[-1]}.yaml")
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file), exist_ok=True)

    if os.path.exists(file):
        with open(file, "r") as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data is None:
            yaml_data = {}
    else:
        yaml_data = {}

    TAG = args.split

    # API interaction
    api = wandb.Api(timeout=19)
    run_data = api.run(os.path.join("argo_po_8p_dn19", args.run))
    if "malconfig" in run_data.tags:
        Log.malconfig("Original run is malconfigured.")
        exit(-1)

    args.linerep = LINE(run_data.config["linerep"])
    if type(run_data.config["activations"]) == str:
        args.activations = [ACTIVATION(a) for a in run_data.config["activations"].split(",")]
    else:
        print(run_data.config["activations"])
        print(type(run_data.config["activations"]))
        args.activations = [ACTIVATION(a) for a in run_data.config["activations"]]

    args.anchors = AnchorDistribution(run_data.config["anchors"])
    args.num_predictors = int(run_data.config["num_predictors"])

    if type(run_data.config["anchor_vars"]) == str:
        args.anchor_vars = [AnchorVariables(a) for a in run_data.config["anchor_vars"].split(",")]
    else:
        print(run_data.config["anchor_vars"])
        print(type(run_data.config["anchor_vars"]))
        args.anchor_vars = [AnchorVariables(a) for a in run_data.config["anchor_vars"]]

    name = run_data.name
    explicit_model = os.path.abspath(os.path.join(args.paths.checkpoints, "..", name, "best_model.pth"))

    Log.debug("----- PARAMETERS -----")

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
                Log.debug("%s: %s" % (k, v))
                yaml.safe_dump({k: str(v)}, f)

    Log.info("----- SETUP DONE -----\n")
    Log.upload_params(args)

    Log.warning(f"\n------------ {name} --------------")

    args.paths = Paths(args.dvc, args.split, explicit_model)
    if not os.path.exists(args.paths.model):
        Log.error(f"Could not find model for {name} at {explicit_model}")
        exit(1)

    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split=args.split, args=args,
                                         shuffle=False, augment=False)

    evaluator = Evaluator(args, load_best_model=True, anchors=dataset.anchors)

    yaml_data[name] = {}
    for i, data in enumerate(tqdm(loader)):
        # for i in range(len(dataset)):
        image, grid_tensor, fileinfo, duplicate_info, params = data  # unsqueeze(dataset.__getitem__(i))
        # TODO: this should be done inside, but the threads are annoying
        dataset.params_per_file.update({f: params for f in fileinfo})

        num_duplicates = int(sum(duplicate_info["total_duplicates_in_image"]).item())
        evaluator(image, grid_tensor, i * args.batch_size, fileinfo, apply_nms=False,
                  epoch=evaluator.forward.start_epoch, do_in_uv=False, num_duplicates=num_duplicates)
    scores = evaluator.publish_scores(epoch=evaluator.forward.start_epoch, tag=TAG)

    best_epoch = run_data.summary["epoch/best/val"]
    for kk in tqdm(scores, desc="Iterate Metrics"):
        k = os.path.join(kk, TAG)
        history = run_data.history()
        if k not in history:
            if "confusion" in k:
                continue

            if "dupl" not in k:
                Log.error(f"we are missing {k}")
                Log.error([k for k in history.keys() if not "gradients" in k and not "parameters" in k])
                exit(0)

            Log.warning(f"{k}: {np.nanmean(scores[kk]):.4f} [{history[k.replace('_dupl', '')][best_epoch]:.4f}]")
            yaml_data[name].update({k: float(np.nanmean(scores[kk]))})
            continue
        v = history[k][best_epoch]
        if np.nanmean(scores[kk]) != v:
            Log.error(f"We have different values for {k}: old={v :.4f}; "
                      f"new={np.nanmean(scores[kk]):.4f}")

    Log.scalars(tag=TAG + "_" + name, dict=yaml_data[name], epoch=0)
    with open(file, "w") as f:
        yaml.safe_dump(yaml_data, f)
