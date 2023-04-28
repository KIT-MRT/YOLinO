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

import torch
import yaml
from yolino.utils.enums import Network, Dataset, LINE, AnchorDistribution
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Log
from yolino.utils.system import get_system_specs


def get_tmp():
    return os.path.abspath("../test/dvc")


def get_param_path():
    return os.path.join(get_tmp(), "params.yaml")


def get_default_param_path():
    return os.path.join(get_tmp(), "default_params.yaml")


def get_param_vals(level=None, dataset=str(Dataset.CULANE)):
    params = {}
    params["rotation_range"] = 0.05
    params["crop_range"] = 0.3
    if dataset == str(Dataset.TUSIMPLE):
        params["img_height"] = 320
    else:
        params["img_height"] = 288
    params["model"] = str(Network.YOLO_CLASS)
    params["linerep"] = str(LINE.POINTS)
    params["num_predictors"] = 8
    # params["learning_rate"] = 0.001
    params["confidence"] = 0.5
    params["lw"] = 1
    params["mpxw"] = 3
    params["min_samples"]: 1
    params["eps"]: 40
    params["nxw"] = 2
    params["anchors"] = str(AnchorDistribution.NONE)
    params["best_mean_loss"] = True
    params["batch_size"] = 4
    params["root"] = "../"
    params["dvc"] = "dvc/"
    params["split"] = "train"
    # params["activation"] = "sigmoid"
    params["retrain"] = True
    params["loading_workers"] = 1
    params["ignore_missing"] = True
    params["darknet_cfg"] = 'model/cfg/darknet19_448_d2.cfg'

    user = get_system_specs()["user"]
    if level:
        params["level"] = str(level)
    else:
        params["level"] = "WARN"  # "INFO" if (user == "mrtbuild") else
    params["max_n"] = 1000
    return params


def test_setup(name, dataset, log_dir=None, level=None, additional_vals=None,
               show_params=False, config_file=None):
    params = {}

    params.update(get_param_vals(level, dataset))
    if additional_vals:
        params.update(additional_vals)

    if config_file is not None:
        with open(config_file, "r") as f:
            params.update(yaml.safe_load(f))

    if os.path.isdir("/mrtstorage/datasets"):
        os.environ["DATASET_CULANE"] = "/mrtstorage/datasets/public/CULane"
        Log.warning("Test is using mrtstorage")
    else:
        local_dir = "/home/meyer/02_data/CULane"
        Log.warning("Testing local!")
        os.environ["DATASET_CULANE"] = local_dir
        params["ignore_missing"] = True

    if not os.path.isdir(os.environ["DATASET_CULANE"]):
        raise FileNotFoundError(
            "Neither '/mrtstorage/datasets/public/CULane' nor '/home/meyer/02_data/CULane' could be found")

    if log_dir is None:
        log_dir = dataset

    params["dataset"] = dataset
    params["log_dir"] = log_dir + "_po_8p_dn19"

    dumpdata(params)

    return general_setup(name, config_file=get_param_path(), ignore_cmd_args=True, setup_logging=False,
                         show_params=show_params, default_config=get_default_param_path())


def dumpdata(params):
    if not os.path.exists(os.path.dirname(get_param_path())):
        os.makedirs(os.path.dirname(get_param_path()))

    for k, v in params.items():
        if type(v) == list:
            string = ""
            for i, vi in enumerate(v):
                string += str(vi) + ("," if i < len(v) - 1 else "")
            # string += "]"
            params[k] = string

    with open(get_param_path(), "w") as f:
        yaml.dump(params, f)


def unsqueeze(*batches):
    images = torch.tensor([])
    gts = torch.tensor([])
    names = []
    duplicates = {}
    params = {}
    for batch in batches:
        if len(batch) == 0:
            continue
        img, gt, name, dupl, param = [torch.unsqueeze(set, dim=0) if type(set) == torch.Tensor else set for set in
                                      batch]
        images = torch.cat([images, img], dim=0)
        gts = torch.cat([gts, gt], dim=0)
        names.append(name)

        for k in dupl:
            if not k in duplicates:
                duplicates[k] = []

            duplicates[k].append(dupl[k])

        for k in param:
            if not k in params:
                params[k] = []

            params[k].append(param[k])
    return images, gts, names, duplicates, params
