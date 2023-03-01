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
import math
import os
import sys
import unittest

import numpy as np
import torch
from yolino.dataset.culane_pytorch import CULaneDataSet
from yolino.model.model_factory import get_model
from yolino.utils.enums import Network
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Log
from yolino.utils.test_utils import get_default_param_path


@unittest.skipIf(sys.version_info.major < 3, "not supported in python2")
class DatasetTestEnum(unittest.TestCase):

    def test_cell_size(self):
        for enum in Network:
            if "yolo_class" not in str(enum):
                continue
            args = general_setup(name=str(enum),
                                 config_file=os.path.abspath("../src/yolino/model/cfg/darknet_params.yaml"),
                                 setup_logging=False, ignore_cmd_args=True,
                                 alternative_args=["--model", str(enum), "--root", "..",
                                                   "--dvc", "../../dvc_experiment_mgmt", "--level", "WARN",
                                                   "--ignore_missing", "--img_height", "128"],
                                 default_config=get_default_param_path())

            if os.path.isdir("/mrtstorage/datasets"):
                os.environ["DATASET_CULANE"] = "/mrtstorage/datasets/public/CULane"
                Log.warning("Test is using mrtstorage")
            else:
                local_dir = "/home/meyer/02_data/CULane"
                Log.warning("Testing local!")
                os.environ["DATASET_CULANE"] = local_dir

            data_set = CULaneDataSet("train", args)
            model = get_model(args, coords=data_set.coords)
            input = torch.rand(args.batch_size, 3, args.img_size[0], args.img_size[1])
            pred = model(input)

            cells = np.prod(np.divide(args.img_size, args.cell_size))
            expected_shape = [args.batch_size, cells, args.num_predictors, data_set.coords.num_vars_to_train()]
            self.assertTrue(np.all(np.equal(pred.shape, expected_shape)),
                            "Expected for %s shape=%s, but got %s" % (enum, str(expected_shape), str(pred.shape)))

            ratio = args.img_size[1] / args.img_size[0]
            smaller = math.sqrt(cells / ratio)
            self.assertTrue(np.all(args.grid_shape == [smaller, ratio * smaller]), enum)
