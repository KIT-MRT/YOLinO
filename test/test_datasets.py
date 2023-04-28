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
import unittest

import yaml
from yolino.dataset.dataset_factory import Dataset, DatasetFactory
from yolino.utils.enums import LINE
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Level, Log
from yolino.utils.system import get_system_specs
# @unittest.skipIf(get_system_specs()["user"] == "mrtbuild", "not to be run on CI")
from yolino.utils.test_utils import get_default_param_path


class DatasetTestEnum(unittest.TestCase):
    path = "tmp_params.yaml"

    params = {}
    params["rotation_range"] = 0.1
    params["img_height"] = 544
    params["model"] = "yolo_class"
    params["linerep"] = str(LINE.POINTS)
    params["num_predictors"] = 8
    params["learning_rate"] = 0.001
    params["root"] = "../"
    params["split"] = "train"

    user = get_system_specs()["user"]
    params["level"] = "WARN"
    params["max_n"] = 2

    def setUp(self) -> None:
        Log.setup_cmd(viz_level=Level.INFO, name=self._testMethodName + "_setup")

        DatasetTestEnum.params["ignore_missing"] = True
        if os.getenv("DATASET_CULANE") is None:
            os.environ["DATASET_CULANE"] = "/mrtstorage/datasets/public/CULane"
            DatasetTestEnum.params["ignore_missing"] = False
        if os.getenv("DATASET_TUSIMPLE") is None:
            os.environ["DATASET_TUSIMPLE"] = "/mrtstorage/datasets/public/tusimple_lane_detection"
            DatasetTestEnum.params["ignore_missing"] = False

        return super().setUp()

    def dumpdata(self, dataset, log_dir=None):
        if log_dir is None:
            log_dir = dataset

        tmp = DatasetTestEnum.params
        tmp["dataset"] = dataset
        tmp["log_dir"] = log_dir + "_po_8p_dn19"
        tmp["dvc"] = "dvc"

        with open(DatasetTestEnum.path, "w") as f:
            yaml.dump(tmp, f)

    def testCULaneFactory(self):
        self.dumpdata("culane")
        args = general_setup(self._testMethodName, os.path.abspath(DatasetTestEnum.path), ignore_cmd_args=True,
                             setup_logging=False, default_config=get_default_param_path())

        dataset, loader = DatasetFactory.get(args.dataset, only_available=False, split=args.split, args=args,
                                             shuffle=True, augment=False)
        self.assertEqual(dataset.enum, Dataset.CULANE)
        self.assertIsNotNone(loader)

    def testTusFactory(self):
        self.dumpdata("tusimple")
        args = general_setup(self._testMethodName, os.path.abspath(DatasetTestEnum.path), ignore_cmd_args=True,
                             setup_logging=False, show_params=False, default_config=get_default_param_path())
        dataset, loader = DatasetFactory.get(args.dataset, only_available=False, split=args.split, args=args,
                                             shuffle=True, augment=False)
        self.assertEqual(dataset.enum, Dataset.TUSIMPLE)
        self.assertIsNotNone(loader)

    @unittest.skipIf("RUNNING_IN" in os.environ, "Do not run in docker")
    def testArgo2Factory(self):
        DatasetTestEnum.params["img_height"] = 1536
        self.dumpdata("argo2")
        args = general_setup(self._testMethodName, os.path.abspath(DatasetTestEnum.path), ignore_cmd_args=True,
                             setup_logging=False, show_params=False, default_config=get_default_param_path())
        dataset, loader = DatasetFactory.get(args.dataset, only_available=False, split=args.split, args=args,
                                             shuffle=True, augment=False)
        self.assertEqual(dataset.enum, Dataset.ARGOVERSE2)
        self.assertIsNotNone(loader)

    def testCulaneVal(self):
        DatasetTestEnum.params["split"] = "val"
        self.dumpdata("culane")
        args = general_setup(self._testMethodName, os.path.abspath(DatasetTestEnum.path), ignore_cmd_args=True,
                             setup_logging=False, default_config=get_default_param_path())

        dataset, loader = DatasetFactory.get(args.dataset, only_available=False, split=args.split, args=args,
                                             shuffle=True, augment=False)
        self.assertEqual(dataset.enum, Dataset.CULANE)
        self.assertIsNotNone(loader)

    def testCULaneFactoryNotAvailable(self):
        self.dumpdata("culane")
        args = general_setup(self._testMethodName, os.path.abspath(DatasetTestEnum.path), ignore_cmd_args=True,
                             setup_logging=False, task_type=None, default_config=get_default_param_path())

        dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split=args.split, args=args,
                                             shuffle=True, augment=False)
        self.assertEqual(dataset.enum, Dataset.CULANE)
        self.assertIsNotNone(loader)
