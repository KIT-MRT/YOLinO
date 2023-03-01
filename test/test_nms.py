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
import sys
import unittest

import numpy as np
import torch

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid import Grid
from yolino.postprocessing.nms import nms
from yolino.runner.evaluator import Evaluator
from yolino.utils.enums import Dataset, AnchorDistribution
from yolino.utils.test_utils import test_setup, unsqueeze

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


@unittest.skipIf(sys.version_info.major < 3, "not supported in python2")
class NmsTest(unittest.TestCase):
    prediction_grid: Grid

    def test_nms_on_dublicate(self):
        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 2, "num_predictors": 8,
                                           "eps": 107,
                                           "nxw": 0, "lw": 0, "anchors": str(AnchorDistribution.NONE)},
                          # level=Level.DEBUG
                          )

        dataset, _ = DatasetFactory.get(args.dataset, only_available=True, split=args.split, args=args,
                                        shuffle=False, augment=False)
        evaluator = Evaluator(args, prepare_forward=False, load_best_model=False,
                              anchors=dataset.anchors)  # we only pushed model.pth to git
        # prevent debug tracemalloc
        images, _, fileinfo, dupl, params = unsqueeze(dataset.__getitem__(0), dataset.__getitem__(1))

        predictions = torch.cat(
            [dataset.full_grid(train_vars=False).unsqueeze(0),
             dataset.full_grid(train_vars=False).unsqueeze(0)])

        preds_uv, gt_uv = evaluator.prepare_uv(
            preds=predictions[:, :, :, dataset.coords.get_position_of_training_vars()],
            grid_tensors=predictions, filenames=fileinfo, images=images)
        lines = evaluator.get_nms_results(filenames=fileinfo, preds_uv=preds_uv, gt_uv=gt_uv, images=images, epoch=0,
                                          tag="unittest", num_duplicates=int(sum(dupl["total_duplicates_in_image"])))

        for b in range(len(lines)):
            for i in range(0, len(lines[b]), 8):
                self.assertTrue(np.array_equal(lines[b, i], preds_uv[b, i]),
                                msg="Expected the first line of each cell (batch=%d, idx=%d) to just conf=1 and "
                                    "be the representative of the cell, but we have %s and expected %s"
                                    % (b, i, lines[b, i], preds_uv[b, i:i + 8]))
                for j in range(1, 8):
                    self.assertTrue(np.array_equal(lines[b, i + j], [*preds_uv[b, i + j][0:-1], 0]),
                                    msg="Expected the %dth line of each cell in batch=%d to get conf=0, but we have %s"
                                        % (i + j, b, lines[b, i + j]))


if __name__ == '__main__':
    unittest.main()
