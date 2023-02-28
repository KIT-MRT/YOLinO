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

    # def test_nms(self):
    #
    #     args = test_setUp(self._testMethodName, str(Dataset.TUSIMPLE),
    #                       additional_vals={"batch_size": 2},
    #                       level=Level.DEBUG
    #                       )
    #
    #     prediction = np.expand_dims(np.asarray([
    #         [64., 180., 60., 187., 0.7],
    #         [63., 163., 60., 179., 0.6],
    #         [71., 230., 63., 216., 0.99],  # match
    #         [61., 224., 59., 212., 1],  # match
    #         [104., 384., 96., 365., 1]
    #     ]), axis=0)
    #
    #     # newlines = nms(prediction, grid_shape=[10, 20], cell_size=32, confidence_threshold=0.5, orientation_weight=1,
    #     #                length_weight=1, midpoint_weight=2, epsilon=250, min_samples=2)  # with factor 5 in weights
    #     # newlines = nms(deepcopy(prediction), grid_shape=[10, 20], cell_size=32, confidence_threshold=0.5, orientation_weight=1,
    #     #                length_weight=1, midpoint_weight=2, epsilon=70, min_samples=2)  # with softmax
    #     # newlines = nms(deepcopy(prediction), grid_shape=[10, 20], cell_size=32, confidence_threshold=0.5,
    #     #                orientation_weight=1, length_weight=1, midpoint_weight=2, epsilon=311, min_samples=2)  # pure weights
    #     newlines, reduced = nms(deepcopy(prediction), grid_shape=[10, 20], cell_size=[32, 32], confidence_threshold=0.5,
    #                             orientation_weight=1, length_weight=1, midpoint_weight=2, epsilon=2,
    #                             min_samples=2, plot_debug=True)  # normalized weights per dimension
    #
    #     self.assertEqual(np.sum(newlines[:, :, -1] == 0), 2)
    #     self.assertEqual(np.sum(newlines[:, :, -1] == 1), 1)
    #
    #     for i, p in enumerate(newlines[0]):
    #         if i == 2:
    #             self.assertEqual(p[-1], prediction[0, 2:4, -1].max(),
    #                              msg="We expect the conf to be set to the max of the cluster")
    #             self.assertTrue(np.array_equal(p[0:-1],
    #                                            np.average(prediction[0, 2:, 0:4], axis=0,
    #                                                       weights=np.power(prediction[0, 2:, -1], 10))),
    #                             msg="We expect it to be the mean of the %dth and %dth prediction %s, but we have %s"
    #                                 % (2, 4, prediction[0, 2:, 0:4].mean(axis=0), p[0:-1]))
    #         elif i == 3 or i == 4:
    #             self.assertEqual(p[-1], 0, msg="We expect the conf to be set to 0")
    #             self.assertTrue(np.array_equal(p[0:-1], prediction[0, i, 0:-1]),
    #                             msg="We expect it to be the same as before")
    #         else:
    #             self.assertTrue(np.array_equal(p, prediction[0, i]))
    #
    # def test_nms_on_reduced_train_data(self):
    #
    #     args = test_setUp(self._testMethodName, str(Dataset.CULANE),
    #                       additional_vals={"batch_size": 1, "num_predictors": 4,
    #                                        "training_variables": [Variables.GEOMETRY, Variables.CONF],
    #                                        "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID],
    #                                        # "confidence": 0.5, "eps": 30
    #                                        },
    #                       # level=Level.DEBUG
    #                       )
    #
    #     predictions = np.asarray([[15., 10, 25, 20, 0.9], [35, 30, 45, 40, 0.1],
    #                               [0, 0, 10, 10, 0.1], [5, 0, 5, 10, 0.8]])
    #     preds_uv = np.tile(predictions, (args.batch_size, 1, 1))
    #
    #     lines, reduced = nms(deepcopy(preds_uv), grid_shape=args.grid_shape, cell_size=args.cell_size,
    #                          confidence_threshold=args.confidence, orientation_weight=args.nxw,
    #                          length_weight=args.lw,
    #                          midpoint_weight=args.mpxw, epsilon=args.eps, min_samples=args.min_samples,
    #                          weight_samples=False)
    #     for i in [1, 2, 3]:
    #         self.assertTrue(np.array_equal(lines[0, i], [*predictions[i, 0:4], 0]),
    #                         msg="Expected the %dth line to just get 0 conf as it is below the threshold, but we have %s" %
    #                             (i, lines[0, i]))
    #     self.assertTrue(
    #         np.array_equal(lines[0, 0],
    #                        [*np.average(predictions[[0, 3], 0:4], weights=predictions[[0, 3], -1] ** 10, axis=0), 0.9]),
    #         msg="Expected the first line (the representative) to get the max conf of 0.9 and the average values of "
    #             "0 and 3, but we have %s" % (lines[0, 0]))

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

    def test_nms(self):
        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 2, "num_predictors": 8,
                                           "eps": 107,
                                           "nxw": 0, "lw": 0, "anchors": str(AnchorDistribution.NONE)},
                          # level=Level.DEBUG
                          )

        random = np.random.random(size=(100, 4)) * 1000
        zeros = np.zeros((100, 1))
        preds = np.concatenate([random, zeros], axis=1).reshape((1, 100, 5))

        preds, reduced = nms(preds, grid_shape=args.grid_shape, cell_size=args.cell_size,
                             confidence_threshold=args.confidence,
                             orientation_weight=args.nxw, length_weight=args.lw, midpoint_weight=args.mpxw,
                             epsilon=args.eps, min_samples=args.min_samples, plot_debug=False, normalize=False,
                             weight_samples=False)

    # def test_eval(self):
    #     args = test_setUp(self._testMethodName, str(Dataset.TUSIMPLE),
    #                       additional_vals={"batch_size": 2, "num_predictors": 8,
    #                                        "min_samples": 1, "eps": 40, "nxw": 2,
    #                                        "lw": 1, "mpxw": 3, "confidence": 0.5,
    #                                        "max_n": 2,
    #                                        "anchors": str(AnchorDistribution.NONE),
    #                                        "explicit_model": "log/checkpoints/tus_checkpoint.pth"},
    #                       # level=Level.DEBUG
    #                       )
    #     evaluator = Evaluator(args, prepare_forward=True, load_best_model=False)
    #     # prevent debug tracemalloc
    #     images, grid_tensors, fileinfo, _ = unsqueeze(evaluator.dataset.__getitem__(0),
    #                                                   evaluator.dataset.__getitem__(1))
    #
    #     outputs, lines = evaluator(images, labels=grid_tensors, idx=evaluator.forward.start_epoch,
    #                                filenames=fileinfo, tag="eval_unittest", apply_nms=True, fit_line=False)
    #
    #     self.assertEqual(lines[0].shape[1], 3, lines[0].shape)
    #     self.assertEqual(lines[1].shape[1], 4, lines[1].shape)


if __name__ == '__main__':
    unittest.main()
