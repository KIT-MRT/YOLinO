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
import unittest
from copy import copy

import numpy as np
import torch
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.eval.distances import get_midpoints, point_to_line_distance
from yolino.eval.iccv_metrics import iccv_f1
from yolino.runner.evaluator import Evaluator
from yolino.utils.enums import Dataset, Variables, ACTIVATION, LOSS, Distance, AnchorDistribution
from yolino.utils.test_utils import test_setup, unsqueeze


class MetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = test_setup(self._testMethodName, str(Dataset.CULANE),
                               additional_vals={"batch_size": 2, "num_predictors": 8,
                                                "training_variables": [Variables.GEOMETRY, Variables.CLASS,
                                                                       Variables.CONF],
                                                "activations": [ACTIVATION.LINEAR, ACTIVATION.SOFTMAX,
                                                                ACTIVATION.SIGMOID],
                                                "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM, LOSS.MSE_SUM],
                                                "explicit_model": "log/checkpoints/test_metrics_on_dublicate.pth",
                                                "matching_gate": 1024, "anchors": str(AnchorDistribution.NONE),
                                                "association_metric": str(Distance.SQUARED)},
                               # level=Level.DEBUG
                               )

    def test_detection_metrics_with_iccv(self):
        args = copy(self.args)
        args.batch_size = 1
        args.num_predictors = 4
        args.training_variables = [Variables.GEOMETRY, Variables.CONF]
        args.activations = [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID]
        args.confidence = 0.5

        evaluator = Evaluator(args, prepare_forward=False, anchors=None)

        predictions = torch.tensor(
            [[15., 10, 25, 20, 0.8], [35, 30, 45, 40, 0.1], [0, 0, 10, 10, 0.1], [6, 0, 5, 10, 0.8]])
        preds_uv = torch.tile(predictions, dims=(args.batch_size, 1, 1))

        gt_uv = [torch.tensor([[10., 10, 20, 20, 1, 0, 0, 0, 0, 1], [30, 30, 40, 40, 1, 0, 0, 0, 0, 1]])]

        # evaluator.get_scores_uv(preds_uv=preds_uv, gt_uv=gt_uv, epoch=None, tag="unittest", filenames=["test.png"])
        precision, recall = iccv_f1(preds_uv=preds_uv, gt_uv=gt_uv[0].unsqueeze(0), img_size=[320, 640], threshold=0.5,
                                    conf_idx=4)
        evaluator.scores["precision"] = [[precision]]
        evaluator.scores["recall"] = [[recall]]

        self.assert_metrics(scores=evaluator.scores, prefix="", recall=0.5, precision=0.5833)

    def test_detection_metrics(self):
        args = copy(self.args)
        args.batch_size = 1
        args.num_predictors = 4
        args.training_variables = [Variables.GEOMETRY, Variables.CONF]
        args.activations = [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID]
        args.confidence = 0.5
        args.cell_size = [64, 64]
        args.grid_shape = [1, 1]
        args.img_size = [64, 64]
        args.matching_gate = 100 / 64.
        args.matching_gate_px = 100

        dataset, _ = DatasetFactory.get(dataset_enum=args.dataset, only_available=True, split=args.split, args=args,
                                        shuffle=False, augment=False)
        evaluator = Evaluator(args, prepare_forward=False, anchors=None)

        predictions = torch.tensor(
            [[15., 10, 25, 20, 0.8], [35, 30, 45, 40, 0.1], [0, 0, 10, 10, 0.1], [5, 0, 6, 10, 0.8]])
        preds_uv = torch.tile(predictions, dims=(args.batch_size, 1, 1))

        # cell match with first predictions
        gt_uv = [torch.tensor([[10., 10, 20, 20, 1, 0, 0, 0, 0, 1], [30, 30, 40, 40, 1, 0, 0, 0, 0, 1]])]

        evaluator.get_scores_uv(gt_uv=gt_uv, preds_uv=preds_uv, epoch=None, filenames=["test.png"], num_duplicates=0,
                                tag="unittest")
        evaluator.get_nms_results(filenames=["test.png"], preds_uv=preds_uv, gt_uv=gt_uv,
                                  images=dataset.dummy_image().unsqueeze(0), epoch=0,
                                  tag="unittest", num_duplicates=0)

        rmse = math.sqrt((math.pow(0.1, 2) + math.pow(0.2, 2) + math.pow(0.8, 2) + math.pow(0.9, 2)) / 4)
        perp = math.cos(math.pi / 4) * 5
        self.assert_metrics(scores=evaluator.scores, prefix="uv_metrics", confusion=[[[1.], [1.]], [[1.], [1.]]],
                            recall=0.5, precision=0.5, f1=0.5, accuracy=0.5, tp=1, fp=1, tn=1, fn=1,
                            conf_rmse=0.2, conf_mae=0.2, conf_all_rmse=rmse, conf_all_mae=(0.1 + 0.8 + 0.2 + 0.9) / 4.,
                            geom_x_rmse=math.sqrt((5 * 5 + 5 * 5) / 2), geom_x_mae=5, geom_y_rmse=0, geom_y_mae=0,
                            geom_perp_rmse=math.sqrt((math.pow(perp, 2))), geom_perp_mae=perp,
                            geom_midpoint_rmse=math.sqrt((math.pow(5, 2))), geom_midpoint_mae=5,
                            geom_ang_rmse=0, geom_ang_mae=0, geom_length_rmse=0, geom_length_mae=0,
                            # sample_ang_rmse=0, sample_x_rmse=5, sample_y_rmse=0,
                            # sample_ang_mae=0, sample_x_mae=5, sample_y_mae=0
                            )

        gt_lines_cell = torch.tile(gt_uv[0], (1, args.grid_shape[0] * args.grid_shape[1], 1, 1))
        gt_lines_cell[:, :, :, 0:4] /= args.cell_size[0]

        p_lines_cell = torch.tile(preds_uv, (1, args.grid_shape[0] * args.grid_shape[1], 1, 1))
        p_lines_cell[:, :, :, 0:4] /= args.cell_size[0]

        nan_lines = torch.ones((1, args.grid_shape[0] * args.grid_shape[1], 2, 10)) * torch.nan
        gt_lines_cell_w_nan = torch.concat([gt_lines_cell, nan_lines], axis=2)
        evaluator.get_scores_in_cell(grid_tensor=gt_lines_cell_w_nan, preds=p_lines_cell, epoch=None,
                                     filenames=["test.png"], num_duplicates=0, tag="unittest", do_matching=False)

        self.assert_metrics(scores=evaluator.scores, prefix="cell_metrics", confusion=[[[1.], [1.]], [[1.], [1.]]],
                            recall=0.5, precision=0.5, f1=0.5, accuracy=0.5, tp=1, fp=1, tn=1, fn=1,
                            conf_rmse=0.2, conf_mae=0.2, conf_all_rmse=rmse, conf_all_mae=(0.1 + 0.8 + 0.2 + 0.9) / 4.,
                            geom_x_rmse=math.sqrt((5 * 5 + 5 * 5) / 2), geom_x_mae=5, geom_y_rmse=0, geom_y_mae=0,
                            geom_perp_rmse=math.sqrt((math.pow(perp, 2))), geom_perp_mae=perp,
                            geom_midpoint_rmse=math.sqrt((math.pow(5, 2))), geom_midpoint_mae=5,
                            geom_ang_rmse=0, geom_ang_mae=0, geom_length_rmse=0, geom_length_mae=0,
                            # sample_ang_rmse=0, sample_x_rmse=5, sample_y_rmse=0,
                            # sample_ang_mae=0, sample_x_mae=5, sample_y_mae=0
                            )

    def test_class_metrics(self):
        args = copy(self.args)
        args.batch_size = 1
        args.num_predictors = 4
        args.matching_gate_px = 124

        dataset, _ = DatasetFactory.get(dataset_enum=args.dataset, only_available=True, split=args.split, args=args,
                                        shuffle=False, augment=False)
        evaluator = Evaluator(args, prepare_forward=False, anchors=None)

        predictions = torch.tensor(
            [[0., 0, 0, 0, 1, 0, 0, 0, 0, 1], [35, 30, 45, 40, 1, 0, 0, 0, 0, 1], [0, 0, 10, 10, 0, 0, 0, 1, 0, 0],
             [5, 0, 5, 10, 0, 0, 0, 1, 0, 1]])
        preds_uv = torch.tile(predictions, dims=(args.batch_size, 1, 1))

        gt_uv = [torch.tensor(
            [[15., 10, 25, 20, 0, 1, 0, 0, 0, 1], [35, 30, 45, 40, 1, 0, 0, 0, 0, 1], [0, 0, 10, 10, 0, 0, 1, 0, 0, 1],
             [5, 0, 5, 10, 0, 0, 0, 1, 0, 1]])]

        # GTxPred
        # Geom-TP: 1x1, 3x3
        # Geom-FP: P0
        # Geom-FN: P2
        # Geom-TN: /

        # Von allen TPs
        # Class-TP: C0:P1, C3:P3
        # Class-FP: /

        evaluator.get_scores_uv(gt_uv=gt_uv, preds_uv=preds_uv, epoch=None, filenames=["test.png"], num_duplicates=0,
                                tag="unittest")
        evaluator.get_nms_results(filenames=["test.png"], preds_uv=preds_uv, gt_uv=gt_uv,
                                  images=dataset.dummy_image().unsqueeze(0), epoch=0,
                                  tag="unittest", num_duplicates=0)

        num_classes = dataset.coords[Variables.CLASS]
        expected_confusion = np.zeros((num_classes, num_classes))
        expected_confusion[0, 0] = 1
        expected_confusion[3, 3] = 1

        self.assert_class_metrics(scores=evaluator.scores, confusion=expected_confusion,
                                  recall=[1, None, None, 1, None], precision=[1, None, None, 1, None],
                                  f1=[1, None, None, 1, None], tp=[1, 0, 0, 1, 0], fp=[0, 0, 0, 0, 0],
                                  fn=[0, 0, 0, 0, 0],
                                  tn=[1, 2, 2, 1, 2], accuracy=1)

    def test_metrics_on_dublicate(self):

        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 2, "num_predictors": 8,
                                           "training_variables": [Variables.GEOMETRY, Variables.CLASS,
                                                                  Variables.CONF],
                                           "activations": [ACTIVATION.LINEAR, ACTIVATION.SOFTMAX,
                                                           ACTIVATION.SIGMOID],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "explicit_model": "log/checkpoints/test_metrics_on_dublicate.pth",
                                           "matching_gate": 1024,
                                           "eps": 107,
                                           "nxw": 0, "lw": 0, "anchors": str(AnchorDistribution.NONE)},
                          # level=Level.DEBUG
                          )

        dataset, _ = DatasetFactory.get(dataset_enum=args.dataset, only_available=True, split=args.split, args=args,
                                        shuffle=False, augment=False)
        evaluator = Evaluator(args, load_best_model=False, anchors=None)  # we only pushed model.pth to git
        # prevent debug tracemalloc
        images, _, fileinfo, _, params = unsqueeze(dataset.__getitem__(0), dataset.__getitem__(1))

        predictions = torch.cat(
            [dataset.full_grid().unsqueeze(0), dataset.full_grid().unsqueeze(0)])

        preds_uv, gt_uv = evaluator.prepare_uv(
            preds=predictions[:, :, :, dataset.coords.get_position_of_training_vars()],
            grid_tensors=predictions, filenames=fileinfo, images=images)
        evaluator.get_scores_uv(gt_uv=gt_uv, preds_uv=preds_uv, epoch=0, filenames=["test.png", "test2.png"],
                                num_duplicates=0, tag="unittest")
        evaluator.get_nms_results(filenames=fileinfo, preds_uv=preds_uv,
                                  # reduced gt as the nms matching becomes quite infeasible when we have 2000
                                  # valid matches which is practically impossible in reality (a match in every cell)
                                  gt_uv=gt_uv,
                                  images=images, epoch=0,
                                  tag="unittest", num_duplicates=0)

        valid_gt_flags = ~torch.any(predictions.isnan(), dim=3)
        num_lines = torch.sum(valid_gt_flags)
        class_position = dataset.coords.get_position_of(Variables.CLASS)
        num_lines_per_class = torch.sum(predictions[valid_gt_flags][:, class_position], dim=0)

        expected_confusion = [[[0], [0]], [[0], [num_lines]]]

        num_cells = args.batch_size * np.prod(args.grid_shape)
        num_preds_wo_match = num_lines - num_cells
        expected_nms_confusion = [[[0], [0]], [[num_preds_wo_match], [num_cells]]]

        recall = 1. / args.num_predictors
        self.assert_metrics(scores=evaluator.scores, prefix="uv_metrics", confusion=expected_confusion, accuracy=1,
                            recall=1, precision=1,
                            f1=1, tp=num_lines, tn=0, fp=0, fn=0, conf_rmse=0, conf_mae=0,
                            conf_all_rmse=0, conf_all_mae=0,
                            geom_x_rmse=0, geom_x_mae=0, geom_y_rmse=0, geom_y_mae=0,
                            geom_perp_rmse=0, geom_perp_mae=0,
                            geom_midpoint_rmse=0, geom_midpoint_mae=0,
                            geom_ang_rmse=0, geom_ang_mae=0, geom_length_rmse=0, geom_length_mae=0)
        self.assert_metrics(scores=evaluator.scores, prefix="uv_metrics_nms",
                            confusion=expected_nms_confusion, accuracy=recall,
                            recall=recall, precision=1, f1=2 * recall / (recall + 1),
                            tp=num_cells, fn=num_preds_wo_match, fp=0, tn=0,
                            conf_rmse=0, conf_mae=0,
                            conf_all_rmse=math.sqrt(num_preds_wo_match / num_lines),
                            conf_all_mae=num_preds_wo_match / num_lines,
                            geom_x_rmse=0, geom_x_mae=0, geom_y_rmse=0, geom_y_mae=0,
                            geom_perp_rmse=0, geom_perp_mae=0,
                            geom_midpoint_rmse=0, geom_midpoint_mae=0,
                            geom_ang_rmse=0, geom_ang_mae=0, geom_length_rmse=0, geom_length_mae=0,
                            )

        self.assert_class_metrics(scores=evaluator.scores, accuracy=1, recall=[1, None, None, None, None],
                                  precision=[1, None, None, None, None], f1=[1, None, None, None, None],
                                  tp=num_lines_per_class, tn=torch.sum(num_lines_per_class) - num_lines_per_class,
                                  fp=[0, 0, 0, 0, 0], fn=[0, 0, 0, 0, 0])

    # @unittest.skipIf(get_system_specs()["user"] == "mrtbuild", "not to be run on CI")
    def test_metrics_matching_gate(self):
        self.args.matching_gate_px = 500

        dataset, _ = DatasetFactory.get(dataset_enum=self.args.dataset, only_available=True, split=self.args.split,
                                        args=self.args, shuffle=False, augment=False)
        evaluator = Evaluator(self.args, load_best_model=False, anchors=None)  # we only push model.pth to git
        images, grid_tensors, fileinfo, dupl, params = unsqueeze(dataset.__getitem__(0),
                                                                 dataset.__getitem__(0))

        predictions = torch.cat([dataset.full_grid().unsqueeze(0), dataset.full_grid().unsqueeze(0)])
        total_num_predicted_lines = self.args.batch_size * np.prod(self.args.grid_shape) * self.args.num_predictors
        preds_uv, gt_uv = evaluator.prepare_uv(preds=predictions, grid_tensors=grid_tensors,
                                               filenames=fileinfo, images=images)
        num_duplicates = int(sum(dupl["total_duplicates_in_image"]))
        evaluator.get_scores_uv(gt_uv=gt_uv, preds_uv=preds_uv, epoch=0, filenames=["test.png", "test2.png"],
                                num_duplicates=num_duplicates, tag="unittest")

        evaluator.get_nms_results(filenames=fileinfo, preds_uv=preds_uv, gt_uv=gt_uv, images=images, epoch=0,
                                  tag="unittest", num_duplicates=num_duplicates)

        self.assert_metrics(scores=evaluator.scores, prefix="uv_metrics",
                            confusion=[[[0], [total_num_predicted_lines], [0], [0]]])

    def test_metrics_on_homogenous(self):
        # self.args.matching_gate_px = 500 # no match
        self.args.matching_gate_px = 1000

        dataset, _ = DatasetFactory.get(dataset_enum=self.args.dataset, only_available=True, split=self.args.split,
                                        args=self.args, shuffle=False, augment=False)
        evaluator = Evaluator(self.args, load_best_model=False, anchors=None)
        # prevent debug tracemalloc
        images, grid_tensors, fileinfo, dupl, params = unsqueeze(dataset.__getitem__(0),
                                                                 dataset.__getitem__(0))

        predictions = torch.cat(
            [dataset.full_grid().unsqueeze(0), dataset.full_grid().unsqueeze(0)])
        total_num_predicted_lines = self.args.batch_size * np.prod(self.args.grid_shape) * self.args.num_predictors
        preds_uv, gt_uv = evaluator.prepare_uv(preds=predictions, grid_tensors=grid_tensors,
                                               filenames=fileinfo, images=images)
        num_duplicates = int(sum(dupl["total_duplicates_in_image"]))
        evaluator.get_scores_uv(gt_uv=gt_uv, preds_uv=preds_uv, epoch=0, filenames=["test.png", "test2.png"],
                                num_duplicates=num_duplicates, tag="unittest")

        # TODO assert values
        evaluator.get_nms_results(filenames=fileinfo, preds_uv=preds_uv, gt_uv=gt_uv, images=images, epoch=0,
                                  tag="unittest", num_duplicates=num_duplicates)

        # conf as cols, match as rows
        expected_confusion = [[[0], [total_num_predicted_lines - 12]], [[0], [12]]]

        mae = (total_num_predicted_lines - 12) / total_num_predicted_lines
        rmse = math.sqrt(mae)

        rmse_x = 0
        mae_x = 0
        rmse_mp = 0
        mae_mp = 0
        rmse_l = 0
        mae_l = 0
        rmse_perp = 0
        mae_perp = 0
        p_l = math.sqrt(32 ** 2 + 32 ** 2)
        matches = [951, 1135, 1143, 1327, 1335, 1519]
        for idx in matches:
            r = math.floor(idx / 8 / 27.)
            c = math.floor(idx / 8) % 27
            for batch in range(self.args.batch_size):
                gt = torch.round(grid_tensors[batch, r * 27 + c, 0, 0:4] * self.args.cell_size[0])
                p = torch.round(predictions[batch, r * 27 + c, 0, 0:4] * self.args.cell_size[0])

                # x
                x_diff = p[[0, 2]] - gt[[0, 2]]
                x_diff = torch.round(x_diff)

                # midpoints
                mp = (get_midpoints(gt) - get_midpoints(p))
                mp = np.linalg.norm(mp)

                # lengths
                gt_l = np.linalg.norm(gt[2:4] - gt[0:2])
                l = p_l - gt_l

                # perp
                start_perp = point_to_line_distance(gt_l, gt, p[0:2]).item()
                end_perp = point_to_line_distance(gt_l, gt, p[2:4]).item()

                # pow / sum
                rmse_x += torch.pow(x_diff, 2).sum()
                rmse_mp += math.pow(mp, 2)
                rmse_l += math.pow(l, 2)
                rmse_perp += math.pow(start_perp, 2) + math.pow(end_perp, 2)
                mae_x += torch.abs(x_diff).sum()
                mae_mp += abs(mp)
                mae_l += abs(l)
                mae_perp += abs(start_perp) + abs(end_perp)

        # mean / sqrt
        rmse_x /= len(matches) * self.args.batch_size * 2
        rmse_x = math.sqrt(rmse_x)
        mae_x /= len(matches) * self.args.batch_size * 2
        rmse_mp /= len(matches) * self.args.batch_size
        rmse_mp = math.sqrt(rmse_mp)
        mae_mp /= len(matches) * self.args.batch_size
        rmse_l /= len(matches) * self.args.batch_size
        rmse_l = math.sqrt(rmse_l)
        mae_l /= len(matches) * self.args.batch_size
        rmse_perp /= len(matches) * self.args.batch_size * 2
        rmse_perp = math.sqrt(rmse_perp)
        mae_perp /= len(matches) * self.args.batch_size * 2

        ten_ang_diff = math.degrees(math.atan(10 / 32)) + 45  # 4
        nine_ang_diff = math.degrees(math.atan(9 / 32)) + 45  # 2
        mae_ang = abs(4 * ten_ang_diff + 2 * nine_ang_diff) / 6
        rmse_ang = math.sqrt((4 * (ten_ang_diff ** 2) + 2 * (nine_ang_diff ** 2)) / 6)

        self.assert_metrics(scores=evaluator.scores, prefix="uv_metrics", confusion=expected_confusion,
                            accuracy=12 / total_num_predicted_lines, recall=1, precision=12 / total_num_predicted_lines,
                            f1=(2 * 12 / total_num_predicted_lines) / (1 + 12 / total_num_predicted_lines),
                            tp=12, tn=0, fp=total_num_predicted_lines - 12, fn=0, conf_rmse=0, conf_mae=0,
                            conf_all_rmse=rmse, conf_all_mae=mae,
                            geom_y_rmse=0, geom_y_mae=0,
                            geom_x_rmse=rmse_x, geom_x_mae=mae_x,
                            geom_perp_rmse=rmse_perp, geom_perp_mae=mae_perp,
                            geom_midpoint_rmse=rmse_mp, geom_midpoint_mae=mae_mp,
                            geom_ang_rmse=rmse_ang, geom_ang_mae=mae_ang,
                            geom_length_rmse=rmse_l, geom_length_mae=mae_l,
                            )

        self.assert_class_metrics(scores=evaluator.scores, accuracy=0, recall=[None, 0, None, None, None],
                                  precision=[0, None, None, None, None], f1=[0, 0, None, None, None],
                                  tp=[0, 0, 0, 0, 0], tn=[0, 0, 0, 0, 0], fp=[12, 0, 0, 0, 0], fn=[0, 12, 0, 0, 0], )

    def assert_metrics(self, scores, prefix="", **kwargs):
        for metric_key, expected_metric_value in kwargs.items():

            if prefix != "":
                metric_key = prefix + "/" + metric_key
            self.assertIn(metric_key, scores, f"{metric_key} not in scores: {scores.keys()}")
            if not np.isscalar(expected_metric_value):
                flattened_expectation = np.asarray(expected_metric_value).flatten()

                flattened_scores = np.asarray(scores[metric_key]).flatten()
                for i, expected_array_value in enumerate(flattened_expectation):
                    if expected_array_value is None:
                        self.assertTrue(np.isnan(flattened_scores[i]),
                                        msg="For %s idx=%d we have %s, but expected nan"
                                            % (metric_key, i, scores[metric_key]))
                    else:
                        self.assertAlmostEqual(flattened_expectation[i], flattened_scores[i], places=4,
                                               msg="For %s idx=%d we have %s, but expected %s"
                                                   % (metric_key, i, scores[metric_key], expected_array_value))
            else:
                if expected_metric_value is None:
                    self.assertTrue(np.isnan(scores[metric_key][0][0]),
                                    msg="For %s we have %s, but expected nan" % (metric_key, scores[metric_key]))
                else:
                    self.assertAlmostEqual(expected_metric_value, scores[metric_key][0][0], places=4,
                                           msg="For %s we have %s, but expected %s" % (metric_key, scores[metric_key],
                                                                                       expected_metric_value))

    def assert_class_metrics(self, scores, **kwargs):
        new_kwargs = {}
        for k in kwargs:
            new_kwargs["class/" + k] = kwargs[k]

        return self.assert_metrics(scores, prefix="uv_metrics", **new_kwargs)


if __name__ == '__main__':
    unittest.main()
