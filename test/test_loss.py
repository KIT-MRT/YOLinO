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
import os.path
import socket
import timeit
import unittest

import numpy as np
import torch
from tqdm import tqdm
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.eval.matcher_cell import CellMatcher
from yolino.eval.matcher_uv import UVMatcher
from yolino.grid.grid_factory import GridFactory
from yolino.model.anchors import Anchor
from yolino.model.loss import get_loss, NormLoss, MeanSquaredErrorLoss, CrossEntropyCellLoss
from yolino.model.model_factory import load_checkpoint
from yolino.runner.forward_runner import ForwardRunner
from yolino.runner.trainer import TrainHandler, TRAIN_TAG
from yolino.utils.enums import Dataset, LOSS, CoordinateSystem, ImageIdx, Scheduler, Variables, ACTIVATION, Distance, \
    AnchorDistribution
from yolino.utils.logger import Log
from yolino.utils.test_utils import test_setup, unsqueeze


class TestLoss(unittest.TestCase):

    def test_loss_composition(self):

        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 1, "img_height": 128, "num_predictors": 1,
                                           "max_n": 10,
                                           "training_variables": [Variables.GEOMETRY, Variables.CLASS, Variables.CONF],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SOFTMAX, ACTIVATION.SIGMOID],
                                           "loss": [LOSS.NORM_SUM, LOSS.CROSS_ENTROPY_SUM, LOSS.MSE_SUM]})
        loss_fct = get_loss(losses=args.loss, args=args,
                            coords=DatasetFactory.get_coords(split=args.split, args=args),
                            weights=[1, 1, 1], anchors=Anchor.get(args, args.linerep),
                            conf_weights=args.conf_match_weight)
        self.assertEqual(type(loss_fct.losses[0]), NormLoss)
        self.assertEqual(type(loss_fct.losses[1]), CrossEntropyCellLoss)
        self.assertEqual(type(loss_fct.losses[2]), MeanSquaredErrorLoss)

    def test_loss_empty_class(self):
        for loss in LOSS:
            args = test_setup(self._testMethodName + "_" + str(loss), str(Dataset.CULANE),
                              additional_vals={"batch_size": 1, "img_height": 128, "num_predictors": 1,
                                               "max_n": 10, "training_variables": str(Variables.CLASS),
                                               "activations": [ACTIVATION.SOFTMAX], "loss": [loss]})

            if loss == LOSS.CROSS_ENTROPY_SUM or loss == LOSS.CROSS_ENTROPY_MEAN:
                # TODO: calc true value
                continue

            dataset, loader = DatasetFactory.get(Dataset.CULANE, only_available=False, split="train", args=args,
                                                 shuffle=True, augment=False)

            loss_weights = TrainHandler.__init_loss_weights__(num_train_vars=len(dataset.coords.train_vars()),
                                                              cuda=args.cuda,
                                                              loss_weighting=args.loss_weight_strategy,
                                                              is_exponential=[True, True])

            conf_loss_weights = TrainHandler.__init_conf_loss_weights__(cuda=args.cuda,
                                                                        loss_weighting=args.loss_weight_strategy,
                                                                        is_exponential=[True, True])

            if loss == LOSS.NORM_SUM or loss == LOSS.NORM_MEAN:
                with self.assertRaises(NotImplementedError):
                    loss_fct = get_loss(losses=[loss], args=args, coords=dataset.coords, weights=loss_weights,
                                        anchors=dataset.anchors, conf_weights=conf_loss_weights)
                continue
            else:
                loss_fct = get_loss(losses=[loss], args=args, coords=dataset.coords, weights=loss_weights,
                                    anchors=dataset.anchors, conf_weights=conf_loss_weights)

            # Loss works with shapes preds=torch.Size([1, 243, 8, 4]) and gt=torch.Size([1, 243, 8, 10])
            dummy_labels = torch.unsqueeze(dataset.full_grid(train_vars=False), dim=0)

            dummy_preds = torch.unsqueeze(dataset.empty_grid(), dim=0)
            dummy_preds = dummy_preds[:, :, :, dataset.coords.get_position_of(Variables.CLASS)]

            print(dummy_preds.shape)
            print(dummy_labels.shape)
            if loss == LOSS.BINARY_CROSS_ENTROPY_SUM or loss == LOSS.BINARY_CROSS_ENTROPY_MEAN:
                with self.assertRaises(NotImplementedError):
                    loss_fct(preds=dummy_preds, grid_tensor=dummy_labels, filenames=["test.png"], epoch=0)
                continue
            loss_vals, sum_loss, _ = loss_fct(preds=dummy_preds, grid_tensor=dummy_labels, filenames=["test.png"],
                                              epoch=0)
            self.assertTrue(loss_vals[0] == 0,
                            msg="We applied %s to identical tensors, but received loss=%s" % (loss, loss_vals))

    def test_loss_similar_class(self):
        for loss in LOSS:
            args = test_setup(self._testMethodName + "_" + str(loss), str(Dataset.CULANE),
                              additional_vals={"batch_size": 2, "img_height": 128, "num_predictors": 1,
                                               "max_n": 2, "training_variables": str(Variables.CLASS),
                                               "activations": [ACTIVATION.SOFTMAX], "loss": [loss]})

            dataset, loader = DatasetFactory.get(Dataset.CULANE, only_available=False, split="train", args=args,
                                                 shuffle=True, augment=False)
            images, dummy_labels, filenames, _, _ = next(iter(loader))
            loss_weights = TrainHandler.__init_loss_weights__(num_train_vars=len(dataset.coords.train_vars()),
                                                              cuda=args.cuda, loss_weighting=args.loss_weight_strategy,
                                                              is_exponential=[True, True])

            conf_loss_weights = TrainHandler.__init_conf_loss_weights__(cuda=args.cuda,
                                                                        loss_weighting=args.loss_weight_strategy,
                                                                        is_exponential=[True, True])

            if loss == LOSS.NORM_SUM or loss == LOSS.NORM_MEAN:
                with self.assertRaises(NotImplementedError):
                    loss_fct = get_loss(losses=[loss], args=args, coords=dataset.coords, weights=loss_weights,
                                        anchors=dataset.anchors, conf_weights=conf_loss_weights)
                continue
            else:
                loss_fct = get_loss(losses=[loss], args=args, coords=dataset.coords, weights=loss_weights,
                                    anchors=dataset.anchors, conf_weights=conf_loss_weights)

            dummy_preds = dummy_labels[:, :, :, dataset.coords.get_position_of(Variables.CLASS, one_hot=True)]

            print(dummy_preds.shape)
            print(dummy_labels.shape)

            # Loss works with shapes preds=torch.Size([1, 243, 8, 4]) and gt=torch.Size([1, 243, 8, 10])
            if loss == LOSS.BINARY_CROSS_ENTROPY_MEAN or loss == LOSS.BINARY_CROSS_ENTROPY_SUM:
                with self.assertRaises(NotImplementedError):
                    loss_fct(preds=dummy_preds, grid_tensor=dummy_labels, filenames=["test.png"], epoch=0)
                    continue
            else:
                loss_vals, sum_loss, _ = loss_fct(preds=dummy_preds, grid_tensor=dummy_labels, filenames=["test.png"],
                                                  epoch=0)

            if loss == LOSS.CROSS_ENTROPY_SUM or loss == LOSS.CROSS_ENTROPY_MEAN or loss == LOSS.BINARY_CROSS_ENTROPY_MEAN or loss == LOSS.BINARY_CROSS_ENTROPY_SUM:
                # TODO: calc true value
                continue

            self.assertTrue(loss_vals[0] == 0,
                            msg="We applied %s to identical tensors, but received loss=%s" % (loss, loss_vals))

    def test_multi_anchor_loss(self):

        start = timeit.default_timer()
        self.run_multi_loss(use_anchors=True)
        end = timeit.default_timer()
        Log.info("Anchor loss takes %ss" % (end - start))

    def test_multi_loss(self):
        start = timeit.default_timer()
        self.run_multi_loss(use_anchors=False)
        end = timeit.default_timer()
        Log.info("Matching loss takes %ss" % (end - start))

    def run_multi_loss(self, use_anchors):
        batch_size = 2
        additional_args = {"batch_size": batch_size, "img_height": 128, "num_predictors": 8 if use_anchors else 12,
                           "max_n": 2, "training_variables": [str(Variables.GEOMETRY),
                                                              str(Variables.CLASS),
                                                              str(Variables.CONF)],
                           "loss": [LOSS.MSE_SUM, LOSS.CROSS_ENTROPY_SUM, LOSS.MSE_SUM],
                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SOFTMAX, ACTIVATION.SIGMOID],
                           "association_metric": str(Distance.SQUARED),
                           "offset": False}
        if use_anchors:
            additional_args["anchors"] = str(AnchorDistribution.EQUAL)
        else:
            additional_args["anchors"] = str(AnchorDistribution.NONE)

        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals=additional_args)

        dataset, loader = DatasetFactory.get(Dataset.CULANE, only_available=False, split="train", args=args,
                                             shuffle=True, augment=False)
        images, grid_tensor, filenames, _, _ = next(iter(loader))
        num_lines = torch.sum(torch.all(~grid_tensor[:, :, :, 0:4].isnan(), dim=3))
        total_num_preds = args.batch_size * args.grid_shape[0] * args.grid_shape[1] * args.num_predictors
        num_unmatched = total_num_preds - num_lines
        preds_template = torch.tile(torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
                                    (batch_size, args.grid_shape[0] * args.grid_shape[1], args.num_predictors, 1))

        losses = [LOSS.MSE_SUM, LOSS.MSE_SUM, LOSS.MSE_SUM]
        loss_weights = TrainHandler.__init_loss_weights__(num_train_vars=len(dataset.coords.train_vars()),
                                                          cuda=args.cuda,
                                                          loss_weighting=args.loss_weight_strategy,
                                                          is_exponential=[True, True, True])
        conf_loss_weights = TrainHandler.__init_conf_loss_weights__(cuda=args.cuda,
                                                                    loss_weighting=args.loss_weight_strategy,
                                                                    is_exponential=[True, True])
        loss_fct = get_loss(losses=losses, args=args, coords=dataset.coords, weights=loss_weights,
                            anchors=dataset.anchors, conf_weights=conf_loss_weights)

        idx = 0
        for item in dataset.coords.items():
            variable, count = item
            if count == 0:
                continue
            num_batch, num_cells, num_predictors, num_values = grid_tensor.shape

            set_value = torch.zeros((num_batch, num_cells, num_predictors, dataset.coords[variable]))
            conf_val = 0.6
            if variable == Variables.GEOMETRY:
                set_value[:, :, :, 2:4] = 0.5
            elif variable == Variables.CLASS:
                set_value[:, :, :, 1] = 1  # class id = 1 => 0 1 0 0 0 as one hot
            elif variable == Variables.CONF:
                set_value[:, :, :, 0] = conf_val
            pos = dataset.coords.get_position_of(variable, one_hot=True)

            from copy import deepcopy
            preds = deepcopy(preds_template)
            preds[~torch.isnan(grid_tensor)] = grid_tensor[~torch.isnan(grid_tensor)]
            preds[:, :, :, pos] = set_value
            # preds = preds.permute((0, 3, 1, 2))

            sum_losses, sum_loss, mean_losses = loss_fct(preds=preds, grid_tensor=grid_tensor, filenames=filenames,
                                                         epoch=0)
            Log.debug("Losses: %s" % str(sum_losses))

            for i in range(len(sum_losses)):
                if i == idx:
                    self.assertGreater(sum_losses[i].item(), 0,
                                       msg="We applied %s to not identical tensors, but received loss=%.3f"
                                           % (losses[i], sum_losses[i]))

                    if variable == Variables.CONF:
                        self.assertAlmostEqual(mean_losses[i].item(), 0.5 * ((1 - conf_val) ** 2 + conf_val ** 2),
                                               msg="We applied %s to not identical tensors, but received loss=%.3f"
                                                   % (losses[i], sum_losses[i]), places=3)
                else:
                    self.assertAlmostEqual(sum_losses[i].item(), 0, places=3,
                                           msg="We applied %s to identical tensors, but received loss=%.3f"
                                               % (losses[i], sum_losses[i]))
                    self.assertAlmostEqual(mean_losses[i].item(), 0, places=3,
                                           msg="We applied %s to identical tensors, but received loss=%.3f"
                                               % (losses[i], mean_losses[i]))
                print(f"{variable}: {sum_losses[i].item()}")
            idx += 1

    def test_loss_association_with_duplicates(self):

        for loss in LOSS:
            if loss == LOSS.BINARY_CROSS_ENTROPY_MEAN or loss == LOSS.BINARY_CROSS_ENTROPY_SUM:
                continue
            args = test_setup(self._testMethodName + "_" + str(loss), str(Dataset.CULANE),
                              additional_vals={"batch_size": 2, "num_predictors": 8, "max_n": 1,
                                               "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                               "loss": [loss, LOSS.MSE_SUM],
                                               "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID]
                                               })

            dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split="train", args=args,
                                                 shuffle=True, augment=True)

            model, scheduler_checkpoint, model_epoch = load_checkpoint(args, dataset.coords)
            forward = ForwardRunner(args, model, model_epoch)

            images, grid_tensors, filenames, _, _ = next(iter(loader))
            outputs = forward(images, is_train=True, epoch=model_epoch)

            # use 10x the label for duplicate check
            grid_tensors = torch.tile(grid_tensors[:, :, [0], :], [1, 1, args.num_predictors, 1])
            loss_weights = TrainHandler.__init_loss_weights__(num_train_vars=len(dataset.coords.train_vars()),
                                                              cuda=args.cuda,
                                                              loss_weighting=args.loss_weight_strategy,
                                                              is_exponential=[True, True])

            conf_loss_weights = TrainHandler.__init_conf_loss_weights__(cuda=args.cuda,
                                                                        loss_weighting=args.loss_weight_strategy,
                                                                        is_exponential=[True, True])
            loss_fct = get_loss(losses=args.loss, args=args, coords=dataset.coords, weights=loss_weights,
                                anchors=dataset.anchors, conf_weights=conf_loss_weights)
            loss_fct(preds=outputs, grid_tensor=grid_tensors, filenames=filenames, epoch=0)

    def test_loss_association(self):
        for loss in LOSS:
            if loss == LOSS.BINARY_CROSS_ENTROPY_MEAN or loss == LOSS.BINARY_CROSS_ENTROPY_SUM:
                continue
            args = test_setup(self._testMethodName + "_" + str(loss), str(Dataset.CULANE),
                              additional_vals={"batch_size": 2, "num_predictors": 8, "max_n": 1,
                                               "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                               "loss": [loss, LOSS.MSE_SUM],
                                               "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID]})

            dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split="train", args=args,
                                                 shuffle=True, augment=True)

            model, scheduler_checkpoint, model_epoch = load_checkpoint(args, dataset.coords)
            forward = ForwardRunner(args, model, model_epoch)

            images, grid_tensors, filenames, _, _ = next(iter(loader))
            outputs = forward(images, is_train=True, epoch=model_epoch)

            loss_weights = TrainHandler.__init_loss_weights__(
                num_train_vars=len(dataset.coords.train_vars()),
                cuda=args.cuda,
                loss_weighting=args.loss_weight_strategy, is_exponential=[True, True])
            conf_loss_weights = TrainHandler.__init_conf_loss_weights__(cuda=args.cuda,
                                                                        loss_weighting=args.loss_weight_strategy,
                                                                        is_exponential=[True, True])
            loss_fct = get_loss(losses=args.loss, args=args, coords=dataset.coords, weights=loss_weights,
                                anchors=dataset.anchors, conf_weights=conf_loss_weights)
            loss_fct(outputs, grid_tensors, filenames=filenames, epoch=0)

    def test_matches_with_empty(self):

        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 1, "num_predictors": 2, "max_n": 1,
                                           "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID],
                                           "plot": socket.gethostname() == "mrt057",
                                           },
                          )
        coords = DatasetFactory.get_coords("train", args)
        matcher = CellMatcher(coords, args)

        grid_shape = (2, 5)
        grid_tensor = torch.ones(
            size=(int(args.batch_size), int(np.prod(grid_shape)), int(args.num_predictors), coords.get_length()),
            dtype=torch.float)
        grid_tensor[:, :, :, coords.get_position_of(Variables.GEOMETRY)] *= torch.nan

        # second is closer
        p_lines = torch.tensor([[0, 0, 0.8, 0.9, 1], [0.1, 0.1, 1, 1, 1]], dtype=torch.float32)
        preds = torch.tile(p_lines, dims=[1, np.prod(grid_shape), 1, 1])

        matched_predictions, matched_gt = matcher.match(preds, grid_tensor, filenames=["test.png"],
                                                        confidence_threshold=args.confidence)
        msg = "P: %s\nG: %s" % (matched_gt[0:3], matched_predictions[0:3])
        self.assertTrue(np.array_equal(matched_predictions.shape, matched_gt.shape), msg)
        self.assertTrue(np.array_equal([args.batch_size * np.prod(grid_shape), args.num_predictors], matched_gt.shape),
                        msg)
        self.assertLessEqual(torch.max(matched_predictions), args.num_predictors, msg)
        self.assertLessEqual(torch.max(matched_gt), args.num_predictors, msg)

        self.assertTrue(torch.all(matched_predictions == -100), msg)
        self.assertTrue(torch.all(matched_gt == -100), msg)

        sorted_preds, sorted_grid_tensor = matcher.sort_cells_by_geometric_match(preds, grid_tensor,
                                                                                 filenames=["test.png"], epoch=0)
        msg = "P: %s\nG: %s" % (sorted_preds[0:3], sorted_grid_tensor[0:3])
        self.assertTrue(torch.equal(preds.view(-1, coords.num_vars_to_train()), sorted_preds), msg)

    def test_random_match_with_random(self):

        args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                          additional_vals={"batch_size": 1, "num_predictors": 2, "max_n": 1,
                                           "img_height": 160,
                                           "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID],
                                           "plot": socket.gethostname() == "mrt057",
                                           },
                          )

        dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split="train", args=args,
                                             shuffle=False, augment=False)
        matcher = CellMatcher(dataset.coords, args)
        images, grid_tensor, fileinfo, _, params = unsqueeze(dataset.__getitem__(0))

        path = "/tmp/preds.torch"
        if os.path.exists(path):
            preds = torch.load(path)
        else:
            preds = torch.rand_like(grid_tensor)
        torch.save(preds, path)

        path = "/tmp/gt.torch"
        if os.path.exists(path):
            gt = torch.load(path)
        else:
            gt = torch.rand_like(grid_tensor)
            gt[:, :, 1, 0:-1] = torch.nan  # only allow one gt predictor to be set
            torch.save(gt, path)

        matched_predictions, matched_gt = matcher.match(preds, gt, filenames=["test.png"],
                                                        confidence_threshold=args.confidence)

        if args.plot:
            resorted_grid_tensor = matcher.__resort_by_match_ids__(gt, matched_predictions)
            matcher._debug_full_match_plot_(epoch=None, preds=preds, grid_tensor=resorted_grid_tensor,
                                            coordinates=CoordinateSystem.CELL_SPLIT, filenames=fileinfo,
                                            tag="test_matches")

        for i in range(0, len(matched_predictions)):
            self.assertTrue(sum(matched_predictions[i] == -100) == sum(gt[0, i, :, 0].flatten().isnan()))

    def test_matches(self):

        args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                          additional_vals={"batch_size": 1, "num_predictors": 2, "max_n": 1,
                                           "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID],
                                           "plot": socket.gethostname() == "mrt057",
                                           },
                          )
        coords = DatasetFactory.get_coords("train", args)
        matcher = CellMatcher(coords, args)

        grid_shape = (2, 5)
        gt_line = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32)
        grid_tensor = torch.tile(gt_line, dims=[1, np.prod(grid_shape), args.num_predictors, 1])

        # second is closer
        p_lines = torch.tensor([[0, 0, 0.8, 0.9, 1], [0.1, 0.1, 1, 1, 1]], dtype=torch.float32)
        preds = torch.tile(p_lines, dims=[1, np.prod(grid_shape), 1, 1])

        matched_predictions, matched_gt = matcher.match(preds, grid_tensor, filenames=["test.png"],
                                                        confidence_threshold=args.confidence)
        msg = "P: %s\nG: %s" % (matched_gt[0:3], matched_predictions[0:3])
        self.assertTrue(np.array_equal(matched_predictions.shape, matched_gt.shape), msg)
        self.assertTrue(np.array_equal([args.batch_size * np.prod(grid_shape), args.num_predictors], matched_gt.shape),
                        msg)
        self.assertLessEqual(torch.max(matched_predictions), args.num_predictors, msg)
        self.assertLessEqual(torch.max(matched_gt), args.num_predictors, "msg")

        if args.plot:
            resorted_grid_tensor = matcher.__resort_by_match_ids__(grid_tensor, matched_predictions)
            matcher._debug_full_match_plot_(epoch=None, preds=preds, grid_tensor=resorted_grid_tensor,
                                            coordinates=CoordinateSystem.CELL_SPLIT, filenames="unittest.png",
                                            tag="test_matches")

        # first is closer
        p_lines2 = torch.tensor([[0, 0, 1, 0.9, 1], [0.1, 0.1, 1, 1, 1]], dtype=torch.float32)
        preds2 = torch.tile(p_lines2, dims=[1, np.prod(grid_shape), 1, 1])

        matched_predictions, matched_gt = matcher.match(preds, preds2, filenames=["test.png"],
                                                        confidence_threshold=args.confidence)
        self.assertTrue(torch.all(matched_predictions[:, 0] == 0), matched_predictions[0:3])
        self.assertTrue(torch.all(matched_predictions[:, 1] == 1), matched_predictions[0:3])
        self.assertTrue(torch.all(matched_gt[:, 0] == 0), matched_gt[0:3])
        self.assertTrue(torch.all(matched_gt[:, 1] == 1), matched_gt[0:3])

        if args.plot:
            resorted_grid_tensor = matcher.__resort_by_match_ids__(preds2, matched_predictions)
            matcher._debug_full_match_plot_(epoch=None, preds=preds, grid_tensor=resorted_grid_tensor,
                                            coordinates=CoordinateSystem.CELL_SPLIT, filenames="unittest.png",
                                            tag="test_matches_closer")

    def test_hard_conf_matches(self):

        args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                          additional_vals={"batch_size": 1, "num_predictors": 2, "max_n": 1,
                                           "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID],
                                           "plot": socket.gethostname() == "mrt057",
                                           "match_by_conf_first": True},
                          # level=Level.INFO,
                          show_params=True
                          )
        coords = DatasetFactory.get_coords("train", args)
        matcher = CellMatcher(coords, args)

        grid_shape = (2, 5)
        p_lines2 = torch.tensor([[0, 0, 1, 1, 1], [0.9, 1, 0, 0.1, 1]], dtype=torch.float32)
        grid_tensor = torch.tile(p_lines2, dims=[1, np.prod(grid_shape), 1, 1])

        p_lines = torch.tensor([[0, 0, 1, 1, 0.1], [0, 0.1, 0.9, 1, 0.9]], dtype=torch.float32)
        preds = torch.tile(p_lines, dims=[1, np.prod(grid_shape), 1, 1])

        matched_predictions, matched_gt = matcher.match(preds, grid_tensor, filenames=["test.png"],
                                                        confidence_threshold=args.confidence)

        if args.plot:
            resorted_grid_tensor = matcher.__resort_by_match_ids__(grid_tensor, matched_predictions)
            matcher._debug_full_match_plot_(epoch=None, preds=preds, grid_tensor=resorted_grid_tensor,
                                            coordinates=CoordinateSystem.CELL_SPLIT, filenames="unittest.png",
                                            tag="test_matches_hard")

        self.assertTrue(torch.all(matched_predictions[:, 0] == 1), matched_predictions[0:3])
        self.assertTrue(torch.all(matched_predictions[:, 1] == 0), matched_predictions[0:3])
        self.assertTrue(torch.all(matched_gt[:, 0] == 1), matched_gt[0:3])
        self.assertTrue(torch.all(matched_gt[:, 1] == 0), matched_gt[0:3])

    def test_matches_with_nan(self):

        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 1, "num_predictors": 3, "max_n": 1,
                                           "training_variables": [Variables.GEOMETRY, Variables.CLASS, Variables.CONF],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SOFTMAX, ACTIVATION.SIGMOID]},
                          )
        coords = DatasetFactory.get_coords("train", args)
        matcher = CellMatcher(coords, args)

        grid_shape = (2, 5)
        # with nan GT
        gt_line = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                                [torch.nan] * 10],
                               dtype=torch.float32)
        grid_tensor = torch.tile(gt_line, dims=[1, np.prod(grid_shape), 1, 1])

        p_lines = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                                [0, 0, 0.5, 0.5, 1, 0, 0, 0, 0, 1]],
                               dtype=torch.float32)
        preds = torch.tile(p_lines, dims=[1, np.prod(grid_shape), 1, 1])

        matched_predictions, matched_gt = matcher.match(preds, grid_tensor, filenames=["test.png"],
                                                        confidence_threshold=args.confidence)

        self.assertTrue(torch.all(matched_predictions[:, 2] == 0), matched_predictions[0:3])
        self.assertTrue(torch.all(matched_predictions[:, 0] == 1) or torch.all(matched_predictions[:, 0] == -100),
                        matched_predictions[0:3])
        self.assertTrue(torch.all(matched_predictions[:, 1] == -100) or torch.all(matched_predictions[:, 1] == 1),
                        matched_predictions[0:3])
        self.assertTrue(torch.all(matched_gt[:, 0] == 2), matched_gt[0:3])
        self.assertTrue(torch.all(matched_gt[:, 1] == 0) or torch.all(matched_gt[:, 1] == 1), matched_gt[0:3])
        self.assertTrue(torch.all(matched_gt[:, 2] == -100), matched_gt[0:3])

    def test_uv_matches_with_nan(self):
        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 1, "num_predictors": 3, "max_n": 1,
                                           "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID],
                                           "anchors": str(AnchorDistribution.NONE),
                                           "association_metric": str(Distance.SQUARED),
                                           "plot": socket.gethostname() == "mrt057"},
                          )
        coords = DatasetFactory.get_coords("train", args)

        # with nan GT
        # match is GT0 with Pred2
        # match is GT1 with Pred1
        gt_line = torch.tensor([[0, 0, 30, 30, 1, 0, 1], [10, 10, 50, 50, 1, 0, 1],
                                [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan]],
                               dtype=torch.float32)
        grid_tensor = torch.tile(gt_line, dims=[1, 1, 1])

        p_lines = torch.tensor([[205, 200, 105, 100, 1, 0, 1], [25, 20, 55, 50, 1, 0, 1], [5, 0, 35, 30, 1, 0, 1]],
                               dtype=torch.float32)
        preds = torch.tile(p_lines, dims=[1, 1, 1])

        # set threshold to less than distance between p[2] and gt[0], so they are not matched
        # distance between p[1] and gt[1] should be less
        # from gt[0] to all predictors squared distance should be [92550.0, 2050.0, 50.0]
        # from gt[1] to all it's [79650.0, 350.0, 750.0]
        threshold = 15 * 15 + 100 + 25 + 0
        matcher = UVMatcher(coords, args, distance_threshold=threshold - 10)
        matched_predictions, _ = matcher.match(preds, grid_tensor, filenames=["test.png"])
        self.assertTrue(torch.all(matched_predictions[:, 2] == 0), matched_predictions[0:3])
        self.assertTrue(torch.all(matched_predictions[:, 0:2] == -100), matched_predictions[0:3])

        # self.assertTrue(torch.all(matched_gt[:, 0] == 2), matched_gt[0:3])
        # self.assertTrue(torch.all(matched_gt[:, 1:] == -100), matched_gt[0:3])

        # include second match with higher threshold
        matcher = UVMatcher(coords, args, distance_threshold=threshold + 10)
        matched_predictions, _ = matcher.match(preds, grid_tensor, filenames=["test.png"])
        self.assertTrue(torch.all(matched_predictions[:, 1] == 1), matched_predictions[0:3])
        self.assertTrue(torch.all(matched_predictions[:, 2] == 0), matched_predictions[0:3])
        self.assertTrue(torch.all(matched_predictions[:, 0] == -100), matched_predictions[0:3])

    def test_loss_single_cell(self):
        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 1, "img_height": 128, "num_predictors": 5, "max_n": 1,
                                           "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID]})

        trainer = TrainHandler(args)
        grid_tensor = torch.unsqueeze(trainer.dataset.empty_grid(), dim=0)

        gt_line = torch.tensor([0.4, 0.5, 0, 1, 1, 0, 0, 0, 0, 1], dtype=torch.float32)
        grid_tensor[0, 20, 0] = gt_line

        preds = torch.zeros((args.batch_size, args.grid_shape[0] * args.grid_shape[1],
                             args.num_predictors, trainer.dataset.coords.num_vars_to_train()))

        preds[0, 20, 0] = torch.tensor([0.4, 0.5, 0, 1, 1])
        loss, sum_loss, _ = trainer.loss_fct(preds, grid_tensor, filenames=["test.png"], epoch=0)
        self.assertEqual(torch.sum(torch.tensor(loss)), 0, loss)

    def test_prepate_data(self):

        args = test_setup(self._testMethodName, str(Dataset.CULANE),
                          additional_vals={"batch_size": 1, "img_height": 128, "num_predictors": 5, "max_n": 1,
                                           "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                           "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM],
                                           "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID],
                                           "plot": socket.gethostname() == "mrt057", })

        coords = DatasetFactory.get_coords("train", args)  # split does not matter
        matcher = CellMatcher(coords, args)

        gt_line = torch.tensor([[0.4, 0.5, 0, 1, 1, 0, 0, 0, 0, 1], [0, 0.4, 0.5, 1, 1, 0, 0, 0, 0, 1]],
                               dtype=torch.float32)
        p_lines = torch.tensor(
            [[1, 1, 0, 0, 1], [0.5, 0, 0, 0.5, 1], [0.1, 0, 0.7, 0.6, 1], [0, 0.1, 0.9, 1, 1], [0, 0, 1, 1, 1]],
            dtype=torch.float32)
        preds = torch.tile(p_lines, dims=[1, np.prod(args.grid_shape), 1, 1])

        grid_tensor = torch.tile(gt_line, dims=[1, np.prod(args.grid_shape), 1, 1])
        grid, _ = GridFactory.get(data=grid_tensor, variables=[], coordinate=CoordinateSystem.CELL_SPLIT,
                                  args=args,
                                  input_coords=coords)
        grid_tensor = torch.unsqueeze(grid.tensor(coords=coords, convert_to_lrep=args.linerep), dim=0)

        preds, grid_tensor = matcher.sort_cells_by_geometric_match(preds, grid_tensor, filenames=["test.png"], epoch=0)

        self.assertTrue(torch.all(torch.isnan(grid_tensor[range(0, len(grid_tensor), 5), 0:4])), grid_tensor[0, 0:4])
        self.assertTrue(torch.all(grid_tensor[range(1, len(grid_tensor), 5), 0:4] == gt_line[0][0:4]),
                        grid_tensor[0, 0:4])
        self.assertTrue(torch.all(torch.isnan(grid_tensor[range(2, len(grid_tensor), 5), 0:4])), grid_tensor[0, 0:4])
        self.assertTrue(torch.all(grid_tensor[range(3, len(grid_tensor), 5), 0:4] == gt_line[1][0:4]),
                        grid_tensor[0, 0:4])
        self.assertTrue(torch.all(torch.isnan(grid_tensor[range(4, len(grid_tensor), 5), 0:4])), grid_tensor[0, 0:4])

    def test_trainer_class_loss(self):
        for loss in LOSS:
            if loss == LOSS.BINARY_CROSS_ENTROPY_MEAN or loss == LOSS.BINARY_CROSS_ENTROPY_SUM \
                    or loss == LOSS.NORM_SUM or loss == LOSS.NORM_MEAN:
                continue

            args = test_setup(self._testMethodName + "_" + str(loss), str(Dataset.CULANE),
                              additional_vals={"batch_size": 1, "img_height": 128, "num_predictors": 1,
                                               "max_n": 1, "learning_rate": 0.0001, "retrain": True, "loss": str(loss),
                                               "nondeterministic": False, "scheduler": str(Scheduler.NONE),
                                               "training_variables": str(Variables.CLASS),
                                               "activations": [ACTIVATION.SOFTMAX]})
            trainer = TrainHandler(args)

            images, grid_tensors, filenames, _, params = next(iter(trainer.loader))
            trainer.dataset.params_per_file.update({f: params for i, f in enumerate(filenames)})

            i = 0
            losses = []
            for e in range(5):
                images, preds = trainer(filenames=filenames, images=images, grid_tensor=grid_tensors, is_train=True,
                                        first_run=False, epoch=0, image_idx_in_batch=i)
                Log.info("Loss %s = %s" % (loss, trainer.losses[TRAIN_TAG]._backprops_))

                grid, _ = GridFactory.get(data=torch.unsqueeze(preds.detach()[i], dim=0), variables=[],
                                          coordinate=CoordinateSystem.CELL_SPLIT, args=args,
                                          input_coords=trainer.dataset.coords, only_train_vars=True)

                trainer.plot_debug_class_image(filenames[i], images[i], grid, epoch=e, tag="unittest",
                                               imageidx=ImageIdx.PRED, ignore_classes=[0])
                losses.append(trainer.losses[TRAIN_TAG]._backprops_[i].item())
            self.assertGreater(losses[0] * 0.8, losses[-1],
                               msg="Expected the loss %s to decrease a lot, but we have got\n%s" % (loss, losses))

    def test_trainer_loss(self):
        for loss in LOSS:
            if loss == LOSS.BINARY_CROSS_ENTROPY_MEAN or loss == LOSS.BINARY_CROSS_ENTROPY_SUM:
                continue

            args = test_setup(self._testMethodName + "_" + str(loss), str(Dataset.CULANE),
                              additional_vals={"batch_size": 1, "img_height": 128, "num_predictors": 1,
                                               "max_n": 1, "learning_rate": 0.0001, "retrain": True,
                                               "training_variables": [Variables.GEOMETRY, Variables.CONF],
                                               "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM],
                                               "activations": [ACTIVATION.SIGMOID, ACTIVATION.SIGMOID]
                                               })
            trainer = TrainHandler(args)

            images, grid_tensors, filenames, _, params = next(iter(trainer.loader))
            trainer.dataset.params_per_file.update(
                {f: {k: v[i] for k, v in params.items()} for i, f in enumerate(filenames)})

            losses = []
            for e in tqdm(range(5)):
                images, preds = trainer(filenames=filenames, images=images, grid_tensor=grid_tensors, is_train=True,
                                        first_run=False, epoch=e, image_idx_in_batch=0)
                Log.debug("Loss %s = %.3f" % (loss, trainer.losses[TRAIN_TAG]._backprops_[-1]))

                idx = 0
                grid, _ = GridFactory.get(data=torch.unsqueeze(preds.detach()[idx], dim=0), variables=[],
                                          coordinate=CoordinateSystem.CELL_SPLIT, args=args,
                                          input_coords=trainer.dataset.coords, only_train_vars=True)

                trainer.plot_debug_class_image(filenames[idx], images[idx], grid, epoch=e, tag="unittest",
                                               imageidx=ImageIdx.PRED, ignore_classes=[0])

                losses.append(trainer.losses[TRAIN_TAG]._backprops_[0].item())
            self.assertGreater(losses[0] * 0.8, losses[-1],
                               msg="Expected the loss %s to decrease a lot, but we have got\n%s" % (loss, losses))


if __name__ == '__main__':
    unittest.main()
