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
import os.path
import socket
import unittest

import numpy as np
import torch
from tqdm import tqdm

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid_factory import GridFactory
from yolino.model.activations import Softmax, get_activations, MidLenDirActivation, Sigmoid
from yolino.model.model_factory import get_checkpoint, get_model
from yolino.model.variable_structure import VariableStructure
from yolino.model.yolino_net import YolinoNet
from yolino.runner.evaluator import Evaluator
from yolino.runner.trainer import TrainHandler
from yolino.utils.enums import Dataset, LOSS, Variables, LINE, CoordinateSystem, ImageIdx
from yolino.utils.logger import Log
from yolino.utils.test_utils import test_setup
from yolino.viz.plot import plot_style_grid


class TestYoloClassification(unittest.TestCase):

    def test_checkpoint(self):
        base_dict = {"batch_size": 2, "img_height": 128,
                     "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM, LOSS.MSE_SUM],
                     "training_variables": [Variables.GEOMETRY, Variables.CLASS,
                                            Variables.CONF],
                     "num_predictors": 2, "activations": "sigmoid, softmax, sigmoid",
                     "eval_iteration": 1, "max_n": 10,
                     "darknet_weights": "",
                     "keep": True}

        base_dict["retrain"] = True
        args_retrain = test_setup(self._testMethodName, str(Dataset.CULANE), additional_vals=base_dict)
        coords = DatasetFactory.get_coords("train", args_retrain)

        trainer = TrainHandler(args=args_retrain)
        model = trainer.forward.model
        for epoch in range(0, args_retrain.eval_iteration + 1):
            for i, data in tqdm(enumerate(trainer.loader), total=len(trainer.loader)):
                images, grid_tensor, fileinfo, dupl, params = data
                trainer.dataset.params_per_file.update(
                    {f: {p: params[p][i] for p in params} for i, f in enumerate(fileinfo)})

                images, train_preds = trainer(filenames=fileinfo, images=images, grid_tensor=grid_tensor,
                                              epoch=epoch, image_idx_in_batch=i, is_train=True,
                                              first_run=False)
            trainer.on_train_epoch_finished(epoch=epoch, filenames=fileinfo, images=images, preds=train_preds.detach(),
                                            grid_tensors=grid_tensor)

            images, eval_preds = trainer(filenames=fileinfo, images=images, grid_tensor=grid_tensor, epoch=epoch,
                                         image_idx_in_batch=0, is_train=False, first_run=False)
        trainer.on_val_epoch_finished(epoch)

        path_eval = "/tmp/last_eval_pred.png"
        print("file://%s" % path_eval)
        self.viz(epoch, eval_preds, images, grid_tensor, path_eval, coords, args_retrain)
        path_train = "/tmp/last_train_pred.png"
        print("file://%s" % path_train)
        self.viz(epoch, train_preds, images, grid_tensor, path_train, coords, args_retrain)

        self.assertTrue(os.path.exists(args_retrain.paths.best_model))
        self.assertTrue(os.path.exists(args_retrain.paths.model))

        # Load best_checkpoint
        args_retrain.retrain = False
        best_checkpoint = get_checkpoint(args=args_retrain, load_best=True, print_debug=False)
        self.assertEqual(best_checkpoint["ID"], args_retrain.id)
        self.assertEqual(best_checkpoint["epoch"], epoch)

        self.assert_cp_equal_model(best_checkpoint, model, msg="Best checkpoint is not equal to the model")

        checkpoint = get_checkpoint(args=args_retrain, load_best=False, print_debug=False)
        self.assertEqual(checkpoint["ID"], args_retrain.id)
        self.assertEqual(checkpoint["epoch"], epoch)

        self.assert_cp_equal_model(checkpoint, model, msg="Checkpoint is not equal to the model")

        # test model loading
        tmp_model = get_model(args=args_retrain, coords=coords)

        ok = tmp_model.load_state_dict(best_checkpoint['model_state_dict'])
        self.assertEqual(len(ok.missing_keys) + len(ok.unexpected_keys), 0,
                         msg="Model did not load successfully: %s" % str(ok))
        self.assert_cp_equal_model(best_checkpoint, model, msg="Model did not load checkpoint data")
        self.assert_model_equal_model(model, tmp_model, msg="Model is not filled with best_checkpoint data correctly.")

        # test train prediction from stored checkpoint
        new_trainer = TrainHandler(args=args_retrain)
        new_model = new_trainer.forward.model
        self.assert_model_equal_model(model, new_model, msg="New loaded trainer model is not equal to stored model.")

        # trainer with train true
        new_images, reloaded_train_preds = new_trainer(filenames=fileinfo, images=images, grid_tensor=grid_tensor,
                                                       epoch=new_trainer.forward.start_epoch,
                                                       image_idx_in_batch=0, is_train=True, first_run=False)
        path_train = "/tmp/last_new_trainer_train_pred.png"
        print("file://%s" % path_train)
        self.viz(epoch, reloaded_train_preds, new_images, grid_tensor, path_train, coords, args_retrain)

        self.assertTrue(torch.equal(images, new_images))

        # Load model from best_checkpoint
        eval = Evaluator(args=args_retrain, prepare_forward=True, load_best_model=True, anchors=None)
        self.assert_model_equal_model(model, eval.forward.model,
                                      msg="Evaluator model is not equal to stored model.")
        self.assertEqual(eval.forward.start_epoch, epoch, msg="The evaluator model does not have the correct epoch "
                                                                  "(expected %d, got %d) and thus probably not the "
                                                                  "correct best_checkpoint." % (epoch + 1,
                                                                                                eval.forward.start_epoch))

    def viz(self, epoch, preds, images, grid_tensors, path, coords, args):
        grid, _ = GridFactory.get(data=torch.unsqueeze(preds[0].detach(), dim=0), variables=[],
                                  coordinate=CoordinateSystem.CELL_SPLIT, args=args,
                                  input_coords=coords, only_train_vars=True)
        gt_grid, _ = GridFactory.get(data=torch.unsqueeze(grid_tensors[0], dim=0), variables=[],
                                     coordinate=CoordinateSystem.CELL_SPLIT, args=args,
                                     input_coords=coords, only_train_vars=False)
        plot_style_grid(grid.get_image_lines(coords=coords, image_height=images[0].shape[1]), path, images[0],
                        coords=coords, cell_size=args.cell_size, show_grid=True,
                        coordinates=CoordinateSystem.UV_SPLIT, epoch=epoch, tag="unittest",
                        imageidx=ImageIdx.PRED, threshold=args.confidence,
                        gt=gt_grid.get_image_lines(coords=coords, image_height=images[0].shape[1]),
                        training_vars_only=True, level=1)

    def assert_cp_equal_model(self, checkpoint, model, msg):
        for checkpoint_key in checkpoint["model_state_dict"].keys():
            if not "bias" in checkpoint_key and not "weight" in checkpoint_key:
                continue
            model_entry = model.get_parameter(checkpoint_key)
            checkpoint_entry = checkpoint["model_state_dict"][checkpoint_key]
            self.assertTrue(torch.all(torch.abs(checkpoint_entry - model_entry) < 0.00001),
                            msg="%s\n%s\n%s"
                                % (msg,
                                   checkpoint_entry[
                                       torch.where(torch.abs(checkpoint_entry - model_entry) > 0.00001)].numpy(),
                                   model_entry[
                                       torch.where(
                                           torch.abs(checkpoint_entry - model_entry) > 0.00001)].detach().numpy()))

    def assert_model_equal_model(self, model_a, model_b, msg):
        for a, b in zip(model_a.parameters(), model_b.parameters()):
            self.assertTrue(torch.equal(a, b), msg="%s\n%s vs %s" % (msg, a.shape, b.shape))

    def test_activation(self):
        self.args = test_setup(self._testMethodName, str(Dataset.CULANE),
                               additional_vals={"batch_size": 2, "img_height": 128,
                                                "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM, LOSS.MSE_SUM],
                                                "training_variables": [Variables.GEOMETRY, Variables.CLASS,
                                                                       Variables.CONF],
                                                "num_predictors": 1, "activations": "linear, softmax, sigmoid"})
        coords = VariableStructure(num_classes=3, line_representation_enum=self.args.linerep,
                                   vars_to_train=[Variables.GEOMETRY, Variables.CLASS, Variables.CONF])
        activations = get_activations(self.args.activations, coords, linerep=self.args.linerep)
        logits = torch.randn((self.args.batch_size, self.args.grid_shape[0] * self.args.grid_shape[1],
                              self.args.num_predictors, coords.num_vars_to_train()))
        logits[:, :, :, coords.get_position_of(Variables.CONF)] *= 100

        probs = activations(logits.clone())

        # Geometry
        self.assertTrue(np.array_equal(logits[:, :, :, coords.get_position_of(Variables.GEOMETRY)],
                                       probs[:, :, :, coords.get_position_of(Variables.GEOMETRY)]),
                        msg="Geometry has linear activation and should not changed, but did: %s vs %s" %
                            (logits[0, 0, 0, coords.get_position_of(Variables.GEOMETRY)],
                             probs[0, 0, 0, coords.get_position_of(Variables.GEOMETRY)]))

        # Class
        self.assertAlmostEqual(first=(sum(probs[0, 0, 0, coords.get_position_of(Variables.CLASS)])).item(), second=1,
                               places=1, msg="Class has softmax activation, but does not sum to one: %s" %
                                             (probs[0, 0, 0, coords.get_position_of(Variables.CLASS)]))

        # Confidence
        self.assertLessEqual(probs[0, 0, 0, coords.get_position_of(Variables.CONF)].item(), 1,
                             msg="Confidence has sigmoid activation, but is greater than one: %s" %
                                 (probs[0, coords.get_position_of(Variables.CONF), 0, 0]))
        self.assertGreaterEqual(probs[0, 0, 0, coords.get_position_of(Variables.CONF)].item(), -1,
                                msg="Confidence has sigmoid activation, but is greater than one: %s" %
                                    (probs[0, 0, 0, coords.get_position_of(Variables.CONF)]))

    def test_softmax(self):
        self.args = test_setup(self._testMethodName, str(Dataset.CULANE),
                               additional_vals={"batch_size": 2, "img_height": 128,
                                                "num_predictors": 1})
        num_vars = 3
        data = torch.randn(
            (self.args.batch_size, self.args.grid_shape[0] * self.args.grid_shape[1], self.args.num_predictors,
             num_vars))
        activation = Softmax(dim=3, variable=Variables.CLASS,
                             coords=VariableStructure(num_classes=num_vars, num_conf=0,
                                                      line_representation_enum=self.args.linerep,
                                                      vars_to_train=Variables.CLASS))
        softed = activation(data)
        print(softed[0, 0, 0, :])
        self.assertAlmostEqual(first=(sum(softed[0, 0, 0, :])).item(), second=1, places=1, msg=(softed[0, 0, 0, :]))

    def test_md_activation(self):
        self.args = test_setup(self._testMethodName, str(Dataset.CULANE),
                               additional_vals={"batch_size": 10, "img_height": 128,
                                                "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM, LOSS.MSE_SUM],
                                                "training_variables": [Variables.GEOMETRY, Variables.CLASS,
                                                                       Variables.CONF],
                                                "num_predictors": 1,
                                                "activations": "sigmoid, softmax, sigmoid",
                                                "linerep": str(LINE.MID_DIR),
                                                "offset": False})

        coords = DatasetFactory.get_coords(split="train", args=self.args)
        activations = get_activations(self.args.activations, coords, linerep=self.args.linerep, offset=self.args.offset)
        self.assertIsInstance(activations.activations[0], MidLenDirActivation, activations.activations[0])
        self.assertIsInstance(activations.activations[1], Softmax, activations.activations[1])
        self.assertIsInstance(activations.activations[2], Sigmoid, activations.activations[2])
        self.assertEqual(len(activations.activations), 3, activations)

        logits = torch.randn((self.args.batch_size, self.args.grid_shape[0] * self.args.grid_shape[1],
                              self.args.num_predictors, coords.num_vars_to_train())) * 100
        probs = activations(logits.clone())

        input_means = logits.view(-1, coords.num_vars_to_train()).mean(axis=0)
        means = probs.view(-1, coords.num_vars_to_train()).mean(axis=0)

        self.assertTrue((means[0:2] >= 0.4).all(), means[0:3])
        self.assertTrue((means[0:2] <= 0.6).all(), means[0:3])
        self.assertTrue((means[2:4] >= -0.2).all(),
                        msg="Direction should be shifted to [-1,1] and thus have its average between [-0.1, 0.1]. "
                            "We've got mean=%s" % means[3:5])
        self.assertTrue((means[2:4] <= 0.2).all(),
                        msg="Direction should be shifted to [-1,1] and thus have its average between [-0.1, 0.1]. "
                            "We've got mean=%s" % means[3:5])

        max = probs.view(-1, coords.num_vars_to_train()).max(axis=0)[0]
        self.assertTrue((max[0:2] >= 0.9).all(), max[0:2])
        self.assertTrue((max[2:4] >= 0.9).all(), max[2:4])

        min = probs.view(-1, coords.num_vars_to_train()).min(axis=0)[0]
        self.assertTrue((min[0:2] <= 0.11).all(), min[0:2])
        self.assertTrue((min[2:4] <= -0.9).all(), min[2:4])

        def sigmoid(v):
            return 1 / (1 + math.exp(-v))

        for j in [0, 5, 20, 41]:
            for i in range(0, 2):
                self.assertAlmostEqual(sigmoid(logits[0, j, 0, i]), probs[0, j, 0, i].item(), places=5)
            self.assertAlmostEqual(sigmoid(logits[0, j, 0, 2]) * 2 - 1, probs[0, j, 0, 2].item(), places=5)
            self.assertAlmostEqual(sigmoid(logits[0, j, 0, 3]) * 2 - 1, probs[0, j, 0, 3].item(), places=5)

    def test_md_activation_offset(self):
        self.args = test_setup(self._testMethodName, str(Dataset.CULANE),
                               additional_vals={"batch_size": 10, "img_height": 128,
                                                "loss": [LOSS.MSE_SUM, LOSS.MSE_SUM, LOSS.MSE_SUM],
                                                "training_variables": [Variables.GEOMETRY, Variables.CLASS,
                                                                       Variables.CONF],
                                                "num_predictors": 1, "activations": "sigmoid, softmax, sigmoid",
                                                "linerep": str(LINE.MID_DIR), "offset": True})
        coords = DatasetFactory.get_coords(split="train", args=self.args)
        activations = get_activations(self.args.activations, coords, linerep=self.args.linerep, offset=self.args.offset)
        self.assertIsInstance(activations.activations[0], MidLenDirActivation, activations.activations[0])
        self.assertIsInstance(activations.activations[1], Softmax, activations.activations[1])
        self.assertIsInstance(activations.activations[2], Sigmoid, activations.activations[2])
        self.assertEqual(len(activations.activations), 3, activations)

        logits = torch.randn((self.args.batch_size, self.args.grid_shape[0] * self.args.grid_shape[1],
                              self.args.num_predictors, coords.num_vars_to_train())) * 100
        probs = activations(logits.clone())

        input_means = logits.view(-1, coords.num_vars_to_train()).mean(axis=0)
        means = probs.view(-1, coords.num_vars_to_train()).mean(axis=0)

        self.assertTrue((means[0:4] >= -0.2).all(), means[0:4])
        self.assertTrue((means[0:4] <= 0.2).all(), means[0:4])

        max = probs.view(-1, coords.num_vars_to_train()).max(axis=0)[0]
        self.assertTrue((max[0:2] >= 0.8).all(), max[0:2])
        self.assertTrue((max[2:4] >= 1.8).all(), max[2:4])

        min = probs.view(-1, coords.num_vars_to_train()).min(axis=0)[0]
        self.assertTrue((min[0:2] <= -0.8).all(), min[0:2])
        self.assertTrue((min[2:4] <= -1.8).all(), min[2:4])

        def sigmoid(v):
            return 1 / (1 + math.exp(-v))

        def tanh(x):
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

        for j in [0, 5, 20, 41]:
            for i in range(0, 2):
                self.assertAlmostEqual(sigmoid(logits[0, j, 0, i]) * 2 - 1, probs[0, j, 0, i].item(), places=6)
            self.assertAlmostEqual(sigmoid(logits[0, j, 0, 2]) * 4 - 2, probs[0, j, 0, 2].item(), places=6)
            self.assertAlmostEqual(sigmoid(logits[0, j, 0, 3]) * 4 - 2, probs[0, j, 0, 3].item(), places=6)

    def test_reshapes(self):
        args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                          additional_vals={"batch_size": 1, "max_n": 1, "img_height": 128,
                                           "loss": [LOSS.MSE_SUM],
                                           "training_variables": [Variables.CONF],
                                           "num_predictors": 3, "activations": "sigmoid",
                                           "linerep": str(LINE.POINTS),
                                           "weights": 1,
                                           "plot": socket.gethostname() == "mrt057"},
                          # show_params=True
                          )

        coords = DatasetFactory.get_coords(split="train", args=args)
        net = YolinoNet(args, coords)

        input = torch.linspace(0, 256, args.grid_shape[0] * args.grid_shape[1]).reshape(
            [args.grid_shape[0], args.grid_shape[1]])
        input = torch.tile(input, [args.batch_size, args.num_predictors, 1, 1])

        output = net.reshape_prediction(input)

        if args.plot:
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.subplot(211)
            plt.title("Input")
            plt.imshow(input.squeeze(0).permute(1, 2, 0) / 256.)

            plt.subplot(212)
            plt.title("Output")
            plt.imshow(output.squeeze(0).reshape([args.grid_shape[0], args.grid_shape[1], 3]) / 256.)

            path = "/tmp/reshape_comparison.png"
            Log.warning("Plot to file://%s" % path)
            plt.savefig(path)

        compare_view = input.squeeze(0).permute(1, 2, 0)
        out_compare_view = output.squeeze(0).reshape([args.grid_shape[0], args.grid_shape[1], 3])
        self.assertTrue(torch.equal(compare_view, out_compare_view), msg="%s vs %s" % (compare_view, out_compare_view))

    def test_scale(self):
        for scale in [32, 16, 8]:
            args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                              additional_vals={"batch_size": 1, "max_n": 1, "img_height": 160,
                                               "num_predictors": 3,
                                               "plot": socket.gethostname() == "mrt057",
                                               "scale": scale},
                              )

            dataset, ok = DatasetFactory.get(dataset_enum=args.dataset, split="train", args=args, only_available=True,
                                             shuffle=False, augment=False)
            model = get_model(args, dataset.coords)

            if scale <= 16:
                self.assertTrue(hasattr(model, "upsample"))
                self.assertTrue(hasattr(model, "postup"))

                if scale <= 8:
                    self.assertTrue(hasattr(model, "upsample2"))
                    self.assertTrue(hasattr(model, "postup2"))

            model = model.train(True)

            images, _, _, _, _ = dataset.__getitem__(0)
            logits = model(images.unsqueeze(0))
            expected_shape = [args.batch_size, np.prod(np.asarray(args.img_size) / scale),
                              args.num_predictors, dataset.coords.num_vars_to_train()]
            self.assertTrue(np.array_equal(logits.shape, expected_shape),
                            msg="Logits at scale %d do not match: %s vs %s" % (scale, logits.shape, expected_shape))

            activations = get_activations(args.activations, dataset.coords, args.linerep)
            outputs = activations(logits)

            self.assertTrue(np.array_equal(outputs.shape, expected_shape),
                            msg="Outputs at scale %d do not match: %s vs %s" % (scale, outputs.shape, expected_shape))


if __name__ == '__main__':
    unittest.main()
