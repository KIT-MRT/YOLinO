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

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid_factory import GridFactory
from yolino.model.activations import get_activations
from yolino.model.loss import get_loss
from yolino.model.loss_container import LossContainer
from yolino.model.model_factory import load_checkpoint, save_best_checkpoint, save_checkpoint
from yolino.model.optimizer_factory import get_optimizer
from yolino.runner.evaluator import Evaluator
from yolino.runner.forward_runner import ForwardRunner
from yolino.utils.enums import CoordinateSystem, ImageIdx, Variables, LossWeighting, AnchorDistribution
from yolino.utils.logger import Log
from yolino.viz.plot import plot_cell_class, convert_to_torch_image, plot_style_grid

VAL_TAG = "val"
TRAIN_TAG = "train"


class TrainHandler:
    def __init__(self, args) -> None:
        self.args = args

        # Load data
        self.dataset, self.loader = DatasetFactory.get(args.dataset, only_available=True, split="train", args=args,
                                                       shuffle=True, augment=True, ignore_duplicates=False)
        self.val_dataset, self.val_loader = DatasetFactory.get(args.dataset, only_available=True, split="val",
                                                               args=args, shuffle=False, augment=False,
                                                               ignore_duplicates=False)
        Log.upload_params({"train_imgs": len(self.dataset), "val_imgs": len(self.val_dataset),
                           "dataset_path": self.dataset.dataset_path})

        # model
        self.model, scheduler_checkpoint, self.model_epoch = load_checkpoint(args, self.dataset.coords)

        num_train_vars = len(self.dataset.coords.train_vars())
        num_cells = np.prod(args.grid_shape)

        # Argoverse has (per scale) 32: 106; 16: 201; 8: 389 line segments in an image (on average)
        num_lines = 106 if args.scale == 32 else (201 if args.scale == 16 else 389)

        activation_is_exp = [get_activations(args.activations, coords=self.dataset.coords,
                                             linerep=self.dataset.coords.line_representation.enum).activations[
                                 i].is_exp() for i in range(num_train_vars)]
        self.loss_weights = self.__init_loss_weights__(num_train_vars=num_train_vars,
                                                       cuda=self.args.cuda, init_weights=args.weights,
                                                       loss_weighting=args.loss_weight_strategy,
                                                       average_num_lines=num_lines, num_cells=int(num_cells),
                                                       num_preds=args.num_predictors, is_exponential=activation_is_exp)

        self.conf_loss_weights = self.__init_conf_loss_weights__(cuda=self.args.cuda,
                                                                 init_weights=args.conf_match_weight,
                                                                 loss_weighting=args.loss_weight_strategy,
                                                                 average_num_lines=num_lines, num_cells=int(num_cells),
                                                                 num_preds=args.num_predictors,
                                                                 is_exponential=[activation_is_exp[
                                                                                     -1]] * num_train_vars)

        self.forward = ForwardRunner(args, self.model, self.model_epoch)
        self.loss_fct = get_loss(losses=args.loss, args=args, coords=self.dataset.coords,
                                 weights=self.loss_weights, anchors=self.dataset.anchors,
                                 conf_weights=self.conf_loss_weights)
        self.optimizer, self.scheduler = get_optimizer(args, net=self.forward.model,
                                                       loss_weights=[*self.loss_weights, *self.conf_loss_weights])
        Log.gradients(model=self.model)

        self.evaluator = Evaluator(args, coords=self.dataset.coords, prepare_forward=False,
                                   anchors=self.dataset.anchors)
        self.evaluator.forward = self.forward

        self.best_mean_loss = np.inf
        self.best_gradient_loss = np.inf
        self.best_epoch = -1
        self.last_loss = torch.inf

        self.losses = {}
        self.losses[TRAIN_TAG] = LossContainer(self.args.training_variables, len(self.loader), self.loss_weights)
        self.losses[VAL_TAG] = LossContainer(self.args.training_variables, len(self.val_loader), self.loss_weights)

    @staticmethod
    def __init_loss_weights__(num_train_vars, cuda, loss_weighting: LossWeighting, is_exponential,
                              init_weights=None, average_num_lines=120, num_cells=120, num_preds=8):
        return TrainHandler.__init_general_loss_weights(num_train_vars, cuda, loss_weighting, is_exponential,
                                                        init_weights, average_num_lines,
                                                        num_cells, num_preds, ignore_weight=-1)

    @staticmethod
    def __init_general_loss_weights(num_train_vars, cuda, loss_weighting: LossWeighting, is_exponential,
                                    init_weight_factor=None, average_num_lines=120, num_cells=120, num_preds=8,
                                    ignore_weight=None):
        # initialize
        if init_weight_factor is None:
            init_weight_factor = [1] * num_train_vars

        if "calculate" in init_weight_factor:
            # reduce by number of geom vars = 4
            init_weight_factor = [((num_cells * num_preds) / average_num_lines) / 4, *[1.] * (num_train_vars - 1)]
            Log.warning(f"We calculated the weights={init_weight_factor}, "
                        f"we had ({num_cells} * {num_preds}) / {average_num_lines}")

        init_weight_factor = torch.tensor(init_weight_factor)

        # calculate trainable weight
        if loss_weighting == LossWeighting.FIXED_NORM:
            train_weights = init_weight_factor / sum(init_weight_factor)
            Log.warning(f"Final weights={train_weights} with strategy {loss_weighting}")

        elif loss_weighting == LossWeighting.FIXED:
            train_weights = init_weight_factor
            Log.warning(f"Final weights={train_weights} with strategy {loss_weighting}")

        elif loss_weighting == LossWeighting.LEARN_LOG or loss_weighting == LossWeighting.LEARN_LOG_NORM:
            train_weights = torch.zeros((num_train_vars), dtype=float)
            for i in range(num_train_vars):

                if loss_weighting == LossWeighting.LEARN_LOG_NORM:
                    wi = init_weight_factor[i] / sum(init_weight_factor)  # init also with normed values
                else:
                    wi = init_weight_factor[i]

                if is_exponential[i]:
                    train_weights[i] = torch.log(1 / wi)
                else:
                    train_weights[i] = torch.log(1 / (2. * wi))

            # conf weight should not be trained if we also have individual conf weights
            if ignore_weight is not None:
                if ignore_weight == -1:
                    ignore_weight = num_train_vars - 1
                train_weights[ignore_weight] = 1

            Log.warning(f"Final trainable weights s=log(sigma**2)={train_weights.numpy()} "
                        f"with strategy {loss_weighting}")
            Log.warning(f"Final weight factor w=(1 / (2*e^s))={(1 / (2 * torch.exp(train_weights))).numpy()} "
                        f"with strategy {loss_weighting}")

        elif loss_weighting == LossWeighting.LEARN:
            train_weights = torch.ones((num_train_vars), dtype=float)

            # conf weight should not be trained if we also have individual conf weights
            if ignore_weight is not None:
                if ignore_weight == -1:
                    ignore_weight = num_train_vars - 1
                train_weights[ignore_weight] = 1

            Log.warning(f"Final trainable weights s={train_weights.numpy()} "
                        f"with strategy {loss_weighting}")
            Log.warning(f"Final weight factor w=(1 / (2*s^2))={(1 / (2 * (train_weights ** 2))).numpy()} "
                        f"with strategy {loss_weighting}")
        else:
            raise NotImplementedError(f"We do not know {loss_weighting}")

        if len(train_weights) != num_train_vars:
            raise ValueError("You want %s variables, but provided a list of %s" % (num_train_vars, train_weights))

        # make optimizable if necessary
        do_training = [(loss_weighting == LossWeighting.LEARN_LOG or loss_weighting == LossWeighting.LEARN
                        or loss_weighting == LossWeighting.LEARN_LOG_NORM)
                       and (ignore_weight is None or ignore_weight != i)
                       for i in range(num_train_vars)]
        grad_train_weights = [torch.tensor((train_weights[i]), dtype=torch.float,
                                           # conf weight should not be trained if we also have individual conf weights
                                           requires_grad=do_training[i],
                                           device=cuda) for i in range(num_train_vars)]
        return grad_train_weights

    @staticmethod
    def __init_conf_loss_weights__(cuda, loss_weighting: LossWeighting, is_exponential,
                                   init_weights=None, average_num_lines=120, num_cells=120, num_preds=8):
        return TrainHandler.__init_general_loss_weights(num_train_vars=2, cuda=cuda, loss_weighting=loss_weighting,
                                                        init_weight_factor=init_weights,
                                                        average_num_lines=average_num_lines,
                                                        num_cells=num_cells, num_preds=num_preds,
                                                        is_exponential=is_exponential)

    def __call__(self, filenames, images, grid_tensor, epoch, image_idx_in_batch, is_train=True, first_run=False):
        if is_train:
            self.optimizer.zero_grad()

        outputs = self.forward(images, is_train=is_train, epoch=epoch, first_run=first_run)

        if image_idx_in_batch == 0 and epoch == 0:
            self.on_data_loaded(filenames[0], images[0], grid_tensor[0], outputs[0].detach().cpu(), is_train=is_train)
            Log.debug("Data reporting finished..")

        # Loss + Backward
        if self.args.cuda not in str(outputs.device):
            Log.warning("Somehow the output of the network was not on %s" % str(self.args.cuda))
            outputs = outputs.to(self.args.cuda)

        if self.args.cuda not in str(grid_tensor.device):
            grid_tensor = grid_tensor.to(self.args.cuda)

        if False:
            for b in range(len(grid_tensor)):
                for c in range(len(grid_tensor[b])):

                    col = c % self.args.grid_shape[1]
                    row = int(c / self.args.grid_shape[1])
                    for p in torch.where(grid_tensor[b, c, :, 4] > self.args.confidence):
                        self.dataset.anchors.add_conf_heatmap(row, col, p)

        if False:
            plt.clf()
            confs = outputs[:, :, :, 4].detach().flatten().numpy()
            print(np.histogram(confs, bins=12, range=[-0.1, 1.1])[0])
            plt.hist(confs, 12, range=[-0.1, 1.1])
            name = "/tmp/seaborn_conf_hist.png"
            Log.warning("Log to file://%s" % name)
            plt.savefig(name)

        loss_weight_dict = {}
        try:
            losses, sum_loss, mean_losses = self.loss(grid_tensor, outputs, epoch=epoch, filenames=filenames,
                                                      tag=TRAIN_TAG if is_train else VAL_TAG)  # input shape [batch, cells, preds, vars]
            if is_train:
                self.backward(sum_loss)

                for j, v in enumerate(self.dataset.coords.train_vars()):
                    std_1 = torch.exp(self.loss_weights[j]) ** 0.5
                    Log.debug("Weight for %s is %f with std %f" % (v, self.loss_weights[j].item(), std_1.item()))
                    loss_weight_dict[os.path.join("loss_" + "conf" if v == Variables.CONF else str(v), "weight")] = \
                        self.loss_weights[j]

            self.losses[TRAIN_TAG if is_train else VAL_TAG].add_backprop(epoch=epoch,
                                                                         loss=sum_loss.detach().cpu().item(),
                                                                         i=image_idx_in_batch)
            self.losses[TRAIN_TAG if is_train else VAL_TAG].add(epoch=epoch, mean_losses=mean_losses,
                                                                config_losses=losses, i=image_idx_in_batch)
        except (ValueError, NotImplementedError, RuntimeError) as e:
            Log.debug(e)
            pred_grid, _ = GridFactory.get(torch.unsqueeze(outputs[0].detach().cpu(), dim=0), [],
                                           CoordinateSystem.CELL_SPLIT, self.args,
                                           input_coords=self.dataset.coords, threshold=self.args.confidence,
                                           only_train_vars=True, anchors=self.dataset.anchors)
            grid, _ = GridFactory.get(torch.unsqueeze(grid_tensor[0].cpu(), dim=0), [],
                                      CoordinateSystem.CELL_SPLIT, self.args,
                                      input_coords=self.dataset.coords, threshold=self.args.confidence,
                                      only_train_vars=False, anchors=self.dataset.anchors)

            pred_grid_uv_lines = pred_grid.get_image_lines(coords=self.dataset.coords,
                                                           image_height=images[0].shape[1])

            if is_train:
                full_size_img = self.dataset.get_full_size_image(filename=filenames[0])
            else:
                full_size_img = self.val_dataset.get_full_size_image(filename=filenames[0])

            self.plot_debug_images(filenames[0], full_size_img,
                                   pred_grid_uv_lines, epoch,
                                   tag=TRAIN_TAG if is_train else VAL_TAG,
                                   suffix="indef",
                                   imageidx=ImageIdx.GRID,
                                   coordinates=CoordinateSystem.UV_SPLIT,
                                   gt=grid.get_image_lines(coords=self.dataset.coords,
                                                           image_height=full_size_img.shape[1]))

            Log.debug("Indef loss")
            self.on_indef(epoch, filenames, exception=e, is_train=is_train, batch_index=image_idx_in_batch)
            import math
        except AttributeError as e:
            Log.error("Something is wrong with these losses: %s. We continue..." % str(e))
            self.losses[TRAIN_TAG if is_train else VAL_TAG].add(epoch, mean_losses=mean_losses, config_losses=losses,
                                                                i=image_idx_in_batch)

        return images, outputs

    def on_indef(self, epoch, filenames, exception, batch_index, is_train=True):
        if self.best_mean_loss != np.inf or epoch > 10 or is_train:
            epoch_losses = self.losses[TRAIN_TAG if is_train else VAL_TAG]._backprops_[-1]
            Log.fail_summary(epoch, epoch_losses if len(epoch_losses) > 0 else [-1],
                             tag=TRAIN_TAG if is_train else VAL_TAG, level=1)
            raise Exception(filenames) from exception
        else:
            if epoch > 0:
                Log.warning("We ran into nan - continue")
            self.losses[TRAIN_TAG if is_train else VAL_TAG].add(epoch=epoch, mean_losses=torch.nan,
                                                                config_losses=torch.nan, i=batch_index)

    def on_data_loaded(self, filename, image, grid_tensor, preds, is_train=True):
        epoch = 0
        if image.shape[1] != self.args.img_size[0] or image.shape[2] != self.args.img_size[1]:
            raise ValueError("Image has %s, we want %s" % (image.shape, self.args.img_size))

        if filename in self.dataset.file_names:
            full_size_img = self.dataset.get_full_size_image(filename=filename)
        elif filename in self.val_dataset.file_names:
            full_size_img = self.val_dataset.get_full_size_image(filename=filename)
        else:
            full_size_img = image

        grid, _ = GridFactory.get(torch.unsqueeze(grid_tensor, dim=0), [], CoordinateSystem.CELL_SPLIT, self.args,
                                  input_coords=self.dataset.coords, threshold=self.args.confidence,
                                  anchors=self.dataset.anchors)

        grid_uv_lines = grid.get_image_lines(coords=self.dataset.coords, image_height=full_size_img.shape[1])

        if preds is not None:
            pred_grid, _ = GridFactory.get(torch.unsqueeze(preds, dim=0), [], CoordinateSystem.CELL_SPLIT,
                                           self.args, anchors=self.dataset.anchors,
                                           input_coords=self.dataset.coords, threshold=self.args.confidence,
                                           only_train_vars=True)
            pred_grid_uv_lines = pred_grid.get_image_lines(coords=self.dataset.coords,
                                                           image_height=full_size_img.shape[1], is_training_data=True)
            ok = self.plot_debug_images(filename, full_size_img,
                                        pred_grid_uv_lines, epoch,
                                        tag=TRAIN_TAG if is_train else VAL_TAG, suffix="preview",
                                        imageidx=ImageIdx.PRED,
                                        coordinates=CoordinateSystem.UV_SPLIT, gt=grid_uv_lines,
                                        training_vars_only=True, cell_size=grid.get_cell_size(full_size_img.shape[1]))
        else:
            ok = self.plot_debug_images(filename, full_size_img,
                                        grid_uv_lines, epoch, tag=TRAIN_TAG if is_train else VAL_TAG, suffix="preview",
                                        imageidx=ImageIdx.GRID,
                                        coordinates=CoordinateSystem.UV_SPLIT,
                                        cell_size=grid.get_cell_size(full_size_img.shape[1]))

        del grid

    def on_train_epoch_finished(self, epoch, filenames, images, preds, grid_tensors):
        if self.args.gpu:
            preds = preds.to("cpu")
            images = images.to("cpu")

        if self.losses[TRAIN_TAG].current_epoch == epoch:
            self.losses[TRAIN_TAG].log(epoch=epoch, tag=TRAIN_TAG)
            sum_loss = self.losses[TRAIN_TAG].sum(epoch=epoch)
            if sum_loss > 4 * abs(self.last_loss):
                Log.glitch(self.args, preds)
            self.last_loss = sum_loss
        else:
            Log.error("Wrong epoch")

        if self.is_time_for_image(epoch):
            self.plot_prediction(epoch, filenames, grid_tensors, images, preds, indices=[0])  # range(len(filenames)))

        scores = []
        if self.is_time_for_val(epoch):
            scores = self.evaluator.publish_scores(epoch=epoch, tag=TRAIN_TAG)

        # Checkpoint
        save_checkpoint(self.args, self.model, self.optimizer, self.scheduler, epoch, self.args.id)

        if False:
            self.plot_heatmap(self.dataset.anchors.conf_heatmap, "conf")
            self.plot_heatmap(self.dataset.anchors.heatmap)

        Log.push(epoch)
        return scores

    def plot_heatmap(self, heatmap, title=""):
        import seaborn as sns
        sns_dict = {"anchor": [], "heat": [], "row": [], "col": []}
        for r in range(len(heatmap)):
            for c in range(len(heatmap[r])):
                for a in range(len(heatmap[r, c])):
                    sns_dict["heat"].append(heatmap[r, c, a].item())
                    sns_dict["anchor"].append(a)
                    sns_dict["row"].append(r)
                    sns_dict["col"].append(c)
        import pandas as pd
        sns_df = pd.DataFrame(sns_dict)
        g = sns.FacetGrid(sns_df, col="anchor")

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            sns.heatmap(d, **kwargs)

        g.map_dataframe(draw_heatmap, "col", "row", "heat")
        name = "/tmp/seaborn_heatmap_%s.png" % title
        Log.warning("Log to file://%s" % name)
        plt.savefig(name)

    def plot_prediction(self, epoch, filenames, grid_tensors, images, preds, indices,
                        image_idx: ImageIdx = ImageIdx.PRED):
        for idx in indices:
            grid, _ = GridFactory.get(data=torch.unsqueeze(preds[idx], dim=0), variables=[],
                                      coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                      input_coords=self.dataset.coords, only_train_vars=True,
                                      anchors=self.dataset.anchors)
            gt_grid, _ = GridFactory.get(data=torch.unsqueeze(grid_tensors[idx], dim=0), variables=[],
                                         coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                         input_coords=self.dataset.coords, only_train_vars=False,
                                         anchors=self.dataset.anchors)

            if Variables.CLASS in self.dataset.coords.train_vars():
                ok = self.plot_debug_class_image(filenames[idx], images[idx], grid, epoch=epoch, tag=TRAIN_TAG,
                                                 imageidx=image_idx, ignore_classes=[0])
            try:
                full_size_img = self.dataset.get_full_size_image(filename=filenames[idx])
            except FileNotFoundError as e:
                full_size_img = images[idx]
            self.plot_debug_images(full_basename=filenames[idx], image=full_size_img,
                                   uv_lines=grid.get_image_lines(coords=self.dataset.coords,
                                                                 image_height=full_size_img.shape[1],
                                                                 is_training_data=True),
                                   epoch=epoch,
                                   tag=TRAIN_TAG, show_grid=True, imageidx=image_idx,
                                   coordinates=CoordinateSystem.UV_SPLIT,
                                   gt=gt_grid.get_image_lines(coords=self.dataset.coords,
                                                              image_height=full_size_img.shape[1]),
                                   training_vars_only=True, cell_size=grid.get_cell_size(full_size_img.shape[1]))

    def on_images_finished(self, preds, grid_tensor, filenames, images, epoch, is_train, num_duplicates):
        if self.is_time_for_val(epoch):
            if torch.all(preds.isnan()):
                Log.error("Prediction is nan. Continue.")
                return

            if self.args.full_eval:
                try:
                    preds_uv, gt_uv = self.evaluator.prepare_uv(preds=preds, grid_tensors=grid_tensor,
                                                                filenames=filenames,
                                                                images=images)
                except ValueError as ex:
                    if epoch <= 5:
                        Log.error("We ran in to nan - continue for now")
                        return
                    else:
                        raise ex

                self.evaluator.get_scores_uv(gt_uv=gt_uv, preds_uv=preds_uv, epoch=epoch, filenames=filenames,
                                             num_duplicates=num_duplicates, tag=TRAIN_TAG if is_train else VAL_TAG,
                                             do_matching=True)
            else:
                self.evaluator.get_scores_in_cell(grid_tensor, preds, epoch, filenames, num_duplicates=num_duplicates,
                                                  tag=TRAIN_TAG if is_train else VAL_TAG,
                                                  do_matching=self.args.anchors == AnchorDistribution.NONE)

        if not is_train:
            self.plot_val_summary(epoch, filenames, grid_tensor, images, preds)

    def plot_val_summary(self, epoch, filenames, grid_tensor, images, preds):
        if self.args.gpu:
            preds = preds.to("cpu")
            images = images.to("cpu")
        viz_files = [
            "val/a33a44fb-6008-3dc2-b7c5-2d27b70741e8/sensors/cameras/ring_front_center/315967270349927216.jpg",
            "val/42f92807-0c5e-3397-bd45-9d5303b4db2a/sensors/cameras/ring_front_center/315976319849927222.jpg",
            "val/02a00399-3857-444e-8db3-a8f58489c394/sensors/cameras/ring_front_center/315966079549927215.jpg",
            "val/24642607-2a51-384a-90a7-228067956d05/sensors/cameras/ring_front_center/315970919849927217.jpg",
            "val/6803104a-bb06-402e-8471-e5af492db0a8/sensors/cameras/ring_front_center/315977901249927221.jpg",
            "val/9a448a80-0e9a-3bf0-90f3-21750dfef55a/sensors/cameras/ring_front_center/315975805899927216.jpg",
            "val/d3ca0450-2167-38fb-b34b-449741cb38f3/sensors/cameras/ring_front_center/315968247249927218.jpg",
            "val/e1d68dde-22a9-3918-a526-0850b21ff2eb/sensors/cameras/ring_front_center/315969765649927217.jpg",
            "val/0fb7276f-ecb5-3e5b-87a8-cc74c709c715/sensors/cameras/ring_front_center/315968086249927214.jpg",
            # tusimple
            "0531/1492626287507231547/20.jpg",
        ]

        if not "ring_front_center" in filenames[0] or self.args.max_n == 1:
            viz_files.append(filenames[0])

        for viz_file_name in viz_files:
            if viz_file_name in filenames:
                idx = np.where(viz_file_name == np.asarray(filenames))[0][0]
            else:
                continue

            grid, _ = GridFactory.get(data=torch.unsqueeze(preds[idx], dim=0), variables=[],
                                      coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                      input_coords=self.dataset.coords, only_train_vars=True,
                                      anchors=self.dataset.anchors)
            gt_grid, _ = GridFactory.get(data=torch.unsqueeze(grid_tensor[idx], dim=0), variables=[],
                                         coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                         input_coords=self.dataset.coords, only_train_vars=False,
                                         anchors=self.dataset.anchors)
            self.plot_debug_images(full_basename=filenames[idx], image=images[idx],
                                   uv_lines=grid.get_image_lines(coords=self.dataset.coords,
                                                                 image_height=images[idx].shape[1]), epoch=epoch,
                                   tag=VAL_TAG, show_grid=True, imageidx=ImageIdx.PRED,
                                   coordinates=CoordinateSystem.UV_SPLIT,
                                   gt=gt_grid.get_image_lines(coords=self.dataset.coords,
                                                              image_height=images[idx].shape[1]))

    def on_val_epoch_finished(self, epoch):
        self.losses[VAL_TAG].log(epoch=epoch, tag=VAL_TAG)

        new_best_mean_loss = self.losses[VAL_TAG].mean(epoch)
        gradient_loss = self.losses[VAL_TAG].backprop(epoch)

        if self.args.best_mean_loss:
            if new_best_mean_loss < self.best_mean_loss - 0.0001:
                self.best_gradient_loss = gradient_loss
                self.best_mean_loss = new_best_mean_loss
                self.best_epoch = epoch
                Log.info('Best mean loss: %f' % self.best_mean_loss)
                self.save_new_best(epoch)
        else:
            if gradient_loss < self.best_gradient_loss - 0.0001:
                self.best_gradient_loss = gradient_loss
                self.best_mean_loss = new_best_mean_loss
                self.best_epoch = epoch
                Log.info('Best gradient loss: %f' % self.best_gradient_loss)
                self.save_new_best(epoch)

        self.evaluator.publish_scores(epoch=epoch, tag=VAL_TAG)
        Log.push(epoch)

    def save_new_best(self, epoch):
        save_best_checkpoint(self.args, self.forward.model, self.optimizer, self.scheduler, epoch, self.args.id)
        Log.scalars(tag=VAL_TAG, dict={"loss/best/mean": self.best_mean_loss,
                                       "loss/best/gradient": self.best_gradient_loss,
                                       "epoch/best": self.best_epoch}, epoch=epoch)

    def is_converged(self, epoch):
        if epoch < self.args.earliest_stop:
            return False

        if self.args.patience <= 0:
            return False

        converged = (epoch - self.best_epoch) / self.args.eval_iteration >= self.args.patience
        if converged:
            Log.warning("No improvement for %d epochs. We stop with patience=%d." % (
                epoch - self.best_epoch, self.args.patience))
            Log.tag("early_stop")
        return converged

    def on_training_finished(self, epoch, do_nms):

        Log.warning('**** Best Model Eval %s ****' % (self.args.id))
        best_evaluator = Evaluator(args=self.evaluator.args, anchors=self.evaluator.anchors,
                                   coords=self.evaluator.coords, load_best_model=True)
        # best_evaluator.plot = True
        for i, data in tqdm(enumerate(self.loader), total=len(self.loader)):
            images, grid_tensor, fileinfo, dupl, params = data

            num_duplicates = int(sum(dupl["total_duplicates_in_image"]).item())
            preds, _ = best_evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo, tag="best/val",
                                      epoch=None, num_duplicates=num_duplicates)
            self.plot_val_summary(epoch, filenames=fileinfo, grid_tensor=grid_tensor, images=images, preds=preds)
            preds, _ = best_evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo, tag="best/val",
                                      do_in_uv=False, epoch=None, num_duplicates=num_duplicates)
        best_evaluator.publish_scores(epoch=None, tag="best/" + VAL_TAG)
        Log.push(None)

        if do_nms:
            Log.warning('**** NMS Training Data Eval %s ****' % (self.args.id))
            for i, data in tqdm(enumerate(self.loader), total=len(self.loader)):
                images, grid_tensor, fileinfo, dupl, params = data
                num_duplicates = int(sum(dupl["total_duplicates_in_image"]).item())
                self.evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo, tag="nms_train",
                               apply_nms=True, epoch=epoch, num_duplicates=num_duplicates)
            self.evaluator.publish_scores(epoch=epoch, tag=TRAIN_TAG)

            Log.warning('**** NMS Validation Data Eval %s ****' % (self.args.id))
            for i, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                images, grid_tensor, fileinfo, dupl, params = data
                num_duplicates = int(sum(dupl["total_duplicates_in_image"]).item())
                self.evaluator(images=images, grid_tensor=grid_tensor, idx=i, filenames=fileinfo, tag="nms_val",
                               apply_nms=True, epoch=epoch, num_duplicates=num_duplicates)

            self.evaluator.publish_scores(epoch=epoch, tag=VAL_TAG)

        Log.info('Finished Training')
        Log.finish()

    def loss(self, grid_tensor, outputs, filenames, epoch, tag="dummy_trainer"):
        """

        Args:
            grid_tensor (torch.tensor):
                with shape [batch, cells, preds, vars]

            outputs (list):
                with num_train_vars entries
        """
        losses, sum_loss, mean_losses = self.loss_fct(outputs, grid_tensor, filenames=filenames, epoch=epoch,
                                                      tag=tag)  # should have (batch, ..., classes)
        if torch.isnan(sum_loss) or torch.isinf(sum_loss):
            raise ValueError("Loss ran into NaN or infinity")

        return losses, sum_loss, mean_losses

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()

    def is_time_for_val(self, epoch):
        return epoch % self.args.eval_iteration == 0

    def plot_debug_class_image(self, full_basename, image, grid, epoch, tag="unknown", imageidx=ImageIdx.DEFAULT,
                               ignore_classes=[]):
        img_path = self.args.paths.generate_debug_image_file_path(full_basename + "_class.png", imageidx, suffix=epoch)
        _, ok = plot_cell_class(grid=grid, image=image, name=img_path, epoch=epoch, tag=tag + "_class",
                                imageidx=imageidx,
                                max_class=self.dataset.num_classes(), ignore_classes=ignore_classes,
                                threshold=self.args.confidence)
        return ok

    def plot_debug_images(self, full_basename, image, uv_lines, epoch=-1, tag="train",
                          show_grid=True, imageidx: ImageIdx = ImageIdx.LABEL,
                          coordinates=CoordinateSystem.UV_CONTINUOUS, gt=None, training_vars_only=False, suffix="",
                          cell_size=None):
        img_path = self.args.paths.generate_debug_image_file_path(full_basename, imageidx, suffix=suffix + "_" + tag)
        ok = plot_style_grid(uv_lines, img_path, image, coords=self.dataset.coords,
                             cell_size=self.args.cell_size if cell_size is None else cell_size,
                             show_grid=show_grid, coordinates=coordinates, epoch=epoch, tag=tag,
                             imageidx=imageidx, threshold=self.args.confidence, gt=gt,
                             training_vars_only=training_vars_only,
                             level=1)
        return ok

    def plot_debug_class_grid(self, preds, images, epoch, filenames, tag, imageidx):
        plot_images = []
        for idx in range(len(filenames)):
            grid, _ = GridFactory.get(data=torch.unsqueeze(preds[idx], dim=0), variables=[],
                                      coordinate=CoordinateSystem.CELL_SPLIT, args=self.args,
                                      input_coords=self.dataset.coords, only_train_vars=True,
                                      anchors=self.dataset.anchors)

            img, ok = plot_cell_class(grid=grid, image=images[idx], name=None, epoch=epoch, tag=tag + "_class",
                                      imageidx=imageidx, max_class=self.dataset.num_classes(), ignore_classes=[0],
                                      threshold=self.args.confidence)
            plot_images.append(convert_to_torch_image(img))

        name = self.args.paths.generate_debug_image_file_path("class", imageidx)
        Log.grid(name=name, images=plot_images, epoch=epoch, imageidx=imageidx, tag=tag + "_class")

    def is_time_for_image(self, epoch):
        return epoch % 5 == 0
