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

import torch
import torch.nn as nn

from yolino.eval.distances import linesegment_euclidean_distance
from yolino.eval.matcher_cell import CellMatcher
from yolino.model.activations import get_activations
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import LOSS, Variables, LossWeighting, AnchorDistribution, ImageIdx
from yolino.utils.logger import Log


class AbstractLoss:
    def __init__(self, coords: VariableStructure, gpu, cuda, one_hot, variable, function, conf_match_weight, reduction,
                 activation_is_exp, loss_weight_strategy):
        self.loss_weight_strategy = loss_weight_strategy
        self.coords = coords
        self.variable = variable
        self.gpu = gpu
        self.cuda = cuda
        self.one_hot = one_hot
        self.function = function
        if self.gpu:
            self.function.cuda()
        self.conf_match_weight = conf_match_weight
        self.reduction = reduction
        self.activation_is_exp = activation_is_exp

    # removes nan
    # Creates a torch view on the relevant variables only
    # returns a tensor of shape (?, num_variables)
    def __prepare_data__(self, preds, grid_tensor):
        # preds = preds.view(-1, preds.shape[-1])
        # grid_tensor = grid_tensor.view(-1, grid_tensor.shape[-1])

        breaks = self.__get_breaks_in_coords__()
        _, labels_for_loss, _ = torch.tensor_split(grid_tensor, torch.tensor(breaks), dim=1)
        if not self.one_hot:
            labels_for_loss = torch.argmax(labels_for_loss, dim=1)

        breaks = self.__get_breaks_in_coords__(only_training=True)
        _, preds_for_loss, _ = torch.tensor_split(preds, torch.tensor(breaks), dim=1)

        return preds_for_loss, labels_for_loss

    # TODO test
    def __get_breaks_in_coords__(self, only_training=False):
        breaks = []
        position = 0
        included_idx = -1
        for idx, var in enumerate(self.coords):
            if self.coords[var] == 0 or (only_training and not var in self.coords.train_vars()):
                continue

            if var == self.variable:
                included_idx = idx

            position += self.coords[var]
            if included_idx == -1:
                if len(breaks) == 0:
                    breaks.append(position)
                else:
                    breaks[0] = position  # collect all excluded up to the included
            elif included_idx == idx:
                if len(breaks) == 0:
                    breaks.append(0)
                breaks.append(position)
                break
            elif included_idx + 1 == idx:
                breaks.append(position)
            else:
                if idx < len(self.coords) - 1:
                    breaks[-1] = position  # collect all excluded after the included
        return breaks

    @staticmethod
    def __get_flat__(preds, grid_tensor):
        grid_tensor = grid_tensor.flatten()
        preds = preds.flatten()
        return preds, grid_tensor

    def split_nans(self, preds, grid_tensor):
        """

        Args:
            preds (torch.tensor): with shape [batch, cells, preds, vars]

        """
        invalid_flags = torch.any(grid_tensor[:, self.coords.get_position_of_training_vars()].isnan(), dim=-1)
        valid_flags = ~invalid_flags

        return preds[valid_flags], grid_tensor[valid_flags], preds[invalid_flags], grid_tensor[invalid_flags]

    def replace_nans(self, grid_tensor, variable):
        indices = torch.any(grid_tensor[:, self.coords.get_position_of(variable)].isnan(), dim=1)
        grid_tensor[indices, self.coords.get_position_of(variable)] = 0
        return grid_tensor

    def __call__(self, preds, grid_tensor):
        if self.cuda not in str(grid_tensor.device):
            Log.debug("Moved labels from %s to %s" % (grid_tensor.device, self.cuda))
            grid_tensor.to(self.cuda)

        if self.cuda not in str(preds.device):
            Log.debug("Moved prediction from %s to %s" % (preds.device, self.cuda))
            preds.to(self.cuda)

        # for the confidence we also punish unmatched predictions which are assigned a nan GT line and so we
        # want to split them into matched and unmatched
        preds, grid_tensor, preds_unmatched, grid_tensor_unmatched = self.split_nans(preds, grid_tensor)

        if self.variable == Variables.CONF:
            # for the confidence all nans in the GT should have a confidence of 0
            grid_tensor_unmatched = self.replace_nans(grid_tensor_unmatched, variable=self.variable)
            preds_unmatched, grid_tensor_unmatched = self.__prepare_data__(preds_unmatched, grid_tensor_unmatched)

        preds, grid_tensor = self.__prepare_data__(preds, grid_tensor)

        if not self.variable == Variables.CONF and (len(preds) == 0 or len(grid_tensor) == 0):
            msg = "No valid data for loss %s in variable %s" % (type(self), self.variable)
            Log.info(msg)
            raise ValueError(msg)

        return preds, grid_tensor, preds_unmatched, grid_tensor_unmatched

    def __str__(self):
        return "%s for %s %s" \
               % (str(self.__class__).replace("<", "").replace(">", "").replace("'", "").replace(
            "class yolino.model.loss.", ""),
                  self.variable, self.coords.get_position_of(self.variable))

    def __apply_function__(self, preds, grid_tensor, p_unmatched, gt_unmatched, tag="none", epoch=None):
        if len(preds) > 0 and len(grid_tensor) > 0:
            Log.debug("%s on e.g. %s vs gt=%s" % (str(self), preds[0], grid_tensor[0]))
        elif self.variable == Variables.CONF and len(p_unmatched) > 0 and len(gt_unmatched) > 0:
            Log.debug("Unmatched %s on e.g. %s vs gt=%s" % (str(self), p_unmatched[0], gt_unmatched[0]))

        matched_loss = torch.tensor(0, dtype=grid_tensor.dtype, device=grid_tensor.device)
        unmatched_loss = torch.tensor(0, dtype=grid_tensor.dtype, device=grid_tensor.device)


        # handle only not nan labels
        if not torch.all(grid_tensor.isnan()):
            matched_loss = self.function(preds, grid_tensor)

        normalizing_matched = len(preds) if self.reduction == "sum" else 1
        normalizing_unmatched = len(p_unmatched) if self.reduction == "sum" else 1

        mean_matched_loss = matched_loss / normalizing_matched
        mean_unmatched_loss = 0

        if self.variable == Variables.CONF:
            if not torch.all(gt_unmatched.isnan()):
                unmatched_loss = self.function(p_unmatched, gt_unmatched)

            mean_unmatched_loss = unmatched_loss / normalizing_unmatched
            Log.scalars(tag=tag, dict={"loss_conf_batch/match/mean": mean_matched_loss,
                                       "loss_conf_batch/unmatch/mean": mean_unmatched_loss},
                        epoch=epoch)

            add_weights, weight_factors = get_actual_weight(epoch, "conf/match",
                                                            weight_strategy=self.loss_weight_strategy,
                                                            weight=self.conf_match_weight[0],
                                                            activation_is_exponential=self.activation_is_exp)
            weighted_loss = weight_factors * matched_loss + add_weights
            add_weights, weight_factors = get_actual_weight(epoch, "conf/nomatch",
                                                            weight_strategy=self.loss_weight_strategy,
                                                            weight=self.conf_match_weight[1],
                                                            activation_is_exponential=self.activation_is_exp)
            weighted_loss += weight_factors * unmatched_loss + add_weights
            mean_loss = (mean_matched_loss + mean_unmatched_loss) * 0.5
        else:
            weighted_loss = mean_matched_loss
            mean_loss = mean_matched_loss

        return weighted_loss, mean_loss


class CrossEntropyCellLoss(AbstractLoss):

    def __init__(self, gpu, coords: VariableStructure, cuda, variable: Variables, conf_match_weight, reduction,
                 activation_is_exp, loss_weight_strategy):
        # weight = torch.tensor([4/100, 24/100, 24/100, 24/100, 24/100])
        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=False, variable=variable,
                         function=nn.CrossEntropyLoss(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)  # output shape (?, 10)
        Log.debug("%s on e.g. %s vs gt=%s; Shapes pred=%s vs gt=%s" % (str(self), preds[0], grid_tensor[0], preds.shape,
                                                                       grid_tensor.shape))

        try:
            return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)
        except Exception as ex:
            Log.error("Calculate loss with shapes %s and %s" % (preds.shape, grid_tensor.shape))
            raise ex


class BinaryCrossEntropyCellLoss(AbstractLoss):
    def __init__(self, gpu, coords: VariableStructure, cuda, variable: Variables, conf_match_weight, reduction,
                 activation_is_exp, loss_weight_strategy):
        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=nn.BCELoss(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)  # shape (?, 10)

        if preds.shape[1] > 1 and self.variable == Variables.CLASS:
            raise NotImplementedError(
                "Calculate binary cross entropy for %d classes! Full shape %s" % (preds.shape[1], preds.shape))

        Log.debug("%s on e.g. %s vs gt=%s" % (str(self), preds[0], grid_tensor[0]))
        preds, grid_tensor = self.__get_flat__(preds, grid_tensor)

        return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)


class MeanSquaredErrorLoss(AbstractLoss):
    def __init__(self, reduction, coords: VariableStructure, gpu, cuda, variable: Variables, batch_size: int,
                 conf_match_weight, activation_is_exp, loss_weight_strategy) -> None:
        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=torch.nn.MSELoss(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)
        self.batch_size = batch_size

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)
        return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)

    def __str__(self):
        string = super().__str__()
        return self.function.reduction.capitalize() + string


class NormLoss(AbstractLoss):
    def __init__(self, coords: VariableStructure, gpu, cuda, variable: Variables, conf_match_weight, activation_is_exp,
                 loss_weight_strategy, reduction="mean") -> None:
        if variable != Variables.GEOMETRY:
            raise NotImplementedError("We only implemented the norm loss for geometry.")

        class norm_loss_fct:
            def __init__(self, reduction):
                self.reduction = reduction

            def __call__(self, preds, gts):
                vls = [linesegment_euclidean_distance(gt=g, pred=p.unsqueeze(dim=0), coords=coords, use_conf=False)
                       for p, g in zip(preds, gts)]
                if self.reduction == "mean":
                    return torch.cat(vls).mean()
                else:
                    return torch.cat(vls).sum()

            def cuda(self):
                pass

        super().__init__(coords=coords, gpu=gpu, cuda=cuda, one_hot=True, variable=variable,
                         function=norm_loss_fct(reduction=reduction), conf_match_weight=conf_match_weight,
                         reduction=reduction, activation_is_exp=activation_is_exp,
                         loss_weight_strategy=loss_weight_strategy)

    def __call__(self, preds, grid_tensor, tag="none", epoch=None):
        preds, grid_tensor, p_unmatched, gt_unmatched = super().__call__(preds, grid_tensor)
        return self.__apply_function__(preds, grid_tensor, p_unmatched, gt_unmatched, tag=tag, epoch=epoch)


# TODO: sum of losses is proportional to batch_size!
def get_loss(losses, args, coords: VariableStructure, weights: list, anchors, conf_weights):
    loss_weight_strategy = args.loss_weight_strategy

    functions = []
    assert (len(coords.train_vars()) == len(losses))
    for i, loss in enumerate(losses):
        activation_is_exp = get_activations(args.activations, coords=coords,
                                            linerep=coords.line_representation.enum).activations[i].is_exp()
        variable = coords.train_vars()[i]
        if loss == LOSS.CROSS_ENTROPY_MEAN:
            functions.append(CrossEntropyCellLoss(gpu=args.gpu, coords=coords, cuda=args.cuda, variable=variable,
                                                  conf_match_weight=conf_weights, reduction="mean",
                                                  activation_is_exp=activation_is_exp,
                                                  loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.CROSS_ENTROPY_SUM:
            functions.append(CrossEntropyCellLoss(gpu=args.gpu, coords=coords, cuda=args.cuda, variable=variable,
                                                  conf_match_weight=conf_weights, reduction="sum",
                                                  activation_is_exp=activation_is_exp,
                                                  loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.BINARY_CROSS_ENTROPY_MEAN:
            functions.append(BinaryCrossEntropyCellLoss(gpu=args.gpu, coords=coords, cuda=args.cuda, variable=variable,
                                                        conf_match_weight=conf_weights, reduction="mean",
                                                        activation_is_exp=activation_is_exp,
                                                        loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.BINARY_CROSS_ENTROPY_SUM:
            functions.append(BinaryCrossEntropyCellLoss(gpu=args.gpu, coords=coords, cuda=args.cuda, variable=variable,
                                                        conf_match_weight=conf_weights, reduction="sum",
                                                        activation_is_exp=activation_is_exp,
                                                        loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.MSE_SUM:
            functions.append(MeanSquaredErrorLoss(reduction="sum", coords=coords, gpu=args.gpu, cuda=args.cuda,
                                                  variable=variable, batch_size=args.batch_size,
                                                  conf_match_weight=conf_weights,
                                                  activation_is_exp=activation_is_exp,
                                                  loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.MSE_MEAN:
            functions.append(MeanSquaredErrorLoss(reduction="mean", coords=coords, gpu=args.gpu, cuda=args.cuda,
                                                  variable=variable, batch_size=args.batch_size,
                                                  conf_match_weight=conf_weights,
                                                  activation_is_exp=activation_is_exp,
                                                  loss_weight_strategy=loss_weight_strategy))
        elif loss == LOSS.NORM_MEAN:
            functions.append(
                NormLoss(coords=coords, gpu=args.gpu, cuda=args.cuda, variable=variable, conf_match_weight=conf_weights,
                         activation_is_exp=activation_is_exp, loss_weight_strategy=loss_weight_strategy,
                         reduction="mean"))
        elif loss == LOSS.NORM_SUM:
            functions.append(
                NormLoss(coords=coords, gpu=args.gpu, cuda=args.cuda, variable=variable, conf_match_weight=conf_weights,
                         activation_is_exp=activation_is_exp, loss_weight_strategy=loss_weight_strategy,
                         reduction="sum"))
        else:
            raise NotImplementedError("Unknown loss type %s" % loss)
    composed_loss = LossComposition(losses=functions, args=args, coords=coords, weights=weights, anchors=anchors)
    return composed_loss


def get_actual_weight(epoch, variable_str, weight_strategy, weight, activation_is_exponential):
    if weight_strategy == LossWeighting.FIXED or weight_strategy == LossWeighting.FIXED_NORM:
        weight_factor = weight
        add_weight = 0  # add no regularization (=0)
    elif weight_strategy == LossWeighting.LEARN:
        if torch.any(weight > 20) or torch.any(weight < 0.05):
            Log.warning(f"Clamp necessary: {weight}")
            weight = torch.clamp(weight, min=0.05, max=20)

        # this is the thing we actually learn: sigma
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "pure_learn_weight"): weight})
        # this is the actual variance sigma ** 2
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "var"): math.pow(weight, 2)})

        if activation_is_exponential:
            # else we have 1 / s^2
            weight_factor = torch.pow(weight, -2)
        else:
            # when using softmax/sigmoid we have 1 / (2s^2)
            weight_factor = 1 / (2 * torch.pow(weight, 2))

        # the regularization is on log(sigma)
        add_weight = torch.log(weight)
    elif weight_strategy == LossWeighting.LEARN_LOG or weight_strategy == LossWeighting.LEARN_LOG_NORM:
        if torch.any(weight > math.log(20)) or torch.any(weight < math.log(0.05)):
            Log.warning(f"Clamp necessary: {weight}")
            weight = torch.clamp(weight, min=math.log(0.05), max=math.log(20))

        # this is the thing we acutally learn: log(sigma ** 2)
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "pure_learn_weight"): weight})
        # this is the actual varianz sigma ** 2 = e^(log(sigma ** 2))
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "var"): math.exp(weight)})

        if activation_is_exponential:
            # else we have e^-s
            weight_factor = torch.exp(-1. * weight)
        else:
            # when using softmax/sigmoid we have 1 / ( 2 * e^s)
            weight_factor = 0.5 * torch.exp(-1. * weight)

        # the regularization is on log(sigma), we train log(sigma ** 2)
        add_weight = 0.5 * weight
    else:
        raise NotImplementedError("We do not know %s" % weight_strategy)

    if weight_strategy == LossWeighting.LEARN \
            or weight_strategy == LossWeighting.LEARN_LOG \
            or weight_strategy == LossWeighting.LEARN_LOG_NORM:
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "actual_weight"): weight_factor})
    elif epoch == 0:
        Log.scalars(tag="", epoch=epoch, dict={os.path.join("loss_" + variable_str, "actual_weight"): weight_factor})

    return add_weight, weight_factor


# TODO: sum of losses is proportional to batch_size!
class LossComposition:
    def __init__(self, losses, args, coords: VariableStructure, weights: list, anchors):
        self.add_weights = []
        self.losses = losses

        self.anchors = anchors
        self.args = args
        self.coords = coords
        # self.weights = torch.nn.functional.softmax(weights, dim=0)
        self.weights = weights
        self.matcher = CellMatcher(coords, args)
        Log.debug("Weights=%s" % self.weights)
        if len(self.losses) != len(self.weights):
            raise ValueError("Please specify the same number of loss terms as weights, we got %s loss terms, "
                             "but %s weights." % (self.losses, self.weights))

    def __call__(self, preds, grid_tensor, filenames, epoch, tag="dummy_loss"):
        """

        Args:
            preds (torch.tensor):
                with shape [batch, cells, preds, vars]
            grid_tensor (torch.tensor):
                with shape [batch, cells, preds, vars]

        Returns:
            loss (list):
                with len num_vars_to_train
            weighted_losses (torch.tensor)
        """
        if torch.any(torch.isnan(preds)):
            raise ValueError("Prediction can not contain nans!")

        num_batches, num_cells, num_predictors, _ = preds.shape
        # Log.debug("Within the loss")
        weighted_losses = torch.zeros((1), device=self.args.cuda, dtype=torch.float32)
        losses = []
        mean_losses = []

        from datetime import datetime
        from datetime import timedelta
        start = datetime.now()
        if self.args.anchors == AnchorDistribution.NONE:
            preds, reduced_grid_tensor = self.matcher.sort_cells_by_geometric_match(preds=preds,
                                                                                    grid_tensor=grid_tensor,
                                                                                    epoch=epoch, tag=tag,
                                                                                    filenames=filenames)
        else:
            self.matcher._debug_full_match_plot_(epoch=epoch, preds=preds, grid_tensor=grid_tensor, filenames=filenames,
                                                 tag="loss", anchors=self.anchors, idx=ImageIdx.LOSS)
            reduced_grid_tensor = grid_tensor.reshape(-1, self.coords.get_length())
            preds = preds.reshape(-1, self.coords.num_vars_to_train())

        end = datetime.now()
        seconds = ((end - start) / timedelta(milliseconds=1))
        Log.debug("Matching done in %dms" % (seconds))

        nice_idx = (6 * 27 + 1) % preds.shape[0]
        Log.debug("Loss on e.g.\n%s\n%s" % (preds[nice_idx], reduced_grid_tensor[nice_idx]))
        for i, t in enumerate(self.losses):
            if torch.all(reduced_grid_tensor[:, self.coords.get_position_of(t.variable)].isnan()):
                Log.info("We skip loss calc for %s as there are no labels" % str(t.variable))
                losses.append(0)
                mean_losses.append(0)
                weighted_losses = 0
            elif torch.all(reduced_grid_tensor[:, self.coords.get_position_of(Variables.GEOMETRY)].isnan()) \
                    and t.variable != Variables.CONF:
                Log.warning("We skip loss calc for %s as there is no geometry and thus no matching"
                            % str(t.variable))
                losses.append(0)
                mean_losses.append(0)
                weighted_losses = 0
            else:
                # try:
                t: AbstractLoss
                loss_val, mean_loss_val = t(preds, reduced_grid_tensor, tag=tag, epoch=epoch)
                loss_val: torch.tensor

                variable_strings = ("conf" if self.losses[i].variable == Variables.CONF
                                    else str(self.losses[i].variable))
                is_exp = get_activations(self.args.activations, self.coords, self.args.linerep).activations[i].is_exp()

                add_weights, weight_factors = get_actual_weight(epoch, variable_strings,
                                                                weight_strategy=self.args.loss_weight_strategy,
                                                                weight=self.weights[i],
                                                                activation_is_exponential=is_exp)

                mean_losses.append(mean_loss_val.detach().cpu())  # / (num_predictors * num_cells))
                losses.append(loss_val.detach().cpu())  # / (num_predictors * num_cells))
                l = loss_val * weight_factors + add_weights
                Log.scalars(tag=tag, epoch=epoch,
                            dict={os.path.join("loss_" + variable_strings + "_batch", "sum",
                                               "weighted"): l.item() / len(preds)})
                Log.scalars(tag=tag, epoch=epoch, dict={"loss_batch/sum/weighted": weighted_losses})

                weighted_losses += l
            Log.debug("After %s we get [sum] loss = %s (total=%s)" % (str(t), losses[i], weighted_losses))
            Log.scalars(tag=tag, epoch=epoch, dict={"loss_batch/sum/weighted": weighted_losses})

        return losses, weighted_losses, mean_losses

    def __repr__(self):
        string = "Loss Composition <"
        for t in self.losses:
            string += str(t) + ", "
        string += ">"
        return string

    def is_exp_activation(self, index):
        return get_activations(self.args.activations, self.coords, self.args.linerep).activations[index].is_exp()
