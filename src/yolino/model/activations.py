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
from abc import abstractmethod

import torch

from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import ACTIVATION, LINE, Variables
from yolino.utils.logger import Log


class AbstractActivation:
    def __init__(self, variable, coords: VariableStructure, function):
        self.variable = variable
        self.coords = coords
        self.function = function

    def __call__(self, logits):
        """

        Args:
            logits (torch.tensor):
                with shape (batch, cells, preds, vars)
        """
        position = self.coords.get_position_within_prediction(variables=self.variable)
        logits[:, :, :, position] = self.function(logits[:, :, :, position])
        return logits

    def __str__(self):
        string = ""
        string += str(self.__class__) + " ["
        string += str(self.variable) + "]"
        return string

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def is_exp(self):
        pass


class Tanh(AbstractActivation):
    def is_exp(self):
        return True

    def __init__(self, variable, coords: VariableStructure):
        super().__init__(variable=variable, coords=coords, function=torch.nn.Tanh())


class Softmax(AbstractActivation):
    def is_exp(self):
        return True

    def __init__(self, dim, variable, coords: VariableStructure):
        super().__init__(variable=variable, coords=coords, function=torch.nn.Softmax(dim=dim))


class Sigmoid(AbstractActivation):
    def is_exp(self):
        return True

    def __init__(self, variable, coords: VariableStructure, offset=False):
        if offset:
            super().__init__(variable=variable, coords=coords,
                             function=lambda logits: torch.nn.Sigmoid()(logits) * 2 - 1)
        else:
            super().__init__(variable=variable, coords=coords, function=torch.nn.Sigmoid())


class Linear(AbstractActivation):
    def is_exp(self):
        return False

    def __init__(self, variable, coords: VariableStructure):
        super().__init__(variable=variable, coords=coords, function=lambda logits: logits)


class MidLenDirActivation(AbstractActivation):
    def __init__(self, fct, variable, coords: VariableStructure):
        super().__init__(variable=variable, coords=coords, function=fct)


class ActivationComposition:
    def __init__(self, activations):
        self.activations = activations

    def __call__(self, logits):
        """

        Args:
            logits (torch.tensor):
                with shape [batch, cells, preds, vars]

        Returns:
            torch.tensor:
                with shape [batch, cells, preds, vars]
        """
        nice_idx = (0, (6 * 27 + 1) % logits.shape[1], 0)
        Log.debug("Activation on e.g.\n%s" % logits[nice_idx])
        for t in self.activations:
            logits = t(logits)
            Log.debug(
                "After %s for %s %s we get\n%s" % (str(t.__class__).replace("class 'yolino.model.activations.", ""),
                                                   t.variable, str(t.coords.get_position_of(t.variable)),
                                                   logits[nice_idx]))

        return logits

    def __repr__(self):
        string = "Activation Composition <"
        for t in self.activations:
            string += str(t) + ", "
        string += ">"
        return string


def get_activations(activations, coords: VariableStructure, linerep: LINE, offset=False):
    if not len(activations) == len(coords.train_vars()):
        raise ValueError("%d activations should be equal to %d training variables.\n%s\n%s" % (len(activations),
                                                                                               len(coords.train_vars()),
                                                                                               activations,
                                                                                               coords.train_vars()))
    composed = []
    for variable, activation in zip(coords.train_vars(), activations):
        fct = None
        if variable == Variables.GEOMETRY:
            if linerep == LINE.EULER:
                fct = Tanh(variable=Variables.GEOMETRY, coords=coords)
            elif linerep == LINE.MID_LEN_DIR:
                if activation == ACTIVATION.SIGMOID:
                    fct = MidLenDirActivation(fct=mid_len_dir_activation_offset if offset else mid_len_dir_activation,
                                              variable=variable, coords=coords)
                elif activation == ACTIVATION.LINEAR:
                    fct = Linear(variable=variable, coords=coords)
                else:
                    raise NotImplementedError("Mid-length-direction lines can not be trained with activation "
                                              "%s" % activation)
            elif linerep == LINE.MID_DIR:
                if activation == ACTIVATION.SIGMOID:  # else its just the raw sigmoid
                    fct = MidLenDirActivation(fct=mid_dir_activation_offset if offset else mid_dir_activation,
                                              variable=variable, coords=coords)
                elif activation == ACTIVATION.LINEAR:
                    fct = Linear(variable=variable, coords=coords)
                else:
                    raise NotImplementedError("Mid-length-direction lines can not be trained with activation "
                                              "%s" % activation)
            elif linerep != LINE.POINTS:
                raise NotImplementedError("Unknown linerep %s" % linerep)
        if fct is None:
            if activation == ACTIVATION.LINEAR:
                fct = Linear(variable=variable, coords=coords)
            elif activation == ACTIVATION.SIGMOID:
                fct = Sigmoid(variable=variable, coords=coords, offset=offset)
            elif activation == ACTIVATION.SOFTMAX:
                fct = Softmax(dim=3, variable=variable, coords=coords)
            else:
                raise NotImplementedError(activation)

        composed.append(fct)

    return ActivationComposition(composed)


def mid_len_dir_activation(v: torch.tensor):
    return torch.cat([torch.nn.Sigmoid()(v[:, :, :, [0, 1]]),  # interval [0,1]
                      torch.nn.Sigmoid()(v[:, :, :, [2]]) * math.sqrt(2),
                      torch.nn.Sigmoid()(v[:, :, :, [3, 4]]) * 2 - 1  # interval [-1,1]
                      ], axis=3)


def mid_len_dir_activation_offset(v: torch.tensor):
    return torch.cat([torch.nn.Sigmoid()(v[:, :, :, [0, 1]]) * 2 - 1,  # interval [-1,1]
                      torch.nn.Sigmoid()(v[:, :, :, [2]]) * math.sqrt(2),
                      torch.nn.Sigmoid()(v[:, :, :, [3, 4]]) * 4 - 2  # interval [-2,2]
                      ], axis=3)


def mid_dir_activation(v: torch.tensor):
    return torch.cat([torch.nn.Sigmoid()(v[:, :, :, [0, 1]]),  # interval [0,1]
                      torch.nn.Sigmoid()(v[:, :, :, [2, 3]]) * 2 - 1  # interval [-1,1]
                      ], axis=3)


def mid_dir_activation_offset(v: torch.tensor):
    return torch.cat([torch.nn.Sigmoid()(v[:, :, :, [0, 1]]) * 2 - 1,  # interval [-1,1]
                      torch.nn.Sigmoid()(v[:, :, :, [2, 3]]) * 4 - 2  # interval [-2,2]
                      ], axis=3)
