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
from copy import deepcopy

import numpy as np

from yolino.utils.enums import Variables, LINE, AnchorVariables
from yolino.utils.logger import Log


class VariableStructure(dict):

    def __init__(self, line_representation_enum: LINE, num_classes=0, num_conf=1, num_angles=0, one_hot=True,
                 vars_to_train=None):

        if num_classes == 0 and vars_to_train is not None and Variables.CLASS in vars_to_train:
            Log.error("The dataset does not contain classes, but you selected to train classes.")
            exit(2)

        from yolino.model.line_representation import LineRepresentation
        self.line_representation = LineRepresentation.get(line_representation_enum)
        dict_data = {
            Variables.GEOMETRY: self.line_representation.num_params,  # Regression
            Variables.CLASS: num_classes,  # Classification
            Variables.INSTANCE: 0,  # Classification !?
            Variables.FOLLOWER: 0,  # Classification
            Variables.POSITION_IN_LINE: 0,  # Classification
            Variables.CONF: num_conf,  # Regression
            Variables.SAMPLE_ANGLE: num_angles,
        }
        super().__init__(dict_data)

        if vars_to_train is None:
            self.vars_to_train = []
        elif type(vars_to_train) == Variables:
            self.vars_to_train = [vars_to_train]
        else:
            self.vars_to_train = vars_to_train

        self.one_hot = one_hot

    def clone(self, line_representation: LINE):
        cp = deepcopy(self)
        cp: VariableStructure

        from yolino.model.line_representation import LineRepresentation
        linerep_object = LineRepresentation.get(line_representation)
        cp[Variables.GEOMETRY] = linerep_object.num_params
        cp.line_representation = linerep_object
        return cp

    def not_train_vars(self):
        return np.setdiff1d(list(self.keys()), self.vars_to_train)

    def vars(self):
        return [k for k, v in self.items() if v > 0]

    def train_vars(self):
        return self.vars_to_train

    def num_vars_to_train(self, one_hot=None):
        return self.get_length(variables=self.vars_to_train, one_hot=one_hot)

    def get_position_of_training_vars(self, one_hot=None):
        return self.get_position_of(self.vars_to_train, one_hot=one_hot)

    def get_position_within_prediction_except(self, exclude_variables, one_hot=None):
        all_variables = [k for k in Variables]
        remaining = np.setdiff1d(all_variables, exclude_variables)
        return self.get_position_within_prediction(remaining, one_hot=one_hot)

    def get_position_within_prediction(self, variables, one_hot=None):
        ranges = []
        position = 0

        if one_hot is None:
            one_hot = self.one_hot

        if type(variables) == Variables:
            variables = [variables]
        elif type(variables) != list and type(variables) != np.ndarray:
            raise ValueError("Unknown type for variables %s" % type(variables))

        if len(variables) == 1 and variables[0] not in self.train_vars():
            return []

        # Log.info("Pos=%d" % position)
        for k in self.train_vars():
            v = self[k]
            if v == 0:
                continue

            if k == Variables.CLASS and not one_hot:
                end = position + 1
            else:
                end = position + v

            if variables is None or k in variables:
                ranges.append(range(position, end))

            position = end

        if len(ranges) == 0:
            return ranges

        return np.concatenate([[i for i in r] for r in ranges])

    def get_position_of(self, variables, anchor_vars=None, one_hot=None):
        ranges = []
        position = 0

        if one_hot is None:
            one_hot = self.one_hot

        if type(variables) == Variables:
            variables = [variables]
        elif type(variables) != list and type(variables) != np.ndarray:
            from yolino.utils.logger import Log
            Log.error(self)
            raise ValueError("Unknown type for variables %s" % type(variables))

        for k, v in self.items():
            if v == 0:
                continue

            if k == Variables.CLASS and not one_hot:
                end = position + 1
            else:
                end = position + v

            if variables is None or k in variables:
                if k == Variables.GEOMETRY and anchor_vars is not None:
                    if AnchorVariables.POINTS in anchor_vars:
                        ranges.append(range(position, end))
                    else:
                        if AnchorVariables.MIDPOINT in anchor_vars:
                            ranges.append(range(position, position + 2))
                        if AnchorVariables.DIRECTION in anchor_vars:
                            ranges.append(range(position + 2, end))
                else:
                    ranges.append(range(position, end))

            position = end

        if ranges == []:
            return []
        else:
            return np.concatenate([[i for i in r] for r in ranges])

    def get_position_of_except(self, exclude_variables, one_hot=None):
        all_variables = [k for k in Variables]
        remaining = np.setdiff1d(all_variables, exclude_variables)
        return self.get_position_of(remaining, one_hot=one_hot)

    def get_length(self, variables=None, one_hot=None):
        position = 0

        if type(variables) == Variables:
            variables = [variables]

        if one_hot is None:
            one_hot = self.one_hot

        for k, v in self.items():
            if variables and k not in variables:
                continue

            if k == Variables.CLASS and not one_hot:
                position = position + 1
            else:
                position = position + v

        return position
