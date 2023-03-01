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

import numpy as np
import torch

from yolino.eval.distances import get_midpoints
from yolino.model.line_representation import MidLenDirLines, EulerLines, OneDLines, class_lookup, MidDirLines
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import LINE, Variables as V, Variables
from yolino.utils.logger import Log


class Predictor:
    def __init__(self, values, label=[], confidence=-1, linerep=LINE.POINTS, is_prediction=False):
        if np.sum(label) > 1.1:
            raise ValueError("label values should be one-hot encoding but we have %s!" % str(label))

        if values is not None:
            if linerep == LINE.POINTS:
                self.set_points(values)
            elif linerep == LINE.EULER:
                self.set_euler(values)
            elif linerep == LINE.ONE_D:
                self.set_oned(values)
            elif linerep == LINE.MID_LEN_DIR:
                self.set_mld(values, raise_error=not is_prediction)
            else:
                raise NotImplementedError("Line representation %s" % linerep)

            self.validate()
        else:
            self.start = None
            self.end = None

        label = label.item() if type(label) == torch.Tensor else label
        self.label = np.reshape(label, -1)
        if len(self.label) > 0 and max(self.label) > 1:  # we store one_hot!
            raise ValueError("%s is not one_hot" % self.label)

        self.linerep = linerep

        # for labels this is the polyline ID!
        self.confidence = confidence.item() if type(confidence) == torch.Tensor else confidence

        self.id = -1

    def __str__(self):
        return "%s -> %s [%s] -- %f" % (self.start, self.end, self.label, self.confidence)

    def set_points(self, values, from_anchor=np.asarray([0, 0, 0, 0]), is_offset=False):
        if len(values) == 0:
            return False

        if not len(values) == 4:
            raise ValueError("We expect 4 values for the points line representation. We've got %s" % values)

        if is_offset and from_anchor is not None:
            self.start = np.array(values[0:2] + np.asarray(from_anchor[0:2]), dtype=float)
            self.end = np.array(values[2:4] + np.asarray(from_anchor[2:4]), dtype=float)
        else:
            self.start = np.array(values[0:2], dtype=float)
            self.end = np.array(values[2:4], dtype=float)

        return True

    def set_euler(self, values):
        if len(values) == 0:
            return False

        if len(values) != 4:
            raise ValueError("We expect 4 values for the euler line representation. But we've got %s" % values)

        cart_values = EulerLines.to_cart(values)
        self.start = cart_values[0:2]
        self.end = cart_values[2:4]
        return True

    def set_oned(self, values):
        if len(values) == 0:
            return False
        if len(values) != 2:
            raise ValueError("We expect 2 values for the oned line representation. But we've got %s" % values)
        cart_values = OneDLines.to_cart(values)
        self.start = cart_values[0:2]
        self.end = cart_values[2:4]
        return True

    def set_mld(self, values, from_anchor=np.asarray([0, 0, 0, 0, 0]), is_offset=False, raise_error=False):
        if len(values) == 0:
            return False

        if len(values) != 5:
            raise ValueError("We expect 5 values for the mld line representation. But we've got %s" % values)

        if is_offset and from_anchor is not None:
            values = np.array(values + np.asarray(from_anchor), dtype=float)

        values = MidLenDirLines.to_cart(values, raise_error=raise_error)
        self.start = np.array(values[0:2], dtype=float)
        self.end = np.array(values[2:4], dtype=float)
        return True

    def set_md(self, values, from_anchor=np.asarray([0, 0, 0, 0]), is_offset=False, raise_error=False):
        if len(values) == 0:
            return False

        if len(values) != 4:
            raise ValueError("We expect 5 values for the mld line representation. But we've got %s" % values)

        if is_offset and from_anchor is not None:
            values = np.array(values + np.asarray(from_anchor), dtype=float)

        values = MidDirLines.to_cart(values, raise_error=raise_error)
        self.start = np.array(values[0:2], dtype=float)
        self.end = np.array(values[2:4], dtype=float)
        return True

    def validate(self):
        if type(self.start) == np.ndarray and type(self.end) == np.ndarray:
            return True
        self.start = self.start.numpy()
        self.end = self.end.numpy()

    def angle(self):
        return math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])

    def get_anchor_idx(self, anchors):
        final_indices, offsets = anchors.get_specific_anchors(self)
        return final_indices

    def numpy(self, coords):
        return self.tensor(coords=coords).numpy()

    def tensor(self, coords: VariableStructure, convert_to_lrep: LINE = LINE.POINTS, one_hot=True, set_conf=-1,
               extrapolate=True):
        put_geometry = V.GEOMETRY in coords and coords[V.GEOMETRY] > 0
        put_class = V.CLASS in coords and coords[V.CLASS] > 0
        put_conf = V.CONF in coords and coords[V.CONF] > 0

        if convert_to_lrep is None:
            if coords.line_representation != LINE.POINTS:
                raise ValueError("We expect points coords, but got %s" % coords.line_representation)
        else:
            if coords.line_representation.enum != convert_to_lrep:
                raise ValueError("We expect %s coords, but got %s" % (convert_to_lrep, coords.line_representation.enum))

        array = torch.ones((coords.get_length(one_hot=one_hot)), dtype=torch.float32) * torch.nan

        geom_pos = coords.get_position_of(V.GEOMETRY, one_hot=one_hot)
        if put_geometry:
            if convert_to_lrep == LINE.POINTS or convert_to_lrep is None:
                array[geom_pos] = torch.tensor(np.array([self.start, self.end]).flatten(), dtype=torch.float32)
            else:
                data = class_lookup[convert_to_lrep].from_cart(self.start, self.end, extrapolate=extrapolate)
                array[geom_pos] = torch.tensor(data, dtype=torch.float32)

        class_pos = coords.get_position_of(V.CLASS, one_hot=one_hot)
        if put_class:
            if one_hot:
                array[class_pos] = torch.tensor(self.label, dtype=torch.float32)
            else:
                array[class_pos] = torch.tensor([np.argmax(self.label)], dtype=torch.float32)

        if put_conf:
            if set_conf >= 0:
                array[coords.get_position_of(V.CONF, one_hot=one_hot)] = torch.tensor(set_conf, dtype=torch.float32)
            else:
                array[coords.get_position_of(V.CONF, one_hot=one_hot)] = torch.tensor(self.confidence,
                                                                                      dtype=torch.float32)

        return array.flatten()

    def length(self):
        return np.linalg.norm(self.end - self.start)

    def assert_is_upwards(self):
        if abs(self.angle()) < math.pi / 2.:
            raise ValueError("%s has invalid orientation with %s" % (self, self.angle()))

    @classmethod
    def from_linesegment(cls, line_segment, line_representation, input_coords: VariableStructure, is_prediction=False,
                         is_offset=False, anchor=None):
        if isinstance(line_segment, torch.Tensor):
            line_segment.cpu()
            line_segment = np.array(line_segment)

        if np.isnan(line_segment).any():
            raise ValueError("Nan in line segment")

        if is_prediction:
            length = input_coords.num_vars_to_train()
            coords_in_data = input_coords.train_vars()
            geom_idx = input_coords.get_position_within_prediction(Variables.GEOMETRY)
            label_idx = input_coords.get_position_within_prediction(Variables.CLASS)
            conf_idx = input_coords.get_position_within_prediction(Variables.CONF)
        else:
            length = input_coords.get_length()
            coords_in_data = input_coords.vars()
            geom_idx = input_coords.get_position_of(Variables.GEOMETRY)
            label_idx = input_coords.get_position_of(Variables.CLASS)
            conf_idx = input_coords.get_position_of(Variables.CONF)

        if not np.equal(np.shape(line_segment), length):
            raise ValueError("Geometry failed for %s line %s. "
                             "We expect length of %d with \n%s and training vars\n%s" % (
                                 "prediction" if is_prediction else "gt", line_segment, length,
                                 {k.value: v for k, v in input_coords.items()}, input_coords.train_vars()))

        if Variables.CONF in coords_in_data and input_coords[Variables.CONF] > 0:
            conf = line_segment[conf_idx]
        else:
            conf = 1
        predictor = Predictor(values=None, label=line_segment[label_idx], confidence=conf)

        if line_representation == LINE.POINTS:
            predictor.set_points(line_segment[geom_idx], from_anchor=anchor, is_offset=is_offset)
        elif line_representation == LINE.ONE_D:
            predictor.set_oned(line_segment[geom_idx])
        elif line_representation == LINE.EULER:
            predictor.set_euler(line_segment[geom_idx])
        elif line_representation == LINE.MID_LEN_DIR:
            predictor.set_mld(line_segment[geom_idx], from_anchor=anchor, is_offset=is_offset)
        elif line_representation == LINE.MID_DIR:
            predictor.set_md(line_segment[geom_idx], from_anchor=anchor, is_offset=is_offset)
        else:
            Log.error("Invalid linerep %s!" % line_representation)
            raise ValueError("Invalid linerep %s!" % line_representation)
        return predictor

    def update(self, new_data):
        Log.debug("Update predictor %s with %s" % (self, new_data))
        self.label += new_data.label
        self.confidence += min(1, new_data.confidence)
        self.start = np.mean([self.start, new_data.start], axis=0)
        self.end = np.mean([self.end, new_data.end], axis=0)
        Log.debug("Updated to %s" % self)

    def midpoints(self):
        return get_midpoints(torch.tensor([*self.start, *self.end]))
