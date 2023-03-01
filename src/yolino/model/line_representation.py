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

from yolino.eval.distances import to_mld_cell_space, to_md_cell_space
from yolino.utils.enums import LINE
from yolino.utils.geometry import intersection_square, reformat2shapely
from yolino.utils.logger import Log


class LineRepresentation:
    def __init__(self, num_params, enum):
        self.num_params = num_params
        self.enum = enum

    @classmethod
    def get_dummy(cls, torch=True):
        pass

    @classmethod
    def from_cart(cls, start_xy, end_xy, **kwargs):
        return np.concatenate([start_xy, end_xy]).flatten()

    @classmethod
    def to_cart(cls, values):
        return values

    @classmethod
    def extrapolate_cart(cls, start, end):
        mld = MidLenDirLines.from_cart(start, end)
        pseudo_start = mld[0:2] - mld[3:5] / mld[2]
        pseudo_end = mld[0:2] + mld[3:5] / mld[2]

        geom_intersection, geom_box = intersection_square(top_left=[0, 0], width=1,
                                                          geom_line=reformat2shapely([pseudo_start, pseudo_end]))
        start = np.asarray([geom_intersection.xy[0][0], geom_intersection.xy[1][0]]).round(8)
        end = np.asarray([geom_intersection.xy[0][1], geom_intersection.xy[1][1]]).round(8)

        if np.any(start < 0) or np.any(start > 1):
            raise ValueError("Start is out of the cell")

        if np.any(end < 0) or np.any(end > 1):
            raise ValueError("Start is out of the cell")

        return start, end

    @classmethod
    def __get_enum__(cls, name_string):
        for v in LINE.__iter__():
            if v.value == name_string:
                return v, cls.get(v)
        raise ValueError("%s is not a line representation. Choose from: %s"
                         % (name_string, [v.value for v in LINE.__iter__()]))

    @classmethod
    def get(cls, enum):
        return class_lookup[enum]

    @classmethod
    def get_num_params(cls, name_string):
        return getattr(cls.__get_enum__(name_string)[1], "num_params")


class EulerLines(LineRepresentation):
    @classmethod
    def get_dummy(cls, use_torch=True):
        cart = PointsLines().get_dummy(use_torch=use_torch)
        return cls.from_cart(cart[0:2], cart[2:4])

    def __init__(self):
        super().__init__(4, LINE.EULER)

    @classmethod
    def from_cart(cls, start_xy, end_xy, extrapolate=True):

        values = []
        for x, y in [start_xy, end_xy]:
            if not extrapolate and not x in [0, 1] and not y in [0, 1]:
                return [math.nan] * 4
            y = y - 0.5
            x = x - 0.5
            angle = math.atan2(y, x)
            values.append(math.cos(angle))
            values.append(math.sin(angle))
        return values

    @classmethod
    def to_cart(cls, values):
        start_cos_sin = values[0:2]
        end_cos_sin = values[2:4]

        cart_values = []
        for c, s in [start_cos_sin, end_cos_sin]:
            angle = torch.atan2(c, s)
            if angle < 0:
                angle += 2 * math.pi
            # right border
            if angle <= math.pi / 4 or angle > 7 * (math.pi / 4):
                distance = math.tan(angle) * 0.125
                cart_values.append(0.375 - distance)
            # top border
            elif angle <= 3 * math.pi / 4:
                distance = math.tan(angle - math.pi / 2) * 0.125
                cart_values.append(0.125 - distance)
            # left border
            elif angle <= 5 * math.pi / 4:
                distance = math.tan(angle - math.pi) * 0.125
                cart_values.append(0.875 - distance)
            # bottom border
            elif angle <= 7 * math.pi / 4:
                distance = math.tan(angle - (3 * math.pi / 2)) * 0.125
                cart_values.append(0.625 - distance)

        return OneDLines.to_cart(cart_values)


class OneDLines(LineRepresentation):
    @classmethod
    def get_dummy(cls, use_torch=True):
        cart = PointsLines().get_dummy(use_torch=torch)
        return cls.from_cart(cart[0:2], cart[2:4])

    def __init__(self):
        super().__init__(2, LINE.ONE_D)

    @classmethod
    def from_cart(cls, start, end, extrapolate=True):
        values = []
        if extrapolate:
            start, end = cls.extrapolate_cart(start, end)

        for x, y in [start, end]:
            if x == 0:
                values.append((0 + y) / 4)
            elif x == 1:
                values.append((3 - y) / 4)
            elif y == 0:
                values.append((4 - x) / 4)
            elif y == 1:
                values.append((1 + x) / 4)
            else:
                return [math.nan] * 2
        return values

    @classmethod
    def to_cart(cls, values):
        cart_values = []
        for v in values:
            if v < 0.25:
                x = 0
                y = v * 4
            elif v < 0.5:
                x = (v - 0.25) * 4
                y = 1
            elif v < 0.75:
                x = 1
                y = 1 - (v - 0.5) * 4
            else:
                x = 1 - (v - 0.75) * 4
                y = 0
            cart_values.append(x)
            cart_values.append(y)
        return cart_values


class PointsLines(LineRepresentation):
    @classmethod
    def get_dummy(cls, use_torch=True):
        v = [0., 0, 1, 1]
        if use_torch:
            return torch.tensor(v, dtype=torch.float32)
        else:
            return np.asarray(v, dtype=torch.float32)

    def __init__(self):
        super().__init__(4, LINE.POINTS)


class MidLenDirLines(LineRepresentation):
    @classmethod
    def get_dummy(cls, use_torch=True):
        cart = PointsLines().get_dummy(use_torch=torch)
        return cls.from_cart(cart[0:2], cart[2:4])

    def __init__(self):
        super().__init__(5, LINE.MID_LEN_DIR)

    @classmethod
    def from_cart(cls, start, end, **kwargs):
        return to_mld_cell_space(np.expand_dims(np.concatenate([start, end]), axis=0))[0]

    @classmethod
    def to_cart(cls, values, raise_error=True):
        x, y, l, dx, dy = values[0:5]

        angle = np.arctan2(dy, dx)
        c = np.cos(angle)
        s = np.sin(angle)
        lower_position = torch.tensor([x, y, x - c * l / 2., y - s * l / 2])
        upper_position = torch.tensor([x, y, x + c * l / 2., y + s * l / 2])

        position = torch.cat([lower_position[2:4], upper_position[2:4]]).flatten()

        new_length = torch.linalg.norm(upper_position[2:4] - lower_position[2:4])
        if abs(new_length) - l > 0.1:
            Log.error("The length of the line (=%s) is not the specified length %s" % (new_length, l))
            if raise_error:
                raise AttributeError("The length of the line (=%s) is not the specified length %s. "
                                     "Input was %s" % (new_length, l, values))

        new_angle = torch.atan2(position[3] - position[1], position[2] - position[0])
        if new_angle - angle > 0.01:
            Log.error("The angle of the line (=%s) is not the specified angle %s" % (new_angle, angle))
        return position


class MidDirLines(LineRepresentation):
    @classmethod
    def get_dummy(cls, use_torch=True):
        cart = PointsLines().get_dummy(use_torch=torch)
        return cls.from_cart(cart[0:2], cart[2:4])

    def __init__(self):
        super().__init__(4, LINE.MID_DIR)

    @classmethod
    def from_cart(cls, start, end, **kwargs):
        return to_md_cell_space(np.expand_dims(np.concatenate([start, end]), axis=0))[0]

    @classmethod
    def to_cart(cls, values, raise_error=True):
        x, y, dx, dy = values[0:4]
        angle = math.atan2(dy, dx)
        position = torch.tensor([x - dx / 2., y - dy / 2, x + dx / 2., y + dy / 2])

        new_angle = torch.atan2(position[3] - position[1], position[2] - position[0])
        if new_angle - angle > 0.01 and not (math.pi - 0.01 <= abs(angle) <= math.pi + 0.01):
            msg = "The angle of the line (=%s) is not the specified angle %s" % (new_angle, angle)
            Log.error(msg)
            if raise_error:
                raise AttributeError(msg)

        return position


class_lookup = {
    LINE.EULER: EulerLines(),
    LINE.POINTS: PointsLines(),
    LINE.ONE_D: OneDLines(),
    LINE.MID_LEN_DIR: MidLenDirLines(),
    LINE.MID_DIR: MidDirLines()
}
