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

import numpy as np
import torch

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid import Grid
from yolino.grid.predictor import Predictor
from yolino.model.line_representation import OneDLines, PointsLines, MidDirLines
from yolino.utils.enums import Dataset, LINE
from yolino.utils.test_utils import test_setup


class LineRepTest(unittest.TestCase):
    def prepare(self, line_representation: LINE):
        args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                          additional_vals={"linerep": str(line_representation)}
                          # level=Level.DEBUG
                          )
        coords = DatasetFactory.get_coords(split=args.split, args=args)
        grid = Grid(img_height=args.img_height, args=args)

        p = Predictor([0, 0, 1, 1])
        grid.append(0, 0, predictor=p)
        p = Predictor([1, 0, 0.2, 0.8])
        grid.append(0, 0, predictor=p)
        p = Predictor([0.8, 0.4, 0.9, 0.2])
        grid.append(0, 0, predictor=p)
        return grid, coords

    def test_points(self):
        grid, coords = self.prepare(line_representation=LINE.POINTS)
        t = grid.tensor(convert_to_lrep=LINE.POINTS, coords=coords)

        self.assertEqual(t[0, 0, 0], 0, msg="Got %s" % str(t[0, 0]))
        self.assertEqual(t[0, 0, 1], 0, msg="Got %s" % str(t[0, 0]))
        self.assertEqual(t[0, 0, 2], 1, msg="Got %s" % str(t[0, 0]))
        self.assertEqual(t[0, 0, 3], 1, msg="Got %s" % str(t[0, 0]))

        self.assertEqual(t[0, 1, 0], 1, msg="Got %s" % str(t[0, 1]))
        self.assertEqual(t[0, 1, 1], 0, msg="Got %s" % str(t[0, 1]))
        self.assertEqual(t[0, 1, 2], 0.2, msg="Got %s" % str(t[0, 1]))
        self.assertEqual(t[0, 1, 3], 0.8, msg="Got %s" % str(t[0, 1]))

        self.assertEqual(t[0, 2, 0], 0.8, msg="Got %s" % str(t[0, 2]))
        self.assertEqual(t[0, 2, 1], 0.4, msg="Got %s" % str(t[0, 2]))
        self.assertEqual(t[0, 2, 2], 0.9, msg="Got %s" % str(t[0, 2]))
        self.assertEqual(t[0, 2, 3], 0.2, msg="Got %s" % str(t[0, 2]))

        self.assertEqual(t.shape[2], coords.get_length())
        self.assertEqual(t.shape[2], 5, "We expect 4 geometric values and one confidence value, but have shape=%s "
                                        "and tensor %s" % (t.shape, t[0, 0]))

    def test_md2cart(self):
        points = MidDirLines.to_cart([0.5, 0.5, 1, 1])
        diff = points - torch.tensor([0, 0, 1, 1.])
        self.assertTrue(torch.all(diff <= 0.0001), points)

        points = MidDirLines.to_cart([0.6, 0.4, -0.8, 0.8])
        diff = points - torch.tensor([1, 0, 0.2, 0.8])
        self.assertTrue(torch.all(diff <= 0.0001), points)

        points = MidDirLines.to_cart([0.85, 0.3, 0.1, -0.2])
        diff = points - torch.tensor([0.8, 0.4, 0.9, 0.2])
        self.assertTrue(torch.all(diff <= 0.0001), points)

        mld = [0.46081206, 0.5623043, 0.95877594, -0.34685016]
        MidDirLines.to_cart(mld, raise_error=True)

        mld = [0.4489532, 0.46790153, 0.9539267, 0.77268744]
        MidDirLines.to_cart(mld, raise_error=True)

        mld = [0.4351161, 0.44949794, 0.9644572, -0.6883276]
        MidDirLines.to_cart(mld, raise_error=True)

    def test_md(self):
        grid, coords = self.prepare(line_representation=LINE.MID_DIR)
        t = grid.tensor(convert_to_lrep=LINE.MID_DIR, coords=coords)

        self.assertEqual(t[0, 0, 0], 0.5, msg="Got %s" % str(t[0, 0]))
        self.assertEqual(t[0, 0, 1], 0.5, msg="Got %s" % str(t[0, 0]))
        self.assertEqual(t[0, 0, 2], math.cos(math.pi / 4) * math.sqrt(2), msg="Got %s" % str(t[0, 0]))
        self.assertEqual(t[0, 0, 3], math.sin(math.pi / 4) * math.sqrt(2), msg="Got %s" % str(t[0, 0]))

        self.assertAlmostEqual(t[0, 1, 0], 0.6, msg="Got %s" % str(t[0, 1]), places=2)
        self.assertAlmostEqual(t[0, 1, 1], 0.4, msg="Got %s" % str(t[0, 1]), places=2)
        self.assertAlmostEqual(t[0, 1, 2], -0.8, msg="Got %s" % str(t[0, 1]), places=2)
        self.assertAlmostEqual(t[0, 1, 3], 0.8, msg="Got %s" % str(t[0, 1]), places=2)

        self.assertAlmostEqual(t[0, 2, 0], 0.85, msg="Got %s" % str(t[0, 2]), places=2)
        self.assertAlmostEqual(t[0, 2, 1], 0.3, msg="Got %s" % str(t[0, 2]), places=2)
        self.assertAlmostEqual(t[0, 2, 2], 0.1, msg="Got %s" % str(t[0, 2]), places=2)
        self.assertAlmostEqual(t[0, 2, 3], -0.2, msg="Got %s" % str(t[0, 2]), places=2)

        self.assertEqual(t.shape[2], coords.get_length())
        self.assertEqual(coords.get_length(), 5,
                         "We expect 5 geometric values and one confidence value, but have shape=%s "
                         "and tensor %s" % (t.shape, t[0, 0]))

    def test_oned(self):
        grid, coords = self.prepare(line_representation=LINE.ONE_D)
        t = grid.tensor(convert_to_lrep=LINE.ONE_D, line_as_is=True, coords=coords)
        self.assertEqual(t[0, 0, 0], 0, msg="Got %s" % str(t[0, 0]))
        self.assertEqual(t[0, 0, 1], 0.5, msg="Got %s" % str(t[0, 0]))
        self.assertTrue(t[0, 1, 0:2].isnan().all(), msg="Got %s" % str(t[0, 1]))
        self.assertTrue(t[0, 2, 0:2].isnan().all(), msg="Got %s" % str(t[0, 2]))

        t = grid.tensor(convert_to_lrep=LINE.ONE_D, line_as_is=False, coords=coords)
        self.assertEqual(t[0, 0, 0], 0, msg="Got %s" % str(t[0, 0]))
        self.assertEqual(t[0, 0, 1], 0.5, msg="Got %s" % str(t[0, 0]))

        self.assertAlmostEqual(t[0, 1, 0].item(), 0.75, msg="Got %s" % str(t[0, 1]), places=2)
        self.assertAlmostEqual(t[0, 1, 1].item(), 0.25, msg="Got %s" % str(t[0, 1]), places=2)

        self.assertAlmostEqual(t[0, 2, 0].item(), 0.5 - (0.25 / 2.), msg="Got %s" % str(t[0, 2]), places=2)
        self.assertAlmostEqual(t[0, 2, 1].item(), 0.75, msg="Got %s" % str(t[0, 2]), places=2)

        self.assertEqual(t.shape[2], coords.get_length())
        self.assertEqual(coords.get_length(), 3,
                         "We expect 2 geometric values and one confidence value, but have shape=%s "
                         "and tensor %s" % (t.shape, t[0, 0]))

    def test_euler(self):
        cos_sin_45_degree = math.cos(math.pi / 4.)

        grid, coords = self.prepare(line_representation=LINE.EULER)
        t = grid.tensor(convert_to_lrep=LINE.EULER, line_as_is=True, coords=coords)
        self.assertAlmostEqual(t[0, 0, 0], -cos_sin_45_degree, msg="Got %s" % str(t[0, 0]))  # cos
        self.assertAlmostEqual(t[0, 0, 1], -cos_sin_45_degree, msg="Got %s" % str(t[0, 0]))  # sin
        self.assertAlmostEqual(t[0, 0, 2], cos_sin_45_degree, msg="Got %s" % str(t[0, 0]))  # cos
        self.assertAlmostEqual(t[0, 0, 3], cos_sin_45_degree, msg="Got %s" % str(t[0, 0]))  # sin
        self.assertTrue(torch.all(t[0, 1, 0:4].isnan()), msg="Got %s" % str(t[0, 1]))
        self.assertTrue(torch.all(t[0, 2, 0:4].isnan()), msg="Got %s" % str(t[0, 2]))

        t = grid.tensor(convert_to_lrep=LINE.EULER, line_as_is=False, coords=coords)
        self.assertAlmostEqual(t[0, 0, 0], -cos_sin_45_degree, msg="Got %s" % str(t[0, 0]), places=4)  # cos
        self.assertAlmostEqual(t[0, 0, 1], -cos_sin_45_degree, msg="Got %s" % str(t[0, 0]), places=4)  # sin
        self.assertAlmostEqual(t[0, 0, 2], cos_sin_45_degree, msg="Got %s" % str(t[0, 0]), places=4)  # cos
        self.assertAlmostEqual(t[0, 0, 3], cos_sin_45_degree, msg="Got %s" % str(t[0, 0]), places=4)  # sin

        self.assertAlmostEqual(t[0, 1, 0], cos_sin_45_degree, msg="Got %s" % str(t[0, 1]), places=4)
        self.assertAlmostEqual(t[0, 1, 1], -cos_sin_45_degree, msg="Got %s" % str(t[0, 1]), places=4)
        self.assertAlmostEqual(t[0, 1, 2], -cos_sin_45_degree, msg="Got %s" % str(t[0, 1]), places=4)
        self.assertAlmostEqual(t[0, 1, 3], cos_sin_45_degree, msg="Got %s" % str(t[0, 1]), places=4)

        self.assertAlmostEqual(t[0, 2, 0], 0.3 / math.sqrt(0.3 ** 2 + 0.1 ** 2), msg="Got %s" % str(t[0, 2]), places=4)
        self.assertAlmostEqual(t[0, 2, 1], -0.1 / math.sqrt(0.3 ** 2 + 0.1 ** 2), msg="Got %s" % str(t[0, 2]), places=4)
        self.assertAlmostEqual(t[0, 2, 2], 0.4 / math.sqrt(0.4 ** 2 + 0.3 ** 2), msg="Got %s" % str(t[0, 2]), places=4)
        self.assertAlmostEqual(t[0, 2, 3], -0.3 / math.sqrt(0.4 ** 2 + 0.3 ** 2), msg="Got %s" % str(t[0, 2]), places=4)

        self.assertEqual(t.shape[2], coords.get_length())
        self.assertEqual(coords.get_length(), 5,
                         "We expect 4 geometric values and one confidence value, but have shape=%s "
                         "and tensor %s" % (t.shape, t[0, 0]))

    def test_md_conversion(self):
        test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                   additional_vals={"linerep": str(LINE.MID_DIR)}
                   # level=Level.DEBUG
                   )

        points = [
            [0., 0, 1, 1],
            [0.1, 0.1, 0.9, 0.9],
            [0.1, 0.4, 0.9, 0.6],
            [0.8, 0.4, 0.9, 0.2]
        ]
        for point in points:
            md_point = list(MidDirLines.from_cart(point[0:2], point[2:4]))
            x_dif = point[2] - point[0]
            y_dif = point[3] - point[1]
            expected_md = [x_dif / 2 + point[0], y_dif / 2 + point[1], x_dif, y_dif]

            for idx in range(0, 4):
                self.assertAlmostEqual(md_point[idx], expected_md[idx],
                                       msg="For point %s, we get mld=%s, but expected %s" % (point,
                                                                                             np.round(md_point, 3),
                                                                                             np.round(expected_md, 3)))
            converted_point = list(MidDirLines.to_cart(md_point).numpy())
            for idx in range(0, 4):
                self.assertAlmostEqual(converted_point[idx], point[idx],
                                       msg="For point %s, we get points=%s, but expected %s" % (
                                           point, converted_point, expected_md))

    def test_oned_conversion(self):
        test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                   additional_vals={"linerep": str(LINE.ONE_D)}
                   # level=Level.DEBUG
                   )

        point = [0.1, 0.1, 0.9, 0.9]
        oned_point = OneDLines.from_cart(point[0:2], point[2:4], extrapolate=True)
        self.assertEqual(oned_point, [0, 0.5], oned_point)
        converted_point = OneDLines.to_cart(oned_point)
        self.assertEqual(converted_point, [0, 0, 1, 1], converted_point)

        point = [0.1, 0.4, 0.9, 0.6]
        oned_point = OneDLines.from_cart(point[0:2], point[2:4], extrapolate=True)
        expected_oned = [0.375 / 4, 0.75 - (0.625 / 4)]
        self.assertEqual(oned_point, expected_oned, oned_point)
        converted_point = OneDLines.to_cart(oned_point)
        self.assertEqual(converted_point, [0, 0.375, 1, 0.625], converted_point)
        
        point = [0.8, 0.4, 0.9, 0.2]
        oned_point = OneDLines.from_cart(point[0:2], point[2:4], extrapolate=True)
        expected_oned = [0.375, 0.75]
        self.assertEqual(oned_point, expected_oned, oned_point)
        converted_point = OneDLines.to_cart(oned_point)
        self.assertEqual(converted_point, [0.5, 1, 1, 0], converted_point)

    def test_extrapolate(self):
        test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                   additional_vals={"linerep": str(LINE.ONE_D)}
                   # level=Level.DEBUG
                   )

        point = [0.1, 0.1, 0.9, 0.9]
        extrapolated_start, extrapolated_end = PointsLines.extrapolate_cart(point[0:2], point[2:4])
        self.assertEqual(list(extrapolated_start), [0, 0])
        self.assertEqual(list(extrapolated_end), [1, 1])

        point = [0.1, 0.4, 0.9, 0.6]
        extrapolated_start, extrapolated_end = PointsLines.extrapolate_cart(point[0:2], point[2:4])
        self.assertEqual(list(extrapolated_start), [0, 0.375])
        self.assertEqual(list(extrapolated_end), [1, 0.625])

        point = [0.8, 0.4, 0.9, 0.2]
        extrapolated_start, extrapolated_end = PointsLines.extrapolate_cart(point[0:2], point[2:4])
        self.assertEqual(list(extrapolated_start), [0.5, 1])
        self.assertEqual(list(extrapolated_end), [1, 0])

        point = [0.1, 0.4, 0.9, 0.4]
        extrapolated_start, extrapolated_end = PointsLines.extrapolate_cart(point[0:2], point[2:4])
        self.assertEqual(list(extrapolated_start.round(2)), [0, 0.4])
        self.assertEqual(list(extrapolated_end.round(2)), [1, 0.4])
