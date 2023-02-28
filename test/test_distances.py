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

from yolino.eval.distances import get_angular_distance, get_perpendicular_sum_distance, get_parallel_distance, \
    linesegment_pauls_distance


class DistanceTest(unittest.TestCase):
    def test_angular(self):
        for factor in [1, 2.4]:
            l1 = torch.tensor([0, 0, 1, 1.]) * factor
            len_l1 = torch.linalg.norm(l1[0:2] - l1[2:4])

            opposite = torch.tensor([1., 1, 0, 0]) * factor
            len_opposite = torch.linalg.norm(opposite[0:2] - opposite[2:4])

            self.assertEqual(get_angular_distance(l1, len_l1, l1, len_l1), 0)
            self.assertEqual(get_angular_distance(l1, len_l1, opposite, len_opposite), 180)

            perp = torch.tensor([0., 1, 1, 0]) * factor
            len_perp = torch.linalg.norm(perp[0:2] - perp[2:4])
            self.assertAlmostEqual(get_angular_distance(l1, len_l1, perp, len_perp), 90, places=3)

            parallel = torch.tensor([0, 0.1, 0.9, 1]) * factor
            len_parallel = torch.linalg.norm(parallel[0:2] - parallel[2:4])
            self.assertAlmostEqual(get_angular_distance(l1, len_l1, parallel, len_parallel), 0, places=3)

    def test_perp(self):
        for factor in [1, 2.4]:
            l1 = torch.tensor([0, 0, 1, 1.]) * factor
            len_l1 = torch.linalg.norm(l1[0:2] - l1[2:4])

            opposite = torch.tensor([1., 1, 0, 0]) * factor
            len_opposite = torch.linalg.norm(opposite[0:2] - opposite[2:4])

            self.assertEqual(get_perpendicular_sum_distance(l1, len_l1, l1, len_l1), 0)
            self.assertEqual(get_perpendicular_sum_distance(l1, len_l1, opposite, len_opposite), 0)

            perp = torch.tensor([0., 1, 1, 0]) * factor
            len_perp = torch.linalg.norm(perp[0:2] - perp[2:4])
            self.assertEqual(get_perpendicular_sum_distance(l1, len_l1, perp, len_perp), len_l1 / 2. * 4)

            parallel = torch.tensor([0, 0.1, 0.9, 1]) * factor
            len_parallel = torch.linalg.norm(parallel[0:2] - parallel[2:4])

            if factor == 1:
                self.assertAlmostEqual(get_perpendicular_sum_distance(l1, len_l1, parallel, len_parallel).item(),
                                       np.linalg.norm([parallel[1] - 0.041 * factor, parallel[0] - 0.041 * factor]) * 4,
                                       places=2)  # number is a guess
            else:
                self.assertAlmostEqual(get_perpendicular_sum_distance(l1, len_l1, parallel, len_parallel).item(),
                                       np.linalg.norm([0.1 * 2.4 - 0.096, 0 * 2.4 - 0.089]) * 4,
                                       places=2)  # number is a guess

    def test_parallel(self):
        for factor in [1, 2.4]:
            l1 = torch.tensor([0, 0, 1, 1.]) * factor
            len_l1 = torch.linalg.norm(l1[0:2] - l1[2:4])

            opposite = torch.tensor([1., 1, 0, 0]) * factor
            len_opposite = torch.linalg.norm(opposite[0:2] - opposite[2:4])

            self.assertEqual(get_parallel_distance(l1, len_l1, l1, len_l1), 0)
            self.assertEqual(get_parallel_distance(l1, len_l1, opposite, len_opposite), 2 * len_l1)

            perp = torch.tensor([0., 1, 1, 0]) * factor
            len_perp = torch.linalg.norm(perp[0:2] - perp[2:4])
            self.assertEqual(get_parallel_distance(l1, len_l1, perp, len_perp), len_l1)

            parallel = torch.tensor([0, 0.1, 0.9, 1]) * factor
            len_parallel = torch.linalg.norm(parallel[0:2] - parallel[2:4])
            self.assertAlmostEqual(get_parallel_distance(l1, len_l1, parallel, len_parallel).item(),
                                   (abs(len_l1 - len_parallel)).item(),
                                   places=2)

    def test_pauls(self):
        for factor in [1, 2.4]:
            l1 = torch.tensor([0, 0, 1, 1.]) * factor
            len_l1 = torch.linalg.norm(l1[0:2] - l1[2:4])

            self.assertEqual(linesegment_pauls_distance(l1, l1), 0)

            opposite = torch.tensor([1., 1, 0, 0]) * factor
            self.assertAlmostEqual(linesegment_pauls_distance(l1, opposite).item(),
                                   math.sqrt(math.pow(1 * factor / len_l1 + 1 * factor / len_l1, 2) * 2), places=2)

            perp = torch.tensor([0., 1, 1, 0]) * factor
            self.assertAlmostEqual(linesegment_pauls_distance(l1, perp).item(),
                                   math.sqrt(math.pow(1 * factor / len_l1 + 1 * factor / len_l1, 2) +
                                             math.pow(1 * factor / len_l1 - 1 * factor / len_l1, 2)), places=2)

            # TODO: test parallel - norm_diff is missing
            # parallel = torch.tensor([0, 0.1, 0.9, 1]) * factor
            # len_parallel = torch.linalg.norm(parallel[0:2]-parallel[2:4])
            # self.assertAlmostEqual(linesegment_pauls_distance(l1, parallel).item(),
            #                        math.sqrt(math.pow(0.04, 4)+math.pow(len_parallel-len_l1, 2)),
            #                        places=2)


if __name__ == '__main__':
    unittest.main()
