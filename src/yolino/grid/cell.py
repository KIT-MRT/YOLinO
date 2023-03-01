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
class Cell:
    def __init__(self, row, col, num_predictors, predictors, suppress=False, threshold=-1):
        """

        :type predictors: [Predictors]
        """
        self.predictors = predictors
        self.row = row
        self.col = col
        self.threshold = threshold
        self.num_predictors = num_predictors

    def __iter__(self):
        return iter(self.predictors)

    def append(self, data, update=False):
        if len(self.predictors) >= self.num_predictors:
            if update:
                self.predictors[0].update(data)
                return True
            else:
                return False
        else:
            self.predictors.append(data)
            return True

    def __len__(self):
        return len(self.predictors)

    def __str__(self):
        return "Cell with %s Predictors" % self.predictors
