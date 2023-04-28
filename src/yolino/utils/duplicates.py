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
import torch

from yolino.utils.logger import Log


class LineDuplicates:
    def __init__(self, filename, grid_shape, num_predictors):
        self.filename = filename
        self.__total_in_image__ = 0
        self.__total_correct_in_image = 0
        self.__total_per_cell__ = torch.zeros([*grid_shape, num_predictors], dtype=torch.int)

    def add(self, row, col, predictor_id):
        self.__total_in_image__ += 1
        self.__total_per_cell__[row, col, predictor_id] += 1

    def add_ok(self):
        self.__total_correct_in_image += 1

    def __len__(self):
        return self.__total_in_image__

    def summarize(self, grid_rows=20):
        if len(self) == 0:
            Log.debug("No duplicates")
        else:
            height = self.height()
            if height[2] >= grid_rows / 2.:
                Log.debug(
                    "We have to neglected %d (%.2f %%) GT elements, mean height is %d, median is %.1f, max is %d, as we have duplicate matches."
                    "We continue, but you might want to fix that for %s"
                    % (len(self), self.percentage(), height[0], height[1], height[2], self.filename))
            else:
                Log.debug("We have %d duplicates, but only at horizon (height<=%d)." % (len(self), height[2]))

    def bad_boys(self, threshold=0):
        where = torch.where(self.__total_per_cell__ > threshold)
        return torch.concat([torch.stack(where, dim=1), self.__total_per_cell__[where].reshape((-1, 1))], dim=1)

    def worst_boy(self):
        return torch.stack(torch.where(self.__total_per_cell__ == self.max()), dim=1)[0].type(torch.IntTensor)

    def max(self):
        return torch.max(self.__total_per_cell__).type(torch.IntTensor)

    def __str__(self):
        return f"{self.__total_in_image__} ({self.percentage():.1f}%) duplicates\n" \
               f"mean/median/max height is {self.height()[0]:.1f}/{self.height()[1]:d}/{self.height()[2]:d}\n" \
               f"Cell {self.worst_boy().numpy()} is one of the worst with {self.max()}"

    def dict(self):
        return {"total_duplicates_in_image": self.total(), "cells": self.__total_per_cell__,
                "worst_anchor": self.worst_boy(), "max_duplicate_on_anchor": self.max(),
                "mean_height": self.height()[0], "median_height": self.height()[1], "max_height": self.height()[2]}

    def total(self):
        return torch.tensor(self.__total_in_image__, dtype=torch.int)

    def percentage(self):
        return self.__total_in_image__ / (self.__total_correct_in_image + self.__total_in_image__) * 100

    def height(self):
        if not torch.any(self.__total_per_cell__):
            return -1, -1, -1

        dupl_per_row = torch.sum(torch.sum(self.__total_per_cell__, dim=1), dim=1)
        valid_rows = torch.where(dupl_per_row > 0)[0]
        dupl_per_row = dupl_per_row[valid_rows]
        rep = torch.repeat_interleave(valid_rows, dupl_per_row)
        med = torch.median(rep)
        mean = torch.mean(rep * 1.)
        max = torch.max(rep)

        return mean.type(torch.FloatTensor), med.type(torch.IntTensor), max.type(torch.IntTensor)
