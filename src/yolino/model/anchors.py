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
import timeit
from abc import abstractmethod

import torch
from yolino.model.line_representation import LineRepresentation
from yolino.model.variable_structure import VariableStructure
from yolino.tools.kmeans_anchor_fit import get_kmeans_cluster
from yolino.utils.enums import AnchorDistribution, AnchorVariables, LINE, Variables
from yolino.utils.logger import Log


class Anchor:
    @classmethod
    def get(cls, args, linerep: LINE, angle_range=1. * math.pi):
        if linerep == LINE.POINTS:
            return PointsAnchor(args=args)
        elif linerep == LINE.MID_DIR:
            return MLDAnchor(args=args, angle_range=angle_range)
        elif args.anchors == AnchorDistribution.NONE:
            return Anchor(args=args, linerep=args.linerep, angle_range=angle_range)
        else:
            raise NotImplementedError()

    def __init__(self, args, linerep, angle_range=1. * math.pi):

        self.args = args
        self.heatmap = None
        self.clear_heatmap()
        self.conf_heatmap = None
        self.clear_conf_heatmap()

        self.angle_range = angle_range
        self.offset = -0.5 * math.pi
        self.linerep = linerep
        self.is_offset = args.offset
        self.anchor_coords = VariableStructure(line_representation_enum=linerep, num_conf=0)

        if self.args.anchors == AnchorDistribution.NONE:
            self.bins = torch.tensor([])
        else:
            Log.warning("Generate %s anchors for %s lines "
                        "based on %s variable." % (args.anchors, linerep, args.anchor_vars))
            timer = timeit.default_timer()
            self.bins = self.generate()
            Log.time(key="anchor_generation", value=(timeit.default_timer() - timer))
        self.validate()

    def __getitem__(self, item):
        return self.bins.__getitem__(item)

    def __len__(self):
        return self.bins.__len__()

    @classmethod
    def get_columns(self, anchor_vars):

        if AnchorVariables.POINTS in anchor_vars:
            kmeans_columns = ["x_s", "y_s", "x_e", "y_e"]
        else:
            kmeans_columns = []
            if AnchorVariables.DIRECTION in anchor_vars:
                kmeans_columns.append("dx")
                kmeans_columns.append("dy")
            if AnchorVariables.MIDPOINT in anchor_vars:
                kmeans_columns.append("mx")
                kmeans_columns.append("my")

        sort_by = "cluster"
        return kmeans_columns, sort_by

    def __generate_angles__(self, num_predictors: int = -1):
        return torch.linspace(-1, 1, num_predictors + 2)[1:-1] * self.angle_range + self.offset

    def __generate_directions__(self, num_predictors, do_plot=False):
        import math
        rt = math.sqrt(1)
        data = torch.tensor([[1, 0],
                             [0, 1],
                             [-1, 0],
                             [0, -1],
                             [rt, rt],
                             [rt, -rt],
                             [-rt, rt],
                             [-rt, -rt]
                             ])

        if num_predictors != len(data):
            Log.malconfig("We can only generate %d equally distributed direction anchors." % len(data))

        if do_plot:
            import math
            import matplotlib.pyplot as plt
            factor = 4
            fig, axs = plt.subplots(int(len(data) / factor), factor, sharex=True, sharey=True)
            for i, d in enumerate(data):
                c = i % factor
                r = int(i / factor)

                axs[r, c].arrow(0.5 - d[0], 0.5 - d[1], d[0], d[1], head_width=0.1, head_starts_at_zero=True)
                axs[r, c].set_xlim(-0.1, 1.1)
                axs[r, c].set_ylim(-0.1, 1.1)
                axs[r, c].set_aspect('equal', 'box')
                axs[r, c].invert_yaxis()
                # axs[r, c].set_title(str(d))
            plt.show()
        return data

    def __generate_midpoints__(self, num_predictors: int = -1, do_plot=False):
        data = torch.tensor([[0.25, 0.25],
                             [0.75, 0.75],
                             [0.25, 0.75],
                             [0.75, 0.25],
                             [0.25, 0.5],
                             [0.75, 0.5],
                             [0.5, 0.25],
                             [0.5, 0.75],
                             ])

        if num_predictors != len(data):
            Log.malconfig("We can only generate %d equally distributed midpoint anchors." % len(data))

        if do_plot:
            import math
            import matplotlib.pyplot as plt
            factor = 4
            fig, axs = plt.subplots(int(len(data) / factor), factor, sharex=True, sharey=True)
            for i, d in enumerate(data):
                c = i % factor
                r = int(i / factor)

                axs[r, c].scatter(d[0], d[1], marker="x")
                axs[r, c].set_xlim(-0.1, 1.1)
                axs[r, c].set_ylim(-0.1, 1.1)
                axs[r, c].set_aspect('equal', 'box')
                axs[r, c].invert_yaxis()
                # axs[r, c].set_title(str(d))
            plt.show()
        return torch.tensor(data, dtype=torch.float)

    def __generate_midpoints_direction__(self, num_predictors: int = -1, combine=True, do_plot=False):
        data = torch.tensor([
            # left towards center
            [0.25, 0.25, 0.5, -0.5],
            [0.25, 0.25, -0.5, 0.5],

            [0.75, 0.75, 0.5, -0.5],
            [0.75, 0.75, -0.5, 0.5],

            # right towards center
            [0.75, 0.25, -0.5, -0.5],
            [0.75, 0.25, 0.5, 0.5],

            [0.25, 0.75, -0.5, -0.5],
            [0.25, 0.75, 0.5, 0.5],

            # horizon other
            [0.25, 0.5, 0, 1],
            [0.25, 0.5, 0, -1],

            [0.75, 0.5, 0, 1],
            [0.75, 0.5, 0, -1],

            # vertical other
            [0.5, 0.25, 1, 0],
            [0.5, 0.25, -1, 0],

            [0.5, 0.75, 1, 0],
            [0.5, 0.75, -1, 0],
        ])

        if num_predictors == 24:
            data = torch.vstack([data, torch.tensor([
                # center to right corner
                [0.5, 0.5, 1, -1],
                [0.5, 0.5, -1, 1],

                # center to left corner
                [0.5, 0.5, -1, -1],
                [0.5, 0.5, 1, 1],

                # horizon center
                [0.5, 0.5, 0, 1],
                [0.5, 0.5, 0, -1],

                # vertical center
                [0.5, 0.5, 1, 0],
                [0.5, 0.5, -1, 0],

            ])])

        if num_predictors != len(data):
            Log.malconfig("We can only generate 16 or 24 equally distributed midpoint-direction anchors.")

        if do_plot:
            import math
            import matplotlib.pyplot as plt
            factor = 4
            fig, axs = plt.subplots(int(len(data) / factor), factor, sharex=True, sharey=True)
            for i, d in enumerate(data):
                c = i % factor
                r = int(i / factor)

                axs[r, c].arrow(d[0] - d[2], d[1] - d[3], 2 * d[2], 2 * d[3], head_width=0.1, head_starts_at_zero=True)
                axs[r, c].set_xlim(-0.1, 1.1)
                axs[r, c].set_ylim(-0.1, 1.1)
                axs[r, c].set_aspect('equal', 'box')
                axs[r, c].invert_yaxis()
                # axs[r, c].set_title(str(d))
            plt.show()

        return data

    @abstractmethod
    def generate(self):
        pass

    def __generate_position_from_kmeans__(self):

        if not self.args.anchors == AnchorDistribution.KMEANS:
            raise NotImplementedError()

        if self.args.linerep == LINE.MID_DIR:
            columns = ["mx", "my", "dx", "dy"]
        elif self.args.linerep == LINE.POINTS:
            columns = ["x_s", "y_s", "x_e", "y_e"]
        else:
            raise NotImplementedError()

        data = get_kmeans_cluster(self.args)
        if not "anchor_kmeans" in data:
            raise AttributeError("No kmeans anchors in your specs file")

        bins = torch.zeros((self.args.num_predictors, 4))
        try:
            for i in range(self.args.num_predictors):
                bins[i] = torch.tensor([data["anchor_kmeans"][i][key] for key in columns])
        except KeyError as ex:
            kmeans_yaml_file = self.args.paths.generate_specs_file_name(dataset=self.args.dataset, split="train",
                                                                        anchor_vars=self.args.anchor_vars,
                                                                        scale=self.args.scale,
                                                                        num_predictors=self.args.num_predictors)
            Log.error("We remove %s. It is missing a key" % kmeans_yaml_file)
            Log.error(ex)
            os.remove(kmeans_yaml_file)
            bins = self.__generate_position_from_kmeans__()

        return bins

    def add_conf_heatmap(self, row, col, anchor_id):
        self.conf_heatmap[row, col, anchor_id] += 1

    def add_heatmap(self, row, col, anchor_id):
        if self.heatmap[row, col, anchor_id] < 0:
            self.heatmap[row, col, anchor_id] = 0

        self.heatmap[row, col, anchor_id] += 1

    def get_specific_anchors(self, predictor: torch.tensor):
        # get position of actual anchor variable
        geom_pos = self.anchor_coords.get_position_of(Variables.GEOMETRY, anchor_vars=self.args.anchor_vars)
        offsets = predictor[geom_pos] - self.bins[:, geom_pos]  # TODO this assumes geometry is always first in coords!

        norms = torch.linalg.norm(offsets, dim=1)
        min_val = torch.min(norms)

        eps = 0.0001  # TODO: proper epsilon for anchor range
        indices = torch.where(torch.logical_and(norms >= min_val - eps, norms <= min_val + eps))[0]

        # TODO handle multi assignment anchors; this deletes all dublicates
        if len(indices) > 1:
            indices = indices[[0]]

        # the offset has been calculated in the anchor cluster space, but we want the full description
        geom_pos = self.anchor_coords.get_position_of(Variables.GEOMETRY)
        offsets = predictor[geom_pos] - self.bins[:, geom_pos]  # TODO this assumes geometry is always first in coords!

        full_predictor_with_offset = torch.tile(predictor, (len(indices), 1))
        full_predictor_with_offset[:, geom_pos] = offsets[indices]
        return indices, full_predictor_with_offset

    def __repr__(self):
        return self.bins.__repr__()

    def validate(self):
        if self.args.anchors != AnchorDistribution.NONE:
            if len(self.bins) != self.args.num_predictors:
                raise ValueError("We found more/loess bins compared to what you required. %s vs %s"
                                 % (self.bins.shape, self.args.num_predictors))

            num_params = LineRepresentation.get(self.linerep).num_params
            if self.bins.shape[1] != num_params:
                raise ValueError("We found more/loess bins for %s compared to what you required. %s vs %s"
                                 % (self.linerep, self.bins.shape, num_params))

    def clear_heatmap(self):
        self.heatmap = torch.ones((self.args.grid_shape[0], self.args.grid_shape[1], self.args.num_predictors)) * -10

    def clear_conf_heatmap(self):
        self.conf_heatmap = torch.ones(
            (self.args.grid_shape[0], self.args.grid_shape[1], self.args.num_predictors)) * -10


class PointsAnchor(Anchor):

    def __init__(self, args):
        super().__init__(args, linerep=LINE.POINTS)

    def generate(self, do_plot=False):
        if self.args.anchors == AnchorDistribution.EQUAL:
            bins = torch.tensor([
                # center
                [0, 0, 1, 1],
                [0, 0.5, 1, 0.5],
                [0, 1, 1, 0],
                [0.5, 1, 0.5, 0],
                [1, 1, 0, 0],
                [1, 0.5, 0, 0.5],
                [1, 0, 0, 1],
                [0.5, 0, 0.5, 1]])

            if self.args.num_predictors == 24:
                bins = torch.vstack([bins, torch.tensor([
                    # corners/borders
                    [0.25, 0, 0, 0.25],
                    [0, 0.25, 0.25, 0.],
                    [1, 0.75, 0.75, 1],
                    [0.75, 1, 1, 0.75],

                    [0.75, 0, 1, 0.25],
                    [1, 0.25, 0.75, 0],
                    [0, 0.75, 0.25, 1],
                    [0.25, 1, 0, 0.75],

                    [0.25, 0, 0.25, 1],
                    [0.25, 1, 0.25, 0],
                    [0.75, 0, 0.75, 1],
                    [0.75, 1, 0.75, 0],

                    [0, 0.25, 1, 0.25],
                    [1, 0.25, 0, 0.25],
                    [0, 0.75, 1, 0.75],
                    [1, 0.75, 0, 0.75]
                ])])

            if self.args.num_predictors != len(bins):
                Log.malconfig("We can only generate %d equally distributed points anchors." % len(bins))
        elif self.args.anchors == AnchorDistribution.KMEANS:
            bins = self.__generate_position_from_kmeans__()
        else:
            raise NotImplementedError("we did not implement %s for %s" % (self.args.anchors, self.linerep))

        if do_plot:
            import math
            import matplotlib.pyplot as plt
            factor = 4
            fig, axs = plt.subplots(int(len(bins) / factor), factor, sharex=True, sharey=True)
            for i, d in enumerate(bins):
                c = i % factor
                r = int(i / factor)

                axs[r, c].arrow(d[0] - d[2], d[1] - d[3], 2 * d[2], 2 * d[3], head_width=0.1,
                                head_starts_at_zero=True)
                axs[r, c].set_xlim(-0.1, 1.1)
                axs[r, c].set_ylim(-0.1, 1.1)
                axs[r, c].set_aspect('equal', 'box')
                axs[r, c].invert_yaxis()
                # axs[r, c].set_title(str(d))
            plt.show()

        return bins


class MLDAnchor(Anchor):
    def __init__(self, args, angle_range=1. * math.pi):
        super().__init__(args, LINE.MID_DIR, angle_range=angle_range)

    def generate(self):
        if self.args.anchors == AnchorDistribution.EQUAL:
            mpx = torch.tensor([0.5] * self.args.num_predictors)
            mpy = torch.tensor([0.5] * self.args.num_predictors)
            dx = torch.tensor([1.] * self.args.num_predictors)
            dy = torch.tensor([1.] * self.args.num_predictors)

            if AnchorVariables.DIRECTION in self.args.anchor_vars:
                if AnchorVariables.MIDPOINT in self.args.anchor_vars:
                    bins = self.__generate_midpoints_direction__(self.args.num_predictors, combine=True)
                    positions = torch.stack([bins[:, 0], bins[:, 1], bins[:, 2], bins[:, 3]], dim=1).reshape((-1, 4))
                else:
                    bins = self.__generate_directions__(self.args.num_predictors)
                    positions = torch.stack([mpx, mpy, bins[:, 0], bins[:, 1]], dim=-1).reshape((-1, 4))
            elif AnchorVariables.MIDPOINT in self.args.anchor_vars:
                bins = self.__generate_midpoints__(self.args.num_predictors)
                positions = torch.stack([bins[:, 0], bins[:, 1], dx, dy], dim=-1).reshape((-1, 4))

            else:
                raise NotImplementedError("With line representation %s "
                                          "we do not support %s" % (self.linerep, self.args.anchor_vars))
        elif self.args.anchors == AnchorDistribution.KMEANS:

            columns = []
            if AnchorVariables.DIRECTION in self.args.anchor_vars:
                columns.append("dx")
                columns.append("dy")
            if AnchorVariables.MIDPOINT in self.args.anchor_vars:
                columns.append("mx")
                columns.append("my")
            positions = self.__generate_position_from_kmeans__()
        elif self.args.anchors == AnchorDistribution.NONE:
            return torch.tensor([])
        else:
            raise NotImplementedError(self.args.anchors)

        return positions
