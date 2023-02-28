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

from yolino.grid.cell import Cell
from yolino.grid.coordinates import validate_input_structure
from yolino.grid.grid import Grid
from yolino.grid.predictor import Predictor
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import CoordinateSystem, Variables, LINE, AnchorDistribution
from yolino.utils.logger import Log


def __strip_geometry__(lines, coordinate: CoordinateSystem):
    if coordinate == CoordinateSystem.UV_CONTINUOUS:  # "(batch, instances, control points, ?)"
        return lines[:, :, :, 0:2], lines[:, :, :, 2:]
    else:
        raise NotImplementedError


class GridFactory:
    @classmethod
    def get(cls, data, variables, coordinate: CoordinateSystem, args, input_coords, threshold=0, scale=1,
            variables_have_conf=True, only_train_vars=False, allow_down_facing=True, anchors=None,
            plot_image=False) -> (Grid, []):
        validate_input_structure(data, coordinate, args)

        if coordinate == CoordinateSystem.EMPTY:
            return Grid(img_height=args.img_size[0], args=args), []
        elif coordinate == CoordinateSystem.UV_CONTINUOUS:
            return cls.__grid_from_uv__(data, variables, args=args, coords=input_coords,
                                        variables_have_conf=variables_have_conf, allow_down_facing=allow_down_facing,
                                        plot_image=plot_image)
        elif coordinate == CoordinateSystem.UV_SPLIT:
            raise NotImplementedError
            # return cls.__grid_from_uv__(data, args.cell_size, args.num_predictors, args.grid_shape)
        elif coordinate == CoordinateSystem.CELL_SPLIT:
            if len(variables) > 0:
                Log.warning(
                    "Please put variables into the data for cell based handling. Separate data will be ignored.")

            if args.offset and args.anchors != AnchorDistribution.NONE and anchors is None:
                raise ValueError("We need anchors to retrieve absolute values from predicted offset values. "
                                 "This is a bug in the code not the configuration. Fix it!")
            return cls.__grid_from_prediction__(prediction=data, img_height=args.img_size[0], args=args,
                                                coords=input_coords, scale=scale,
                                                confidence_threshold=threshold, only_train_vars=only_train_vars,
                                                allow_down_facing=allow_down_facing, anchors=anchors)
        else:
            raise NotImplementedError

    # variables are appended to each geometry entry and are expected to have (batch, instances, num_vars)
    @classmethod
    def __grid_from_uv__(cls, geometry, variables, args, coords,
                         scale=1, swap=False, allow_out_of_bounds=False, plot_image=False, variables_have_conf=True,
                         allow_down_facing=True):

        errors = {}
        if len(geometry) == 0:
            Log.warning("No lines found. Created empty grid.")
            return Grid(img_height=args.img_size[0], args=args), errors

        # TODO: do we need to handle class labels here?
        # geometry = deepcopy(geometry)
        grid = Grid(img_height=args.img_size[0], args=args)

        if len(geometry) > args.num_predictors:
            Log.warning("You probably end up with more line segments per cell than the grid takes. " +
                        "Your data contains a huge number of polylines (%s) that is larger than your " % (
                            len(geometry)) +
                        "predictors per cell (%s)!" % (args.num_predictors))

        import matplotlib.pyplot as plt
        points_coords = coords.clone(LINE.POINTS)
        for b_idx, batch in enumerate(geometry):
            for instance_id, instance in enumerate(batch):
                valid_indices = np.where(np.logical_not(np.isnan(instance[:, 0])))  # check if first value is nan
                line = instance[valid_indices]

                if len(line) == 0:
                    continue

                if len(line) == 1:
                    Log.info("A GT line only has a single valid point. We remove the point %s in image of size %s"
                             % (line, args.img_size[0]))
                    # errors.append([b_idx, instance_id])
                    continue

                if len(np.unique(line)) == 2:
                    Log.info("A GT line only has a single valid point. We remove the point %s in image of size %s"
                             % (line, args.img_size[0]))
                    continue

                line = np.asarray(line) * scale
                row_col, position_in_cell, portion_in_cell = grid.get_position_of_line_segment(line)
                if len(row_col) < 2:
                    Log.warning("A line %s is only within a cell (%s). We ignore that!" % (str(line), str(row_col)))
                    continue

                for r, c in grid.get_unique_cell_corner_pairs_uv(row_col):
                    segments, geom_box = grid.slice_and_straighten_line(line, r, c, plot_image=plot_image)
                    for i, line_segment in enumerate(segments):
                        if np.any(line_segment < 0):
                            Log.warning("Negative position. Probably due to crop.")
                            continue

                        if np.any(np.isnan(line_segment)):
                            Log.error("Invalid line segment with nan!")
                            exit(3)

                        # TODO use params here
                        if len(variables) == 0:
                            grid.add_single_slice_uv(line_segment, [], r, c,
                                                     coords=points_coords,
                                                     allow_out_of_bounds=allow_out_of_bounds,
                                                     variables_have_conf=variables_have_conf,
                                                     allow_down_facing=allow_down_facing)
                        else:
                            grid.add_single_slice_uv(line_segment, variables[b_idx][instance_id], r, c,
                                                     coords=points_coords,
                                                     allow_out_of_bounds=allow_out_of_bounds,
                                                     variables_have_conf=variables_have_conf,
                                                     allow_down_facing=allow_down_facing)

            if grid.errors["knots_removed"] > 0:
                summary = {}
                summary["knots_removed"] = grid.errors["knots_removed"]

                if "area_removed" in grid.errors and len(grid.errors["area_removed"]) > 0:
                    summary["area_removed"] = {"median": np.median(grid.errors["area_removed"]),
                        "sum": np.sum(grid.errors["area_removed"]),
                        "max": np.max(grid.errors["area_removed"])}

                    if summary["area_removed"]["max"] > 16 * 16:
                        Log.info("Straighten the lines to the grid induced the following unusual errors:\n%s" % summary)
                        errors[b_idx] = summary

                if "max_distance" in grid.errors and len(grid.errors["max_distance"]) > 0:
                    summary["max_distance"] = {
                            "median": np.median([np.median(d) for d in grid.errors["max_distance"].values()]),
                            "sum": np.sum([np.sum(d) for d in grid.errors["max_distance"].values()]),
                            "max": np.max([np.max(d) for d in grid.errors["max_distance"].values()])
                        }

                    if summary["max_distance"]["max"] > 16:
                        Log.info(
                            "Straighten the lines to the grid induced the following unusual errors:\n%s" % summary)
                        errors[b_idx] = summary

                else:
                    Log.error(f"We removed {grid.errors['knots_removed']} knots, but did not measure the distance.")

                Log.scalars(tag="unknown_tag", dict=summary, epoch=None)
        return grid, errors

    # CoordinateSystem.CELL_SPLIT with (batch, cells, <=predictors, 2 * 2 + ?)
    @classmethod
    def __grid_from_prediction__(cls, prediction, img_height, args,
                                 coords: VariableStructure, confidence_threshold=0, scale=1, allow_nan=False,
                                 only_train_vars=False, allow_down_facing=True, anchors=None):
        Log.debug("Get grid from prediction")
        grid = Grid(img_height, args=args)
        shape = args.grid_shape

        use_conf = Variables.CONF in coords.train_vars() and coords[Variables.CONF] > 0
        if only_train_vars:
            conf_idx = coords.get_position_within_prediction(Variables.CONF)
        else:
            conf_idx = coords.get_position_of(Variables.CONF)

        for b_idx, batch in enumerate(prediction):
            # for c_idx, cell in enumerate(batch):
            for i, grid_cell in enumerate(batch):
                row = math.floor(i / shape[1])
                col = i % shape[1]
                cell = Cell(row, col, args.num_predictors, [])
                if len(grid_cell) > args.num_predictors:
                    Log.warning("Expected to add %s predictors, but we have set num_predictors=%s" % (
                        len(grid_cell), args.num_predictors))
                for p_idx, predictor in enumerate(grid_cell):
                    # try:
                    predictor = np.asarray(predictor)
                    if np.any(np.isnan(predictor)):
                        continue

                    if use_conf and confidence_threshold > 0 and predictor[conf_idx] < confidence_threshold:
                        # Log.debug("Skip cause conf=%f < %f" % (predictor[conf_idx], confidence_threshold))
                        continue

                    try:
                        line = Predictor.from_linesegment(predictor, args.linerep, input_coords=coords,
                                                          is_prediction=only_train_vars, is_offset=args.offset,
                                                          anchor=None if (anchors is None or len(anchors) == 0) else
                                                          anchors[p_idx])
                    except ValueError as e:
                        if allow_nan:
                            Log.warning("We do not add nan")
                            continue
                        else:
                            raise e

                    if not allow_down_facing:
                        line.assert_is_upwards()

                    cell.append(line)
                if len(cell) > 0:
                    grid.insert(cell)
        return grid, []
