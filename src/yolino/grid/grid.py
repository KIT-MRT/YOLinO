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
# from yolino_polyline_detection.data.grid_centerlines import cart_to_euler, oned_to_cart, euler_to_oned, loc_label_to_linepoints, \
# milean_to_cart
import csv
import math
import os
import pickle
import numpy as np
# from yolino.utils.distances_for_eval import line_segment_to_sampled_points
import torch
from matplotlib import pyplot as plt
from shapely.geometry import Polygon

from yolino.grid.cell import Cell
from yolino.grid.coordinates import validate_input_structure
from yolino.grid.predictor import Predictor
from yolino.model.anchors import Anchor
from yolino.model.line_representation import LineRepresentation
from yolino.model.variable_structure import VariableStructure
from yolino.utils.duplicates import LineDuplicates
from yolino.utils.enums import CoordinateSystem, LINE, Variables, ColorStyle, ImageIdx
from yolino.utils.geometry import line_segment_to_sampled_points, reformat2shapely, intersection_box, \
    max_polyline_2_line_distance
from yolino.utils.logger import Log
from yolino.viz.plot import plot_debug_geometry, finish_plot_debug_geometry, plot_debug_geometry_area, get_color

from yolino.model.line_representation import MidDirLines

class Grid:
    def __iter__(self):
        return iter(self.cells)

    def __len__(self):
        length = 0
        for row in self.cells:
            for cell in row:
                if cell is None:
                    continue
                length += len(cell.predictors)
        return length

    def __init__(self, img_height, args):
        self.paths = args.paths
        self.default_img_height = img_height
        self.shape = args.grid_shape
        self.cells = np.empty(args.grid_shape, dtype=Cell)

        self._internal_cell_size = args.cell_size

        self.linerep = args.linerep
        self.num_predictors = args.num_predictors
        self.errors = {"knots_removed": 0, "area_removed": [], "max_distance": {}}

        self.plot = args.plot

    def __str__(self):
        return str(sum([a is not None for a in self.cells.flatten()]))

    def __getitem__(self, item):
        return self.cells.__getitem__(item)

    def to_csv(self, path, coords, image_height=-1):
        lines = self.get_image_lines(coords=coords, image_height=image_height, confidence_threshold=0)
        Log.info("Write CSV to file://%s" % path)
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['start_x', 'end_x', 'end_y', 'start_y', 'class', 'confidence']
            writer = csv.writer(csvfile)

            writer.writerow(fieldnames)

            for instance in lines:
                for line in instance:
                    writer.writerow(line)

    def insert(self, cell: Cell, overwrite=False):
        """

        :Cell data: data to be inserted
        """

        if self.cells[cell.row, cell.col] is None or overwrite:
            if len(cell.predictors) > self.num_predictors:
                Log.error("Invalid num predictors!")
                raise ValueError

            self.cells[cell.row, cell.col] = cell
            return True

        return False

    def append(self, row, col, predictor: Predictor, overwrite=False):

        if self.cells[row, col] is None:
            self.cells[row, col] = Cell(row, col, num_predictors=self.num_predictors, predictors=[predictor])
        else:
            if len(self.cells[row, col]) == self.num_predictors and not overwrite:
                Log.error("Cell is already full")
                raise ValueError("Cell is already full")

            self.cells[row, col].append(predictor)
        return True

    def get_image_points(self, sample_distance, coords, image_height=-1, confidence_threshold=-1):
        """
        get points in uv coordinates
        :float sample_distance: sample distance in pixels
        :int image_height: height of the expected image; -1 does not scale the image
        """
        scaled_lines = self.get_image_lines(coords=coords, image_height=image_height,
                                            confidence_threshold=confidence_threshold)
        total_points = self.sample_along_lines(sample_distance, scaled_lines, coords=coords)

        return total_points

    def slice_and_straighten_line(self, geom_line, r, c, plot_image=False):
        # input: 
        #   geom_line: flat list or ?x2
        #   r: row index of the box to slice it through
        #   c: col index of the box to slice it through
        # return: 
        #   segments: [x,y], [x,y] for start and end point
        #   geom_box: shapely box, use like xs=geom_box.boundary.xy[0], ys=geom_box.boundary.xy[1])

        cs = self._internal_cell_size

        geom_line = reformat2shapely(geom_line)
        if geom_line.length <= 2:
            Log.info(f"The line {geom_line} is only {geom_line.length}px in length.")
            return [], None

        geom_intersection, geom_box = intersection_box((r, c), cs[0], cs[1], geom_line)

        segments = []
        if not geom_intersection.is_empty:
            segments = self.handle_geometry_type(c, cs, geom_box, geom_intersection, geom_line, plot_image, r)
        else:
            if geom_line.within(geom_box):
                # remove knots of the polylines within a cell by taking only first and last point of the box-crop
                # TODO: use some interpolation?
                segments = np.asarray([[[geom_line.coords.xy[0][0], geom_line.coords.xy[1][0]],
                                        [geom_line.coords.xy[0][-1], geom_line.coords.xy[1][-1]]]])

                Log.warning(
                    "There was no intersection with the box (%s) - is the line (%s) still straightened? Debug image might be dumped to tmp/straight.png" % (
                        geom_box, geom_line))
                plot_debug_geometry(geom_box, geom_line, segments[0])
                finish_plot_debug_geometry()

        if len(segments) > 3:
            Log.error(f"You should only have two segments in a cell, "
                             f"but {int(r / 32)}x{int(c / 32)} / {int(r)}x{int(c)} has {len(segments)}")

        return segments, geom_box

    def handle_geometry_type(self, c, cs, geom_box, geom_intersection, geom_line, plot_image, r):
        cell_string = f"{r},{c}"

        segments = []
        # TODO calc straightening error
        # TODO rather use linear approximation instead of just removing knots
        # TODO return the small chunk remaining after slice at the end of a line as we do not extrapolate or cut here!
        if geom_intersection.type == 'MultiLineString':
            segments = np.zeros((len(geom_intersection.geoms), 2, 2), dtype=float)
            for idx, geom_intersection_line in enumerate(geom_intersection.geoms):
                # remove knots of the polylines within a cell by taking only first and last point of the box-crop
                segments[idx] = np.asarray(
                    [[geom_intersection_line.coords.xy[0][0], geom_intersection_line.coords.xy[1][0]],
                     [geom_intersection_line.coords.xy[0][-1], geom_intersection_line.coords.xy[1][-1]]])
                self.errors["knots_removed"] += len(geom_intersection_line.coords) - 2
                np_inters = np.asarray(geom_intersection_line.coords.xy).transpose()
                max_distance = max_polyline_2_line_distance(polyline=np_inters, line_p1=segments[idx][0],
                                                            line_p2=segments[idx][1])

                if cell_string not in self.errors["max_distance"]:
                    self.errors["max_distance"][cell_string] = []
                self.errors["max_distance"][cell_string].append(max_distance)
                if plot_image:
                    plot_debug_geometry(geom_box, geom_line, geom_intersection_line)
            if plot_image:
                finish_plot_debug_geometry()
        elif geom_intersection.type == 'LineString':
            # remove knots of the polylines within a cell by taking only first and last point of the box-crop
            segments = np.asarray(
                [[[geom_intersection.coords.xy[0][0], geom_intersection.coords.xy[1][0]], [
                    geom_intersection.coords.xy[0][-1], geom_intersection.coords.xy[1][-1]]]])

            if plot_image:
                plot_debug_geometry(geom_box, geom_line, segments[0])
                finish_plot_debug_geometry()

            if len(geom_intersection.coords) > 2:
                self.errors["knots_removed"] += len(geom_intersection.coords) - 2

                np_inters = np.asarray(geom_intersection.coords.xy).transpose()
                max_distance = max_polyline_2_line_distance(polyline=np_inters, line_p1=segments[0, 0],
                                                            line_p2=segments[0, 1])

                if cell_string not in self.errors["max_distance"]:
                    self.errors["max_distance"][cell_string] = []
                self.errors["max_distance"][cell_string].append(max_distance)

                polygon = Polygon([*list(geom_intersection.coords),
                                   (
                                       geom_intersection.coords.xy[0][0],
                                       geom_intersection.coords.xy[1][0])])

                self.errors["area_removed"].append(polygon.area)
                if polygon.area > 5 and plot_image:
                    plot_debug_geometry_area(
                        c, cs, r, polygon, geom_box, geom_line, True,
                        title="Straightening error area = %s" % polygon.area)
        elif geom_intersection.type == 'Point' or geom_intersection.type == "MultiPoint":
            self.errors["knots_removed"] += 1

            if plot_image:
                plot_debug_geometry(geom_box, geom_line, None)
                finish_plot_debug_geometry()

            Log.debug("A line only touched a grid cell in a single point. We did not add this to the cell.")
        elif geom_intersection.type == 'GeometryCollection':
            if len(geom_intersection.geoms) > 2:
                Log.error(f"We have a lot of slices within a cell! Please check cell {r},{c}")
            if plot_image:
                plot_debug_geometry(geom_box, geom_line, None)
                finish_plot_debug_geometry()

            Log.debug("We found a collection of %s" % [a.type for a in geom_intersection.geoms])
            for idx, geom_intersection_line in enumerate(geom_intersection.geoms):
                s = self.handle_geometry_type(c, cs, geom_box, geom_intersection_line, geom_line, plot_image, r)
                if len(s) > 0:
                    if len(segments) == 0:
                        segments = s
                    else:
                        segments = np.concatenate([segments, s])
        else:
            raise AttributeError(
                "Invalid intersection type received from shapely: %s" % geom_intersection.type)

        plt.close("all")
        plt.clf()
        return segments

    @classmethod
    def sample_along_lines(cls, sample_distance, scaled_lines, coords, return_variables=True):
        validate_input_structure(scaled_lines, coordinate=CoordinateSystem.UV_SPLIT)
        num_angle_coords = int(coords[Variables.GEOMETRY] / 2.) + 1 \
                           + (len(coords.get_position_of_except([Variables.GEOMETRY])) if return_variables else 0)
        total_points = np.empty((0, num_angle_coords), dtype=float)
        for instance in scaled_lines:
            for line in instance:
                if line[0] == line[2] and line[1] == line[3]:
                    Log.debug("Found line of length 0")
                    continue
                # TODO: class id is missing!
                geom_indices = coords.get_position_of(Variables.GEOMETRY)

                if return_variables:
                    variable_indices = coords.get_position_of_except([Variables.GEOMETRY])
                else:
                    variable_indices = []
                points = line_segment_to_sampled_points(line_segment=line[geom_indices],
                                                        variables=line[variable_indices],
                                                        sample_distance=sample_distance)
                # print("%s vs %s" % (np.shape(total_points), np.shape(points)))
                total_points = np.concatenate([total_points, points])
        return total_points

    def numpy(self, coords: VariableStructure, init=np.nan):
        array = np.ones((self.shape[0], self.shape[1], self.num_predictors, coords.get_length()),
                        dtype=float) * init

        for row, row_cells in enumerate(self.cells):
            for col, cell in enumerate(row_cells):
                if cell is None:
                    continue
                for id, predictor in enumerate(cell):
                    array[row, col, id] = predictor.numpy(coords=coords)
        return array

    def tensor(self, convert_to_lrep: LINE, coords: VariableStructure, fill_nan=True, one_hot=True, set_conf=-1,
               cuda="cpu", anchors: Anchor = None, line_as_is=True, as_offset=False,
               duplicates: LineDuplicates = None, filename=None, ignore_duplicates=False, store_lines=False,
               pkl_file=None):

        # fill with dummy cell containing only background class
        dummy_line = torch.zeros(coords.get_length(one_hot=one_hot), dtype=torch.float32, device=cuda)
        if fill_nan:
            dummy_line.fill_(np.nan)

        if Variables.CLASS in coords and coords[Variables.CLASS] > 0:
            background = torch.tensor([1, *torch.zeros(coords[Variables.CLASS] - 1)], dtype=torch.float32,
                                      device=cuda) if one_hot else [0]
            dummy_line[coords.get_position_of(Variables.CLASS, one_hot)] = background
            del background

        if Variables.CONF in coords and coords[Variables.CONF] > 0:
            # set conf=0 initially so all predictors will have no confidence
            # only actual gt lines should have a confidence of 1
            dummy_line[coords.get_position_of(Variables.CONF, one_hot)] = 0

        array = torch.tile(dummy_line, (self.shape[0] * self.shape[1], self.num_predictors, 1))
        del dummy_line

        converter = LineRepresentation.get(coords.line_representation.enum)

        ax_anchor, ax_gt, ax_gt_dupl, colors, cs_4d = self.prepare_plot(anchors)
        self.add_anchor_to_plot(anchors, ax_anchor, colors, converter, cs_4d, [0, 0, 0, 0])

        for row, row_cells in enumerate(self.cells):
            for col, cell in enumerate(row_cells):

                top_left_corner = [row * cs_4d[0], col * cs_4d[1]]
                tlc_4d = np.asarray([*top_left_corner, *top_left_corner])

                if cell is None:
                    continue

                cell_idx = row * self.shape[1] + col

                for idx, predictor in enumerate(cell):

                    indices = [idx]
                    t_predictor = predictor.tensor(coords=coords, convert_to_lrep=convert_to_lrep,
                                                   one_hot=one_hot, set_conf=set_conf,
                                                   extrapolate=not line_as_is)
                    if anchors is not None and len(anchors) > 0:
                        indices, offset = anchors.get_specific_anchors(t_predictor)
                        if as_offset:
                            t_predictor = offset
                        else:
                            t_predictor = torch.tile(t_predictor, (len(indices), 1))
                    else:
                        t_predictor = t_predictor.unsqueeze(0)

                    if store_lines and convert_to_lrep == LINE.POINTS:
                        np_p = t_predictor[0, coords.get_position_of(Variables.GEOMETRY)].numpy()
                        dict_data = {"x_s": np_p[0], "y_s": np_p[1], "x_e": np_p[2], "y_e": np_p[3]}
                        dict_data["mx"], dict_data["my"], dict_data["dx"], dict_data["dy"] = MidDirLines.from_cart(
                            start=np_p[0:2], end=np_p[2:4])

                        Log.info(f"Write to {pkl_file}")
                        with open(pkl_file, "ab") as f:
                            pickle.dump(dict_data, f, protocol=2)

                    if len(indices) > 1:
                        Log.info("We have more than one anchor matching a line.")
                    for p_idx, final_index in enumerate(indices):
                        if False:
                            anchors.add_heatmap(row, col, final_index)
                        if not array[cell_idx, final_index, 0].isnan():
                            msg = "We have more than one GT element per anchor %s:\n" % anchors[final_index].numpy()
                            msg += "%s-Anchors are %s\n" % (anchors.args.anchor_vars, anchors)
                            msg += "Already in [%d, %d]: %s\n " \
                                   % (row, col, array[cell_idx, final_index, 0:4])
                            msg += "New one: %s" % (predictor.tensor(coords=coords, convert_to_lrep=convert_to_lrep))

                            Log.info(msg)
                            if duplicates is not None and not ignore_duplicates:
                                duplicates.add(row, col, final_index.item())

                            self.add_gt_to_plot(ax_gt=ax_gt_dupl, line=t_predictor[p_idx, 0:4],
                                                color=colors[final_index], tlc_4d=tlc_4d, converter=converter,
                                                cs_4d=cs_4d, linerep=anchors.linerep, anchors=anchors, idx=final_index)
                            continue

                        if duplicates and not ignore_duplicates:
                            duplicates.add_ok()
                        array[cell_idx, final_index] = t_predictor[p_idx]

                        if anchors is not None:
                            self.add_gt_to_plot(ax_gt=ax_gt, line=t_predictor[p_idx, 0:4], color=colors[final_index],
                                                tlc_4d=tlc_4d, converter=converter, cs_4d=cs_4d,
                                                linerep=anchors.linerep, anchors=anchors, idx=final_index)

        if duplicates and not ignore_duplicates and (
                len(duplicates) > 15 or duplicates.height()[2] >= self.shape[0] / 3.):
            self.finish_anchor_plot(anchors, [ax_anchor, ax_gt, ax_gt_dupl], self._internal_cell_size,
                                    filename=filename, duplicates=duplicates)
        return array

    def finish_anchor_plot(self, anchors, axs, cs, filename=None, duplicates=None):
        if self.plot and anchors is not None:
            major_yticks = np.arange(0, self.default_img_height, cs[0])

            startx, endx = axs[1].get_xlim()
            int_start = math.floor(startx / cs[1]) * cs[1]
            int_end = math.ceil(endx / cs[1]) * cs[1]
            major_xticks = np.arange(int_start, int_end + cs[1], cs[1])

            for ax in axs[1:]:
                ax.set_ylim((0, self.default_img_height))
                ax.yaxis.set_ticks(major_yticks)

                ax.set_xlim((startx, endx))
                ax.xaxis.set_ticks(major_xticks)
                ax.grid(which='major')
                ax.invert_yaxis()
                ax.set_aspect('equal', adjustable='box')

            axs[0].invert_yaxis()
            axs[0].set_aspect('equal', adjustable='box')
            axs[0].set_title(f"{anchors.args.anchor_vars} Anchors")
            axs[0].legend()
            axs[1].set_title("Generated GT")
            axs[2].set_title(str(duplicates))
            plt.tight_layout()

            if filename:
                plt.suptitle(filename)
            # path = "/tmp/anchors.png"
            path = self.paths.generate_anchor_image_file_path(file_name=filename, **anchors.args.__dict__)
            Log.warning(f"Save anchor debug to file://{path}", level=1)
            Log.plt(epoch=0, fig=plt, tag=os.path.join(str(ImageIdx.ANCHOR), filename))
            plt.savefig(path)

    def add_anchor_to_plot(self, anchors, ax_anchor, colors, converter, cs_4d, tlc_4d):
        if self.plot and anchors is not None and len(anchors) > 0:
            for a_idx, a in enumerate(anchors):
                if anchors.linerep == LINE.POINTS:
                    cart = tlc_4d + np.asarray(a) * cs_4d
                else:
                    cart = tlc_4d + np.asarray(converter.to_cart(a)) * cs_4d
                ax_anchor.arrow(cart[1], cart[0], cart[3] - cart[1], cart[2] - cart[0], color=colors[a_idx],
                                head_width=1, label=a_idx if np.all(cart < cs_4d[0]) else "", width=0.1)

    def add_gt_to_plot(self, ax_gt, line, color, tlc_4d, converter, cs_4d, linerep, anchors, idx):
        if self.plot and ax_gt:

            if anchors.offset:
                line += anchors[idx]

            if linerep == LINE.POINTS:
                cart = tlc_4d + np.asarray(line) * cs_4d
            else:
                cart = tlc_4d + np.asarray(converter.to_cart(line, raise_error=False)) * cs_4d

            ax_gt.arrow(cart[1], cart[0], cart[3] - cart[1], cart[2] - cart[0], color=color, head_width=2, width=1)

    def prepare_plot(self, anchors):
        cs = self._internal_cell_size
        cs_4d = np.asarray([*cs, *cs])

        if self.plot and anchors is not None and len(anchors) > 0:
            fig, (ax_anchor, ax_gt, ax_gt_dupl) = plt.subplots(1, 3, figsize=(20, 10), dpi=200)

            converter = LineRepresentation.get(anchors.linerep)

            colors = np.stack(
                [get_color(ColorStyle.ANCHOR, idx=a_idx, anchors=anchors) for a_idx in range(len(anchors))]) / 255.
            return ax_anchor, ax_gt, ax_gt_dupl, colors, cs_4d
        else:
            return None, None, None, np.zeros(self.num_predictors), cs_4d

    def get_row_col_for_uv_lines(self, confidence_threshold=-1):
        array = np.empty((0, 2), dtype=float)
        for row, row_cells in enumerate(self.cells):
            for col, cell in enumerate(row_cells):
                if cell is None:
                    continue
                for idx, predictor in enumerate(cell.predictors):
                    if predictor.confidence < confidence_threshold:
                        continue
                    array = np.vstack([array, [row, col]])

        return array

    def get_cluster_ids(self, confidence_threshold=-1):
        array = np.empty((0), dtype=float)
        for row, row_cells in enumerate(self.cells):
            for col, cell in enumerate(row_cells):
                if cell is None:
                    continue
                for idx, predictor in enumerate(cell.predictors):
                    if predictor.confidence < confidence_threshold:
                        continue
                    array = np.append(array, predictor.id)

        return array

    def get_image_lines(self, coords: VariableStructure, image_height=-1, confidence_threshold=-1,
                        coordinates=CoordinateSystem.UV_SPLIT,
                        is_training_data=False, round_to_int=True):

        # Output should have (batch, line_segments, 2 * 2 + ?)
        if coordinates != CoordinateSystem.UV_SPLIT:
            raise NotImplementedError

        if coords.line_representation != LINE.POINTS:
            coords = coords.clone(LINE.POINTS)
            # raise ValueError("Image lines should only be put into points representation.")

        if image_height > 0:
            scale = self.get_cell_size(image_height)
        else:
            scale = self._internal_cell_size

        if is_training_data:
            length = coords.num_vars_to_train()
        else:
            length = coords.get_length()

        array = np.empty((0, length), dtype=float)
        rows = 0
        cols = 0
        # predictors = 0
        for row, row_cells in enumerate(self.cells):
            rows += 1
            for col, cell in enumerate(row_cells):
                cols += 1
                if cell is None:
                    continue
                v = row * scale[0]
                h = col * scale[1]
                # len(cell.predictors)
                for idx, predictor in enumerate(cell.predictors):
                    # predictors += 1

                    if predictor.confidence < confidence_threshold:
                        continue

                    val = np.empty((length), dtype=float)

                    if coords[Variables.GEOMETRY] > 0:
                        if round_to_int:
                            round = np.round
                        else:
                            round = lambda a: a

                        if is_training_data and Variables.GEOMETRY in coords.train_vars():
                            val[coords.get_position_within_prediction(Variables.GEOMETRY)] = [
                                round(v + predictor.start[0] * scale[0]),
                                round(h + predictor.start[1] * scale[0]),
                                round(v + predictor.end[0] * scale[0]),
                                round(h + predictor.end[1] * scale[1])]
                        elif not is_training_data:
                            val[coords.get_position_of(Variables.GEOMETRY)] = [
                                round(v + predictor.start[0] * scale[0]),
                                round(h + predictor.start[1] * scale[0]),
                                round(v + predictor.end[0] * scale[0]),
                                round(h + predictor.end[1] * scale[1])]

                    if coords[Variables.CLASS] > 0:
                        if is_training_data and Variables.CLASS in coords.train_vars():
                            val[coords.get_position_within_prediction(Variables.CLASS)] = predictor.label
                        elif not is_training_data:
                            val[coords.get_position_of(Variables.CLASS)] = predictor.label

                    if coords[Variables.CONF] > 0:
                        if is_training_data and Variables.CONF in coords.train_vars():
                            val[coords.get_position_within_prediction(Variables.CONF)] = predictor.confidence
                        elif not is_training_data:
                            val[coords.get_position_of(Variables.CONF)] = predictor.confidence

                    array = np.vstack([array, val])

        # Output should have (batch, line_segments, 2 * 2 + ?)
        array = np.expand_dims(array, 0)
        validate_input_structure(array, coordinates)
        return array

    def get_cell_size(self, image_height=-1):
        if image_height > 0 and image_height != self.default_img_height:
            return np.round(
                np.multiply(
                    np.divide(image_height, self.default_img_height),
                    self._internal_cell_size),
                0).astype(int)
        else:
            return self._internal_cell_size

    def get_cell_diagonal_length(self, image_height=-1):
        cell_size = self.get_cell_size(image_height)
        return math.sqrt(2 * cell_size * cell_size)

    def get_cell_id_range(self, row_col, axis=0):
        assert (np.shape(row_col) == (2, 2))

        if row_col[0, axis] <= row_col[1, axis]:
            range_step = 1
        else:
            range_step = -1

        range_start = row_col[0, axis]
        range_end = row_col[1, axis] + range_step
        return range(range_start, range_end, range_step)

    def get_cell_ids(self, row_col, axis=0):
        return list(range(self.get_cell_id_range(row_col, axis)))

    def get_unique_cell_id_pairs(self, row_col):
        # Log.info("Row_col with shape %s and data %s" % (row_col.shape, row_col.flatten()))
        pairs = self.get_cell_id_pairs(row_col)
        # Log.info("Pairs with shape %s and data %s" % (pairs.shape, pairs.flatten()))
        try:
            return np.unique(pairs, axis=0)
        except ValueError as e:
            raise ValueError("Exception %s\nPairs %s, %s\n Row_col %s, %s" % (
                str(e), pairs.shape, pairs.flatten(), row_col.shape, row_col.flatten()))

    def get_cell_id_pairs(self, row_col):
        pairs = np.empty((0, 2), float)
        for idx in range(len(row_col) - 1):
            rows = list(self.get_cell_id_range(row_col[idx:idx + 2], axis=0))
            cols = list(self.get_cell_id_range(row_col[idx:idx + 2], axis=1))
            # Log.info("%s, %s" % (str(rows), str(cols)))
            pairs = np.vstack([pairs, np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)])
        return pairs

    def get_cell_corner_pairs_uv(self, row_col):
        pairs = self.get_cell_id_pairs(row_col)
        pairs = pairs * self._internal_cell_size
        return pairs

    def get_unique_cell_corner_pairs_uv(self, row_col):
        pairs = self.get_unique_cell_id_pairs(row_col)
        pairs = pairs * self._internal_cell_size
        return pairs

    def iterate_cell_ids(self, row_col, axis=0):
        range_start, range_end, range_step = self.get_cell_id_range(row_col, axis)
        return range(range_start, range_end, range_step)

    def iterate_cell_corners_uv(self, row_col, axis=0):
        range_start, range_end, range_step = self.get_cell_id_range(row_col, axis)
        return range(range_start * self._internal_cell_size[axis], range_end * self._internal_cell_size[axis],
                     range_step * self._internal_cell_size[axis])

    def get_position_of_line_segment(self, line, swap=False):
        """
        line as [x,y],[x,y],[...]
        :(row_col, position_in_cell, portion_in_cell):
            row_col: row and column numbers of line as int for all points; e.g. (5,16), (6,17)
            position_in_cell: pixel coordinates in cell (2,2) as int; always smaller than cell_size; e.g. (16,16) -> (32,32)
            portion_in_cell: proportion coordinates of the cell (2,2) as float; e.g. (0.5,0.5) -> (1,1)
        """
        if len(np.shape(line)) != 2:
            raise ValueError("Line %s should only have 2 dimensions!" % line)
        assert (np.shape(line)[1] == 2)
        line = np.array(line)

        repeated_cs = self._internal_cell_size
        row_col = np.floor(np.divide(line, repeated_cs)).astype(int)
        position_in_cell = np.mod(line, repeated_cs)
        portion_in_cell = np.divide(np.mod(line, repeated_cs), repeated_cs)
        corner_cases = np.zeros_like(line, dtype=bool)

        for p_idx in range(len(line)):
            if np.any(position_in_cell[p_idx] == 0):

                for idx in range(len(position_in_cell[p_idx])):
                    # for axis in range(len(position_in_cell[idx])):
                    p = position_in_cell[p_idx][idx]

                    # p is on border
                    if p == 0:

                        # it is the right/bottom outer border and we can use the previous cell index
                        if row_col[p_idx][idx] > row_col[(p_idx + 1) % 2][idx]:
                            row_col[p_idx][idx] -= 1
                            position_in_cell[p_idx][idx] = self._internal_cell_size[idx % 2]
                            portion_in_cell[p_idx][idx] = 1

                            corner_cases[p_idx][idx] = True

                # if corner_cases.any():
                #     print("Found a corner case at %s with pixels\n%s\n%s" % (row_col, line.flatten(), position_in_cell.flatten()))

        # limit to grid range
        row_col[row_col < 0] = 0
        for idx in range(len(row_col)):
            for axis in range(len(row_col[idx])):
                if row_col[idx][axis] >= self.shape[axis % 2]:
                    # Log.warning("Limit value to grid range. Before: %s for grid of shape %s"
                    #             % (row_col[idx], str(self.shape)))
                    row_col[idx][axis] = self.shape[axis % 2] - 1
                    # Log.warning("Limit value to grid range. After: %s for grid of shape %s"
                    #             % (row_col[idx], str(self.shape)))
        return row_col, position_in_cell, portion_in_cell

    def add_single_slice_uv(self, line_segment_uv, variables, row, col, coords,
                            allow_out_of_bounds=False, allow_nan=False, variables_have_conf=True,
                            allow_down_facing=True):
        variables = np.asarray(variables)
        row_col, _, line_segment_grid_coords = self.get_position_of_line_segment(line_segment_uv)
        row, col = row_col[0]

        if not allow_out_of_bounds and (row >= self.shape[0] or col >= self.shape[1]):
            Log.warning("Segment out of range %s" % line_segment_uv)
            return

        try:
            data = [line_segment_grid_coords.flatten(), variables.flatten()]

            if not variables_have_conf and Variables.CONF in coords and coords[Variables.CONF] > 0:
                data.append([1])
            line_segment_grid_coords = Predictor.from_linesegment(np.concatenate(data), LINE.POINTS,
                                                                  input_coords=coords)
        except ValueError as e:
            if not allow_nan:
                raise e

        if not allow_down_facing:
            line_segment_grid_coords.assert_is_upwards()

        if Variables.CLASS in coords.train_vars() and np.sum(line_segment_grid_coords.label) != 1:
            raise ValueError("The classes are not one hot! input: %s, predictor: %s" % (
                str(variables.flatten()), str(line_segment_grid_coords.label)))

        if self.cells[row, col] is None:
            data = Cell(row, col, self.num_predictors, [line_segment_grid_coords])
            self.insert(data, overwrite=True)
        else:
            self.cells[row, col].append(line_segment_grid_coords)

    def clear(self, r, c):
        self.cells[r, c] = None

    # def get_uv_for_cell(self, r, c):
    #     self.get_position_of_line_segment()
