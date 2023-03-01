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
import os
from abc import abstractmethod

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from yolino.eval.distances import get_points_distances, get_hungarian_match
from yolino.grid.coordinates import validate_input_structure
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import Variables, CoordinateSystem, ColorStyle, ImageIdx, LINE
from yolino.utils.logger import Log
from yolino.viz.plot import plot


class Matcher:
    def __init__(self, coords: VariableStructure, args, distance_threshold, is_cell_based, no_match_index=-100):
        self.no_match_index = no_match_index
        self.coords = coords
        self.args = args
        self.plot = args.plot
        self.img_count = 0
        self.distance_threshold = distance_threshold
        self.dummy_order = torch.tensor(range(self.args.num_predictors), device=args.cuda)
        self.is_cell_based = is_cell_based

    @abstractmethod
    def match(self, preds, grid_tensor, filenames, confidence_threshold):
        """
        Matches prediction and ground truth in 1:1 manner.
        The distance metric is set with the constructor.

        Returns:
            matched_predictions (torch.tensor):
                Will be filled with list of GT IDs at the position of its matching prediction.
                The remainder is filled with -100 by default.
            matched_gt (torch.tensor):
                Will be filled with a list of prediction IDs at the position of its matching GT.
                The remainder is filled with -100 by default.
        """
        pass

    @staticmethod
    def __resort_by_match_ids__(grid_tensor, matched_predictions):
        # resort matched GT entries and have nan tensors in the remaining places
        _, num_cells, _, num_vars = grid_tensor.shape
        resorted_grid_tensor = torch.ones_like(grid_tensor) * torch.nan
        for e_idx, pred_in_cell in enumerate(matched_predictions):
            batch_idx, cell_idx = (int(e_idx / num_cells), e_idx % num_cells)

            for p_idx, gt_idx in enumerate(pred_in_cell):
                if gt_idx >= 0:
                    resorted_grid_tensor[batch_idx, cell_idx, int(p_idx)] = grid_tensor[
                        batch_idx, cell_idx, int(gt_idx)]
        return resorted_grid_tensor

    def get_matches_in_set(self, pred_subset, gt_subset, distance_metric, filename, use_conf=False):
        """
        Match a subset of the GT with a subset of the prediction as 1:1. A subset can be a cell or a full image.
        If given in the constructor, a distance threshold is applied.
        The distance metric is also specified in the constructor.

        Args:
            pred_subset (torch.tensor):
                with shape [lines, vars]
            gt_subset (torch.tensor):
                with shape [lines, vars]

            E.g. if the Gt with ID=1 and prediction with ID=2 and GT-ID=0 and P-ID=1 are a match,
            'out_pred_matched' would be [-100, 0, 1] and 'out_gt_matched' would be [1, 2, -100].
        Returns:
            out_pred_matches (torch.tensor):
                Output variable filled with the matching GT ID or -100 if there is no match at the position
                of the predictions in the input.
            out_gt_matches (torch.tensor):
                Output variable filled with the matching prediction ID or -100 if there is no match at the position
                of the GT in the input.
        """
        if torch.all(torch.isnan(gt_subset[:, self.coords.get_position_of(Variables.GEOMETRY)])):
            if self.plot:
                self._debug_single_match_plot_(gt=gt_subset.cpu().numpy(), pred=pred_subset.detach().cpu().numpy(),
                                               p_match=np.ones(len(pred_subset), dtype=int) * self.no_match_index)
            # Log.warning("Nothing to match. All GT is %s" % (gt_subset))
            return False, [], []

        if len(pred_subset) == 0:
            raise ValueError("Nothing to match. We have no prediction here!")

        out_pred_matches = torch.ones(len(pred_subset), dtype=torch.int64,
                                      device=pred_subset.device) * self.no_match_index
        out_gt_matches = torch.ones(len(gt_subset), dtype=torch.int64, device=pred_subset.device) * self.no_match_index

        # get ids of valid subset; preds is always valid
        # get distances for subset (result matrix is preds x gt)
        # everything > max value is inf
        # Cost matrix with preds x gt (rowsxcols)
        cost_matrix = get_points_distances(p_cell=pred_subset.detach(), gt_cell=gt_subset,
                                           distance_metric=distance_metric, coords=self.coords,
                                           distance_threshold=self.distance_threshold, use_conf=use_conf,
                                           max_geom_value=1 if self.is_cell_based else self.args.img_size[1])
        if torch.all(torch.isinf(cost_matrix)) \
                or (self.distance_threshold > 0 and torch.all(cost_matrix > self.distance_threshold)) \
                or len(cost_matrix) == 0:
            return False, [], []

        gt_isnan_flags = torch.cat(
            [gt_subset[:, 0].isnan(), torch.ones(len(pred_subset), dtype=bool, device=cost_matrix.device)])
        if sum(~gt_isnan_flags) == 1:
            try:
                match_pred_id = torch.argmin(cost_matrix[:, 0])

                # match is only on the top left section of the matrix.
                # The bottom and most-right parts are placeholders for non-matching.
                if match_pred_id < len(out_pred_matches):
                    gt_id_match = torch.where(~gt_isnan_flags)[0].item()
                    out_pred_matches[match_pred_id] = gt_id_match
                    out_gt_matches[gt_id_match] = match_pred_id
            except IndexError as e:
                Log.error("Distance Metric: %s" % distance_metric)
                Log.error("Pred %s\n%s" % (pred_subset.shape, pred_subset))
                Log.error("GT %s\n%s" % (gt_subset.shape, gt_subset))
                Log.error("Distance Threshold: %s" % self.distance_threshold)
                Log.error("GT is Nan?: %s" % gt_isnan_flags)
                Log.error("Match ID: %s" % match_pred_id)
                Log.error("Cost Matrix: %s\n%s" % (cost_matrix.shape, cost_matrix))
                Log.error("P Matches: %s\n%s" % (out_pred_matches.shape, out_pred_matches))
                Log.error("GT Matches: %s\n%s" % (out_gt_matches.shape, out_gt_matches))
                raise e

        else:
            try:
                _, gt_ids_at_matched_pred_position = get_hungarian_match(cost_matrix)
                if sum(gt_ids_at_matched_pred_position) != np.sum(range(0, len(pred_subset))) and \
                        sum(gt_ids_at_matched_pred_position) != np.sum(range(len(pred_subset) + len(gt_subset))):
                    raise ValueError(
                        "GT matching IDs should contain all values from 0 to %d only once, but we have %s"
                        % (len(pred_subset), out_gt_matches))

                # 1. get indices of predictions where the gt match is not nan
                pred_indices = torch.where(~gt_isnan_flags[gt_ids_at_matched_pred_position][0:len(pred_subset)])[0]
                # 2. set those to the matching GT ID
                out_pred_matches[pred_indices] = gt_ids_at_matched_pred_position[pred_indices]

                # 3. set all gt to their match if they were not nan
                for p_idx, g_idx in enumerate(gt_ids_at_matched_pred_position):
                    if p_idx < len(pred_subset) and g_idx < len(gt_subset) and not gt_subset[g_idx, 0].isnan():
                        out_gt_matches[g_idx] = p_idx

            except ValueError as e2:
                Log.error("%s: Tried CPU match, but did not work" % filename)
                raise e2

        return True, out_gt_matches, out_pred_matches

    def _debug_single_match_plot_(self, gt, pred, p_match, suffix=""):
        if not self.plot:
            return
        if self.img_count > 10 or len(gt) > 100 or len(pred) > 100:
            return
        if sum(p_match >= 0) < 1:
            return

        fn = 0
        fp = 0
        tp = 0
        tn = 0

        self.img_count += 1
        cmap = plt.get_cmap("jet")  # "gist_ncar")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max(p_match))  # self.args.num_predictors)
        scalarMap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        gt_color = (1, 0, 0.5, 0.5)
        gt_color_full = (1, 0, 0.5, 1)
        tp_color = (35 / 255, 222 / 255, 91 / 255, 1)
        fp_color = (35 / 255, 101 / 255, 222 / 255, 1)
        fn_color = (41 / 255, 0, 148 / 255, 1)
        fig, axs = plt.subplots(3, sharey=True, sharex=True)

        # plot GT below everything else
        for gt_idx, gt_entry in tqdm(enumerate(gt), total=len(gt)):
            gt_cartesian = self.coords.line_representation.to_cart(gt_entry)
            xs = gt_cartesian[[0, 2]]
            ys = gt_cartesian[[1, 3]]
            axs[0].plot(ys, xs, label="gt_" + str(gt_idx), color=gt_color)
            axs[1].plot(ys, xs, label="gt_" + str(gt_idx), color=gt_color)
            axs[2].plot(ys, xs, label="gt_" + str(gt_idx), color=gt_color_full)

        # plot false positives next
        for p_idx, gt_idx in tqdm(enumerate(p_match)):
            p_cartesian = self.coords.line_representation.to_cart(pred[int(p_idx)])
            xs = p_cartesian[[0, 2]]
            ys = p_cartesian[[1, 3]]
            if gt_idx < 0:
                if pred[int(p_idx)][self.coords.get_position_within_prediction(Variables.CONF)] > 0.5:
                    color = fp_color
                    axs[1].plot(ys, xs, label="pred_" + str(p_idx), color=color)
                    axs[0].plot(ys, xs, label="pred_" + str(p_idx), color=color)
                    fp += 1
                else:
                    tn += 1
                    continue

        # plot true positives on top
        for p_idx, gt_idx in tqdm(enumerate(p_match)):
            p_cartesian = self.coords.line_representation.to_cart(pred[int(p_idx)])
            xs = p_cartesian[[0, 2]]
            ys = p_cartesian[[1, 3]]
            if gt_idx >= 0:
                if Variables.CONF not in self.coords.train_vars() \
                        or pred[int(p_idx)][self.coords.get_position_within_prediction(Variables.CONF)] > 0.5:
                    tp += 1
                    color = tp_color
                    axs[0].plot(ys, xs, label="pred_" + str(p_idx), color=color)
                else:
                    fn += 1
                    color = fn_color
                    axs[1].plot(ys, xs, label="pred_" + str(p_idx), color=color)

        legend_elements = [Line2D([0], [0], color=tp_color, lw=4, label='TP'),
                           Line2D([0], [0], color=fp_color, lw=4, label='FP'),
                           Line2D([0], [0], color=fn_color, lw=4, label='FN')]
        axs[0].legend(handles=legend_elements)
        axs[0].set_title("All conf > t; TP: %s, FP: %s" % (tp, fp))
        axs[1].set_title("All false; FN: %s, FP: %s" % (fn, fp))
        axs[2].set_title("Ground Truth")
        plt.gca().invert_yaxis()

        path = os.path.join(self.args.paths.debug_folder, "pred_gt_match_%s.png" % suffix)
        Log.warning("Store prediction gt match debug image file://%s" % path, level=1)
        plt.savefig(path)
        plt.close(fig)

    def _debug_full_match_plot_(self, epoch, preds, grid_tensor, filenames, coordinates=CoordinateSystem.CELL_SPLIT,
                                tag="dummy_matching", anchors=None, idx: ImageIdx = ImageIdx.MATCH):

        if not self.plot or self.coords[Variables.GEOMETRY] == 0:
            return

        validate_input_structure(preds[[0]], coordinates)
        validate_input_structure(torch.unsqueeze(grid_tensor[0], dim=0), coordinates)

        for i in [0]:  # range(len(preds)):
            tmp_preds = torch.clone(preds).cpu().detach()
            scale = 4
            if coordinates != CoordinateSystem.UV_SPLIT:

                from yolino.grid.grid_factory import GridFactory
                full_pred_grid, _ = GridFactory.get(tmp_preds[[i]], [],
                                                    coordinate=coordinates,
                                                    args=self.args, input_coords=self.coords, only_train_vars=True,
                                                    anchors=anchors)

                grid, _ = GridFactory.get(grid_tensor[[i]].cpu(), [],
                                          coordinate=coordinates, anchors=anchors,
                                          args=self.args, input_coords=self.coords)

                # set all predictions to nan that have no match in the GT (= nan in GT)
                tmp_preds[torch.sum(torch.isnan(grid_tensor[:, :, :, 0:4]), dim=-1) == 4] = torch.nan

                pred_grid, _ = GridFactory.get(tmp_preds[[i]], [],
                                               coordinate=coordinates, anchors=anchors,
                                               args=self.args, input_coords=self.coords, only_train_vars=True)

                points_coords = self.coords.clone(LINE.POINTS)
                full_pred_uv_lines = full_pred_grid.get_image_lines(coords=points_coords,
                                                                    image_height=self.args.img_size[0] * scale,
                                                                    is_training_data=True)
                gt_uv_lines = grid.get_image_lines(coords=points_coords, image_height=self.args.img_size[0] * scale)
                matched_pred_uv_lines = pred_grid.get_image_lines(coords=points_coords,
                                                                  image_height=self.args.img_size[0] * scale,
                                                                  is_training_data=True)
            else:
                full_pred_uv_lines = torch.clone(tmp_preds[[i]]) * scale
                tmp_preds[i][torch.sum(torch.isnan(grid_tensor[i, :, 0:4]), dim=-1) == 4] = torch.nan
                gt_uv_lines = grid_tensor[[i]] * scale
                matched_pred_uv_lines = tmp_preds[[i]] * scale

            img = torch.ones((3, self.args.img_size[0] * scale, self.args.img_size[1] * scale), dtype=torch.float32)

            path = self.args.paths.generate_debug_image_file_path(file_name=filenames[i], idx=idx,
                                                                  suffix=tag + "_gray_pred")
            img, ok = plot(full_pred_uv_lines[[0]], path, img, coords=self.coords, show_grid=True,
                           cell_size=(int(self.args.cell_size[0] * scale), int(self.args.cell_size[1] * scale)),
                           threshold=0,
                           colorstyle=ColorStyle.CONFIDENCE_BW if Variables.CONF in self.coords.vars_to_train else ColorStyle.UNIFORM,
                           color=None if Variables.CONF in self.coords.vars_to_train else (1, 1, 1),
                           coordinates=CoordinateSystem.UV_SPLIT,
                           imageidx=idx, epoch=epoch, training_vars_only=True)

            path = self.args.paths.generate_debug_image_file_path(file_name=filenames[i], idx=idx,
                                                                  suffix=tag + "_gray_pred_color")
            img, ok = plot(gt_uv_lines[[0]], path, img, coords=self.coords, show_grid=False,
                           cell_size=(int(self.args.cell_size[0] * scale), int(self.args.cell_size[1] * scale)),
                           threshold=0, colorstyle=ColorStyle.ID, coordinates=CoordinateSystem.UV_SPLIT,
                           imageidx=idx, thickness=2, epoch=epoch)

            path = self.args.paths.generate_debug_image_file_path(file_name=filenames[i], idx=idx, suffix=tag)
            img, ok = plot(matched_pred_uv_lines[[0]], path, img, coords=self.coords, show_grid=False,
                           cell_size=(int(self.args.cell_size[0] * scale), int(self.args.cell_size[1] * scale)),
                           threshold=0, colorstyle=ColorStyle.ID, coordinates=CoordinateSystem.UV_SPLIT, tag=tag,
                           imageidx=idx, thickness=6, epoch=epoch,
                           training_vars_only=True)
        return path
