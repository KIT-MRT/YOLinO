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
import timeit

import numpy as np
import torch

from yolino.eval.matcher import Matcher
from yolino.grid.coordinates import validate_input_structure
from yolino.model.line_representation import PointsLines
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import Variables, CoordinateSystem, Distance
from yolino.utils.logger import Log


class UVMatcher(Matcher):
    def __init__(self, coords: VariableStructure, args, distance_threshold: int = -1):

        if not coords[Variables.GEOMETRY] == PointsLines().num_params:
            raise ValueError("The given coordinates are not specified for the correct line representation. "
                             "We expect points for the matching.")

        super().__init__(coords, args, distance_threshold, is_cell_based=False)
        self.distance_metric = self.args.association_metric

    def match(self, preds: torch.tensor, grid_tensor: list, filenames, confidence_threshold=0):
        """
        Match GT with prediction as 1:1. If given in the constructor, a distance threshold is applied.
        The distance metric is also specified in the constructor.

        Args:
            preds (torch.tensor):
                in uv with shape [batch, lines, vars]
            grid_tensor (list of torch.tensor):
                in uv with shape [batch, lines, vars]

        Returns:
            matched_preds (torch.tensor): Will always have shape [batch, number of lines] and the remainder is
                filled with -100.
            matched_gt (torch.tensor): Will always have shape [batch, number of lines] and the remainder is filled
                with -100.

        """
        start = timeit.default_timer()
        # (batch, line_segments, 2 * 2 + ?)
        num_batch, num_lines, _ = preds.shape

        matched_preds = torch.ones((num_batch, num_lines), dtype=torch.int64,
                                   device=preds.device) * self.no_match_index
        matched_gt = torch.ones((num_batch, num_lines), dtype=torch.int64, device=preds.device) * self.no_match_index

        # only if there are labels
        if torch.all(torch.tensor([len(g) for g in grid_tensor]) == 0):
            return matched_preds, matched_gt

        Log.info("Waiting for uv match...")
        for b_idx, batch in enumerate(preds):
            validate_input_structure(preds[[b_idx]], CoordinateSystem.UV_SPLIT)
            validate_input_structure(torch.unsqueeze(grid_tensor[b_idx], dim=0), CoordinateSystem.UV_SPLIT)

            Log.debug("UV match for image %d pred %s and gt %s"
                      % (b_idx, str(preds[b_idx].shape), str(grid_tensor[b_idx].shape)))
            Log.debug("%s" % grid_tensor[b_idx][0:5])

            ids = torch.where(preds[b_idx, :, -1] >= confidence_threshold)[0]

            if len(ids) > 0:
                confident_predictions = preds[b_idx, ids]
                ok, g, p = self.get_matches_in_set(confident_predictions, grid_tensor[b_idx],
                                                   distance_metric=self.distance_metric,
                                                   filename=filenames[b_idx], use_conf=False)

                if not ok:
                    continue

                matched_gt[b_idx][0:len(grid_tensor[b_idx])] = torch.tensor(
                    [ids[i] if i >= 0 else torch.tensor(self.no_match_index)
                    for i in g])
                matched_preds[b_idx, ids] = p

            if torch.any(matched_gt[b_idx, 0:len(grid_tensor[b_idx])] == self.no_match_index):
                inconfident_prediction_ids = torch.where(preds[b_idx, :, -1] < confidence_threshold)[0]
                inconfident_predictions = preds[b_idx, inconfident_prediction_ids]

                if len(inconfident_predictions) > 0:
                    unmatched_gt_ids = torch.where(matched_gt[b_idx, 0:len(grid_tensor[b_idx])] == self.no_match_index)
                    ok, g, p = self.get_matches_in_set(inconfident_predictions, grid_tensor[b_idx][unmatched_gt_ids],
                                                       distance_metric=self.distance_metric, filename=filenames[b_idx],
                                                       use_conf=False)
                    if not ok:
                        continue
                    matched_gt[b_idx][unmatched_gt_ids] = torch.tensor(
                        [inconfident_prediction_ids[i] if i >= 0 else torch.tensor(self.no_match_index)
                         for i in g])

                    matched_preds[b_idx, inconfident_prediction_ids] = torch.tensor([
                        unmatched_gt_ids[0][i] if i >= 0 else torch.tensor(self.no_match_index)
                        for i in p])

            if self.plot:
                self._debug_single_match_plot_(gt=grid_tensor[b_idx].numpy(), pred=preds[b_idx].detach().cpu().numpy(),
                                               p_match=matched_preds[b_idx].cpu().numpy(), suffix="uv")
            if torch.any(matched_gt[b_idx] != -100):
                good_gt_idx = torch.where(matched_gt[b_idx] >= 0)[0][0]
                Log.debug("Match e.g. %s with GT %s" % (
                    preds[b_idx][matched_gt[b_idx][good_gt_idx]], grid_tensor[b_idx][good_gt_idx]))

        Log.time(key="matching_uv", value=timeit.default_timer() - start)
        return matched_preds.to(preds.device), None

    def sort_lines_by_geometric_match(self, preds: torch.tensor, grid_tensor: list, filenames, epoch,
                                      tag="dummy_matcher",
                                      never_plot=False):
        """
        Calculate the distance between all predictions and GTs within an image.

        Args:
            preds (torch.tensor):
                with shape [batch, lines, vars]
            grid_tensor (list of torch.tensor):
                with shape [batch, lines, vars]

        Returns:
            torch.tensor:
                with shape [batch*lines, vars]
        """
        if Variables.GEOMETRY not in self.coords.train_vars() or \
                not np.any(np.asarray([len(g) for g in grid_tensor]) > 0):
            if Variables.GEOMETRY not in self.coords.train_vars():
                Log.warning("We need the geometry to calculate an association. We continue without matching.")
            elif not np.any(np.asarray([len(g) for g in grid_tensor]) > 0):
                Log.warning("We need labels to calculate an association. We continue without matching.")
            grid_tensor = torch.cat(grid_tensor)
            preds = preds.view(-1, self.coords.num_vars_to_train())
            return preds, grid_tensor

        Log.debug("UV Matching works with shapes preds=%s and e.g. gt=%s" % (preds.shape, grid_tensor[0].shape))
        matched_predictions, _ = self.match(preds=preds, grid_tensor=grid_tensor, filenames=filenames,
                                            confidence_threshold=self.args.confidence)

        # resort matched GT entries and have nan tensors in the remaining places
        # TODO rather not create a new one
        num_batch, num_lines, num_vars = preds.shape
        resorted_grid_tensor = torch.ones((num_batch, num_lines, self.coords.get_length())) * torch.nan
        for b_idx, pred_in_cell in enumerate(matched_predictions):
            for p_idx, gt_idx in enumerate(pred_in_cell):
                if gt_idx >= 0:
                    resorted_grid_tensor[b_idx, int(p_idx)] = grid_tensor[b_idx][int(gt_idx)]

        if not never_plot:
            self._debug_full_match_plot_(epoch, preds, resorted_grid_tensor, filenames=filenames,
                                         coordinates=CoordinateSystem.UV_SPLIT, tag=tag + "_resorted")

        resorted_grid_tensor = resorted_grid_tensor.view(-1, self.coords.get_length())
        preds = preds.view(-1, self.coords.num_vars_to_train())

        return preds, resorted_grid_tensor, matched_predictions


class UVPointMatcher(UVMatcher):
    def __init__(self, coords: VariableStructure, args, distance_threshold: int = -1):
        super().__init__(coords, args, distance_threshold)
        self.distance_metric = Distance.POINT
