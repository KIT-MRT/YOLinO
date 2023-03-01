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

import torch
from yolino.eval.matcher import Matcher
from yolino.grid.coordinates import validate_input_structure
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import CoordinateSystem, Variables
from yolino.utils.logger import Log


class CellMatcher(Matcher):
    def __init__(self, coords: VariableStructure, args):
        super().__init__(coords, args, distance_threshold=-1, is_cell_based=True)

    def sort_cells_by_geometric_match(self, preds, grid_tensor, filenames, epoch, tag="dummy_matcher"):
        """
        Calculate the distance between all predictions and GTs within a cell.
        We assume all GT have a match and thus provide the sorted GT to match the assigned predictions.

        Args:
            preds (torch.tensor):
                with shape [batch, cells, preds, vars]
            grid_tensor (torch.tensor):
                with shape [batch, cells, preds, vars]

        Returns:
            torch.tensor:
                with shape [batch*cells*preds, vars]
        """
        Log.debug("Loss works with shapes preds=%s and gt=%s" % (preds.shape, grid_tensor.shape))
        if Variables.GEOMETRY not in self.coords.train_vars() \
                or torch.all(grid_tensor[:, :, :, self.coords.get_position_of(Variables.GEOMETRY)].isnan()):
            if Variables.GEOMETRY not in self.coords.train_vars():
                Log.warning("We need the geometry to be training variable to calculate association loss. "
                            "We continue without matching.")
            else:
                Log.warning("No geometry labels so we continue without matching.")

            try:
                grid_tensor = grid_tensor.reshape(-1, self.coords.get_length())
                preds = preds.reshape(-1, self.coords.num_vars_to_train())
            except RuntimeError as re:
                Log.error(f"Reshape a Grid of {grid_tensor.shape} to (-1,{self.coords.get_length()})")
                Log.error(f"Reshape the Prediction of {preds.shape} to (-1,{self.coords.num_vars_to_train()})")
                raise re
            return preds, grid_tensor

        matched_predictions, _ = self.match(preds=preds, grid_tensor=grid_tensor, filenames=filenames,
                                            confidence_threshold=self.args.confidence)
        resorted_grid_tensor = self.__resort_by_match_ids__(grid_tensor, matched_predictions)

        self._debug_full_match_plot_(epoch, preds, resorted_grid_tensor, filenames, CoordinateSystem.CELL_SPLIT,
                                     tag=tag)

        resorted_grid_tensor = resorted_grid_tensor.view(-1, self.coords.get_length())
        preds = preds.reshape(-1, self.coords.num_vars_to_train())

        return preds, resorted_grid_tensor

    def match(self, preds, grid_tensor, filenames, confidence_threshold):
        """
        Determines a 1:1 matching between GT and prediction.
        The distance metric is specified in the constructor.

        Args:
            preds (torch.tensor):
                with shape [batch, cells, preds, vars]
            grid_tensor (torch.tensor):
                with shape [batch, cells, preds, vars]

        Returns:
            matched_preds (torch.tensor):
                Will be filled with the matching GT ID or -100 if there is no match at the position
                of the predictions in the input. Will always have shape
                [batch * number of cells, number of predictors].
            matched_gt (torch.tensor):
                Output variable filled with the matching prediction ID or -100 if there is no match at the position
                of the GT in the input. Will always have shape [batch * number of cells, number of predictors].
        """
        start = timeit.default_timer()
        Log.debug("Cell match pred %s and gt %s" % (str(preds.shape), str(grid_tensor.shape)))
        validate_input_structure(preds, CoordinateSystem.CELL_SPLIT)
        validate_input_structure(grid_tensor, CoordinateSystem.CELL_SPLIT)
        num_batch, num_cells, _, _ = preds.shape
        matched_preds = torch.ones((num_batch * num_cells, self.args.num_predictors),
                                   dtype=torch.int64, device=preds.device) * self.no_match_index
        matched_gt = torch.ones((num_batch * num_cells, self.args.num_predictors),
                                dtype=torch.int64, device=preds.device) * self.no_match_index

        Log.warning("Waiting for cell match...")
        start = timeit.default_timer()
        gt_is_valid = ~torch.isnan(grid_tensor[:, :, :, 0])
        where_valid = torch.where(torch.any(gt_is_valid, dim=2))
        for b_idx, c_idx in zip(where_valid[0], where_valid[1]):
            grid_tensor_cell = grid_tensor[b_idx, c_idx]
            pred_cell = preds[b_idx, c_idx]
            idx = b_idx * num_cells + c_idx
            if self.args.match_by_conf_first:
                ids = torch.where(pred_cell[:, -1] >= confidence_threshold)[0]
            else:
                ids = range(len(pred_cell))
            if len(ids) > 0:
                confident_predictions = pred_cell[ids]
                ok, g, p = self.get_matches_in_set(confident_predictions, grid_tensor_cell,
                                                   distance_metric=self.args.association_metric,
                                                   filename=filenames[b_idx],
                                                   # use confidence in distance measure when we have the
                                                   # onestage-match-all version
                                                   use_conf=False if self.args.match_by_conf_first else
                                                   self.args.use_conf_in_loss_matching)
                if not ok:
                    Log.warning("Match not ok for cell %s" % c_idx)
                    continue

                matched_gt[idx] = torch.tensor([ids[i] if i >= 0 else torch.tensor(self.no_match_index)
                                                for i in g], device=preds.device)
                matched_preds[idx, ids] = p.to(preds.device)

            if self.args.match_by_conf_first and \
                    torch.any(matched_gt[idx, gt_is_valid[b_idx][c_idx]] == self.no_match_index):
                inconfident_prediction_ids = torch.where(pred_cell[:, -1] < confidence_threshold)[0]
                inconfident_predictions = pred_cell[inconfident_prediction_ids]

                unmatched_gt_ids = torch.where(matched_gt[idx] == self.no_match_index)
                unmatched_gt = grid_tensor_cell[unmatched_gt_ids]
                ok, g, p = self.get_matches_in_set(inconfident_predictions, unmatched_gt,
                                                   distance_metric=self.args.association_metric,
                                                   filename=filenames[b_idx], use_conf=False)
                if not ok:
                    Log.warning("Match not ok for cell %s" % c_idx)
                    continue

                matched_gt[idx][unmatched_gt_ids] = torch.tensor(
                    [inconfident_prediction_ids[i] if i >= 0 else torch.tensor(self.no_match_index)
                     for i in g], device=preds.device)

                matched_preds[idx, inconfident_prediction_ids] = torch.tensor([
                    unmatched_gt_ids[0][i] if i >= 0 else torch.tensor(-100)
                    for i in p], device=preds.device)

            if False and self.plot:
                self._debug_single_match_plot_(gt=grid_tensor_cell.numpy(), pred=pred_cell.detach().cpu().numpy(),
                                               p_match=matched_preds[idx].cpu().numpy(),
                                               suffix="cell_%d_%d" % (b_idx, c_idx))
        if len(where_valid[0]) > 0:
            Log.debug("Match e.g. on %s with GT %s:\n%s\n%s"
                      % (pred_cell, grid_tensor_cell, matched_gt[idx], matched_preds[idx]))
            Log.warning("Cell matching took %fs" % (timeit.default_timer() - start))
            Log.time(key="cell_match", value=(timeit.default_timer() - start))

        Log.time(key="matching_cell", value=timeit.default_timer() - start)
        return matched_preds, matched_gt
