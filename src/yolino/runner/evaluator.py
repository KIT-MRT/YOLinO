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
import timeit
import warnings

import numpy as np
import torch

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.eval.iccv_metrics import iccv_f1
from yolino.eval.matcher_cell import CellMatcher
from yolino.eval.matcher_uv import UVMatcher
from yolino.eval.metrics import ClassificationMetrics, DetectionMetrics, GeometryMetrics
from yolino.grid.grid_factory import GridFactory
from yolino.postprocessing.line_fit import fit_lines
from yolino.runner.forward_runner import ForwardRunner
from yolino.runner.nms_handler import NmsHandler
from yolino.utils.enums import CoordinateSystem, ImageIdx, Variables, LINE, AnchorDistribution
from yolino.utils.logger import Log
from yolino.viz.plot import plot_style_grid


class Evaluator:

    def __init__(self, args, anchors, coords=None, prepare_forward=True, load_best_model=True) -> None:

        self.anchors = anchors
        self.scores = {}
        self.args = args
        self.plot = self.args.plot

        if coords is None:
            self.coords = DatasetFactory.get_coords(split=args.split, args=args)
        else:
            self.coords = coords

        self.points_coords = self.coords.clone(LINE.POINTS)
        self.matcher = UVMatcher(self.points_coords, self.args, distance_threshold=args.matching_gate_px)
        self.cell_matcher = CellMatcher(self.points_coords, self.args)

        if prepare_forward:
            args.retrain = False
            Log.warning("We set retrain==False")
            self.forward = ForwardRunner(args, coords=self.coords, load_best=load_best_model)
            if self.forward.start_epoch == 0:
                raise FileNotFoundError("Please provide a valid, trained model.")
        else:
            self.forward = None

        self.matching_metrics = ClassificationMetrics(metric_selections=self.args.metrics,
                                                      coords=self.points_coords, max_class=1)

        self.class_metrics = ClassificationMetrics(metric_selections=self.args.metrics,
                                                   coords=self.points_coords,
                                                   max_class=self.points_coords[Variables.CLASS] - 1)
        self.detection_metrics = DetectionMetrics(metric_selections=self.args.metrics, coords=self.points_coords)
        self.geom_metrics = GeometryMetrics(metric_selections=self.args.metrics, coords=self.points_coords)

        self.nms = NmsHandler(args, coords=self.coords)

    def on_data_loaded(self, filename, image, grid_tensor, epoch):
        if image.shape[1] != self.args.img_size[0] or image.shape[2] != self.args.img_size[1]:
            raise ValueError("Image has %s, we want %s" % (image.shape, self.args.img_size))

        grid, errors = GridFactory.get(np.expand_dims(grid_tensor, axis=0), [], CoordinateSystem.CELL_SPLIT, self.args,
                                       input_coords=self.coords, threshold=0.5, anchors=self.anchors)

        path = self.args.paths.generate_eval_label_file_path(filename, idx=ImageIdx.GRID)
        plot_style_grid(grid.get_image_lines(coords=self.points_coords, image_height=image.shape[1]), path, image,
                        show_grid=True,
                        cell_size=grid.get_cell_size(image.shape[1]),
                        coordinates=CoordinateSystem.UV_SPLIT, tag=self.args.split, imageidx=ImageIdx.GRID,
                        coords=self.coords, epoch=epoch)

        del grid

    def __call__(self, images, grid_tensor, idx, filenames, epoch, num_duplicates, tag="dummy_eval", do_in_uv=True,
                 apply_nms=False, fit_line=False):

        preds = self.forward(images, epoch=epoch, is_train=False)
        preds = preds.detach()
        if self.args.gpu:
            preds = preds.cpu()

        if self.plot:
            for i in range(len(preds)):
                pred_grid, _ = GridFactory.get(torch.unsqueeze(preds[i].detach().cpu(), dim=0), [],
                                               CoordinateSystem.CELL_SPLIT, self.args,
                                               input_coords=self.coords,
                                               only_train_vars=True, anchors=self.anchors)
                grid, _ = GridFactory.get(torch.unsqueeze(grid_tensor[i], dim=0), [],
                                          CoordinateSystem.CELL_SPLIT, self.args,
                                          input_coords=self.coords,
                                          only_train_vars=False, anchors=self.anchors)

                pred_grid_uv_lines = pred_grid.get_image_lines(coords=self.coords,
                                                               image_height=images[i].shape[1])
                gt_grid_uv_lines = grid.get_image_lines(coords=self.coords,
                                                        image_height=images[i].shape[1])

                img_path = self.args.paths.generate_debug_image_file_path(filenames[i], ImageIdx.PRED)
                ok = plot_style_grid(pred_grid_uv_lines, img_path, images[i], coords=self.coords,
                                     cell_size=self.args.cell_size,
                                     show_grid=True, coordinates=CoordinateSystem.UV_SPLIT, epoch=None, tag=tag,
                                     imageidx=ImageIdx.PRED, threshold=self.args.confidence, gt=gt_grid_uv_lines,
                                     training_vars_only=True,
                                     level=1)

        # Normal
        start = timeit.default_timer()

        fitted_lines = []
        if do_in_uv:
            Log.info("Run evaluation on full UV")
            try:
                preds_uv, gt_uv = self.prepare_uv(preds=preds, grid_tensors=grid_tensor, filenames=filenames,
                                                  images=images)
            except ValueError as ex:
                if epoch <= 5:
                    Log.error(str(ex))
                    return preds, None
                else:
                    raise ex
            uv_conversion = timeit.default_timer()
            Log.time(key="uv_conversion", value=uv_conversion - start)
            self.get_scores_uv(gt_uv=gt_uv, preds_uv=preds_uv, epoch=epoch, filenames=filenames,
                               num_duplicates=num_duplicates, tag=tag)

            # NMS
            if apply_nms:
                preds = self.get_nms_results(filenames, gt_uv, epoch, images, preds_uv, tag)

            start = timeit.default_timer()
            if fit_line:
                fitted_lines = []
                for i in range(len(preds_uv)):
                    lines = fit_lines(lines_uv=preds[[i]], coords=self.coords,
                                      confidence_threshold=self.args.confidence,
                                      adjacency_threshold=self.args.adjacency_threshold,
                                      grid_shape=self.args.grid_shape,
                                      min_segments_for_polyline=self.args.min_segments_for_polyline,
                                      cell_size=self.args.cell_size, image=images[i], file_name=filenames[i],
                                      paths=self.args.paths, args=self.args, split=self.args.split)
                    fitted_lines.append(lines)
            Log.time(key="fit_lines", value=timeit.default_timer() - start)
        else:
            self.get_scores_in_cell(grid_tensor, preds, self.forward.start_epoch, filenames,
                                    num_duplicates=num_duplicates, tag=tag,
                                    do_matching=self.args.anchors == AnchorDistribution.NONE)

        return preds, fitted_lines

    def get_nms_results(self, filenames, gt_uv, epoch, images, preds_uv, tag, num_duplicates):
        set = []

        start = timeit.default_timer()
        for i in range(len(preds_uv)):
            lines, reduced = self.nms(images=images[[i]], preds=preds_uv[[i]], labels=gt_uv[i].unsqueeze(dim=0),
                                      filenames=[filenames[i]],
                                      epoch=self.forward.start_epoch if self.forward else 0)
            set.append(lines)

        nms = np.concatenate(set)
        after_nms = timeit.default_timer()
        Log.time(key="nms", value=after_nms - start)
        # should not be the reduced nms here ! otherwise there is no fn etc
        self.get_scores_uv(gt_uv=gt_uv, preds_uv=torch.tensor(nms), epoch=epoch, filenames=filenames,
                           num_duplicates=num_duplicates, tag=tag + "_nms", prefix="uv_metrics_nms")
        Log.time(key="nms_eval", value=timeit.default_timer() - after_nms)
        # TODO eval nms scores in test
        return nms

    def prepare_uv(self, grid_tensors, preds, filenames, images):

        if torch.all(torch.isnan(preds)):
            raise ValueError("Prediction is nan")

        num_batch, num_cells, num_preds, num_train_vars = preds.shape
        _, _, _, num_vars = grid_tensors.shape

        preds_uv = []
        gt_uv = []

        for batch in range(num_batch):
            pred_grid, _ = GridFactory.get(preds[[batch]], [], CoordinateSystem.CELL_SPLIT, args=self.args,
                                           input_coords=self.coords, only_train_vars=True, anchors=self.anchors)
            grid, _ = GridFactory.get(grid_tensors[[batch]], [], CoordinateSystem.CELL_SPLIT, self.args,
                                      input_coords=self.coords,
                                      only_train_vars=False, anchors=self.anchors)
            if len(pred_grid) < len(grid) or len(pred_grid) != num_cells * num_preds:
                raise ValueError("Prediction is malformed. The prediction has by definition a line per predictor, "
                                 "thus we expect %d lines. Also, the GT is expected to have quite less lines, "
                                 "but at most the same as the prediction. We have GT=%d, pred=%d."
                                 % (num_batch * num_cells * num_preds, len(grid), len(pred_grid)))

            preds_uv.append(torch.tensor(
                pred_grid.get_image_lines(coords=self.points_coords, image_height=images[batch].shape[1],
                                          is_training_data=True),
                dtype=torch.float32))
            gt_uv.append(
                torch.squeeze(
                    torch.tensor(grid.get_image_lines(coords=self.points_coords, image_height=images[batch].shape[1]),
                                 dtype=torch.float32), dim=0))

        return torch.cat(preds_uv), gt_uv

    def get_scores_in_cell(self, grid_tensor, preds, epoch, filenames, num_duplicates, tag="dummy_eval",
                           do_matching=True):

        start = timeit.default_timer()
        if do_matching:
            resorted_pred, resorted_grid_tensor = self.cell_matcher.sort_cells_by_geometric_match(preds=preds,
                                                                                                  grid_tensor=grid_tensor,
                                                                                                  epoch=epoch,
                                                                                                  tag=tag,
                                                                                                  filenames=filenames)
        else:
            resorted_pred = preds.reshape(-1, self.points_coords.num_vars_to_train())
            resorted_grid_tensor = grid_tensor.view(-1, self.points_coords.get_length())

        assert torch.sum(~grid_tensor[:, :, :, 0].isnan()) == torch.sum(~resorted_grid_tensor[:, 0].isnan())
        matched_prediction_flags = ~resorted_grid_tensor[:, 0].isnan()
        matching_time_end = timeit.default_timer()
        Log.time(key="eval_cell_matching", value=matching_time_end - start)

        scores, has_tps = self.get_scores(filenames=filenames, gt_uv=resorted_grid_tensor,
                                          matched_prediction_flags=matched_prediction_flags, preds_uv=resorted_pred,
                                          prefix="cell_metrics", num_duplicates=num_duplicates,
                                          geom_px_scale=self.args.cell_size[0])

        if not has_tps:
            Log.info("We do not have any true positives in the prediction and thus will not calculate the "
                     "RMSE, MAE nor the class metrics")

        self.add_scores(scores)
        Log.eval_summary(self.scores)

    def get_scores_uv(self, gt_uv: list, preds_uv: torch.tensor, epoch, filenames, num_duplicates, tag="dummy_eval",
                      prefix="uv_metrics", do_matching=True, calculate_iccv=False):
        """
        Calculate scores for the training variables. Confidence and geometry must be training variables!
        First match the prediction and GT geometrically in UV-coordinates and calculate F1, recall, ...
        Then, for all TPs, calculate recall, precision, f1 score, accuracy, fn-rate, fp-rate on the geometric match.
        A line segment is true positive, when confidence > threshold and the matching GT lies within the
        distance threshold.

        Geometry:
            - RMSE for u, v, length, orthogonal distance, parallel distance and angular distance
            - subsample the true-positive line segments and calculate the RSME and MAE of the points (u,v)-positions
        Classificaiton:
            - calculate recall, precision, f1 score, accuracy, fn-rate, fp-rate for each class
        Confidence:
            - RMSE and MAE on confidence values of all pred/GT

        Args:
            preds_uv (torch.tensor): the prediction in UV coords, can be a full batch.
            gt_uv (list): a list of the GTs in UV coords, can be a full batch
        """

        start = timeit.default_timer()
        if not Variables.GEOMETRY in self.points_coords.train_vars():
            Log.warning("We do not eval scores without geometry")
            return
        if calculate_iccv:
            for t in [0.5, 0.99]:
                precision_tag = "iccv_precision_%s" % t
                self.scores[precision_tag] = 0
                recall_tag = "iccv_recall_%s" % t
                self.scores[recall_tag] = 0
                f1_tag = "iccv_f1_%s" % t
                self.scores[f1_tag] = 0
                for b, batch in enumerate(preds_uv):
                    precision, recall = iccv_f1(preds_uv=batch.unsqueeze(0), gt_uv=gt_uv[b].unsqueeze(0),
                                                img_size=[self.args.img_size[1], self.args.img_size[0]],
                                                conf_idx=self.points_coords.get_position_within_prediction(
                                                    Variables.CONF),
                                                threshold=t)
                    self.scores[precision_tag] += precision
                    self.scores[recall_tag] += recall
                    if recall > 0 and precision > 0:
                        self.scores[f1_tag] += 2. * precision * recall / (precision + recall)

                self.scores[precision_tag] /= len(preds_uv)
                self.scores[recall_tag] /= len(preds_uv)
                self.scores[f1_tag] /= len(preds_uv)
            Log.time(key="iccv", value=timeit.default_timer() - start)
        start = timeit.default_timer()

        # 2. match 1:1 with distance threshold
        # matched_preds_indices contains at the prediction position the matching GT ID
        # matched_gt_indices contains at the GT position the matching prediction ID
        if do_matching:
            preds_uv, gt_uv, matched_preds_indices = self.matcher.sort_lines_by_geometric_match(preds=preds_uv,
                                                                                                grid_tensor=gt_uv,
                                                                                                epoch=epoch,
                                                                                                tag=tag,
                                                                                                filenames=filenames)
            matched_prediction_flags = torch.where(matched_preds_indices == -100, 0, 1).flatten()
        else:
            gt_uv = torch.cat(gt_uv)
            preds_uv = preds_uv.view(-1, self.points_coords.num_vars_to_train())
            matched_prediction_flags = ~gt_uv[:, 0].isnan()
            if len(matched_prediction_flags) < len(preds_uv):
                matched_prediction_flags = torch.cat([matched_prediction_flags,
                                                      [False] * (len(preds_uv) - len(matched_prediction_flags))])

        matching_time_end = timeit.default_timer()
        Log.time(key="eval_uv_matching", value=matching_time_end - start)
        scores, has_tps = self.get_scores(filenames, gt_uv, matched_prediction_flags, preds_uv, prefix,
                                          num_duplicates=num_duplicates)
        if not has_tps:
            Log.warning("We do not have any true positives in the prediction with matching gate=%s and thus will "
                        "not calculate the RMSE, MAE nor the class metrics"
                        % self.args.matching_gate_px)

        self.add_scores(scores)
        Log.eval_summary(self.scores)

    def add_scores(self, scores):
        for k, v in scores.items():
            if k not in self.scores:
                if k == "confusion":
                    self.scores[k] = np.empty(
                        (self.matching_metrics.max_class + 1, self.matching_metrics.max_class + 1, 0))
                else:
                    self.scores[k] = np.empty((0, 1 if np.isscalar(v) else len(v)))

            try:
                if k == "confusion":
                    self.scores[k] = np.concatenate([self.scores[k], np.expand_dims(v, axis=-1)],
                                                    axis=-1)  # => 5,5,x
                else:
                    self.scores[k] = np.vstack([self.scores[k], v])  # => x, 5
            except ValueError as ex:
                Log.error("%s\nTried to stack %s on top of %s for key=%s" % (ex, v, self.scores[k].shape, k))
                continue

    def get_scores(self, filenames, gt_uv, matched_prediction_flags, preds_uv, prefix, num_duplicates, geom_px_scale=1):
        """

        @param geom_px_scale: Provides a scaling factor in order to scale the input geometry values to actual pixels.
        Especially for cell based evaluation this should be set to the cell size as the geometry values range between
        0 and 1.
        """
        start = timeit.default_timer()
        scores = {}
        if Variables.CONF in self.points_coords.train_vars():
            # 3. Metrics on matches
            # TP: conf >= args.conf && matched with GT
            # FP: conf >= args.conf && not matched with GT
            # TN: conf < args.conf && not matched with GT
            # FN: conf < args.conf && matched with GT
            conf_position = self.points_coords.get_position_within_prediction(Variables.CONF)
            pred_is_confident_flags = (preds_uv[:, conf_position] >= self.args.confidence).to(torch.int).flatten()
            scores.update(self.matching_metrics.get(preds=pred_is_confident_flags, labels=matched_prediction_flags,
                                                    num_duplicates=0, prefix=prefix))
            scores.update(self.matching_metrics.get(preds=pred_is_confident_flags, labels=matched_prediction_flags,
                                                    num_duplicates=num_duplicates, prefix=prefix + "_dupl"))

            # select only true positives => conf > args.conf and match
            tp_flags = torch.logical_and(matched_prediction_flags, pred_is_confident_flags)
        else:
            tp_flags = torch.logical_and(matched_prediction_flags, matched_prediction_flags)
        matching_metrics_time_end = timeit.default_timer()
        Log.time(key="matching_metrics", value=matching_metrics_time_end - start)
        if torch.any(tp_flags):
            prediction_position = self.points_coords.get_position_within_prediction(Variables.GEOMETRY)
            label_position = self.points_coords.get_position_of(Variables.GEOMETRY)

            # # # 4. GEOM
            # sample_geom_scores = self.sample_metrics.get(prefix="sample", preds=preds_uv[tp_flags].unsqueeze(dim=0),
            #                                              labels=gt_uv[tp_flags].unsqueeze(dim=0))
            # scores.update(sample_geom_scores)

            # 4. GEOM
            #   RMSE, MAE on x,y,l_diff,orth,parallel,ang of TPs
            prefix = (prefix + "/" if len(prefix) > 0 else prefix)
            geom_scores = self.geom_metrics.get(preds=preds_uv[tp_flags][:, prediction_position],
                                                labels=gt_uv[tp_flags][:, label_position], filenames=filenames,
                                                prefix=prefix + "geom", geom_px_scale=geom_px_scale)
            scores.update(geom_scores)
            geom_metrics_end = timeit.default_timer()
            Log.time(key="geom_metrics", value=geom_metrics_end - matching_metrics_time_end)

            # 5. CLASS
            #   ClassficiationMetrics on TP matches values or all if there was no conf
            if Variables.CLASS in self.points_coords.train_vars():
                prediction_position = self.points_coords.get_position_within_prediction(Variables.CLASS)
                label_position = self.points_coords.get_position_of(Variables.CLASS)

                Log.error("We did not check the confusion matrix for classification in case of duplicates!")
                class_scores = self.class_metrics.get(preds=torch.argmax(preds_uv[tp_flags][:, prediction_position],
                                                                         dim=1),
                                                      labels=torch.argmax(gt_uv[tp_flags][:, label_position], dim=1),
                                                      num_duplicates=num_duplicates, prefix=prefix + "class")
                scores.update(class_scores)
                class_metrics_end = timeit.default_timer()
                Log.time(key="class_metrics", value=class_metrics_end - geom_metrics_end)

            if Variables.CONF in self.points_coords.train_vars():
                tp_mse_metrics_start = timeit.default_timer()
                # 6. CONF
                #   RMSE / MAE on TP only
                prediction_position = self.points_coords.get_position_within_prediction(Variables.CONF)
                label_position = self.points_coords.get_position_of(Variables.CONF)
                gt_conf = gt_uv[:, label_position]
                gt_conf[gt_conf.isnan()] = 0
                geom_scores = self.detection_metrics.get(preds=preds_uv[tp_flags][:, prediction_position],
                                                         labels=gt_conf[tp_flags], filenames=filenames,
                                                         num_duplicates=-1, prefix=prefix + "conf")
                scores.update(geom_scores)
                tp_mse_metrics_end = timeit.default_timer()
                Log.time(key="tp_conf_metrics", value=tp_mse_metrics_end - tp_mse_metrics_start)

                # 7. CONF
                #   RMSE / MAE on value of all (TP, TN, FN, FP)
                geom_scores = self.detection_metrics.get(preds=preds_uv[:, prediction_position], labels=gt_conf,
                                                         filenames=filenames, num_duplicates=num_duplicates,
                                                         prefix=prefix + "conf_all")
                scores.update(geom_scores)
                tp_mse_metrics_end = timeit.default_timer()
                Log.time(key="all_conf_metrics", value=timeit.default_timer() - tp_mse_metrics_end)

        return scores, torch.any(tp_flags)

    def publish_scores(self, epoch, tag):
        scores = {}
        for k in self.scores:
            # we want to build the average on all epochs but not on the classes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scores[k] = np.nanmean(self.scores[k], axis=0).tolist()

        if "confusion" in scores:
            del scores["confusion"]
        if "confusion_mean" in scores:
            del scores["confusion_mean"]

        if self.matcher.plot:
            Log.info("You can find evaluation debug image in file://%s or file://%s" %
                     (
                         os.path.join(self.args.paths.debug_folder,
                                      str(epoch) + "_" + str(self.args.split) + "_debug.png"),
                         self.args.paths.generate_eval_image_file_path("<file>", idx=ImageIdx.PRED)))
        Log.scalars(tag=tag, dict=scores, epoch=epoch)
        Log.eval_summary(scores)
        self.scores = {}
        return scores
