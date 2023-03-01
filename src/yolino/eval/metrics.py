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
import warnings

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from yolino.eval.distances import get_angular_distance, \
    get_perpendicular_distances, midpoints_distance
from yolino.eval.matcher_uv import UVPointMatcher
from yolino.grid.grid import Grid
from yolino.model.line_representation import PointsLines
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import Metric, Variables, LINE
from yolino.utils.logger import Log


class DetectionMetrics:
    def __init__(self, metric_selections, coords):
        self.__metric_selection__ = metric_selections
        self._labels_ = None
        self._predictions_ = None
        self._filenames_ = None

        if not coords[Variables.GEOMETRY] == PointsLines().num_params:
            raise ValueError("The fiven coordinates are not specified for the corrrect linerepresentation. "
                             "We expect points for detection metrics.")
        self.coords = coords

    def rmse(self):
        try:
            return mean_squared_error(self._labels_, self._predictions_, squared=False)
        except ValueError as e:
            Log.error("\n%s\n%s" % (self._labels_, self._predictions_))
            raise e

    def mae(self):
        return mean_absolute_error(self._labels_, self._predictions_)

    def get(self, preds, labels, filenames, num_duplicates, prefix="", geom_px_scale=1):
        preds = np.asarray(preds).flatten()
        labels = np.asarray(labels).flatten()
        if len(prefix) > 0:
            prefix += "_"

        self._predictions_ = preds
        self._labels_ = labels
        self._filenames_ = filenames

        result = {}
        for method in [Metric.RMSE, Metric.MAE]:
            if not method in self.__metric_selection__:
                continue
            prefixed_enum = prefix + str(method)
            result[prefixed_enum] = getattr(self, str(method))() * geom_px_scale
        return result


class SampleGeometryMetrics(DetectionMetrics):
    def __init__(self, metric_selections, coords, args):
        super().__init__(metric_selections, coords)
        self.args = args

        match_coords = VariableStructure(line_representation_enum=LINE.POINTS, num_classes=0, num_conf=0, num_angles=1,
                                         vars_to_train=[Variables.GEOMETRY, Variables.SAMPLE_ANGLE])
        self.matcher = UVPointMatcher(coords=match_coords, args=args)

    def get(self, preds, labels, filenames, num_duplicates, prefix=""):
        sample_coords = VariableStructure(line_representation_enum=LINE.POINTS, num_classes=0, num_conf=0, num_angles=1,
                                          vars_to_train=[Variables.GEOMETRY, Variables.SAMPLE_ANGLE])
        gt_samples = Grid.sample_along_lines(sample_distance=1, scaled_lines=labels.numpy(),
                                             coords=sample_coords, return_variables=False)
        p_samples = Grid.sample_along_lines(sample_distance=1, scaled_lines=preds.numpy(),
                                            coords=sample_coords, return_variables=False)
        # [batch, lines, xya]
        matched_preds_indices, _ = self.matcher.match(preds=torch.unsqueeze(torch.tensor(p_samples), dim=0),
                                                      grid_tensor=list([torch.tensor(gt_samples)]), filenames=filenames)
        # [batch*lines, x]
        preds_uv, gt_uv = self.matcher.sort_lines_by_geometric_match(
            preds=torch.unsqueeze(torch.tensor(p_samples), dim=0),
            grid_tensor=list([torch.tensor(gt_samples)]),
            epoch=None, never_plot=True, filenames=filenames)
        matched_prediction_flags = torch.where(matched_preds_indices == -100, False, True).flatten()

        result = {}

        # position
        result.update(
            super().get(preds_uv[matched_prediction_flags, 0], gt_uv[matched_prediction_flags, 0], filenames=filenames,
                        num_duplicates=-1, prefix=prefix + "_x"))
        result.update(
            super().get(preds_uv[matched_prediction_flags, 1], gt_uv[matched_prediction_flags, 1], filenames=filenames,
                        num_duplicates=-1, prefix=prefix + "_y"))

        # angular
        result.update(
            super().get(preds_uv[matched_prediction_flags, 2], gt_uv[matched_prediction_flags, 2], filenames=filenames,
                        num_duplicates=-1, prefix=prefix + "_ang"))

        return result


class GeometryMetrics(DetectionMetrics):
    def __init__(self, metric_selections, coords):
        super().__init__(metric_selections, coords)

    def get(self, preds, labels, filenames, prefix="", geom_px_scale=1):
        result = {}

        # position
        result.update(super().get(preds[:, [0, 2]], labels[:, [0, 2]], filenames=filenames, num_duplicates=-1,
                                  prefix=prefix + "_x", geom_px_scale=geom_px_scale))
        result.update(super().get(preds[:, [1, 3]], labels[:, [1, 3]], filenames=filenames, num_duplicates=-1,
                                  prefix=prefix + "_y", geom_px_scale=geom_px_scale))

        # length
        len_pred = np.linalg.norm(preds[:, 0:2] - preds[:, 2:4], axis=1)
        len_gt = np.linalg.norm(labels[:, 0:2] - labels[:, 2:4], axis=1)
        len_dist = len_pred - len_gt
        result.update(super().get(len_dist, np.zeros_like(len_dist), filenames=filenames, num_duplicates=-1,
                                  prefix=prefix + "_length", geom_px_scale=geom_px_scale))

        # perpendicular
        perpendicular_distances = np.stack([get_perpendicular_distances(labels[i], len_gt[i], preds[i], len_pred[i])
                                            for i in range(len(labels))])[:, [0, 2]]  # prediction points to GT only
        result.update(super().get(perpendicular_distances, np.zeros_like(perpendicular_distances), filenames=filenames,
                                  num_duplicates=-1, prefix=prefix + "_perp", geom_px_scale=geom_px_scale))

        # midpoint
        parallel_distances = np.stack([midpoints_distance(gt=labels[i], pred=preds[i]) for i in range(len(labels))])
        result.update(
            super().get(parallel_distances, np.zeros_like(parallel_distances), filenames=filenames, num_duplicates=-1,
                        prefix=prefix + "_midpoint", geom_px_scale=geom_px_scale))

        # angular
        parallel_distances = np.stack([get_angular_distance(labels[i], len_gt[i], preds[i], len_pred[i])
                                       for i in range(len(labels))])
        result.update(
            super().get(parallel_distances, np.zeros_like(parallel_distances), filenames=filenames, num_duplicates=-1,
                        prefix=prefix + "_ang"))

        return result


class ClassificationMetrics:
    def __init__(self, metric_selections, coords, max_class):
        self.__metric_selection__ = metric_selections
        self._labels_ = None
        self._predictions_ = None
        self._confusion_ = None

        if not coords[Variables.GEOMETRY] == PointsLines().num_params:
            raise ValueError("The given coordinates are not specified for the correct line representation. "
                             "We expect points for the metrics.")
        self.coords = coords
        self.max_class = max_class
        self.binary = max_class == 1

        self.__tp__ = None
        self.__fp__ = None
        self.__fn__ = None
        self.__tn__ = None

    def tp(self):
        if self.__tp__ is None:
            if self.binary:
                return np.sum(np.logical_and(
                    np.equal(self._predictions_, self._labels_),  # true
                    np.equal(self._predictions_, 1)))  # positive
            else:
                self.__tp__ = np.diag(self._confusion_)

        return self.__tp__

    def fp(self):
        if self.__fp__ is None:
            if self.binary:
                self.__fp__ = np.sum(np.logical_and(
                    np.not_equal(self._predictions_, self._labels_),  # false
                    np.equal(self._predictions_, 1)))  # positive
            else:
                self.__fp__ = np.sum(self._confusion_, axis=0) - self.tp()
        return self.__fp__

    def fn(self):
        if self.__fn__ is None:
            if self.binary:
                self.__fn__ = np.sum(np.logical_and(
                    np.not_equal(self._predictions_, self._labels_),  # false
                    np.equal(self._predictions_, 0)))  # negative
            else:
                self.__fn__ = np.sum(self._confusion_, axis=1) - self.tp()
        return self.__fn__

    def tn(self):
        if self.__tn__ is None:
            if self.binary:
                self.__tn__ = np.sum(np.logical_and(
                    np.equal(self._predictions_, self._labels_),  # true
                    np.equal(self._predictions_, 0)))  # negative
            else:
                self.__tn__ = np.sum(np.diag(self._confusion_)) - self.tp()
        return self.__tn__

    # can only be per class
    def precision(self):
        tp = self.tp()
        if self.binary:
            fp = self.fp()
            np_sum = tp + fp
        else:
            np_sum = np.sum(self._confusion_, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            return np.divide(tp, np_sum)

    # can only be per class
    def recall(self):
        tp = self.tp()
        if self.binary:
            np_sum = tp + self.fn()
        else:
            np_sum = np.sum(self._confusion_, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            return np.divide(tp, np_sum)

    def f1(self):
        with warnings.catch_warnings(record=True) as w:
            f1 = f1_score(self._labels_, self._predictions_,
                          labels=range(self.max_class + 1),
                          average=None, zero_division="warn")

        if len(w) > 0 and issubclass(w[-1].category, UndefinedMetricWarning):
            Log.debug(w[-1])
            f1_new = f1_score(self._labels_, self._predictions_,
                              labels=range(self.max_class + 1),
                              average=None, zero_division=1)
            f1[np.logical_and(f1_new == 1, f1 == 0)] = np.nan

        if self.binary:
            return f1[1]  # binary regards class 1 as the True one
        else:
            return f1

    def confusion(self):
        # gt as cols, pred as rows
        # TODO: checkout https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html#sklearn.metrics.multilabel_confusion_matrix

        matrix = confusion_matrix(self._labels_, self._predictions_, labels=range(self.max_class + 1))
        if not np.array_equal(matrix.shape, [self.max_class + 1, self.max_class + 1]):
            Log.warning("We expected %d classes in confusion matrix %s, "
                        "Labels %s, Predictions %s" % (self.max_class + 1, matrix.shape,
                                                       np.unique(self._labels_), np.unique(self._predictions_)))

        return matrix

    def accuracy(self):
        return accuracy_score(self._labels_, self._predictions_)

    def pr(self):
        # TODO: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve
        raise NotImplementedError("We did not implement precision recall curve")

    # TODO use more sklearn 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score 
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    # TODO use wandb viz https://docs.wandb.ai/guides/track/log/plots
    def get(self, preds, labels, num_duplicates, ignore_class=-1, prefix=""):
        self.clear()

        preds = np.asarray(preds).flatten()
        labels = np.asarray(labels).flatten()
        if len(prefix) > 0:
            prefix += "/"

        valid_indices = np.where(labels != ignore_class)
        if len(valid_indices) == 0 or np.size(valid_indices) == 0:
            Log.warning("No valid predictions for evaluation when ignoring classes: %s" % ignore_class)
            result = {}
            for enum in self.__metric_selection__:
                prefixed_enum = prefix + str(enum)
                result[prefixed_enum] = [np.nan]
                if not self.binary:
                    result[prefixed_enum + "_mean"] = np.nan
            return result

        self._predictions_ = preds[valid_indices]
        self._labels_ = labels[valid_indices]

        if self.max_class == 1:
            if num_duplicates > 0:
                self._labels_ = np.concatenate([self._labels_, np.ones(num_duplicates)])
                self._predictions_ = np.concatenate([self._predictions_, np.zeros(num_duplicates)])
            elif num_duplicates == -1:
                raise NotImplementedError("You did not provide any duplicates. Is this on purpose?")
        else:
            Log.error("We do not know how to handle more than one class (%s) with duplicates!" % self.max_class)

        self._confusion_ = self.confusion()

        result = {}
        for enum in self.__metric_selection__:
            prefixed_enum = prefix + str(enum)

            try:
                result[prefixed_enum] = getattr(self, str(enum))()
            except AttributeError as e:
                continue

            if not self.binary:
                if np.any(np.isnan(result[prefixed_enum]) == False):
                    result[prefixed_enum + "_mean"] = np.nanmean(result[prefixed_enum])
                else:
                    result[prefixed_enum + "_mean"] = np.nan

        return result

    def clear(self):
        self.__init__(self.__metric_selection__, self.coords, self.max_class)
