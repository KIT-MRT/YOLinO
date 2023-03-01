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
import json
import os
from datetime import datetime

import numpy as np

from yolino.utils.enums import ImageIdx
from yolino.utils.logger import Log


class MetricsStorage:
    def __init__(self):
        self.pre_nms_recalls = []
        self.pre_nms_precisions = []

        self.recalls = []
        self.precisions = []
        self.inferences = []
        self.distances = {}
        self.files = []

    def save(self, cuda, epoch, log_dir, logger, out, file_out):
        metrics = {}

        metrics["inference" + cuda] = np.mean(self.inferences)
        recall = np.mean(self.recalls)
        metrics["recall"] = recall
        prec = np.mean(self.precisions)
        metrics["precision"] = prec
        metrics["pre_nms_recall"] = np.mean(self.pre_nms_recalls)
        metrics["pre_nms_precision"] = np.mean(self.pre_nms_precisions)
        metrics["F1"] = self.getF1(prec, recall)
        metrics["epoch"] = epoch
        metrics["val_files"] = len(np.unique(self.files))
        metrics["experiment"] = log_dir

        today = datetime.now()
        metrics["date"] = today.strftime("%b-%d-%Y_%H-%M-%S")

        if logger:
            logger.report_scalar(title='ROC', series='Pre NMS Recall', value=metrics["pre_nms_recall"],
                                 iteration=epoch)
            logger.report_scalar(title='ROC', series='Pre NMS Precision', value=metrics["pre_nms_precision"],
                                 iteration=epoch)
            logger.report_scalar(title='ROC', series='Recall', value=metrics["recall"], iteration=epoch)
            logger.report_scalar(title='ROC', series='Precision', value=metrics["precision"], iteration=epoch)
            logger.report_scalar(title='ROC', series='F1', value=metrics["F1"], iteration=epoch)

            logger.report_histogram(title='FileBased', series='Recall',
                                    iteration=epoch, values=self.recalls, xlabels=self.files, xaxis='Samples',
                                    yaxis='Recall')
            logger.report_histogram(title='FileBased', series='Precision',
                                    iteration=epoch, values=self.precisions, xlabels=self.files, xaxis='Samples',
                                    yaxis='Precision')
        store_metrics("point", metrics, out)
        store_metrics("point", self.distances, file_out)

    @classmethod
    def F1_score(cls, prec, recall):
        return 2 * ((prec * recall) / (prec + recall))

    def append(self, file_name, precision, recall, path_kind=ImageIdx.NMS, pre_nms=False):

        self.files.append(file_name)

        if pre_nms:
            self.pre_nms_precisions.append(precision)
            self.pre_nms_recalls.append(recall)
        else:
            self.precisions.append(precision)
            self.recalls.append(recall)

            if not file_name in self.distances:
                self.distances[file_name] = {}
            self.distances[file_name][path_kind.value + "f1"] = self.F1_score(precision, recall)
            self.distances[file_name][path_kind.value + "recall"] = recall
            self.distances[file_name][path_kind.value + "precision"] = precision

    def __len__(self):
        return len(self.files)


def store_metrics(metric_type, key_value_pairs, filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                metrics = json.load(f)
                if metric_type not in metrics:
                    metrics[metric_type] = {}
            except BaseException:
                print("Error in parsing json. Overwrite old metrics.json")
                metrics = {}
                metrics[metric_type] = {}
    else:
        metrics = {metric_type: {}}

    metrics[metric_type] = key_value_pairs

    Log.info("Write eval metrics to file://%s" % filename)
    with open(filename, "w") as f:
        json.dump(metrics, f)

    return
