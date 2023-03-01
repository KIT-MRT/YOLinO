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
import shutil
import threading
import time
from pathlib import Path

from yolino.utils.enums import ImageIdx
from yolino.utils.logger import Log


class Paths:
    def __init__(self, dvc, split, explicit_model=None, debug_tool_name="no_description", metrics_prefix="metrics",
                 id="unkown", keep_checkpoints=False):
        """

        :rtype: object
        """
        self._lock = threading.Lock()

        qeval = os.path.join("qeval", split)
        dvc = os.path.abspath(dvc)
        if ".json" == metrics_prefix[-5:]:
            self.metrics_file = os.path.join(dvc, qeval, metrics_prefix)
            self.filebased_metrics_file = os.path.join(dvc, qeval, "file_" + metrics_prefix)
        else:
            self.metrics_file = os.path.join(dvc, qeval, metrics_prefix + ".json")
            self.filebased_metrics_file = os.path.join(dvc, qeval, "file_" + metrics_prefix + ".json")

        self.qeval_points = Path(os.path.abspath(os.path.join(dvc, qeval, 'points')))
        self.create_dir(self.qeval_points)
        self.qeval_tusimple = Path(os.path.abspath(os.path.join(dvc, qeval, 'tusimple')))
        self.create_dir(self.qeval_tusimple)

        self.prediction = Path(os.path.abspath(os.path.join(dvc, qeval, "prediction")))
        self.create_dir(self.prediction)
        self.prediction_grid = Path(os.path.abspath(os.path.join(dvc, qeval, "grid")))
        self.create_dir(self.prediction_grid)
        self.anchors = Path(os.path.abspath(os.path.join(dvc, qeval, "anchors")))
        self.create_dir(self.anchors)

        self.experiment_dir = Path(os.path.abspath(os.path.join(dvc, 'log')))
        self.logs = Path(os.path.abspath(os.path.join(self.experiment_dir, 'logs')))

        self.train_configuration = self.experiment_dir.joinpath('train_configuration.txt')

        if explicit_model is None:
            checkpoint_top_folder = os.path.join(self.experiment_dir, "checkpoints")
            self.cleanup_checkpoints(checkpoint_top_folder)

            self.checkpoints = Path(os.path.join(checkpoint_top_folder, id))
            if keep_checkpoints:
                self.create_dir(self.checkpoints)

            self.pretrain_checkpoints = self.checkpoints
            self.model = Path(os.path.join(self.checkpoints, "model.pth"))
            self.pretrain_model = self.model
        else:
            self.checkpoints = Path(os.path.dirname(explicit_model))
            self.pretrain_checkpoints = Path(os.path.dirname(explicit_model))
            self.model = Path(explicit_model)
            self.pretrain_model = Path(explicit_model)

        self.best_model = Path(os.path.join(self.checkpoints, "best_model.pth"))
        self.torch_script = Path(os.path.join(self.checkpoints, "best_model.pt"))
        self.train_config = Path(os.path.join(self.checkpoints, "train_configuration.txt"))

        self.pretrain_best_model = Path(os.path.join(self.pretrain_checkpoints, "best_model.pth"))
        self.torch_script = Path(os.path.join(self.pretrain_checkpoints, "best_model.pt"))
        self.pretrain_train_config = Path(os.path.join(self.pretrain_checkpoints, "train_configuration.txt"))

        self.specs_folder = os.path.join(dvc, "specs")
        self.specs_cells_folder = os.path.join(dvc, "specs", "cells")

        self.general_debug_folder = os.path.join(dvc, "debug", debug_tool_name)
        self.debug_folder = os.path.join(self.general_debug_folder, id)
        print("Debug folder: file://%s" % self.debug_folder)
        self.cleanup_debug()

    @staticmethod
    def create_dir(dir):
        if not os.path.exists(dir):
            Log.debug("Create dir %s" % dir)
            os.makedirs(dir, exist_ok=True)

    def cleanup_checkpoints(self, checkpoints):
        days = 90
        numseconds = days * 24 * 60 * 60
        now = time.time()
        Log.info("Cleanup %s" % checkpoints)
        for r, d, _ in os.walk(checkpoints):
            for folder in d:
                folder_path = os.path.join(r, folder)
                if os.path.exists(folder_path):
                    timestamp = os.path.getmtime(folder_path)
                    if now - numseconds > timestamp:
                        Log.warning("We removed checkpoint data from more than %f days ago: %s" % (days, folder_path))
                        shutil.rmtree(folder_path, ignore_errors=True)

    def cleanup_debug(self):
        days = 30
        numseconds = days * 24 * 60 * 60
        now = time.time()
        Log.info("Cleanup %s" % self.general_debug_folder)
        for r, d, _ in os.walk(self.general_debug_folder):
            for folder in d:
                folder_path = os.path.join(r, folder)
                if os.path.exists(folder_path):
                    timestamp = os.path.getmtime(folder_path)
                    if now - numseconds > timestamp:
                        Log.warning("We removed debug data from more than %f days ago: %s" % (days, folder_path))
                        shutil.rmtree(folder_path, ignore_errors=True)

        if os.path.exists(self.debug_folder):
            try:
                shutil.rmtree(self.debug_folder, ignore_errors=True)
            except:
                pass

        self.create_dir(self.debug_folder)

    def cleanup_training(self):
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir, ignore_errors=True)

        if not os.path.exists(self.logs):
            self.logs.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(self.checkpoints):
            self.checkpoints.mkdir(exist_ok=True, parents=True)

    def cleanup_prediction(self, enhance=False):
        if not enhance:
            if os.path.exists(self.prediction):
                shutil.rmtree(self.prediction, ignore_errors=True)
            if os.path.exists(self.prediction_grid):
                shutil.rmtree(self.prediction_grid, ignore_errors=True)

        if not os.path.exists(self.prediction):
            self.prediction.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(self.prediction_grid):
            self.prediction_grid.mkdir(exist_ok=True, parents=True)

    def cleanup_eval(self, enhance=False):
        if not enhance:
            if os.path.exists(self.qeval_points):
                shutil.rmtree(self.qeval_points, ignore_errors=True)
            if os.path.exists(self.qeval_tusimple):
                shutil.rmtree(self.qeval_tusimple, ignore_errors=True)

        if not os.path.exists(self.qeval_points):
            self.qeval_points.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(self.qeval_tusimple):
            self.qeval_tusimple.mkdir(exist_ok=True, parents=True)

    def generate_file_stub(self, file_name, idx=ImageIdx.DEFAULT, boxplot=False, suffix="", prefix=""):
        idx_string = idx.value

        if os.path.isabs(file_name):
            Log.warning("We shall create a file stub for %s" % file_name)

        splits = file_name.split("/")
        if len(splits) > 1:
            return os.path.join(*splits[:-1], prefix + splits[-1] + "_"
                                + (("_" + idx_string) if len(idx_string) > 0 else "")
                                + ("_bp" if boxplot else "")
                                + (("_" + str(suffix)) if len(str(suffix)) > 0 else ""))
        else:
            return prefix + splits[-1] + (("_" + idx_string) if len(idx_string) > 0 else "") \
                   + ("_bp" if boxplot else "") \
                   + (("_" + str(suffix)) if len(str(suffix)) > 0 else "")

    def generate_specs_file_name(self, dataset, split, anchor_vars, num_predictors, scale):
        return os.path.join(self.specs_folder,
                            "%s_%s_%s_%d_means_%dcs_anchors.yaml" % (
                                dataset, split, "_".join([str(a) for a in anchor_vars]), num_predictors, scale))

    def generate_specs_eval_file_name(self, dataset, split, anchor_vars, num_predictors, img_size, cell_size, anchors):
        return os.path.join(self.specs_folder,
                            "%s_%s_%s_%s_%d_%s_%d_means_anchors_eval.yaml" % (
                                dataset, split, "_".join([str(a) for a in anchor_vars]), str(anchors), cell_size[0],
                                f"{img_size[0]}x{img_size[1]}", num_predictors))

    def generate_specs_pkl_file_name(self, dataset, split, img_size, scale, prefix=""):
        return os.path.join(self.specs_folder, "%s%s_%s_%dx%d_%dcs_per_line_data.pkl" %
                            (prefix, dataset, split, img_size[0], img_size[1], scale))

    def generate_specs_pkl_per_cell_file_name(self, dataset, split, cell_size, img_size, idx=-1):
        return os.path.join(self.specs_cells_folder,
                            "%s_%s_%d_%dx%d_%d_per_cell_data.pkl" % (dataset, split, cell_size[0],
                                                                     img_size[0], img_size[1], idx))

    def generate_file_specific_debug_folder(self, file_name):
        return os.path.join(self.debug_folder, file_name)

    def generate_rec_field_image_file_path(self, file_name, cfg_name):

        path = os.path.join(self.prediction,
                            "%s_rf_%s.png" % (
                            "-".join(self.generate_file_stub(file_name).split("/")), os.path.basename(cfg_name)))
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        return path

    def generate_debug_image_file_path(self, file_name, idx, boxplot=False, suffix=""):

        if os.path.isabs(file_name):
            Log.warning("%s should not be abs path, but be inside the debug folder!" % file_name, level=1)

        file_path = os.path.join(self.debug_folder, self.generate_file_stub(file_name, idx, boxplot, suffix) + ".png")

        folder = os.path.dirname(file_path)

        with self._lock:
            if not os.path.isdir(folder):
                Log.info("Create %s" % folder)
                os.makedirs(folder)
        return file_path

    def generate_anchors_image_file_path(self, dataset, anchor_vars, num_predictors, anchors, scale):
        anchor_str = '_'.join([str(a) for a in anchor_vars])
        return os.path.join(self.specs_folder, f"{dataset}_{anchor_str}_"
        # "{scale}x{scale}_" # we do not need different anchors per scale 
                                               f"{num_predictors}_{anchors}_{scale}cs_anchors.png")

    def generate_anchor_image_file_path(self, file_name, anchors, anchor_vars, num_predictors, scale, **kwargs):

        if os.path.isabs(file_name):
            Log.warning("%s should not be abs path, but be inside the debug folder!" % file_name, level=1)

        stub = self.generate_file_stub(os.path.basename(file_name))
        anchor_str = '_'.join([str(a) for a in anchor_vars])
        file_path = os.path.join(self.anchors, os.path.dirname(file_name), "anchors",
                                 f"{stub}_{anchors}_{anchor_str}_p{num_predictors}_cs{scale}.png")

        folder = os.path.dirname(file_path)
        if not os.path.isdir(folder):
            Log.info("Create %s" % folder)
            os.makedirs(folder)
        return file_path

    def generate_loss_image_path(self, suffix=""):
        return os.path.join(self.debug_folder, "loss_" + suffix + ".png")

    def generate_prediction_torch_file_path(self, file_name):
        return os.path.join(self.prediction, file_name + ".torch")

    def generate_prediction_file_path(self, file_name):
        path = os.path.join(self.prediction, file_name + ".torch")
        self.create_dir(os.path.dirname(path))
        return path

    def generate_prediction_grid_file_path(self, file_name, swap):
        path = os.path.join(self.prediction_grid, file_name + ("_swap" if swap else "") + ".pkl")
        self.create_dir(os.path.dirname(path))
        return path

    def generate_prediction_csv_grid_file_path(self, file_name, swap, idx=ImageIdx.DEFAULT):
        idx_string = idx.value
        path = os.path.join(self.prediction_grid, file_name + ("_swap" if swap else "") + "_" + idx_string + ".csv")
        self.create_dir(os.path.dirname(path))
        return path

    def generate_prediction_csv_file_path(self, file_name, swap, idx=ImageIdx.DEFAULT):
        idx_string = idx.value
        path = os.path.join(self.prediction, file_name + ("_swap" if swap else "") + "_" + idx_string + ".csv")
        self.create_dir(os.path.dirname(path))
        return path

    def generate_confusion_matrix_img_path(self, epoch, tag="", idx=ImageIdx.DEFAULT):
        idx_string = idx.value
        path = os.path.join(self.prediction, "confusion_" + str(epoch) + "_" + tag + "_" + idx_string + ".png")
        self.create_dir(os.path.dirname(path))
        return path

    def generate_eval_image_file_path(self, file_name, idx, boxplot=False, suffix=""):
        path = os.path.join(self.qeval_points, self.generate_file_stub(file_name, idx, boxplot, suffix) + ".jpg")
        self.create_dir(os.path.dirname(path))
        return path

    def generate_eval_json_file_path(self, file_name, idx, boxplot=False):
        path = os.path.join(self.qeval_points, self.generate_file_stub(file_name, idx, boxplot) + ".json")
        self.create_dir(os.path.dirname(path))
        return path

    def generate_eval_label_file_path(self, file_name, idx, boxplot=False):
        path = os.path.join(self.qeval_points, self.generate_file_stub(file_name, idx, boxplot) + ".jpg")
        self.create_dir(os.path.dirname(path))
        return path

    def generate_epoch_model_path(self, epoch):
        path = os.path.join(self.checkpoints, 'ep' + str(epoch).zfill(4) + '_model.pth')
        self.create_dir(os.path.dirname(path))
        return path
