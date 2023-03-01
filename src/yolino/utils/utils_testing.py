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
import pickle
import random
import shutil

import cv2
import numpy as np
import torch

from yolino.utils.logger import Log
from yolino.utils.paths import Paths
from yolino.viz.plot import plot_style_grid


def add_noise(self, polylines, sigma=3, num=7):
    noise = []
    for line_id in range(0, len(polylines)):

        for line_segment in polylines[line_id]:
            for i in range(0, num):
                noise_start_x = random.gauss(0, sigma)
                noise_start_y = random.gauss(0, sigma)
                noise_end_x = random.gauss(0, sigma)
                noise_end_y = random.gauss(0, sigma)

                if len(noise) <= line_id:
                    noise.append([])
                    noise[line_id] = [(np.array(line_segment[0]) + (noise_start_x, noise_start_y),
                                       np.array(line_segment[1]) + (noise_end_x, noise_end_y))]
                else:
                    noise[line_id].append((np.array(line_segment[0]) + (noise_start_x, noise_start_y),
                                           np.array(line_segment[1]) + (noise_end_x, noise_end_y)))
    return noise


def setupPaths(args, filename, kind):
    base_folder = os.path.join(os.getcwd(), "test_data")
    paths = Paths(args.dvc, split=args.split, debug_tool_name="unittest", keep_checkpoints=args.keep)

    if filename:
        debug_folder = os.path.join(base_folder, "debug", filename, kind)
        if os.path.exists(debug_folder):
            shutil.rmtree(debug_folder)
        os.makedirs(debug_folder)

        return paths, debug_folder
    else:
        return paths, None


def load_image(img_size, filename, paths):
    image_path = paths.generate_eval_image_npy_file_import_path(filename)
    Log.info("Load image from %s" % image_path)
    img = np.load(image_path, allow_pickle=True)
    img = cv2.resize(img, dsize=(img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
    return np.asarray(img)


def load_prediction(args, coords, filename, paths, image):
    npy_path = paths.generate_prediction_file_path(filename)
    Log.info("Load prediction from %s" % npy_path)
    prediction = torch.load(npy_path)

    try:
        grid_path = paths.generate_prediction_grid_file_path(filename, swap=True)
        Log.info("Try to load swapped prediction grid from %s" % grid_path)
        with open(grid_path, "rb") as f:
            prediction_grid = pickle.load(f)
    except:
        grid_path = paths.generate_prediction_grid_file_path(filename, swap=False) + "_swap.pkl"
        Log.info("Failed... Try to load swapped prediction grid from %s" % grid_path)
        with open(grid_path, "rb") as f:
            prediction_grid = pickle.load(f)
    plot_style_grid(prediction_grid.get_image_lines(coords=coords, image_height=image.shape[0]),
                    os.path.join(args.debug_folder, filename + "_raw_prediction_points.png"), image,
                    show_grid=True, threshold=0.5, cell_size=prediction_grid.get_cell_size())
    return prediction, prediction_grid
