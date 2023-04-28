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
import math
import os

import numpy as np
import torch
from tqdm import tqdm
from yolino.dataset.dataset_base import DatasetInfo
from yolino.utils.duplicates import LineDuplicates
from yolino.utils.enums import Dataset
from yolino.utils.logger import Log


def is_sorted(data, down=True):
    if down:
        p_i = max(data) + 1
    else:
        p_i = min(data) - 1

    for i in range(len(data)):
        if math.isnan(data[i]) or data[i] < 0:
            continue

        if (down and data[i] > p_i) or (not down and data[i] < p_i):
            return i
        p_i = data[i]
    return -1


class TusimpleDataset(DatasetInfo):
    @classmethod
    def height(self) -> int:
        return 640

    @classmethod
    def width(self) -> int:
        return 1280

    def __init__(self, split, args, augment=False, sky_crop=80, side_crop=0, load_only_labels=False,
                 show=False, load_full_dataset=False, lazy=False, ignore_duplicates=False, store_lines=False):
        super().__init__(Dataset.TUSIMPLE, split, args, sky_crop=sky_crop, side_crop=side_crop, augment=augment,
                         num_classes=0, train=2858 + 410, test=2782, val=358,
                         override_dataset_path="/mrtstorage/datasets/public/tusimple_lane_detection",
                         load_only_labels=load_only_labels, show=show, load_sequences=load_full_dataset, lazy=lazy,
                         ignore_duplicates=ignore_duplicates, store_lines=store_lines)

        self.lanes = []
        self.h_samples = []
        self.file_names = []
        self.img_list = []  # including main dataset path and folders

        self.gather_files()

    def gather_files(self):
        if self.lazy:
            return

        if self.split == "train":
            for kind in ["0313", "0601"]:
                label_file_path = os.path.join(self.dataset_path, "train_set", "label_data_" + kind + ".json")
                example_filename = self.read_label_file(label_file_path)
        elif self.split == "val":
            label_file_path = os.path.join(self.dataset_path, "train_set", "label_data_0531.json")
            example_filename = self.read_label_file(label_file_path)
        elif self.split == "test":
            label_file_path = os.path.join(self.dataset_path, "test_label.json")
            example_filename = self.read_label_file(label_file_path)

        if len(self.file_names) == 0 and self.explicit_in_split is not None:
            raise FileNotFoundError("Explicit filename %s not found. Files need to have the pattern of e.g. %s. "
                                    "We will now look for all files." % (self.explicit_in_split, example_filename))
        self.on_load()

    def read_label_file(self, label_file_path, initial_i=0):
        try:
            with open(label_file_path, "r") as f:
                data = [json.loads(line) for line in f.readlines()]
            f.close()
        except json.decoder.JSONDecodeError as e:
            Log.error("JSON error %s with file %s" % (e.msg, label_file_path))
            raise json.decoder.JSONDecodeError("file://%s" % label_file_path, doc=e.doc, pos=e.pos) from e

        for i, entry in tqdm(enumerate(data), desc=os.path.basename(label_file_path), initial=initial_i,
                             total=initial_i + len(data)):
            if self.abort():
                break

            if self.skip(entry["raw_file"], initial_i + i):
                continue

            image_file_path = entry["raw_file"]

            full_path = self.__construct_filename__(image_file_path)
            if not self.correct(full_path):
                continue

            self.img_list.append(full_path)
            self.file_names.append(image_file_path)
            self.lanes.append(entry["lanes"])  # TODO: maybe we want the actual direction and thus invert all lanes
            self.h_samples.append(entry["h_samples"])  # always 56? nope!

            if self.load_full:
                folder = os.path.dirname(full_path)
                subfolder = os.path.dirname(image_file_path)
                for _, _, filenames in os.walk(folder):
                    for f in filenames:
                        if self.abort():
                            print("break on %s" % f)
                            break
                        if "20.jpg" in f:
                            continue
                        else:
                            full_path = os.path.join(folder, f)
                            self.img_list.append(full_path)
                            self.file_names.append(os.path.join(subfolder, f))
                            self.lanes.append([])
                            self.h_samples.append([])

        return entry["raw_file"]

    def on_load(self):
        if len(self.file_names) != len(self.img_list):
            raise IndexError("Error reading data for %s. We ended up with %d filenames and %d images" % (
                self.enum, len(self.file_names), len(self.img_list)))

        if len(self.file_names) != len(self.lanes):
            raise IndexError("Error reading data for %s. We ended up with %d filenames and %d labels" % (
                self.enum, len(self.file_names), len(self.lanes)))

        if len(self.file_names) != len(self.h_samples):
            raise IndexError("Error reading data for %s. We ended up with %d filenames and %d markings" % (
                self.enum, len(self.file_names), len(self.h_samples)))

        super().on_load()

    def __len__(self):
        return len(self.img_list)

    def __get_labels__(self, idx):
        gridable_lines = torch.ones((5, len(self.h_samples[idx]), 2), dtype=torch.float32) * torch.nan

        # Tusimple GT runs from horizon to bottom of the image.
        # We want to encode the driving direction in the arrows and thus reverse the GT.
        for i in range(len(self.lanes[idx])):
            gridable_lines[i, :, 1] = torch.tensor(list(reversed(self.lanes[idx][i])))
            gridable_lines[i, :, 0] = torch.tensor(list(reversed(self.h_samples[idx])))

        gridable_lines[gridable_lines[:, :, 1] < 0] = torch.tensor([torch.nan, torch.nan])

        if False:
            # import matplotlib
            # matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.close("all")
            plt.clf()

            fig, ax = plt.subplots()
            for line in gridable_lines:
                for i in range(1, len(line)):
                    if torch.any(torch.isnan(line[i - 1:i + 1])):
                        continue
                    x = line[i - 1, 1]
                    dx = line[i, 1] - x
                    y = line[i - 1, 0]
                    dy = line[i, 0] - y
                    plt.plot([x, x + dx], [y, y + dy])
                    plt.scatter(x + dx, y + dy, marker="*")

            start, end = ax.get_ylim()
            ax.yaxis.set_ticks(np.arange(start, end, 32))
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.arange(start, end, 32))

            plt.grid()

            path = "/tmp/raw_tusimple.png"
            Log.info("Plot to file://%s" % path)
            plt.savefig(path)
            if self.show:
                plt.show()
            plt.close("all")
            plt.clf()

        return gridable_lines

    def __construct_filename__(self, filename):
        return os.path.join(self.dataset_path, "train_set", filename)

    def __getitem__(self, idx):
        if self.load_only_labels:
            image = self.dummy_image([720, 1280])
        else:
            cv_image = self.__load_image__(idx)
            image = self.__make_torch__(cv_image)
            del cv_image

        lines = self.__get_labels__(idx)
        image, lines, params = self.__augment__(idx, image, lines)

        try:
            duplicates = LineDuplicates(filename=self.file_names[idx], grid_shape=self.args.grid_shape,
                                        num_predictors=self.args.num_predictors)
            grid_tensor, grid = self.__get_grid_labels__(torch.unsqueeze(lines, dim=0), [],
                                                         idx, image=image,
                                                         duplicates=duplicates)
        except ValueError as e:
            Log.error("Error in %s" % (self.img_list[idx]))
            raise e

        if idx == 0:
            Log.debug("Shapes from TuSimple:")
            Log.debug("\tImage: %s" % str(image.shape))  # (3, h, w)
            Log.debug("\tGrid Lines: %s" % str(grid_tensor.shape))  # (grid_shape, preds, coords)

        return image, grid_tensor, self.file_names[idx], duplicates.dict(), params

    def check_img_size(self):
        if not np.all(math.isclose(np.divide(*self.args.img_size), self.height() / self.width())) or \
                not np.all(np.mod(self.args.img_size, 32) == 0):
            # 1280 x 720 px
            Log.warning("Tusimple together with Grid-YOLO on 32x32 px cells only accepts "
                        "an aspect ratio of %f. In addition the height and width must be dividable by 32. "
                        "You provided %s. "
                        "We suggest 320x640 or 640x1280." %
                        (self.height() / self.width(), str(self.args.img_size)))
            return False

        return True

    @classmethod
    def get_max_image_size(self):
        return 640, 1280
