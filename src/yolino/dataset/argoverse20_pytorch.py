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
import math
import os.path

import numpy as np
import torch
from tqdm import tqdm
from yolino.dataset.dataset_base import DatasetInfo
from yolino.utils.duplicates import LineDuplicates
from yolino.utils.enums import ColorStyle
from yolino.utils.enums import Dataset, Variables, ImageIdx, CoordinateSystem
from yolino.utils.logger import Log
from yolino.viz.plot import plot


class Argoverse2Dataset(DatasetInfo):
    def __init__(self, split, args, augment=False, sky_crop=2048 - 1536, side_crop=int((1550 - 1536) / 2),
                 load_only_labels=False, ignore_duplicates=False, show=False, load_full_dataset=False, lazy=False,
                 store_lines=False):
        super().__init__(Dataset.ARGOVERSE2, split, args, sky_crop=sky_crop, side_crop=side_crop, augment=augment,
                         num_classes=0, train=11240, val=2404,  # expecting preprocessing with -sdr 20
                         override_dataset_path="/mrtstorage/users/meyer/02_data/argoverse20",
                         load_only_labels=load_only_labels, show=show, load_sequences=load_full_dataset, lazy=lazy,
                         norm_mean=[0.4338504672050476, 0.4435146152973175, 0.47162434458732605],
                         norm_std=[0.2141847014427185, 0.21603362262248993, 0.24518834054470062],
                         ignore_duplicates=ignore_duplicates, store_lines=store_lines)

        self.dataset_img_path = self.get_dataset_path(str(Dataset.ARGOVERSE2) + "_IMG",
                                                      "/mrtstorage/datasets/public/argoverse20")
        self.allow_down_facing = True

        # filename should look like test/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/sensors/cameras/ring_front_center/315971450049927211.jpg
        self.file_names = []

        # should look like /mrtstorage/datasets/tmp/argoverse20/sensor/test/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/sensors/cameras/ring_front_center/315971450049927211.jpg
        self.img_list = []
        self.log_ids = {}
        self.lanes = []

        self.gather_files()

    def __construct_filename__(self, filename):
        # filename should look like test/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/sensors/cameras/ring_front_center/315971450049927211.jpg
        return os.path.join(self.dataset_img_path, "sensor", filename)

    def __construct_filename_from_label__(self, filename):
        # filename should look like test/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/sensors/cameras/ring_front_center/315971450049927211.jpg
        return os.path.join(self.dataset_img_path, "sensor", os.path.splitext(filename)[0] + ".jpg")

    def __construct_label_filename__(self, img_filename):
        # filename should look like test/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/sensors/cameras/ring_front_center/315971450049927211.jpg
        return os.path.join(self.dataset_path, "sensor", os.path.splitext(img_filename)[0] + ".npy")

    def load_label(self, full_path):
        try:
            self.lanes.append(torch.tensor(np.load(full_path), dtype=torch.float).round())
        except ValueError as e:
            Log.error(e)
            Log.error(f"{full_path} is maybe not a proper numpy array?")
            raise e

        assert (os.path.relpath(self.dataset_img_path, self.img_list[-1]) == os.path.relpath(self.dataset_path,
                                                                                             full_path))

    def gather_files(self):
        if self.lazy:
            return

        base_path = os.path.join(self.dataset_path, "sensor")
        image_base_path = os.path.join(self.dataset_img_path, "sensor")

        Log.warning("Load labels from %s" % base_path)
        Log.warning("Load images from %s" % image_base_path)
        index = 0

        walking_folder = ""
        if self.explicit_in_split is not None and len(self.explicit_in_split) == 1:
            walking_folder = os.path.join(base_path, self.split, self.explicit_in_split[0])
            Log.warning("Try walk in %s" % walking_folder)
            split = self.explicit_in_split[0].split("/")
            if not os.path.isdir(walking_folder):
                walking_folder = os.path.join(base_path, self.split, split[0])
                Log.warning("Try walk in %s" % walking_folder)
            if not os.path.isdir(walking_folder):
                walking_folder = os.path.join(base_path, split[0], split[1])
                Log.warning("Try walk in %s" % walking_folder)

        if not os.path.isdir(walking_folder):
            walking_folder = os.path.join(base_path, self.split)
            Log.warning("Try walk in %s" % walking_folder)

        for root, dirs, files in tqdm(os.walk(walking_folder),
                                      desc=f"Walk folders .../{'/'.join(walking_folder.split('/')[-2:])}",
                                      total=self.__numbers__[self.split]):
            if self.abort(count=len(self.file_names)):
                break

            for f in files:
                if not f.endswith(".npy"):
                    if f.endswith(".jpg"):
                        Log.error(f"We found .jpg files ({f}). Is {walking_folder} the correct label folder? "
                                  "Please set $DATASET_ARGO2_IMG and $DATASET_ARGO2 to the correct path.")
                        exit(1)
                    continue
                if len(f) != 22:
                    continue
                if self.abort(count=len(self.file_names)):
                    break

                full_label_path = os.path.join(root, f)
                label_file_path = os.path.relpath(full_label_path, base_path)
                image_file_path = os.path.splitext(label_file_path)[0] + ".jpg"
                full_path = self.__construct_filename__(image_file_path)

                if self.skip(image_file_path, index):
                    index += 1
                    continue

                if not self.correct(full_path):
                    Log.error(f"Image is missing for {full_path}")
                    continue

                if not self.correct(full_label_path):
                    continue

                self.file_names.append(image_file_path)
                self.img_list.append(full_path)

                self.load_label(full_label_path)
                index += 1

        if self.explicit_in_split is not None and len(self.file_names) < len(self.explicit_in_split):
            raise FileNotFoundError("Not all explicit filenames found. Files need to have the pattern of e.g. %s. "
                                    "We will now look for all files. "
                                    % (
                                            "%s/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68/sensors/cameras/ring_front_center/315971450049927211.jpg" % self.split) +
                                    "We only found %s" % self.file_names)

        self.on_load()

    def on_load(self):
        if len(self.file_names) != len(self.img_list):
            raise IndexError("Error reading data for %s. We ended up with %d filenames and %d images" % (
                self.enum, len(self.file_names), len(self.img_list)))

        if len(self.file_names) != len(self.lanes):
            raise IndexError("Error reading data for %s. We ended up with %d filenames and %d labels" % (
                self.enum, len(self.file_names), len(self.lanes)))

        super().on_load()

    def __len__(self):
        return len(self.img_list)

    def __get_labels__(self, idx):
        t = self.lanes[idx]
        return t

    def __getitem__(self, idx):
        if self.load_only_labels:
            image = self.dummy_image([2048, 1550])
        else:
            cv_image = self.__load_image__(idx)
            image = self.__make_torch__(cv_image)

        lines = self.__get_labels__(idx)
        if self.args.plot:
            name = self.args.paths.generate_debug_image_file_path(self.file_names[idx], ImageIdx.LABEL, suffix="raw")
            plot(torch.unsqueeze(lines, 0), name, image, show_grid=True, coords=self.coords,
                 colorstyle=ColorStyle.ORIENTATION,  # could also be ID
                 coordinates=CoordinateSystem.UV_CONTINUOUS, epoch=-1, tag="raw",
                 imageidx=ImageIdx.LABEL, cell_size=[self.args.cell_size[0] * 2, self.args.cell_size[1] * 2])

        image, lines, params = self.__augment__(idx, image, lines)

        try:
            duplicates = LineDuplicates(filename=self.file_names[idx], grid_shape=self.args.grid_shape,
                                        num_predictors=self.args.num_predictors)
            if self.coords[Variables.CLASS] == 0:
                one_hot_class = []
            else:
                one_hot_class = torch.zeros((1, 100, 100))
                one_hot_class[0].fill_diagonal_(1)

                if len(lines) > 100:
                    raise ValueError(len(lines))

                one_hot_class = one_hot_class[:, 0:len(lines)]
            grid_tensor, grid = self.__get_grid_labels__(torch.unsqueeze(lines, dim=0), one_hot_class,
                                                         idx, image=image, duplicates=duplicates)
        except Exception as e:
            Log.error("Error in %s" % (self.img_list[idx]))
            raise e

        if idx == 0:
            Log.debug("Shapes from Argoverse:")
            Log.debug("\tImage: %s" % str(image.shape))  # (3, h, w)
            Log.debug("\tGrid Lines: %s" % str(grid_tensor.shape))  # (grid_shape, preds, coords)

        # FIXME: this is weird but helps
        if self.ignore_duplicates:
            duplicate_dict = {}
        else:
            duplicate_dict = {
                k: v.shape if k == "cells" else (v.type(torch.FloatTensor) if type(v) == torch.Tensor
                                                 else torch.tensor(v, dtype=torch.float))
                for k, v in duplicates.dict().items()}

        return image, grid_tensor, self.file_names[idx], duplicate_dict, params

    @classmethod
    def get_max_image_size(self):
        return (1536, 1536)

    @classmethod
    def height(self) -> int:
        return 1536

    @classmethod
    def width(self) -> int:
        return 1536

    def check_img_size(self):
        if not np.all(math.isclose(np.divide(*self.args.img_size), self.height() / self.width())) or \
                not np.all(np.mod(self.args.img_size, 32) == 0):
            Log.warning("Argoverse together with Grid-YOLO on 32x32 px cells only accepts "
                        "an aspect ratio of %f. In addition the height and width must be dividable by 32. "
                        "You provided %s. "
                        "We suggest 384x384, 768x768 or %dx%d." %
                        (self.height() / self.width(), str(self.args.img_size), self.height(), self.width()))
            return False

        return True
