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
import os

import numpy as np
import torch
from tqdm import tqdm

from yolino.dataset.dataset_base import DatasetInfo
from yolino.utils.duplicates import LineDuplicates
from yolino.utils.enums import Dataset, Variables
from yolino.utils.logger import Log


class CULaneDataSet(DatasetInfo):

    @classmethod
    def height(self) -> int:
        return 544

    @classmethod
    def width(self) -> int:
        return 1632

    def __init__(self, split, args, augment=False, sky_crop=46, side_crop=4, load_only_labels=False,
                 show=False, load_full_dataset=False, lazy=False, store_lines=False, ignore_duplicates=False):
        super().__init__(Dataset.CULANE, split, args, sky_crop=sky_crop, side_crop=side_crop, augment=augment,
                         num_classes=5, train=88880, test=34690, val=9675,
                         override_dataset_path="/mrtstorage/datasets/public/CULane", load_only_labels=load_only_labels,
                         show=show, load_sequences=load_full_dataset, lazy=lazy, store_lines=store_lines,
                         ignore_duplicates=ignore_duplicates)

        # read file list txt 
        file = self.split + '.txt' if self.split == "test" else self.split + '_gt.txt'
        with open(os.path.join(self.dataset_path, "list", file), "r") as f:
            file_lines = f.readlines()
        f.close()

        self.labels = []
        self.markings_exist = []

        if split == "test":
            self.has_labels = False

        self.gather_files(file_lines)
        if len(self.file_names) == 0 and self.explicit_in_split:
            raise FileNotFoundError("Explicit filename %s not found in %s; files need to have the pattern of e.g. %s. "
                                    "We will now look for all files." % (
                                        split, os.path.join(self.dataset_path, self.explicit_in_split[0]),
                                        self.parse_txt_line(file_lines[0])[0]))

        self.img_path = self.dataset_path
        self.gt_path = self.dataset_path

        self.on_load()

    def check_img_size(self):
        if not np.all(math.isclose(np.divide(*self.args.img_size), self.height() / self.width())) or \
                not np.all(np.mod(self.args.img_size, 32) == 0):
            Log.warning("CULane together with Grid-YOLO on 32x32 px cells only accepts "
                        "an aspect ratio of %.2f. In addition the height and width must be dividable by 32. "
                        "You provided %s. "
                        "We suggest [128, 384], [288, 864], [544, 1632]." %
                        (self.height() / self.width(), str(self.args.img_size)))
            return False

        return True

    @classmethod
    def get_max_image_size(self):
        return (544, 1632)

    def gather_files(self, file_lines):
        # iterate all listed files
        txt_data = {}
        for i, line in tqdm(enumerate(file_lines), total=len(file_lines)):
            filename, _, txt_markings = self.parse_txt_line(line)

            if self.abort(count=len(txt_data)):
                break
            if self.skip(filename, i):
                continue

            txt_data[filename] = txt_markings

        if self.explicit_in_split is not None and len(self.explicit_in_split) > len(txt_data):
            raise FileNotFoundError("Could not find all explicit files, %d missing:\n%s\nWe only have\n%s"
                                    % (len(self.explicit_in_split) - len(txt_data),
                                       str(np.asarray(self.explicit_in_split)[
                                               np.where([e not in txt_data for e in self.explicit_in_split])]),
                                       str(self.explicit_in_split)))

        Log.debug("Found %d filenames" % (len(txt_data)))
        for i, (filename, txt_markings) in tqdm(enumerate(txt_data.items()), total=len(txt_data)):

            # check if specified files exist
            path = os.path.join(self.dataset_path, filename)
            if not self.correct(path):
                continue

            self.img_list.append(path)  # including main dataset path and folders
            self.file_names.append(filename)  # including folders

            # retrieve further labels?
            if not self.has_labels:
                continue

                # store the existance of four lane markings from left to right => always 4
            self.markings_exist.append(txt_markings)

            # read labels txt file
            path = os.path.join(self.dataset_path, os.path.splitext(filename)[0] + ".lines.txt")
            if not self.correct(path):
                Log.warning("We found an image without label [%s] - assuming test data." % filename)
                self.has_labels = False
                continue
            self.labels.append(path)

    def on_load(self):
        if len(self.file_names) != len(self.img_list):
            raise IndexError("Error reading data for %s. We ended up with %d filenames and %d images" % (
                self.enum, len(self.file_names), len(self.img_list)))

        if self.has_labels and len(self.file_names) != len(self.labels):
            raise IndexError("Error reading data for %s. We ended up with %d filenames and %d labels" % (
                self.enum, len(self.file_names), len(self.labels)))

        if self.has_labels and len(self.file_names) != len(self.markings_exist):
            raise IndexError("Error reading data for %s. We ended up with %d filenames and %d markings" % (
                self.enum, len(self.file_names), len(self.markings_exist)))

        super().on_load()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.load_only_labels:
            image = self.dummy_image([590, 1640])
        else:
            cv_image = self.__load_image__(idx)
            image = self.__make_torch__(cv_image)
            del cv_image

        if self.has_labels:
            lines, classes_per_line_one_hot = self.__get_labels__(idx)
        else:
            lines = self.dummy_lines()
            classes_per_line_one_hot = self.dummy_class()

        image, lines, params = self.__augment__(idx, image, lines)

        duplicates = LineDuplicates(filename=self.file_names[idx], grid_shape=self.args.grid_shape,
                                    num_predictors=self.args.num_predictors)
        grid_tensor, grid = self.__get_grid_labels__(torch.unsqueeze(lines, dim=0),
                                                     torch.unsqueeze(classes_per_line_one_hot,
                                                                     dim=0),
                                                     idx,
                                                     image=image, duplicates=duplicates)

        if idx == 0:
            Log.debug("Shapes from CULane:")
            Log.debug("\tImage: %s" % str(image.shape))  # (3, h, w)
            Log.debug("\tGrid Lines: %s" % str(grid_tensor.shape))  # (grid_shape, preds, coords)

        return image, grid_tensor, self.file_names[idx], duplicates.dict(), params

    def __get_labels__(self, idx):
        classes_one_hot = torch.zeros((4, self.coords[Variables.CLASS]), dtype=torch.float32)
        classes_one_hot[:, 0] = 1  # set all cells to background class

        # get lines of the file; only existing markings have coordinates, thus <= 4 lines
        with open(self.labels[idx]) as l:
            lines = l.readlines()
        l.close()

        max_num_markings = 75
        gridable_lines = torch.ones((4, max_num_markings, int(self.coords[Variables.GEOMETRY] / 2)),
                                    dtype=torch.float32) * torch.nan  # x, y

        one_idx = 0  # counts the index within all existing markings

        # iterate all markings left2, left1, right1, right2 => always 4 times
        for m_idx, m in enumerate(self.markings_exist[idx]):
            if m == 0:
                # marking does not exist => has no coords
                continue

            # get coordinates from markings txt file
            marking_coords = torch.tensor([float(x) for x in lines[one_idx].strip().split(" ")])
            marking_coords = marking_coords.reshape((-1, 2))
            marking_coords = self.culane2pixel(marking_coords)

            gridable_lines[m_idx] = torch.concat(
                [marking_coords, torch.ones((int(max_num_markings - len(marking_coords))), 2) * torch.nan])

            # store classification for left2, left1, right1, right2 in torch format
            onehot = torch.zeros(self.coords[Variables.CLASS])
            onehot[m_idx + 1] = 1
            classes_one_hot[m_idx] = onehot
            one_idx += 1

        if torch.max(gridable_lines) > self.args.img_size[1] * 1.1:
            raise ValueError("Scaling went wrong. We have labels with pixels up to %f, but the image is %s."
                             % (torch.max(gridable_lines).item(), str(self.args.img_size)))

        # coordinates of the lines (batch, instances, control points, ?) and class per line as onehot
        return gridable_lines, classes_one_hot

    # in case we want image classification as well
    def get_image_classes(self, grid_tensor):
        # classes as a one-hot vector for each image (batch_size, 5, cells, predictors)
        # num_cells, num_preds, num_coords = grid_tensor.shape
        image_class = grid_tensor[:, :, self.coords.get_position_of(Variables.CLASS)]
        if self.one_hot:
            image_class = torch.permute(image_class, (2, 0, 1))
        else:
            image_class = torch.unsqueeze(torch.argmax(image_class, dim=2), dim=0)  # indices

        return image_class

    def culane2pixel(self, line):
        new_line = torch.empty_like(line)
        new_line[:, 1] = line[:, 0]  # x -> width -> col
        new_line[:, 0] = line[:, 1]  # y -> height -> row
        return new_line

    def parse_txt_line(self, line):
        splits = line.strip().split(" ")

        if len(splits) == 1:
            gt = ""
            markings = []
        else:
            markings = np.array(splits[2:], dtype=np.float32)
            gt = splits[1][1:]
        return splits[0][1:], gt, markings


def polyline2linesegments(line):
    new_line = torch.zeros((line.shape[0] - 1, line.shape[1] * 2))
    for idx in range(len(line) - 1):
        new_line[idx] = torch.tensor([line[idx][0], line[idx][1], line[idx + 1][0], line[idx + 1][1]])
    return new_line
