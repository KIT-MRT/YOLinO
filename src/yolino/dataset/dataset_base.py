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
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset

from yolino.dataset.augmentation import DatasetTransformer
from yolino.grid.grid import Grid
from yolino.grid.grid_factory import GridFactory
from yolino.model.anchors import Anchor
from yolino.model.line_representation import LineRepresentation
from yolino.model.variable_structure import VariableStructure
from yolino.utils.duplicates import LineDuplicates
from yolino.utils.enums import Variables, CoordinateSystem, AnchorDistribution, LINE
from yolino.utils.logger import Log
from yolino.viz.plot import plot


class DatasetInfo(torchDataset, metaclass=ABCMeta):
    @classmethod
    def height(cls) -> int:
        pass

    @classmethod
    def width(cls) -> int:
        pass

    def __init__(self, enum, split, args, sky_crop: int, side_crop: int, augment, num_classes=0, train=-1, test=-1,
                 val=-1, norm_mean=None, norm_std=None,
                 override_dataset_path=None, load_only_labels=False, show=False,
                 load_sequences=False, lazy=False, ignore_duplicates=False, store_lines=False) -> None:
        Log.info("---- Load %s [%s] ----" % (enum, split))

        self.img_list = []
        self.file_names = []
        self.params_per_file = {}
        self.override_dataset_path = override_dataset_path
        self.allow_down_facing = False
        self.ignore_duplicates = ignore_duplicates

        self.args = args
        self.lazy = lazy
        self.explicit_in_split = self.args.explicit if split == args.split else None
        self.enum = enum
        self.load_only_labels = load_only_labels
        self.load_full = load_sequences  # load all images also non-labeled ones

        if split == "train" and not self.lazy:
            self.pkl_file = args.paths.generate_specs_pkl_file_name(dataset=args.dataset, split=split, scale=args.scale,
                                                                    img_size=args.img_size)

            self.store_lines = store_lines
            if os.path.exists(self.pkl_file):
                Log.error("Data is already generated in %s" % self.pkl_file)
                self.store_lines = False
        else:
            self.pkl_file = None
            self.store_lines = False

        if not self.lazy or (self.lazy and "training_variables" in args):
            self.coords = VariableStructure(num_classes=num_classes, vars_to_train=args.training_variables,
                                            line_representation_enum=args.linerep)

            self.sky_crop = sky_crop
            self.side_crop = side_crop
            self.augmentor = DatasetTransformer(args, sky_crop, side_crop, augment, norm_mean=norm_mean,
                                                norm_std=norm_std)

        self.dataset_path = self.get_dataset_path(enum.value, self.override_dataset_path)

        self.split = split
        self.__numbers__ = {"train": train, "val": val, "test": test}
        if self.args.max_n < 0:
            self.args.max_n = math.ceil(self.__numbers__[self.split] * 100 / max(1, args.subsample_dataset_rhythm))
        if self.args.max_n > 0:
            Log.info("Limit the search to a maximum of %d files" % self.args.max_n)

        self.plot = "plot" in self.args and self.args.plot
        self.show = show
        self.skipped = False
        self.has_labels = True  # should be set to false on errors

        if not self.lazy:
            Log.info("Gather data for %s for split %s" % (self.dataset_path, self.split))
            self.anchor_sorting = args.anchors != AnchorDistribution.NONE

            if self.allow_down_facing:
                self.anchors = Anchor.get(self.args, args.linerep, angle_range=2. * math.pi)
            else:
                self.anchors = Anchor.get(self.args, args.linerep)

            self.check_img_size()

    def get_dataset_path(self, enum_str, override_dataset_path):
        env_var = "DATASET_" + enum_str.upper()
        dataset_path = os.getenv(env_var)
        Log.info("Dataset path is set to %s=%s" % (env_var, dataset_path))
        if dataset_path is None or dataset_path == "":
            dataset_path = override_dataset_path
            Log.info("Dataset path is set to internal path=%s" % (dataset_path))
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(
                "Dataset path %s does not exist. Try to set with $%s" % (dataset_path, env_var))

        return dataset_path

    def abort(self, count=-1):
        if count >= 0:
            return count >= self.args.max_n
        else:
            return len(self.file_names) >= self.args.max_n

    def skip(self, filename, i):
        explicit_mode = self.explicit_in_split is not None
        if explicit_mode:
            filname_correct = np.any([e == filename for e in self.explicit_in_split])
            folder_correct = np.any([e in filename for e in self.explicit_in_split])

            if not filname_correct and not folder_correct:  # and is_train_split:
                return True
        else:
            if self.args.subsample_dataset_rhythm > 0 and i % self.args.subsample_dataset_rhythm != 0:
                return True

        return False

    def correct(self, path):
        if not os.path.isfile(path):
            msg = "We found at least one invalid path at %s" % path

            if self.skipped:
                return False
            if self.args.ignore_missing:
                Log.warning(msg)
                self.skipped = True
                return False
            else:
                raise FileNotFoundError(msg)

        return True

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    def __get_grid_labels__(self, lines, variables, idx, image=None, duplicates: LineDuplicates = None):
        if duplicates is None:
            duplicates = LineDuplicates(filename=self.file_names[idx], grid_shape=self.args.grid_shape,
                                        num_predictors=self.args.num_predictors)

        # wrap the labels in a grid structure for easy access
        try:
            grid, errors = GridFactory.get(lines, variables,
                                           CoordinateSystem.UV_CONTINUOUS, self.args, input_coords=self.coords,
                                           variables_have_conf=False, allow_down_facing=self.allow_down_facing,
                                           anchors=self.anchors, plot_image=False)  # self.plot)
        except ValueError as e:
            Log.error("Label error in %s" % self.file_names[idx])
            raise e

        if self.plot and len(grid) > 0:
            path = os.path.join(self.args.paths.debug_folder, "%s.png" % self.file_names[idx])

            points_coords = self.coords.clone(LINE.POINTS)

            full_size_image = self.get_full_size_image(filename=self.file_names[idx])
            plot(lines=grid.get_image_lines(coords=points_coords, image_height=full_size_image.shape[1]), name=path,
                 image=full_size_image, coords=self.coords, show_grid=True,
                 cell_size=grid.get_cell_size(full_size_image.shape[1]), show=self.show, threshold=0.5)

        for batch in errors:
            Log.info("Grid found an unusual error in %s [b=%d]:\n%s" % (self.file_names[idx], batch, errors[batch]))
        try:
            grid_tensor = grid.tensor(coords=self.coords, one_hot=True, set_conf=1,
                                      anchors=self.anchors, convert_to_lrep=self.args.linerep,
                                      as_offset=self.args.offset, duplicates=duplicates, filename=self.file_names[idx],
                                      ignore_duplicates=self.ignore_duplicates, store_lines=self.store_lines,
                                      pkl_file=self.pkl_file)
        except ValueError as e:
            path = os.path.join(self.args.paths.debug_folder, "%s.png" % idx)
            print(path)
            points_coords = self.coords.clone(LINE.POINTS)
            plot(grid.get_image_lines(coords=points_coords, image_height=self.args.img_size[0]),
                 path, None, self.coords,
                 show_grid=True,
                 cell_size=grid.get_cell_size(self.args.img_size[0]))
            raise e

        if not self.ignore_duplicates:
            duplicates.summarize(grid.shape[0])
        return grid_tensor, grid

    def __augment__(self, idx, image, lines):
        image, lines, params = self.augmentor(image, lines, self.file_names[idx])
        self.params_per_file[self.file_names[idx]] = params
        return image, lines, params

    def num_classes(self):
        return self.coords[Variables.CLASS]

    def num_geom_coords(self):
        return self.coords[Variables.GEOMETRY]

    def num_conf(self):
        return self.coords[Variables.CONF]

    def num_coords(self, one_hot=True):
        return self.coords.get_length(one_hot=one_hot)

    def dummy_image(self, size=None):
        if size is None:
            return torch.ones((3, self.args.img_size[0], self.args.img_size[1]), dtype=torch.float32)  # as BGR
        else:
            return torch.ones((3, size[0], size[1]), dtype=torch.float32)  # as BGR

    def dummy_lines(self):
        return torch.empty((0, 0, int(self.coords[Variables.GEOMETRY] / 2)), dtype=torch.float32)  # x, y

    def dummy_class(self):
        shape = (0, self.num_classes())
        return torch.empty(shape, dtype=torch.float32)  # onehot-class

    def empty_grid(self):
        grid = Grid(img_height=self.args.img_size, args=self.args)

        return grid.tensor(coords=self.coords, fill_nan=True, set_conf=1, convert_to_lrep=self.args.linerep)

    def full_grid(self, train_vars=True, random=False):

        num_vars = self.coords.num_vars_to_train() if train_vars else self.coords.get_length()
        array = torch.zeros((self.args.num_predictors, num_vars), dtype=torch.float32)

        if self.anchor_sorting:
            raise NotImplementedError()
        else:
            linerep = LineRepresentation.get(self.args.linerep)
            if random:
                origin = torch.rand(linerep.num_params)
            else:
                origin = linerep.get_dummy(use_torch=True)
            geom_val = torch.tile(origin, dims=[self.args.num_predictors, 1])

        idx = torch.randint(self.coords[Variables.CLASS], (1,)) if random else 0
        class_val = torch.zeros((self.coords[Variables.CLASS]))
        class_val[idx] = 1

        for i in range(self.args.num_predictors):
            if train_vars:
                if Variables.GEOMETRY in self.coords.train_vars():
                    array[i, self.coords.get_position_within_prediction(Variables.GEOMETRY)] = geom_val[i]
                if Variables.CLASS in self.coords.train_vars():
                    array[i, self.coords.get_position_within_prediction(Variables.CLASS)] = class_val
                if Variables.CONF in self.coords.train_vars():
                    array[i, self.coords.get_position_within_prediction(Variables.CONF)] = torch.tensor([1.])
            else:
                if self.coords[Variables.GEOMETRY] > 0:
                    array[i, self.coords.get_position_of(Variables.GEOMETRY)] = geom_val[i]
                if self.coords[Variables.CLASS] > 0:
                    array[i, self.coords.get_position_of(Variables.CLASS)] = class_val
                if self.coords[Variables.CONF] > 0:
                    array[i, self.coords.get_position_of(Variables.CONF)] = torch.tensor([1.])

        data = torch.tile(array, dims=(int(np.prod(self.args.grid_shape)), 1, 1))
        if random:
            data[:, :, -1] = torch.rand((int(np.prod(self.args.grid_shape)), self.args.num_predictors))
        return data

    def is_available(self):
        Log.warning("Dataset %s is loading from path %s for split %s" % (self.enum, self.dataset_path, self.split))
        return os.path.isdir(self.dataset_path)

    def __load_image__(self, idx):
        import cv2
        image = cv2.imread(self.img_list[idx])
        if image is None:
            raise FileNotFoundError(self.img_list[idx])

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def __construct_filename__(self, filename):
        return os.path.join(self.dataset_path, filename)

    def get_full_size_image(self, idx=-1, filename=None):
        import cv2
        if idx > 0:
            full_path = self.img_list[idx]
            filename = self.file_names[idx]
        elif filename is not None:
            full_path = self.__construct_filename__(filename)
        else:
            raise AttributeError("We either need an index or a filename")

        if not os.path.exists(full_path):
            raise FileNotFoundError(full_path)

        image = cv2.imread(full_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = torch.from_numpy(image / 255.).permute(2, 0, 1).contiguous().float()

        if filename in self.params_per_file:
            image, _, _ = self.augmentor.reproduce(image=image, uv_lines=None, filename=filename,
                                                   params=self.params_per_file[filename],
                                                   target_size=self.get_max_image_size())

        else:
            raise ValueError("No augmentation params found for %s: %s" % (filename, self.params_per_file))
        return image

    def __make_torch__(self, cv_image):
        # Make torch image
        torch_image = torch.from_numpy(cv_image / 255.).permute(2, 0, 1).contiguous().float()
        if torch_image.shape[0] != 3:
            raise ValueError("Image shape should be (3, h, w), but we have %s" % str((torch_image.shape)))

        return torch_image

    def on_load(self):
        if self.lazy:
            return

        Log.info("Found %d files for split %s" % (len(self), self.split))
        if len(self) == 0:
            raise FileNotFoundError("Could not find any data for %s and %s split" % (self.enum, self.split))

        if self.explicit_in_split is None and self.__numbers__[self.split] >= 0 and len(self) != self.__numbers__[
            self.split] \
                and len(self) != self.args.max_n:
            Log.warning("Your dataset is smaller than expected %s. We have only %d images for %s found in %s." % (
                self.__numbers__, len(self), self.split, self.dataset_path))
            if not self.args.ignore_missing:
                raise FileNotFoundError(
                    f"We found {len(self)} files, this is smaller than expected ({self.__numbers__[self.split]}!")

        if len(self) < self.args.batch_size:
            self.args.batch_size = len(self)
            Log.warning(
                "ATTENTION: your dataset is smaller than your batch size. We adapt the batch_size=%s accordingly." % self.args.batch_size)

    def __str__(self) -> str:
        return "%s for %s split: %d items" % (self.enum, self.split, len(self))

    @abstractmethod
    def get_max_image_size(self):
        Log.error("You should not use the dataset base!")
        pass

    @classmethod
    def get_img_width(cls, height) -> int:
        return math.ceil(height * cls.width() / cls.height())

    @abstractmethod
    def check_img_size(self):
        pass
