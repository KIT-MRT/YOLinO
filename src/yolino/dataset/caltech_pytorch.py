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

import torchvision
import torchvision.transforms as transforms
from yolino.dataset.dataset_base import DatasetInfo
from yolino.utils.enums import Dataset


class CaltechDataSet(DatasetInfo):
    def get_max_image_size(self):
        return 337, 510

    @classmethod
    def height(self) -> int:
        return 337

    @classmethod
    def width(self) -> int:
        return 510

    def check_img_size(self):
        return True

    def __init__(self, split, args, augment, load_only_labels, show=False, load_full_dataset=False, store_lines=False):
        if not os.path.exists("tmp/caltech"):
            os.makedirs("tmp/caltech")
        super().__init__(Dataset.CALTECH, split, args, sky_crop=0, side_crop=0, augment=augment, num_classes=0,
                         override_dataset_path='tmp/caltech', load_only_labels=load_only_labels, show=show,
                         load_sequences=load_full_dataset, store_lines=store_lines)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(self.args.img_size)])

        if not self.lazy:
            self.__caltech__ = torchvision.datasets.Caltech101(root=self.dataset_path,
                                                               transform=self.transform, download=True)
        self.has_labels = False
        self.on_load()

    def __len__(self):
        return len(self.__caltech__)

    def __getitem__(self, idx):
        image, _ = self.__caltech__.__getitem__(idx)

        return image, self.empty_grid(), str(self.enum) + str(idx), 0
