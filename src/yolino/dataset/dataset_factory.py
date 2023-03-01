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

from torch.utils.data import DataLoader

from yolino.dataset.argoverse20_pytorch import Argoverse2Dataset
from yolino.dataset.dataset_base import DatasetInfo
from yolino.utils.enums import Dataset, Network, LINE
from yolino.utils.logger import Log


class DatasetFactory:
    from yolino.dataset.tusimple_pytorch import TusimpleDataset
    from yolino.dataset.caltech_pytorch import CaltechDataSet
    from yolino.dataset.cifar10_pytorch import CifarDataSet
    from yolino.dataset.culane_pytorch import CULaneDataSet
    datasets = {
        Dataset.CULANE: CULaneDataSet,
        Dataset.CIFAR: CifarDataSet,
        Dataset.CALTECH: CaltechDataSet,
        Dataset.TUSIMPLE: TusimpleDataset,
        Dataset.ARGOVERSE2: Argoverse2Dataset,
    }

    @classmethod
    def __str__(self) -> str:
        return str(self.datasets.keys())

    @classmethod
    def get_coords(self, split, args):
        if not args.dataset in DatasetFactory.datasets:
            raise NotImplementedError("We did not set class for %s" % args.dataset)

        dataset_class = DatasetFactory.datasets[args.dataset]
        coords = dataset_class(split, args, lazy=True).coords
        return coords

    @classmethod
    def get_path(self, split, args):
        if not args.dataset in DatasetFactory.datasets:
            raise NotImplementedError("We did not set class for %s" % args.dataset)

        dataset_class = DatasetFactory.datasets[args.dataset]
        path = dataset_class(split, args, lazy=True).dataset_path
        img_path = dataset_class(split, args, lazy=True).dataset_img_path
        return path, img_path

    @classmethod
    def get_img_size(self, dataset, img_height):
        if not dataset in DatasetFactory.datasets:
            raise NotImplementedError("We did not setup %s" % dataset)

        dataset_class = DatasetFactory.datasets[dataset]
        width = dataset_class.get_img_width(img_height)
        return [img_height, width]

    @classmethod
    def get_max_image_size(cls, dataset):
        if not dataset in DatasetFactory.datasets:
            raise NotImplementedError("We did not setup %s" % dataset)

        dataset_class = DatasetFactory.datasets[dataset]
        img_size = dataset_class.get_max_image_size()
        return img_size

    @classmethod
    def get(self, dataset_enum: Dataset, only_available, split, args, shuffle, augment, load_only_labels=False,
            show=False, load_full=False, ignore_duplicates=False, store_lines=False) -> (DatasetInfo, DataLoader):
        if dataset_enum in DatasetFactory.datasets:
            dataset_class = DatasetFactory.datasets[dataset_enum]
            dataset = dataset_class(split=split, args=args, augment=augment, load_only_labels=load_only_labels,
                                    show=show, load_full_dataset=load_full, ignore_duplicates=ignore_duplicates,
                                    store_lines=store_lines)

            if dataset.is_available():
                Log.debug("Load data from %s with batch=%d" % (dataset_enum, args.batch_size))
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=True,
                                    num_workers=args.loading_workers, pin_memory=args.gpu)
                return dataset, loader
            else:
                if only_available:
                    raise FileNotFoundError("Could not find the data for %s" % (dataset_enum))
                else:
                    return dataset, None
        else:
            raise ValueError("%s not found. Please choose from %s" % (dataset_enum, DatasetFactory.datasets.keys()))
