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


# ------------------------------------------------------------
# --------------------- FACTORY ------------------------------ 
# ------------------------------------------------------------ 

class DatasetFactory():
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

    # @classmethod
    # def get(self, dataset_str: str, only_available, split, args, shuffle, augment):
    #     return self.get(Dataset[dataset_str.lower()], only_available, split, args, shuffle, augment)

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

        # ------------------------------------------------------------


# ------------------------ MAIN ------------------------------
# ------------------------------------------------------------ 

if __name__ == '__main__':
    from yolino.utils.general_setup import general_setup
    import yaml


    def dumpdata(dataset, path, log_dir=None):
        if log_dir is None:
            log_dir = dataset

        params = {}
        params["rotation_range"] = 0.1
        params["img_height"] = 1640
        params["model"] = Network.YOLO_CLASS
        params["linerep"] = LINE.POINTS
        params["num_predictors"] = 8
        params["learning_rate"] = 0.001
        params["conf"] = 0.99
        params["lw"] = 0.016
        params["mpxw"] = 1.5
        params["debug"] = True
        params["cell_size"] = "[100,100]"
        params["max_n"] = 2

        tmp = params
        tmp["dataset"] = dataset
        tmp["log_dir"] = log_dir + "_po_8p_dn19"

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            yaml.dump(tmp, f)


    path = "tmp/params.yaml"
    dumpdata("culane", path)
    os.environ["DATASET_CULANE"] = "/mrtstorage/datasets/public/CULane"
    args = general_setup("Dataset Main", path)

    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split="runner", args=args, shuffle=True,
                                         augment=True)

    assert (len(dataset) >= 2)
