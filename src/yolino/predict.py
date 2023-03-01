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
import os.path

from tqdm import tqdm
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.runner.evaluator import Evaluator
from yolino.utils.enums import TaskType, ImageIdx
from yolino.utils.general_setup import general_setup

if __name__ == "__main__":
    args = general_setup("Prediction", task_type=TaskType.TEST)
    args.plot = True

    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split=args.split, args=args, shuffle=False,
                                         augment=False, load_full=True)
    evaluator = Evaluator(coords=dataset.coords, args=args)
    TAG = args.split

    for i, data in enumerate(tqdm(loader)):
        images, grid_tensors, fileinfos, _, params = data
        dataset.params_per_file.update({fileinfos[i]: params[i] for i in range(len(fileinfos))})

        # predict without nms
        preds = evaluator(images, labels=grid_tensors, idx=i, filenames=fileinfos, tag="predict",
                          apply_nms=True, fit_line=True)

    path = args.paths.generate_debug_image_file_path(fileinfos[0], ImageIdx.PRED)
    path_nms = args.paths.generate_debug_image_file_path(fileinfos[0], ImageIdx.NMS)
    print("You'll find the resulting images in file://%s "
          "(file pattern %s or %s)" % (os.path.dirname(path_nms), os.path.basename(path),
                                       os.path.basename(path_nms)))
