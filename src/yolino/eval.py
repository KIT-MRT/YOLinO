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
from tqdm import tqdm

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.runner.evaluator import Evaluator
from yolino.utils.enums import TaskType
from yolino.utils.general_setup import general_setup

if __name__ == "__main__":
    args = general_setup("Evaluation", task_type=TaskType.TEST)

    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split=args.split, args=args, shuffle=False,
                                         augment=False)
    evaluator = Evaluator(args, load_best_model=True, anchors=dataset.anchors)
    TAG = args.split

    for i, data in enumerate(tqdm(loader)):
        # for i in range(len(dataset)):
        image, grid_tensor, fileinfo, duplicate_info, params = data  # unsqueeze(dataset.__getitem__(i))
        # TODO: this should be done inside, but the threads are annoying 
        dataset.params_per_file.update({f: params for f in fileinfo})

        # img = torch.ones((3, args.img_size[0] * 4, args.img_size[1] * 4), dtype=torch.float32)
        #
        # grid, errors = GridFactory.get(grid_tensor[[0]], [],
        #                                coordinate=CoordinateSystem.CELL_SPLIT,
        #                                args=args, input_coords=dataset.coords,
        #                                anchors=dataset.anchors)

        if i == 0:
            # plot the GT
            evaluator.on_data_loaded(fileinfo[0], image[0], grid_tensor[0], epoch=i)

        num_duplicates = int(sum(duplicate_info["total_duplicates_in_image"]).item())
        evaluator(image, grid_tensor, i * args.batch_size, fileinfo, apply_nms=False,
                  epoch=evaluator.forward.start_epoch, do_in_uv=args.full_eval, num_duplicates=num_duplicates)
    evaluator.publish_scores(epoch=evaluator.forward.start_epoch, tag=TAG)
