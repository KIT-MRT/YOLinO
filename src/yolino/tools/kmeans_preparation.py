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
from copy import copy

import torch
from tqdm import tqdm
from yolino.model.line_representation import MidDirLines
from yolino.utils.enums import AnchorDistribution, LINE, Variables, TaskType
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Log


def run():
    args = general_setup("Kmeans Preparation", show_params=False, task_type=TaskType.PARAM_OPTIMIZATION,
                         project_suffix="_kmeans_prep", setup_logging=True)

    args.offset = False
    args.anchors = AnchorDistribution.NONE
    args.ignore_missing = False

    if args.max_n <= 0:
        Log.error("We do not recommend to use all lines for this. Rather use a subset with --max_n <= 100.")

    Log.upload_params(args)

    pkl_file = args.paths.generate_specs_pkl_file_name(dataset=args.dataset, split="train",
                                                       img_size=args.img_size, scale=args.scale)

    if os.path.exists(pkl_file):
        Log.error("Data is already generated in %s" % pkl_file)
        Log.tag("duplicate")
        exit(0)

    Log.info("Start collecting lines from the dataset into %s" % pkl_file)
    if not os.path.exists(args.paths.specs_folder):
        os.makedirs(args.paths.specs_folder)
    if not os.path.exists(args.paths.specs_cells_folder):
        os.makedirs(args.paths.specs_cells_folder)

    args.anchors = AnchorDistribution.NONE
    columns = ["f1", "precision", "recall", "accuracy", "x_s", "y_s", "x_e", "y_e",
               "mx", "my", "dx", "dy"]

    per_line_data, _ = collect_dataframes(columns, args)
    Log.info(f"Write {len(per_line_data)} lines to {pkl_file}.")
    per_line_data.to_pickle(pkl_file)


def collect_dataframes(columns, args):
    import pandas as pd
    from yolino.dataset.dataset_factory import DatasetFactory

    points_args = copy(args)
    points_args.linerep = LINE.POINTS
    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split="train", args=points_args,
                                         shuffle=True, augment=False, load_only_labels=True)
    per_line_data = pd.DataFrame(columns=columns)

    total_data = pd.Series(0, index=["images", "scores", "lines"])
    total_data["images"] = 0
    total_data["lines"] = 0
    image_idx = 0
    line_idx = 0
    Log.info("Collect data...")
    for i, dataset_entry in enumerate(tqdm(loader, desc="Load dataset")):
        images, grid_tensor, fileinfo, _, params = dataset_entry

        total_data["images"] += len(images)

        for b_idx, batch in enumerate(grid_tensor):
            for c_idx, cell in enumerate(batch):
                for a_idx, predictor in enumerate(cell):
                    if torch.any(torch.isnan(predictor)):
                        continue

                    p = predictor[dataset.coords.get_position_of(Variables.GEOMETRY)]
                    p_np = p.numpy()

                    total_data["lines"] += 1
                    dict_data = {"x_s": p_np[0], "y_s": p_np[1], "x_e": p_np[2], "y_e": p_np[3]}
                    dict_data["mx"], dict_data["my"], dict_data["dx"], dict_data["dy"] = MidDirLines.from_cart(
                        start=p_np[0:2], end=p_np[2:4])
                    per_line_data = pd.concat([per_line_data, pd.DataFrame(dict_data, columns=columns, index=[0])],
                                              ignore_index=True, axis=0)
                    line_idx += 1
            image_idx += 1
    Log.info(total_data)
    Log.scalars(tag="specs", epoch=0, dict=total_data)
    Log.debug("...Done")

    return per_line_data, total_data


if __name__ == '__main__':
    run()
