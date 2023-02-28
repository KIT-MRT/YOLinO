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
import numpy as np
import torch
import yaml
from tqdm import tqdm

from yolino.model.line_representation import MidDirLines, LineRepresentation
from yolino.utils.enums import TaskType, LINE
from yolino.utils.logger import Log
from yolino.utils.test_utils import unsqueeze


def run():
    from yolino.utils.general_setup import general_setup

    args = general_setup("Kmeans Anchor", show_params=False, task_type=TaskType.PARAM_OPTIMIZATION,
                         project_suffix="_kmeans", setup_logging=True)
    args.offset = False
    Log.upload_params(args)
    anchors = evaluate(args, only_viz=False)

    # -------- Plot anchors -----
    from yolino.tools.kmeans_anchor_fit import plot_anchor
    if args.linerep != LINE.POINTS:
        cart_anchors = np.stack([[*LineRepresentation.get(args.linerep).to_cart(anchor), anchor[0], anchor[1], anchor[2], anchor[3]] for anchor in anchors])
    else:
        cart_anchors = anchors

    plot_anchor(args, cart_anchors)
    Log.push(None)


def evaluate(args, only_viz=False):
    from yolino.dataset.dataset_factory import DatasetFactory
    # ------ Evaluate -----
    if not only_viz:
        Log.warning("\n------------------ EVALUATE -------------------")
    # augment: False, True
    # split: train, val
    yaml_data = []
    for augment in [False]:
        for split in ["train", "val"]:
            Log.info("Evaluate %s on %s" % ("With" if augment else "Without", split))
            single_yaml_data = {"augment": augment, "split": split}
            single_yaml_data["files"] = {}

            dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split=split, args=args,
                                                 shuffle=False, augment=augment, load_only_labels=True)
            if only_viz:
                break

            total_num_duplicates = 0
            total_num_lines = 0

            pbar = tqdm(loader, desc="Iterate %s Dataset" % split)
            for i in pbar:  # range(len(dataset))):  # loader):
                _, grid_tensor, filenames, duplicates, params = i

                for entry in range(len(filenames)):
                    Log.info("%s has %s" % (filenames[entry], duplicates["total_duplicates_in_image"][entry]))
                    single_yaml_data["files"][filenames[entry]] = {k: duplicates[k][entry].numpy().tolist() for k in
                                                                   duplicates.keys() if k != "cells"}
                    if len(duplicates["cells"]) > 0 and len(duplicates["cells"]) > entry:
                        single_yaml_data["files"][filenames[entry]]["bad_boys"] = np.stack(
                            np.where(duplicates["cells"][entry] > 0), axis=1).tolist()
                    else:
                        single_yaml_data["files"][filenames[entry]]["bad_boys"] = list()

                    num_correct_lines = torch.sum(~torch.isnan(grid_tensor[entry, :, :, 0]))
                    Log.info("%s has %d correct lines" % (filenames[entry], num_correct_lines.item()))
                    single_yaml_data["files"][filenames[entry]]["correct"] = num_correct_lines.item()

                    total_num_duplicates += duplicates["total_duplicates_in_image"][entry].item()
                    total_num_lines += num_correct_lines.item()

            Log.info("%s augment on %s we've got" % ("With" if augment else "Without", split))
            single_yaml_data["total_duplicates"] = total_num_duplicates
            single_yaml_data["total_correct"] = total_num_lines
            yaml_data.append(single_yaml_data)
            Log.scalars(tag=split + ("_augment" if augment else ""), epoch=0,
                        dict={"duplicates": single_yaml_data["total_duplicates"],
                              "correct": single_yaml_data["total_correct"],
                              "percentage": single_yaml_data["total_duplicates"]
                                            / (single_yaml_data["total_duplicates"]
                                               + single_yaml_data["total_correct"])})

    # ------ Write yaml -----
    if not only_viz:
        Log.warning("\n------------------ WRITE EVAL YAML -------------------")
        path = args.paths.generate_specs_eval_file_name(dataset=args.dataset, split="train",
                                                        anchors=args.anchors, anchor_vars=args.anchor_vars,
                                                        num_predictors=args.num_predictors,
                                                        cell_size=args.cell_size, img_size=args.img_size)
        Log.warning("Write yaml specs to file://%s" % path)
        # for entry in yaml_data:
        with open(path, "w") as f:
            yaml.dump(yaml_data, f)

    Log.push(next_epoch=0)
    return dataset.anchors.bins


if __name__ == '__main__':
    run()

