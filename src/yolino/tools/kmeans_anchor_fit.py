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
import sys

import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from yolino.utils.enums import TaskType, AnchorDistribution, AnchorVariables
from yolino.utils.logger import Log
from yolino.viz.colors import COLOR


def run():
    from yolino.utils.general_setup import general_setup

    args = general_setup("Kmeans Anchor", show_params=False, task_type=TaskType.PARAM_OPTIMIZATION,
                         project_suffix="_kmeans", setup_logging=True)
    args.offset = False
    args.anchors = AnchorDistribution.NONE

    yaml_data = get_kmeans_cluster(args, force=True)

    anchors = np.asarray([[a["x_s"], a["y_s"], a["x_e"], a["y_e"], a["mx"], a["my"], a["dx"], a["dy"]]
                          for a in yaml_data["anchor_kmeans"]])

    plot_anchor(args, anchors)
    Log.upload_params(args)


def get_kmeans_cluster(args, force=False):
    from yolino.model.anchors import Anchor
    # /home/meyer/.local/lib/python3.8/site-packages
    sys.path.append("/home/meyer/.local/lib/python3.8/site-packages")
    import pandas as pd

    kmeans_yaml_file = args.paths.generate_specs_file_name(dataset=args.dataset, split="train",
                                                           anchor_vars=args.anchor_vars, scale=args.scale,
                                                           num_predictors=args.num_predictors)


    Log.warning(f"We try to load anchors from {kmeans_yaml_file}.")
    if not force and os.path.exists(kmeans_yaml_file):
        with open(kmeans_yaml_file, "r") as f:
            data = yaml.safe_load(f)
        if "anchor_kmeans" in data:
            Log.warning("Load data from %s" % kmeans_yaml_file)
            return data

    pkl_file = args.paths.generate_specs_pkl_file_name(dataset=args.dataset, split="train", scale=args.scale,
                                                       img_size=args.img_size)

    if os.path.exists(pkl_file):
        per_line_data = pd.read_pickle(pkl_file).astype(float)
        Log.warning(
            f"We calculate the anchors as there is no yaml file... Read {len(per_line_data)} dataset lines from %s" % pkl_file)
    else:
        Log.error(
            f"Please run line extraction first with kmeans_preparation.py (same usage as this). We expect {pkl_file}")
        exit(1)
    args.anchors = AnchorDistribution.KMEANS

    yaml_data = {}
    yaml_data["yolino"] = args.yolino
    kmeans_columns, sort_by = Anchor.get_columns(args.anchor_vars)
    np_data = per_line_data[kmeans_columns].to_numpy()
    if len(np_data.shape) == 1:
        np_data = np_data.reshape(-1, 1)
    Log.info("Get %d-means on %s" % (args.num_predictors, kmeans_columns))
    kmeans = KMeans(n_clusters=args.num_predictors, random_state=0).fit(np_data)
    per_line_data["cluster"] = kmeans.labels_
    cluster_means = per_line_data[
        np.unique(["x_s", "x_e", "y_s", "y_e", "cluster", "mx", "my", "dx", "dy", *kmeans_columns])].groupby(
        by="cluster").mean().sort_values(by=sort_by)
    yaml_data["anchor_kmeans"] = [{kk: vv for kk, vv in v.items()} for k, v in
                                  dict(cluster_means.transpose()).items()]
    # ------ Write yaml -----
    Log.print("\n------------------ WRITE YAML -------------------")
    Log.info("Write yaml specs to file://%s" % kmeans_yaml_file)

    yaml_data["anchor_kmeans"] = sorted(yaml_data["anchor_kmeans"], key=lambda x: x["x_s"] + x["y_s"])
    with open(kmeans_yaml_file, "w") as f:
        yaml.dump(yaml_data, f)
    return yaml_data


def plot_anchor(args, anchors):
    Log.print("\n------------------ PLOT -------------------")
    Log.info("Plot %d-means" % (args.num_predictors))

    plt.figure(figsize=(5, 5))
    scale = 1

    colors = COLOR(root=args.root)
    for i, n in enumerate(anchors):
        color = colors[i].plt()
        if AnchorVariables.MIDPOINT in args.anchor_vars:
            plt.gca().add_patch(plt.Circle((n[5] * scale, n[4] * scale), radius=0.05, color=color))
        if AnchorVariables.DIRECTION in args.anchor_vars:
            plt.arrow(n[1] * scale, n[0] * scale, n[7] * scale, n[6] * scale, color=color,
                      label=("[%.1f,%.1f]-" % (n[4], n[5]) if AnchorVariables.MIDPOINT in args.anchor_vars else "")
                            + "[%.1f,%.1f]" % (n[6], n[7]), head_width=scale / 20)

        if AnchorVariables.POINTS in args.anchor_vars:
            plt.arrow(n[1] * scale, n[0] * scale,
                      (n[3] - n[1]) * scale, (n[2] - n[0]) * scale,
                      color=color, label="[%.1f,%.1f]-[%.1f,%.1f]" % (n[0], n[1], n[2], n[3]), head_width=scale / 20)
    name = args.paths.generate_anchors_image_file_path(dataset=args.dataset, anchor_vars=args.anchor_vars,
                                                       num_predictors=args.num_predictors, anchors=args.anchors,
                                                       scale=args.scale)

    plt.xlim((-0.1 * scale, scale * 1.1))
    plt.ylim((-0.1 * scale, scale * 1.1))
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.tight_layout()
    plt.gca().set_aspect('equal', 'box')

    Log.info("Export to file://%s" % (os.path.abspath(name)))
    plt.savefig(name)
    Log.plt(epoch=0, fig=plt, tag=name)

    # Log.info("Export to file://%s_.tex" % (os.path.abspath(name)))
    # tikzplotlib.save(name + ".tex")


if __name__ == '__main__':
    run()
    exit()
