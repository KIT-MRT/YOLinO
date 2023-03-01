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
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
import yolino.tools.vonMisesMixtures.mixture as vonmises
import yolino.tools.vonMisesMixtures.tools as vonmisestools
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from yolino.utils.enums import TaskType
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Log
from yolino.viz.plot import draw_angles


def plot_kmeans(args, angles, split, finish=True):
    plt.clf()
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, figsize=(25, 10))

    rmses = []
    for n in range(2, nrows * ncols + 2):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(angles.reshape(-1, 1))
        x_histo = vonmisestools.histogram(angles, bins=100)

        rmse = np.sqrt(np.power(kmeans.cluster_centers_[kmeans.labels_][:, 0] - angles, 2).mean())
        rmses.append(rmse)

        row = (n - 2) // ncols
        col = (n - 2) % ncols
        ax[row, col].set_title("K-Means with n=%d" % n)
        ax[row, col].plot(x_histo[0], x_histo[1], color=(0.5, 0.5, 0.5))
        for c in sorted(kmeans.cluster_centers_):
            ax[row, col].plot([c, c], [0, max(x_histo[1]) * 2.], label="%.1f" % c)
        ax[row, col].legend()

    optimum = np.argmin(rmses)

    title = "%s of %s\noptimum n=%d" % (split, args.dataset, optimum)
    plt.suptitle(title, fontsize=20)

    if finish:
        path = os.path.join(args.paths.specs_folder, "%s_%s_means.png" % (args.dataset, split))
        Log.warning("Write k-means plot to file://%s" % path)
        plt.savefig(path)
        plt.close()
    return optimum


def plot_vonmises(args, per_line_data, split):
    number_of_mixes = [2]  # , 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for n in number_of_mixes:
        Log.info("Plot von mises n=%d" % n)
        x = deepcopy(per_line_data["angles"].to_numpy())
        # Von Mises Fit
        # calculate the coefficients
        try:
            m = vonmises.mixture_pdfit(x, n=n, threshold=math.pow(10, -(n ** n)))
        except ValueError as e:
            Log.error("Von mises failed with n=%d\n%s" % (n, e))
            continue

        # plot the empirical distribution
        x_histo = vonmisestools.histogram(x, bins=100)

        plt.clf()
        plt.title("Von Mises Mix with n=%d" % n)
        p1 = plt.plot(x_histo[0], x_histo[1], label='raw')

        # plot the distribution using the parameters obtained from the EM algorithm
        f = np.zeros(len(x_histo[0]))
        for i in range(m.shape[1]):
            f += m[0, i] * vonmises.density(x_histo[0], m[1, i], m[2, i])
        p2 = plt.plot(x_histo[0], f / np.sum(f), label='fit')

        # display the two plots on the same figure
        plt.legend()
        path = os.path.join(args.paths.specs_folder, "%s_%s_mises_%d.png" % (args.dataset, split, n))
        Log.warning("Write von Mises plot to file://%s" % path)
        plt.savefig(path)
        # plt.show()
        plt.close()


def run():
    args = general_setup("Specs", show_params=True, task_type=TaskType.PARAM_OPTIMIZATION,
                         setup_logging=False
                         )
    pkl_file = args.paths.generate_specs_pkl_file_name(args.dataset, args.img_size, scale=args.scale)
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(pkl_file)
    per_line_data = pd.read_pickle(pkl_file)
    yaml_data = {}
    yaml_data["yolino"] = args.yolino

    # --------- k-means in total-------
    print("K-Means in total")
    if per_line_data["angles"].abs().min() < 1:
        raise ValueError("We expect the half circle of -1/2pi to 1/2pi over -pi as angles, but have abs "
                         "values smaller with \n%s" % np.asarray(per_line_data["angles"])[
                             np.where(np.abs(per_line_data["angles"]) < 1)])

    per_line_data["moved_angles"] = (per_line_data["angles"] + math.pi / 2.) % (math.pi * 2)
    n_angles = per_line_data["moved_angles"].to_numpy()
    optimum = plot_kmeans(args, n_angles, "train")

    kmeans = KMeans(n_clusters=optimum, random_state=0).fit(n_angles.reshape(-1, 1))
    per_line_data["cluster"] = kmeans.labels_
    per_line_data["cluster_angle"] = kmeans.cluster_centers_[kmeans.labels_]
    cluster_means = per_line_data[["x_s", "x_e", "y_s", "y_e", "cluster", "cluster_angle", "moved_angles"]].groupby(
        by="cluster").mean().sort_values(by="moved_angles")
    cluster_means["atan2"] = np.arctan2(cluster_means.y_e - cluster_means.y_s, cluster_means.x_e - cluster_means.x_s)

    if cluster_means["atan2"].abs().min() < math.pi / 2.:
        where = np.where(np.abs(cluster_means["atan2"]) < 1)
        raise ValueError("We expect the half circle of -1/2pi to 1/2pi over -pi as angles, but have abs "
                         "values smaller at %s \n%s" % (where, cluster_means.loc[where]))

    yaml_data["anchor_kmeans"] = [{kk: vv for kk, vv in v.items()} for k, v in
                                  dict(cluster_means.transpose()).items()]

    # --------- k-means in cells -------
    print("K-Means per cell")
    plt.clf()
    fig, axes = plt.subplots(nrows=args.grid_shape[0], ncols=args.grid_shape[1], figsize=(20, 10), sharex=True,
                             sharey=True)
    title = "Train of %s" % (args.dataset)
    plt.suptitle(title, fontsize=20)

    cells_folder = args.specs_cells_folder
    for file in tqdm(os.listdir(cells_folder)):
        if not file.startswith("%s_train_%d_%dx%d_" % (args.dataset, args.cell_size[0], args.img_size[0],
                                                       args.img_size[1])) \
                or not file.endswith(".pkl") \
                or not "per_cell_data" in file:
            continue
        c_idx = int(file.split("_")[4])
        r = c_idx // args.grid_shape[1]
        c = c_idx % args.grid_shape[1]

        file = os.path.join(cells_folder, file)
        per_cell_data = pd.read_pickle(file)
        if len(per_cell_data) <= args.num_predictors:
            text_kwargs = dict(ha='center', va='center', fontsize=28)
            axes[r, c].text(0.5, 0.5, '%d' % len(per_cell_data), **text_kwargs)
            continue

        if per_cell_data["angles"].abs().min() < 1:
            raise ValueError("We expect the half circle of -1/2pi to 1/2pi over -pi as angles, but have abs "
                             "values smaller with up to %s" % per_line_data["angles"].abs().min())

        per_cell_data = per_cell_data.drop(per_cell_data[per_cell_data.angles == 0].index)
        per_cell_data["moved_angles"] = (per_cell_data["angles"] + math.pi / 2.) % (math.pi * 2)
        n_angles = per_cell_data["moved_angles"].to_numpy()
        optimum = args.num_predictors

        kmeans = KMeans(n_clusters=optimum, random_state=0).fit(n_angles.reshape(-1, 1))
        per_cell_data["cluster"] = kmeans.labels_
        if len(kmeans.cluster_centers_) != args.num_predictors:
            raise ValueError("We have %d kmean proposals but only %d predictors"
                             % (len(kmeans.cluster_centers_), args.num_predictors))
        per_cell_data["cluster_angle"] = kmeans.cluster_centers_[kmeans.labels_]
        cluster_means = per_cell_data[["x_s", "x_e", "y_s", "y_e", "cluster", "cluster_angle", "moved_angles"]].groupby(
            by="cluster").mean().sort_values(by="moved_angles")
        cluster_means["atan2"] = np.arctan2(cluster_means.y_e - cluster_means.y_s,
                                            cluster_means.x_e - cluster_means.x_s)
        yaml_data["anchor_kmeans_in_cell_%d" % c_idx] = [{kk: vv for kk, vv in v.items()} for k, v in
                                                         dict(cluster_means.transpose()).items()]
        draw_angles(cluster_means["atan2"], ax=axes[r, c], finish=False)

        axes[r, c].set_yticks([])
        axes[r, c].set_xticks([])
        axes[r, c].axes.yaxis.set_ticklabels([])
        axes[r, c].axes.xaxis.set_ticklabels([])

    path = os.path.join(args.paths.specs_folder, "%s_train_cell_kmeans_anchors.png" % args.dataset)
    Log.warning("Write cell anchor image to file://%s" % path)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().invert_yaxis()
    plt.savefig(path, bbox_inches='tight', transparent="False", pad_inches=0)
    plt.close()

    path = os.path.join(args.paths.specs_folder, "%s_%s_anchors.yaml" % (args.dataset, "train"))
    Log.warning("Write yaml specs to file://%s" % path)
    with open(path, "w") as f:
        yaml.dump(yaml_data, f)


if __name__ == '__main__':
    run()
    exit()
