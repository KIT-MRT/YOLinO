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
import timeit

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

from yolino.eval.distances import to_nms_space
from yolino.grid.coordinates import validate_input_structure
from yolino.utils.enums import CoordinateSystem, ImageIdx
from yolino.utils.logger import Log
from yolino.viz.plot import plot


def nms(preds_uv, grid_shape, cell_size, confidence_threshold, orientation_weight, length_weight, midpoint_weight,
        epsilon, min_samples, plot_debug=False, normalize=False, weight_samples=False):
    validate_input_structure(preds_uv, CoordinateSystem.UV_SPLIT)
    lines_uv = preds_uv[0]  # only single batch
    reduced = np.empty((1, preds_uv.shape[2]), dtype=preds_uv.dtype)

    tic = timeit.default_timer()
    scale = 20 / grid_shape[1]  # 1.0 for normal, 0.5 for up, 0.25 for upup
    epsilon *= scale  # 20 for grid shape 4*12
    Log.debug("Epsilon: %s; Dimensions: %s" % (epsilon, np.max(lines_uv[:, 0:4], axis=0)))
    # dt = distance_threshold

    pauls_lines = to_nms_space(lines_uv)

    # Interesting lines have sufficient confidence and non-zero length
    nonzero_conf_idx = lines_uv[:, -1] > float(confidence_threshold)
    nonzero_len_idx = np.logical_not(np.isclose(pauls_lines[:, 2], 0))

    interesting = np.logical_and(nonzero_conf_idx, nonzero_len_idx)
    if np.sum(interesting) == 0:
        return preds_uv, reduced

    Log.debug("Interesting: %s %s" % (interesting.shape, np.count_nonzero(interesting)))

    # Weight dimensions
    weights = np.asarray([midpoint_weight, midpoint_weight, length_weight, orientation_weight, orientation_weight])
    pauls_lines[interesting] *= weights

    ticDBSCAN = timeit.default_timer()
    db = DBSCAN(float(epsilon), min_samples=int(min_samples))
    if weight_samples:
        db = db.fit(pauls_lines[interesting],
                    sample_weight=np.power(lines_uv[interesting, -1], 10))  # Power can also be up to 10 w/o change
    else:
        db = db.fit(pauls_lines[interesting])  # Power can also be up to 10 w/o change

    tocDBSCAN = timeit.default_timer()
    Log.debug("Sklearn:  %s" % (tocDBSCAN - ticDBSCAN))

    labels = np.full((lines_uv.shape[0]), -1)
    labels[interesting] = db.labels_
    lines_uv[np.logical_not(interesting), -1] = 0.

    cmap = plt.get_cmap("jet")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(np.unique(labels)))
    scalarMap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    if plot_debug:
        x1, y1, x2, y2 = np.max(lines_uv[:, 0:4], axis=0)
        img = torch.ones((3, int(max(x1, x2)), int(max(y1, y2))), dtype=torch.float32)
        img2 = torch.ones((3, int(max(x1, x2)), int(max(y1, y2))), dtype=torch.float32)

    # Wighted mean within clusters
    if len(np.unique(labels)) == 1:
        Log.warning("No clusters found")

    for label in np.unique(labels):
        if label < 0:
            continue
        with_this_label = np.argwhere(labels == label)[:, 0]
        lines_with_this_label = lines_uv[with_this_label, :]
        m = np.average(lines_with_this_label, axis=0, weights=np.power(lines_with_this_label[:, -1], 10))
        m[-1] = np.max(lines_with_this_label[:, -1])
        np.append(reduced, m)

        if plot_debug:
            img, ok = plot(lines=np.expand_dims(lines_uv[with_this_label], axis=0), name="/tmp/debug_nms.png",
                           image=img, threshold=confidence_threshold, colorstyle=None,
                           coordinates=CoordinateSystem.UV_SPLIT,
                           tag="nms", imageidx=ImageIdx.NMS, color=np.asarray(scalarMap.to_rgba(label))[0:3] * 255.)

        lines_uv[with_this_label, -1] = 0.  # remove all cluster elements
        lines_uv[with_this_label[0], :] = m  # rewrite first cluster element with average

        if plot_debug:
            img2, ok = plot(lines=np.expand_dims(lines_uv[with_this_label], axis=0), name="/tmp/debug_nms2.png",
                            image=img2, threshold=confidence_threshold, colorstyle=None,
                            coordinates=CoordinateSystem.UV_SPLIT,
                            tag="nms", imageidx=ImageIdx.NMS, color=np.asarray(scalarMap.to_rgba(label))[0:3] * 255.)

    Log.debug("Sklearn with postproc: %s" % (tocDBSCAN - ticDBSCAN))

    toc = timeit.default_timer()
    Log.debug("Total NMS: %s" % (toc - tic))
    return np.expand_dims(lines_uv, axis=0), reduced
