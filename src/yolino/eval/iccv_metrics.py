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

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt


def line_segment_to_sampled_points(line_segment, sample_distance):
    # get direction of the segment
    vector = line_segment[2:4] - line_segment[0:2]
    direction = np.arctan2(vector[1], vector[0])
    if direction < 0:
        direction = direction * -1 + math.pi
    a = math.degrees(direction)
    swap_axis = a % (90) == 0

    # get the x,y points given the fixed sampling interval
    length = np.linalg.norm(vector)
    num_sample_points = max(math.floor(length / sample_distance), 2)
    if swap_axis:
        xp = line_segment[[1, 3]]
        fp = line_segment[[0, 2]]
    else:
        xp = line_segment[[0, 2]]
        fp = line_segment[[1, 3]]
    if xp[0] > xp[1]:
        xp = xp[::-1]
        fp = fp[::-1]
    x = np.linspace(xp[0], xp[1], num_sample_points)
    y = np.interp(x, xp, fp, left=None, right=None, period=None)

    if swap_axis:
        points = np.stack((y, x, np.zeros(len(x)) + a)).transpose()
    else:
        points = np.stack((x, y, np.zeros(len(x)) + a)).transpose()

    return points


def polylines_to_points(polylines, sample_distance):
    total_points = np.array([[0, 0, 0]])
    for polyline in polylines:
        for line_segment in polyline:
            points = line_segment_to_sampled_points(np.array(line_segment), sample_distance)
            total_points = np.concatenate((total_points, points))
    return total_points[1:]


def get_captured_from_to(source_points, target_points, threshold, weights=[1, 1, 1]):
    true_positives = []  # 0: false_positive, 1: true_positive
    min_distances = np.empty((0, 4))
    true_positives_predictions = np.zeros((len(target_points)), dtype=np.bool)

    if len(source_points) == 0 or len(target_points) == 0:
        recall = 0
        precision = 0

        min_distances = np.ones((len(source_points), 4)) * np.inf
        true_positives = np.zeros((len(source_points)), dtype=np.bool)
        return recall, precision, true_positives, min_distances, true_positives_predictions

    for source_point in source_points:
        # xy distances
        xy_distances = np.abs(target_points[:, :2] - source_point[:2])
        # xy_distances[:, 0] *= weights[0]
        # xy_distances[:, 1] *= weights[1]

        # angular distances
        a = np.abs(target_points[:, 2] - source_point[2])
        b = np.abs(np.abs(target_points[:, 2] - source_point[2]) - 360)
        angular_distances = np.min(np.stack((a, b)), axis=0)

        # combined
        distances_raw = np.hstack((xy_distances, np.expand_dims(angular_distances, 1))
                                  )  # <class 'tuple'>: (N, 3) => x, y, a
        distances = distances_raw
        distances *= weights
        norm = np.linalg.norm(distances, axis=1)
        distances_raw = np.hstack((distances_raw, np.reshape(norm, (-1, 1)))
                                  )  # <class 'tuple'>: (N, 4) => x, y, a, dist

        true_positives.append(min(norm) <= threshold)
        idx = np.argmin(norm)
        true_positives_predictions[idx] = True
        min_distances = np.vstack((min_distances, distances_raw[idx]))

        # to check for class correctness
        # todo: find which target_points has the closest distance, and then check if class values are the same

    recall = sum(true_positives) / len(source_points)
    precision = sum(true_positives) / len(target_points)
    # min_distances <class 'list'>: (N, 4) x, y, a, dist
    return recall, precision, true_positives, min_distances, true_positives_predictions


def iccv_f1(preds_uv, gt_uv, img_size, sample_distance=1, threshold=0.5, conf_idx=-1):
    gt_points = polylines_to_points(gt_uv, sample_distance=sample_distance)

    if len(preds_uv[0]) > 0:
        # filter by confidence
        unbound_pred_lines = preds_uv[np.where(preds_uv[:, :, conf_idx] > threshold)[0:2]][:, :4]
    else:
        unbound_pred_lines = []
    # convert label and pred to points
    # args.sample_distance = 1
    pred_points = polylines_to_points([unbound_pred_lines], sample_distance)
    # calculate precision and recall
    width = img_size[0]
    threshold = width / 50  # TODO: threshold eval; abhaengig von Zeile?!
    height = img_size[1]
    aspect_ratio = width / height
    weights = [1.0, aspect_ratio, 1.0]
    # precision, TP_P, p_distances = get_captured_from_to(pred_points, gt_points, threshold, weights)
    recall, precision, TP, r_distances, TP_pred = get_captured_from_to(gt_points, pred_points, threshold, weights)

    # tp/fp/fn/tn image
    if False:
        # fig = plt.figure(figsize=(5*aspect_ratio,5))
        # ax = fig.add_subplot(111)

        fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(5 * aspect_ratio, 10))
        fig.suptitle("%s: Precision %.2f, Recall %.2f." % ("test.png", precision, recall))

        tp_pred_patch = mpatches.Patch(color=(0, 0.5, 0, 1), label='TP Prediction')
        tp_gt_patch = mpatches.Patch(color=(0, 1, 0, 1), label='TP Ground Truth')
        pred_patch = mpatches.Patch(color=(0, 0, 1, 1), label='FN Prediction')
        gt_patch = mpatches.Patch(color=(1, 0, 0, 1), label='FN Ground Truth')
        axs[0].legend(handles=[tp_pred_patch, tp_gt_patch])
        axs[1].legend(handles=[pred_patch, gt_patch])
        # ax.legend(handles=[tp_pred_patch, tp_gt_patch, pred_patch, gt_patch])

        for i, (m, out, tp) in enumerate([('o', pred_points, TP_pred), ('^', gt_points, TP)]):
            xs = out[:, 1]
            ys = height - out[:, 0]
            axs[2].scatter(xs, ys, marker=m, s=5, label=["pred", "gt"][i])

            ys = height - out[tp, 0]
            xs = out[tp, 1]
            color = [(0, 0.5 + 0.5 * i, 0, 1)]
            axs[0].scatter(xs, ys, marker=m, c=color, s=2)

            if i == 1:
                axs[1].scatter(xs, ys, marker=m, c=color, s=2)
            ys = height - out[~np.array(tp), 0]
            xs = out[~np.array(tp), 1]
            color = [(i, 0, 1 - i, 1)]
            axs[1].scatter(xs, ys, marker=m, c=color, s=2)

        axs[2].legend()

        path = "/tmp/4_result.png"
        Log.warning("plot to %s" % path)
        plt.savefig(path)
        plt.close()

        # boxplots
        fig, axs = plt.subplots(2, 2, sharey=True)
        fig.suptitle("Boxplot")
        axs[0, 0].set_title('Euclidean x distances')
        x_bp = axs[0, 0].boxplot(r_distances[:, 0])
        axs[0, 1].set_title('Euclidean y distances')
        y_bp = axs[0, 1].boxplot(r_distances[:, 1])
        axs[1, 0].set_title('Angle distances')
        a_bp = axs[1, 0].boxplot(r_distances[:, 2])
        axs[1, 1].set_title('Total distances')
        axs[1, 1].boxplot(r_distances[:, 3])

        path = "/tmp/boxplots.png"
        print("plot to %s" % path)
        plt.savefig(path)
        plt.close()

    return precision, recall


