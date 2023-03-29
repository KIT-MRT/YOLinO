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
import timeit
from copy import copy

import numpy as np
import torch
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import Distance, Variables, LINE
from yolino.utils.logger import Log


def get_hungarian_match(cost_matrix):
    start = timeit.default_timer()
    if len(cost_matrix) == 1:
        return torch.tensor(0, device=cost_matrix.device), torch.argmin(cost_matrix, dim=1)
    elif len(cost_matrix[0]) == 1:
        return torch.argmin(cost_matrix, dim=0), torch.tensor(0, device=cost_matrix.device)

    from scipy.optimize import linear_sum_assignment
    # is ok as this is not part of the actual loss that would need to be differentiable
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    except ValueError as e:
        m = cost_matrix.detach().cpu().numpy()
        Log.error("Cost matrix %s" % str(m.shape))
        for row in m:
            Log.error(row)
        raise e
    return torch.tensor(row_ind, device=cost_matrix.device), torch.tensor(col_ind, device=cost_matrix.device)


def get_hungarian_match_gpu(cost_matrix, eps=1):
    start = timeit.default_timer()
    if len(cost_matrix) == 1:
        return torch.tensor(0, device=cost_matrix.device), torch.argmin(cost_matrix, dim=1)
    elif len(cost_matrix[0]) == 1:
        return torch.argmin(cost_matrix, dim=0), torch.tensor(0, device=cost_matrix.device)

    Log.warning("Apply auction lap")
    from yolino.tools.auction_lap import auction_lap
    _, curr_ass, _ = auction_lap(cost_matrix * -1, eps=eps)
    Log.warning("Done")
    Log.time(key="hungarian_gpu", value=timeit.default_timer() - start)
    return torch.tensor(range(len(cost_matrix)), device=cost_matrix.device), curr_ass


def linesegment_hausdorff_distance(gt, pred, gt_class_indices, pred_class_indices):
    len_gt = torch.linalg.norm(gt)
    len_pred = torch.linalg.norm(pred)

    ang_dist = get_angular_hausdorff_distance(gt, len_gt, pred, len_pred)
    parallel_dist = get_parallel_distance(gt, len_gt, pred, len_pred)
    perp_dist = get_perpendicular_sum_distance(gt, len_gt, pred, len_pred)

    return torch.sqrt(torch.pow(ang_dist, 2) + torch.pow(parallel_dist, 2) + torch.pow(perp_dist, 2))


def cosine_similarity(gt, len_gt, pred, len_pred):
    dot = torch.dot(gt[2:4] - gt[0:2], pred[2:4] - pred[0:2])
    cos = dot / (len_gt * len_pred)
    return max(-1, min(1, cos))  # sometimes this is marginally above 1


def linesegment_cosine_distance(gt, preds, coords: VariableStructure, use_conf=False):
    Log.warning("Cosine distance does not use class or conf")
    distances = []
    for idx in range(len(preds)):
        diff = gt[2:4] - gt[0:2]
        pred_diff = preds[idx, 2:4] - preds[idx, 0:2]

        len_gt = torch.linalg.norm(diff)
        len_pred = torch.linalg.norm(pred_diff)

        distances.append(cosine_distance(gt=gt, pred=preds[idx], len_gt=len_gt, len_pred=len_pred))

    return torch.tensor(distances)


def cosine_distance(gt, pred, len_gt, len_pred):
    return 1 - cosine_similarity(gt, len_gt, pred, len_pred)


def get_perpendicular_pred_rmse(gt, len_gt, pred, len_pred):
    dists = get_perpendicular_distances(gt, len_gt, pred, len_pred)
    return math.sqrt(torch.pow(torch.tensor([dists[0], dists[2]]), 2).mean())  # only pred points


def get_perpendicular_distances(gt, len_gt, pred, len_pred):
    """

    Returns:
        list: of 4 perpendicular distances
        0: start pred point on gt line
        1: start gt point on pred line
        2: end pred point on gt line
        3: end gt point on pred line
    """
    if len_gt is None:
        len_gt = torch.linalg.norm(gt[2:4] - gt[0:2])

    if len_pred is None:
        len_pred = torch.linalg.norm(pred[2:4] - pred[0:2])

    perp_distances = []
    for idx in [[0, 1], [2, 3]]:
        perp_distances.append(point_to_line_distance(len_gt, gt, pred[idx]))
        perp_distances.append(point_to_line_distance(len_pred, pred, gt[idx]))

    return perp_distances


def point_to_line_distance(line_length, line, point):
    if line_length == 0:
        return np.linalg.norm(line[0:2] - point)
    else:
        return abs(
            (line[2] - line[0]) * (line[1] - point[1]) - (line[0] - point[0]) * (line[3] - line[1])) / line_length


# min of start and end distance to the GT line and vice versa => 4 the actual dist
def get_perpendicular_min_distance(gt, len_gt, pred, len_pred):
    perp_distances = get_perpendicular_distances(gt, len_gt, pred, len_pred)
    perp_dist = min(perp_distances)
    return perp_dist


# sum of start and end distance to the GT line and vice versa => 4 the actual dist
def get_perpendicular_sum_distance(gt, len_gt, pred, len_pred):
    perp_distances = get_perpendicular_distances(gt, len_gt, pred, len_pred)
    perp_dist = torch.sum(torch.tensor(perp_distances))
    return perp_dist


# remainder of projection
def get_parallel_distance(gt, len_gt, pred, len_pred):
    dot = torch.dot(gt[2:4] - gt[0:2], pred[2:4] - pred[0:2])
    len_projection_on_gt = dot / len_gt
    len_projection_on_pred = dot / len_pred
    parallel_dist = max(len_gt - len_projection_on_gt, len_pred - len_projection_on_pred)
    return parallel_dist


def get_parallel_mover_distance(gt, len_gt, pred, len_pred):
    # direction
    dx = gt[2] - gt[0]
    dy = gt[3] - gt[1]
    # normals
    n1 = torch.tensor([-dy, dx])

    line_length = torch.linalg.norm(n1)
    perps_p_s = point_to_line_distance(line_length, line=[*gt[0:2], *(gt[0:2] + n1)], point=pred[[0, 1]])
    perps_p_e = point_to_line_distance(line_length, line=[*gt[2:4], *(gt[2:4] + n1)], point=pred[[2, 3]])

    # direction
    dx = pred[2] - pred[0]
    dy = pred[3] - pred[1]
    # normals
    n1 = torch.tensor([-dy, dx])

    line_length = torch.linalg.norm(n1)
    perps_g_s = point_to_line_distance(line_length, line=[*pred[0:2], *(pred[0:2] + n1)], point=gt[[0, 1]])
    perps_g_e = point_to_line_distance(line_length, line=[*pred[2:4], *(pred[2:4] + n1)], point=gt[[2, 3]])

    set = [perps_p_s, perps_p_e, perps_g_s, perps_g_e]
    return min(set)


def get_angular_hausdorff_distance(gt, len_gt, pred, len_pred):
    sin_value = get_angular_distance(gt, len_gt, pred, len_pred)
    ang_dist = min(len_gt, len_pred) * sin_value
    return ang_dist


def get_angular_distance(gt, len_gt, pred, len_pred):
    cos = cosine_similarity(gt, len_gt, pred, len_pred)
    acos = math.acos(cos)
    return math.degrees(acos)


def linesegment_euclidean_distance(gt, pred, coords: VariableStructure, use_conf=False):
    vars = copy(coords.vars_to_train)
    if not use_conf and Variables.CONF in vars:
        vars.remove(Variables.CONF)

    gt_idx = coords.get_position_of(vars)
    pred_idx = coords.get_position_within_prediction(vars)

    points_distances = torch.linalg.norm(gt[gt_idx] - pred[:, pred_idx], dim=-1)
    points_distances = torch.flatten(points_distances)
    return points_distances


def variables_euclidean_distance(coords, gt, pred, start_dist, use_conf=False):
    from yolino.utils.enums import Variables

    angle_dist = torch.zeros_like(start_dist)
    if Variables.SAMPLE_ANGLE in coords.train_vars():
        gt_idx = coords.get_position_of(Variables.SAMPLE_ANGLE)
        p_idx = coords.get_position_within_prediction(Variables.SAMPLE_ANGLE)
        angle_dist[:] = torch.linalg.norm(gt[gt_idx] - pred[:, p_idx], dim=-1)

    class_dist = torch.zeros_like(start_dist)
    if Variables.CLASS in coords.train_vars():
        gt_idx = coords.get_position_of(Variables.CLASS)
        p_idx = coords.get_position_within_prediction(Variables.CLASS)
        class_dist[:] = torch.linalg.norm(gt[gt_idx] - pred[:, p_idx], dim=-1)

    conf_dist = torch.zeros_like(start_dist)
    if use_conf and Variables.CONF in coords.train_vars():
        gt_idx = coords.get_position_of(Variables.CONF)
        p_idx = coords.get_position_within_prediction(Variables.CONF)
        conf_dist[:] = torch.linalg.norm(gt[gt_idx] - pred[:, p_idx], dim=-1)
    return angle_dist, class_dist, conf_dist


def point_squared_distance(gt, pred, coords: VariableStructure, use_conf=False):
    # TODO normalize to have max of 1 also in full image distances?

    start_dist = torch.pow(gt[0:2] - pred[:, 0:2], 2)

    angle_dist, class_dist, conf_dist = variables_squared_distance(coords, gt, pred, start_dist, use_conf)

    points_distances = torch.sum(torch.stack([start_dist, angle_dist, class_dist, conf_dist]), dim=0)
    points_distances = torch.flatten(points_distances)
    return points_distances


def variables_squared_distance(coords, gt, pred, start_dist, use_conf=False):
    from yolino.utils.enums import Variables
    angle_dist = torch.zeros_like(start_dist)
    if Variables.SAMPLE_ANGLE in coords.train_vars():
        gt_idx = coords.get_position_of(Variables.SAMPLE_ANGLE)
        p_idx = coords.get_position_within_prediction(Variables.SAMPLE_ANGLE)
        angle_dist[:] = torch.pow(gt[gt_idx] - pred[:, p_idx], 2)

    class_dist = torch.zeros_like(start_dist)
    if Variables.CLASS in coords.train_vars():
        gt_idx = coords.get_position_of(Variables.CLASS)
        p_idx = coords.get_position_within_prediction(Variables.CLASS)
        class_dist[:] = torch.sum(torch.pow(gt[gt_idx] - pred[:, p_idx], 2), dim=1)

    conf_dist = torch.zeros_like(start_dist)
    if use_conf and Variables.CONF in coords.train_vars():
        gt_idx = coords.get_position_of(Variables.CONF)
        p_idx = coords.get_position_within_prediction(Variables.CONF)
        conf_dist[:] = torch.pow(gt[gt_idx] - pred[:, p_idx], 2).flatten()

    return angle_dist, class_dist, conf_dist


def linesegment_squared_distance(gt, pred, coords: VariableStructure, use_conf=False):
    vars = copy(coords.vars_to_train)
    if not use_conf and Variables.CONF in vars:
        vars.remove(Variables.CONF)

    gt_idx = coords.get_position_of(vars)
    pred_idx = coords.get_position_within_prediction(vars)

    points_distances = torch.sum(torch.pow(gt[gt_idx] - pred[:, pred_idx], 2), dim=1)
    points_distances = torch.flatten(points_distances)
    return points_distances


def linesegment_pauls_distance(gt, pred):
    # Convert to dx, dy,
    diff = gt[2:4] - gt[0:2]
    pred_diff = pred[2:4] - pred[0:2]

    lengths = torch.linalg.norm(diff)
    pred_lengths = torch.linalg.norm(pred_diff)

    midpoint_x, midpoint_y = midpoints_distances(gt, pred)

    norm_diff = np.divide(diff, lengths, where=lengths != 0)
    norm_pred_diff = np.divide(pred_diff, pred_lengths, where=pred_lengths != 0)

    return torch.sqrt(torch.pow(midpoint_x, 2) + torch.pow(midpoint_y, 2) +
                      torch.pow(lengths - pred_lengths, 2) +
                      torch.pow(norm_diff[0] - norm_pred_diff[0], 2) + torch.pow(norm_diff[1] - norm_pred_diff[1], 2))


# def aml_to_cart(line):
#     from yolino.dataset.dataset_base import calc_geom_from_angle
#     return calc_geom_from_angle(x=line[1], y=line[2], angle=line[0], max_length=[line[3], line[3]])


def aml_to_cart(aml):
    from yolino.dataset.dataset_base import calc_geom_from_angle
    upper_position = calc_geom_from_angle(x=aml[1], y=aml[2], angle=aml[0], max_length=[aml[3], aml[3]])
    lower_position = calc_geom_from_angle(x=aml[1], y=aml[2], angle=aml[0], max_length=[aml[3], aml[3]], invert=True)
    position = torch.cat([lower_position[2:4], upper_position[2:4]])
    return position


def to_aml(line):
    new_lines = torch.ones((4), dtype=line.dtype) * -1
    # Convert to dx, dy,
    diff = line[2:4] - line[0:2]

    # angle
    new_lines[0] = np.arctan2(diff[1], diff[0])

    # midpoints
    new_lines[1:3] = get_midpoints(line)

    # lengths
    new_lines[3] = torch.linalg.norm(diff)

    return new_lines


def to_pauls_space_raw(lines):
    new_lines = np.ones((len(lines), 5), dtype=lines.dtype) * -1

    # Convert to dx, dy,
    xdiffs = lines[:, 2] - lines[:, 0]
    ydiffs = lines[:, 3] - lines[:, 1]

    # midpoints
    new_lines[:, 0:2] = get_midpoints_from_np_set(lines)

    # normalized lengths
    lengths = np.linalg.norm([xdiffs, ydiffs], axis=0)  # / math.sqrt(2)
    new_lines[:, 2] = lengths

    # Normalize xdiffs by length
    new_lines[:, 3] = np.divide(xdiffs, lengths, where=new_lines[:, 2] != 0)
    new_lines[:, 4] = np.divide(ydiffs, lengths, where=new_lines[:, 2] != 0)

    return new_lines


def to_md_cell_space(lines):
    new_lines = np.ones((len(lines), 4), dtype=lines.dtype) * -1

    # Convert to dx, dy,
    new_lines[:, 2:4] = lines[:, 2:4] - lines[:, 0:2]

    # midpoints
    new_lines[:, 0:2] = get_midpoints_from_np_set(lines)

    return new_lines


def to_mld_cell_space(lines):
    new_lines = np.ones((len(lines), 5), dtype=lines.dtype) * -1

    # Convert to dx, dy,
    xdiffs = lines[:, 2] - lines[:, 0]
    ydiffs = lines[:, 3] - lines[:, 1]

    # midpoints
    new_lines[:, 0:2] = get_midpoints_from_np_set(lines)

    # normalized lengths
    lengths = np.linalg.norm([xdiffs, ydiffs], axis=0)
    new_lines[:, 2] = lengths

    # Normalize xdiffs by length
    new_lines[:, 3] = np.divide(xdiffs, lengths, where=new_lines[:, 2] != 0)
    new_lines[:, 4] = np.divide(ydiffs, lengths, where=new_lines[:, 2] != 0)

    return new_lines


def to_nms_space(lines):
    new_lines = np.ones((len(lines), 5), dtype=lines.dtype) * -1

    # Convert to dx, dy,
    xdiffs = lines[:, 2] - lines[:, 0]
    ydiffs = lines[:, 3] - lines[:, 1]

    # midpoints
    new_lines[:, 0:2] = get_midpoints_from_np_set(lines)

    # normalized lengths
    lengths = np.linalg.norm([xdiffs, ydiffs], axis=0)
    new_lines[:, 2] = lengths

    # Normalize xdiffs by length
    new_lines[:, 3] = np.divide(xdiffs, lengths, where=new_lines[:, 2] != 0)
    new_lines[:, 4] = np.divide(ydiffs, lengths, where=new_lines[:, 2] != 0)

    return new_lines


def midpoints_distances(gt, pred):
    midpoints = get_midpoints(gt)
    pred_midpoints = get_midpoints(pred)
    return midpoints[0] - pred_midpoints[0], midpoints[1] - pred_midpoints[1]


def get_midpoints(line):
    midpoints = torch.stack([torch.mean(line[[2, 0]]), torch.mean(line[[3, 1]])])
    return midpoints


def get_midpoints_from_np_set(lines):
    midpoints = np.stack([np.mean(lines[:, [2, 0]], axis=1), np.mean(lines[:, [3, 1]], axis=1)], axis=1)
    return midpoints


def midpoints_distance(gt, pred):
    x, y = midpoints_distances(gt, pred)
    return np.linalg.norm([x, y])


def get_points_distances(p_cell, gt_cell, distance_metric: Distance, coords: VariableStructure, max_geom_value,
                         distance_threshold=-1, use_conf=False):
    """
    Calculate cost_matrix between line segments with the given distance metric. All valus higher than max will be set to inf.

    Returns:
        torch.tensor: Cost matrix with preds x gt (rowsxcols)
    """
    # if len(p_cell) != len(gt_cell):
    #     raise ValueError("We want to use the same number of GT and pred but have %s and %s"
    #                      % (len(gt_cell), len(p_cell)))

    # start = timeit.default_timer()
    if len(p_cell) == 0:
        return torch.ones((0, 0), dtype=torch.float32, device=p_cell.device)

    artificial_max = get_max_value(distance_metric=distance_metric,
                                   max_distance=max_geom_value, coords=coords,
                                   use_conf=use_conf)
    clutter_cost = distance_threshold if distance_threshold > 0 else artificial_max
    # default_value = torch.iinfo(torch.int).max if max_val < 0 else max_val  # torch.iinfo(torch.int).max
    pseudo_inf = max(clutter_cost + 100, artificial_max)

    if distance_metric == Distance.EUCLIDEAN:
        fct = linesegment_euclidean_distance
    elif distance_metric == Distance.SQUARED:
        # TODO should the class distance be weighted?
        # fct = linesegment_euclidean_distance  # uses class distance as well
        fct = linesegment_squared_distance
    elif distance_metric == Distance.COSINE:
        fct = linesegment_cosine_distance  # TODO enable class diff
    elif distance_metric == Distance.POINT:
        fct = point_squared_distance
    else:
        raise NotImplementedError("No implementation for %s" % distance_metric)

    gt_isnan_flag = gt_cell[:, 0].isnan()
    if distance_threshold <= 0 and len(p_cell) == len(gt_cell):
        num_rows = num_cols = len(p_cell)
        cost_matrix = torch.ones((num_rows, num_cols), dtype=torch.float32, device=p_cell.device) * pseudo_inf
    else:
        num_rows = num_cols = len(p_cell) + len(gt_cell)
        cost_matrix = torch.ones((num_rows, num_cols), dtype=torch.float32, device=p_cell.device) * pseudo_inf

        # cost for being clutter => assigned to themselves
        cost_matrix[len(p_cell):, 0:len(gt_cell)] = clutter_cost
        cost_matrix[0:len(p_cell):, len(gt_cell):] = clutter_cost
        cost_matrix[len(p_cell):, len(gt_cell):] = clutter_cost

    for idx in torch.where(~gt_isnan_flag)[0]:
        cost_matrix[0:len(p_cell), idx] = fct(gt_cell[idx], p_cell, coords, use_conf)

    return cost_matrix


def get_max_value(distance_metric, max_distance, coords: VariableStructure, use_conf=False):
    max_distance = max(max_distance, 1)

    gt_class = torch.zeros(coords[Variables.CLASS])
    if len(gt_class) > 0:
        gt_class[1] = 1

    pred_class = []
    if Variables.CLASS in coords.train_vars():
        pred_class = torch.zeros(coords[Variables.CLASS])
        pred_class[0] = 1

    if coords.line_representation.enum == LINE.POINTS:
        tensor_pred = torch.tensor([[max_distance, max_distance, 0, 0, *pred_class,
                                     *torch.ones(coords.num_vars_to_train() - 4 - len(pred_class))]])
        tensor_gt = torch.tensor([0, 0, max_distance, max_distance, *gt_class,
                                  *torch.zeros(coords.get_length() - 4 - len(gt_class))])
    elif coords.line_representation.enum == LINE.MID_LEN_DIR and max_distance == 1:
        tensor_pred = torch.tensor([[0, 0, 0.1, -1, -1, *pred_class,
                                     *torch.ones(coords.num_vars_to_train() - 5 - len(pred_class))]])
        tensor_gt = torch.tensor([1, 1, math.sqrt(2), 1, 1, *gt_class,
                                  *torch.zeros(coords.get_length() - 5 - len(gt_class))])
    elif coords.line_representation.enum == LINE.MID_DIR and max_distance == 1:
        tensor_pred = torch.tensor([[0, 0, -1, -1, *pred_class,
                                     *torch.ones(coords.num_vars_to_train() - 4 - len(pred_class))]])
        tensor_gt = torch.tensor([1, 1, 1, 1, *gt_class,
                                  *torch.zeros(coords.get_length() - 4 - len(gt_class))])
    else:
        raise NotImplementedError(coords.line_representation)

    if distance_metric == Distance.SQUARED:
        return linesegment_squared_distance(tensor_gt, tensor_pred, coords=coords, use_conf=use_conf).item()
    elif distance_metric == Distance.EUCLIDEAN:
        return linesegment_euclidean_distance(tensor_gt, tensor_pred, coords=coords, use_conf=use_conf).item()
    elif distance_metric == Distance.COSINE:
        return linesegment_cosine_distance(tensor_gt, tensor_pred, coords=coords, use_conf=use_conf).item()
    elif distance_metric == Distance.POINT:
        return point_squared_distance(tensor_gt, tensor_pred, coords=coords, use_conf=use_conf).item()
    else:
        raise NotImplementedError("No implementation for %s" % distance_metric)
