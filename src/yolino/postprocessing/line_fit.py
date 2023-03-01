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
from copy import deepcopy

import numpy as np
from scipy import interpolate

from yolino.dataset.tusimple_pytorch import TusimpleDataset
from yolino.grid.coordinates import validate_input_structure
from yolino.model.variable_structure import VariableStructure
from yolino.tools.tusimple_benchmark import LaneEval
from yolino.utils.enums import CoordinateSystem, Variables, ImageIdx, ColorStyle, LINE
from yolino.viz.plot import plot, get_color


def breadth_first_connected_component(graph, start):
    # keep track of all visited nodes
    explored = []
    explored_tuple = []
    # keep track of nodes to be checked
    level = 0
    queue = [(start, level)]
    # keep looping until there are nodes still to be checked
    while queue:
        # pop shallowest node (first node) from queue
        node, level = queue.pop(0)
        if node not in explored:
            # add node to list of checked nodes
            explored.append(node)
            explored_tuple.append((node, level))
            neighbours = graph[node]
            level += 1
            # add neighbours of node to queue
            for neighbour in neighbours:
                queue.append((neighbour, level))
    return explored_tuple


def fit_lines(lines_uv, coords: VariableStructure, confidence_threshold, adjacency_threshold, grid_shape,
              min_segments_for_polyline,
              cell_size, image, file_name, paths, args, split):
    validate_input_structure(lines_uv, CoordinateSystem.UV_SPLIT)
    conf_pos = coords.get_position_within_prediction(Variables.CONF)
    geom_pos = coords.get_position_within_prediction(Variables.GEOMETRY)

    # filter by conf
    lines_uv = lines_uv[np.where(lines_uv[:, :, conf_pos] > confidence_threshold)[0:2]]

    name = paths.generate_debug_image_file_path(file_name=file_name, idx=ImageIdx.PRED, suffix="filter")
    img = deepcopy(image)
    img, ok = plot(np.expand_dims(lines_uv, axis=0), name, img, coords=coords,
                   colorstyle=ColorStyle.UNIFORM,
                   coordinates=CoordinateSystem.UV_SPLIT, imageidx=ImageIdx.PRED, training_vars_only=True)

    segments = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in lines_uv[:, geom_pos]]
    startpoints = np.array(lines_uv[:, 0:2])
    endpoints = np.array(lines_uv[:, 2:4])
    confidences = np.array(lines_uv[:, conf_pos])

    # build adjacency list
    adjacency, reversed_adjacency = get_adjacency_list(adjacency_threshold, endpoints, startpoints)

    # Infer roots
    roots = [node for node in adjacency.keys() if not adjacency[node]]

    # Find connected components
    polylines, polylines_as_segment_ids = get_connected_components(confidences, file_name, image,
                                                                   min_segments_for_polyline, paths, reversed_adjacency,
                                                                   roots, segments)

    # Smooth poylines
    end_start_distances, smooth_polylines = smoothing_polylines(file_name, image, paths, polylines)

    # if more than one polylines contain the same segments, only one can remain (possibly based on connectedness)
    to_be_removed = remove_duplicates(end_start_distances, polylines_as_segment_ids)

    # make spline
    splines = fit_spline(cell_size, file_name, image, paths, smooth_polylines, to_be_removed)

    # # Create submission
    # y_samples = np.linspace(160, 710, 56)
    # splines_reformatted = [[spline[0] * 2, spline[1] * 2 + 80] for spline in splines]
    # submission = []
    # for spline in splines_reformatted:
    #     x = spline[0]
    #     y = spline[1]
    #     x_result_values = np.interp(y_samples, y, x, -2, -2)
    #     submission.append(x_result_values.tolist())

    # # Evaluate
    # tus = TusimpleDataset(split=split, args=args, load_only_labels=True)
    # gt_lanes = tus.lanes

    # # TODO fix tus benchmark
    # le = LaneEval()
    # accuracy, fp, fn = le.bench(submission, gt_lanes, y_samples, 0)
    # results_string = "(Acc %.3f, FP %.2f, FN %.2f)" % (accuracy, fp, fn)

    # from skimage.transform import resize
    # upscaled_image = resize(image.copy() / 255, (640, 1280))
    # mask = np.zeros((720, 1280, 3))
    # mask[80:, :, :] += upscaled_image

    return splines


def fit_spline(cell_size, file_name, image, paths, smooth_polylines, to_be_removed):
    splines = []
    for idx, smooth_polyline in enumerate(smooth_polylines):
        if idx in to_be_removed:
            continue
        lp = [l[0] for l in smooth_polyline]
        lp.append(smooth_polyline[-1][1])
        lp = np.array(lp)
        x = lp[:, 0] / cell_size[0]
        y = lp[:, 1] / cell_size[1]
        # TODO we could use the confidence here as well
        # Larger s means more smoothing while smaller values of s indicate less smoothing.
        tck, u = interpolate.splprep([x, y], s=0.05)
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)
        splines.append(np.array(out) * cell_size[0])
    name = paths.generate_debug_image_file_path(file_name=file_name, idx=ImageIdx.PRED, suffix="spline")
    img = deepcopy(image)
    plot_coords = VariableStructure(line_representation_enum=LINE.POINTS, num_conf=0,
                                    vars_to_train=[Variables.GEOMETRY])
    # for instance in splines:
    converted_splines = np.expand_dims(
        np.asarray([[[x, y] for x, y in zip(instance[0], instance[1])] for instance in splines]), axis=0)
    plot(converted_splines, name, img, coords=plot_coords,
         colorstyle=ColorStyle.ID,
         coordinates=CoordinateSystem.UV_CONTINUOUS, imageidx=ImageIdx.PRED, training_vars_only=True)
    return splines


def remove_duplicates(end_start_distances, polylines_as_segment_ids):
    duplicates = set([])
    for id, pl in enumerate(polylines_as_segment_ids):
        others = range(len(polylines_as_segment_ids))
        for seg_id in pl:
            for other in others:
                if other != id:
                    if seg_id in polylines_as_segment_ids[other]:
                        duplicates.add((id, other))
    to_be_removed = []
    for duplicate in duplicates:
        if end_start_distances[duplicate[0]] <= end_start_distances[duplicate[1]]:
            to_be_removed.append(duplicate[0])
        else:
            to_be_removed.append(duplicate[1])
    if to_be_removed:
        print("to_be_removed", to_be_removed)
    return to_be_removed


def smoothing_polylines(file_name, image, paths, polylines):
    smooth_polylines = []
    end_start_distances = []
    for polyline in polylines:
        smoothed = []
        prev = None
        end_start_distance = 0
        for l_idx, line in enumerate(polyline):
            if prev is not None:
                end_start_distance += np.linalg.norm(np.array(prev[1]) - np.array(line[0]))
                midpoint = tuple((np.array(prev[1]) + np.array(line[0])) / 2)
                smoothed.append((prev[0], midpoint))
                prev = (midpoint, line[1])
            else:
                prev = line
        smoothed.append(prev)
        end_start_distances.append(end_start_distance)
        smooth_polylines.append(smoothed)
    name = paths.generate_debug_image_file_path(file_name=file_name, idx=ImageIdx.PRED, suffix="smooth")
    img = deepcopy(image)
    plot_coords = VariableStructure(line_representation_enum=LINE.POINTS, num_conf=0,
                                    vars_to_train=[Variables.GEOMETRY])
    for i, instance in enumerate(smooth_polylines):
        color = get_color(colorstyle=ColorStyle.ID, idx=i)
        img, ok = plot(np.asarray([instance]).reshape((1, -1, 4)), name, img, coords=plot_coords,
                       colorstyle=ColorStyle.UNIFORM, color=color, coordinates=CoordinateSystem.UV_SPLIT,
                       imageidx=ImageIdx.PRED, training_vars_only=True)
    return end_start_distances, smooth_polylines


def get_connected_components(confidences, file_name, image, min_segments_for_polyline, paths, reversed_adjacency, roots,
                             segments):
    # visits all the nodes of a graph (connected component) using BFS
    polylines = []
    polylines_as_segment_ids = []
    for root in roots:
        polyline = []
        polyline_as_segment_ids = []
        bfs_result = breadth_first_connected_component(reversed_adjacency, root)
        current_level = 0
        merge_segments = []
        merge_confidences = []
        for segment_id, level in bfs_result:
            polyline_as_segment_ids.append(segment_id)
            if level == current_level:
                merge_segments.append(np.array(segments[segment_id]))
                merge_confidences.append(confidences[segment_id])
            else:
                weights = np.array(merge_confidences) / sum(merge_confidences)
                merged_segment = np.sum(
                    np.array([merge_segment * weight for merge_segment, weight in zip(merge_segments, weights)]),
                    axis=0)
                merged_segment = tuple([tuple(x) for x in merged_segment])
                polyline.append(merged_segment)
                merge_segments = [np.array(segments[segment_id])]
                merge_confidences = [(confidences[segment_id])]
                current_level = level
        weights = np.array(merge_confidences) / sum(merge_confidences)
        merge_segments[0] * weights[0]
        merged_segment = np.sum(
            np.array([merge_segment * weight for merge_segment, weight in zip(merge_segments, weights)]), axis=0)
        merged_segment = tuple([tuple(x) for x in merged_segment])
        polyline.append(merged_segment)
        polylines.append(polyline[::-1])
        polylines_as_segment_ids.append(polyline_as_segment_ids)
    polyline_num_threshold = int(min_segments_for_polyline)
    polylines = [pl for pl in polylines if len(pl) >= polyline_num_threshold]
    polylines_as_segment_ids = [pl for pl in polylines_as_segment_ids if len(pl) >= polyline_num_threshold]

    name = paths.generate_debug_image_file_path(file_name=file_name, idx=ImageIdx.PRED, suffix="cc")
    img = deepcopy(image)
    plot_coords = VariableStructure(line_representation_enum=LINE.POINTS, num_conf=0,
                                    vars_to_train=[Variables.GEOMETRY])
    for i_idx, instance in enumerate(polylines):
        c = get_color(ColorStyle.ID, idx=i_idx)
        img, ok = plot(np.asarray(instance).reshape((1, -1, 4)), name, img, coords=plot_coords,
                       colorstyle=ColorStyle.UNIFORM,
                       color=c, #(int(instance[0][0][1]) % 255, int(instance[0][0][0]) % 255, int(instance[0][1][0]) % 255),
                       coordinates=CoordinateSystem.UV_SPLIT, imageidx=ImageIdx.PRED, training_vars_only=True)
    return polylines, polylines_as_segment_ids


def get_adjacency_list(adjacency_threshold, endpoints, startpoints):
    adjacency_d_threshold = adjacency_threshold
    adjacency = dict()
    for idx, endpoint in enumerate(endpoints):
        distances = np.sum((startpoints - endpoint) ** 2, axis=1)
        d_argsort = distances.argsort()
        successor_idx = 0 if d_argsort[0] != idx else 1
        d_value = distances[d_argsort[successor_idx]]
        y_distance = startpoints[d_argsort[successor_idx]][1] - endpoint[1]
        # distance must be below threshold and it cannot be a segment in the (bottom half of the) bottom most row
        # and the y distance from current to its successor can not decrease too much (to prevent rings/circular graphs)
        # ideally this is the point where you'd want to implement a more
        # sophisticated graph building algorithm that features proper optimization
        # terms
        if d_value <= adjacency_d_threshold and \
                y_distance >= -0.25 * 32 * 32:
            # endpoint[1] < (grid_shape[0] - 0.5) and \
            adjacency[idx] = [d_argsort[0]] if d_argsort[0] != idx else [d_argsort[1]]
        else:
            adjacency[idx] = []

    # Build reveserd adjacency lsit
    revesed_adjacency = {}
    for key, value in adjacency.items():
        if value:
            value = value[0]
            if value not in revesed_adjacency.keys():
                revesed_adjacency[value] = [key]
            else:
                revesed_adjacency[value].append(key)
    for key, value in adjacency.items():
        if key not in revesed_adjacency.keys():
            revesed_adjacency[key] = []

    return adjacency, revesed_adjacency
