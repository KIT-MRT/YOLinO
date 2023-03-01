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

import numpy as np
import torch
from shapely.geometry import Polygon, LineString
from yolino.utils.logger import Log


def calc_geom_from_angle(x, y, angle, max_length=[1, 1], invert=False):
    if invert:
        return torch.tensor([x, y, x - math.cos(angle) * (max_length[0] - x),
                             y - math.sin(angle) * (max_length[1] - y)])
    else:
        return torch.tensor([x, y, x + math.cos(angle) * (max_length[0] - x),
                             y + math.sin(angle) * (max_length[1] - y)])


def calc_geom_from_direction(x, y, direction, max_length=[1, 1], invert=False):
    if invert:
        return torch.tensor([x, y, x - direction[0] * (max_length[0] / 2), y - direction[1] * (max_length[1] / 2)])
    else:
        return torch.tensor([x, y, x + direction[0] * (max_length[0] / 2), y + direction[1] * (max_length[1] / 2)])


def intersection_square(top_left, width, geom_line):
    return intersection_box(top_left, width, width, geom_line)


def handle_shapely_intersection(geom_intersection, segments, offset_h, offset_w):
    if geom_intersection.type == "LineString":
        try:
            segments = torch.concat([segments, torch.ones((1, 500, 2)) * torch.nan])
            for idx, point in enumerate(zip(geom_intersection.xy[0], geom_intersection.xy[1])):
                segments[-1, idx] = (torch.tensor(point) - torch.tensor([offset_h, offset_w]))
        except NotImplementedError:
            Log.error(
                "Invalid line string after crop %s with %s" % (geom_intersection.type, str(geom_intersection)))
            raise NotImplementedError(
                "Somehow we received an invalid type of linestring from shapely. Please check %s" % str(
                    geom_intersection))
    elif geom_intersection is None or geom_intersection.is_empty:
        Log.debug("Full polyline will be erased from the labels")
        return []

    elif geom_intersection.type == "MultiLineString" or geom_intersection.type.lower() == "geometrycollection":
        connect = False
        for i in range(len(geom_intersection.geoms)):
            g = geom_intersection.geoms[i]
            xy = torch.tensor(g.xy)
            xy[0] -= offset_h
            xy[1] -= offset_w

            if 0 < i < len(geom_intersection.geoms) - 1:
                next_xy = torch.tensor(geom_intersection.geoms[i + 1].xy)
                next_xy[0] -= offset_h
                next_xy[1] -= offset_w

                prev_xy = torch.tensor(geom_intersection.geoms[i - 1].xy)
                prev_xy[0] -= offset_h
                prev_xy[1] -= offset_w

                if len(xy[0]) == 2 and torch.all(xy[:, 0] == next_xy[:, 0]) and torch.all(prev_xy[:, -1] == xy[:, 0]):
                    last_valid_index = torch.where(segments[-1, :, 0].isnan())[0][0] - 1

                    Log.info(f"The center split is just a stage around p1={xy[0, 0].item(), xy[1, 0].item()}. "
                             f"The remainder of that stage has only p2={xy[0, 1].item(), xy[1, 1].item()}. "
                             f"We removed it. Segments so far\n{segments[-1, last_valid_index - 4:last_valid_index + 1]}...")
                    connect = True
                    continue

            if not connect:
                total_idx = 0
                segments = torch.concat([segments, torch.ones((1, 500, 2)) * torch.nan])

            for idx, point in enumerate(zip(xy[0], xy[1])):
                if connect:
                    Log.info(f"We append\n{xy[:, 1:3]}")
                    connect = False
                    continue

                segments[-1, total_idx] = torch.tensor(point)
                total_idx += 1
    return segments


def point_2_line_distance(point, line_p1, line_p2):
    line = line_p2 - line_p1
    norm = np.linalg.norm(line)

    if norm == 0:
        return np.linalg.norm(line_p1 - point)

    return np.abs(np.linalg.norm(np.cross(line, line_p1 - point))) / norm


def max_polyline_2_line_distance(polyline, line_p1, line_p2):
    max_dist = 0
    for point in polyline:
        line = line_p2 - line_p1
        norm = np.linalg.norm(line)

        if norm == 0:
            dist = np.linalg.norm(line_p1 - point)
        else:
            dist = np.abs(np.linalg.norm(np.cross(line, line_p1 - point))) / norm
        if dist > max_dist:
            max_dist = dist
    return max_dist


def intersection_segments(top_left, width, height, geom_line: np.ndarray, segments: np.ndarray, lib="shapely"):
    r, c = top_left

    geom_intersection, geom_box = intersection_box(top_left, width, height, geom_line)
    if lib == "shapely":
        segments = handle_shapely_intersection(geom_intersection, segments, offset_h=r, offset_w=c)
    else:
        raise NotImplementedError(lib)

    return segments


def intersection_box(top_left, width, height, geom_line):
    r, c = top_left

    if type(geom_line) == np.ndarray and np.any(np.isnan(geom_line)):
        Log.error("Nan")
        return None, None
    geom_box = Polygon([(r, c),
                        (r + height, c),
                        (r + height, c + width),
                        (r, c + width),
                        (r, c)])

    if geom_box.area == 0:
        Log.warning("Invalid box %s" % (geom_box))
        raise ValueError("Invalid box %s" % (geom_box))

    geom_intersection = geom_box.intersection(geom_line.simplify(0.8))
    return geom_intersection, geom_box


def getExtrapoledLine(p1, p2):
    'Creates a line extrapoled in p1->p2 direction'

    length = (math.pow(p2[1] - p1[1], 2) + math.pow(p2[0] - p1[0], 2))
    EXTRAPOL_RATIO = 10
    a = p1
    b = (p1[0] + EXTRAPOL_RATIO * (p2[0] - p1[0]), p1[1] + EXTRAPOL_RATIO * (p2[1] - p1[1]))
    return LineString([a, b])


def reformat2shapely(line_segment):
    line_segment = np.reshape(line_segment, (-1, 2))
    is_nan = np.all(np.isnan(line_segment), axis=1)

    # check if any adjacent points have a same coordinate
    where_first = line_segment[:-1, 0] == line_segment[1:, 0]
    where_snd = line_segment[:-1, 1] == line_segment[1:, 1]

    where_only_first = np.logical_and(where_first, np.logical_not(where_snd))
    if np.any(where_only_first):
        line_segment[[*where_only_first, False], 0] += np.random.random(1) / 1000.

    where_only_snd = np.logical_and(np.logical_not(where_first), where_snd)
    if np.any(where_only_snd):
        line_segment[[*where_only_snd, False], 1] += np.random.random(1) / 1000.

    where_first = line_segment[:-1, 0] == line_segment[1:, 0]
    where_snd = line_segment[:-1, 1] == line_segment[1:, 1]
    where_both = np.logical_and(where_first, where_snd)
    if np.any(where_both):
        bad_indices = np.where(where_both)[0]
        Log.info(f"{len(bad_indices)} times we have the same point in a line at indices={bad_indices}."
                 f"We delete the first of each pair as shapely is not able to process otherwise.")
        line_segment = np.stack([np.delete(line_segment[:, 0], bad_indices),
                                 np.delete(line_segment[:, 1], bad_indices)],
                                axis=1)

    if len(line_segment) == 1:
        raise ValueError("This line segment is only a point %s" % str(line_segment))

    geom_line = LineString(line_segment.tolist())
    return geom_line


def line_segment_to_sampled_points(line_segment, variables, sample_distance):
    # get direction of the segment
    vector = line_segment[2:4] - line_segment[0:2]
    dist = np.linalg.norm(vector)
    points = np.array([line_segment[0:2] + i * vector / dist for i in range(0, int(dist), sample_distance)])

    direction = np.arctan2(vector[1], vector[0])
    if direction < 0:
        direction = direction * -1 + math.pi
    a = math.degrees(direction)
    points = np.column_stack((points, np.repeat(a, len(points)), np.tile(variables, [len(points), 1])))

    return points


def t_cart2pol(point):
    """
    angle is in range of 0 to PI
    distance is in range -sqrt(2) to sqrt(2)

    converts a tansor of [x_values, y_values] to [rhos, phis]
    list of points can be converted to proper input format via
    data.transpose(1,0)
    """
    x, y = point
    rho = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    rho[phi < 0] *= -1
    phi[phi < 0] += math.pi
    return torch.stack((rho, phi))


def t_pol2cart(polar):
    """converts a tansor of [rhos, phis] to [x_values, y_values]"""
    x = polar[0] * torch.cos(polar[1])
    y = polar[0] * torch.sin(polar[1])
    return torch.stack((x, y))


def shift_polylines(polylines, height_offset, width_offset):
    Log.debug("Shift labels by %sx%s" % (height_offset, width_offset))
    height_offset = int(height_offset)
    width_offset = int(width_offset)
    for line in polylines:
        for segment in line:
            segment -= [height_offset, width_offset]  # pass by reference, here

    return polylines
