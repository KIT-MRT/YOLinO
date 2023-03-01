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
import colorsys
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import LineString
from yolino.eval.distances import aml_to_cart
from yolino.grid.coordinates import validate_input_structure
from yolino.grid.predictor import Predictor
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import ImageIdx, CoordinateSystem, ColorStyle, Variables, LINE
from yolino.utils.logger import Log

try:
    import cv2
except:
    Log.error("Somehow cv2 not imported")


def export(lines, debug_folder, name="0313-1_10000.json"):
    jsondict = {"instances": {}}
    for idx, instance in enumerate(lines):

        json_segments = []
        for segment in instance:
            json_segments.append({"start": [int(segment[0][0]), int(segment[0][1])],
                                  "end": [int(segment[1][0]), int(segment[1][1])]})
        jsondict["instances"][idx] = json_segments

    Log.debug("Export to %s" % os.path.join(debug_folder, name))
    with open(os.path.join(debug_folder, name), "w") as f:
        json.dump(jsondict, f)


# put in range of [0, 1]
def squeeze(conf, min=0.9, max=1):
    return (conf - min) / (max - min)


def unsqueeze(r, min=0.9, max=1):
    return r * (max - min) + min


hue_space = []


def draw_aml_cell(cell, image, scale=1, draw_label=True):
    cartesian_cell = torch.ones((len(cell), 4))
    for i, line in enumerate(cell):
        cartesian_cell[i] = aml_to_cart(line)
    cartesian_cell *= scale
    draw_cell(cell=cartesian_cell, image=image, has_conf=False, valid_count=0, total_idx=0, coords=None, cell_size=None,
              threshold=0, colorstyle=ColorStyle.ORIENTATION, cell_indices=None, color=None, thickness=4,
              training_vars_only=False, anchors=None, draw_label=draw_label,
              labels=["a=%.1f m=[%.1f,%.1f] l=%.1f" % (n[0], n[1], n[2], n[3]) for n in cell])


def draw_text(image, label, x1, x2, y1, y2, fontScale=1, thickness=1, colorstyle=ColorStyle.ORIENTATION, idx=None):
    color = get_color(colorstyle=colorstyle, idx=idx, x1=x1, x2=x2, y1=y1, y2=y2)
    color.append(0.5)
    cv2.putText(image, str(label), (int(y1), int(x1)), color=color, fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=fontScale, thickness=thickness)
    return color


def draw_cell(cell, image, valid_count=0, total_idx=0, coords: VariableStructure = None, cell_size=None,
              threshold=0, colorstyle=ColorStyle.ORIENTATION, cell_indices=None, color=None, thickness=4,
              training_vars_only=False, anchors=None, has_conf=True, draw_label=False, labels=[], name="",
              standalone_scale=1):
    points_coords = coords.clone(LINE.POINTS) if coords else None

    if len(cell) == 0:
        Log.warning("Cell was empty")
        return

    if image is None:
        image = np.ones((cell_size[0] * standalone_scale, cell_size[1] * standalone_scale, 3), dtype=float) * 255

    for i_idx, instance in enumerate(cell):  # e.g. segmet = tensor([ 0.3262,  0.2997, -0.3307, -0.6456,  0.0008])
        total_idx += 1

        x1 = y1 = 0 * standalone_scale
        x2 = y2 = 1 * standalone_scale
        geom_pos = [0, 1, 2, 3]
        if points_coords:
            if training_vars_only and Variables.GEOMETRY in points_coords.vars_to_train:
                geom_pos = points_coords.get_position_within_prediction(Variables.GEOMETRY)
            elif not training_vars_only and points_coords[Variables.GEOMETRY] > 0:
                geom_pos = points_coords.get_position_of(Variables.GEOMETRY)

        if np.all(np.isnan(np.asarray(instance[geom_pos]))):
            continue

        if len(geom_pos) == 2:
            x1, y1 = instance[geom_pos] * standalone_scale
            x2 = x1
            y2 = y1
        elif len(geom_pos) == 4:
            x1, y1, x2, y2 = instance[geom_pos] * standalone_scale

        if x1 == x2 and y1 == y2:
            continue

        classes = None
        max_class = None
        if points_coords:
            if training_vars_only and Variables.CLASS in points_coords.vars_to_train:
                classes = np.argmax(
                    np.asarray(instance[points_coords.get_position_within_prediction(Variables.CLASS)]))
            elif not training_vars_only and points_coords[Variables.CLASS] > 0:
                classes = np.argmax(np.asarray(instance[points_coords.get_position_of(Variables.CLASS)]))
            max_class = points_coords[Variables.CLASS]

        conf = -1
        if points_coords:
            if training_vars_only and Variables.CONF in points_coords.vars_to_train:
                conf = instance[points_coords.get_position_within_prediction(Variables.CONF)]
            elif not training_vars_only and points_coords[Variables.CONF] > 0:
                conf = instance[points_coords.get_position_of(Variables.CONF)]
            else:
                conf = 1
        elif has_conf:
            conf = instance[-1]
        else:
            conf = 1

        if cell_indices is not None:
            r, c = cell_indices[total_idx]
            r = int(r * cell_size[0])
            c = int(c * cell_size[1])
            draw_rectangle(image, x1=r, y1=c, x2=r + cell_size[0], y2=c + cell_size[1], idx=i_idx, conf=conf,
                           classification=classes, threshold=threshold,
                           colorstyle=colorstyle, max_class=max_class, color=color, anchors=anchors)

        valid_count += 1
        draw_line(image, x1, y1, x2, y2, idx=i_idx, conf=conf, classification=classes, threshold=threshold,
                  colorstyle=colorstyle, max_class=max_class, color=color, thickness=thickness, anchors=anchors,
                  use_conf=has_conf)
        if draw_label:
            color = draw_text(image, labels[i_idx], x1, x2, y1, y2, colorstyle=colorstyle, fontScale=1,
                              thickness=int(thickness / 2.), idx=i_idx)

    if name is not None and len(name) > 0:
        if image is None:
            Log.error("Could not plot image. Nothing there")
        else:
            Log.warning("Export cell image to file://%s" % (os.path.abspath(name)), level=1)
            cv2.imwrite(name, image)
            Log.img(name, image[..., ::-1], epoch=None, tag=os.path.basename(name), imageidx=ImageIdx.LABEL, level=1)
    return valid_count


def draw_angles(anchor_angles, dataset, anchor_distribution, split, ax=None, finish=True):
    import numpy as np
    from yolino.viz.plot import get_color
    from yolino.utils.enums import ColorStyle

    plt.clf()
    origin = 0.5
    labels = {}
    if ax is None:
        ax = plt

    for i, angle in enumerate(anchor_angles):
        dx = math.cos(angle) * 0.4
        dy = math.sin(angle) * 0.4
        if not abs(angle) <= math.pi:
            raise ValueError()

        atan_ = math.atan2(dy, dx)
        if abs(angle) < math.pi:
            if abs(atan_ - angle) > 0.001:
                raise ValueError("Tangent=%f of x=%f and y=%f should be the same as the angle (%f) "
                                 "constructing x and y." % (atan_, dx, dy, angle))

        color = get_color(ColorStyle.ORIENTATION, angle=angle, bgr=False)
        label = "%.2f at %d" % (angle, i)
        arrow = ax.arrow(origin, origin, dx=dy, dy=dx, label=label,
                         color=np.divide(color, 255.), head_width=0.05, length_includes_head=True)

        labels[arrow] = label

    if finish:
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.gca().invert_yaxis()
        plt.legend(labels.keys(), labels.values(), loc='lower right', ncol=math.ceil(math.sqrt(len(anchor_angles))),
                   fancybox=True)
        plt.title("%s-Anchors %s %s" % (dataset, anchor_distribution, split))

        path = "/tmp/%s_%s_%s_anchors.png" % (dataset, anchor_distribution, split)
        Log.warning("Save anchor to file://%s" % path, level=1)
        plt.savefig(path)
        plt.close("all")
        plt.clf()


def get_color(colorstyle, idx=None, x1=None, x2=None, y1=None, y2=None, angle=None, class_id=-1, max_class=10, conf=-1,
              threshold=0, anchors=None, color=None, bgr=True):
    """

    Args:
        bgr (bool): use True for OpenCV, False for matplotlib
    """
    if color is not None:
        return color

    if colorstyle == ColorStyle.ID:
        # value should be the id
        color = [
            *reversed(tuple(cv2.applyColorMap(np.array((idx + 1) * 30, dtype=np.uint8), COLORMAP).flatten().tolist()))]

    elif colorstyle == ColorStyle.CLASS:
        color = tuple(
            cv2.applyColorMap(np.array(round(class_id) * 255 / max_class, dtype=np.uint8), COLORMAP).flatten().tolist())

    elif colorstyle == ColorStyle.RANDOM:
        color = tuple(cv2.applyColorMap(np.array(random.randint(0, 255), dtype=np.uint8), COLORMAP).flatten().tolist())

    elif colorstyle == ColorStyle.CONFIDENCE:

        conf = max(0, min(1, conf))  # clip to range
        # value should be the confidence [0,1] or [t, 1]
        color = tuple(reversed(cv2.applyColorMap(np.array(squeeze(
            conf, min=threshold) * 255, dtype=np.uint8), COLORMAP).flatten().tolist()))
    elif colorstyle == ColorStyle.CONFIDENCE_BW:
        # value should be the confidence [0,1] or [t, 1]
        color = np.asarray([(1 - squeeze(np.clip(np.asarray(conf), 0, 1), min=threshold)) * 255] * 3).flatten().tolist()
    elif colorstyle == ColorStyle.ORIENTATION:
        # value should be the radian angle
        if angle is None:
            diff = np.asarray([x2, y2]) - np.asarray([x1, y1])
            angle = np.arctan2(diff[1], diff[0])

        # we get angles in range [-pi,pi], but need positive hue values
        offset_angle = angle + math.pi

        # choose a color set that fits your use by shifting through the range
        # use test_anchors.py for visualization of typical line orientations
        color_offset = 0.9
        offset_angle = (offset_angle + color_offset * 2 * math.pi) % (2 * math.pi)

        if offset_angle > 2 * math.pi or offset_angle < 0:
            raise ValueError("Invalid angle=%f for color definition." % offset_angle)

        hue = offset_angle / (2 * math.pi)
        hue_space.append(hue)
        # color = tuple(cv2.applyColorMap(np.array(hue * 255, dtype=np.uint8), COLORMAP).flatten().tolist())
        color = tuple((np.asarray(colorsys.hsv_to_rgb(hue, 1, 1)) * 255).astype(int).tolist())
    elif colorstyle == ColorStyle.ANCHOR:
        color = [*reversed(tuple(
            cv2.applyColorMap(np.array((idx + 1) * 255 / len(anchors), dtype=np.uint8), COLORMAP).flatten().tolist()))]
    else:
        color = tuple(cv2.applyColorMap(np.array(100, dtype=np.uint8), COLORMAP).flatten().tolist())

    if bgr:
        color = [*reversed(color)]
    return color


COLORMAP = cv2.COLORMAP_TURBO


def draw_color_bar(image, colorstyle, threshold):
    if colorstyle == ColorStyle.RANDOM or colorstyle == ColorStyle.UNIFORM or colorstyle == ColorStyle.ID or colorstyle == ColorStyle.CLASS:
        return image

    cell_size = 32
    width = 8 if colorstyle == ColorStyle.ORIENTATION else 4  # times cellsize
    extended_image = np.ones((image.shape[0], image.shape[1] + cell_size * width, 3), dtype=image.dtype) * 255
    extended_image[:, 0:image.shape[1], :] = image

    if colorstyle == ColorStyle.ORIENTATION:
        steps = extended_image.shape[0] / 255.
        for a in range(0, math.ceil(2 * math.pi) * 100, 1):
            angle = a / 100.
            hue = (angle / (2 * math.pi))
            color = np.asarray(colorsys.hsv_to_rgb(hue, 1, 1)) * 255

            x_center = int(image.shape[0] / 2.)
            ext_x = int(x_center - math.cos(angle) * width / 3. * cell_size)
            y_center = int(image.shape[1] + width / 2. * cell_size)
            ext_y = int(y_center - math.sin(angle) * width / 3. * cell_size)
            cv2.line(extended_image, (y_center, x_center), (ext_y, ext_x), color=color, thickness=2)
    elif colorstyle == ColorStyle.CONFIDENCE:
        for i in range(1, 100):
            conf = i / 100.
            # print("%s at bound %s" % (conf, threshold))
            color = get_color(colorstyle, conf=conf, threshold=threshold)
            pt1 = (image.shape[1] + cell_size, math.ceil(5 * i))
            pt2 = (extended_image.shape[1] - 2 * cell_size, math.ceil(5 * (i + 1)))
            cv2.rectangle(extended_image, pt1, pt2, color, -1)

            if i % 5 == 0:
                cv2.putText(extended_image, str(conf), (pt2[0] + 10, pt2[1]), color=(0, 0, 0),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, lineType=cv2.LINE_AA)
    else:
        return image

    return extended_image


def draw_grid(image, cell_size):
    for x in range(0, image.shape[1], cell_size[0]):
        cv2.line(image, tuple(map(int, [x, 0])), tuple(map(int, [x, image.shape[0]])), (50, 50, 50), thickness=1)

    for y in range(0, image.shape[0], cell_size[1]):
        cv2.line(image, tuple(map(int, [0, y])), tuple(map(int, [image.shape[1], y])), (50, 50, 50), thickness=1)


def convert_to_torch_image(image):
    assert (image.shape[2] == 3)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32)
    image = image / 255.

    image = torch.tensor(image)
    image = image.permute(2, 0, 1)

    return image


def convert_torch_to_plt_style(image):
    assert (image.shape[0] == 3)

    image = image.permute(1, 2, 0)

    image = image.numpy()
    image = image * 255.
    image = image.astype(np.uint8)

    return image


def convert_cv2_to_plt_style(image):
    assert (image.shape[2] == 3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.


def convert_to_cv2_image(image):
    assert (image.shape[0] == 3)
    image = image.permute(1, 2, 0)

    image = image.numpy()
    image = image * 255.
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def draw_line(image, x1, y1, x2, y2, idx, conf, classification, threshold, colorstyle, max_class, anchors, color=None,
              thickness=4, use_conf=True):
    if not use_conf or conf > threshold:
        # (w, h)
        color = get_color(colorstyle, idx=idx, x1=x1, x2=x2, y1=y1, y2=y2,
                          class_id=classification,
                          conf=conf,
                          threshold=threshold, max_class=max_class, color=color, anchors=anchors)

        cv2.arrowedLine(image, tuple(map(int, [y1, x1])), tuple(map(int, [y2, x2])),
                        color,
                        thickness=thickness,
                        tipLength=0.15)


def draw_rectangle(image, x1, y1, x2, y2, idx, conf, classification, threshold, colorstyle, max_class, anchors,
                   color=None):
    if colorstyle == ColorStyle.ID and idx == -1:
        return

    if conf is None or conf > threshold:
        # (w, h)
        color = get_color(colorstyle, idx=idx, x1=x1, x2=x2, y1=y1, y2=y2,
                          class_id=classification,
                          conf=conf,
                          threshold=threshold, max_class=max_class, color=color, anchors=anchors)
        cv2.rectangle(image, (y1, x1), (y2, x2), color=color, thickness=1)


def plot_cell_class(grid, name, image, epoch, tag, imageidx: ImageIdx, ignore_classes=[], max_class=10, fill=False,
                    threshold=0):
    image = np.array(image)
    assert (image.shape[0] <= 3)
    assert (image.dtype == np.float32)
    image = np.transpose(image, (1, 2, 0))
    image = image * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    valid_count = 0
    for r, row in enumerate(grid.cells):
        for c, _ in enumerate(row):
            if grid.cells[r, c] is None:
                continue
            current_predictors = grid.cells[r, c].predictors
            for p_idx, p in enumerate(current_predictors):
                p: Predictor
                if p.confidence < threshold:
                    continue

                cs = grid.get_cell_size(image.shape[0])

                if len(p.label) == 0:
                    Log.error("No class label to plot in %d,%d" % (r, c))
                    continue

                class_id = int(np.argmax(p.label))
                if len(ignore_classes) == 0 or class_id not in ignore_classes:
                    color = get_color(ColorStyle.CLASS, class_id=class_id, max_class=max_class)
                    x1 = int((r * cs[0]) + 2)
                    y1 = int((c * cs[1]) + 2)
                    x2 = int(((r + 1) * cs[0]) - 2)
                    y2 = int(((c + 1) * cs[1]) - 2)

                    valid_count += 1

                    try:
                        # make thicker if multiple predictors are stacked so all are visible
                        cv2.rectangle(image, (y1, x1), (y2, x2), color=color,
                                      thickness=-1 if fill else 2 * (len(current_predictors) - p_idx + 1))
                    except cv2.error as ex:
                        Log.error("Rectangle cannot be build from (%f, %f) and (%f, %f) with color=%s" % (
                            y1, x1, y2, x2, str(color)))
                        raise ex

    if name != "" and name is not None:
        if valid_count == 0:
            Log.warning("We plotted no data, is that correct? Ignore=%s\n%s"
                        % (ignore_classes, Log.get_pretty_stack()))
        Log.warning("Export %d classification to file://%s" % (valid_count, os.path.abspath(name)))

        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))

        cv2.imwrite(name, image)
        # TODO: use https://github.com/nschloe/tikzplotlib

        Log.img(name, image[..., ::-1], epoch, tag=tag, imageidx=imageidx)

    return image, valid_count


def save_grid(images, path, title="grid"):
    import torchvision
    grid = torchvision.utils.make_grid(images, nrow=math.ceil((len(images) / 4)))
    grid = convert_to_cv2_image(grid)
    import os
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    Log.warning(("Write " + title + " to file://%s") % path, level=1)
    cv2.imwrite(path, grid)


# image should have shape (3, w, h) as float [0,1]
def plot(lines, name, image, coords: VariableStructure = None, show_grid=False, cell_size=None, threshold=0,
         colorstyle=ColorStyle.ORIENTATION, show_color_bar=False,
         coordinates: CoordinateSystem = CoordinateSystem.UV_SPLIT, epoch=-1, tag="runner",
         imageidx: ImageIdx = ImageIdx.DEFAULT, cell_indices=None, color=None, thickness=4,
         training_vars_only=False, anchors=None, show=False, level=0):
    import timeit
    start_time = timeit.default_timer()
    if image is None:
        if coordinates == CoordinateSystem.UV_SPLIT:
            if type(lines) == torch.tensor:
                x1, y1, x2, y2 = torch.max(lines[:, :, 0:4], dim=1)[0][0]
            else:
                x1, y1, x2, y2 = np.nanmax(np.nanmax(lines[:, :, 0:4], axis=1), axis=0)
        elif coordinates == CoordinateSystem.UV_CONTINUOUS:
            x1, y1 = np.nanmax(np.nanmax(lines[0], axis=1), axis=0)
            x2 = x1
            y2 = y1
        else:
            raise NotImplementedError(coordinates)
        image = torch.ones((3, int(max(x1, x2)), int(max(y1, y2))), dtype=torch.float32)

    image = np.array(image)

    if len(image.shape) > 3:
        raise ValueError("Please provide one image at a time with 3xhxw. Instead we have got %s" % str(image.shape))

    if image.shape[0] <= 3:
        assert (image.dtype == np.float32)
        image = np.transpose(image, (1, 2, 0))
        image = image * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    assert (image.shape[2] <= 3)

    validate_input_structure(lines, coordinates)

    if show_grid and cell_size is not None:
        draw_grid(image, cell_size)

    # (batch, instances, control points, ?)
    valid_count = 0
    total_idx = -1
    if coordinates == CoordinateSystem.UV_CONTINUOUS:
        for b_idx, batch in enumerate(lines):
            for i_idx, instance in enumerate(batch):
                for idx in range(len(instance) - 1):
                    total_idx += 1
                    if type(instance) == torch.Tensor:
                        if torch.all(instance[idx + 1] == 0) or torch.any(torch.isnan(instance[idx:idx + 2])):
                            continue
                    elif type(instance) == np.ndarray:
                        if np.all(instance[idx + 1] == 0) or np.any(np.isnan(instance[idx:idx + 2])):
                            continue

                    x1 = instance[idx][0]
                    y1 = instance[idx][1]
                    x2 = instance[idx + 1][0]
                    y2 = instance[idx + 1][1]
                    valid_count += 1

                    # TODO: handle conf and class
                    draw_line(image, x1, y1, x2, y2, idx=i_idx,
                              conf=-1, classification=None, threshold=0, colorstyle=colorstyle, max_class=None,
                              color=color, thickness=thickness, anchors=anchors, use_conf=False)
    elif coordinates == CoordinateSystem.UV_SPLIT:
        for b_idx, batch in enumerate(lines):
            valid_count = draw_cell(cell=batch, image=image, valid_count=valid_count, total_idx=total_idx,
                                    coords=coords, cell_size=cell_size, threshold=threshold, colorstyle=colorstyle,
                                    cell_indices=cell_indices, color=color, thickness=thickness,
                                    training_vars_only=training_vars_only, anchors=anchors)
    elif coordinates == CoordinateSystem.CELL_SPLIT:
        raise AttributeError("Plot can only visualize UV based coordinates.")
    else:
        raise NotImplementedError

    if show_color_bar:
        image = draw_color_bar(image, colorstyle, threshold=threshold)

    if name != "" and name is not None:
        Log.warning("Export %d lines (%s) to file://%s" % (valid_count, coordinates.name, os.path.abspath(name)),
                    level=level + 1)

        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))

        cv2.imwrite(name, image)

        Log.img(name, image[..., ::-1], epoch, tag=tag, imageidx=imageidx, level=level + 1)

    if False:
        hist, edges = np.histogram(hue_space, bins=20)
        printable_hist = {e: h for h, e in zip(hist, edges)}

        plt.clf()
        plt.hist(hue_space, 20)

        path = "tmp/hue.png"
        Log.warning("Plot to file://%s" % path)
        plt.savefig(path)
        plt.close('all')

        Log.debug("Hue space is %s" % (printable_hist))

    Log.time(key="plot", value=timeit.default_timer() - start_time)
    return image, valid_count


def plot_style_grid(lines, name, image, coords: VariableStructure = None, show_grid=False, cell_size=None, threshold=0,
                    show_color_bar=False, coordinates: CoordinateSystem = CoordinateSystem.UV_SPLIT, epoch=-1,
                    tag="runner", show_only_train=True,
                    imageidx: ImageIdx = ImageIdx.DEFAULT, gt=None, training_vars_only=False, anchors=None, level=0):
    grid_images = []
    thickness = int(image.shape[2] / 160)

    styles = [ColorStyle.ORIENTATION, ColorStyle.CLASS, ColorStyle.CONFIDENCE, ColorStyle.RANDOM]
    for style in styles:
        if style == ColorStyle.CLASS and (coords[Variables.CLASS] == 0
                                          or (show_only_train and not Variables.CLASS in coords.train_vars())):
            continue

        if style == ColorStyle.CONFIDENCE and not Variables.CONF in coords.train_vars():
            continue

        if style == ColorStyle.CONFIDENCE:
            thresholds = [0]
        else:
            thresholds = np.unique([threshold])

        for t in thresholds:
            line_img, ok = plot(lines, name="", image=image, coords=coords,
                                show_grid=show_grid,
                                cell_size=cell_size,
                                threshold=t, show_color_bar=show_color_bar,
                                coordinates=coordinates, epoch=epoch,
                                tag=tag, imageidx=imageidx, colorstyle=style, training_vars_only=training_vars_only,
                                anchors=anchors, level=level + 1,
                                thickness=math.ceil(thickness / 2) if style == ColorStyle.RANDOM else thickness)
            line_img = cv2.rectangle(img=line_img, pt1=(0, 0), pt2=(200, 25), color=(255, 255, 255), thickness=-1)
            line_img = cv2.putText(img=line_img, text="%s [t=%s]" % (style, t), org=(5, 20), color=(0, 0, 0),
                                   fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)
            grid_images.append(convert_to_torch_image(line_img))

    if gt is not None:
        line_img, ok = plot(gt, name="", image=image, coords=coords, show_grid=show_grid,
                            cell_size=cell_size,
                            show_color_bar=show_color_bar, coordinates=coordinates, epoch=epoch,
                            tag=tag, imageidx=imageidx,
                            colorstyle=ColorStyle.CLASS
                            if Variables.CLASS in coords.train_vars() else ColorStyle.ORIENTATION,
                            anchors=anchors, level=level + 1, thickness=thickness)
        line_img = cv2.rectangle(img=line_img, pt1=(0, 0), pt2=(200, 25), color=(255, 255, 255), thickness=-1)
        line_img = cv2.putText(img=line_img, text="ground truth", org=(5, 20), color=(0, 0, 0),
                               fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)
        grid_images.append(convert_to_torch_image(line_img))
    Log.grid(name=name, images=grid_images, epoch=epoch, tag=tag, imageidx=imageidx, level=level + 1)

    expected_num_images = (len(styles) + 1 - int(Variables.CLASS not in coords.train_vars()) - int(
        Variables.CONF not in coords.train_vars()) - int(gt is None))
    return len(grid_images) == expected_num_images


def plot_debug_geometry(geom_box, geom_line, line_segment):
    plt.plot(
        geom_box.boundary.xy[1], geom_box.boundary.xy[0])
    plt.plot(geom_line.xy[1], geom_line.xy[0], '.r-')

    if line_segment is not None:
        if type(line_segment) == LineString:
            plt.plot(line_segment.coords.xy[1],
                     line_segment.coords.xy[0], '.g-')
        else:
            plt.plot(line_segment[:, 1],
                     line_segment[:, 0], '.g-')


def finish_plot_debug_geometry(suffix=""):

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    debug_image_path = "tmp/straight%s.png" % suffix
    import logging
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    logging.getLogger(
        'matplotlib.font_manager').disabled = True
    Log.warning("Export debug viz to file://%s" %
                os.path.abspath(debug_image_path), level=1)

    start, end = plt.gca().get_ylim()
    int_end = math.floor(end / 32) * 32
    int_start = int(start / 32) * 32
    major_ticks = np.arange(int_start, int_end + 32, 32)
    plt.gca().yaxis.set_ticks(major_ticks)

    start, end = plt.gca().get_xlim()
    int_end = math.floor(end / 32) * 32
    int_start = int(start / 32) * 32
    major_ticks = np.arange(int_start, int_end + 32, 32)
    plt.gca().xaxis.set_ticks(major_ticks)

    plt.grid()

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()

    plt.title("Straighten line")
    plt.savefig(debug_image_path)
    plt.close()
    plt.clf()
    plt.close('all')


def plot_debug_geometry_area(c, cs, r, polygon_area, geom_box, long_uv_line, plot_image, title="Straighten line"):
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([c - cs[1], c + 2 * cs[1]])
    axes.set_xlim([r - cs[0], r + 2 * cs[0]])
    plt.plot(geom_box.boundary.xy[0], geom_box.boundary.xy[1])
    plt.plot(long_uv_line.xy[0], long_uv_line.xy[1], '.r-')

    plt.plot(polygon_area.exterior.xy[0], polygon_area.exterior.xy[1], color='#00ff00')

    if plot_image:
        debug_image_path = "tmp/straightarea.png"
        import logging
        pil_logger = logging.getLogger('PIL')
        pil_logger.setLevel(logging.INFO)
        logging.getLogger(
            'matplotlib.font_manager').disabled = True
        Log.warning("Export debug viz of straightening to file://%s" %
                    debug_image_path)
        plt.title(title)
        plt.savefig(debug_image_path)
        plt.close()
    plt.close('all')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Add grid to image")
    parser.add_argument("image", type=str, help="Provide path to an image")
    parser.add_argument("-s", "--grid_shape", nargs=2, type=int, default=None,
                        help="Specify number of grid cells, [rows, cols].")
    parser.add_argument("-c", "--cell_size", nargs=2, type=int, default=None,
                        help="Specify pixels per cell, [height, width].")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        Log.error("Unknown path %s" % args.image)
        exit(2)
    path, ext = os.path.splitext(args.image)

    import cv2

    if ext == ".png" or ext == ".jpg":
        img = cv2.imread(args.image)
    elif ext == ".npy":
        img = np.load(args.image, allow_pickle=True)
    else:
        Log.error("Please provide either png, jpg or npy files. We found %s" % ext)
        exit(3)

    if args.grid_shape is None:
        if args.cell_size is None:
            Log.error("Please provide one of --cell_size and --grid_shape.")
        else:
            args.grid_shape = np.array([img.shape[0], img.shape[1]]) / np.array(args.cell_size)

    Log.debug("Grid Size: %s" % args.grid_shape)
    Log.debug("Image Size: %s" % str(img.shape))

    img, _ = plot([], path + "_grid.png", img.astype(np.uint8), show_grid=True, cell_size=args.cell_size)
