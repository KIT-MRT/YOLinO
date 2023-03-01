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
import unittest
from copy import copy

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.grid_factory import GridFactory
from yolino.model.anchors import Anchor
from yolino.model.line_representation import LineRepresentation, MidDirLines
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import Dataset, AnchorVariables, AnchorDistribution, ColorStyle, LINE, Variables, ImageIdx, \
    CoordinateSystem, ACTIVATION
from yolino.utils.test_utils import test_setup
from yolino.viz.plot import draw_angles, draw_cell, plot


class AnchorTest(unittest.TestCase):
    def test_anchor_point_sorting(self):
        self.do_anchor_sorting(LINE.POINTS, [AnchorVariables.POINTS])

    def test_anchor_md_sorting(self):
        self.do_anchor_sorting(LINE.MID_DIR, [AnchorVariables.MIDPOINT, AnchorVariables.DIRECTION], 16)

    def do_anchor_sorting(self, linerep, anchor_var, num_p=8):
        args = test_setup(self._testMethodName + "_" + str(linerep), str(Dataset.TUSIMPLE),
                          additional_vals={"num_predictors": num_p,
                                           "explicit": "clips/0313-2/36600/20.jpg",
                                           "anchors": str(AnchorDistribution.EQUAL),
                                           "linerep": str(linerep),
                                           "anchor_vars": anchor_var,
                                           # "plot": True
                                           },
                          # level=Level.DEBUG
                          )
        dataset, _ = DatasetFactory.get(Dataset.TUSIMPLE, only_available=False, split="train", args=args,
                                        shuffle=True, augment=True)
        # avoid ugly debug with data loader
        image, grid_tensor, fileinfo, _, params = dataset.__getitem__(0)
        dataset.params_per_file.update({fileinfo: params})

        import numpy as np
        image = np.ones((args.cell_size[0] * 100, args.cell_size[1] * 100, 3), dtype=float) * 255

        name_total = args.paths.generate_debug_image_file_path(file_name=fileinfo[0], idx=ImageIdx.ANCHOR,
                                                               suffix="_".join(
                                                                   [str(a) for a in args.anchor_vars]) + "_total")
        for a in range(len(dataset.anchors.bins)):
            name = args.paths.generate_debug_image_file_path(file_name=fileinfo[0], idx=ImageIdx.ANCHOR,
                                                             suffix="_".join(
                                                                 [str(a) for a in args.anchor_vars]) + "_" + str(a))
            line = dataset.anchors.bins[[a]]
            if linerep == LINE.MID_DIR:
                line = MidDirLines.to_cart(line[0]).unsqueeze(0)
            draw_cell(line * 32, image=None, coords=dataset.anchors.anchor_coords,
                      cell_size=args.cell_size, name=name if args.plot else "", standalone_scale=100, draw_label=True,
                      labels=[str(dataset.anchors.bins[a].numpy())], thickness=4)
            draw_cell(line * 32, image=image, coords=dataset.anchors.anchor_coords,
                      cell_size=args.cell_size, name=name_total if args.plot else "", standalone_scale=100,
                      draw_label=True, labels=[str(dataset.anchors.bins[a].numpy())], thickness=4)

        plt.clf()
        for c_idx, cell in enumerate(grid_tensor):
            for i, predictor in enumerate(cell):
                if torch.isnan(predictor[0]):
                    continue

                if linerep == LINE.MID_DIR:
                    plot_predictor = MidDirLines.to_cart(
                        predictor[dataset.coords.get_position_of(Variables.GEOMETRY)])
                else:
                    plot_predictor = predictor
                y_diff = plot_predictor[3] - plot_predictor[1]
                x_diff = plot_predictor[2] - plot_predictor[0]
                from yolino.viz.plot import get_color
                import numpy as np
                from yolino.utils.enums import ColorStyle
                color = get_color(ColorStyle.ANCHOR, anchors=dataset.anchors, idx=i, bgr=False)

                geom_pos = [0, 1, 2, 3]
                norms = torch.linalg.norm(predictor[geom_pos] - dataset.anchors.bins, axis=1)
                anchor_min = torch.min(norms)
                anchor_min_idx = torch.where(
                    torch.logical_and(norms >= anchor_min - 0.0001, norms <= anchor_min + 0.0001))[0]
                self.assertIn(i, anchor_min_idx,
                              "The predictor %s should be at the anchor position %s\n%s" % (i, anchor_min_idx, norms))

                plt.arrow(plot_predictor[1], plot_predictor[0], dx=y_diff, dy=x_diff,
                          label="%s at %d" % (str(predictor), i),
                          color=np.divide(color, 255.), head_width=0.05, length_includes_head=True)

                plt.gca().invert_yaxis()
                plt.title("Actual lines matched to %s anchors with %s" % (args.num_predictors, args.anchor_vars))
            path = "/tmp/anchors_in_img.png"
            from yolino.utils.logger import Log
            if args.plot:
                Log.warning("Save to file://%s" % path)
                plt.savefig(path)
                plt.clf()

    def test_viz_anchors(self):
        args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                          additional_vals={"num_predictors": 8, "explicit": "clips/0313-2/36600/20.jpg",
                                           "anchors": str(AnchorDistribution.EQUAL),
                                           "anchor_vars": [AnchorVariables.DIRECTION],
                                           "linerep": str(LINE.MID_DIR)},
                          # level=Level.DEBUG
                          )
        anchors = Anchor.get(args, args.linerep)
        draw_angles(torch.atan2(anchors[:, 2], anchors[:, 3]), args.dataset, args.anchors, args.split)

    def test_point_viz(self):
        self.eval_viz_anchor_positions(LINE.POINTS, [AnchorVariables.POINTS])

    def test_md_viz(self):
        self.eval_viz_anchor_positions(LINE.MID_DIR, [AnchorVariables.MIDPOINT, AnchorVariables.DIRECTION], num_p=16)

    def test_md_midp_viz(self):
        self.eval_viz_anchor_positions(LINE.MID_DIR, [AnchorVariables.MIDPOINT])

    def test_md_dir_viz(self):
        self.eval_viz_anchor_positions(LINE.MID_DIR, [AnchorVariables.DIRECTION])

    def eval_viz_anchor_positions(self, linerep, anchor_vars, num_p=8):
        args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                          additional_vals={"num_predictors": num_p,
                                           "explicit": "clips/0313-2/36600/20.jpg",
                                           "anchors": str(AnchorDistribution.EQUAL),
                                           "anchor_vars": anchor_vars,
                                           "linerep": str(linerep)},
                          # level=Level.DEBUG
                          )
        coords = VariableStructure(line_representation_enum=args.linerep, num_conf=0,
                                   vars_to_train=[Variables.GEOMETRY])

        scale = 1000
        anchors = Anchor.get(args, args.linerep).bins

        from yolino.utils.logger import Log
        Log.warning("The generated anchors\n%s" % anchors)

        image = np.ones((1 * scale, 1 * scale, 3), dtype=float) * 255
        plot_positions = torch.stack([LineRepresentation.get(args.linerep).to_cart(a) * scale for a in anchors])

        draw_cell(cell=plot_positions, image=image,
                  coords=coords, labels=[["%.1f" % b for b in a.numpy()] for a in anchors], draw_label=True,
                  cell_size=[scale, scale], colorstyle=ColorStyle.ID, thickness=2)
        from yolino.utils.logger import Log
        path = "/tmp/anchor_%s_%s.png" % (linerep, "_".join([str(a) for a in anchor_vars]))
        if args.plot:
            Log.warning("Save to file://%s" % path)
            cv2.imwrite(path, image)

    def test_anchor_offset(self):

        args_offset = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                                 additional_vals={"num_predictors": 24,
                                                  "explicit": "clips/0313-2/36600/20.jpg",
                                                  "anchors": str(AnchorDistribution.EQUAL),
                                                  "anchor_vars": [AnchorVariables.MIDPOINT, AnchorVariables.DIRECTION],
                                                  "linerep": str(LINE.MID_DIR),
                                                  "activations": [ACTIVATION.LINEAR, ACTIVATION.SIGMOID],
                                                  "offset": True},
                                 # level=Level.DEBUG
                                 )
        args_absolute = copy(args_offset)
        args_absolute.offset = False

        # Get data
        dataset_offset, _ = DatasetFactory.get(Dataset.TUSIMPLE, only_available=False, split="train", args=args_offset,
                                               shuffle=False, augment=False)
        dataset_absolute, _ = DatasetFactory.get(Dataset.TUSIMPLE, only_available=False, split="train",
                                                 args=args_absolute,
                                                 shuffle=False, augment=False)

        image, grid_tensor_offset, fileinfo, _, params = dataset_offset.__getitem__(0)
        dataset_offset.params_per_file.update({fileinfo: params})

        _, grid_tensor_absolute, fileinfo_abs, _, params = dataset_absolute.__getitem__(0)
        dataset_absolute.params_per_file.update({fileinfo: params})

        geom_pos = dataset_offset.coords.get_position_of(Variables.GEOMETRY)

        for a_idx in range(len(dataset_offset.anchors)):
            self.assertTrue(torch.equal(dataset_offset.anchors[a_idx], dataset_absolute.anchors[a_idx]),
                            "Anchors should be equal not matter the learning strategy, but we have"
                            "\nOffset\n%s\nAbsolute\n%s\n---" % (dataset_offset.anchors[a_idx],
                                                                 dataset_absolute.anchors[a_idx]))
        # assert is correct offset
        for c_idx in range(len(grid_tensor_absolute)):
            for p_idx, predictor_absolute in enumerate(grid_tensor_absolute[c_idx]):
                if torch.all(predictor_absolute[geom_pos].isnan()):
                    offset_position = torch.where(~grid_tensor_offset[c_idx, :, 0].isnan())[0].numpy()
                    absolute_position = torch.where(~grid_tensor_absolute[c_idx, :, 0].isnan())[0].numpy()
                    self.assertTrue(torch.all(grid_tensor_offset[c_idx, p_idx, geom_pos].isnan()),
                                    "ID=%s/%s: We expect the offset to be nan, when the absolute is nan, "
                                    "but in cell %s the absolute predictor\n%s\nis at position %s "
                                    "and the offset predictor is at position %s"
                                    % (c_idx, p_idx, c_idx, grid_tensor_absolute[c_idx, absolute_position],
                                       absolute_position, offset_position))
                    continue

                offset_position = torch.where(~grid_tensor_offset[c_idx, :, 0].isnan())[0].numpy()
                absolute_position = torch.where(~grid_tensor_absolute[c_idx, :, 0].isnan())[0].numpy()
                self.assertTrue(torch.all(~grid_tensor_offset[c_idx, p_idx, geom_pos].isnan()),
                                "ID=%s/%s: We expect the offset not to be nan, "
                                "when the absolute is not nan, but in cell %s the absolute predictor\n%s\n"
                                "is at position %s and the offset predictor is at position %s"
                                % (c_idx, p_idx, c_idx, grid_tensor_absolute[c_idx, absolute_position],
                                   absolute_position, offset_position))

                self.assertTrue(torch.all(torch.less_equal(
                    dataset_offset.anchors[p_idx] + grid_tensor_offset[c_idx, p_idx, geom_pos]
                    - predictor_absolute[geom_pos],
                    0.0001)),
                    "ID=%s/%s\nThe offset\t\t\t%s does not fit the"
                    "\nabsolute position\t%s and"
                    "\nthe anchor\t\t\t%s with"
                    "\nthe sum\t\t\t\t%s"
                    % (c_idx, p_idx, grid_tensor_offset[c_idx, p_idx, geom_pos],
                       predictor_absolute[geom_pos],
                       dataset_offset.anchors[p_idx],
                       dataset_offset.anchors[p_idx] + grid_tensor_offset[c_idx, p_idx, geom_pos]))

    def test_viz_offset(self):

        args_offset = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                                 additional_vals={"num_predictors": 8,
                                                  "explicit": "clips/0313-2/36600/20.jpg",
                                                  "anchors": str(AnchorDistribution.EQUAL),
                                                  "anchor_vars": [AnchorVariables.POINTS],
                                                  "linerep": str(LINE.POINTS),
                                                  "activations": [ACTIVATION.LINEAR, ACTIVATION.SIGMOID],
                                                  "offset": True},
                                 # level=Level.DEBUG
                                 )
        args_absolute = copy(args_offset)
        args_absolute.offset = False

        # Get data
        dataset_offset, _ = DatasetFactory.get(Dataset.TUSIMPLE, only_available=False, split="train", args=args_offset,
                                               shuffle=False, augment=False)
        dataset_absolute, _ = DatasetFactory.get(Dataset.TUSIMPLE, only_available=False, split="train",
                                                 args=args_absolute, shuffle=False, augment=False)

        image, grid_tensor_offset, fileinfo, _, params = dataset_offset.__getitem__(0)
        dataset_offset.params_per_file.update({fileinfo: params})

        _, grid_tensor_absolute, fileinfo_abs, _, params = dataset_absolute.__getitem__(0)
        dataset_absolute.params_per_file.update({fileinfo: params})

        # Get Grid
        grid_offset, _ = GridFactory.get(grid_tensor_offset.unsqueeze(0), [], coordinate=CoordinateSystem.CELL_SPLIT,
                                         args=args_offset, input_coords=dataset_offset.coords,
                                         anchors=dataset_offset.anchors)
        grid_absolute, _ = GridFactory.get(grid_tensor_absolute.unsqueeze(0), [],
                                           coordinate=CoordinateSystem.CELL_SPLIT,
                                           args=args_absolute, input_coords=dataset_absolute.coords,
                                           anchors=dataset_absolute.anchors)

        point_coords = dataset_offset.coords.clone(LINE.POINTS)
        offset_raw = grid_offset.numpy(point_coords, init=-1)
        equal = np.less_equal(np.abs(grid_absolute.numpy(point_coords, init=-1) - offset_raw),
                              0.0001)
        where = np.where(~equal)
        where = [[where[0][a], where[1][a], where[2][a], where[3][a]] for a in range(len(where[0]))]

        self.assertTrue(np.all(equal), "The grid is not correct. "
                                       "The offset grid is deviating from the absolute one in %s/%s positions."
                                       "\n%s..." % (
                            np.sum(~equal), np.sum(~np.isnan(grid_offset.numpy(point_coords))), where[0:10]))

        # Plot
        img_offset, ok_offset = plot(grid_offset.get_image_lines(image_height=image.shape[1],
                                                                 coords=dataset_offset.coords),
                                     name="/tmp/offset.png", image=image, coords=dataset_offset.coords,
                                     colorstyle=ColorStyle.UNIFORM,
                                     color=(100, 100, 100), training_vars_only=False, anchors=dataset_offset.anchors,
                                     coordinates=CoordinateSystem.UV_SPLIT)

        img_absolute, ok_absolute = plot(grid_absolute.get_image_lines(image_height=image.shape[1],
                                                                       coords=dataset_absolute.coords),
                                         name="/tmp/absolute.png", image=image, coords=dataset_absolute.coords,
                                         colorstyle=ColorStyle.UNIFORM,
                                         color=(100, 100, 100), training_vars_only=False,
                                         anchors=dataset_absolute.anchors,
                                         coordinates=CoordinateSystem.UV_SPLIT)

        self.assertEqual(ok_offset, ok_absolute)
        self.assertGreaterEqual(ok_absolute, 0)
        self.assertTrue(np.all(np.equal(img_offset, img_absolute)),
                        np.where(~np.equal(img_offset, img_absolute)))


if __name__ == '__main__':
    unittest.main()
