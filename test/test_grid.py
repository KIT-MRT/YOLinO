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
import unittest

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

from yolino.grid.grid_factory import GridFactory
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import CoordinateSystem, Dataset, Variables, LINE
from yolino.utils.general_setup import general_setup
from yolino.utils.system import get_system_specs
from yolino.utils.test_utils import test_setup, get_default_param_path
from yolino.viz.plot import ColorStyle, get_color, plot_style_grid

if sys.version_info.major == 3:
    from yolino.grid.cell import Cell
    from yolino.grid.predictor import Predictor
    from yolino.viz.plot import plot
else:
    print("Version is %s" % sys.version_info)
from yolino.utils.logger import Level, Log


@unittest.skipIf(sys.version_info.major < 3, "not supported in python2")
class GridTest(unittest.TestCase):
    params_file = "tmp_params.yaml"
    dump_path = "tmp"

    params = {}
    params["rotation_range"] = 0.1
    params["img_height"] = 128
    params["model"] = "yolo_class"
    params["linerep"] = str(LINE.POINTS)
    params["num_predictors"] = 8
    params["learning_rate"] = 0.001
    params["split"] = "train"
    params["root"] = ".."

    user = get_system_specs()["user"]
    params["level"] = "WARN"
    params["max_n"] = 100
    params["epoch"] = 10

    def dumpdata(self, dataset, log_dir=None):
        if log_dir is None:
            log_dir = dataset

        tmp = GridTest.params
        tmp["dataset"] = dataset
        tmp["log_dir"] = log_dir + "_po_8p_dn19"
        tmp["dvc"] = "tmp"

        with open(GridTest.params_file, "w") as f:
            yaml.dump(tmp, f)

    def setUp(self):
        Log.setup_cmd(viz_level=Level.INFO, name=self._testMethodName + "_setup")

        self.dumpdata("culane")
        self.args = general_setup(self._testMethodName, GridTest.params_file, ignore_cmd_args=True, setup_logging=False,
                                  task_type=None, default_config=get_default_param_path())

        if not os.path.exists(GridTest.dump_path):
            os.makedirs(GridTest.dump_path)

        # ---- Image 
        self.image = torch.ones((3, self.args.img_size[0], self.args.img_size[1]), dtype=torch.float32)  # as BGR

        # ---- GRID
        rows = self.args.grid_shape[0]
        cols = self.args.grid_shape[1]

        data = np.ones([1, rows * cols, self.args.num_predictors, 7]) * [0, 0, 1, 1, 0, 1, 1]  # class + conf

        # data should be (batch, cells, <=predictors, 2 * 2 + ?)
        self.coords = VariableStructure(num_classes=2, num_conf=1, line_representation_enum=self.args.linerep)
        self.grid, _ = GridFactory.get(data, [], CoordinateSystem.CELL_SPLIT, self.args,
                                       input_coords=self.coords)

        # Points: x-y coords starting at top left corner; x down, y sideways

    # Assert that the expected grid shape meets the actual grid. Further assert that the row and col ids of a cell
    # are correct and the values are set correctly for all cells.
    def testGridInit(self):
        self.assertTrue((np.shape(self.grid.cells) == np.asarray(self.args.grid_shape)).all())
        for r_idx, row in enumerate(self.grid.cells):
            for c_idx, cell in enumerate(row):
                self.assertEqual(cell.row, r_idx)
                self.assertEqual(cell.col, c_idx)

                for predictor in cell:
                    self.assertEqual(predictor.label[0], 0)
                    self.assertEqual(predictor.label[1], 1)
                    self.assertEqual(predictor.start.tolist(), [0., 0.])
                    self.assertEqual(predictor.end.tolist(), [1., 1.])
                    self.assertEqual(predictor.confidence, 1)
                    self.assertEqual(predictor.linerep, LINE.POINTS)

    # Check cell insertion.
    def testGridModify(self):
        row_id = int(self.args.grid_shape[0] / 2)
        col_id = int(self.args.grid_shape[1] / 3)

        pred = [0.7, 0.7, 0.3, 0.3]
        data = [Predictor(pred[0:4], label=[1, 0]), Predictor(pred[0:4], label=[1, 0])]
        data = Cell(row_id, col_id, self.args.num_predictors, predictors=data)
        self.grid.insert(data, overwrite=True)

        for r_idx, row in enumerate(self.grid.cells):
            for c_idx, cell in enumerate(row):
                for predictor in cell:
                    if r_idx == row_id and c_idx == col_id:
                        self.assertEqual(predictor.label[0], 1)
                        self.assertEqual(predictor.label[1], 0)
                        self.assertEqual(predictor.start.tolist(), pred[0:2])
                        self.assertEqual(predictor.end.tolist(), pred[2:4])
                        self.assertEqual(predictor.confidence, -1)
                        self.assertEqual(predictor.linerep, LINE.POINTS)
                    else:
                        self.assertEqual(predictor.label[0], 0)
                        self.assertEqual(predictor.label[1], 1)
                        self.assertEqual(predictor.start.tolist(), [0, 0])
                        self.assertEqual(predictor.end.tolist(), [1, 1])
                        self.assertEqual(predictor.confidence, 1)
                        self.assertEqual(predictor.linerep, LINE.POINTS)

    # Check cell insertion non-overwrite mode.
    def testGridModifyNoOverwrite(self):
        row_id = int(self.args.grid_shape[0] / 2)
        col_id = int(self.args.grid_shape[1] / 3)

        pred = [0.2, 0.2, 0.3, 0.3]
        data = [Predictor(pred), Predictor(pred)]
        data = Cell(row_id, col_id, self.args.num_predictors, predictors=data)
        self.grid.insert(data, overwrite=False)

        for r_idx, row in enumerate(self.grid.cells):
            for c_idx, cell in enumerate(row):
                for predictor in cell:
                    self.assertEqual(predictor.label[0], 0)
                    self.assertEqual(predictor.label[1], 1)
                    self.assertEqual(predictor.start.tolist(), [0, 0])
                    self.assertEqual(predictor.end.tolist(), [1, 1])
                    self.assertEqual(predictor.confidence, 1)
                    self.assertEqual(predictor.linerep, LINE.POINTS)

    # Dashed pattern on half the image, full stripes on the other half. Check pixels are correctly drawn.
    def testGridImage(self):
        for i in range(0, int(np.prod(self.args.grid_shape) / 2.)):
            # first half of the image should have dashed stripes, the others should be continuous
            self.grid.insert(Cell(i % self.grid.shape[0], int(i / self.grid.shape[0]), self.grid.num_predictors,
                                  [Predictor(values=[0, 0, 0.5, 0.5], label=[1, 0], confidence=1)]), overwrite=True)

        # (batch, line_segments, 2 * 2 + ?)
        uv_lines = self.grid.get_image_lines(coords=self.coords, image_height=self.image.shape[1])
        cell_size = np.asarray(self.grid.get_cell_size(image_height=self.image.shape[1]))
        center_cell = np.floor(0.5 * cell_size)
        self.assertTrue((uv_lines[0][0] == [0, 0, center_cell[0], center_cell[1], 1, 0, 1]).all(),
                        msg="%s should be [0,0,%s,%s,1,1,1]" % (uv_lines[0][0], 0.5 * cell_size[0], 0.5 * cell_size[1]))

        plot_style_grid(uv_lines, os.path.join("tmp", "debug_grid_image_with_grid.png"), self.image,
                        show_grid=True, cell_size=self.grid.get_cell_size(), coords=self.coords)

        img, ok = plot(uv_lines, os.path.join("tmp", "debug_grid_image.png"), self.image, colorstyle=ColorStyle.UNIFORM,
                       show_grid=False, coords=self.coords)
        npimg = np.array(img)
        print(npimg.shape)

        uniform_color = get_color(ColorStyle.UNIFORM)  # should be (138, 246,  63)
        three_quarter_cell = np.floor(0.75 * cell_size).astype(int)

        for i in range(0, int(np.prod(self.args.grid_shape) / 2.)):
            r = i % self.grid.shape[0]
            c = int(i / self.grid.shape[0])
            # first half of cell has line
            self.assertTrue((npimg[r * cell_size[0], c * cell_size[1]] == uniform_color).all(),
                            "We expect the first half of the image to have %s, but for cell %d, "
                            "%d we have at pixel %d, %d color %s" %
                            (str(uniform_color), r, c, r * cell_size[0], c * cell_size[1],
                             str(npimg[r * cell_size[0], c * cell_size[1]])))

            # snd half of cell has no line
            self.assertTrue((npimg[r * cell_size[0] + three_quarter_cell[0], c * cell_size[1] + three_quarter_cell[
                1]] != uniform_color).all(),
                            npimg[r * cell_size[0] + three_quarter_cell[0], c * cell_size[1] + three_quarter_cell[1]])

        for i in range(int(np.prod(self.args.grid_shape) / 2.), np.prod(self.args.grid_shape)):
            r = i % self.grid.shape[0]
            c = int(i / self.grid.shape[0])
            # first half of cell has line
            self.assertTrue((npimg[r * cell_size[0], c * cell_size[1]] == uniform_color).all())

            # snd half of cell also has line
            self.assertTrue((npimg[r * cell_size[0] + three_quarter_cell[0], c * cell_size[1] + three_quarter_cell[
                1]] == uniform_color).all())

    # check line points are sampled along diagonale and have angle 45°
    def testGetPoints(self):
        points = self.grid.get_image_points(coords=self.coords, sample_distance=1, image_height=self.image.shape[1])

        uv_lines = self.grid.get_image_lines(coords=self.coords, image_height=self.image.shape[1])
        plot_style_grid(lines=uv_lines, name=os.path.join(GridTest.dump_path, "dummy_grid_points.png"),
                        image=self.image, show_grid=False,
                        coords=self.coords)

        cell_size = self.grid.get_cell_size(image_height=self.image.shape[1])
        for point in points:
            self.assertAlmostEqual(point[0] % cell_size[0], point[1] % cell_size[1], places=5,
                                   msg=point)  # Should be on diagonale inside grid cell
            self.assertAlmostEqual(point[2], 45, point)  # the angle

    # Test position is correctly calculated for a line including struck at border
    def testGridLinePosition(self):
        line = [[5, 0], [self.image.shape[1], self.image.shape[2]]]
        row_col, position_in_cell, portion_in_cell = self.grid.get_position_of_line_segment(line)

        self.assertTrue(np.all(row_col == [[0, 0], [self.args.grid_shape[0] - 1, self.args.grid_shape[1] - 1]]),
                        row_col)
        self.assertTrue(np.all(position_in_cell == [[5, 0], [self.args.cell_size[0], self.args.cell_size[1]]]),
                        position_in_cell)
        self.assertTrue(np.all(portion_in_cell == [[5 / self.args.cell_size[0], 0], [1, 1]]), portion_in_cell)

    # Test Slices along horizontal line
    def testGeneralSliceOfLabelsH(self):
        line = np.asarray([10., 0, 10, 2, 10, 30, 10, self.image.shape[1]])
        Log.info("Slice line %s along grid of shape %s" % (line, self.args.grid_shape))
        lines, _ = self.grid.slice_and_straighten_line(
            line, 0, 0, plot_image=False)

        lines = np.asarray(lines)
        self.assertTrue(lines is not None)

        self.assertTrue(np.all(np.equal(np.shape(lines), (1, 2, 2))), np.shape(lines))
        self.assertTrue(np.all(np.equal(line[0:2], lines[0][0][0:2])),
                        lines)  # start stays the same, end will be on the first grid border

        self.assertAlmostEqual(lines[0, 1, 0], 10, places=2, msg=str(lines))
        self.assertAlmostEqual(lines[0, 1, 1], self.args.cell_size[1], places=2, msg=str(lines))

    # Test Slices along vertical line
    def testGeneralSliceOfLabelsV(self):
        line = np.asarray([0., 10, 1, 10, 2, 10, self.image.shape[1], 10])
        Log.info("Slice line %s along grid of shape %s" % (line, self.args.grid_shape))

        lines, _ = self.grid.slice_and_straighten_line(line, 0, 0, plot_image=False)
        lines = np.asarray(lines)
        self.assertTrue(lines is not None)

        self.assertTrue(np.all(np.equal(np.shape(lines), (1, 2, 2))), np.shape(lines))
        self.assertTrue(np.all(np.equal(line[0:2], lines[0][0][0:2])), lines)

        self.assertAlmostEqual(lines[0, 1, 0], self.args.cell_size[0], places=2, msg=str(lines))
        self.assertAlmostEqual(lines[0, 1, 1], 10, places=2, msg=str(lines))

    # Test Slices along vertical line, just a point on the grid; should return empty
    def testGeneralSliceOfLabelsPoint(self):
        line = np.asarray([0., 10, 0, 10])
        Log.info("Slice line %s along grid of shape %s" % (line, self.args.grid_shape))

        with self.assertRaises(ValueError):
            lines, _ = self.grid.slice_and_straighten_line(line, 0, 0, plot_image=False)

    def testGeneralSliceOfLabelsDuplicatePoint(self):
        line = np.asarray([0., 10, 0, 10.5, 0, 14])
        Log.info("Slice line %s along grid of shape %s" % (line, self.args.grid_shape))
        lines, _ = self.grid.slice_and_straighten_line(line, 0, 0, plot_image=False)
        self.assertEqual(len(lines), 1, np.shape(lines))
        self.assertAlmostEqual(line[0], lines[0, 0, 0], places=2, msg=lines)
        self.assertAlmostEqual(line[1], lines[0, 0, 1], places=2, msg=lines)
        self.assertAlmostEqual(line[4], lines[0, 1, 0], places=2, msg=lines)
        self.assertAlmostEqual(line[5], lines[0, 1, 1], places=2, msg=lines)

    def testGeneralSliceOfLabelsSmallInsideCell(self):
        line = np.asarray([0., 10, 0, 12])
        Log.info("Slice line %s along grid of shape %s" % (line, self.args.grid_shape))
        lines, _ = self.grid.slice_and_straighten_line(
            line, 0, 0, plot_image=False)

        self.assertTrue(np.all(np.equal(np.shape(lines), (1, 2, 2))), np.shape(lines))

        self.assertTrue(np.all(np.equal(line[0:2], lines[0][0])), lines)
        self.assertTrue(np.all(np.equal(line[2:], lines[0][1])), lines)

    def testGeneralSliceOfLabelsSmallInsideCellReverse(self):
        line = np.asarray([0., 12, 0, 10])
        Log.info("Slice line %s along grid of shape %s" % (line, self.args.grid_shape))
        lines, _ = self.grid.slice_and_straighten_line(
            line, 0, 0, plot_image=False)

        self.assertTrue(np.all(np.equal(np.shape(lines), (1, 2, 2))), np.shape(lines))

        self.assertTrue(np.all(np.equal(line[0:2], lines[0][0])),
                        "FIXME! test is correct but slicing changes the order of the lines! %s" % lines)
        self.assertTrue(np.all(np.equal(line[2:], lines[0][1])), lines)

    def testGeneralSliceOfLabelsInsideCell(self):
        line = np.asarray([10., 10, 20, 20])
        Log.info("Slice line %s along grid of shape %s" % (line, self.args.grid_shape))
        lines, _ = self.grid.slice_and_straighten_line(line, 0, 0, plot_image=False)

        self.assertTrue(
            np.all(np.equal(np.shape(lines), (1, 2, 2))), np.shape(lines))

        self.assertTrue(np.all(np.equal(line[0:2], lines[0][0])), lines)
        self.assertTrue(np.all(np.equal(line[2:], lines[0][1])), lines)

    def testSliceInCShape(self):
        GridTest.params["img_height"] = 1280
        GridTest.params["anchors"] = "none"
        self.dumpdata("argo2")

        args = general_setup(self._testMethodName, GridTest.params_file, ignore_cmd_args=True, setup_logging=False,
                             default_config="tmp/default_params.yaml")
        # args.plot = True

        coords = VariableStructure(num_classes=0, num_conf=0, line_representation_enum=self.args.linerep)

        line = np.asarray(
            [[337.0, 528.0], [337.0, 529.0], [338.0, 530.0], [338.0, 531.0], [338.0, 532.0], [338.0, 533.0],
             [339.0, 534.0], [339.0, 535.0], [339.0, 535.0], [339.0, 536.0], [340.0, 537.0], [340.0, 538.0],
             [340.0, 539.0], [340.0, 540.0], [341.0, 541.0], [341.0, 541.0], [341.0, 542.0], [342.0, 543.0],
             [342.0, 544.0], [342.0, 544.0], [342.0, 545.0], [343.0, 546.0], [343.0, 546.0], [343.0, 547.0],
             [343.0, 548.0], [344.0, 549.0], [344.0, 549.0], [344.0, 550.0], [345.0, 551.0], [345.0, 552.0],
             [345.0, 552.0], [345.0, 553.0], [346.0, 554.0], [346.0, 555.0], [346.0, 556.0], [347.0, 557.0],
             [347.0, 558.0], [347.0, 559.0], [347.0, 560.0], [348.0, 561.0], [348.0, 562.0], [348.0, 563.0],
             [349.0, 563.0], [349.0, 565.0], [349.0, 566.0], [350.0, 567.0], [350.0, 568.0], [350.0, 569.0],
             [350.0, 570.0], [351.0, 571.0], [351.0, 571.0], [351.0, 571.0], [351.0, 571.0], [351.0, 572.0],
             [351.0, 572.0], [352.0, 572.0], [352.0, 573.0], [352.0, 573.0], [352.0, 574.0], [352.0, 574.0],
             [352.0, 574.0], [353.0, 575.0], [353.0, 575.0], [353.0, 575.0], [353.0, 576.0], [353.0, 576.0],
             [353.0, 576.0], [354.0, 577.0], [354.0, 577.0], [354.0, 578.0], [354.0, 578.0], [354.0, 578.0],
             [354.0, 579.0], [355.0, 579.0], [355.0, 579.0], [355.0, 579.0], [355.0, 580.0], [355.0, 580.0],
             [356.0, 580.0], [356.0, 580.0], [356.0, 580.0], [356.0, 580.0], [356.0, 580.0], [357.0, 580.0],
             [357.0, 580.0], [357.0, 580.0], [357.0, 580.0], [358.0, 580.0], [358.0, 580.0], [358.0, 580.0],
             [358.0, 580.0], [358.0, 580.0], [359.0, 580.0], [359.0, 580.0], [359.0, 580.0], [359.0, 580.0],
             [360.0, 580.0], [360.0, 581.0], [360.0, 581.0], [360.0, 581.0], [360.0, 581.0], [361.0, 580.0],
             [361.0, 580.0], [362.0, 579.0], [363.0, 578.0], [363.0, 577.0], [364.0, 576.0], [364.0, 575.0],
             [365.0, 574.0], [366.0, 573.0], [366.0, 572.0], [367.0, 571.0], [367.0, 570.0], [368.0, 569.0],
             [369.0, 568.0], [369.0, 566.0], [370.0, 565.0], [371.0, 564.0], [371.0, 562.0], [372.0, 561.0],
             [373.0, 560.0], [374.0, 558.0], [374.0, 557.0], [375.0, 556.0], [376.0, 554.0], [376.0, 553.0],
             [377.0, 551.0], [378.0, 550.0], [379.0, 548.0], [380.0, 547.0], [380.0, 545.0], [381.0, 544.0],
             [382.0, 542.0], [383.0, 540.0], [384.0, 539.0], [385.0, 537.0], [386.0, 535.0], [386.0, 534.0],
             [387.0, 532.0], [388.0, 530.0], [389.0, 528.0], [390.0, 526.0], [391.0, 525.0], [392.0, 523.0],
             [393.0, 521.0], [394.0, 519.0], [395.0, 517.0], [396.0, 515.0], [397.0, 513.0], [398.0, 511.0],
             ])

        if args.plot:
            plt.clf()
            plt.plot(line[:, 1], line[:, 0])
            plt.gca().invert_yaxis()
            plt.gca().yaxis.set_ticks(np.arange(int(min(line[:, 0]) / 32) * 32, max(line[:, 0]) + 32, 32))
            plt.gca().xaxis.set_ticks(np.arange(int(min(line[:, 1]) / 32) * 32, max(line[:, 1]) + 32, 32))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.savefig("/tmp/grid1.png")

        line = np.unique(line, axis=0)
        grid, errors = GridFactory.get(data=np.expand_dims(np.expand_dims(line, 0), 0), variables=[],
                                       coordinate=CoordinateSystem.UV_CONTINUOUS, args=args, input_coords=coords,
                                       variables_have_conf=False, plot_image=args.plot)

        if args.plot:
            t = grid.get_image_lines(coords=coords, image_height=args.img_height)[0]
            plt.clf()
            for l in t:
                color = np.asarray(get_color(colorstyle=ColorStyle.RANDOM)) / 255.
                plt.plot(l[[1, 3]], l[[0, 2]], color=color)
                plt.scatter(l[3], l[2], color=color, marker="*")
            plt.gca().invert_yaxis()
            plt.gca().yaxis.set_ticks(np.arange(int(min(line[:, 0]) / 32) * 32, max(line[:, 0]) + 32, 32))
            plt.gca().xaxis.set_ticks(np.arange(int(min(line[:, 1]) / 32) * 32, max(line[:, 1]) + 32, 32))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid()
            plt.savefig("/tmp/grid2.png")

        for topleft in np.asarray([[320, 512], [320, 544],
                                   [352, 512], [352, 576],
                                   [384, 480], [384, 512]]):
            r, c = (topleft / 32).astype(int)
            self.assertIsNotNone(grid[r, c], f"None in {r},{c} {topleft}")
            self.assertEqual(len(grid[r, c]), 1,
                             f"Cell {r},{c} has {grid[r, c]}, we expect one line segment there.")
        topleft = np.asarray([352, 544])
        r, c = (topleft / 32).astype(int)
        # print(f"{r},{c} {topleft}: {[str(g) for g in grid[r, c].predictors] if grid[r, c] is not None else 'none'}")
        self.assertIsNotNone(grid[r, c], f"None in {r},{c} {topleft}")
        self.assertEqual(len(grid[r, c]), 2, f"Cell {r},{c} has {grid[r, c]}, we expect one line segment there.")

        self.assertEqual(len(grid), 8)
        self.assertEqual(len(errors), 0)

    def testOneHot(self):

        self.args = test_setup(self._testMethodName, str(Dataset.CULANE))
        coords = VariableStructure(num_classes=3, num_conf=1, line_representation_enum=self.args.linerep)
        grid, _ = GridFactory.get(data=[], variables=[], coordinate=CoordinateSystem.EMPTY, args=self.args,
                                  input_coords=coords)

        # ATTENTION label as one hot!
        with self.assertRaises(ValueError):
            Predictor([0, 0, 1, 1], label=[2], confidence=1, linerep=LINE.POINTS)

        label = [0, 0, 1]
        p = Predictor([0, 0, 1, 1], label=label, confidence=1, linerep=LINE.POINTS)
        data = Cell(5, 5, num_predictors=1, predictors=[p])
        grid.insert(data)
        predictor = grid[5, 5].predictors[0]
        self.assertEqual(len(predictor.numpy(coords=coords)), coords.get_length(one_hot=True))
        self.assertTrue(np.all(np.equal(predictor.label, label)), "Expect labels to remain %s, but got %s"
                        % (label, predictor.label))
        index_label_predictor = predictor.tensor(coords=coords, one_hot=False)
        self.assertEqual(index_label_predictor[coords.get_position_of(Variables.CLASS, one_hot=False)],
                         np.argmax(label), "%s should not have one hot" % index_label_predictor)


if __name__ == '__main__':
    unittest.main()
