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
from unittest import TestCase

import numpy as np

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.grid.cell import Cell
from yolino.grid.grid import Grid
from yolino.grid.grid_factory import GridFactory
from yolino.grid.predictor import Predictor
from yolino.model.variable_structure import VariableStructure
from yolino.utils.enums import CoordinateSystem, ImageIdx, Dataset, Variables, LINE
from yolino.utils.test_utils import test_setup
from yolino.viz.plot import plot_cell_class, plot


class TestPlot(TestCase):
    def setUp(self) -> None:
        self.args = test_setup(self._testMethodName + "_setup", str(Dataset.CULANE))

    def test_plot_cell_class(self):

        output_path = os.path.join("tmp", "debug_class_image.png")
        if os.path.isfile(output_path):
            os.remove(output_path)

        grid, _ = GridFactory.get(data=[], variables=[], coordinate=CoordinateSystem.EMPTY, args=self.args,
                                  input_coords={Variables.CLASS: 3})

        cells = [
            {"pos": [5, 5], "color": [255, 255, 255]},
            {"pos": [self.args.grid_shape[0] - 1, self.args.grid_shape[1] - 1], "color": [26., 228., 182.]},
            {"pos": [6, 6], "color": [250., 186., 57.]}]

        p = Predictor([0, 0, 1, 1], label=[1, 0, 0], confidence=1, linerep=LINE.POINTS)
        data = Cell(cells[0]["pos"][0], cells[0]["pos"][1], num_predictors=1, predictors=[p])
        grid.insert(data)

        p = Predictor([0, 0, 1, 1], label=[0, 1, 0], confidence=1, linerep=LINE.POINTS)
        data = Cell(cells[1]["pos"][0], cells[1]["pos"][1], num_predictors=1, predictors=[p])
        grid.insert(data)

        p = Predictor([0, 0, 1, 1], label=[0, 0, 1], confidence=1, linerep=LINE.POINTS)
        data = Cell(cells[2]["pos"][0], cells[2]["pos"][1], num_predictors=1, predictors=[p])
        grid.insert(data)

        image = np.ones((3, self.args.img_size[0], self.args.img_size[1]), dtype=np.float32)
        img, ok = plot_cell_class(grid, output_path, image, epoch=0, tag="unittest",
                                  imageidx=ImageIdx.DEFAULT, ignore_classes=[0], max_class=3, fill=True,
                                  threshold=self.args.confidence)

        self.assertTrue(os.path.isfile(output_path), "could not find %s" % output_path)

        for cell in cells:
            r, c = cell["pos"]
            cell_color = img[int((r + .5) * self.args.cell_size[0]), int((c + .5) * self.args.cell_size[1])]
            self.assertTrue(np.all(np.equal(cell_color, cell["color"])), "Cell at %d, %d should be %s but is %s" %
                            (cell["pos"][0], cell["pos"][1], str(cell["color"]), str(cell_color)))

    def test_dml(self):
        args = test_setup(self._testMethodName, str(Dataset.TUSIMPLE),
                          additional_vals={"linerep": str(LINE.MID_DIR), "ignore_missing": True}
                          # level=Level.DEBUG
                          )
        coords = DatasetFactory.get_coords(split=args.split, args=args)
        grid = Grid(img_height=args.img_height, args=args)

        p = Predictor([0, 0, 1, 1])
        grid.append(0, 0, predictor=p)
        p = Predictor([1, 0, 0.2, 0.8])
        grid.append(0, 0, predictor=p)
        values = [0.8, 0.4, 0.9, 0.2]
        p = Predictor(values)
        grid.append(0, 0, predictor=p)

        t = grid.tensor(convert_to_lrep=LINE.MID_DIR, coords=coords)

        x_dif = values[2] - values[0]
        y_dif = values[3] - values[1]
        expected_md = [x_dif / 2 + values[0], y_dif / 2 + values[1], x_dif, y_dif]

        for i in range(len(expected_md)):
            self.assertAlmostEqual(t[0, 2, i].item(), expected_md[i], places=5,
                                   msg="Grid calculated %s, but we expected %s" % (t[0, 2], expected_md))

        gt_grid, _ = GridFactory.get(data=t.unsqueeze(0), variables=[],
                                     coordinate=CoordinateSystem.CELL_SPLIT, args=args,
                                     input_coords=coords, only_train_vars=False)
        img_lines = gt_grid.get_image_lines(coords=coords, image_height=320)

        path = "/tmp/test_dm_plot.png"
        plot(lines=img_lines, name=path, image=None, coords=coords, show_grid=True,
             cell_size=args.cell_size, coordinates=CoordinateSystem.UV_SPLIT)

        self.assertEqual(list(img_lines[0, 2, 0:2]), [round(values[0] * 32), round(values[1] * 32)])

    def test_plot_confidence(self):
        random = np.random.random(size=(100, 4)) * 1000
        zeros = np.zeros((100, 1))
        stack = np.concatenate([random, zeros], axis=1).reshape((1, 100, 5))
        img, _ = plot(lines=stack, name="", image=np.zeros((1000, 1000, 3)), cell_size=[10, 10],
                      coords=VariableStructure(LINE.POINTS), coordinates=CoordinateSystem.UV_SPLIT, threshold=0.5)
        self.assertEqual(np.sum(img), 0)
