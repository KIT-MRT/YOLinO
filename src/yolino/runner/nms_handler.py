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
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.postprocessing.nms import nms
from yolino.utils.enums import ImageIdx, ColorStyle, CoordinateSystem
from yolino.viz.plot import plot


class NmsHandler:
    def __init__(self, args, coords=None):
        self.do_plot = args.plot
        self.args = args
        if coords is None:
            self.coords = DatasetFactory.get_coords(args.split, args=args)
        else:
            self.coords = coords

    def __call__(self, images, preds, labels, filenames, epoch):

        np_preds_uv = preds.numpy()

        if self.do_plot:
            path = self.args.paths.generate_debug_image_file_path(file_name=filenames[0], idx=ImageIdx.PRED,
                                                                  suffix="nms")
            plot(lines=np_preds_uv, name=path, image=images[0], coords=self.coords,
                 show_grid=True, cell_size=self.args.cell_size, threshold=self.args.confidence,
                 colorstyle=ColorStyle.ORIENTATION, coordinates=CoordinateSystem.UV_SPLIT,
                 tag="nms", imageidx=ImageIdx.PRED, epoch=epoch, training_vars_only=True)

        lines, reduced = nms(np_preds_uv, grid_shape=self.args.grid_shape, cell_size=self.args.cell_size,
                             confidence_threshold=self.args.confidence, orientation_weight=self.args.nxw,
                             length_weight=self.args.lw,
                             midpoint_weight=self.args.mpxw, epsilon=self.args.eps, min_samples=self.args.min_samples)
        if self.do_plot:
            path = self.args.paths.generate_debug_image_file_path(file_name=filenames[0], idx=ImageIdx.NMS)
            plot(lines=lines, name=path, image=images[0], coords=self.coords,
                 show_grid=True, cell_size=self.args.cell_size, threshold=self.args.confidence,
                 colorstyle=ColorStyle.ORIENTATION, coordinates=CoordinateSystem.UV_SPLIT,
                 tag="nms", imageidx=ImageIdx.NMS, epoch=epoch, training_vars_only=True)

        return lines, reduced
