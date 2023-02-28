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

import torch
import torch.nn as nn

from yolino.model.skudlik.darknet import Darknet
from yolino.utils.enums import Variables, Network
from yolino.utils.logger import Log


class ClYolo(nn.Module):
    cell_size = [32, 32]
    specs = {"cell_size": cell_size}

    def __init__(self, args, coords):
        super(ClYolo, self).__init__()

        self.empty_geom = torch.tile(torch.tensor([0, 0, 1, 1], device=torch.device(args.cuda)),
                                     (args.batch_size, args.grid_shape[0] * args.grid_shape[1], args.num_predictors, 1))

        self.darknet = Darknet(args.darknet_cfg)
        if args.darknet_weights:
            self.darknet.load_weights(args.darknet_weights)

        self.num_classes = coords[Variables.CLASS]
        self.num_coords = coords.get_length(one_hot=True)
        self.output_channels = args.num_predictors * self.num_coords

        self.yolo = nn.Conv2d(1024, self.output_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.darknet(x)
        x = self.yolo(x)
        x = self.reshape_prediction(x)
        return x

    def reshape_prediction(self, pred):
        Log.debug("Reshape raw net output %s" % str(pred.shape))
        batch_size = pred.shape[0]
        # pred = pred.view(batch_size, self.output_channels, -1)
        pred = pred.view(batch_size, self.num_classes, -1, self.num_predictors)  # [2, 5, 918, 1]
        # pred = pred.permute(0, 2, 1)
        # batch_size, cell_count, out_channels = pred.shape
        # pred = pred.view(batch_size, cell_count, -1, self.num_coords)
        Log.debug("--> %s" % str(pred.shape))
        return pred


def get_test_input(shape, batch_size):
    return torch.rand(batch_size, shape[2], shape[0], shape[1])


if __name__ == "__main__":
    from yolino.utils.general_setup import general_setup
    from yolino.model.model_factory import get_model

    args = general_setup(name="YOLOv1", config_file="darknet_params.yaml", setup_logging=False, ignore_cmd_args=True,
                         alternative_args=["--model", Network.YOLO_CLASS])
    model = get_model(args, dataset=5)  # TODO should be a dataset!
    inp = get_test_input((args.img_size[0], args.img_size[1], 3),
                         args.batch_size)  # torch.Size([8, 200, 8, 5]) => 10x20
    # inp = get_test_input((640, 1280, 3), args.batch_size)  # torch.Size([8, 800, 8, 5]) => 20x40
    pred = model(inp)
    Log.debug(pred.shape)

    import numpy as np

    cells = np.prod(np.divide(args.img_size, ClYolo.cell_size))
    assert (np.all(np.equal(pred.shape, [args.batch_size, cells, 8, 5])))

    ratio = args.img_size[1] / args.img_size[0]
    smaller = math.sqrt(cells / ratio)
    assert (np.all(args.grid_shape == [smaller, ratio * smaller]))
