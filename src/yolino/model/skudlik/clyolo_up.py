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
import torch
import torch.nn as nn

from yolino.model.skudlik.darknet import Darknet
from yolino.utils.logger import Log


class ClYoloUp(nn.Module):
    cell_size = [16, 16]
    specs = {"cell_size": cell_size}

    def __init__(self, args, coords):
        super(ClYoloUp, self).__init__()

        self.darknet = Darknet(args.darknet_cfg, return_intermediate=True)
        if args.darknet_weights:
            self.darknet.load_weights(args.darknet_weights)

        self.num_coords = coords.get_length(one_hot=True)
        self.output_channels = args.num_predictors * self.num_coords

        self.upsample = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            nn.ConvTranspose2d(1024, 1024, 3, 2, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.postup = nn.Sequential(
            nn.Conv2d(1536, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1536, 1024, 1, 1, 0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.yolo = nn.Conv2d(1024, self.output_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        x2, x1, x = self.darknet(x)
        # x = self.darknet2(x1)
        x = self.upsample(x)
        x = torch.cat((x, x1), 1)
        x = self.postup(x)
        x = self.yolo(x)
        x = self.reshape_prediction(x)
        return x

    def reshape_prediction(self, pred):
        batch_size = pred.shape[0]
        pred = pred.view(batch_size, self.output_channels, -1)
        pred = pred.permute(0, 2, 1)
        batch_size, cell_count, out_channels = pred.shape
        pred = pred.view(batch_size, cell_count, -1, self.num_coords)
        return pred


def get_test_input(shape, batch_size):
    return torch.rand(batch_size, shape[2], shape[0], shape[1])


if __name__ == "__main__":
    from yolino.utils.general_setup import general_setup

    args = general_setup(name="YOLOv1 1x up", config_file="darknet_params.yaml", setup_logging=False,
                         ignore_cmd_args=True,
                         alternative_args={"model": "clyoloup"})

    model = ClYoloUp(num_predictors=args.num_predictors, num_coords=5, darknet_cfg=args.darknet_cfg,
                     darknet_weights=args.darknet_weights)
    inp = get_test_input((320, 640, 3), args.batch_size)
    pred = model(inp)  # torch.Size([8, 800, 8, 5]) => 20x40
    Log.debug(pred.shape)
