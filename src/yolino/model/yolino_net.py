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

import torch
import torch.nn as nn
from yolino.model.darknet import Darknet
from yolino.utils.logger import Log


class YolinoNet(nn.Module):
    def __init__(self, args, coords):
        super(YolinoNet, self).__init__()

        self.cuda = args.cuda
        self.scale = args.scale
        self.darknet = Darknet(args.darknet_cfg, return_intermediate=True)
        if args.darknet_weights and os.path.isfile(args.darknet_weights):
            Log.info("Load weights from %s" % args.darknet_weights)
            self.darknet.load_weights(args.darknet_weights)

        self.coords = coords
        if len(self.coords.get_position_of_training_vars()) == 0:
            raise ValueError("Network is configured to predict 0 variables! Please fix %s, %s" % (self.coords,
                                                                                                  self.coords.vars_to_train))

        self.num_predictors = args.num_predictors

        in_channels = 1024
        if self.scale == 16 or self.scale == 8:
            self.upsample = nn.Sequential(
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

            in_channels = 1024
            if self.scale == 8:
                self.upsample2 = nn.Sequential(
                    nn.ConvTranspose2d(1024, 1024, 3, 2, 1, 1),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.1, inplace=True)
                )

                self.postup2 = nn.Sequential(
                    nn.Conv2d(1280, 1280, 3, 1, 1),
                    nn.BatchNorm2d(1280),
                    nn.LeakyReLU(0.1, inplace=True),

                    nn.Conv2d(1280, 512, 1, 1, 0),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.1, inplace=True),

                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.1, inplace=True)
                )
                in_channels = 512

        self.yolo = nn.Conv2d(in_channels=in_channels,
                              out_channels=self.num_predictors * len(self.coords.get_position_of_training_vars()),
                              kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        """

        Args:
            x (torch.tensor):
                with shape [batch, 3, height, width]
                dtype=torch.float32
                values in [0,1]

        Returns:
            torch.tensor:
                with shape [batch, cells, preds, vars]
        """
        x2, x1, x = self.darknet(x)

        if self.scale <= 16:
            x = self.upsample(x)
            x = torch.cat((x, x1), 1)  # here comes the skip; cat produces 1536 channels
            x = self.postup(x)

            if self.scale <= 8:
                x = self.upsample2(x)
                x = torch.cat((x, x2), 1)  # here comes the skip
                x = self.postup2(x)
                if self.scale <= 4:
                    raise NotImplementedError("We only implemented to upsampling layers. "
                                              "Use --scale with 32, 16 or 8.")
        x = self.yolo(x)  # output [batch, preds*vars, rows, cols]
        x = self.reshape_prediction(x)  # should output [batch, cells, preds, vars]
        return x

    def reshape_prediction(self, pred):
        """

        Args:
            pred (torch.tensor):
                with shape [batch, preds*vars, rows, cols]

        Returns:
            torch.tensor:
                with shape [batch, cells, preds, vars]
        """
        batch_size = pred.shape[0]
        pred = pred.permute(0, 2, 3, 1)
        pred = pred.reshape(batch_size, -1, self.num_predictors, self.coords.num_vars_to_train())
        return pred

    def receptive_field(self, input_size):
        """
        input_size: (channels, H, W)
        """
        from torch_receptive_field import receptive_field
        return receptive_field(self, input_size=input_size)


def get_test_input(shape, batch_size):
    return torch.rand(batch_size, shape[2], shape[0], shape[1])


def get_test_label(cells, batch_size):
    return torch.rand(batch_size, cells, 1, 1)
