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
import timeit

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.model.activations import get_activations
from yolino.model.model_factory import load_checkpoint
from yolino.model.variable_structure import VariableStructure
from yolino.utils.logger import Log

import torch

class ForwardRunner:
    def __init__(self, args, preloaded_model=None, start_epoch=-1, coords: VariableStructure = None,
                 load_best=False) -> None:
        self.args = args

        if coords is None:
            if preloaded_model is None:
                coords = DatasetFactory.get_coords("train", args)  # split does not matter
            else:
                coords = preloaded_model.coords

        if preloaded_model:
            self.model = preloaded_model
            self.start_epoch = start_epoch
        else:
            self.model, _, self.start_epoch = load_checkpoint(args, coords, allow_failure=False, load_best=load_best)

        if self.args.cuda not in str(next(self.model.parameters()).device):
            self.model = self.model.to(self.args.cuda)

        self.activations = get_activations(self.args.activations, coords, self.args.linerep)

    def __call__(self, images, is_train, epoch, first_run=False):
        """

        Args:
            images (torch.tensor):
                [batch, 3, height, width]

        Returns:
            torch.tensor:
                with shape [batch, cells, preds, vars]
        """
        if self.args.cuda not in str(images.device):
            Log.debug("Moved images from %s to %s" % (images.device, self.args.cuda))
            images = images.to(self.args.cuda)

        inference_start = timeit.default_timer()
        self.model = self.model.train(is_train)
        if is_train:
            logits = self.model(images)  # [batch, cells, preds, vars]
            outputs = self.activations(logits)  # [batch, cells, preds, vars]
        else:
            with torch.no_grad():
                logits = self.model(images)  # [batch, cells, preds, vars]
                outputs = self.activations(logits)  # [batch, cells, preds, vars]

        Log.time(key="raw_infer", value=timeit.default_timer() - inference_start, epoch=epoch)

        if first_run:
            Log.graph(self.model, images)
        return outputs  # [batch, cells, preds, vars]
