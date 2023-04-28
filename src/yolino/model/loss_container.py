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

import numpy as np
import torch
from yolino.utils.logger import Log


class LossContainer:
    def __init__(self, training_variables, num_iterations, loss_weights):
        self.variables = training_variables
        self._means_ = torch.ones((num_iterations, len(training_variables)), dtype=torch.float) * torch.nan

        # this can be sum or mean depending on --loss; conf is also weighted!
        self._configs_ = torch.ones((num_iterations, len(training_variables)), dtype=torch.float) * torch.nan

        # this is the weighted loss as specified in args and put into the net
        self._backprops_ = torch.ones((num_iterations, 1), dtype=torch.float) * torch.nan
        self.weights = loss_weights

        self.current_epoch = None

    def add_backprop(self, epoch, loss, i):
        if epoch is None:
            raise ValueError("Epoch is none")

        if self.current_epoch is not None and self.current_epoch > epoch:
            Log.error(f"We cannot add loss for epoch {epoch}. We already log {self.current_epoch}.")
            return

        if type(loss) == torch.Tensor:
            loss = loss.detach().cpu().item()

        self._backprops_[i] = loss

    def add(self, epoch, mean_losses, config_losses, i):
        if self.current_epoch is not None and self.current_epoch > epoch:
            Log.error(f"We cannot add loss for epoch {epoch}. We already log {self.current_epoch}.")
            return

        if type(mean_losses) == torch.Tensor:
            mean_losses = mean_losses.detach().cpu()

        if type(config_losses) == torch.Tensor:
            config_losses = config_losses.detach().cpu()

        self.current_epoch = epoch
        self._means_[i] = torch.tensor(mean_losses)
        self._configs_[i] = torch.tensor(config_losses)  # this can be sum or mean depending on --loss

    def backprop(self, epoch):
        if self.current_epoch is not None and epoch != self.current_epoch:
            Log.error(f"We do not have data for epoch {epoch}")
            return None
        return torch.sum(self._backprops_)

    def mean(self, epoch, i=None, variable=None):
        if self.current_epoch is not None and epoch != self.current_epoch:
            Log.error(f"We do not have data for epoch {epoch}")
            return None

        if variable is not None:
            i = np.where(variable == self.variables)[0]
        if i is not None:
            data = self._means_[:, i]
        else:
            data = self._means_
        return torch.nanmean(data).item()

    def sum(self, epoch, i=None, variable=None):
        if self.current_epoch is not None and epoch != self.current_epoch:
            Log.error(f"We do not have data for epoch {epoch}")
            return None

        if variable is not None:
            i = np.where(variable == self.variables)[0]
        if i is not None:
            data = self._configs_[:, i]
        else:
            data = self._configs_
        return torch.nansum(data).item()

    def log(self, tag, epoch):
        data = {
            os.path.join("loss", "mean"): self.mean(epoch),
            os.path.join("loss", "sum"): self.sum(epoch),
            os.path.join("loss", "backprop"): self.backprop(epoch),
            "epoch": epoch
        }

        for i, t in enumerate(self.variables):
            data[os.path.join(f"loss_{str(t)}", "mean")] = self.mean(epoch, i)
            data[os.path.join(f"loss_{str(t)}", "sum")] = self.sum(epoch, i)

        Log.scalars(tag=tag, epoch=self.current_epoch, dict=data)
        Log.print('%s sum losses (sum over images): %s' % (tag.capitalize(), self.sum(epoch)), level=1)
        Log.print('%s mean losses (mean over images): %s' % (tag.capitalize(), self.mean(epoch)), level=1)
        Log.print("\n", level=1)
