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

from yolino.utils.enums import Optimizer, Scheduler
from yolino.utils.logger import Log


def get_optimizer(args, net, loss_weights):
    Log.debug("Optimizer %s" % str(args.optimizer))
    params = ([p for p in net.parameters()] + [l for l in loss_weights if l.requires_grad])
    if args.optimizer == Optimizer.ADAM:
        if args.learning_rate != 0.001 or args.decay_rate != 0:
            Log.info("The optimizer by default uses lr=0.001 and weight_decay=0, you set lr=%f and weight_decay=%f."
                        % (args.learning_rate, args.decay_rate))
        optimizer = torch.optim.Adam(
            params,
            lr=args.learning_rate,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == Optimizer.SGD:
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == Optimizer.RMS_PROP:
        optimizer = torch.optim.RMSprop(params, lr=args.learning_rate,
                                        alpha=0.99, eps=1e-08, weight_decay=args.decay_rate,
                                        momentum=args.momentum, centered=False)
    elif args.optimizer == Optimizer.ADA_DELTA:
        optimizer = torch.optim.Adadelta(params, lr=10 * args.learning_rate, rho=0.9, eps=1e-06,
                                         weight_decay=args.decay_rate)
    else:
        raise NotImplementedError("Unknown optimizer %s" % args.optimizer)

    if args.scheduler == Scheduler.NONE:
        scheduler = None
    else:
        raise NotImplementedError("Unknown scheduler %s" % args.scheduler)

    return optimizer, scheduler
