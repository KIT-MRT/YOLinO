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
from yolino.utils.logger import Log


def getCuda(args):
    if args.gpu:
        if not torch.cuda.is_available() or not torch.cuda.device_count() > 0:
            Log.warning("CUDA is%s available and we have only %d gpus" % (
                "not " if torch.cuda.is_available() else "", torch.cuda.device_count()))
            args.gpu = False

    if args.gpu:
        CUDA = "cuda"
        Log.debug("Cuda is available: " + str(torch.cuda.is_available()))
        if "CUDA_AVAILABLE_DEVICES" in os.environ:
            Log.info("You provided CUDA_AVAILABLE_DEVICES=%s" % os.environ["CUDA_AVAILABLE_DEVICES"])
            if len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1:
                CUDA += ":"
                CUDA += str(os.environ["CUDA_VISIBLE_DEVICES"])
        elif args.gpu_id >= 0:
            CUDA += ":"
            CUDA += str(args.gpu_id)
        Log.warning("Let's use %s GPUs (%s)!" % (torch.cuda.device_count(), CUDA))
    else:
        CUDA = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        Log.warning("Let's use the CPU!")
    Log.info("We set %s" % CUDA)
    return CUDA


def getFreeGpus(num=-1, gpu_subset=None, free_limit=2000):
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    except ModuleNotFoundError as e:
        Log.error(e)
        return []

    nvmlInit()
    free_gpus = []

    if gpu_subset is None:
        gpu_subset = range(nvmlDeviceGetCount())

    for gpu in gpu_subset:
        h = nvmlDeviceGetHandleByIndex(int(gpu))
        info = nvmlDeviceGetMemoryInfo(h)
        if info.free > free_limit * 1e+6:
            free_gpus = free_gpus.append(str(gpu))
            Log.debug("GPU %s free" % (gpu))
        else:
            Log.debug("GPU %s: %s MB / %s MB allocated" % (gpu, info.used, info.total))

    return free_gpus[:num]
