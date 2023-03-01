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
from argparse import Namespace
from time import sleep

import cv2
import numpy as np
import torch
from tqdm import tqdm

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.utils import enums
from yolino.utils.general_setup import general_setup

path = os.path.join(os.environ["DATASET_ARGO2_IMG"], "sensor", "train")
args = general_setup(name="Mean / Std")
ds, loader = DatasetFactory.get(enums.Dataset.ARGOVERSE2, only_available=True, split=args.split, args=args,
                                shuffle=False, augment=False)

means = []
psum = 0
psum_sq = 0
print("Look into %s" % path)


def get_them(psum, psum_sq):
    num_pixels = len(ds) * args.img_size[0] * args.img_size[1]
    mean = psum / num_pixels
    var = (psum_sq / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)
    print(f"mu={mean.numpy().tolist()},s={std.numpy().tolist()},s2={var.numpy().tolist()}\r")


for data in loader:
    img, _, _, _, _ = data
    psum += img.sum(axis=[0, 2, 3])
    psum_sq += (img ** 2).sum(axis=[0, 2, 3])
    get_them(psum, psum_sq)

print("----- TOTAL -----")
get_them(psum, psum_sq)

