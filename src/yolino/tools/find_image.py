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
import os

import PIL
import numpy as np
from PIL import ImageChops
from tqdm import tqdm

source = PIL.Image.open("/home/meyer/Downloads/Bild1.jpg")

min_diff = math.inf
diff_path = ""
for root, dirs, files in os.walk("/mrtstorage/datasets/public/tusimple_lane_detection/train_set/clips/0531"):
    # if not "ring_front_center" in root:
    #     continue
    print(root)
    for f in tqdm(files):
        # if not f.startswith("ring_front_center_"):
        #     continue
        if not f.endswith("20.jpg"):
            continue
        path = os.path.join(root, f)
        other = PIL.Image.open(path)
        other = other.resize(source.size)

        diff_image = ImageChops.difference(source, other)

        abs_diff = np.sum(np.abs(diff_image))
        if abs_diff < min_diff:
            tmp_diff = "/tmp/diff.png"
            diff_image.save(tmp_diff)
            min_diff = abs_diff
            diff_path = path

            print("The current closest image is file://%s with diff file://%s and diff=%s" % (diff_path, tmp_diff,
                                                                                              abs_diff))

print("The closest image is file://%s" % diff_path)


