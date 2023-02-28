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
import os.path

import numpy as np
import seaborn as sns

#
# COLOR = {
#     "darkgreen": _Color_(0, 176, 109),
#     "green": _Color_(0, 108, 66),
#     "lightgreen": _Color_(115, 154, 131),
#     "brightgreen": _Color_(0, 180, 111),
#     "happygreen": _Color_(146, 208, 80),
#     "superlightgreen": _Color_(191, 205, 196),
#     "red": _Color_(255, 0, 71),
#     "blue": _Color_(0, 100, 163),
#     "lightblue": _Color_(136, 195, 230),
#     "yellow": _Color_(250, 187, 8)}
from yolino.utils.logger import Log


class COLOR:

    class _Color_:
        def __init__(self, r, g, b):
            self.r = float(r)
            self.g = float(g)
            self.b = float(b)

        def value(self):
            return self.r, self.g, self.b

        def numpy(self):
            return np.asarray(self.value())

        def plt(self):
            return np.asarray(self.value()) / 255.

        def __str__(self) -> str:
            return str(self.value)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __getitem__(self, item):
        return list(self.__dict__.items())[item % len(self)][1]

    def __len__(self):
        return len(self.__dict__)

    def __init__(self, filename="/home/meyer/00_documents/01_projects/diss/latex/res/colors.tex", root=""):
        if not os.path.exists(filename):
            new_filename = os.path.join(root, "res/colors.tex")
            Log.error(f"Latex colors not found at {filename}. We use local copy at {new_filename}")
            filename = new_filename

        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line == "\n":
                    break
                splits = line.splitlines()[0].replace("}", "").split("{")
                vals = splits[3].split(",")
                setattr(self, splits[1].replace("fzi", "").replace("-", "").replace("}", ""), COLOR._Color_(*vals))

    def get_palette(self):
        return sns.color_palette([c.plt() for c in self.__dict__.values()])
