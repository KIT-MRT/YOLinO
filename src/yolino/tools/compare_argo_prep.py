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
from tqdm import tqdm
from yolino.tools.prepare_argoverse2 import define_arguments
from yolino.utils.enums import TaskType
from yolino.utils.general_setup import __push_commit__
from yolino.utils.general_setup import __set_hash__, __set_cmd_logger__, set_progress_logger, __set_paths__
from yolino.utils.logger import Log

if __name__ == '__main__':

    name = "compare npy"
    parser = define_arguments(name)
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Provide path to generated dataset")
    args, unknown = parser.parse_known_args()

    args.log_dir = "argo2_prep"
    args.resume_log = False
    args.split = "none"
    args.explicit_model = "none"
    args.debug_tool_name = name.replace(" ", "-").lower()

    params = __push_commit__(args.root)
    args.yolino = params["yolino"]
    if params["yolino"]["branch"] != "master":
        if args.tags is None:
            args.tags = []
        args.tags.append(params["yolino"]["branch"][0:10])
    Log.debug("----- LOGGER -----")
    __set_hash__(args)
    __set_cmd_logger__(args, name)
    set_progress_logger(args, TaskType.TRAIN)
    args = __set_paths__(args)

    if len(args.input) < 2:
        raise ValueError("We need at least two input folders")
    Log.warning(f"We take {args.input[0]} as iterator.")

    for folder in args.input:
        if not os.path.exists(folder):
            Log.error("Nothing to prepare for %s. Path not found. " % folder)
            continue

    for r, dirs, files in tqdm(os.walk(args.input[0]), total=2607):
        for f in files:
            if f.endswith(".npy"):
                for i in range(1, len(args.input)):
                    absfile = os.path.join(r, f)
                    relfile = os.path.relpath(absfile, args.input[0])
                    checkfile = os.path.join(args.input[i], relfile)

                    if not os.path.exists(checkfile):
                        Log.error(f"Could not validate {checkfile}")
                        Log.warning("---------------------------")
                        continue

                    abs_size = os.stat(absfile).st_size / (1024)
                    check_size = os.stat(checkfile).st_size / (1024)

                    abs_shape = np.load(absfile).shape
                    check_shape = np.load(checkfile).shape

                    ljust = max(len(absfile), len(checkfile))

                    if abs_shape[0] != check_shape[0]:
                        Log.warning(
                            f"{absfile}".ljust(ljust) + "\t\t" + f"{int(abs_size)} KB" + "\t\t" + f"{abs_shape}")
                        Log.warning(
                            f"{checkfile}".ljust(ljust) + "\t\t" + f"{int(check_size)} KB" + "\t\t" + f"{check_shape}")
                        Log.error("Shape is different!")
                        Log.warning("---------------------------")
                        continue
