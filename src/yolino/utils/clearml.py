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
import argparse
import os
import socket
from datetime import datetime

import git
import numpy as np
from yolino.utils.clearml import Task
from yolino.utils.logger import Log


def setup(log_dir, dvc, config):
    kind = os.path.splitext(os.path.basename(argparse._sys.argv[0]))[0]

    if config is not None and "project_name" in config and "task_name" in config and not "kolumbos" in socket.gethostname():
        Log.debug("Use %s and %s for trains" % (config["project_name"], config["task_name"]))
        task = Task.init(project_name=config["project_name"],
                         task_name=config["task_name"],
                         task_type=Task.TaskTypes.testing)
    else:
        today = datetime.now()
        task = Task.init(project_name="YOLinO_" + log_dir[0:3],
                         task_name=log_dir + "_" + kind + "_" + today.strftime("%b-%d-%Y_%H-%M-%S"),
                         task_type=Task.TaskTypes.testing if "train" in kind else Task.TaskTypes.testing)

    try:
        repo = git.Repo(dvc)
        dvc_commit = str(repo.rev_parse("HEAD"))
    except git.exc.InvalidGitRepositoryError as e:
        Log.error("Your DVC is not a git folder %s; we cannnot store the commit hash" % e)
        dvc_commit = ""

    Log.debug("%s --> %s" % (task._project_name, task.task_id))
    splits = log_dir.split("_")
    task.add_tags(np.append([log_dir, "v5.0", kind], splits))
    task.connect({'dvc_commit': dvc_commit, "server": socket.gethostname()})

    logger = task.get_logger()
    return logger
