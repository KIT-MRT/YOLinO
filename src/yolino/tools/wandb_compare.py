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

import yaml
from yolino.utils.logger import Log
from yolino.viz.translation import experiment_param_keys


def run():
    import wandb
    api = wandb.Api()
    prefix = "argo_po_8p_dn19"

    run_names = ["default", "oth7kag4", "7p637hyk", "qw0nc08x", "lam9onr9", "od4ik720", "fcxc8nu9"]
    if len(run_names) <= 1:
        raise ValueError("We need at least 2 runs!")
    run_configs = {}

    default_path = "default_params.yaml"
    for run_name in run_names:
        if run_name == "default":
            with open(default_path, "r") as f:
                run_configs[run_name] = yaml.safe_load(f)
        else:
            try:
                run_configs[run_name] = api.run(os.path.join(prefix, run_name)).config
            except wandb.errors.CommError as ex:
                run_configs[run_name] = {}
                Log.error(ex.message)
                continue

    template_value = None
    justi = 30

    END = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'

    print(BOLD + " ".ljust(justi) + " ".join([r.ljust(justi) for r in run_names]) + END)
    for key in experiment_param_keys:
        ss = str(key).ljust(justi) + "\t"

        for r, run_name in enumerate(run_names):

            configs = run_configs[run_name]
            if key not in configs:
                ss += "-".ljust(justi) + "\t"
                continue
            test_value = configs[key]

            if type(test_value) == list:
                portion = int(justi / len(test_value))
                joined = ",".join([str(v)[0:portion] for v in test_value])
                test_value = joined
            test_value = str(test_value)[0:justi]

            if template_value is None:
                template_value = test_value
                ss += str(template_value).ljust(justi) + "\t"
                continue

            if template_value == str(test_value):
                ss += GREEN + str(test_value).ljust(justi) + END + "\t"
            else:
                ss += RED + str(test_value).ljust(justi) + END + "\t"

        print(ss)
        template_value = None


if __name__ == '__main__':
    run()
