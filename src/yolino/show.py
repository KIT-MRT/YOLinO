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

from yolino.dataset.dataset_factory import DatasetFactory
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Log


def run():
    args = general_setup("Show", setup_logging=False, show_params=False)
    args.plot = True
    args.max_n = 100
    args.augment = ""
    dataset, loader = DatasetFactory.get(args.dataset, only_available=True, split=args.split, args=args, shuffle=False,
                                         augment=args.augment, show=False)

    for data in iter(loader):
        try:
            _, _, filenames, _, params = data
            dataset.params_per_file.update({f: {k: v[i] for k,v in params.items()} for i, f in enumerate(filenames)})
        except (Exception, BaseException) as e:
            Log.error("Error with file %s" % (str(filenames)))
            raise e

    if args.explicit:
        for path in args.explicit:
            os.system("eog %s.png" % os.path.join(args.paths.debug_folder, path))
    else:
        os.system("xdg-open %s" % args.paths.debug_folder)


if __name__ == '__main__':
    run()

