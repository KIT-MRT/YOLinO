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
# TODO: could be also implemented as self-check classes

from yolino.utils.enums import CoordinateSystem


def validate_input_structure(data, coordinate: CoordinateSystem, args=None):
    import numpy as np
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if coordinate == CoordinateSystem.UV_CONTINUOUS:
            if len(data) > 1:
                raise ValueError("Please provide one batch at a time to the grid. We've got %s" % str(data.shape))

            for batch in data:
                for instance in batch:
                    if len(np.shape(instance)) != 2:
                        raise AttributeError("%s, but have (?,?, %s)" % (str(coordinate), str(np.shape(instance))))

                    if np.shape(instance)[1] < 2:
                        raise AttributeError("%s, but have (?,?, %s)" % (str(coordinate), str(np.shape(instance))))

        elif coordinate == CoordinateSystem.UV_SPLIT:
            if np.shape(data)[0] > 1:
                raise ValueError("Please provide one batch at a time to the grid. We've got %s" % str(data.shape))

            if len(np.shape(data)) != 3:
                raise AttributeError("%s, but have %s" % (str(coordinate), str(np.shape(data))))

            if args is not None and np.shape(data)[0] != args.batch_size:
                raise AttributeError(
                    "%s, but have %s with batch=%d" % (str(coordinate), str(data.shape), args.batch_size))

        elif coordinate == CoordinateSystem.CELL_SPLIT:
            if len(np.shape(data)) != 4:
                raise AttributeError("%s, but have %s" % (str(coordinate), str(data.shape)))

            if args is not None and np.shape(data)[2] > args.num_predictors:
                raise AttributeError(
                    "%s, but have %s with batch=%d" % (str(coordinate), str(data.shape), args.batch_size))

        elif coordinate == CoordinateSystem.EMPTY:
            return True
        else:
            raise NotImplementedError

    return True
