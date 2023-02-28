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
import unittest

from yolino.utils.enums import Dataset, Logger
from yolino.utils.logger import Log
from yolino.utils.test_utils import test_setup


class SetupTest(unittest.TestCase):
    def test_file_logger(self):
        args = test_setup(self._testMethodName, dataset=str(Dataset.CULANE),
                          additional_vals={"loggers": str(Logger.FILE)})

        from datetime import datetime
        now = datetime.now()
        Log.debug("hallo")
        Log.warning("123")

        self.assertTrue(os.path.exists(Log.log_file_path))

        with open(Log.log_file_path, "r") as f:
            lines = f.readlines()

        self.assertGreaterEqual(len(lines), 2)

        # DEBUG
        self.assertIn(self._testMethodName, lines[-2])
        self.assertIn("test_setup.py:36", lines[-2])
        self.assertIn("DEBUG", lines[-2])
        self.assertEqual("hallo", lines[-2][-10:-5])

        date_time = now.strftime("%Y-%m-%d %H:%M")
        self.assertEqual(date_time, lines[-2][24:40])

        # WARN
        self.assertIn(self._testMethodName, lines[-2])
        self.assertIn("test_setup.py:37", lines[-1])
        self.assertIn("WARN", lines[-1])
        self.assertEqual("123", lines[-1][-8:-5])

        date_time = now.strftime("%Y-%m-%d %H:%M")
        self.assertEqual(date_time, lines[-1][24:40])

        Log.__file_log__ = None


if __name__ == '__main__':
    unittest.main()
