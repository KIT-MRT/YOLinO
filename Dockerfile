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
FROM python:3.8-bullseye
RUN apt update \
    && apt install git ffmpeg libsm6 libxext6 -y
RUN python -m pip install --upgrade pip
RUN pip install virtualenv

RUN mkdir /usr/bin/deps/
RUN mkdir /usr/bin/deps/src
COPY setup.cfg /usr/bin/deps/setup.cfg
RUN echo "from setuptools import setup\nsetup()" > /usr/bin/deps/setup.py
RUN pip install -e /usr/bin/deps

RUN export GIT_PYTHON_REFRESH=quiet
