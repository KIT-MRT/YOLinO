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
from copy import deepcopy

import cv2
import numpy as np
from tqdm import tqdm

from yolino.postprocessing.nms import nms
from yolino.utils.enums import ColorStyle
from yolino.utils.logger import Log
from yolino.utils.paths import Paths


def parse_args():
    parser = argparse.ArgumentParser('Viz')
    # parser.add_argument("--pred_file_name", required=True, help="Prediction path")
    # parser.add_argument("--img_file_name", required=True, help="Image path")

    # test on ring_front_center_315973066019521256.jpg
    # suffix: _swap.pkl
    parser.add_argument('--max_n', type=int, default=-1, help='Runs for only max_n images')
    parser.add_argument("--root", default="../yolino/src/yolino",
                        help="Folder containing code and models weights")
    parser.add_argument("--dvc", default=".", help="Folder containing dvc management")
    # parser.add_argument("--enhance", action="store_true", help="Only add new files, nothing will be deleted.")
    parser.add_argument("--explicit", default="",
                        help="Provide an explicit filename, e.g. '000100' to process (no extension!). Will ignore enhance, max_n and every_n")
    parser.add_argument("--explicit_folder", default="",
                        help="Provide an explicit folder, e.g. '000100' to process. Could be a subset of the actual folder name (we use <explicit_folder> in <path>)")
    parser.add_argument("--split", default="val", help="Provide split folder name (train, val, test)")
    parser.add_argument("--bgr", action="store_true",
                        help="Convert image as bgr (might be useful for foreign trainings with bgr images)")
    parser.add_argument("-c", "--fixed_crop_range", default=[-1, -1, -1, -1], nargs=4, type=int,
                        help="Crop the images by a defined window: [left, upper, right, lower]")
    parser.add_argument("--pred_suffix", required=True, type=str,
                        help="Provide suffix in filename incl filetype ending. Should be readable with np.load and pickle")
    parser.add_argument('--height', type=int, default=-1,
                        help='Specifiy height of the output image; this will overwrite the value from params.yaml')
    parser.add_argument('--style', type=ColorStyle, default=ColorStyle.ORIENTATION,
                        help='Specifiy the colorstyle mapping as one of %s' % list(
                            ColorStyle._value2member_map_.keys()))
    parser.add_argument("--color_bar", action="store_true", help="Show color bar on the side")

    args = parser.parse_args()
    return args


def do(args):
    old_img_size = args.img_size
    paths = Paths(args.dvc, args.root, args.dataroot)

    label_path = os.path.join(args.dataroot, args.split, "labels")
    image_path = os.path.join(args.dataroot, args.split, "images")
    image_file_list = getPresplitFiles(label_path, image_path, args.split, args.explicit, args.explicit_folder,
                                       args.max_n)

    # crop = np.asarray([0,448,0,960])
    crops = [
        # np.asarray([150,448,0,960]),
        # np.asarray([int(150 / 16) * 16, 448, int(400 / 16) * 16, int(500 / 16) * 16]),
        # np.asarray([0,897,0,1919])
        np.asarray([0])
    ]

    for crop in crops:

        crop_folder_name = "_".join(crop.astype(str))
        main_folder = os.path.join(paths.prediction, "matplotlib", crop_folder_name, str(args.style))
        Log.warning("Main folder is %s" % main_folder)

        for filename in tqdm(sorted(image_file_list)):
            file_stub = os.path.splitext(filename)[0]
            full_file_path = os.path.join(image_path, filename)

            prediction_file_path = paths.generate_prediction_grid_file_path(file_name=file_stub, swap=True)
            pred = np.load(prediction_file_path, allow_pickle=True)
            image = cv2.imread(full_file_path)
            if np.all(np.asarray(args.fixed_crop_range) >= 0):
                image = image[args.fixed_crop_range[1]:args.fixed_crop_range[3],
                        args.fixed_crop_range[0]:args.fixed_crop_range[2]]

            if args.height > 0:
                args.img_size = (args.height, int(args.height / image.shape[0] * image.shape[1]))

            scale = args.img_size[0] / old_img_size[0]

            # print("Prediction: %s" % np.shape(pred))
            # print("Image: %s" % str(np.shape(image)))
            lines = pred.get_image_lines(coords=, image_height=image.shape[0])

            # factor = 1.5 * scale 

            # image = cv2.resize(image, np.array(image.shape[1::-1]) * scale)
            empty_image = np.ones_like(image) * 255
            # lines[0,:,0:4] *= factor 

            # nxw = 0.05
            # mpxw = 0.06
            # lw=0.00016
            # min_samples=2

            # crop *= scale 
            crop_folder_name = "_".join(crop.astype(str))

            main_folder = os.path.join(paths.prediction, "matplotlib", crop_folder_name, str(args.style))
            os.makedirs(os.path.join(main_folder, os.path.dirname(filename)), exist_ok=True)

            ####################### Plot Pre-NMS ###########################

            for do_img in [True, False]:
                for do_grid in [False, True]:
                    for t in [0, 0.9, 0.99]:  # 0,0.5,
                        name = os.path.join(main_folder, file_stub + "_t%s%s%s_prenms.jpg" % (
                            t, "_img" if do_img else "", "_grid" if do_grid else ""))
                        img, ok = plot(lines, "", image if do_img else empty_image, show_grid=do_grid, grid_shape=[
                            28, 60], colorstyle=args.style, class_idx=-2, conf_idx=-1, threshold=t,
                                       show_color_bar=args.color_bar)
                        if len(crop) == 4:
                            img = img[crop[0]:crop[1], crop[2]:crop[3]]

                        Log.warning("Write file to %s" % name)
                        cv2.imwrite(name, img)

            ####################### Plot NMS cluster ###########################

            for do_img in [True, False]:
                for do_grid in [True, False]:
                    for t in [0.9, 0.99]:  # 0,0.5,
                        nms_cluser_lines = deepcopy(lines)
                        nms_cluser_lines = nms(nms_cluser_lines, grid_rows=28, grid_cols=60, num_predictors=8, conf=t,
                                               nxw=args.nxw, lw=args.lw, mpxw=args.mpxw / scale, epsilon=0.02,
                                               min_samples=args.min_samples, classify_only=True)
                        img, ok = plot(nms_cluser_lines, "", image if do_img else empty_image, show_grid=do_grid,
                                       grid_shape=[
                                           28, 60], colorstyle=ColorStyle.CLASS, class_idx=-2, threshold=t,
                                       show_color_bar=args.color_bar)
                        if len(crop) == 4:
                            img = img[crop[0]:crop[1], crop[2]:crop[3]]

                        name = os.path.join(main_folder, file_stub + "_t%s%s%s_nmscluster.jpg" % (
                            t, "_img" if do_img else "", "_grid" if do_grid else ""))
                        Log.warning("Write file to %s" % name)
                        cv2.imwrite(name, img)

            ####################### Plot NMS result ###########################
            for do_img in [True, False]:

                for do_grid in [True, False]:
                    for t in [0.9, 0.99]:  # 0,0.5,

                        nms_lines = deepcopy(lines)
                        nms_lines = nms(nms_lines, grid_rows=28, grid_cols=60, num_predictors=8, conf=t, nxw=args.nxw,
                                        lw=args.lw, mpxw=args.mpxw / scale, epsilon=0.02, min_samples=args.min_samples,
                                        classify_only=False)

                        img, ok = plot(nms_lines, "", image if do_img else empty_image, show_grid=do_grid,
                                       grid_shape=[28, 60],
                                       colorstyle=args.style, class_idx=-2, conf_idx=-1, threshold=t,
                                       show_color_bar=args.color_bar)
                        if len(crop) == 4:
                            img = img[crop[0]:crop[1], crop[2]:crop[3]]

                        name = os.path.join(main_folder, file_stub + "_t%s%s%s_nms.jpg" % (
                            t, "_img" if do_img else "", "_grid" if do_grid else ""))

                        Log.warning("Write file to %s" % name)
                        cv2.imwrite(name, img)


if __name__ == '__main__':
    import coloredlogs

    os.environ["COLOREDLOGS_LOG_FORMAT"] = '%(filename)s:%(lineno)d\t %(message)s'
    os.environ[
        "COLOREDLOGS_LEVEL_STYLES"] = 'spam=22;debug=28;verbose=34;notice=220;warning=202;success=118,bold;error=124;critical=background=red'
    coloredlogs.install(level="WARN")

    args = parse_args()

    args = add_params_from_file(args)
    args = set_dataroot(args)
    args.batch_size = 1

    do(args)
