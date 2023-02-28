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

import PIL
import cv2
import numpy as np


def generate_argparse(name):
    parser = argparse.ArgumentParser(name)

    parser.add_argument("--height", type=int, required=True,
                        help="Output height of the video.")
    # parser.add_argument("--width", type=int, required=True,
    #                     help="Output width of the video")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the output video")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input folder containing the images")
    parser.add_argument("-p", "--pattern", type=str, required=True,
                        help="Pattern to identify the files. Will only check the end of the file, so provide e.g. "
                             "nms.png if your images are named 1_nms.png")
    parser.add_argument("-x", "--execute", action="store_true",
                        help="Open the video file afterwards")
    return parser.parse_args()


def genVideo(path, videoFile, end_pattern, height):

    images = []
    if not os.path.isdir(path):
        print("Invalid path: " + str(path))
        return
    for dirpath, dirnames, files in os.walk(path):
        for f in files:
            if f.endswith(end_pattern):
                filename = os.path.join(dirpath, f)
                images.append(filename)
    if len(images) == 0:
        print("No images found in " + str(path))
        return
    images.sort()

    # cap = cv2.VideoCapture(videoFile)

    # cvImg = cv2.imread(images[0])
    pil_image = PIL.Image.open(images[0]).convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    cvImg = open_cv_image[:, :, ::-1].copy()

    size = cvImg.shape[:2]
    width = int(height * size[1] / size[0])
    cvImg = cv2.resize(cvImg, (height, width))
    size = cvImg.shape[:2]
    print("use %s as size" % str(size))
#    fourcc = cv2.VideoWriter_fourcc(*'h264')
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    # fourcc = cv2.VideoWriter_fourcc(*"x264")
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

#    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoOut = cv2.VideoWriter(videoFile, fourcc, 15.0, (size[1], size[0]))  # 4096 2048
    print("Press ENTER to start video generation")
    input()

    i = 0
    for image in images:
        print("Process " + str(image))
        # cvImg = cv2.imread(images[0])
        pil_image = PIL.Image.open(images[0]).convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        cvImg = open_cv_image[:, :, ::-1].copy()

        if cvImg is None:
            print("Could not process %s" % image)
            continue
        if cvImg.shape[:2] != size:
            cvImg = cv2.resize(cvImg, size[:2])

        videoOut.write(cvImg)
        i += 1

    videoOut.release()

    print("Put video into " + str(videoFile))


if __name__ == '__main__':
    args = generate_argparse("Videos")
    # genVideo(path=args.input, videoFile=args.output + str('.mp4'), end_pattern=args.pattern, height=args.height)

    for root, dirs, files in os.walk(args.input):
        if len(dirs) == 0:
            id = os.path.split(root)[-1]
            output_path = "%s/%s_%s.mp4" % (args.output, id,
                                            args.pattern.replace("_", "").replace(".", "").replace("png", "").replace(
                                                "jpg", ""))
            ok = os.system("ffmpeg -framerate 10 " +
                           "-i %s/" % root + "%00d" + "%s " % args.pattern +
                           "-vcodec libx264 " +
                           "-vf scale=-1:%s " % args.height +
                           output_path + " -y > /dev/null 2>&1")
                
            if ok == 0: 
                print("vlc %s" % (output_path))
            elif ok == 256: 
                print("No data found for %s" % id)
            else:
                print("Unknown error %s for %s" % (ok, id))

    if args.execute:
        os.system("xdg-open %s" % args.output)
