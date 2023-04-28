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
import math
import os
import random
import shutil
import threading
from pathlib import Path
from time import sleep
from typing import List, Union

import av2.utils.io as io_utils
import matplotlib
import numpy as np
import yaml
from av2.datasets.sensor.constants import RingCameras, StereoCameras
from rdp import rdp
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.utils.argparser import add_level, add_dataset, add_explicit, add_max_n, add_root, \
    add_ignore_missing, add_plot, add_tags, add_loggers, add_dvc, add_subsample, add_loading_workers, add_keep
from yolino.utils.enums import TaskType, ImageIdx
from yolino.utils.general_setup import __push_commit__
from yolino.utils.general_setup import __set_hash__, __set_cmd_logger__, set_progress_logger, __set_paths__
from yolino.utils.logger import Log
from yolino.utils.system import get_system_specs


def argparse(name):
    parser = define_arguments(name)
    parser.add_argument("-i", "--input", help="Provide path to original dataset", default=None)

    args, unknown = parser.parse_known_args()

    args = enrich_args(args, name, unknown)

    return args, parser


def enrich_args(args, name, unknown):
    args.log_dir = "argo2_prep"
    args.resume_log = False
    Log.info(f"Unknown args {unknown}")
    # args.subsample_dataset_rhythm = -1
    # Log.setup_cmd(viz_level=args.level, name="Argoverse 2.0")
    args.split = "none"
    args.explicit_model = "none"
    args.debug_tool_name = name.replace(" ", "-").lower()
    mrt_path = "/mrtstorage/datasets/public/argoverse20"
    if args.input is None:
        if "DATASET_ARGO2_IMG" in os.environ and os.path.exists(os.environ["DATASET_ARGO2_IMG"]):
            env_path = os.environ["DATASET_ARGO2_IMG"]
            Log.warning("You did not choose an input folder")
            answer = input(f"Do you want to collect the raw labels from DATASET_ARGO2_IMG={env_path}? [yN]")
            if answer == "y":
                args.input = env_path
                Log.warning(f"Please add '--input {args.input}' to your command.")
    if args.input is None:
        if os.path.exists(mrt_path):
            Log.warning("You did not choose an input folder")
            answer = input(f"Do you want to collect the raw labels from {mrt_path}? [yN]")
            if answer == "y":
                Log.warning(f"Please add '--input {mrt_path}' to your command.")
                args.input = mrt_path
    if args.input is None:
        Log.error(f"We do not know where your data is. We tried {mrt_path} and $DATASET_ARGO2_IMG. Please set --input.")
        exit(1)
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
    return args


def define_arguments(name):
    try:
        # Test if configargparse is available (not avail on unittests in CI)
        import configargparse
        parser = configargparse.ArgumentParser(name)
    except ModuleNotFoundError as ex:
        import socket
        host = socket.gethostname()
        user = pwd.getpwuid(os.getuid())[0]

        if "mrtbuild" != user:  # this should only happen in CI
            Log.error("%s with %s: No configargparse available" % (host, user))
            raise ex

        Log.warning("%s with %s: No configargparse available" % (host, user))
        CONFIG_AVAILABLE = False

        parser = argparse.ArgumentParser(name)

    try:
        from av2.datasets.sensor.constants import RingCameras
        from av2.datasets.sensor.constants import StereoCameras
        parser.add_argument("-cam", "--camera", help="Choose the camera image",
                            choices=[*list(RingCameras), *list(StereoCameras)], default=RingCameras.RING_FRONT_CENTER)
    except ModuleNotFoundError as ex:
        parser.add_argument("-cam", "--camera", help="Choose the camera image", default="ring_front_center")
    parser.add_argument("-y", "--yes", action="store_true", help="Just run and delete all old files without asking.")
    parser.add_argument("--dry_run", action="store_true", help="Do nothing for now. Only count images.")
    parser.add_argument("--enhance", action="store_true", help="Do not delete anything. Only add missing files.")
    add_dataset(parser)
    add_level(parser)
    add_max_n(parser)
    add_root(parser)
    add_explicit(parser)
    add_ignore_missing(parser)
    add_plot(parser)
    add_tags(parser)
    add_loggers(parser)
    add_dvc(parser)
    add_subsample(parser)
    add_loading_workers(parser)
    add_keep(parser)
    return parser


def outside_image(uv, img):
    return uv[1] > img.shape[0] or uv[0] > img.shape[1] or np.any(uv < 0)


def outside_image_np(uvs, img):
    return np.logical_or(np.logical_or(uvs[:, 1] > img.shape[0], uvs[:, 0] > img.shape[1]), np.any(uvs < 0, axis=1))


def get_lane(polyline, ego_SE3_city, pinhole_cam, depth_map, img):
    from av2.utils.typing import NDArrayInt, NDArrayFloat
    polyline_ego_frame = ego_SE3_city.transform_point_cloud(polyline)
    pinhole_cam.project_ego_to_img(polyline_ego_frame)
    # no need to motion compensate, since these are points originally from the city frame.
    uv, points_cam, is_valid_camera_point = pinhole_cam.project_ego_to_img(polyline_ego_frame)

    if not np.any(is_valid_camera_point):
        return [], False, False

    line_segments_arrays = []
    # occluded_arrays = []
    u: NDArrayInt = np.round(uv[:, 0]).astype(np.int32)
    v: NDArrayInt = np.round(uv[:, 1]).astype(np.int32)

    lane_z = points_cam[is_valid_camera_point, 2]
    if depth_map is not None:

        # ----------- adapted from av2 ------------
        MIN_DISTANCE_AWAY_M = 30.0  # assume max noise starting at this distance (meters)
        MAX_ALLOWED_NOISE_M = 15.0  # meters
        dists_away: NDArrayFloat = np.linalg.norm(points_cam[is_valid_camera_point], axis=1)  # type: ignore
        max_dist_away = dists_away.max()
        max_dist_away = max(max_dist_away, MIN_DISTANCE_AWAY_M)
        allowed_noise: NDArrayFloat = (dists_away / max_dist_away) * MAX_ALLOWED_NOISE_M
        # ----------- end av2 --------------
        is_not_occluded_in_valid_only = np.logical_or(lane_z - 0.01 <=
                                                      depth_map[v[is_valid_camera_point],
                                                                u[is_valid_camera_point]] + allowed_noise,
                                                      np.isnan(depth_map[v[is_valid_camera_point],
                                                                         u[is_valid_camera_point]]))
        is_not_occluded = np.asarray(
            [is_not_occluded_in_valid_only[sum(is_valid_camera_point[0:i])] if valid else False for i, valid in
             enumerate(is_valid_camera_point)])
    else:
        is_not_occluded = np.ones(lane_z.shape, dtype=bool)

    is_valid_not_occluded = np.logical_and(is_not_occluded, is_valid_camera_point)
    start_is_occluded = ~is_not_occluded[0]
    end_is_occluded = ~is_not_occluded[-1]

    occlusion_break_points = [0, *[o for o in range(1, len(is_valid_not_occluded))
                                   if is_valid_not_occluded[o] != is_valid_not_occluded[o - 1]],
                              len(is_valid_not_occluded)]
    is_block_occluded = ~is_valid_not_occluded[occlusion_break_points[:-1]]

    for occ_idx in range(len(occlusion_break_points) - 1):

        if is_block_occluded[occ_idx]:
            continue

        not_occluded_block_indices = np.arange(occlusion_break_points[occ_idx], occlusion_break_points[occ_idx + 1])
        line_segments_arr: NDArrayInt = np.hstack([u[not_occluded_block_indices].reshape(-1, 1),
                                                   v[not_occluded_block_indices].reshape(-1, 1)])

        if not_occluded_block_indices[0] > 0 and len(line_segments_arr) > 1:
            last_invalid_index = not_occluded_block_indices[0] - 1
            last_invalid_uv = np.asarray([u[last_invalid_index], v[last_invalid_index]])
            if outside_image(last_invalid_uv, img):
                line_segments_arr = np.concatenate([[last_invalid_uv], line_segments_arr])

        if not_occluded_block_indices[-1] < len(uv) - 1 and len(line_segments_arr) > 1:
            next_invalid_index = not_occluded_block_indices[-1] + 1
            next_invalid_uv = np.asarray([u[next_invalid_index], v[next_invalid_index]])
            if outside_image(next_invalid_uv, img):
                line_segments_arr = np.concatenate([line_segments_arr, [next_invalid_uv]])

        line_segments_arrays.append(line_segments_arr)

        if False:
            plt.clf()
            vis_depth_map(img, depth_map, False)
            plt.subplot(1, 2, 2)
            plt.scatter(u[is_valid_camera_point][~is_not_occluded_in_valid_only],
                        v[is_valid_camera_point][~is_not_occluded_in_valid_only],
                        c=np.reshape([255, 0, 0] * sum(~is_not_occluded), (-1, 3)) / 255., label="occluded")
            plt.scatter(u[is_valid_not_occluded], v[is_valid_not_occluded],
                        c=np.reshape([0, 255, 0] * sum(is_valid_not_occluded), (-1, 3)) / 255., label="visible")
            plt.legend()
            plt.savefig("/tmp/depth.png")
    return line_segments_arrays, start_is_occluded, end_is_occluded


def get_fresh_instance_id():
    global LAST_INSTANCE_ID
    LAST_INSTANCE_ID += 1

    COLORS[LAST_INSTANCE_ID] = np.asarray([np.random.randint(0, 255),
                                           np.random.randint(0, 255),
                                           np.random.randint(0, 255)]) \
                               / 255
    return LAST_INSTANCE_ID


import matplotlib.pyplot as plt


def add_centerline(id, instance_id, ids, avm, im_pose, city_SE3_ego, lanes, pinhole_cam, depth_map, img, axs,
                   num_valid_in_log=-1):
    if id in ids and len(ids[id]) > 0:
        Log.warning("DUPLICATE")
        return

    try:
        cl = avm.get_lane_segment_centerline(lane_segment_id=id)
    except KeyError as e:
        return

    distance_from_ego = np.linalg.norm(cl[:, 0:2] - im_pose, axis=-1)
    if np.any(distance_from_ego < max_dist):

        Log.debug("Process %s as instance %s" % (id, instance_id))
        ids[id].append(instance_id)

        import av2.geometry.interpolate as interp_utils
        cl = interp_utils.interp_arc(t=50, points=cl)

        try:
            distance_from_ego = np.linalg.norm(cl[:, 0:2] - im_pose, axis=-1)
            if distance_from_ego[0] >= max_dist:
                cl = cl[np.where(distance_from_ego < max_dist)[0][0]:]

            distance_from_ego = np.linalg.norm(cl[:, 0:2] - im_pose, axis=-1)
            if distance_from_ego[-1] >= max_dist:
                cl = cl[:np.where(distance_from_ego < max_dist)[-1][-1]]
        except IndexError as e:
            Log.error(e)
            Log.error(f"Centerline has distances {distance_from_ego}, but max_dist={max_dist}")

        uv_c, start_is_occluded, end_is_occluded = get_lane(cl, ego_SE3_city=city_SE3_ego.inverse(),
                                                            pinhole_cam=pinhole_cam,
                                                            depth_map=depth_map, img=img)
        if len(uv_c) == 0:
            Log.debug("%s has no valid uv points" % id)
        else:
            instance_id = process_uv_coords(id, instance_id, uv_c, lanes, axs, start_is_occluded, end_is_occluded,
                                            overall_i=num_valid_in_log)

    else:
        Log.debug("%s is too far away %f" % (id, min(distance_from_ego)))

    successors = avm.get_lane_segment_successor_ids(id)
    Log.debug("Process %d successors of id %s" % (len(successors), id))
    for s in range(len(successors)):
        if successors[s] in ids:
            Log.debug("Attention we found an successor that has been met: %s" % successors[s])
        else:
            if len(successors) > 1 and len(lanes[instance_id]) > 0:
                instance_id = get_fresh_instance_id()
                Log.debug("New instance id (%s) for successor" % instance_id)
                lanes[instance_id] = []
            ids[successors[s]] = []
            add_centerline(id=successors[s], instance_id=instance_id, num_valid_in_log=num_valid_in_log, ids=ids,
                           lanes=lanes, avm=avm, img=img,
                           im_pose=im_pose, city_SE3_ego=city_SE3_ego, pinhole_cam=pinhole_cam, axs=axs,
                           depth_map=depth_map)


def process_uv_coords(id, initial_instance_id, uv_cs, lanes, axs, start_is_occluded, end_is_occluded, only_viz=False,
                      marker=["*", "x", "+"], overall_i=-1):
    instance_id = initial_instance_id
    marker = marker[random.randint(0, len(marker) - 1)]
    label_is_written = False

    for idx, uv_c in enumerate(uv_cs):

        # between all iterations we want a new id
        if idx > 0 or start_is_occluded:
            instance_id = get_fresh_instance_id()
            lanes[instance_id] = []

        for i in range(len(uv_c) - 1):
            line = uv_c[i:i + 2]
            tmp_instance_id, label_is_written = process_line(id=id, instance_id=instance_id,
                                                             label_is_written=label_is_written, line=line,
                                                             marker=marker, only_viz=only_viz,
                                                             num_valid_in_log=overall_i,
                                                             lanes=lanes, axs=axs)
            if tmp_instance_id == -1:
                continue
            else:
                instance_id = tmp_instance_id

        last_i = len(uv_c) - 1
        if not only_viz:
            lanes[instance_id].append([uv_c[last_i, 1], uv_c[last_i, 0]])

    if end_is_occluded:
        instance_id = get_fresh_instance_id()
        lanes[instance_id] = []

    return instance_id


def process_line(id, instance_id, label_is_written, line, marker, only_viz, num_valid_in_log, lanes, axs):
    if not only_viz:
        lanes[instance_id].append([line[0, 1], line[0, 0]])
    if (args.plot and num_valid_in_log == 1):
        cmap = matplotlib.cm.get_cmap('tab20')
        axs[0].plot(line[:, 0], line[:, 1], color=cmap(id % 20), marker=marker,
                    label=str(id) if not label_is_written and max(line[:, 1]) > 1200 else "")
        if not label_is_written and max(line[:, 1]) > 1200:
            axs[0].text(line[0, 0], line[0, 1], str(id))
            label_is_written = True
    return instance_id, label_is_written


def abort(count, max_n):
    return count >= 0 and count >= max_n


def skip(i, subsample_dataset_rhythm):
    return subsample_dataset_rhythm > 0 and i % subsample_dataset_rhythm != 0


def handle_paths(args):
    dataset_output_path, _ = DatasetFactory.get_path(split=args.split, args=args)
    dataset_output_path = os.path.join(dataset_output_path, "sensor", args.split)

    if not args.dry_run:
        if os.path.exists(dataset_output_path):
            if not args.enhance:
                if not args.yes:
                    ok = input("We will delete everything in %s. Are you sure? [y/N]" % dataset_output_path)
                    if ok != "y":
                        args.enhance = True

                if not args.enhance:
                    shutil.rmtree(dataset_output_path)
                    os.makedirs(dataset_output_path)
        else:
            os.makedirs(dataset_output_path)
    dataset_input_path = os.path.join(args.input, "sensor", args.split)
    if not os.path.exists(dataset_input_path):
        Log.error("Nothing to prepare for %s. Path not found. " % dataset_input_path +
                  "Maybe you want to explicitly set --input?")
        exit(1)

    return dataset_output_path, dataset_input_path


file_checks = {
    "train": {
        "logs": 700
    },
    "test": {
        "logs": 150
    },
    "val": {
        "logs": 150
    }
}


def plt_add_image(img_fpath):
    import av2.utils.io as io_utils
    img = np.clip((
            np.repeat(
                np.expand_dims(
                    np.mean(
                        io_utils.read_img(img_fpath),
                        axis=2),
                    axis=-1), 3,
                axis=2) * 2),
        a_min=0, a_max=255).astype(int)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10 / img.shape[1] * img.shape[0]))
    axs[0].imshow(img)
    axs[1].imshow(img)
    return img, axs, fig


def finish_plot(axs, fig, img_height=None):
    axs[0].set_title("Map")
    axs[1].set_title("Numpy")
    for i in [0, 1]:
        start, end = axs[i].get_ylim()
        start = min(img_height * 1.1, start)
        axs[i].set_ylim(start, end)
        int_end = math.floor(end / 32) * 32
        int_start = int(start / 32) * 32
        major_ticks = np.arange(int_start, int_end + 40, 320)
        minor_ticks = np.arange(int_start, int_end + 40, 32)
        axs[i].yaxis.set_ticks(major_ticks)
        axs[i].yaxis.set_ticks(minor_ticks, minor=True)
        start, end = axs[i].get_xlim()
        int_start = int(start / 32) * 32
        int_end = math.floor(end / 32) * 32
        major_ticks = np.arange(int_start, int_end + 40, 320)
        minor_ticks = np.arange(int_start, int_end + 40, 32)
        axs[i].xaxis.set_ticks(major_ticks)
        axs[i].xaxis.set_ticks(minor_ticks, minor=True)
        axs[i].grid(which='minor', alpha=0.2)
        axs[i].grid(which='major', alpha=0.5)
        axs[i].legend(loc="upper right")

    fig.tight_layout()
    fig.gca().set_aspect('equal', adjustable='box')


def skip_log(log_id, explicit):
    if explicit is not None:
        folder_correct = np.any([log_id in e for e in explicit])

        if not folder_correct:
            return True
    return False


def process_log(log_ids: np.ndarray, thread_index, subsample_factor, max_n_in_log=-1):
    max_num_control_points = 0
    if len(log_ids) == 0:
        Log.error(f"Thread {thread_index} has nothing to do?")
        exit(1)

    from yolino.thirdparty.av2_sensor_dataloader import AV2SensorDataLoader
    import av2.geometry.interpolate as interp_utils
    from av2.map.lane_segment import LaneType
    from av2.map.map_api import ArgoverseStaticMap
    Log.info(f"Thread {thread_index} is running for {len(log_ids)} logs...")
    counter = 0
    for log_index, log_id in enumerate(log_ids):
        if abort(counter, args.max_n):
            Log.warning(f"Abort on counter {counter} and max_n={args.max_n}.")
            break

        if skip_log(log_id, args.explicit):
            continue

        if args.enhance:
            npy_folder_path = os.path.join(dataset_output_path, log_id, "sensors", "cameras", args.camera)
            img_folder_path = os.path.join(dataset_input_path, log_id, "sensors", "cameras", args.camera)
            if os.path.exists(npy_folder_path):
                num_npy_files = len([name for name in os.listdir(npy_folder_path) if
                                     os.path.isfile(os.path.join(npy_folder_path, name)) and
                                     os.path.splitext(name)[
                                         1] == ".npy"])
                num_img_files = len([name for name in os.listdir(img_folder_path) if
                                     os.path.isfile(os.path.join(img_folder_path, name)) and
                                     os.path.splitext(name)[
                                         1] == ".jpg"])
                if num_npy_files >= num_img_files:
                    Log.debug(
                        f"{log_id} already done. \nInput {img_folder_path} contains {num_img_files}.\nOutput {npy_folder_path} contains {num_npy_files}.")
                    # pbar.update(int(num_img_files / subsample_factor))
                    continue
                else:
                    Log.debug(
                        f"{log_id} is missing {num_img_files - num_npy_files} files... We continue here.")

        loader = AV2SensorDataLoader(data_dir=Path(dataset_input_path), labels_dir=Path(dataset_input_path),
                                     collect_single_log_id=log_id)

        cam_enum = cam_enums[0]
        cam_name = cam_enum.value
        try:
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
        except FileNotFoundError as e:
            if args.ignore_missing:
                continue
            else:
                raise e

        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)
        num_cam_imgs = len(cam_im_fpaths)

        if args.dry_run:
            del loader
            del pinhole_cam
            del cam_im_fpaths
            continue

        log_map_dirpath = os.path.join(dataset_input_path, log_id, "map")
        Log.debug("Load map from %s" % log_map_dirpath)
        avm = ArgoverseStaticMap.from_map_dir(Path(log_map_dirpath), build_raster=True)
        Log.debug("Loaded Map")

        num_valid_in_log = 0
        successfully_finished = 0
        for i, img_fpath in enumerate(cam_im_fpaths):

            if abort(i, max_n_in_log):
                Log.warning(f"Abort on counter {i} and thread max_n={max_n_in_log}.")
                break

            if skip(num_valid_in_log, args.subsample_dataset_rhythm):
                Log.debug(f"Skip on {i}th position with subsample rhythm of {args.subsample_dataset_rhythm}.")
                num_valid_in_log += 1
                continue

            short_file_name = os.path.splitext(os.path.relpath(img_fpath, dataset_input_path))[0]
            if args.explicit and args.explicit[0][-4:] == ".jpg" and not np.any(
                    [short_file_name in e for e in args.explicit]):
                num_valid_in_log += 1
                continue

            npy_path = os.path.join(dataset_output_path, short_file_name)
            if args.enhance:
                if os.path.exists(npy_path + ".npy"):
                    Log.debug(f"Skip {short_file_name}.npy - it is already generated.")
                    num_valid_in_log += 1
                    continue

            cam_timestamp_ns = int(img_fpath.stem)
            city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            if city_SE3_ego is None:
                Log.debug("missing LiDAR pose")
                continue
            im_pose = city_SE3_ego.translation[0:2]

            lidar_fpath = loader.get_closest_lidar_fpath(log_id, cam_timestamp_ns)
            if lidar_fpath is None:
                Log.debug("missing LiDAR file %s" % lidar_fpath)
                # without depth map, can't do this accurately
                continue

            lidar_points = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
            lidar_timestamp_ns = int(lidar_fpath.stem)
            depth_map = loader.get_depth_map_from_lidar(
                lidar_points=lidar_points,
                cam_name=cam_name,
                log_id=log_id,
                cam_timestamp_ns=cam_timestamp_ns,
                lidar_timestamp_ns=lidar_timestamp_ns,
            )

            ego_lane_segments = avm.get_nearby_lane_segments(im_pose, 2)
            is_intersection = False
            for els in ego_lane_segments:
                if els.lane_type != LaneType.VEHICLE or not Polygon(els.polygon_boundary[:, 0:2]).contains(
                        Point(im_pose)):
                    continue
                else:
                    is_intersection = els.is_intersection
                    if is_intersection:
                        break

            if is_intersection:
                with open(intersections_file_path, "a") as f:
                    f.write(str(img_fpath) + "\n")

            if args.plot and successfully_finished == 1:
                img, axs, fig = plt_add_image(img_fpath)
            else:
                img = np.zeros((2048, 1550, 3))
                axs = None
                fig = None

            lanes = {}
            ids = {}
            for lane_segment in avm.get_scenario_lane_segments():
                if lane_segment.lane_type != LaneType.VEHICLE and lane_segment.lane_type != LaneType.BUS:
                    ids[lane_segment.id] = []
                    continue

                cl = avm.get_lane_segment_centerline(lane_segment.id)
                uv_c, start_is_occluded, end_is_occluded = get_lane(cl, ego_SE3_city=city_SE3_ego.inverse(),
                                                                    # , ok, start_out, end_out
                                                                    pinhole_cam=pinhole_cam, depth_map=depth_map,
                                                                    img=img)

                if args.plot and successfully_finished == 1 and len(uv_c) > 0:
                    process_uv_coords(id=lane_segment.id, initial_instance_id=-1, uv_cs=uv_c, lanes=lanes, axs=axs,
                                      only_viz=True, marker=["+"], overall_i=successfully_finished,
                                      start_is_occluded=start_is_occluded, end_is_occluded=end_is_occluded)

                if lane_segment.id in ids:
                    continue

                # we want to start right at the beginning or at merges
                if len(lane_segment.predecessors) == 1:
                    # only skip if predecessor is in range
                    available_predecessors = [a.id for a in avm.get_scenario_lane_segments() if
                                              a.id in lane_segment.predecessors]
                    if len(available_predecessors) > 0:
                        if not available_predecessors[0] in ids:
                            Log.debug("Skip %s because predecessor %s is in scene" % (lane_segment.id,
                                                                                      lane_segment.predecessors))
                            continue
                        else:
                            Log.debug(
                                "Process %s because predecessor %s is already dealt with" % (lane_segment.id,
                                                                                             lane_segment.predecessors))
                    else:
                        Log.debug("Process %s because predecessor %s is not in scene" % (lane_segment.id,
                                                                                         lane_segment.predecessors))
                else:
                    Log.debug("Process %s because %s predecessors and thus new instance" % (lane_segment.id,
                                                                                            len(lane_segment.predecessors)))
                instance_id = get_fresh_instance_id()
                Log.debug("Look at lane segment with ID %s with pre: %s; current instance id is %s" % (
                    lane_segment.id, lane_segment.predecessors, instance_id))

                lanes[instance_id] = []
                ids[lane_segment.id] = []
                add_centerline(id=lane_segment.id, instance_id=instance_id, num_valid_in_log=successfully_finished,
                               ids=ids, lanes=lanes, avm=avm,
                               im_pose=im_pose, city_SE3_ego=city_SE3_ego, pinhole_cam=pinhole_cam,
                               depth_map=depth_map, img=img, axs=axs)

            if not os.path.exists(os.path.dirname(npy_path)):
                os.makedirs(os.path.dirname(npy_path))

            if len(lanes) == 0:
                Log.error(f"Oh oh {log_id} & {img_fpath}")

            len_npy = 100
            new_lanes = np.empty((0, len_npy, 2), dtype=int)
            for k, l in lanes.items():
                if len(l) <= 1:
                    continue

                l = np.asarray(l)
                diff = l[0:-1] - l[1:]
                is_nonzero_diff = [*np.any(diff != 0, axis=1), True]
                l = l[is_nonzero_diff]
                if len(l) <= 1:
                    continue

                nonzero_diffs = np.asarray([d for d in diff if not np.all(d == 0)])
                line_length = np.sum(np.linalg.norm(nonzero_diffs, axis=1))
                num_samples = max(20, int(line_length / 30))

                l = interp_utils.interp_arc(t=num_samples, points=l)
                l = rdp(l, epsilon=0.1)

                if len(l) > max_num_control_points:
                    max_num_control_points = len(l)
                    Log.debug(f"Currently we need {max_num_control_points} control points.")

                if len(l) > len_npy:
                    raise ValueError(f"We need more than {len(l)} control points.")

                dummy = (np.ones((len_npy - len(l), 2)) * np.nan)
                new_lanes = np.concatenate([new_lanes,
                                            np.expand_dims(np.concatenate([np.asarray(l), dummy]), 0)])

                if args.plot and successfully_finished == 1:
                    if len(l) == 0:
                        continue
                    for l_idx in range(len(l) - 1):
                        axs[1].arrow(l[l_idx, 1], l[l_idx, 0], l[l_idx + 1, 1] - l[l_idx, 1],
                                     l[l_idx + 1, 0] - l[l_idx, 0], color=COLORS[k], label=k if l_idx == 0 else "",
                                     head_width=15)
                    axs[1].text(l[0, 1], l[0, 0], str(k))

            if args.plot and successfully_finished == 1:
                finish_plot(axs, fig, img_height=img.shape[0])
                path = args.paths.generate_debug_image_file_path(short_file_name, idx=ImageIdx.LABEL,
                                                                 suffix="npy")
                Log.debug("Plot stored data to file://%s" % path)
                fig.savefig(path)
                Log.plt(epoch=0, tag=os.path.join(str(ImageIdx.LABEL), split, short_file_name),
                        fig=fig)
                plt.close(fig)
                del axs
                del fig

            Log.debug("Store %s.npy" % npy_path)
            np.save(npy_path, new_lanes.round())
            counter += 1
            num_valid_in_log += 1
            successfully_finished += 1

    if counter < min(max_n_in_log, 300) and not args.ignore_missing and args.explicit is None:
        Log.error(f"We only found {counter} files in thread {thread_index}. That is not enough.")
    else:
        Log.info(f"Thread {thread_index} is finished with {counter} files..")
    return counter


if __name__ == '__main__':
    name = "Argoverse 2.0"
    args, parser = argparse(name)
    _lock = threading.Lock()

    for split in ["train", "test", "val"]:
        print(f"\n%%%%%%%%%%%%% Run on {split} %%%%%%%%%%%%%%%")
        args.split = split
        dataset_output_path, dataset_input_path = handle_paths(args)
        intersections_file_path = os.path.join(dataset_output_path, "intersections.txt")

        if dataset_output_path is None:
            continue

        if not args.dry_run:
            if args.enhance:
                with open(os.path.join(dataset_output_path, "metadata.yaml"), "a") as f:
                    yaml.safe_dump({**get_system_specs(), **__push_commit__(args.root, ignore_dirty=True),
                                    **{k: str(v) for k, v in args.__dict__.items()}}, f)
            else:
                with open(os.path.join(dataset_output_path, "metadata.yaml"), "w") as f:
                    yaml.safe_dump({**get_system_specs(), **__push_commit__(args.root, ignore_dirty=True),
                                    **{k: str(v) for k, v in args.__dict__.items()}}, f)

        cam_enums: List[Union[RingCameras, StereoCameras]] = [args.camera]

        print("Load data from %s" % dataset_input_path)
        log_dirs = os.listdir(dataset_input_path)
        if len(log_dirs) != file_checks[split]["logs"] and not args.ignore_missing:
            raise FileNotFoundError(f"We expect {split} to have {file_checks[split]['logs']} files, "
                                    f"but found only {len(log_dirs)}. If this is on purpose set --ignore_missing.")

        # ----------- PARAMS ---------------
        max_dist = 80
        LAST_INSTANCE_ID = -1
        COLORS = {
            -1: (1, 0, 0)
        }

        print("Store npy files in file://%s" % os.path.join(dataset_output_path, "<log_id>", "sensors", "cameras",
                                                            args.camera))
        counter = 0
        max_num_control_points = 0
        subsample_factor = args.subsample_dataset_rhythm if args.subsample_dataset_rhythm and args.subsample_dataset_rhythm > 0 else 1
        total = min(args.max_n, int(320 * len(log_dirs) / subsample_factor))
        threads = list()
        argoverse_max_n = 3422
        log_id_batch_size = math.ceil(len(log_dirs) / args.loading_workers)
        do_single_thread = 0 < args.max_n < argoverse_max_n or (args.explicit is not None and len(args.explicit) > 0)
        if do_single_thread:
            Log.warning("This is single threading!")
            args.loading_workers = 1

        log_dirs = [l for l in log_dirs if len(l) == 36]
        if len(log_dirs) != file_checks[split]["logs"] and not args.ignore_missing:
            raise ValueError(
                f"We only found {len(log_dirs)}, but expected {file_checks[split]['logs']} logs for {split}.")
        log_dirs = np.asarray(log_dirs)

        Log.info(f"{len(log_dirs)} are valid for this run "
                    f"and will be distributed on {args.loading_workers} threads.")

        for idx in range(args.loading_workers):
            if do_single_thread:
                process_log(log_ids=log_dirs, thread_index=idx,
                            max_n_in_log=args.max_n, subsample_factor=subsample_factor)
            else:
                x = threading.Thread(target=process_log,
                                     kwargs={"log_ids": log_dirs[idx * log_id_batch_size:(idx + 1) * log_id_batch_size],
                                             "thread_index": idx, "max_n_in_log": args.max_n,
                                             "subsample_factor": subsample_factor})
                threads.append(x)
                x.start()

        Log.info("So let's wait...")
        # wait for all to finish
        is_running = np.ones(len(threads), dtype=bool)
        total = min(args.max_n, int(320 * len(log_dirs) / subsample_factor))
        with tqdm(total=total, desc=f"Running in all {len(threads)} threads") as pbar:
            while np.any(is_running):

                where = np.where(is_running)[0]
                for index in where:
                    threads[index].join(timeout=1)
                    is_running[index] = threads[index].is_alive()

                    if not threads[index].is_alive():
                        Log.info(f"Finished thread {index}")

                new_pbar_total = 0
                for DIR in log_dirs:
                    if not os.path.isdir(os.path.join(dataset_output_path, DIR)):
                        continue
                    image_folder = os.path.join(dataset_output_path, DIR, "sensors", "cameras", "ring_front_center")
                    new_pbar_total += len([name for name in os.listdir(image_folder) if
                                           os.path.isfile(os.path.join(image_folder, name))])

                new_where = np.where(is_running)[0]
                if len(new_where) < len(where):
                    pbar.set_description(f"Threads {new_where}")
                pbar.update(new_pbar_total - pbar.last_print_n)

                sleep(60)
