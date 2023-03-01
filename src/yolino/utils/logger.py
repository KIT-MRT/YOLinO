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
import inspect
import logging
import math
import os
import sys
from logging import StreamHandler
from logging.handlers import WatchedFileHandler

import git
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from yolino.utils.enums import ImageIdx, Level, Logger, TaskType
from yolino.utils.system import get_system_specs

try:
    TB_EXIST = True
    from torch.utils.tensorboard import SummaryWriter
except:
    TB_EXIST = False
    logging.error("No tensorboard available")

try:
    from clearml import Task, TaskTypes

    TRAINS_EXIST = True
except:
    logging.error("No trains / clear ML available")
    TRAINS_EXIST = False

try:
    import wandb

    WANDB_EXIST = True
except:
    logging.error("No weights and biases available")
    WANDB_EXIST = False

version = "v3.0"

class LoggerInfo():
    def __init__(self, name, class_type) -> None:
        self.name = name
        self.do = class_type
        self.writer = None


def getClearMlTaskType(task_type: TaskType):
    if not TRAINS_EXIST:
        raise ModuleNotFoundError("No trains module found")

    if TaskType.PRED:
        return TaskTypes.inference
    if TaskType.TEST:
        return TaskTypes.testing
    if TaskType.TRAIN:
        return TaskTypes.training
    if TaskType.PARAM_OPTIMIZATION:
        return TaskTypes.optimizer

    raise NotImplementedError("Task type %s not found" % task_type)


class Log():
    timing = {}
    log_file_path = None
    ljust_space = 30
    __loggers__ = []
    __level__ = Level.WARN
    __tb__ = None
    __file_log__ = None  # logging.getLogger('yolino_file')
    __cmd__ = logging.getLogger('yolino')
    __trains__ = None
    __trains_task__ = None
    __project_name__ = None
    __wandb_run__ = None
    __wandb_buffer__ = {}

    @classmethod
    def get_caller_str(self, level=0):
        stack = inspect.stack()
        idx = min(len(stack), min(2 + level, len(stack) - 1))
        frame, filename, lineno, function_name, lines, index = stack[idx]
        return ("%s:%d" % (os.path.basename(filename), lineno)).ljust(self.ljust_space)

    @classmethod
    def setup(cls, args, task_type: TaskType, project_suffix=""):

        if args.tags is None:
            args.tags = []

        cls.__project_name__ = args.log_dir + project_suffix

        # ATTENTION might set the args.id to another value
        if Logger.WEIGHTSBIASES in args.loggers and WANDB_EXIST:
            cls.init_wandb(args, task_type)

        tb_path = None
        if TB_EXIST:
            if Logger.TENSORBOARD in args.loggers:
                tb_path = os.path.join("runs", args.id)
            elif Logger.CLEARML in args.loggers and TRAINS_EXIST:
                tb_path = os.path.join("trains", args.id)

        if TB_EXIST and (Logger.TENSORBOARD in args.loggers or (Logger.CLEARML in args.loggers and TRAINS_EXIST)):
            cls.init_tb(args, tb_path)

        if Logger.CLEARML in args.loggers and TRAINS_EXIST:
            cls.init_clearml(args, task_type)

        Log.info("We log to %s and cmd" % (", ".join([str(l) for l in Log.__loggers__])))

    @classmethod
    def upload_params(cls, param_dict):
        if Logger.WEIGHTSBIASES in Log.__loggers__:
            try:
                wandb.config.update(param_dict)
            except wandb.sdk.lib.config_util.ConfigError as e:
                Log.warning("wandb %s" % str(e).split("to")[1].split("If")[0])
                wandb.config.update(param_dict, allow_val_change=True)

    @classmethod
    def init_clearml(cls, args, task_type):
        Log.info("Setup trains logging for %s - %s [%s]" % (cls.__project_name__, args.id, str(task_type)))

        Log.__trains_task__ = Task.init(project_name=cls.__project_name__,
                                        task_name=args.id,
                                        task_type=getClearMlTaskType(task_type),
                                        continue_last_task=args.resume_log)
        # reuse_last_task_id (bool ) – Force a new Task (experiment) with a previously used Task ID, and the same project and Task name.
        # continue_last_task (bool ) – Continue the execution of a previously executed Task (experiment)

        try:
            repo = git.Repo(args.dvc)
            dvc_commit = str(repo.rev_parse("HEAD"))
        except git.exc.InvalidGitRepositoryError as e:
            Log.error("Your DVC is not a git folder %s; we cannnot store the commit hash" % e)
            dvc_commit = ""
        if args.tags is None:
            Log.__trains_task__.add_tags([version])
        else:
            Log.__trains_task__.add_tags([version, *args.tags])
        Log.__trains_task__.connect({'dvc_commit': dvc_commit, **get_system_specs()})

        Log.__trains_task__.set_parameters_as_dict({"Args": vars(args)})
        Log.__trains__ = Log.__trains_task__.get_logger()

        Log.__loggers__.append(Logger.CLEARML)

    @classmethod
    def init_tb(cls, args, tb_path):
        Log.info("Setup tensorboard logging for %s" % tb_path)
        Log.__tb__ = SummaryWriter(log_dir=tb_path)
        Log.__loggers__.append(Logger.TENSORBOARD)

    @classmethod
    def init_wandb(cls, args, task_type):

        Log.info("Setup wandb logging for %s" % (cls.__project_name__))
        Log.__wandb_run__ = wandb.init(project=cls.__project_name__, entity="annkit",
                                       tags=[version, *args.tags],
                                       config=args,
                                       job_type=str(task_type), resume="auto" if args.resume_log else "never")
        # resume: "allow", "must", "never", "auto" or None.
        #   None (default): If the new run has the same ID as a previous run, this run overwrites that data.
        #   "auto" (or True): if the preivous run on this machine crashed, automatically resume it. Otherwise, start a new run. Resumes although config changes and args.retrain.
        #   "allow": if id is set with init(id="UNIQUE_ID") or WANDB_RUN_ID="UNIQUE_ID" and it is identical to a previous run, wandb will automatically resume the run with that id.
        #   "never": if id is set with init(id="UNIQUE_ID") or WANDB_RUN_ID="UNIQUE_ID" and it is identical to a previous run, wandb will crash.
        #   "must": if id is set with init(id="UNIQUE_ID") or WANDB_RUN_ID="UNIQUE_ID" and it is identical to a previous run, wandb will automatically resume the run with the id. Otherwise wandb will crash.

        Log.__loggers__.append(Logger.WEIGHTSBIASES)

        Log.__wandb_buffer__["epoch"] = 0
        args.id = wandb.run.name

    @classmethod
    def init_file_logs(cls, name="yolino", log_file=None):
        file_format = name + ' - %(asctime)s - %(message)s'

        if log_file:
            cls.log_file_path = os.path.abspath(log_file)
        else:
            cls.log_file_path = os.path.abspath(os.path.join("tmp", name + ".log"))
        if not os.path.exists(os.path.dirname(cls.log_file_path)):
            os.makedirs(os.path.dirname(cls.log_file_path))

        fh = WatchedFileHandler(filename=cls.log_file_path, mode="w")
        fh.setLevel(logging.DEBUG)
        try:
            import coloredlogs
            formatter = coloredlogs.ColoredFormatter(file_format)
            fh.setFormatter(formatter)
        except:
            Log.error("No coloredlogs available")

        if os.path.exists(os.path.dirname(cls.log_file_path)):
            Log.__cmd__.addHandler(fh)
        else:
            Log.warning("We could not add file logging. Path %s does not exist" % os.path.dirname(cls.log_file_path))

        print("Log to file://%s" % cls.log_file_path)

    @classmethod
    def setup_cmd(cls, setup_file_log=False, log_file=None, viz_level: Level = Level.WARN, name="yolino"):
        sys.excepthook = Log.__exception__

        if Log.__cmd__ is not None:
            Log.debug("Changing logger to %s!" % name)
            sys.stdout.flush()
            sys.stderr.flush()

        Log.__cmd__ = logging.getLogger()
        Log.__cmd__.handlers.clear()

        Log.__level__ = viz_level

        cmd_format = name + ' - %(asctime)s - %(message)s'  # - PID%(process)d

        sh = StreamHandler()
        sh.setLevel(str(Log.__level__))

        try:
            import coloredlogs
            formatter = coloredlogs.ColoredFormatter(cmd_format, datefmt='%H:%M:%S')
            sh.setFormatter(formatter)
        except:
            Log.error("No coloredlogs available")

        Log.__cmd__.setLevel(logging.DEBUG)

        Log.debug("Show log from %s upwards" % Log.__level__)

        if setup_file_log:
            cls.init_file_logs(log_file=log_file, name=name)

        Log.__cmd__.addHandler(sh)

    @classmethod
    def __exception__(cls, exc_type, exc_value, exc_traceback):
        Log.push(None)
        Log.flush()
        Log.error(msg="%s: %s" % (exc_type.__name__, exc_value), exc_info=(exc_type, exc_value, exc_traceback), level=3)

    @classmethod
    def tag(cls, tag_name):

        if Logger.CLEARML in Log.__loggers__:
            Log.__trains_task__.add_tags(tag_name)

        if Logger.WEIGHTSBIASES in Log.__loggers__:
            Log.__wandb_run__.tags += (tag_name,)

    @classmethod
    def info(self, msg, level=0, **kwargs):
        Log.__cmd__.info(msg="INFO".ljust(7) + Log.get_caller_str(level) + " " + str(msg), **kwargs)
        if Log.__file_log__:
            Log.__file_log__.info(msg="INFO".ljust(7) + Log.get_caller_str(level) + " " + str(msg), **kwargs)

    @classmethod
    def debug(self, msg, level=0, **kwargs):
        Log.__cmd__.debug(msg="DEBUG".ljust(7) + Log.get_caller_str(level) + " " + str(msg), **kwargs)
        if Log.__file_log__:
            Log.__file_log__.debug(msg="DEBUG".ljust(7) + Log.get_caller_str(level) + " " + str(msg), **kwargs)

    @classmethod
    def warning(self, msg, level=0, **kwargs):
        Log.__cmd__.warning(msg="WARN".ljust(7) + Log.get_caller_str(level) + " " + str(msg), **kwargs)

        if Log.__file_log__:
            Log.__file_log__.warning(msg="WARN".ljust(7) + Log.get_caller_str(level) + " " + str(msg), **kwargs)

    @classmethod
    def error(self, msg, level=0, **kwargs):
        Log.__cmd__.error(msg="ERROR".ljust(7) + Log.get_caller_str(level) + " " + str(msg), **kwargs)

        if Log.__file_log__:
            Log.__file_log__.error(msg="ERROR".ljust(7) + Log.get_caller_str(level) + " " + str(msg), **kwargs)

    @classmethod
    def flush(self):
        Log.__cmd__.handlers[0].flush()

        if Log.__file_log__:
            Log.__file_log__.handlers[0].flush()

    @classmethod
    def time(cls, key, value, epoch=None):

        dict = {"time/%s" % key: value}
        if Logger.TENSORBOARD in Log.__loggers__:
            Log.__tb__.add_scalar(key, dict)

        if Logger.CLEARML in Log.__loggers__:
            Log.__trains__.report_scalar("time", key, value=value)

        if Logger.WEIGHTSBIASES in Log.__loggers__:
            isin = np.isin(dict.keys(), Log.__wandb_buffer__)
            if np.any(isin):
                Log.warning("We are overwriting logging values %s" % dict.keys()[isin])
                Log.push(epoch)

            if epoch is not None and Log.__wandb_buffer__["epoch"] != epoch:
                Log.push(epoch)

            Log.__wandb_buffer__.update(dict)

        if key not in Log.timing:
            Log.timing[key] = []
        Log.timing[key].append(value)

    @classmethod
    def scalars(self, tag, dict, epoch):
        new_dict = {}
        for k, v in dict.items():

            if type(v) == np.ndarray or type(v) == list:
                if len(v) == 1:
                    new_dict[str(k) + "/" + tag] = v[0]
                else:
                    Log.warning("Miss to push %s: %s" % (k, v))
                    continue
            else:
                new_dict[str(k) + "/" + tag] = v

        if len(new_dict) == 0:
            Log.warning("No scalar to report for %s epoch %s. Input was %s" % (tag, epoch, dict))
            return

        if Logger.TENSORBOARD in Log.__loggers__:
            Log.__tb__.add_scalars(tag, new_dict, epoch)

        if Logger.CLEARML in Log.__loggers__:
            for k, v in new_dict.items():
                Log.__trains__.report_scalar(tag, k, iteration=epoch, value=v)

        if Logger.WEIGHTSBIASES in Log.__loggers__:
            if epoch is not None and Log.__wandb_buffer__["epoch"] != epoch:
                Log.push(epoch)

            Log.debug(f"Push {list(new_dict.keys())} in epoch {epoch}", level=1)
            Log.__wandb_buffer__.update(new_dict)

    @classmethod
    def plt(self, epoch, fig, tag="unknown", level=0):
        if Logger.WEIGHTSBIASES in Log.__loggers__:
            wandb.log({tag: fig})

    @classmethod
    def img(self, name, img, epoch, tag="unknown", imageidx: ImageIdx = ImageIdx.DEFAULT, level=0):

        import cv2
        if tag == "unknown":
            Log.warning("Tag is unkown: stack %s" % Log.get_pretty_stack(), level=level + 1)
            return

        dirs = name.split("/")[-3:]
        path_parts = os.path.basename(dirs[-1]).split("_")
        viz_name = str(imageidx) + "/" + tag
        file_name = os.path.join(*dirs[0:2], path_parts[0])
        tb_name = os.path.join(viz_name, file_name)
        if Logger.TENSORBOARD in Log.__loggers__:
            if type(img) is np.ndarray:
                # somehow tensorboard visualizes differently; this fixes BGR viz
                torch_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                from yolino.viz.plot import convert_to_torch_image
                torch_img = convert_to_torch_image(torch_img)
            else:
                torch_img = img

            # sth like 'grid_label_train_img/driver_23_30frame/05151649_0422.MP4/00030.jpg' as tb constructs tree from that
            Log.__tb__.add_image(tb_name, torch_img, epoch)

        if Logger.WEIGHTSBIASES in Log.__loggers__:
            if epoch is not None and Log.__wandb_buffer__["epoch"] != epoch:
                Log.push(epoch)

            Log.debug("Push %s to wandb at step=%s" % (viz_name, epoch), level=level + 1)
            if "train" in tag:
                Log.__wandb_buffer__[viz_name] = [wandb.Image(img, caption=file_name)]
            else:
                Log.__wandb_buffer__[tb_name] = [wandb.Image(img, caption=file_name)]

    @classmethod
    def get_pretty_stack(cls):
        return ["%s: %d" % (os.path.basename(f.filename), f.lineno) for f in inspect.stack()].reverse()

    @classmethod
    def grid(self, name, images, epoch, tag="unknown", imageidx: ImageIdx = ImageIdx.DEFAULT, level=0):
        grid = torchvision.utils.make_grid(images, nrow=math.ceil((math.sqrt(len(images)))))

        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))

        from yolino.viz.plot import convert_torch_to_plt_style
        Log.warning("Save %s image to file://%s" % (tag, os.path.abspath(name)), level=level + 1)
        plt.imsave(name, convert_torch_to_plt_style(grid))
        Log.img(name, grid, epoch, tag=tag, imageidx=imageidx, level=level + 1)

    @classmethod
    def graph(self, model, images):

        if Logger.WEIGHTSBIASES in Log.__loggers__:
            wandb.watch(model)

    @classmethod
    def eval_summary(self, scores):
        for k, v in scores.items():
            if k == "confusion":
                continue

            val_str = ""
            if np.isscalar(v):
                val_str = " %.2f" % v
            else:
                for scalar in v:
                    val_str += " %s" % str(scalar)

            msg = ("%s:" % k).ljust(20) + " %s" % val_str
            if Logger.WEIGHTSBIASES in Log.__loggers__:
                Log.info(msg, level=1)
            else:
                print(msg)

        Log.push(next_epoch=None)

    @classmethod
    def fail_summary(self, epoch, last_loss, tag="unknown", level=0):
        Log.warning('Failed in epoch %d, %s last loss within this epoch: %s' % (
            epoch, tag.lower(), [l.item() if type(l) == torch.tensor else l for l in last_loss]), level=level + 1)
        Log.scalars(tag.lower(), {"loss": torch.inf, "epoch": epoch}, epoch)
        Log.push(next_epoch=None)

    @classmethod
    def gradients(self, model):
        if Logger.WEIGHTSBIASES in Log.__loggers__:
            wandb.watch(model, log="all", log_freq=3)

    @classmethod
    def finish(cls):
        if Logger.WEIGHTSBIASES in Log.__loggers__ and Log.log_file_path is not None and os.path.exists(
                Log.log_file_path):
            Log.error("Upload log file")
            wandb.save(Log.log_file_path)
        Log.push(next_epoch=None)

    @classmethod
    def confusion(cls, matrix, name, epoch, tag, imageidx):
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix)  # , display_labels=range(0,self.dataset.num_classes))
        conf_plot = disp.plot()

        if Logger.CLEARML in Log.__loggers__:
            Log.__trains__.report_confusion_matrix(str(imageidx) + "_" + tag, name, iteration=epoch,
                                                   matrix=matrix, xaxis="prediction", yaxis="gt")

        if Logger.WEIGHTSBIASES in Log.__loggers__:
            plt = conf_plot.figure_

            if epoch is not None and Log.__wandb_buffer__["epoch"] != epoch:
                Log.push(epoch)
            Log.__wandb_buffer__[name + "_" + str(imageidx) + "_"] = plt

    @classmethod
    def glitch(cls, args, prediction: torch.tensor):
        path = os.path.join(args.paths.debug_folder, "glitch_prediction.pt")
        Log.error("We have a glitch! We store predictions in file://%s" % path, level=1)
        Log.tag("glitch")
        torch.save(prediction, path)
        Log.push(next_epoch=None)

    @classmethod
    def push(cls, next_epoch):
        if Logger.WEIGHTSBIASES in Log.__loggers__:
            if len(Log.__wandb_buffer__.keys()) > 1:
                wandb.log(Log.__wandb_buffer__, step=Log.__wandb_buffer__["epoch"])

                if next_epoch == None:
                    Log.__wandb_buffer__ = {"epoch": Log.__wandb_buffer__["epoch"]}
                else:
                    Log.__wandb_buffer__ = {"epoch": next_epoch}

    @classmethod
    def malconfig(cls, msg):
        Log.error(msg, level=1)
        Log.tag("malconfig")

        from yolino.runner.trainer import VAL_TAG
        Log.scalars(tag=VAL_TAG, dict={"loss/best": math.inf, "best_epoch": 0}, epoch=0)
        Log.push(next_epoch=0)

        exit(0)
