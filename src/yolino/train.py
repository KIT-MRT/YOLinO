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
import timeit

import torch
from tqdm import tqdm

from yolino.runner.trainer import TrainHandler
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Log

if __name__ == "__main__":
    start = timeit.default_timer()
    try:
        args = general_setup("Training")
        trainer = TrainHandler(args)

        if args.gpu:
            if args.gpu_id >= 0:
                import os

                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
                torch.cuda.set_device(args.gpu_id)
            Log.error(torch.cuda.is_available())
            Log.error(torch.cuda.current_device())
            for i in range(torch.cuda.device_count()):
                Log.error(torch.cuda.device(i))
                Log.error(torch.cuda.get_device_name(i))

        Log.time(key="setup", value=(timeit.default_timer() - start))
        for epoch in range(trainer.model_epoch, args.epoch):
            epoch_start = timeit.default_timer()

            Log.info("")
            Log.warning('**** Epoch %d/%s %s ****' % (epoch, args.epoch, args.id))

            ###### TRAIN #######
            for i, data in tqdm(enumerate(trainer.loader), total=len(trainer.loader), desc="Train %s" % args.id):
                try:
                    images, grid_tensor, fileinfo, duplicate_info, params = data
                    for j, f in enumerate(fileinfo):
                        trainer.dataset.params_per_file[f] = {}
                        for k, v in params.items():
                            trainer.dataset.params_per_file[f].update({k: v[j].item()})

                    Log.debug("Iteration %d" % i)
                    # if epoch == 1:
                    inference_start = timeit.default_timer()
                    _, preds = trainer(fileinfo, images, grid_tensor, epoch=epoch, image_idx_in_batch=i,
                                       first_run=(i == 0))
                    # if epoch == 1:
                    Log.time(key="infer", value=timeit.default_timer() - inference_start)

                    Log.debug("Training step finished..")

                    num_duplicates = int(sum(duplicate_info["total_duplicates_in_image"]).item())
                    trainer.on_images_finished(preds=preds.detach().cpu(), grid_tensor=grid_tensor, epoch=epoch,
                                               filenames=fileinfo, images=images, is_train=True,
                                               num_duplicates=num_duplicates)

                    Log.debug("---- Iteration done i=%d ----" % i)
                except (Exception, BaseException) as e:
                    Log.error("Error with file %s, epoch %d, iteration %d" % (str(fileinfo), epoch, i))
                    raise e
                Log.time(key="train_batch", value=timeit.default_timer() - epoch_start)
            trainer.on_train_epoch_finished(epoch, fileinfo, images, preds=preds.detach(), grid_tensors=grid_tensor)

            Log.time(key="train_epoch", value=timeit.default_timer() - epoch_start)
            Log.debug("Training done epoch %d" % epoch)

            ###### EVAL #######
            if trainer.is_time_for_val(epoch):
                Log.info("")
                Log.warning('**** EPOCH %d EVALUATION %s ****' % (epoch, args.id))
                with torch.no_grad():

                    eval_batch_time = timeit.default_timer()
                    for i, data in enumerate(tqdm(trainer.val_loader, desc="Eval %s" % args.id)):
                        images, grid_tensor, fileinfo, duplicate_info, params = data
                        for j, f in enumerate(fileinfo):
                            trainer.val_dataset.params_per_file[f] = {}
                            for k, v in params.items():
                                trainer.val_dataset.params_per_file[f].update({k: v[j].item()})

                        _, preds = trainer(fileinfo, images, grid_tensor, epoch=epoch, image_idx_in_batch=i, is_train=False)

                        num_duplicates = int(sum(duplicate_info["total_duplicates_in_image"]).item())
                        trainer.on_images_finished(preds=preds.detach().cpu(), grid_tensor=grid_tensor, epoch=epoch,
                                                   filenames=fileinfo, images=images, is_train=False,
                                                   num_duplicates=num_duplicates)
                        Log.time(key="eval_batch", value=timeit.default_timer() - eval_batch_time)
                trainer.on_val_epoch_finished(epoch)
                Log.time(key="eval_epoch_finished", value=timeit.default_timer() - eval_batch_time)

                if trainer.is_converged(epoch):
                    break

            # if epoch == 1 or epoch == args.eval_iteration:
            Log.time(key="epoch", value=timeit.default_timer() - epoch_start)

        finish_start = timeit.default_timer()
        trainer.on_training_finished(epoch=epoch, do_nms=args.nms)
        Log.time(key="finish", value=timeit.default_timer() - finish_start)
    except (Exception, BaseException) as e:
        Log.finish()
        raise e
