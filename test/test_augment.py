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
import unittest

import numpy as np
import torch
from matplotlib import pyplot as plt
from yolino.dataset.augmentation import RandomCropWithLabels
from yolino.dataset.dataset_base import DatasetInfo
from yolino.dataset.dataset_factory import DatasetFactory
from yolino.utils.enums import Dataset, Augmentation
from yolino.utils.logger import Log
from yolino.utils.test_utils import test_setup


class TestAugmentation(unittest.TestCase):
    plot = False

    def setUp(self):
        self.crop_portion = 0.75
        self.target_size = torch.tensor([40, 40])
        self.crop_transform = RandomCropWithLabels(crop_portion=self.crop_portion)

    @unittest.skipIf("DATASET_ARGO2" is not in os.environ or "DATASET_ARGO2_IMG" is not in os.environ
                     or not os.path.isdir(os.environ["DATASET_ARGO2"])
                     or not os.path.isdir(os.environ["DATASET_ARGO2_IMG"]),
                     "No data to evaluate on")
    def test_normalization(self):

        args = test_setup(name=self._testMethodName, dataset=str(Dataset.ARGOVERSE2))
        args.ignore_missing = True

        args.max_n = 1
        # args.plot = True

        args.augment = ""
        no_aug_ds, loader = DatasetFactory.get(dataset_enum=args.dataset, only_available=True, split=args.split,
                                               args=args, shuffle=False, augment=False)

        args.augment = [Augmentation.NORM]  # list(Augmentation)
        ds, loader = DatasetFactory.get(dataset_enum=args.dataset, only_available=True, split=args.split, args=args,
                                        shuffle=False, augment=True)
        #
        args.augment = list(Augmentation)
        full_aug_ds, loader = DatasetFactory.get(dataset_enum=args.dataset, only_available=True, split=args.split,
                                                 args=args, shuffle=False, augment=True)

        img_raw, _, _, _, _ = no_aug_ds.__getitem__(0)
        img, _, _, _, _ = ds.__getitem__(0)
        full_aug_ds.__getitem__(0)

        expected_norm_img = torch.stack([((img_raw[i] - ds.augmentor.norm_mean[i]) / ds.augmentor.norm_std[i])
                                         for i in [0, 1, 2]])
        expected_mean = torch.tensor(
            [torch.mean((img_raw[i] - ds.augmentor.norm_mean[i]) / ds.augmentor.norm_std[i]).item()
             for i in [0, 1, 2]])
        means = torch.mean(img, dim=[1, 2])
        ds: DatasetInfo
        self.assertTrue(torch.all(abs(expected_mean - means) < 0.001), torch.stack([expected_mean, means]))

    def test_random_crop_inside(self):
        # c-shape
        # line has sharp turn but is inside => nothing happens, nothing lost
        # line inside the crop => nothing happens
        lines = [[15, 11.], [20, 25], [29, 11]]
        augmented_segments, params = self.apply_augmentation(lines, self.target_size, self.crop_transform,
                                                             title="Easy: line inside")

        moved_input_lines = (torch.tensor(lines) - torch.tensor([params["crop_t"], params["crop_l"]]))
        self.assertTrue(torch.all(augmented_segments[0, 0:3] == moved_input_lines))

    def test_random_crop_one_outside(self):
        # line has one end outside => first part is untouched; end is cropped
        lines = [[39, 1.], [20, 25], [29, 29]]
        augmented_segments, params = self.apply_augmentation(lines, self.target_size, self.crop_transform,
                                                             title="One end outside")

        self.assertEqual(len(augmented_segments), 1, f"Augmentation returned {len(augmented_segments)} "
                                                     f"segments, but we expected 1")
        not_nan = torch.sum(~augmented_segments[0, :, 0].isnan())
        self.assertEqual(not_nan, 3, f"Augmentation returned {not_nan} segments, but we expected 3")

        moved_input_lines = torch.tensor(lines) - torch.tensor([params["crop_t"], params["crop_l"]])
        self.assertTrue(torch.all(augmented_segments[0, 1:3] == moved_input_lines[1:]), "%s, %s"
                        % (augmented_segments[0, 1:3], moved_input_lines))

        self.assertTrue(augmented_segments[0, 0, 0] == 40 - params["crop_t"] - params["crop_b"]
                        or augmented_segments[0, 0, 1] == 0)

    def test_random_crop_both_outside(self):
        # both ends outside => center is preserved
        lines = [[39, 1.], [20, 25], [39, 39]]
        augmented_segments, params = self.apply_augmentation(lines, self.target_size, self.crop_transform,
                                                             title="Both ends outside")

        self.assertEqual(len(augmented_segments), 1, f"Augmentation returned {len(augmented_segments)} "
                                                     f"segments, but we expected 3")
        not_nan = torch.sum(~augmented_segments[0, :, 0].isnan())
        self.assertEqual(not_nan, 3, f"Augmentation returned {not_nan} segments, but we expected 3")

        moved_input_lines = torch.tensor(lines) - torch.tensor([params["crop_t"], params["crop_l"]])
        self.assertTrue(torch.all(augmented_segments[0, 1] == moved_input_lines[1]),
                        "Segments %s, Params %s" % (str(augmented_segments[0, 1]), moved_input_lines))

        self.assertTrue(augmented_segments[0, 0, 0] == 40 - params["crop_t"] - params["crop_b"]
                        or augmented_segments[0, 0, 1] == 0,
                        "Segments %s, Params %s" % (str(augmented_segments[0, 0]), params))
        self.assertTrue(augmented_segments[0, 2, 0] == 40 - params["crop_t"] - params["crop_b"]
                        or augmented_segments[0, 2, 1] == 40 - params["crop_r"] - params["crop_l"],
                        "Segments %s, Params %s" % (str(augmented_segments[0, 2]), params))

    def test_random_crop_both_outside_with_stage(self):
        # both ends outside => center is preserved
        lines = [[39, 1.], [20, 25], [21, 25], [20, 25], [39, 39]]
        augmented_segments, params = self.apply_augmentation(lines, self.target_size, self.crop_transform,
                                                             title="Both ends outside")

        self.assertEqual(len(augmented_segments), 1, f"Augmentation returned {len(augmented_segments)} "
                                                     f"segments, but we expected 1")
        not_nan = torch.sum(~augmented_segments[0, :, 0].isnan())
        self.assertEqual(not_nan, 3, f"Augmentation returned {not_nan} segments, but we expected 3")

        moved_input_lines = torch.tensor(lines) - torch.tensor([params["crop_t"], params["crop_l"]])
        self.assertTrue(torch.all(augmented_segments[0, 1].round() == moved_input_lines[1]),
                        "Segments %s, Params %s" % (str(augmented_segments[0, 1]), moved_input_lines))

        self.assertTrue(augmented_segments[0, 0, 0] == 40 - params["crop_t"] - params["crop_b"]
                        or augmented_segments[0, 0, 1] == 0,
                        "Segments %s, Params %s" % (str(augmented_segments[0, 0]), params))
        self.assertTrue(augmented_segments[0, 2, 0] == 40 - params["crop_t"] - params["crop_b"]
                        or augmented_segments[0, 2, 1] == 40 - params["crop_r"] - params["crop_l"],
                        "Segments %s, Params %s" % (str(augmented_segments[0, 4]), params))

    def test_random_crop_both_center_outside(self):
        # c-shape, both ends outside, but also the center => two lines created
        lines = [[39, 1.], [0, 25], [39, 39]]
        augmented_segments, params = self.apply_augmentation(lines, self.target_size, self.crop_transform,
                                                             title="Both ends outside")

        self.assertEqual(len(augmented_segments), 2, f"Augmentation returned {len(augmented_segments)} "
                                                     f"segments, but we expected 2")

        self.assertTrue(augmented_segments[0, 0, 0] == 40 - params["crop_t"] - params["crop_b"]
                        or augmented_segments[0, 0, 1] == 0,
                        "Segments %s, Params %s" % (str(augmented_segments[0]), params))
        self.assertTrue(augmented_segments[0, 1, 0] == 0,
                        "Segments %s, Params %s" % (str(augmented_segments[0]), params))

        self.assertTrue(augmented_segments[1, 0, 0] == 0,
                        "Segments %s, Params %s" % (str(augmented_segments[1]), params))
        self.assertTrue(augmented_segments[1, 1, 0] == 40 - params["crop_t"] - params["crop_b"]
                        or augmented_segments[1, 1, 1] == 40 - params["crop_l"] - params["crop_r"],
                        "Segments %s, Params %s" % (str(augmented_segments[1]), params))

        # c-shape, both ends inside, but center outside => two lines created

    def apply_augmentation(self, lines, target_size, transform, title, height=40, width=40):
        lines = torch.tensor([lines])
        params = {}
        image = torch.zeros((3, height, width), dtype=float)

        cropped_image, segments, removals = transform(image, lines, params)

        if TestAugmentation.plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            major_ticks = np.arange(0, 41, 10)
            minor_ticks = np.arange(0, 41, 1)

            self.setup_plot(ax, major_ticks, minor_ticks, title)
            plt.plot(lines[0, :, 1], lines[0, :, 0])
            plt.scatter([0, 0, width, width], [0, height, 0, height], marker="*", color=(0, 0, 0))
            plt.scatter(params["crop_l"], params["crop_t"])
            plt.scatter(params["crop_l"], target_size[0] - params["crop_b"])
            plt.scatter(target_size[1] - params["crop_r"], params["crop_t"])
            plt.scatter(target_size[1] - params["crop_r"], target_size[0] - params["crop_b"])
            # plt.show()
            path = "/tmp/augment.png"
            Log.info(f"file://{path}")
            plt.savefig(path)

        # cropped in big picture
        if TestAugmentation.plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            self.setup_plot(ax, major_ticks, minor_ticks, "Cropped in big picture %s" % title)

            for segment in segments:
                non_nan_augmented_segments = segment[torch.where(~segment[:, 0].isnan())]
                plt.plot(non_nan_augmented_segments[:, 1] + params["crop_l"],
                         non_nan_augmented_segments[:, 0] + params["crop_t"])

            plt.scatter(params["crop_l"], params["crop_t"])
            plt.scatter(params["crop_l"], target_size[0] - params["crop_b"])
            plt.scatter(target_size[1] - params["crop_r"], params["crop_t"])
            plt.scatter(target_size[1] - params["crop_r"], target_size[0] - params["crop_b"])
            plt.scatter([0, 0, width, width], [0, height, 0, height], marker="+", color=(0, 0, 0))
            # plt.show()
            path = "/tmp/augment1.png"
            Log.info(f"file://{path}")
            plt.savefig(path)

        # cropped only
        if TestAugmentation.plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            self.setup_plot(ax, major_ticks, minor_ticks, "Cropped %s" % title)

            for segment in segments:
                non_nan_augmented_segments = segment[torch.where(~segment[:, 0].isnan())]
                plt.plot(non_nan_augmented_segments[:, 1],
                         non_nan_augmented_segments[:, 0])

            crop_h = height - params["crop_t"] - params["crop_b"]
            crop_w = width - params["crop_r"] - params["crop_l"]
            plt.scatter([0, 0, crop_h, crop_h], [0, crop_w, 0, crop_w], marker="+", color=(0, 0, 0))
            # plt.show()
            path = "/tmp/augment2.png"
            Log.info(f"file://{path}")
            plt.savefig(path)

        return segments, params

    def setup_plot(self, ax, major_ticks, minor_ticks, title):
        ax.set_xlim(-1, max(minor_ticks) + 1)
        ax.set_ylim(-1, max(minor_ticks) + 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()

        plt.title(title)


if __name__ == '__main__':
    unittest.main()
