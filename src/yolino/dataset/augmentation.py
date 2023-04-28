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

import cv2
import torch
from torchvision.transforms import ColorJitter, RandomErasing, Normalize, RandomCrop, RandomRotation, Resize
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import _setup_angle
from yolino.utils.enums import Augmentation, CoordinateSystem, ColorStyle, ImageIdx
from yolino.utils.geometry import t_cart2pol, t_pol2cart, intersection_segments, reformat2shapely
from yolino.utils.logger import Log
from yolino.viz.plot import plot, save_grid, convert_to_torch_image


def has_op(obj, op):
    return callable(getattr(obj, op, None))


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        errors = []
        params = {}
        for t in self.transforms:
            use_label = getattr(t, "use_label", False)
            if use_label:
                img, label, ok = t(img, label, params=params)
                if not ok:
                    errors.append(t)
            else:
                if isinstance(t, ColorJitter) or isinstance(t, RandomErasing) or isinstance(t, Normalize):
                    img = t(img)
                else:
                    img = t(img, params=params)

        return img, label, errors, params


class DatasetTransformer:
    def __init__(self, args, sky_crop: int, side_crop: int, augment=True, keep_scale=False,
                 norm_mean=None, norm_std=None):
        self.augment = augment

        if norm_mean is None or norm_std is None:
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]
        else:
            self.norm_std = norm_std
            self.norm_mean = norm_mean

        self.erasing_ratio = (0.15, 2.5)
        self.erasing_scale = (0.02, 0.05)
        self.erasing_p = 0.2
        self.erasing_value = "random"

        self.jitter_hue = 0.2
        self.jitter_saturation = 0.5
        self.jitter_contrast = 0.5
        self.jitter_brightness = 0.5

        self.plot = args.plot

        self.crop_portion = 1 - args.crop_range

        self.rotation_degrees = math.degrees(args.rotation_range)

        self.args = args
        self.methods = args.augment if args.augment else []

        self.sky_crop = sky_crop
        self.side_crop = side_crop

        self.transform = self.compose_transforms(augment, keep_scale, side_crop, sky_crop,
                                                 target_size=self.args.img_size)

    def reproduce(self, image, uv_lines, filename, target_size, params):
        transform = self.compose_transforms(augment=self.augment, keep_scale=False, side_crop=self.side_crop,
                                            sky_crop=self.sky_crop, fixed=True, target_size=target_size, **params)
        return self.apply_transforms(filename=filename, image=image, uv_lines=uv_lines, transform=transform)

    def __call__(self, image, uv_lines, filename):
        return self.apply_transforms(filename, image, uv_lines, self.transform)

    def apply_transforms(self, filename, image, uv_lines, transform):

        new_image, new_uv_lines, errors, params = transform(image, uv_lines)

        if (len(errors) > 0 or self.plot) and uv_lines is not None:  # and self.args.level == Level.DEBUG:
            path = self.args.paths.generate_debug_image_file_path(filename, ImageIdx.LABEL, suffix="augment_error")
            if len(errors) > 0:
                Log.warning("Error in augmentation steps %s" % (errors))

            img1, _ = plot(lines=torch.unsqueeze(new_uv_lines, dim=0) if uv_lines is not None else [], name=None,
                           image=new_image, show_grid=True, cell_size=self.args.cell_size,
                           colorstyle=ColorStyle.ORIENTATION, coordinates=CoordinateSystem.UV_CONTINUOUS, epoch=None,
                           tag="augment_error", imageidx=ImageIdx.LABEL)
            cv2.putText(img1, 'Augmented', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            orig_image, old_uv_lines, ok, _ = Compose(
                [FixedCrop(sky_crop=self.sky_crop, left_crop=self.side_crop, right_crop=self.side_crop),
                 FixedScale(new_image.shape[1:])])(image, uv_lines)
            img2, ok = plot(torch.unsqueeze(old_uv_lines, dim=0) if old_uv_lines is not None else [], name=None,
                            image=orig_image, show_grid=True, cell_size=self.args.cell_size,
                            colorstyle=ColorStyle.ORIENTATION, coordinates=CoordinateSystem.UV_CONTINUOUS,
                            epoch=None, imageidx=ImageIdx.LABEL)
            cv2.putText(img2, 'Original', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            grid = [convert_to_torch_image(img2), convert_to_torch_image(img1)]
            save_grid(grid, path, title="augmentation")
            # Log.grid(filename, grid, tag="augment", epoch=-1, imageidx=ImageIdx.LABEL)
        return new_image, new_uv_lines, params

    def compose_transforms(self, augment, keep_scale, side_crop, sky_crop, target_size, fixed=False, **kwargs):
        transforms_to_compose = [FixedCrop(sky_crop=sky_crop, left_crop=side_crop, right_crop=side_crop)]

        if Augmentation.NORM in self.methods:  # apply also if not augmented, but in method list
            if not fixed:
                transforms_to_compose.append(Normalize(mean=self.norm_mean, std=self.norm_std))

        if augment:
            if Augmentation.ROTATION in self.methods:
                if fixed:
                    transforms_to_compose.append(FixedRotationWithLabels(angle=kwargs["rrotate_angle"]))
                else:
                    transforms_to_compose.append(RandomRotationWithLabels(degree=self.rotation_degrees))
            if Augmentation.CROP in self.methods:
                if fixed:
                    transforms_to_compose.append(
                        FixedCrop(sky_crop=kwargs["crop_t"], left_crop=kwargs["crop_l"], right_crop=kwargs["crop_r"],
                                  bottom_crop=kwargs["crop_b"]))
                else:
                    transforms_to_compose.append(RandomCropWithLabels(self.crop_portion))
            if Augmentation.JITTER in self.methods:
                if not fixed:
                    transforms_to_compose.append(ColorJitter(brightness=self.jitter_brightness,
                                                             contrast=self.jitter_contrast,
                                                             saturation=self.jitter_saturation,
                                                             hue=self.jitter_hue))
            if Augmentation.ERASING in self.methods:
                if not fixed:
                    transforms_to_compose.append(RandomErasing(p=self.erasing_p, scale=self.erasing_scale,
                                                               ratio=self.erasing_ratio,
                                                               value=self.erasing_value))
        if not keep_scale:
            transforms_to_compose.append(FixedScale(target_size=target_size))
        text = ""
        for transform in transforms_to_compose:
            text += "\n" + " ".ljust(Log.ljust_space + 20)
            text += str(transform)
        Log.debug("Augment with %s" % text)
        return Compose(transforms_to_compose)


class FixedCrop(torch.nn.Module):
    def __init__(self, sky_crop, left_crop, bottom_crop=0, right_crop=0):
        super().__init__()      
        self.sky_crop = int(sky_crop)
        self.left_crop = int(left_crop)
        self.bottom_crop = int(bottom_crop)
        self.right_crop = int(right_crop)
        self.use_label = True

    def __call__(self, image, label, params):
        if self.sky_crop == image.shape[1] - self.bottom_crop: 
            Log.error(f"This crop will result in no image as you selected range {self.sky_crop}:{image.shape[1] - self.bottom_crop} of the height of {image.shape}. Maybe you confused height and width?")
        if self.left_crop == image.shape[2] - self.right_crop: 
            Log.error(f"This crop will result in no image as you selected range {self.left_crop}:{image.shape[2] - self.right_crop} of the width of {image.shape}. Maybe you confused height and width?")
        image = image[:,
                self.sky_crop:image.shape[1] - self.bottom_crop,
                self.left_crop:image.shape[2] - self.right_crop]
        if label is not None:
            label = label - torch.tensor([self.sky_crop, self.left_crop])
        return image, label, True

    def __repr__(self):
        s = '(sky={}, '.format(self.sky_crop)
        s += 'sides={}), '.format(self.left_crop)
        return self.__class__.__name__ + s


class FixedScale(torch.nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
        h, w = target_size
        self.resize = Resize((h, w), antialias=None)
        self.use_label = True

    def __call__(self, image, label, params):
        _, i_h, i_w = image.shape
        h, w = self.target_size

        scale = 1 / (i_h / h)
        if label is not None:
            label = label * torch.tensor(scale)

        image = self.resize(image)
        return image, label, True

    def __repr__(self):
        s = '(size={})'.format(self.target_size)
        return self.__class__.__name__ + s


class FixedRotationWithLabels(torch.nn.Module):
    def __init__(self, angle):
        super().__init__()
        self.angle = angle
        self.use_label = True

    def forward(self, image, uv_lines, params):
        image = F.rotate(image, float(self.angle))

        if uv_lines is not None:
            _, h, w = image.shape
            for i_idx, instance in enumerate(uv_lines):
                for p_idx, point in enumerate(instance):
                    point[0] = point[0] - h / 2
                    point[1] = point[1] - w / 2

                    # convert to polar
                    polyline_polar = t_cart2pol(point)
                    # rotate
                    polyline_polar[1] += math.radians(self.angle)
                    # convert to carthesian
                    point = t_pol2cart(polyline_polar)
                    # uncenter from image (put back into uv coordiates)
                    point[0] = point[0] + h / 2
                    point[1] = point[1] + w / 2

                    uv_lines[i_idx, p_idx] = point

        return image, uv_lines, True

    def __repr__(self):
        s = '(angle=%.1f)' % self.angle
        return self.__class__.__name__ + s


class RandomRotationWithLabels(torch.nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degrees = _setup_angle(degree, name="degrees", req_sizes=(2,))
        self.use_label = True

    def forward(self, image, uv_lines, params):
        # (3,h,w), (instances, points, 2 geom coords)
        angle = RandomRotation.get_params(self.degrees)
        # Log.debug("Rotate image by %.3f" % angle)
        image = F.rotate(image, angle)

        _, h, w = image.shape
        for i_idx, instance in enumerate(uv_lines):
            for p_idx, point in enumerate(instance):
                point[0] = point[0] - h / 2
                point[1] = point[1] - w / 2

                # convert to polar
                polyline_polar = t_cart2pol(point)
                # rotate
                polyline_polar[1] += math.radians(angle)
                # convert to carthesian
                point = t_pol2cart(polyline_polar)
                # uncenter from image (put back into uv coordiates)
                point[0] = point[0] + h / 2
                point[1] = point[1] + w / 2

                uv_lines[i_idx, p_idx] = point

        params["rrotate_angle"] = angle
        return image, uv_lines, True

    def __repr__(self):
        s = '(degrees=%.1f,%.1f)' % (self.degrees[0], self.degrees[1])
        return self.__class__.__name__ + s


def inside(geom_line, geom_box):
    first_x_inside = geom_line.bounds[0] > geom_box.bounds[0] and geom_line.bounds[0] < geom_box.bounds[2]
    snd_x_inside = geom_line.bounds[2] > geom_box.bounds[0] and geom_line.bounds[2] < geom_box.bounds[2]
    first_y_inside = geom_line.bounds[1] > geom_box.bounds[1] and geom_line.bounds[1] < geom_box.bounds[3]
    snd_y_inside = geom_line.bounds[3] > geom_box.bounds[1] and geom_line.bounds[3] < geom_box.bounds[3]

    return first_x_inside and snd_x_inside and first_y_inside and snd_y_inside


class RandomCropWithLabels(torch.nn.Module):
    def __init__(self, crop_portion):
        """

        crop_portion: float
        portion of the image to crop

        """
        super().__init__()
        self.crop_portion = crop_portion
        self.use_label = True

    def forward(self, image, uv_lines, params):
        """

        image: cv2 image Mat
        The image can have any size. We will crop according to crop_portion given to the init function.

        """
        min_crop_h = int(image.shape[1] * self.crop_portion)
        crop_h = torch.randint(min_crop_h, image.shape[1]-1, size=(1,)).item()

        min_crop_w = int(image.shape[2] * self.crop_portion)
        crop_w = int(min_crop_w * crop_h / min_crop_h)

        # Log.debug("Crop square is %dx%d" % (crop_h, crop_w))
        if crop_h > image.shape[1] or crop_w > image.shape[2]:
            raise ValueError(
                "Random crop specs %d > %d (height), %d > %d (width) (args provided crop of %s, resulting in %dx%d; )" % (
                    crop_h, image.shape[1], crop_w, image.shape[2], self.crop_portion, min_crop_h, min_crop_w))
        i, j, th, tw = RandomCrop.get_params(image, (crop_h, crop_w))
        i = max(i, 1)
        j = max(j, 1)

        # Log.debug("Crop image of %s to %dx%d at (%d,%d)" % (str(image.shape), th, tw, i, j))
        cropped_image = F.crop(image, i, j, th, tw)

        removals = 0
        # segments = torch.ones_like(uv_lines) * torch.nan
        segments = torch.empty((0, 500, 2), dtype=uv_lines.dtype, device=uv_lines.device)
        for i_idx, instance in enumerate(uv_lines):
            if torch.all(torch.isnan(instance)):
                continue

            non_nan_instance = instance[torch.where(torch.logical_not(torch.isnan(instance[:, 0])))]
            if len(non_nan_instance) <= 1:
                Log.debug("Full polyline will be erased from the labels "
                          "with target size %dx%d at (%d, %d) and instance %s" % (th, tw, i, j, instance))
                continue

            segments = intersection_segments((i, j), tw, th, reformat2shapely(non_nan_instance.numpy()),
                                             segments=segments)

        if removals > 0:
            Log.debug("RandomCrop: %d polylines have been erased from the labels with target size %dx%d at (%d, %d)"
                      % (removals, th, tw, i, j))

        params.update({"crop_l": j, "crop_r": int(image.shape[2] - (tw + j)),
                       "crop_b": int(image.shape[1] - (th + i)), "crop_t": i})
        return cropped_image, segments, removals == 0

    def __repr__(self):
        s = '(crop={}, '.format(self.crop_portion)
        return self.__class__.__name__ + s
