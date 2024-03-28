from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class AutoMineDepthDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(AutoMineDepthDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[1193.7613309605542 / 2048, 0, 1019.4029813653916 / 2048, 0],
                           [0, 1191.5502806673906 / 1536, 794.7165379923136 / 1536, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (1024, 512)
        self.side_map = {"left": "left_cam", "right": "right_cam"}
        self.depth_side_map = {"left": "left_depth", "right": "right_depth"}

    def check_depth(self):
        return True

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:.6f}".format(frame_index) + self.img_ext
        image_path = os.path.join(
            self.data_path,
            folder,
            self.side_map[side],
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:.6f}".format(frame_index) + self.img_ext
        depth_path = os.path.join(
            self.data_path,
            folder,
            self.depth_side_map[side],
            f_str)
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256
        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt
