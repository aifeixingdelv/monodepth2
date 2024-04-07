from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from automine_options import MonodepthOptions
import datasets
import networks
import matplotlib.cm
from PIL import Image

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def colorize(value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"):
    if value.ndim > 2:
        return value
    invalid_mask = value == -1
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    # set color
    cmapper = matplotlib.colormaps[cmap]
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[..., :3]
    return img


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    splits_dir = os.path.join(opt.data_path, "AutoMine_Depth")
    filenames = readlines(os.path.join(splits_dir, "AutoMine_Depth_val.txt"))

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_best_weights_folder = os.path.expanduser(opt.load_best_weights_folder)

        assert os.path.isdir(opt.load_best_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_best_weights_folder)

        print("-> Loading weights from {}".format(opt.load_best_weights_folder))

        encoder_path = os.path.join(opt.load_best_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_best_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.AutoMineDepthDataset(opt.data_path, filenames,
                                                encoder_dict['height'], encoder_dict['width'],
                                                [0], 4, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, 8, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=True)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []
        gt_depths = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                gt_depths.append(data["depth_gt"].squeeze(1).cpu().numpy())

        pred_disps = np.concatenate(pred_disps)
        gt_depths = np.concatenate(gt_depths)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_best_weights_folder, "disps_val.npy")
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.save_pred_depths:
        save_depths_dir = os.path.join(opt.load_best_weights_folder, "depths_val")
        print("-> Saving predicted depth_maps to {}".format(save_depths_dir))
        if not os.path.exists(save_depths_dir):
            os.makedirs(save_depths_dir)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    if opt.eval_stereo:
        print("-> Evaluating Stereo - disable median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("-> Evaluating Mono - enable median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        frame_id_str = filenames[i].split()[1]

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        # _, pred_depth = disp_to_depth(pred_disp, 0.1, 100.0)
        pred_depth = 1 / pred_disp
        pred_depth_color = pred_depth.copy()

        mask = gt_depth > 0
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
            if opt.save_pred_depths:
                pred_depth_color *= ratio
                pred_depth_color[pred_depth_color < MIN_DEPTH] = MIN_DEPTH
                pred_depth_color[pred_depth_color > MAX_DEPTH] = MAX_DEPTH
                color_depth_map = Image.fromarray(colorize(pred_depth_color, MIN_DEPTH, MAX_DEPTH), mode='RGB')
                color_depth_map.save(os.path.join(save_depths_dir, "{}.png".format(frame_id_str)))

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        max = np.max(ratios)
        min = np.min(ratios)
    mean_errors = np.array(errors).mean(0)
    depth_metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    if opt.save_pred_metrics:
        output_path = os.path.join(
            opt.load_best_weights_folder, "metrics_val.txt")
        print("-> Saving predicted metrics to ", output_path)
        with open(output_path, 'w') as f:
            for name, value in zip(depth_metric_names, mean_errors.tolist()):
                f.write("{}:{:<8.3f}\n".format(name, value))
            f.write("\n")
            if not opt.disable_median_scaling:
                f.write("Scaling ratios \n--med: {:0.3f}\n--std: {:0.3f}\n--max: {:0.3f}\n--min: {:0.3f}"
                        .format(med, np.std(ratios / med), max, min))
                f.write("\n")
            f.close()
    print("-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
