import networks
import torchvision.transforms as transforms
import torch
from PIL import Image
import cv2
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from torch import nn
import time

torch.cuda.empty_cache()
MamEncoder = networks.MambaEncoder(
    pretrained_path="/root/autodl-tmp/monodepth2/models/mono_1024x320/vmamba_tiny_e292.pth").cuda()
MamDecoder = networks.MambaDepthDecoder().cuda()

total_params = sum(p.numel() for p in MamDecoder.parameters())
trainable_params = sum(p.numel() for p in MamDecoder.parameters() if p.requires_grad)

def format_params(num):
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)

total_params_readable = format_params(total_params)
trainable_params_readable = format_params(trainable_params)

print(f"Total parameters: {total_params_readable}")
print(f"Trainable parameters: {trainable_params_readable}")

img = Image.open("/root/autodl-tmp/monodepth2/assets/kitti.png")
transform = transforms.Compose([
    transforms.Resize((320, 1024)),  # Resize the image to 1024x320
    transforms.ToTensor()  # Convert the image to a torch.Tensor
])
img = transform(img).unsqueeze(0).cuda()
feature = MamEncoder(img)
output = MamDecoder(feature)
pred_disp = output[("disp", 0)].squeeze(0).squeeze(0).cpu().detach().numpy()
print(pred_disp.shape)
disp_resized = cv2.resize(pred_disp, (1216, 352))
vmax = np.percentile(disp_resized, 95)
normalizer = mpl.colors.Normalize(vmin=disp_resized.min(), vmax=vmax)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
colormapped_im = (mapper.to_rgba(disp_resized)[:, :, :3] * 255).astype(np.uint8)
im = pil.fromarray(colormapped_im)
im.save("assets/depth.png")


