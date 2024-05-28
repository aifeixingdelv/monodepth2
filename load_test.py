import torch
from networks.mamba_sys import VSSM

import copy

model = VSSM(
    patch_size=4,
    in_chans=3,
    num_classes=1,
    depths=[2, 2, 9, 2],
    dims=[96, 192, 384, 768],
    d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    norm_layer=torch.nn.LayerNorm, patch_norm=True,
    use_checkpoint=False, final_upsample="expand_first")

model_dict = model.state_dict()

pretrained_model_path = "models/mono_1024x320/mamba_unet.pth"
pretrained_dict = torch.load(pretrained_model_path)
pretrained_dict = pretrained_dict['model']
full_dict = copy.deepcopy(pretrained_dict)
for k, v in pretrained_dict.items():
    if "layers." in k:
        current_layer_num = 3 - int(k[7:8])
        current_k = "layers_up." + str(current_layer_num) + k[8:]
        full_dict.update({current_k: v})

for k in list(full_dict.keys()):
    if k in model_dict:
        if full_dict[k].shape != model_dict[k].shape:
            print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
            del full_dict[k]
print(full_dict.keys())
msg = model.load_state_dict(full_dict, strict=False)
# print(msg)

