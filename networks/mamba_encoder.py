from .mamba_sys import *

class MambaEncoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, depths=[2, 2, 9, 2], dims=96,
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, pretrained_path=None):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.pretrained_path = pretrained_path
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                # dim=dims[i_layer], #int(embed_dim * 2 ** i_layer)
                dim=int(dims[0] * 2 ** i_layer),
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

        if self.pretrained_path is not None:
            self.load_pretrained(self.pretrained_path)
        else:
            print("MambaEncoder learns from scratch")


    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, pretrained_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        pretrained_dict = pretrained_dict['model']
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        msg = self.load_state_dict(model_dict)
        print("MambaEncoder load pretrained model from {}".format(pretrained_path))
        # print(msg)

    # Encoder and Bottleneck
    def forward(self, x):
        x = self.patch_embed(x)
        feature = []
        for layer in self.layers:
            feature.append(x)
            x = layer(x)
        x = self.norm(x)  # B H W C
        feature.append(x)
        for i in range(len(feature)):
            feature[i] = feature[i].permute(0, 3, 1, 2)
        return feature


