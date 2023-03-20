from transformers import ViTModel
from transformers.models.vit.modeling_vit import ViTEncoder
from transformers  import VisionEncoderDecoderModel

import torch.nn as nn
import torch

class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=[32, 320],
                 in_channels=3,
                 embed_dim=768,
                 sub_num=3,
                 patch_size=[4, 4],
                 mode='pope'):
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        ))
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=False),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=False),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=False))
        elif mode == 'linear':
            self.proj = nn.Conv2d(
                1, embed_dim, kernel_size=tuple(patch_size), stride=tuple(patch_size))
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).permute((0, 2, 1))
        return x


class FastTrOCR(nn.Module):
    def __init__(self, config):
        super(FastTrOCR, self).__init__()
        self.embeddings = PatchEmbed(embed_dim=config.hidden_size)
        self.encoder = ViTEncoder(config)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self,x):
        x  = self.embeddings(x)
        encoder_outputs = self.encoder(x)
        encoder_hidden_states = encoder_outputs[0]
        output = self.fc(encoder_hidden_states[:,1:,:])
        return output
