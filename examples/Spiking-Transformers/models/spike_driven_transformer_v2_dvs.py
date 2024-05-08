import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *
from LIFNode import MyNode  # LIFNode setting for Spiking Tranformers
from functools import partial

__all__ = ['spikformer']

'''The input shape of neuromorphic datasets in Spiking Transformer when using Braincog
are used to set to 64*64 '''

'''Here the second version of Spike-driven Transformer only open sourced the
 code for img cla '''


# Modified Operators
class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output
    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channels)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 0, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(BaseModule):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        step=8,
        encode_type='direct',
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__(step=step,encode_type=encode_type,)
        med_channels = int(expansion_ratio * dim)
        self.lif1 = MyNode(step=step,tau=2.0)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 =MyNode(step=step,tau=2.0)
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape
        x = self.lif1(x.flatten(0,1)).reshape(T,B,C,H,W).contiguous()
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x.flatten(0,1)).reshape(T,B,-1,H,W).contiguous()
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x # T B C H W


class MLP(BaseModule):
    #Linear here is subsituted by convs
    def __init__(self, in_features, step=10, encode_type='direct', hidden_features=None, out_features=None, drop=0.):
        super().__init__(step=10, encode_type='direct')
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MyNode(step=step, tau=2.0)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MyNode(step=step, tau=2.0)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape

        x = x.flatten(3)  # T B C N

        _, _, _, N = x.shape 
        
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, C, N).contiguous()
        x = self.fc1_conv(x.flatten(0, 1)) 
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()  # T B C N
        
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        
        return x  # T B C H W


# convs in SDSA V3/V4 should be substituted
class SDSA(BaseModule):
    def __init__(self, dim, step=10, encode_type='direct', num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., sr_ratio=1):
        super().__init__(step=10, encode_type='direct')
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.num_heads = num_heads
        # scale
        self.scale = 0.125

        self.head_lif = MyNode(step=step, tau=2.0) # for spike-drivens

        self.q_conv = RepConv(dim, dim, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = MyNode(step=step, tau=2.0)

        self.k_conv = RepConv(dim, dim, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MyNode(step=step, tau=2.0)

        self.v_conv = RepConv(dim, dim, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MyNode(step=step, tau=2.0)

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MyNode(step=step, tau=2.0)
        self.attn_lif = MyNode(step=step, tau=2.0, v_threshold=0.5, )

        self.proj_conv = RepConv(dim, dim, bias=False)
        self.proj_bn =  nn.BatchNorm2d(dim)
        


    def forward(self, x):
        self.reset()
        
        #different here
        T, B, C, H, W = x.shape

        N  = H * W

        x = self.head_lif(x.flatten(0,1)).reshape(T, B, C, H, W).contiguous()

        x_for_qkv = x.flatten(0, 1)  # TB C H W

        q_conv_out = self.q_conv(x_for_qkv)  # [TB] C H W
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()  # T B C H W
        q_conv_out = self.q_lif(q_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-1,-2)  # T B N C
        q = q_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-1,-2)  # T B N C
        k = k_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-1,-2)  # T B N C
        v = v_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, C, H, W)

        return x # T B C H W


class Block(nn.Module):
    def __init__(self, dim, num_heads, step=10, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SDSA(dim, step=step, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(step=step, in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # residual connection
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
    
class DownSampling(BaseModule):
    def __init__(
        self,
        step=10,
        encode_type='direct',
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
    ):
        super().__init__(step=step,
        encode_type=encode_type,)

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = MyNode(
                tau=2.0,step=step
            )

    def forward(self, x):
        self.reset()
        
        T, B, C, H, W = x.shape

        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x.flatten(0,1)).reshape(T,B,C,H,W).contiguous()
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x

class ConvBlock(BaseModule):
    def __init__(
        self,
        dim,
        step=10,
        encode_type='direct',
        mlp_ratio=4.0,
    ):
        super().__init__(step=step,
        encode_type=encode_type,)

        self.Conv = SepConv(step=step,dim=dim)
        # self.Conv = MHMC(dim=dim)

        self.lif1 = MyNode(step=step,tau=2.0)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio) 
        self.lif2 = MyNode(step=step,tau=2.0)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        self.bn2 = nn.BatchNorm2d(dim)  

    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x.flatten(0,1)))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x.flatten(0, 1)))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x
class Spikformer(BaseModule):
    def __init__(self, step=10, encode_type='direct',
                 img_size_h=64, img_size_w=64, patch_size=4, in_channels=2, num_classes=10,
                 embed_dims=256, num_heads=16, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=4,kd=False,
                 ):
        super().__init__(step=10, encode_type='direct')
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths
        
        self.block3_depths = 1
        # for membrane shortcut
        self.final_lif = MyNode(step=step,tau=2.0)
        # channel for dvs
        # 16 32 64 128 256
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        self.downsample1_1 = DownSampling(
            step=step,
            in_channels=in_channels,
            embed_dims=embed_dims // 16,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [ConvBlock(step=step,dim= embed_dims // 16, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = DownSampling(
            step=step,
            in_channels =  embed_dims // 16,
            embed_dims= embed_dims // 8,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )
        self.ConvBlock1_2 = nn.ModuleList(
            [ConvBlock(step=step,dim=embed_dims // 8, mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = DownSampling(
            step=step,
            in_channels=embed_dims // 8,
            embed_dims=embed_dims // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [ConvBlock(step=step,dim=embed_dims // 4, mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [ConvBlock(step=step,dim=embed_dims // 4, mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = DownSampling(
            step=step,
            in_channels=embed_dims // 4,
            embed_dims=embed_dims // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                Block(
                    step=step,
                    dim=embed_dims // 2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    # drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for j in range(self.block3_depths)
            ]
        )

        self.downsample4 = DownSampling(
            step=step,
            in_channels=embed_dims // 2,
            embed_dims=embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
        )

        self.block4 = nn.ModuleList(
            [
                Block(
                    step=step,
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for j in range(self.depths-self.block3_depths)
            ]
        )

        # classification head
        self.lif = MyNode(step=step,tau=2.0,)
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.kd = kd
        if self.kd:
            self.head_kd = (
                nn.Linear(embed_dims, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.apply(self._init_weights)

        # setattr(self, f"patch_embed", patch_embed)
        # setattr(self, f"block", block)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3: # attention here
            x = blk(x)

        x = self.downsample4(x) # attention here
        for blk in self.block4:
            x = blk(x)
        return x  # T,B,C,H,W
    
    def forward(self, x):
        self.reset()
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = x.flatten(3).mean(3)
        T,B,_ = x.shape
        x_lif = self.lif(x.flatten(0,1)).reshape(T,B,-1)
        x = self.head(x_lif).mean(0)
        if self.kd:
            x_kd = self.head_kd(x_lif).mean(0)
            if self.training:
                return x, x_kd
            else:
                return (x + x_kd) / 2
        return x



# Adjust ur hyperparams here
@register_model
def sd_transformer_v2_dvs(pretrained=False, **kwargs):
    model = Spikformer(step = 8,
        img_size_h=64, img_size_w=64,
        patch_size=4, embed_dims=256, num_heads=16, mlp_ratios=4,
        in_channels=2, num_classes=10, qkv_bias=False,
        depths=2, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
