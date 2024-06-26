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
from functools import partial
from torchvision import transforms
from utils.MyNode import *
from models.TIM import *
__all__ = ['spikformer']

class MLP(BaseModule):
    def __init__(self,in_features,step=10,encode_type='direct',hidden_features=None, out_features=None, drop=0.):
        super().__init__(step=10,encode_type='direct')
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MyNode(step=step,tau=2.0)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MyNode(step=step,tau=2.0)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        self.reset()

        T,B,C,N = x.shape

        x = self.fc1_conv(x.flatten(0,1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N ).contiguous() # T B C N
        x = self.fc1_lif(x.flatten(0,1)).reshape(T, B, self.c_hidden, N).contiguous() 

        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
        x = self.fc2_lif(x.flatten(0,1)).reshape(T, B, C, N ).contiguous() 
        return x


class SSA(BaseModule):
    def __init__(self,dim,step=10,encode_type='direct',num_heads=16,TIM_alpha=0.5,qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__(step=10,encode_type='direct')
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.num_heads = num_heads

        self.in_channels = dim // num_heads

        self.scale = 0.25
        
    
        self.q_conv = nn.Conv1d(dim, dim,kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MyNode(step=step,tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim,kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MyNode(step=step,tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim,kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MyNode(step=step,tau=2.0)
    
        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MyNode(step=step, tau=2.0)
        self.attn_lif = MyNode(step=step, tau=2.0, v_threshold=0.5,)

        self.proj_conv =  nn.Conv1d(dim, dim,kernel_size=1, stride=1,bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MyNode(step=step, tau=2.0,)

        self.TIM = TIM(TIM_alpha=TIM_alpha,in_channels=self.in_channels)
        
    def forward(self, x):

        self.reset()

        T,B,C,N = x.shape

        x_for_qkv = x.flatten(0, 1)  

        q_conv_out = self.q_conv(x_for_qkv)  
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out.flatten(0,1)).reshape(T, B, C ,N) 
        q = q_conv_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out= self.k_lif(k_conv_out.flatten(0,1)).reshape(T, B, C ,N)
        k = k_conv_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out.flatten(0,1)).reshape(T, B, C ,N)
        v = v_conv_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        #TIM on Q
        q = self.TIM(q) 

        #SSA 
        attn = (q @ k.transpose(-2, -1)) 
        x = (attn @ v) * self.scale 
        
        x = x.transpose(3,4).reshape(T, B, C, N).contiguous() 
        x = self.attn_lif(x.flatten(0,1)) 
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, N) 
        
        return x



class Block(nn.Module):
    def __init__(self, dim, num_heads, step=10,TIM_alpha=0.5, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)



        self.attn = SSA(dim, step=step,TIM_alpha=TIM_alpha,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,step=step, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class SPS(BaseModule):
    def __init__(self, step=10, encode_type='direct', img_size_h=64, img_size_w=64, patch_size=4, in_channels=2,
                 embed_dims=256,if_UCF=False):
        super().__init__(step=10, encode_type='direct')
        self.image_size = [img_size_h, img_size_w]

        patch_size = to_2tuple(patch_size)  
        self.patch_size = patch_size  
        self.C = in_channels  
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1] 
        self.num_patches = self.H * self.W  

        self.if_UCF = if_UCF

         
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MyNode(step=step, tau=2.0)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False) 
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)   
        self.proj_lif1 = MyNode(step=step, tau=2.0)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MyNode(step=step, tau=2.0)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MyNode(step=step, tau=2.0)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MyNode(step=step, tau=2.0)

    def forward(self, x):
        self.reset()

        # SHD
        T, B, _ = x.shape
        x = x.reshape(T,B,2,-1) # T B 2 350
        
        x = F.interpolate(x.flatten(0,1), size=256, mode='nearest').reshape(T,B,2,16,16)


        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))  # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x.flatten(0, 1)).contiguous()
        # x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x.flatten(0, 1)).contiguous()
        # x = self.maxpool1(x)
        

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x.flatten(0, 1)).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif3(x.flatten(0, 1)).contiguous()
        x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x)).reshape(T, B, -1 , H // 4,W // 4).contiguous()
        x_rpe = self.rpe_lif(x_rpe.flatten(0,1)).contiguous()
        x = x + x_rpe
        x = x.reshape(T, B, -1, (H//4)*(W//4)).contiguous()
        
       
        return x # T B C N


class Spikformer(nn.Module):
    def __init__(self, step=10,TIM_alpha=0.5,if_UCF=False,
                 img_size_h=64, img_size_w=64, patch_size=16, in_channels=2, num_classes=10,
                 embed_dims=256, num_heads=16, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=4, 
                 ):
        super().__init__()
        self.T = step  # time step
        self.num_classes = num_classes
        self.depths = depths



        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(       step=step, 
                                 if_UCF=if_UCF,
                                 img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)

        block = nn.ModuleList([Block(step=step, TIM_alpha=TIM_alpha,
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)

            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

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

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")


        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2)  
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

# Hyperparams could be adjust here

@register_model
def spikformer_shd(pretrained=False, **kwargs):
    model = Spikformer(TIM_alpha=0.5,step=10,if_UCF=False,num_classes=20,
        # img_size_h=64, img_size_w=64,
        # patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
        # in_channels=2, qkv_bias=False,
        # depths=2, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


