### 任意大小。x编码.nomask
from tkinter import X
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from hydra.utils import to_absolute_path
except:
    print("Hydra not found, using relative paths")
    pass
import logging

import torch
import torch.nn as nn
try:
    from .base import BaseModel
except:
    from base import BaseModel
import mamba.models.layers as layers
from mamba.models.utils.Continues_Scan import continues_scan,rev_continues_scan
# from mamba.models.layers.combinations import *
# from mamba.models.layers.brt_modules import BlockRecurrentAttention
# from mamba.models.layers.network_swinir import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
### 任意大小。x编码.nomask
from tkinter import X
try:
    from hydra.utils import to_absolute_path
except:
    print("Hydra not found, using relative paths")
    pass
import logging

import torch
import torch.nn as nn

from .base import BaseModel
import mamba.models.layers as layers
# from mamba.models.layers.combinations import *
# from mamba.models.layers.brt_modules import BlockRecurrentAttention
# from mamba.models.layers.network_swinir import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from einops import rearrange

import torch
import torch.nn as nn
from einops import rearrange
from collections import OrderedDict

class SS3D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=1, channel_token=False):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.SSAS_orders = [
            'b c t h w',
            'b c w t h',
            'b c t w h',
            'b c h w t',
            'b c w h t',
            'b c h t w',
        ]

        self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
        )

        self.final_linear = nn.Linear(dim, dim)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        B, C, T, H, W = x.shape
        total_out = None

        for idx, SSAS in enumerate(self.SSAS_orders):
            x_perm = rearrange(x, 'b c t h w -> ' + SSAS, b=B, c=C, t=T, h=H, w=W)
            x_scan = continues_scan(x_perm)
            B_, d_model = x_scan.shape[:2]
            assert d_model == self.dim
            n_tokens = x_scan.shape[2:].numel()
            img_dims = x_scan.shape[2:]
            x_flat = x_scan.reshape(B_, d_model, n_tokens).contiguous().transpose(-1, -2)
            x_mamba = self.mamba(x_flat)
            out = x_mamba.transpose(-1, -2).reshape(B_, d_model, *img_dims)
            out = rev_continues_scan(out).contiguous()
            out = rearrange(out, SSAS + ' -> b c t h w', b=B, c=C, t=T, h=H, w=W)
            if total_out is None:
                total_out = out
            else:
                total_out += out

        B, C, T, H, W = total_out.shape
        total_out_flat = total_out.reshape(B, C, -1).transpose(1, 2)  # (B, T*H*W, C)
        total_out_flat = self.final_linear(total_out_flat)           # (B, T*H*W, C)
        total_out = total_out_flat.transpose(1, 2).contiguous().reshape(B, C, T, H, W)

        return total_out


class LACM(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.H, self.W = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.cpe1 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = GSSM(dim=dim, input_resolution= input_resolution,
                                      num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cpe2 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, L, D], where L = C_spec * H * W
        """
        B, L, D = x.shape
        H, W = self.H, self.W
        assert L % (H * W) == 0, "Input length does not match spatial resolution"
        C_spec = L // (H * W)
        x_3d = x.view(B, C_spec, H, W, D).permute(0, 4, 1, 2, 3)  # [B, D, C, H, W]
        x = x + self.cpe1(x_3d).permute(0, 2, 3, 4, 1).reshape(B, L, D)
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x_dwc = self.in_proj(x).view(B, C_spec, H, W, D).permute(0, 4, 1, 2, 3)  # [B, D, C, H, W]
        x_dwc = self.act(self.dwc(x_dwc)).permute(0, 2, 3, 4, 1).reshape(B, L, D)
        x_attn = self.attn(x_dwc)
        x = self.out_proj(x_attn * act_res)
        x = shortcut + self.drop_path(x)
        x_3d = x.view(B, C_spec, H, W, D).permute(0, 4, 1, 2, 3)
        x = x + self.cpe2(x_3d).permute(0, 2, 3, 4, 1).reshape(B, L, D)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class GSSM(nn.Module):
    r""" Linear Attention with 3D LePE and 3D RoPE (Dynamic C_spec supported).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Spatial resolution (H, W).
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.H, self.W = input_resolution  # 只包含 H, W
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.rope = RoPE2D(base=20000)
        self.mamba = SS3D(dim=dim)

    def forward(self, x):
        """
        Args:
            x: Tensor with shape [B, N, C], where N = C_spec * H * W
        """
        B, N, C = x.shape
        H, W = self.H, self.W
        assert N % (H * W) == 0, "Input length is not divisible by H * W"
        C_spec = N // (H * W)

        num_heads = self.num_heads
        head_dim = C // num_heads

        # q, k, v projection
        qk = self.qk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]
        v = x.reshape(B, C_spec, H, W, C)  # 先变成 (B, C_spec, H, W, C)
        v = v.permute(0, 4, 1, 2, 3)  # 再调换成 (B, C, C_spec, H, W)
        v = self.mamba(v)

        # Linear activation
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q_rope = self.rope(q.view(B, C_spec, H, W, C)).view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.view(B, C_spec, H, W, C)).view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        q = q.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        # Linear Attention
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(B, N, C)

        # LePE
        v_3d = v.transpose(1, 2).reshape(B, C_spec, H, W, C).permute(0, 4, 1, 2, 3)  # → [B, D, C, H, W]
        x = x + self.lepe(v_3d).permute(0, 2, 3, 4, 1).reshape(B, N, C)


        return x


class RoPE2D(nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        self.base = base

    def forward(self, x):
        """
        x: shape = (B, C_spec, H, W, dim)
        """
        B, C_spec, H, W, dim = x.shape
        num_dims = 2  # Only H and W
        assert dim % (2 * num_dims) == 0, f"dim should be divisible by 2*{num_dims}"
        k_max = dim // (2 * num_dims)

        device = x.device
        dtype = x.dtype

        # Compute theta_ks
        theta_ks = 1.0 / (self.base ** (torch.arange(k_max, device=device, dtype=dtype) / k_max))

        # Dynamic meshgrid for H, W
        grid_h = torch.arange(H, device=device).view(1, H, 1, 1)
        grid_w = torch.arange(W, device=device).view(1, 1, W, 1)

        # Compute angles for (H, W)
        angles_h = grid_h * theta_ks.view(1, 1, 1, -1)
        angles_w = grid_w * theta_ks.view(1, 1, 1, -1)

        angles = torch.cat([
            angles_h.expand(1, H, W, -1),
            angles_w.expand(1, H, W, -1)
        ], dim=-1)  # shape: (1, H, W, k_max * 2)

        # Compute cos/sin
        cos_pos = torch.cos(angles).unsqueeze(1).unsqueeze(-1)  # (1, 1, H, W, D_half, 1)
        sin_pos = torch.sin(angles).unsqueeze(1).unsqueeze(-1)  # same

        # Reshape input to complex
        x_complex = torch.view_as_complex(x.reshape(B, C_spec, H, W, -1, 2))  # [..., complex]
        pos_complex = torch.view_as_complex(torch.cat([cos_pos, sin_pos], dim=-1))

        out = pos_complex * x_complex
        return torch.view_as_real(out).flatten(-2)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LACM_Layer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            LACM(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])


    def forward(self, x):
        B, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x_flat)
            else:
                x = blk(x_flat)
        out = x.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

class Upsample(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x



class Residual_Block(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    
class Encoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = True,
                 norm_op: Union[None, Type[nn.Module]] = torch.nn.modules.instancenorm.InstanceNorm3d,
                 norm_op_kwargs: dict = {'eps': 1e-05, 'affine': True},
                 nonlin: Union[None, Type[torch.nn.Module]] = torch.nn.modules.activation.LeakyReLU,
                 nonlin_kwargs: dict = {'inplace': True},
                 depths=[2, 4, 8, 4],
                 num_heads=[2, 4, 8, 16],
                 mlp_ratio=4.,
                 qkv_bias = True,
                 drop_rate = 0.,
                 drop_path_rate = 0.1,
                 norm_layer = nn.LayerNorm,
                 ape = False,
                 use_checkpoint = False
                 ):
        super().__init__()

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True
            

        print(f"feature_map_sizes: {feature_map_sizes}")
        print(f"do_channel_token: {do_channel_token}")

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            Residual_Block(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels

        stages = []
        BasicLayers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                Residual_Block(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )

            stages.append(stage)
            input_channels = features_per_stage[s]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(n_stages):
            layer = LACM_Layer(dim=np.prod(feature_map_sizes[i_layer]) if do_channel_token[i_layer] else features_per_stage[i_layer],
                               input_resolution=(64 // (2**((i_layer+1) //2)), 64 // (2**((i_layer+1) //2))),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer)
            self.layers.append(layer)

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.layers[s](x)
            ret.append(x)

        return ret


class ResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]]):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        stages = []
        upsample_layers = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(Upsample(
                conv_op = encoder.conv_op,
                input_channels = input_features_below,
                output_channels = input_features_skip,
                pool_op_kernel_size = stride_for_upsampling,
                mode='nearest'
            ))

            stages.append(nn.Sequential(
                Residual_Block(
                    conv_op = encoder.conv_op,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    nonlin = encoder.nonlin,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    input_channels = input_features_skip,
                    output_channels = input_features_skip,
                    kernel_size = encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                ),
                *[
                    BasicBlockD(
                        conv_op = encoder.conv_op,
                        input_channels = input_features_skip,
                        output_channels = input_features_skip,
                        kernel_size = encoder.kernel_sizes[-(s + 1)],
                        stride = 1,
                        conv_bias = encoder.conv_bias,
                        norm_op = encoder.norm_op,
                        norm_op_kwargs = encoder.norm_op_kwargs,
                        nonlin = encoder.nonlin,
                        nonlin_kwargs = encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s-1] - 1)
                ]
            ))
            
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](input)
            x = x + skips[-(s+2)]
            x = self.stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))
            input = x

        seg_outputs = seg_outputs[::-1]
        r = seg_outputs
        return r
    
class LAMU(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...]=[48, 160, 224],
                 input_channels: int=1,
                 n_stages: int=6,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]]=[32, 64, 128, 256, 320, 320],
                 conv_op: Type[_ConvNd]=torch.nn.modules.conv.Conv3d,
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]]=[[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                 strides: Union[int, List[int], Tuple[int, ...]]=[[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]]=[2, 2, 2, 2, 2, 2],
                 num_classes: int=14,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]]=[2, 2, 2, 2, 2],
                 depths = [2, 2, 6, 2],
                 num_heads = [3, 6, 12, 24],
                 mlp_ratio = 4.,
                 qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False
                 ):
        super().__init__()

        self.encoder = Encoder(
            input_size,
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            depths = depths,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            drop_rate = drop_rate,
            drop_path_rate = drop_path_rate,
            norm_layer = norm_layer,
            ape = ape,
            use_checkpoint = use_checkpoint
        )

        self.decoder = ResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder)
        self.conv_first = conv_op(1, input_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_last = conv_op(num_classes,out_channels=1,kernel_size=3,stride=1,padding=1, bias=True)
        # self.norm = norm_op(features_per_stage[-1], **norm_op_kwargs)
        # self.act = nonlin(**nonlin_kwargs)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_first =self.conv_first(x)
        skips = self.encoder(x_first)
        y = self.decoder(skips)
        y = self.conv_last(y[0]+x_first).squeeze(1)
        return y



class LAMU_Net(BaseModel):
    def __init__(self,base,
        ssl=0,
        n_ssl=0,ckpt=None,):
        super().__init__(**base)
        self.layers_params = layers
        self.ssl = ssl
        self.n_ssl = n_ssl
        logger.debug(f"ssl : {self.ssl}, n_ssl : {self.n_ssl}")

        # self.init_layers()
        
        upscale = 1
        window_size = 8
        height = 64
        width = 64
        self.net = LAMU(input_channels=32,n_stages=6, input_size=[31, 64, 64],strides=[[1,1,1],[1,2,2],[1,1,1],[1,2,2],[1,1,1],[1,2,2]],num_classes=32,features_per_stage=[32, 64, 64, 128, 128, 256],n_conv_per_stage=[2,2,2,1,1,1],n_conv_per_stage_decoder=[2,2,2,1,1],
                             kernel_sizes=[[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],depths = [1, 1, 1, 2, 2, 2], num_heads = [1, 2, 2, 4, 4, 8], mlp_ratio = 4.,qkv_bias=True, drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False)

        logger.info(f"Using SSL : {self.ssl}")
        self.ckpt = ckpt
        if self.ckpt is not None:
            try:
                logger.info(f"Loading ckpt {self.ckpt!r}")
                d = torch.load(to_absolute_path(self.ckpt))
                self.load_state_dict(d["state_dict"])
            except:
                print("Could not load ckpt")
                pass

    def forward(self,x, mode=None, img_id=None, sigmas=None, ssl_idx=None, **kwargs
    ):
        x = self.net(x)
        return x
    


if __name__ == '__main__':
    model = LAMU_Net.to("cuda:1")
    x = torch.randn(1,31,64,64).to("cuda:1")
    y = model(x)
    pass
