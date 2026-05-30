# -*- coding: utf-8 -*-
"""InternImage backbone for MMDetection v3.x.

本檔把 OpenGVLab InternImage 的偵測版 backbone 移植成 MMDet v3 介面，並
**預設使用純 PyTorch 版的 DCNv3（`DCNv3_pytorch`）核心算子**，完全不需要編譯
任何 CUDA 擴充。原因：

  1. 訓練機是 PRO 6000（Blackwell）+ torch 2.12，官方 DCNv3 的舊版 CUDA kernel
     對這麼新的 CUDA / torch 幾乎一定編不過；
  2. Kaggle 2×T4 離線環境編譯自訂 CUDA ops 風險高、易超時。

純 PyTorch 版用 `F.grid_sample` 實作可變形採樣，數值上與 CUDA 版等價（僅較慢），
在工作站與 Kaggle 都能直接跑。若日後要追求極致速度，可在 setup_env.sh 內編譯
官方 ops_dcnv3 並把下方 `core_op='DCNv3'`，本檔會自動嘗試 import CUDA 版。

實作忠實參考 OpenGVLab/InternImage（detection/mmdet_custom/models/backbones/
intern_image.py 與 ops_dcnv3/modules/dcnv3.py），僅改成 MMDet v3 的
`mmdet.registry.MODELS` 註冊與 `mmengine.model.BaseModule` 基底。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.init import constant_, xavier_uniform_

from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmdet.registry import MODELS


# ---------------------------------------------------------------------------
# 基礎工具：channels_first <-> channels_last 轉換、norm / act 建構、DropPath
# InternImage 內部特徵以 channels_last (N, H, W, C) 流動，方便 DCNv3 與 MLP 用
# nn.Linear 在最後一維運算。
# ---------------------------------------------------------------------------
class to_channels_first(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim, norm_layer, in_format='channels_last',
                     out_format='channels_last', eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f'build_norm_layer 不支援 {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer 不支援 {act_layer}')


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# DCNv3 純 PyTorch 核心：可變形卷積的取樣 + 加權求和（與官方等價）
# ---------------------------------------------------------------------------
def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w,
                          dilation_h, dilation_w, pad_h=0, pad_w=0,
                          stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out, dtype=torch.float32, device=device),
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out, dtype=torch.float32, device=device),
        indexing='ij')
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)
    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h,
                             dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w,
            kernel_w, dtype=torch.float32, device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h,
            kernel_h, dtype=torch.float32, device=device),
        indexing='ij')
    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)
    return grid


def dcnv3_core_pytorch(input, offset, mask, kernel_h, kernel_w, stride_h,
                       stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                       group_channels, offset_scale):
    # input: (N, H, W, C)。先 pad 再用 grid_sample 做雙線性可變形採樣。
    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(input.shape, input.device, kernel_h, kernel_w,
                                dilation_h, dilation_w, pad_h, pad_w, stride_h,
                                stride_w)
    grid = _generate_dilation_grids(input.shape, kernel_h, kernel_w, dilation_h,
                                    dilation_w, group, input.device)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).repeat(
        1, 1, 1, group * kernel_h * kernel_w).to(input.device)

    sampling_locations = (ref + grid * offset_scale).repeat(
        N_, 1, 1, 1, 1).flatten(3, 4) + \
        offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    input_ = input.view(N_, H_in * W_in, group * group_channels).transpose(
        1, 2).reshape(N_ * group, group_channels, H_in, W_in)
    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_,
                                         2).transpose(1, 2).flatten(0, 1)
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros',
        align_corners=False)
    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2).reshape(
        N_ * group, 1, H_out * W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(
        N_, group * group_channels, H_out * W_out)
    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()


class DCNv3_pytorch(nn.Module):
    """純 PyTorch 版 DCNv3（無 CUDA 依賴）。"""

    def __init__(self, channels=64, kernel_size=3, dw_kernel_size=None,
                 stride=1, pad=1, dilation=1, group=4, offset_scale=1.0,
                 act_layer='GELU', norm_layer='LN', center_feature_scale=False):
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels({channels}) 必須能被 group({group}) 整除')
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        # 深度可分離卷積負責萃取偏移/mask 所需的局部上下文
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, groups=channels),
            build_norm_layer(channels, norm_layer, 'channels_first',
                             'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(channels, group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(channels, group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        # input: (N, H, W, C)
        N, H, W, _ = input.shape
        x = self.input_proj(input)
        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)  # -> (N, H, W, C)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)
        x = dcnv3_core_pytorch(
            x, offset, mask, self.kernel_size, self.kernel_size, self.stride,
            self.stride, self.pad, self.pad, self.dilation, self.dilation,
            self.group, self.group_channels, self.offset_scale)
        x = self.output_proj(x)
        return x


def _resolve_core_op(core_op):
    """回傳核心算子類別。core_op='DCNv3' 時嘗試載入已編譯的 CUDA 版，
    失敗則自動退回純 PyTorch 版並提示。"""
    if core_op == 'DCNv3_pytorch':
        return DCNv3_pytorch
    if core_op == 'DCNv3':
        try:
            from ops_dcnv3.modules.dcnv3 import DCNv3  # 需先編譯 ops_dcnv3
            return DCNv3
        except Exception as e:  # noqa: BLE001
            MMLogger.get_current_instance().warning(
                f'載入 CUDA 版 DCNv3 失敗（{e}），自動退回純 PyTorch 版 '
                'DCNv3_pytorch。')
            return DCNv3_pytorch
    raise NotImplementedError(f'未知 core_op: {core_op}')


# ---------------------------------------------------------------------------
# 模組積木：Stem / Downsample / MLP / InternImageLayer / InternImageBlock
# ---------------------------------------------------------------------------
class StemLayer(nn.Module):
    """兩層 stride-2 卷積，輸出降到 1/4 解析度，回傳 channels_last。"""

    def __init__(self, in_chans=3, out_chans=96, act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, kernel_size=3,
                               stride=2, padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, kernel_size=3,
                               stride=2, padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    """stage 之間的 2x 下採樣，通道翻倍。輸入/輸出皆 channels_last。"""

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2,
                              padding=1, bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer, 'channels_first',
                                     'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer='GELU', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    """單個 InternImage block：DCNv3 + MLP，含 layer-scale 與 DropPath。"""

    def __init__(self, core_op, channels, groups, mlp_ratio=4., drop=0.,
                 drop_path=0., act_layer='GELU', norm_layer='LN',
                 post_norm=False, layer_scale=None, offset_scale=1.0,
                 with_cp=False):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.with_cp = with_cp
        self.post_norm = post_norm

        self.norm1 = build_norm_layer(channels, 'LN')
        self.dcn = core_op(channels=channels, kernel_size=3, stride=1, pad=1,
                           dilation=1, group=groups, offset_scale=offset_scale,
                           act_layer=act_layer, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(channels, int(channels * mlp_ratio),
                            act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True)
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True)

    def forward(self, x):
        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        # 梯度檢查點：用時間換記憶體，T4/單卡訓練大解析度時很關鍵
        if self.with_cp and x.requires_grad:
            # torch 2.9+ 要求顯式傳 use_reentrant；用 False（官方建議）
            x = checkpoint.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    """一個 stage：堆疊 depth 個 InternImageLayer，可選結尾 downsample。"""

    def __init__(self, core_op, channels, depth, groups, downsample=True,
                 mlp_ratio=4., drop=0., drop_path=0., act_layer='GELU',
                 norm_layer='LN', post_norm=False, offset_scale=1.0,
                 layer_scale=None, with_cp=False):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op, channels=channels, groups=groups,
                mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list)
                else drop_path,
                act_layer=act_layer, norm_layer=norm_layer, post_norm=post_norm,
                layer_scale=layer_scale, offset_scale=offset_scale,
                with_cp=with_cp) for i in range(depth)
        ])
        if not self.post_norm:
            self.norm = build_norm_layer(channels, 'LN')
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for blk in self.blocks:
            x = blk(x)
        if not self.post_norm:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)
        if return_wo_downsample:
            return x, x_
        return x


@MODELS.register_module()
class InternImage(BaseModule):
    """InternImage backbone（MMDet v3 版）。

    InternImage-T 預設參數：channels=64, depths=[4,4,18,4],
    groups=[4,8,16,32]，四個 stage 輸出通道為 [64,128,256,512]，對應 FPN。
    """

    def __init__(self,
                 core_op='DCNv3_pytorch',
                 channels=64,
                 depths=(4, 4, 18, 4),
                 groups=(4, 8, 16, 32),
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False,
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.core_op = core_op
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.out_indices = out_indices
        core_op_cls = _resolve_core_op(core_op)

        self.patch_embed = StemLayer(
            in_chans=3, out_chans=channels, act_layer=act_layer,
            norm_layer='BN')
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = InternImageBlock(
                core_op=core_op_cls,
                channels=int(channels * 2 ** i),
                depth=depths[i],
                groups=groups[i],
                downsample=(i < self.num_levels - 1),
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                offset_scale=offset_scale,
                layer_scale=layer_scale,
                with_cp=with_cp)
            self.levels.append(level)

    def init_weights(self):
        # 若 init_cfg 指定 Pretrained checkpoint，交給 BaseModule（strict=False）
        # 載入；否則對 Linear / LayerNorm 做標準初始化。
        if self.init_cfg is not None:
            super().init_weights()
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)        # (N, H/4, W/4, C)
        x = self.pos_drop(x)
        seq_out = []
        for level_idx, level in enumerate(self.levels):
            x, x_ = level(x, return_wo_downsample=True)
            if level_idx in self.out_indices:
                # 轉回 channels_first 給 FPN（mmdet neck 預期 NCHW）
                seq_out.append(x_.permute(0, 3, 1, 2).contiguous())
        return tuple(seq_out)
