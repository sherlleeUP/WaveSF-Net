import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from einops import rearrange
import pywt

# Factorized Coordinate-Aware Multi-Scale Module************************************************************************
class Flatten(nn.Module):

    def __init__(self, start_dim=1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim

    def forward(self, inp):
        return torch.flatten(inp, start_dim=self.start_dim)


class ChannelGate(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16, use_bias=False):
        super(ChannelGate, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(start_dim=1),
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        attention_weights = self.module(input_tensor).unsqueeze(2).unsqueeze(3)

        return input_tensor * attention_weights


class FCASM(nn.Module):

    def __init__(self, channels, factor=8, gate_reduction_ratio=4):
        super(FCASM, self).__init__()
        self.groups = factor

        group_channels = channels // self.groups

        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(group_channels, group_channels)

        self.conv_hw = nn.Sequential(
            nn.Conv2d(group_channels, group_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(group_channels, group_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1)

        self.conv5x5_dilated = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=2, dilation=2)

        self.conv7x7_dilated = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=3, dilation=3)

        self.gate1 = ChannelGate(group_channels, gate_reduction_ratio)
        self.gate3 = ChannelGate(group_channels, gate_reduction_ratio)
        self.gate5_dilated = ChannelGate(group_channels, gate_reduction_ratio)
        self.gate7_dilated = ChannelGate(group_channels, gate_reduction_ratio)
        self.num_scales = 4
        self.fusion_conv = nn.Conv2d(self.num_scales, 1, kernel_size=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.size()

        group_x = x.reshape(b * self.groups, c // self.groups, h, w)
        group_channels = c // self.groups

        pooled_h = self.pool_h(group_x)

        _pooled_w_before_permute = self.pool_w(group_x)

        permuted_pooled_w = _pooled_w_before_permute.permute(0, 1, 3, 2)
        hw_cat = torch.cat([pooled_h, permuted_pooled_w], dim=2)

        spatial_att_maps = self.conv_hw(hw_cat)

        att_h, att_w = torch.split(spatial_att_maps, [h, w], dim=2)
        x1 = self.gn(group_x * att_h * att_w.permute(0, 1, 3, 2))

        x2_1 = self.gate1(self.conv1x1(group_x))
        x2_3 = self.gate3(self.conv3x3(group_x))
        x2_5_dilated = self.gate5_dilated(self.conv5x5_dilated(group_x))
        x2_7_dilated = self.gate7_dilated(self.conv7x7_dilated(group_x))

        x22_1 = self.sigmoid(x2_1)
        x22_3 = self.sigmoid(x2_3)
        x22_5_dilated = self.sigmoid(x2_5_dilated)
        x22_7_dilated = self.sigmoid(x2_7_dilated)

        x1_pooled = self.agp(x1)
        x11 = self.softmax(x1_pooled.view(b * self.groups, group_channels, -1).permute(0, 2, 1))

        x22_1_flat = x22_1.view(b * self.groups, group_channels, -1)
        x22_3_flat = x22_3.view(b * self.groups, group_channels, -1)
        x22_5_dilated_flat = x22_5_dilated.view(b * self.groups, group_channels, -1)
        x22_7_dilated_flat = x22_7_dilated.view(b * self.groups, group_channels, -1)

        weight_1 = torch.matmul(x11, x22_1_flat)
        weight_3 = torch.matmul(x11, x22_3_flat)
        weight_5 = torch.matmul(x11, x22_5_dilated_flat)
        weight_7 = torch.matmul(x11, x22_7_dilated_flat)

        weight_1_reshaped = weight_1.reshape(b * self.groups, 1, h, w)
        weight_3_reshaped = weight_3.reshape(b * self.groups, 1, h, w)
        weight_5_reshaped = weight_5.reshape(b * self.groups, 1, h, w)
        weight_7_reshaped = weight_7.reshape(b * self.groups, 1, h, w)

        stacked_weights = torch.cat([
            weight_1_reshaped,
            weight_3_reshaped,
            weight_5_reshaped,
            weight_7_reshaped
        ], dim=1)

        combined_weights = self.fusion_conv(stacked_weights)
        combined_weights = self.sigmoid(combined_weights)

        output = (group_x * combined_weights).reshape(b, c, h, w)
        return output


# Dynanic Aware Fusion block***********************************************************************************************************
class DAF(nn.Module):
    def __init__(self, in_channels):
        super(DAF, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        target_h, target_w = x.shape[2:]
        current_h, current_w = y.shape[2:]
        if current_h != target_h or current_w != target_w:
            y = F.interpolate(
                y,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=True
            )
        initial = x + y
        pattn2 = self.relu(initial)
        pattn2 = self.psi(pattn2)
        result = initial + pattn2 * x + (1 - pattn2) * y
        return result


# RefineModule block*****************************************************************************************************
class RefineModule(nn.Module):
    def __init__(self, dim):
        super(RefineModule, self).__init__()
        self.dim = dim
        self.fcsam = FCASM(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f = self.fcsam(x)
        x_out = self.conv(f)
        return x_out


# ConvNext***************************************************************************************************************
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, sp=False,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.sp = sp

        if self.sp:  # for 512x512 input
            self.sfs = nn.ModuleList(
                [FCASM(96),
                 FCASM(192),
                 FCASM(384)]
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.sp:
                x = x + self.sfs[i](x)
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        """
        torch.Size([1, 96, 128, 128])
        torch.Size([1, 192, 64, 64])
        torch.Size([1, 384, 32, 32])
        torch.Size([1, 768, 16, 16])
        """
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_encoder(pretrained=False, in_22k=False, sp=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], sp=sp, **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


# Scale reshape**********************************************************************************************************
class LRDU(nn.Module):
    def __init__(self, in_c, factor):
        super(LRDU, self).__init__()

        self.up_factor = factor
        self.factor1 = factor * factor // 2
        self.factor2 = factor * factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1 * in_c, (1, 7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1 * in_c, self.factor2 * in_c, (7, 1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# ConvBlock**************************************************************************************************************
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            LRDU(ch_in, 2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Decoder******************************************************************************************************************
class decoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(decoder, self).__init__()
        self.up_conv = nn.Sequential(
            LRDU(ch_in, 2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up_conv(x)
        x = self.conv_block(x)
        return x


# Wavelet related modules*********************************************************************************************************

class DWTLevel(nn.Module):

    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

    def forward(self, x_input):
        # x_input: (B, C, H, W)
        batch_size, num_channels, height, width = x_input.shape

        coeffs_ll_batch, coeffs_lh_batch, coeffs_hl_batch, coeffs_hh_batch = [], [], [], []

        for i in range(batch_size):
            img_single_batch = x_input[i]  # (C, H, W)
            channel_lls, channel_lhs, channel_hls, channel_hhs = [], [], [], []
            for ch in range(num_channels):
                coeffs = pywt.dwt2(img_single_batch[ch].detach().cpu().numpy(),
                                   self.wavelet, mode='periodization')
                LL, (LH, HL, HH) = coeffs

                channel_lls.append(torch.from_numpy(LL).unsqueeze(0).to(x_input.device).float())
                channel_lhs.append(torch.from_numpy(LH).unsqueeze(0).to(x_input.device).float())
                channel_hls.append(torch.from_numpy(HL).unsqueeze(0).to(x_input.device).float())
                channel_hhs.append(torch.from_numpy(HH).unsqueeze(0).to(x_input.device).float())

            coeffs_ll_batch.append(torch.cat(channel_lls, dim=0).unsqueeze(0))
            coeffs_lh_batch.append(torch.cat(channel_lhs, dim=0).unsqueeze(0))
            coeffs_hl_batch.append(torch.cat(channel_hls, dim=0).unsqueeze(0))
            coeffs_hh_batch.append(torch.cat(channel_hhs, dim=0).unsqueeze(0))

        ll = torch.cat(coeffs_ll_batch, dim=0)
        lh = torch.cat(coeffs_lh_batch, dim=0)
        hl = torch.cat(coeffs_hl_batch, dim=0)
        hh = torch.cat(coeffs_hh_batch, dim=0)

        return ll, lh, hl, hh


class MultiLevelWaveletTransform(nn.Module):

    def __init__(self, wavelet='haar', num_levels=4, input_channels=3):
        super().__init__()
        self.num_levels = num_levels
        self.dwt_levels = nn.ModuleList([DWTLevel(wavelet) for _ in range(num_levels)])

    def forward(self, x):
        # x: (B, C_in, H, W)
        low_freq_outputs = []
        high_freq_outputs = []

        current_approx = x

        for i in range(self.num_levels):
            ll, lh, hl, hh = self.dwt_levels[i](current_approx)

            low_freq_outputs.append(ll)
            concatenated_hf = torch.cat((lh, hl, hh), dim=1)
            high_freq_outputs.append(concatenated_hf)

            current_approx = ll

        return low_freq_outputs, high_freq_outputs


class WaveletGuidedFusion(nn.Module):
    def __init__(self, encoder_channels, wavelet_low_in_channels, wavelet_high_in_channels,
                 is_deep_layer, intermediate_wavelet_channels_ratio=0.25):
        super().__init__()
        self.is_deep_layer = is_deep_layer

        intermediate_wavelet_channels = max(1, int(encoder_channels * intermediate_wavelet_channels_ratio))

        self.conv_low = nn.Sequential(
            nn.Conv2d(wavelet_low_in_channels, intermediate_wavelet_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_wavelet_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(wavelet_high_in_channels, intermediate_wavelet_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_wavelet_channels),
            nn.ReLU(inplace=True)
        )

        self.gate_low = nn.Parameter(torch.randn(1))
        self.gate_high = nn.Parameter(torch.randn(1))

        if self.is_deep_layer:
            nn.init.constant_(self.gate_low, 0.7)
            nn.init.constant_(self.gate_high, 0.3)
        else:
            nn.init.constant_(self.gate_low, 0.3)
            nn.init.constant_(self.gate_high, 0.7)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(encoder_channels + intermediate_wavelet_channels,
                      encoder_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder_feature, wavelet_low_feature, wavelet_high_feature):
        adapted_low = self.conv_low(wavelet_low_feature)
        adapted_high = self.conv_high(wavelet_high_feature)

        g_low = torch.sigmoid(self.gate_low)
        g_high = torch.sigmoid(self.gate_high)

        combined_wavelet_info = g_low * adapted_low + g_high * adapted_high
        target_h, target_w = encoder_feature.shape[2:]

        current_h, current_w = combined_wavelet_info.shape[2:]

        if current_h != target_h or current_w != target_w:
            combined_wavelet_info = F.interpolate(
                combined_wavelet_info,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=True
            )
        x_cat = torch.cat([encoder_feature, combined_wavelet_info], dim=1)

        fused_output = self.fusion_conv(x_cat)

        return encoder_feature + fused_output


# Model_ConvNext*********************************************************************************************************
class Model_ConvNextDWT(nn.Module):
    def __init__(self, n_class=2, pretrained=True):
        super(Model_ConvNextDWT, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        self.dec3 = decoder(ch_in=768, ch_out=384)
        self.dec2 = decoder(ch_in=384, ch_out=192)
        self.dec1 = decoder(ch_in=192, ch_out=96)
        self.Up = LRDU(96, 4)
        self.convout = nn.Conv2d(96, n_class, kernel_size=1, stride=1, padding=0)
        self.F3 = DAF(384)
        self.F2 = DAF(192)
        self.F1 = DAF(96)
        self.backbone = convnext_encoder(pretrained, True)
        self.fcasm = FCASM(768)
        self.rm1 = RefineModule(96)
        self.rm2 = RefineModule(192)
        self.rm3 = RefineModule(384)
        self.wavelet_transform = MultiLevelWaveletTransform(wavelet='haar', num_levels=4,
                                                            input_channels=self.in_channel)
        self.wavelet_fusion_enc1 = WaveletGuidedFusion(encoder_channels=96,
                                                       wavelet_low_in_channels=self.in_channel,
                                                       wavelet_high_in_channels=3 * self.in_channel,
                                                       is_deep_layer=False)
        self.wavelet_fusion_enc2 = WaveletGuidedFusion(encoder_channels=192,
                                                       wavelet_low_in_channels=self.in_channel,
                                                       wavelet_high_in_channels=3 * self.in_channel,
                                                       is_deep_layer=False)
        self.wavelet_fusion_enc3 = WaveletGuidedFusion(encoder_channels=384,
                                                       wavelet_low_in_channels=self.in_channel,
                                                       wavelet_high_in_channels=3 * self.in_channel,
                                                       is_deep_layer=True)
        self.wavelet_fusion_enc4 = WaveletGuidedFusion(encoder_channels=768,
                                                       wavelet_low_in_channels=self.in_channel,
                                                       wavelet_high_in_channels=3 * self.in_channel,
                                                       is_deep_layer=True)

        self.pool_wavelet_l1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_wavelet_l2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_wavelet_l3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_wavelet_l4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        all_ll_coeffs, all_hf_coeffs = self.wavelet_transform(x)
        enc_l1, enc_l2, enc_l3, enc_l4 = self.backbone(x)
        """
        torch.Size([1, 96, 128, 128])
        torch.Size([1, 192, 64, 64])
        torch.Size([1, 384, 32, 32])
        torch.Size([1, 768, 16, 16])
        """

        fused_l1 = self.wavelet_fusion_enc1(
            enc_l1,
            self.pool_wavelet_l1(all_ll_coeffs[0]),
            self.pool_wavelet_l1(all_hf_coeffs[0])
        )
        fused_l2 = self.wavelet_fusion_enc2(
            enc_l2,
            self.pool_wavelet_l2(all_ll_coeffs[1]),
            self.pool_wavelet_l2(all_hf_coeffs[1])
        )
        fused_l3 = self.wavelet_fusion_enc3(
            enc_l3,
            self.pool_wavelet_l3(all_ll_coeffs[2]),
            self.pool_wavelet_l3(all_hf_coeffs[2])
        )
        fused_l4 = self.wavelet_fusion_enc4(
            enc_l4,
            self.pool_wavelet_l4(all_ll_coeffs[3]),
            self.pool_wavelet_l4(all_hf_coeffs[3])
        )
        processed_l4 = self.fcasm(fused_l4)

        d3 = self.dec3(processed_l4)
        f3 = self.F3(fused_l3, d3)
        f3 = self.rm3(f3) + f3

        d2 = self.dec2(f3)
        f2 = self.F2(fused_l2, d2)
        f2 = self.rm2(f2) + f2

        d1 = self.dec1(f2)
        f1 = self.F1(fused_l1, d1)
        f1 = self.rm1(f1) + f1

        F_out = self.convout(self.Up(f1))

        return F_out


if __name__ == "__main__":

    model = Model_ConvNextDWT(2)
    img = torch.rand((1, 3, 512, 512))
    output = model(img)
    print(output.shape)

    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table

        flops = FlopCountAnalysis(model, img)
        print("FLOPs: %.4f G" % (flops.total() / 1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6))