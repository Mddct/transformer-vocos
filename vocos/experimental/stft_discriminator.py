from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from vocos.utils import Spectrogram


def update_mask(mask_in: torch.Tensor, kernel_size: int, stride: int,
                dilation: int, padding: int):
    """
    Update mask with vectorized min pooling operation.
    mask_in: [B, 1, T_in] binary mask, 1=valid, 0=padding
    Returns mask_out: [B, 1, T_out], updated strict mask
    """
    mask_padded = F.pad(mask_in, (padding, padding), mode='constant', value=0)
    # PyTorch does not provide min_pool1d, simulate with negative max_pool1d
    mask_out = -F.max_pool1d(-mask_padded.float(),
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=0,
                             dilation=dilation)
    return (mask_out > 0.5).to(mask_in.dtype)


def get_2d_padding(kernel_size: Tuple[int, int],
                   dilation: Tuple[int, int] = (1, 1)):
    """
    Compute 'same' padding for 2D convolution with dilation.
    Returns tuple of paddings for height and width.
    """
    return (((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2)


class NormConv2d(nn.Module):
    """
    Conv2d with weight normalization wrapper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)


class DiscriminatorSTFT(nn.Module):
    """
    Single-scale STFT discriminator module.
    Computes STFT spectrogram, applies several Conv2d layers with dilation and stride,
    supports optional mask propagation through layers.
    """

    def __init__(self,
                 filters: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 max_filters: int = 1024,
                 filters_scale: int = 1,
                 kernel_size: Tuple[int, int] = (3, 9),
                 dilations: List = [1, 2, 4],
                 stride: Tuple[int, int] = (1, 2),
                 normalized: bool = True,
                 activation: str = 'LeakyReLU',
                 activation_params: dict = {'negative_slope': 0.2},
                 spec_scale_pow=0.0):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(nn, activation)(**activation_params)

        self.spec_transform = Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window_fn=torch.hann_window,
            window_norm=self.normalized,
            padding='same',
            power=None,
        )

        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(spec_channels,
                       self.filters,
                       kernel_size=kernel_size,
                       padding=get_2d_padding(kernel_size)))
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale**(i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(in_chs,
                           out_chs,
                           kernel_size=kernel_size,
                           stride=stride,
                           dilation=(dilation, 1),
                           padding=get_2d_padding(kernel_size, (dilation, 1))))
            in_chs = out_chs

        out_chs = min((filters_scale**(len(dilations) + 1)) * self.filters,
                      max_filters)
        self.convs.append(
            NormConv2d(in_chs,
                       out_chs,
                       kernel_size=(kernel_size[0], kernel_size[0]),
                       padding=get_2d_padding(
                           (kernel_size[0], kernel_size[0]))))

        self.conv_post = NormConv2d(out_chs,
                                    self.out_channels,
                                    kernel_size=(kernel_size[0],
                                                 kernel_size[0]),
                                    padding=get_2d_padding(
                                        (kernel_size[0], kernel_size[0])))
        self.spec_scale_pow = spec_scale_pow

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass through STFT discriminator.
        x: [B, C, T] input waveform
        mask: optional [B, 1, T] binary mask to indicate valid input frames
        Returns:
            logits: final output tensor
            fmap: list of feature maps from each conv layer
            mask_out: propagated mask after conv layers
        """
        fmap = []
        fmap_mask = []
        z, z_paddings = self.spec_transform(x, ~mask.bool())
        mask = ~z_paddings

        if self.spec_scale_pow != 0.0:
            z = z * torch.pow(z.abs() + 1e-6, self.spec_scale_pow)

        z = torch.cat([z.real, z.imag], dim=1)
        # z = rearrange(z, 'b c f t -> b c t f')
        z = z.transpose(-1, -2)
        z = z * mask[:, None, :, None]
        for layer in self.convs:
            stride_t = layer.conv.stride[0]
            kernel_t = layer.conv.kernel_size[0]
            dilation_t = layer.conv.dilation[0]
            padding_t = layer.conv.padding[0]

            z = layer(z)
            if mask is not None:
                mask = update_mask(mask, kernel_t, stride_t, dilation_t,
                                   padding_t)
            z = z * mask[:, None, :, None]
            z = self.activation(z)
            fmap.append(z)
            fmap_mask.append(mask)

        stride_t = self.conv_post.conv.stride[0]
        kernel_t = self.conv_post.conv.kernel_size[0]
        dilation_t = self.conv_post.conv.dilation[0]
        padding_t = self.conv_post.conv.padding[0]

        if mask is not None:
            mask = update_mask(mask, kernel_t, stride_t, dilation_t, padding_t)
        z = z * mask[:, None, :, None]
        z = self.conv_post(z)

        z_mask = mask

        return z, z_mask, fmap, fmap_mask


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-scale STFT discriminator combining several DiscriminatorSTFT
    with different FFT sizes, hop lengths, and window lengths.
    """

    def __init__(self,
                 filters: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 n_ffts: List[int] = [1024, 2048, 512],
                 hop_lengths: List[int] = [256, 512, 128],
                 win_lengths: List[int] = [1024, 2048, 512],
                 **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              n_fft=n_ffts[i],
                              win_length=win_lengths[i],
                              hop_length=hop_lengths[i],
                              **kwargs) for i in range(len(n_ffts))
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass for multi-scale discriminator.
        x: input waveform tensor [B, C, T]
        mask: optional input mask [B, 1, T]
        Returns:
            logits: list of logits from each discriminator
            fmaps: list of feature maps from each discriminator
            masks: list of propagated masks from each discriminator
        """
        logits, logit_masks, fmaps, fmaps_masks = [], [], [], []
        for disc in self.discriminators:
            logit, logit_mask, fmap, fmap_mask = disc(x, mask)
            logits.append(logit)
            logit_masks.append(logit_mask)
            fmaps.append(fmap)
            fmaps_masks.append(fmap_mask)
        return logits, logit_masks, fmaps, fmaps_masks
