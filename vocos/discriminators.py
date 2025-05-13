from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn, strided
from torch.nn import Conv2d
from wenet.utils.mask import make_pad_mask

from vocos.cqt import CQT
from vocos.loss import cal_mean_with_mask

try:
    from torch.nn.utils.parametrizations import weight_norm
except:
    from torch.nn.utils import weight_norm

from torchaudio.transforms import Spectrogram

from vocos.utils import frame_paddings


class SequenceMultiPeriodDiscriminator(nn.Module):

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [SequenceDiscriminatorP(period=p) for p in periods])

    def forward(
        self,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        logits = []
        logits_masks = []
        fmaps = []
        fmaps_masks = []
        for d in self.discriminators:
            logit, logit_masks, fmap, fmap_masks = d(x=y, mask=mask)
            logits.append(logit)
            logits_masks.append(logit_masks)
            fmaps.append(fmap)
            fmaps_masks.append(fmap_masks)

        return logits, logits_masks, fmaps, fmaps_masks


class SequenceDiscriminatorP(nn.Module):

    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(
                Conv2d(in_channels,
                       32, (kernel_size, 1), (stride, 1),
                       padding=(kernel_size // 2, 0))),
            weight_norm(
                Conv2d(32,
                       128, (kernel_size, 1), (stride, 1),
                       padding=(kernel_size // 2, 0))),
            weight_norm(
                Conv2d(128,
                       512, (kernel_size, 1), (stride, 1),
                       padding=(kernel_size // 2, 0))),
            weight_norm(
                Conv2d(512,
                       1024, (kernel_size, 1), (stride, 1),
                       padding=(kernel_size // 2, 0))),
            weight_norm(
                Conv2d(1024,
                       1024, (kernel_size, 1), (1, 1),
                       padding=(kernel_size // 2, 0))),
        ])

        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1,
                                            padding=(1, 0)))
        self.lrelu_slope = lrelu_slope

    def _apply_mask(self, x: torch.Tensor,
                    current_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if current_mask is not None:
            x = x * current_mask.float()
        return x

    def _update_mask(self, mask: Optional[torch.Tensor],
                     layer: Conv2d) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        kernel_size = (layer.kernel_size[0], 1)
        stride = (layer.stride[0], 1)
        padding = (layer.padding[0], 0)
        mask = -F.max_pool2d(-mask.float(), kernel_size, stride, padding)
        return mask > 0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor],
               List[torch.Tensor]]:
        x = x.unsqueeze(1)
        fmap = []
        b, c, t = x.shape

        # Handle padding and reshaping
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            if mask is not None:
                mask = F.pad(mask, (0, n_pad), value=False)
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)
        current_mask = mask.view(b, 1, t // self.period,
                                 self.period) if mask is not None else None

        masks = []
        # Process through layers
        for i, conv in enumerate(self.convs):
            x = conv(x)
            current_mask = self._update_mask(current_mask, conv)
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self._apply_mask(x, current_mask)
            if i > 0:
                fmap.append(x)
                masks.append(current_mask)

        x = self.conv_post(x)
        current_mask = self._update_mask(current_mask, self.conv_post)
        x = self._apply_mask(x, current_mask)
        fmap.append(x)
        masks.append(current_mask)
        x = torch.flatten(x, 1, -1)
        x_mask = masks[-1].flatten(1, -1)
        return x, x_mask, fmap, masks


class SequenceMultiResolutionDiscriminator(nn.Module):

    def __init__(self, fft_sizes: Tuple[int, ...] = (2048, 1920, 1024, 512)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [SequenceDiscriminatorR(window_length=w) for w in fft_sizes])

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               List[List[torch.Tensor]], List[torch.Tensor]]:
        logits = []
        fmaps = []
        fmaps_mask = []
        logit_masks = []
        for d in self.discriminators:
            logit, logit_mask, fmap, fmap_mask = d(x=y, mask=mask)
            logits.append(logit)
            fmaps.append(fmap)
            fmaps_mask.append(fmap_mask)
            logit_masks.append(logit_mask)
        return logits, logit_masks, fmaps, fmaps_mask


class SequenceDiscriminatorR(nn.Module):

    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float],
                     ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75),
                             (0.75, 1.0)),
        normalize_volume: bool = False,
    ):
        super().__init__()
        self.normalize_volume = normalize_volume
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(n_fft=window_length,
                                   hop_length=int(window_length * hop_factor),
                                   win_length=window_length,
                                   power=None,
                                   center=False)
        n_fft = window_length // 2 + 1
        self.bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.band_convs = nn.ModuleList(
            [self._create_band_convs(channels) for _ in self.bands])
        self.conv_post = weight_norm(
            nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def _create_band_convs(self, channels: int) -> nn.ModuleList:
        return nn.ModuleList([
            weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1),
                                  padding=(1, 4))),
            weight_norm(
                nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(
                nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(
                nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:

        if self.normalize_volume:
            x_mean = cal_mean_with_mask(x, mask, dim=-1)
            x = x - x_mean
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        spec = self.spec_fn(x)
        mask = mask.float()
        out_paddings = frame_paddings(1 - mask.unsqueeze(1),
                                      frame_size=self.spec_fn.n_fft,
                                      hop_size=self.spec_fn.hop_length)
        mask = 1 - out_paddings
        mask = mask.squeeze(1)
        fmaps_mask = mask.unsqueeze(1).unsqueeze(-1)

        spec = torch.view_as_real(spec)
        spec = spec.transpose(1, 3)  # "b f t c -> b c t f"

        x_bands = [spec[..., b[0]:b[1]] for b in self.bands]

        fmaps = []

        for band, conv_stack in zip(x_bands, self.band_convs):
            x_band = band

            for i, conv in enumerate(conv_stack):
                x_band = conv(x_band)
                x_band = F.leaky_relu(x_band, 0.1)
                x_band = x_band * fmaps_mask
                if i > 0:
                    fmaps.append(x_band)

        x = torch.cat(fmaps, dim=-1)
        x = self.conv_post(x)
        fmaps.append(x)
        mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, x.shape[-1])
        x = torch.flatten(x, 1, -1)
        mask = torch.flatten(mask, 1, -1)
        return x, mask, fmaps, fmaps_mask


class SequenceMultiScaleSubbandCQTDiscriminator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SequenceDiscriminatorCQT(
                config.cqtd_filters,
                config.cqtd_max_filters,
                config.cqtd_filters_scale,
                config.cqtd_dilations,
                config.cqtd_in_channels,
                config.cqtd_out_channels,
                config.sample_rate,
                config.cqtd_hop_lengths[i],
                n_octaves=config.cqtd_n_octaves[i],
                bins_per_octave=config.cqtd_bins_per_octaves[i],
            ) for i in range(config.cqtd_hop_lengths)
        ])

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               List[List[torch.Tensor]], List[List[torch.Tensor]]]:

        logits = []
        fmaps = []
        fmaps_mask = []
        logit_masks = []

        for disc in self.discriminators:
            logit, logit_mask, fmap = disc(y, mask)
            fmap_mask = logit_mask
            logits.append(logit)
            fmaps.append(fmap)
            fmaps_mask.append(fmap_mask)
            logit_masks.append(logit_mask)

        return logits, logit_masks, fmaps, fmaps_mask


class SequenceDiscriminatorCQT(nn.Module):

    def __init__(
        self,
        cqtd_filters,
        cqtd_max_filters,
        cqtd_filters_scale,
        cqtd_dilations,
        cqtd_in_channels,
        cqtd_out_channels,
        sample_rate,
        hop_length: int,
        n_octaves: int,
        bins_per_octave: int,
    ):
        super().__init__()

        self.filters = cqtd_filters
        self.max_filters = cqtd_max_filters
        self.filters_scale = cqtd_filters_scale
        self.kernel_size = (3, 9)
        self.dilations = cqtd_dilations
        self.stride = (1, 2)

        self.in_channels = cqtd_in_channels
        self.out_channels = cqtd_out_channels
        self.fs = sample_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        self.cqt_transform = CQT(
            self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
        )

        self.conv_pres = nn.ModuleList([
            nn.Conv2d(
                self.in_channels * 2,
                self.in_channels * 2,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            ) for _ in range(self.n_octaves)
        ])

        self.convs = nn.ModuleList([
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        ])
        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min((self.filters_scale**(i + 1)) * self.filters,
                          self.max_filters)
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        int(in_chs),
                        int(out_chs),
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding(self.kernel_size,
                                                    (dilation, 1)),
                    )))
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale**(len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    int(in_chs),
                    int(out_chs),
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding(
                        (self.kernel_size[0], self.kernel_size[0])),
                )))

        self.conv_post = weight_norm(
            nn.Conv2d(
                int(out_chs),
                self.out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding(
                    (self.kernel_size[0], self.kernel_size[0])),
            ))

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = torchaudio.transforms.Resample(orig_freq=self.fs,
                                                       new_freq=self.fs * 2)

        self.cqtd_normalize_volume = False

    def get_2d_padding(
            self,
            kernel_size: Tuple[int, int],
            dilation: Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        fmap = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - cal_mean_with_mask(x, mask, dim=-1)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)
        mask = mask.repeat_interleave(2, dim=-1)

        z, z_length = self.cqt_transform(x, mask.sum(-1))
        z = torch.view_as_real(z)
        z_mask = ~make_pad_mask(z_length, z.shape[-1])
        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)
        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]
        z_mask = z_mask.unsqueeze(1).unsqueeze(-1)
        z = z * z_mask

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(self.conv_pres[i](
                z[:, :, :,
                  i * self.bins_per_octave:(i + 1) * self.bins_per_octave, ]))
        latent_z = torch.cat(latent_z, dim=-1)
        latent_z = latent_z * z_mask

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)
            latent_z = latent_z * z_mask
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)
        latent_z = latent_z * z_mask

        return latent_z, z_mask, fmap


if __name__ == '__main__':
    model = SequenceDiscriminatorCQT(
        32,
        1024,
        1.0,
        [1, 2, 4],
        1,
        1,
        24000,
        256,
        9,
        48,
    )

    samples = int(24000)

    mask = torch.ones(1, samples, dtype=torch.bool)
    output, z_mask, fmap = model(torch.rand(1, samples), mask)
    print(output.shape, z_mask.shape, fmap[0].shape)
