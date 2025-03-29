from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
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
        mask = F.max_pool2d(mask.float(), kernel_size, stride, padding)
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

    def __init__(self, fft_sizes: Tuple[int, ...] = (2048, 1024, 512)):
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
    ):
        super().__init__()
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

        x = x - x.mean(dim=-1, keepdim=True)
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        spec = self.spec_fn(x)
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
