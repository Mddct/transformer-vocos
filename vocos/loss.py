from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from vocos.utils import STFT, MelSpectrogram


def cal_mean_with_mask(input: torch.Tensor, mask: torch.Tensor, dim=None):
    if mask is None:
        loss_term = torch.mean(torch.clamp(input, min=0))
    else:
        valid_scores = input
        masked_scores = valid_scores * mask.float()

        valid_elements = mask.broadcast_to(input.shape).sum(
            dim=dim, keepdim=True).float()
        valid_elements = torch.clamp(valid_elements, min=1e-6)

        if dim is None:
            loss_term = masked_scores.sum() / valid_elements.sum()
        else:
            loss_term = masked_scores.sum(dim=dim,
                                          keepdim=True) / valid_elements
    return loss_term


def _unwrap(p, dim=-1):
    dd = torch.diff(p, dim=dim)
    dd_mod = (dd + np.pi) % (2 * np.pi) - np.pi

    dd_mod = torch.where((dd_mod == -np.pi) & (dd > 0), np.pi, dd_mod)

    correction = torch.cumsum(dd - dd_mod, dim=dim)

    out = torch.zeros_like(p)
    out.select(dim, 0).copy_(p.select(dim, 0))  # 第一帧保持不变

    slicer = [slice(None)] * p.dim()
    slicer[dim] = slice(1, None)
    out[tuple(slicer)] = p[tuple(slicer)] - correction

    return out


def compute_phase_loss(stft_pred, stft_true, mask):
    B, C, F, T_stft = stft_pred.shape
    stft_pred = stft_pred.view(B, C, F, T_stft)
    stft_true = stft_true.view(B, C, F, T_stft)

    phase_pred = torch.angle(stft_pred)
    phase_true = torch.angle(stft_true)

    unwrapped_phase_pred = _unwrap(phase_pred, dim=-1)
    unwrapped_phase_true = _unwrap(phase_true, dim=-1)

    delta_phase_pred = unwrapped_phase_pred[..., 1:] - unwrapped_phase_pred[
        ..., :-1]
    delta_phase_true = unwrapped_phase_true[..., 1:] - unwrapped_phase_true[
        ..., :-1]

    mag_true = torch.abs(stft_true[..., 1:])
    mask_mag = mag_true > 1e-7

    if mask is not None:
        # mask: (B, T) -> STFT 帧级别 mask
        valid_lengths = mask.sum(-1)

        time_indices = torch.arange(
            T_stft - 1,
            device=stft_pred.device).unsqueeze(0).expand(B, T_stft - 1)
        mask_length = time_indices < valid_lengths.unsqueeze(1)
        mask_length = mask_length.unsqueeze(1).unsqueeze(1).expand(B, C, F, -1)

        mask_mag = mask_mag & mask_length

    loss = torch.abs(delta_phase_pred - delta_phase_true)
    loss = loss[mask_mag]
    loss = loss.mean()

    return loss


def masked_spectral_convergence_loss(x_mag: torch.Tensor, y_mag: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the masked spectral convergence loss for multi-channel inputs.

    This implementation correctly handles arbitrary masks across channels by
    aggregating all unmasked values before the final norm calculation.

    Args:
        x_mag (torch.Tensor): Predicted magnitude spectrogram.
                              Shape: (batch, channels, freq_bins, time_frames)
        y_mag (torch.Tensor): Ground truth magnitude spectrogram.
                              Shape: (batch, channels, freq_bins, time_frames)
        mask (torch.Tensor): Boolean or binary tensor with the same shape as the inputs.
                             The loss will be computed only where the mask is True (or 1).

    Returns:
        torch.Tensor: A loss tensor for each item in the batch. Shape: (batch, channels)
    """
    # Ensure mask is a float for multiplication
    mask_float = mask.float()

    # Apply the mask to the tensors before any calculation
    masked_diff = (y_mag - x_mag) * mask_float
    masked_true = y_mag * mask_float

    # Calculate the squared sum for the numerator and denominator.
    # We sum over all three dimensions (channel, freq, time) - dims [1, 2, 3] -
    # to correctly aggregate all unmasked values.
    numerator_sq_sum = torch.sum(masked_diff**2, dim=[2, 3])
    denominator_sq_sum = torch.sum(masked_true**2, dim=[2, 3])

    # Take the square root to get the true Frobenius norm
    numerator = torch.sqrt(numerator_sq_sum)
    denominator = torch.sqrt(denominator_sq_sum)

    # Add a small epsilon to the denominator for numerical stability
    loss = numerator / (denominator + 1e-8)

    return loss


class MultiScaleSFTLoss(nn.Module):

    def __init__(
        self,
        n_ffts: List[int] = [
            32, 64, 128, 256, 512, 1024, 2048, 44100 // 50 * 4
        ],
        padding="center",
        spectralconv_weight: float = 1,
        log_weight: float = 1,
        lin_weight: float = 0.1,
        phase_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_ffts = n_ffts
        self.losses = torch.nn.ModuleList([
            STFTLoss(
                n_fft,
                n_fft // 4,
                padding,
                spectralconv_weight=spectralconv_weight,
                log_weight=log_weight,
                lin_weight=lin_weight,
                phase_weight=phase_weight,
            ) for n_fft in n_ffts
        ])

    def forward(self, y_hat, y, mask):
        loss = 0.0
        for loss_fn in self.losses:
            loss_dict = loss_fn(y_hat, y, mask)
            loss = loss + loss_dict['loss']
        # TODO: other loss info
        return loss


class STFTLoss(nn.Module):

    def __init__(self,
                 n_fft=1024,
                 hop_length=256,
                 padding="center",
                 window_fn=torch.hann_window,
                 spectralconv_weight: float = 1,
                 log_weight: float = 1,
                 lin_weight: float = 0.1,
                 phase_weight: float = 0.5,
                 power: int = 1) -> None:

        super().__init__()

        #TODO: A_weighting or K_weighting
        self.stft = STFT(n_fft, hop_length, padding, window_fn, power)
        self.log_weight = log_weight
        self.spectralconv_weight = spectralconv_weight
        self.lin_weight = lin_weight
        self.phi_weight = phase_weight

    def forward(self, y_hat, y, mask):
        """
        Args:
            y_hat (Tensor): Predicted audio waveform. [B,C,T]
            y (Tensor): Ground truth audio waveform. [B,C,T]

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        B, C, T = y_hat.shape
        y_hat = y_hat.view(-1, T)
        y = y.view(-1, T)
        spec_hat, spec_hat_padding = self.stft(
            y_hat, ~mask.unsqueeze(1).repeat(1, C, 1).view(-1, T).bool())
        spec, _ = self.stft(
            y, ~mask.unsqueeze(1).repeat(1, C, 1).view(-1, T).bool())
        spec_hat = spec_hat.view(B, C, spec_hat.shape[-2], -1)
        spec = spec.view(B, C, spec.shape[-2], -1)
        spec_mask = ~spec_hat_padding.view(B, C,
                                           spec_hat_padding.shape[-1])[:, 0, :]

        mag_hat = spec_hat.abs()
        mag = spec.abs()

        loss = 0.0
        loss_mag = None
        if self.lin_weight != 0:
            loss_mag = masked_spectral_convergence_loss(
                mag_hat, mag,
                spec_mask.unsqueeze(1).unsqueeze(1)).mean()  # [B,C]
            loss = loss + self.spectralconv_weight * loss_mag

        # log
        loss_log_mag = None
        if self.log_weight != 0:
            loss_log_mag = cal_mean_with_mask(
                torch.abs(
                    torch.log(torch.clip(mag, min=1e-7)) -
                    torch.log(torch.clip(mag_hat, min=1e-7))),
                spec_mask.unsqueeze(1).unsqueeze(2),
                dim=[0, 1, 2, 3]).mean()  # [B,C]
            loss = loss + self.log_weight * loss_log_mag

        # linear
        loss_lin_mag = None
        if self.lin_weight != 0:
            loss_lin_mag = cal_mean_with_mask(
                torch.abs(
                    torch.clip(mag, min=1e-7) - torch.clip(mag_hat, min=1e-7)),
                spec_mask.unsqueeze(1).unsqueeze(2),
                dim=[0, 1, 2, 3]).mean()
            loss = loss + loss_lin_mag * loss_lin_mag

        loss_phase = None
        if self.phi_weight != 0:
            loss_phase = compute_phase_loss(spec_hat, spec, spec_mask)
            loss = loss + self.phi_weight * loss_phase

        return {
            "loss": loss,
            "loss_mag": loss_mag,
            "loss_log_mag": loss_log_mag,
            "loss_lin_mag": loss_lin_mag,
            "loss_phase": loss_phase,
        }


class MultiScaleMelSpecReconstructionLoss(nn.Module):
    """ https://arxiv.org/abs/2306.06546
    """

    def __init__(self,
                 sample_rate: int = 24000,
                 n_ffts: List[int] = [32, 64, 128, 256, 512, 1024, 1920, 2048],
                 n_mels: List[int] = [5, 10, 20, 40, 80, 100, 100, 320],
                 power=1,
                 padding="center",
                 fmin=0,
                 fmax=None,
                 norm="slaney",
                 mel_scale="slaney",
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert len(n_ffts) == len(n_mels)

        self.losses = torch.nn.ModuleList([
            MelSpecReconstructionLoss(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=n_fft // 4,
                n_mels=n_mels[i],
                power=power,
                padding=padding,
                fmin=fmin,
                fmax=fmax,
                norm=norm,
                mel_scale=mel_scale,
            ) for i, n_fft in enumerate(n_ffts)
        ])

    def forward(self, y_hat, y, mask):
        loss = 0.0
        for loss_fn in self.losses:
            loss = loss + loss_fn(y_hat, y, mask)
        return loss / len(self.losses)


class MelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        power=1,
        padding="center",
        fmin=0,
        fmax=8000,
        norm="slaney",
        mel_scale="slaney",
    ):
        super().__init__()
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            fmin=fmin,
            fmax=fmax,
            padding=padding,
            norm=norm,
            mel_scale=mel_scale,
        )

    def forward(self, y_hat, y, mask) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat, mel_hat_padding = self.mel_spec(y_hat, ~mask.bool())
        mel, _ = self.mel_spec(y, ~mask.bool())
        mel_mask = ~mel_hat_padding

        mel_hat = mel_hat * mel_mask.unsqueeze(1)
        mel = mel * mel_mask.unsqueeze(1)

        loss = cal_mean_with_mask(torch.abs(mel - mel_hat),
                                  mel_mask.unsqueeze(1))

        return loss


def compute_generator_loss(
        disc_outputs: List[torch.Tensor],
        masks: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Args:
        disc_outputs (List[Tensor]): List of discriminator outputs from sub-discriminators
        masks (List[Tensor]): List of boolean masks corresponding to each discriminator output

    Returns:
        Tuple[Tensor, List[Tensor]]:
        - Total generator loss (scalar tensor)
         - List of per-discriminator losses
    """
    device = disc_outputs[0].device
    dtype = disc_outputs[0].dtype

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    gen_losses = []

    for dg, mask in zip(disc_outputs, masks):
        loss_term = cal_mean_with_mask((1 - dg)**2, mask)
        gen_losses.append(loss_term.detach())
        total_loss = total_loss + loss_term

    return total_loss / len(gen_losses), gen_losses


def compute_discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
    masks: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Args:
        disc_real_outputs (List[Tensor]): List of discriminator outputs for real samples.
        disc_generated_outputs (List[Tensor]): List of discriminator outputs for generated samples.

     Returns:
        Tuple[Tensor, List[Tensor], List[Tensor]]: A tuple containing the total loss, a list of loss values from
        the sub-discriminators for real outputs, and a list of loss values for generated outputs.
     """
    loss = torch.tensor(0.0,
                        device=disc_real_outputs[0].device,
                        dtype=disc_real_outputs[0].dtype)
    r_losses = []
    g_losses = []
    for dr, dg, mask in zip(disc_real_outputs, disc_generated_outputs, masks):
        r_loss = cal_mean_with_mask((1 - dr)**2, mask)
        g_loss = cal_mean_with_mask(dg**2, mask)
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)

    return loss / len(r_losses), r_losses, g_losses


def compute_feature_matching_loss(
        fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]],
        masks: List[List[torch.Tensor]]) -> torch.Tensor:

    loss = torch.tensor(0,
                        device=fmap_r[0][0].device,
                        dtype=fmap_r[0][0].dtype)
    nums = 0
    for dr, dg, mask in zip(fmap_r, fmap_g, masks):
        for rl, gl, m in zip(dr, dg, mask):
            loss = loss + cal_mean_with_mask(torch.abs(rl - gl), m)
            nums += 1
    return loss / nums


if __name__ == '__main__':

    loss_fn = STFTLoss(882 * 4, 882, phase_weight=0.5)
    wav = torch.rand(1, 2, 48000)
    wav_g = torch.rand(1, 2, 48000)

    mask = torch.ones(1, 48000)

    loss = loss_fn(wav_g, wav, mask)
    print(loss)
