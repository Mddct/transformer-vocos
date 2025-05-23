from typing import List, Tuple

import torch
from torch import nn

from vocos.utils import MelSpectrogram


def cal_mean_with_mask(input: torch.Tensor, mask: torch.Tensor, dim=None):
    if mask is None:
        loss_term = torch.mean(torch.clamp(input, min=0))
    else:
        valid_scores = input
        masked_scores = valid_scores * mask.float()

        valid_elements = mask.broadcast_to(input.shape).sum().float()
        valid_elements = torch.clamp(valid_elements, min=1e-6)

        if dim is None:
            loss_term = masked_scores.sum() / valid_elements
        else:
            loss_term = masked_scores.sum(dim=dim,
                                          keepdim=True) / valid_elements
    return loss_term


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

        loss = cal_mean_with_mask(torch.abs(mel-mel_hat), mel_mask.unsqueeze(1))

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
