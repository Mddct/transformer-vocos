from typing import List, Tuple

import torch
from torch import nn

from vocos.utils import MelSpectrogram


def cal_mean_with_mask(input, mask):
    if mask is None:
        loss_term = torch.mean(torch.clamp(input, min=0))
    else:
        valid_scores = torch.clamp(input, min=0)
        masked_scores = valid_scores * mask.float()

        valid_elements = mask.sum().float()
        valid_elements = torch.clamp(valid_elements, min=1e-6)

        loss_term = masked_scores.sum() / valid_elements
    return loss_term


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


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
    ):
        super().__init__()
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            # padding='center',
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

        mel_hat = safe_log(mel_hat) * mel_mask.unsqueeze(1)
        mel = safe_log(mel) * mel_mask.unsqueeze(1)

        loss = torch.nn.functional.l1_loss(mel, mel_hat)

        return loss


def compute_generatorl_oss(
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
        loss_term = cal_mean_with_mask(1 - dg, mask)
        gen_losses.append(loss_term.detach())
        total_loss += loss_term

    return total_loss, gen_losses


def compute_discriminator_loss(
    self,
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
        r_loss = cal_mean_with_mask(1 - dr, mask)
        g_loss = cal_mean_with_mask(1 + dg, mask)
        loss += r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)

    return loss, r_losses, g_losses


def compute_feature_matching_loss(
        fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]],
        masks: List[List[torch.Tensor]]) -> torch.Tensor:

    loss = torch.tensor(0,
                        device=fmap_r[0][0].device,
                        dtype=fmap_r[0][0].dtype)
    for dr, dg, mask in zip(fmap_r, fmap_g, masks):
        for rl, gl, m in zip(dr, dg, mask):
            loss += cal_mean_with_mask(torch.abs(rl - gl), m)
    return loss
