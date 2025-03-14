import math
from functools import partial

import torch
import torchaudio
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(current_step: int, *,
                                               num_warmup_steps: int,
                                               num_training_steps: int,
                                               num_cycles: float):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps))
    return max(
        0.0,
        0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def frame_paddings(paddings: torch.Tensor, *, frame_size: int,
                   hop_size: int) -> torch.Tensor:
    """Frames paddings.

    Args:
        paddings: A Tensor of shape `[..., seq_len]`.
        frame_size: Size of frames.
        hop_size: Hop size of frames.

    Returns:
        A Tensor of shape `[..., num_windows]`, where each value represents
        the maximum padding value in the corresponding frame.

    Raises:
        ValueError: If the input is invalid.
    """
    if hop_size > frame_size:
        raise ValueError(
            f"hop_size {hop_size} must be smaller than frame_size {frame_size}."
        )

    # Ensure padding is at least frame_size long
    pad_len = (frame_size - hop_size) % frame_size  # Ensures full coverage
    if pad_len > 0:
        paddings = torch.nn.functional.pad(
            paddings, (0, pad_len), value=1)  # Pad with 1s as in your example

    # Unfold to create overlapping frames
    paddings_frame = paddings.unfold(-1, frame_size, hop_size)

    # Compute max padding per frame
    out_paddings = paddings_frame.max(dim=-1).values
    return out_paddings


class MelSpectrogram(torch.nn.Module):

    def __init__(self,
                 sample_rate=24000,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=100,
                 padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, paddings=None, **kwargs):
        """
        Args:
            audio: (B, C, T) or (B, T) or (T,)
            paddings: (B, T) or (T,) optional, corresponding to the padding mask in the audio data
                    (1 indicates padding, 0 indicates valid data).
            (mel_features, out_paddings)

        Returns:
            mel_features: (B, n_mels, T')
            out_paddings: (B, T') propagated padding information.
        """
        # Ensure the input is at least 2D (batch, time)
        if audio.dim() == 1:  # (T,) -> (1, 1, T)
            audio = audio.unsqueeze(0).unsqueeze(0)
            if paddings is not None:
                paddings = paddings.unsqueeze(0)
        elif audio.dim() == 2:  # (B, T) -> (B, 1, T)
            audio = audio.unsqueeze(1)
            if paddings is not None:
                paddings = paddings.unsqueeze(1)  # (B, T) -> (B, 1, T)
        elif audio.dim() == 3 and audio.shape[
                1] > 1:  # (B, C, T) -> (B, 1, T) (take the mean)
            audio = audio.mean(dim=1, keepdim=True)

        # Manual padding is needed when `padding="same"`
        if self.padding == "same":
            pad = (self.mel_spec.win_length
                   or self.mel_spec.n_fft) - self.mel_spec.hop_length
            pad_left, pad_right = pad // 2, pad - pad // 2
            audio = torch.nn.functional.pad(audio, (pad_left, pad_right),
                                            mode="reflect")

            # Padding should also be adjusted
            if paddings is not None:
                paddings = torch.nn.functional.pad(paddings, (pad_left, pad_right), value=1)

        # Compute Mel spectrogram
        mel = self.mel_spec(audio)  # (B, n_mels, T')

        # Compute padding propagation
        if paddings is not None:
            paddings = paddings.squeeze(
                1)  # Remove channel dimension (B, 1, T) -> (B, T)
            out_paddings = frame_paddings(paddings,
                                          frame_size=self.mel_spec.n_fft,
                                          hop_size=self.mel_spec.hop_length)
        else:
            out_paddings = None

        # Avoid log(0) errors
        features = torch.log(torch.clip(mel, min=1e-6))

        return features, out_paddings
