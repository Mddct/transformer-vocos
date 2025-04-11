import math
import os
from functools import partial

import torch
import torch.distributed as dist
import torchaudio
from torch.optim.lr_scheduler import LambdaLR

from vocos.model import Transformer


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

    # Unfold to create overlapping frames
    paddings_frame = paddings.unfold(-1, frame_size, hop_size)

    # Compute max padding per frame
    out_paddings = paddings_frame.max(dim=-1).values
    return out_paddings


class MelSpectrogram(torch.nn.Module):

    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
        power=1,
        fmin=0,
        fmax=8000,
        norm="slaney",
        mel_scale="slaney",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=power,
            f_min=fmin,
            f_max=fmax,
            norm=norm,
            mel_scale=mel_scale,
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
        # Manual padding is needed when `padding="same"`
        pad = (self.mel_spec.win_length
               or self.mel_spec.n_fft) - self.mel_spec.hop_length
        if self.padding == "center":
            pad_left, pad_right = pad // 2, pad - pad // 2
            audio = torch.nn.functional.pad(audio, (pad_left, pad_right),
                                            mode="reflect")
            # Padding should also be adjusted
            if paddings is not None:
                paddings = torch.nn.functional.pad(paddings,
                                                   (pad_left, pad_right),
                                                   value=1)
        elif self.padding == "same":
            audio = torch.nn.functional.pad(audio, (0, pad), mode="reflect")
            # Padding should also be adjusted
            if paddings is not None:
                paddings = torch.nn.functional.pad(paddings, (0, pad), value=1)

        # Compute Mel spectrogram
        mel = self.mel_spec(audio)  # (B, n_mels, T')

        # Compute padding propagation
        out_paddings = frame_paddings(paddings,
                                      frame_size=self.mel_spec.n_fft,
                                      hop_size=self.mel_spec.hop_length)

        # Avoid log(0) errors
        features = torch.log(torch.clip(mel, min=1e-6))

        return features, out_paddings


def init_distributed(configs):

    local_rank = os.environ.get('LOCAL_RANK', 0)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    print('training on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))

    return world_size, local_rank, rank


def get_model(config):
    if hasattr(config, 'model_type'):
        if config.model_type == 'conformer':
            from efficient_conformer.model import Conformer
            return Conformer(config)
    return Transformer(config)
