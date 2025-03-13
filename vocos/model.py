import torch
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.class_utils import (WENET_ACTIVATION_CLASSES,
                                     WENET_ATTENTION_CLASSES,
                                     WENET_MLP_CLASSES)
from wenet.utils.common import mask_to_bias
from wenet.utils.mask import causal_or_lookahead_mask, make_non_pad_mask


class Transformer(BaseEncoder):

    def __init__(self, config):
        super().__init__(
            n_expert=config.n_expert,
            n_expert_activated=config.n_expert_activated,
            attention_heads=config.attention_heads,
            linear_units=config.linear_units,
            num_blocks=config.num_blocks,
            dropout_rate=config.dropout_rate,
            positional_dropout_rate=config.positional_dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            input_layer=config.input_layer,
            pos_enc_layer_type=config.pos_enc_layer_type,
            normalize_before=config.normalize_before,
            static_chunk_size=config.static_chunk_size,
            use_dynamic_chunk=config.use_dynamic_chunk,
            global_cmvn=config.global_cmvn,
            use_dynamic_left_chunk=config.use_dynamic_left_chunk,
            query_bias=config.query_bias,
            key_bias=config.key_bias,
            value_bias=config.value_bias,
            activation_type=config.activation_type,
            gradient_checkpointing=config.gradient_checkpointing,
            use_sdpa=config.use_sdpa,
            layer_norm_type=config.layer_norm_type,
            norm_eps=config.norm_eps,
            n_kv_head=config.n_kv_head,
            head_dim=config.head_dim,
            selfattention_layer_type=config.selfattention_layer_type,
            mlp_type=config.mlp_type,
            mlp_bias=config.mlp_bias,
        )
        self.config = config

    def forward(self, batch, device):
        mels = batch['mels'].to(device)
        mels_lens = batch['mels_lens'].to(device)

        mask = make_non_pad_mask(mels_lens)
        xs, pos_emb, masks = self.embed(mels, mask)
        mask_pad = masks
        att_mask = causal_or_lookahead_mask(masks, self.config.right_context,
                                            self.config.left_context)
        if self.use_sdpa:
            att_mask = mask_to_bias(att_mask, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, att_mask, pos_emb,
                                                  mask_pad)
        else:
            xs = self.forward_layers(xs, att_mask, pos_emb, mask_pad)
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk_by_chunk(self,
                               xs,
                               decoding_chunk_size,
                               num_decoding_left_chunks=-1):
        pass

    def forward_chunk(self, xs, offset, required_cache_size, att_cache,
                      cnn_cache, att_mask):
        pass


class ISTFT(torch.nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor,
                paddding: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec,
                               self.n_fft,
                               self.hop_length,
                               self.win_length,
                               self.window,
                               center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(torch.nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        out_dim = config.n_fft + 2
        self.out = torch.nn.Linear(config.dim, out_dim)
        self.istft = ISTFT(n_fft=config.n_fft,
                           hop_length=config.hop_length,
                           win_length=config.n_fft,
                           padding=config.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio
