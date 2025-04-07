import torch
import torchaudio
from absl import app, flags
from ml_collections import config_flags
from wenet.utils.mask import make_non_pad_mask, make_pad_mask

from vocos.model import ISTFTHead, Transformer
from vocos.utils import MelSpectrogram

flags.DEFINE_string('wav', None, help='audio file', required=True)
flags.DEFINE_string('checkpoint', None, help='model checkpoint', required=True)

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config')


class Vocos(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone = Transformer(config)
        self.head = ISTFTHead(config)

    def forward(self, mels: torch.Tensor, mels_lens: torch.Tensor):
        """ 
        Args:
            mels: [B, T, D]
            mels_lens: [B]
        Returns:
            wav: [B, T]
            wav_mask: [B,1,T]
        """
        mels_mask = ~make_pad_mask(mels_lens, mels.shape[1])
        x, mask = self.backbone(mels, mels_mask)
        wav, wav_mask = self.head(x, mask.squeeze(1))
        wav = wav * wav_mask

        return wav, wav_mask

    def forward_chunk_by_chunk(self):
        pass


def main(_):
    config = FLAGS.config
    print(config)

    # TODO model.from_pretrained
    model = Vocos(config)
    ckpt = torch.load(FLAGS.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    y, sr = torchaudio.load(FLAGS.wav)
    y = y[:1, :]
    if sr != config.sample_rate:
        y = torchaudio.functional.resample(y,
                                           orig_freq=sr,
                                           new_freq=config.sample_rate)
    y_lens = torch.tensor([y.shape[1]], dtype=torch.int64)
    y_padding = make_pad_mask(torch.tensor([y.shape[1]], dtype=torch.int64))
    mel_fn = MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_size,
        n_mels=config.n_mels,
    )
    mels, mels_paddings = mel_fn(y, y_padding, dtype=torch.int64)
    wav, wav_mask = model(mels.transpose(1, 2), (~mels_paddings).sum(-1))
    torchaudio.save("test.wav", wav, config.sample_rate, bits_per_sample=16)


if __name__ == '__main__':
    app.run(main)
