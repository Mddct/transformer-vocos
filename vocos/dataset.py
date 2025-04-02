import json
from functools import partial

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from wenet.dataset.datapipes import WenetRawDatasetSource


def decode_wav(sample):
    obj = json.loads(sample['line'])
    filepath = obj['wav']
    audio, sample_rate = torchaudio.load(filepath)
    return {
        'wav': audio,
        "sample_rate": sample_rate,
    }


def resample(sample, resample_rate=16000):
    """ Resample sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return sample


def filter_by_length(sample, max_seconds=30):
    wav = sample['wav']
    sr = sample['sample_rate']
    if wav.shape[1] / sr <= max_seconds:
        return True
    return False


def padding(data, pad_value=0):
    samples = data

    wavs_lst = [sample['wav'][0] for sample in data]
    wavs_lens = [sample['wav'].shape[1] for sample in data]

    wavs = pad_sequence(wavs_lst, batch_first=True, padding_value=pad_value)
    wavs_lens = torch.tensor(wavs_lens, dtype=torch.int64)
    return {
        'wavs': wavs,
        'wavs_lens': wavs_lens,
    }


def init_dataset_and_dataloader(files,
                                batch_size,
                                num_workers,
                                prefetch,
                                shuffle,
                                steps,
                                drop_last=False,
                                sample_rate=24000,
                                seed=2025):

    dataset = WenetRawDatasetSource(files, cycle=steps, shuffle=shuffle)
    # TODO: stage2 shuffle

    dataset = dataset.map(decode_wav)
    dataset = dataset.filter(filter_by_length)
    dataset = dataset.map(partial(resample, resample_rate=sample_rate))
    dataset = dataset.batch(batch_size,
                            wrapper_class=partial(padding, pad_value=0.0),
                            drop_last=drop_last)

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            persistent_workers=True,
                            prefetch_factor=prefetch,
                            generator=generator)
    return dataset, dataloader
