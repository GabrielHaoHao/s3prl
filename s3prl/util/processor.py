import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence
import torch


def compute_fbank(data,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    fbank_features = []
    for waveform in data:
        waveform = waveform * (1 << 15)
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=16000)
        fbank_features.append(mat)
    return fbank_features

def padding(data):
    assert isinstance(data, list)
    device = data[0].device
    feats_length = torch.tensor([feat.size(0) for feat in data],
                                dtype=torch.int32).to(device)
    padded_feats = pad_sequence(data,
                                batch_first=True,
                                padding_value=0)
    return padded_feats, feats_length

def extract_fbank(wave_inputs):
    # wave_len = [len(wave) for wave in wave_inputs]
    # wave_inputs = pad_sequence(wave_inputs, batch_first=True)
    feats = compute_fbank(wave_inputs)
    padded_feats, feats_length = padding(feats)
    return padded_feats, feats_length
