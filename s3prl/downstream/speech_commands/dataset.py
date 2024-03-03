from random import randint
from pathlib import Path
import torch
import numpy as np
import random

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from .add_noise_func import add_noise


EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.data = []
        self.class_num = 2

    def __getitem__(self, idx):
        audio_path, text, label = self.data[idx]
        wav, _ = apply_effects_file(str(audio_path), EFFECTS)
        wav_fbank = wav
        wav = wav.squeeze(0).numpy()
        # try:
        #     wav = add_noise(wav)
        # except Exception as e:
        #     print("this audio_path fail to add noise:" + audio_path)

        # text = tokenizer(text, return_tensors='pt')
        return wav, wav_fbank, text, label

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)

class SpeechCommandsBaseTestDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.data = []
        self.class_num = 2

    def __getitem__(self, idx):
        # train_data
        audio_path, text, label, type_name = self.data[idx]
        filename = audio_path.split('.')[0]
        wav, _ = apply_effects_file(str(audio_path), EFFECTS)
        wav = wav.squeeze(0).numpy()

        # text = tokenizer(text, return_tensors='pt')
        return wav, text, label, filename, type_name

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)


class SpeechCommandsDataset(SpeechCommandsBaseDataset):
    """Training and validation dataset."""

    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.data = data_list


    def __getitem__(self, idx):
        wav, wav_fbank, text, label = super().__getitem__(idx)
        return wav, wav_fbank, text, label

class SpeechCommandsTestingDataset(SpeechCommandsBaseTestDataset):
    """Training and validation dataset."""

    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.data = data_list


    def __getitem__(self, idx):
        wav, text, label, filename, type_name = super().__getitem__(idx)
        return wav, text, label, filename, type_name
