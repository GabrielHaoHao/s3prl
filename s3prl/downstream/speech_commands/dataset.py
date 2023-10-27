from random import randint
from pathlib import Path

from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
# from transformers import DistilBertTokenizer
# tokenizer = DistilBertTokenizer.from_pretrained('/root/data2/data/dataset/huggingface_model/DistillBert')

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.data = []
        self.class_num = 2

    def __getitem__(self, idx):
        audio_path, text, label = self.data[idx]
        wav, _ = apply_effects_file(str(audio_path), EFFECTS)
        wav = wav.squeeze(0).numpy()
        # text = tokenizer(text, return_tensors='pt')
        return wav, text, label

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
        wav, text, label = super().__getitem__(idx)
        return wav, text, label

