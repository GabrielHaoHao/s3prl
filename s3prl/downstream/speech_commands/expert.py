"""Downstream expert for Spoken Term Detection on Speech Commands."""

import re
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from catalyst.data.sampler import DistributedSamplerWrapper
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import SpeechCommandsDataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim: int, downstream_expert: dict, expdir: str, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        # train_list, valid_list = split_dataset(self.datarc["speech_commands_root"])
        # train_list, valid_list = divid_dataset(self.datarc["speech_commands_root"])
        train_list, valid_list, test_list = split_kws_dataset(self.datarc["speech_commands_root"])

        self.train_dataset = SpeechCommandsDataset(train_list, **self.datarc)
        self.dev_dataset = SpeechCommandsDataset(valid_list, **self.datarc)
        self.test_dataset = SpeechCommandsDataset(test_list, **self.datarc)

        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.projector_text = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        # self.model = model_cls(
        #     input_dim = self.modelrc['projector_dim'],
        #     output_dim = self.train_dataset.class_num,
        #     **model_conf,
        # )
        self.model = Model(
            input_dim = self.modelrc['projector_dim'],
            output_class_num = self.train_dataset.class_num,
            hidden_dim = self.modelrc['projector_dim'],
        )

        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

    def _get_balanced_train_dataloader(self, dataset, drop_last=False):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        # if is_initialized():
        #     sampler = DistributedSamplerWrapper(sampler)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.datarc["batch_size"],
            drop_last=drop_last,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_balanced_dev_dataloader(self, dataset, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.datarc["batch_size"],
            drop_last=drop_last,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.datarc["batch_size"],
            drop_last=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def get_dataloader(self, mode):
        if mode == 'train':
            return self._get_balanced_train_dataloader(self.train_dataset, drop_last=True)
        elif mode == 'dev':
            return self._get_balanced_dev_dataloader(self.dev_dataset, drop_last=False)
        elif mode == 'test':
            return self._get_dataloader(self.test_dataset)
        else:
            raise NotImplementedError

    # Interface
    def forward(self, mode, features, feat_text, labels, records, **kwargs):
        device = features[0].device
        len_audio = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        len_text = torch.IntTensor([len(feat) for feat in feat_text]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        text_features = self.projector_text(feat_text)
        predicted = self.model(features, text_features, len_audio, len_text)

        labels = torch.LongTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records["loss"].append(loss.item())
        records["acc"] += (predicted_classid == labels).view(-1).cpu().float().tolist()


        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["loss", "acc"]:
            values = records[key]
            average = sum(values) / len(values)
            logger.add_scalar(
                f'speech_commands/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir, "log.log"), 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

        with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
            lines = [f"{f} {i}\n" for f, i in zip(records["filename"], records["predict"])]
            file.writelines(lines)

        with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
            lines = [f"{f} {i}\n" for f, i in zip(records["filename"], records["truth"])]
            file.writelines(lines)

        return save_names


def split_dataset(
    root_dir: Union[str, Path], max_uttr_per_class=2 ** 27 - 1
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split Speech Commands into 3 set.
    
    Args:
        root_dir: speech commands dataset root dir
        max_uttr_per_class: predefined value in the original paper
    
    Return:
        train_list: [(class_name, audio_path), ...]
        valid_list: as above
    """
    train_list, valid_list = [], []

    for entry in Path(root_dir).iterdir():
        if not entry.is_dir() or entry.name == "_background_noise_":
            continue

        for audio_path in entry.glob("*.wav"):
            speaker_hashed = re.sub(r"_nohash_.*$", "", audio_path.name)
            hashed_again = hashlib.sha1(speaker_hashed.encode("utf-8")).hexdigest()
            percentage_hash = (int(hashed_again, 16) % (max_uttr_per_class + 1)) * (
                100.0 / max_uttr_per_class
            )

            if percentage_hash < 10:
                valid_list.append((entry.name, audio_path))
            elif percentage_hash < 20:
                pass  # testing set is discarded
            else:
                train_list.append((entry.name, audio_path))

    return train_list, valid_list


def divid_dataset(root_dir):
    train_list, valid_list= [], []
    train_list_path = root_dir + 'train.txt'
    valid_list_path = root_dir + 'val.txt'
    with open(train_list_path, 'r') as ft:
        content = ft.readlines()
        for line in content:
            wav_path = line[:-1]
            label = line.split('/')[-2]
            train_list.append((label, wav_path))
    with open(valid_list_path, 'r') as fv:
        content = fv.readlines()
        for line in content:
            wav_path = line[:-1]
            label = line.split('/')[-2]
            valid_list.append((label, wav_path))
    return train_list, valid_list

def split_kws_dataset(root_dir):
    # train_set
    train_list = []
    train_path = root_dir + 'train/'
    for i in range(1, 5):
        cur_path = train_path + 'libriphrase_diffspk_train_' + str(i) + 'word.csv'
        cur_list = pd.read_csv(cur_path).values.tolist()
        train_cur_list = []
        for line in cur_list:
            wav_path = root_dir + line[1]
            text = line[7]
            label = line[-2]
            train_cur_list.append((wav_path, text, label))
        train_list += train_cur_list

    # dev_set
    val_list = []
    val_path = root_dir + 'val/'
    for i in range(1, 5):
        cur_path = val_path + 'libriphrase_diffspk_val_' + str(i) + 'word.csv'
        cur_list = pd.read_csv(cur_path).values.tolist()
        val_cur_list = []
        for line in cur_list:
            wav_path = root_dir + line[0]
            text = line[6]
            label = line[-2]
            val_cur_list.append((wav_path, text, label))
        val_list += val_cur_list

    # test_set
    test_list = []
    test_path = root_dir + 'test/'
    for i in range(1, 5):
        cur_path = test_path + 'libriphrase_diffspk_test_' + str(i) + 'word.csv'
        cur_list = pd.read_csv(cur_path).values.tolist()
        test_cur_list = []
        for line in cur_list:
            wav_path = root_dir + line[0]
            text = line[6]
            label = line[-2]
            test_cur_list.append((wav_path, text, label))
        test_list += test_cur_list
    return train_list, val_list, test_list
