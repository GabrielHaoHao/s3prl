"""Downstream expert for Spoken Term Detection on Speech Commands."""

import re
import os
import hashlib
from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from catalyst.data.sampler import DistributedSamplerWrapper
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import SpeechCommandsDataset, SpeechCommandsTestingDataset
from .mask import make_pad_mask


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim: int, text_upstream_dim: int, downstream_expert: dict, expdir: str, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        # train_list, valid_list = split_dataset(self.datarc["speech_commands_root"])
        # train_list, valid_list = divid_dataset(self.datarc["speech_commands_root"])
        train_list, valid_list, test_list = split_kws_dataset(self.datarc["speech_commands_root"])

        self.train_dataset = SpeechCommandsDataset(train_list, **self.datarc)
        self.dev_dataset = SpeechCommandsDataset(valid_list, **self.datarc)
        self.test_dataset = SpeechCommandsTestingDataset(test_list, **self.datarc)

        self.speech_embedding = nn.Sequential(nn.Linear(upstream_dim, self.modelrc['projector_dim']),
                                             nn.ReLU())
        self.text_embedding = nn.Sequential(nn.Linear(text_upstream_dim, self.modelrc['projector_dim']),
                                        nn.ReLU())
        self.speechs_batchnorm = nn.BatchNorm1d(self.modelrc['projector_dim'])
        self.text_batchnorm = nn.BatchNorm1d(self.modelrc['projector_dim'])
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.modelrc['projector_dim'])

        self.combined_linear = nn.Sequential(nn.Linear(4 * self.modelrc['projector_dim'], self.modelrc['projector_dim']*2),
                                             nn.ReLU(inplace=True),
                                             self.dropout,
                                             nn.Linear(2 * self.modelrc['projector_dim'], self.modelrc['projector_dim']),
                                             nn.ReLU(inplace=True),)
        
        # self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        # self.projector_text = nn.Linear(text_upstream_dim, self.modelrc['projector_dim'])

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
        
    def local_inference_layer(self, x1, x2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''

        #  BATCH, 100 , 768   --- BATCH , 768 100
        attention = torch.matmul(x1, x2.transpose(1, 2))
        
        #  BATCH, 100 , 100
        weight1 = F.softmax(attention , dim=-1)

        # BATCH, 100 , 100 *  BATCH, 100 , 768
        # BATCH, 100 , 768 
        x1_align = torch.matmul(weight1, x2)

        weight2 = F.softmax(attention.transpose(1, 2) , dim=-1)
        # BATCH, 100 , 100 *  BATCH, 100 , 768
        # BATCH, 100 , 768
        x2_align = torch.matmul(weight2, x1)

        x1_sub,x2_sub = x1 - x1_align, x2 - x2_align
        x1_mul,x2_mul = x1 * x1_align, x2 * x2_align

        # BATCH, 100 , 768 * 4
        x1_output = torch.cat([x1, x1_align, x1_sub, x1_mul], -1)
        # BATCH, 100 , 768 * 4
        x2_output = torch.cat([x2, x2_align, x2_sub, x2_mul], -1)

        # input : BATCH, 100 , 768 * 4
        # output :  BATCH, 100 , 768
        x1_output = self.combined_linear(x1_output)
        x2_output = self.combined_linear(x2_output)

        return x1_output, x2_output

    # Interface
    def forward(self, mode, features, feat_text, text_mask, labels, records, **kwargs):
        device = features[0].device
        audio_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)
        T = len(features)
        audio_mask = ~make_pad_mask(audio_len, 0).unsqueeze(1)  # (B, 1, T)
        text_mask = text_mask.bool().unsqueeze(1) # (B, 1, T)
        # len_text = torch.IntTensor([len(feat) for feat in feat_text]).to(device=device)
        features = pad_sequence(features, batch_first=True)
        # features = self.projector(features)
        # text_features = self.projector_text(feat_text)

        speechs_embedding = self.speech_embedding(features)
        text_embedding = self.text_embedding(feat_text)
        speech_enc = self.layer_norm(speechs_embedding)
        text_enc = self.layer_norm(text_embedding)

        # local inference layer
        speechs_combined , text_combined = self.local_inference_layer(speech_enc,text_enc)

        predicted = self.model(speechs_combined, text_combined, audio_mask, text_mask)

        labels = torch.LongTensor(labels).to(features.device)

        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records["loss"].append(loss.item())
        records["acc"] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records["predict"] += predicted_classid.cpu().tolist()
        records["truth"] += labels.cpu().tolist()
        predicted_list = predicted.cpu().tolist()
        labels_list = labels.cpu().tolist()
        predicted_score = []
        for i in range(len(predicted_list)):
            predicted_score.append(predicted_list[i][labels_list[i]])
        records["predict_score"] += predicted_score

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

        with open(Path(self.expdir) / f"{mode}_predict.txt", "a") as file:
            lines = [f"{f} {t} {i}\n" for f, t, i in zip(records["filename"], records["typename"], records["predict"])]
            file.writelines(lines)

        with open(Path(self.expdir) / f"{mode}_predict_score.txt", "a") as file:
            lines = [f"{f} {t} {i}\n" for f, t, i in zip(records["filename"], records["typename"], records["predict_score"])]
            file.writelines(lines)

        with open(Path(self.expdir) / f"{mode}_truth.txt", "a") as file:
            lines = [f"{f} {t} {i}\n" for f, t, i in zip(records["filename"], records["typename"], records["truth"])]
            file.writelines(lines)

        with open(Path(self.expdir) / f"{mode}_all.txt", "a") as file:
            lines = [f"{f} {te} {ty} {i} {j} {k}\n" for f, te, ty, i, j, k in zip(records["filename"], records['text'] , records["typename"], records["truth"], records["predict_score"], records["predict"])]
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
            anc_pos = [root_dir + line[0], line[2], '1']
            anc_neg = [root_dir + line[0], line[6], line[-2]]
            com_pos = [root_dir + line[4], line[6], '1']
            com_neg = [root_dir + line[4], line[2], line[-2]]
            # wav_path, text, label
            train_cur_list.append((anc_pos[0], anc_pos[1], int(anc_pos[2])))
            train_cur_list.append((anc_neg[0], anc_neg[1], int(anc_neg[2])))
            train_cur_list.append((com_pos[0], com_pos[1], int(com_pos[2])))
            train_cur_list.append((com_neg[0], com_neg[1], int(com_neg[2])))
        train_list += train_cur_list

    # dev_set
    val_list = []
    val_path = root_dir + 'val/'
    for i in range(1, 5):
        cur_path = val_path + 'libriphrase_diffspk_val_' + str(i) + 'word.csv'
        cur_list = pd.read_csv(cur_path).values.tolist()
        val_cur_list = []
        for line in cur_list:
            anc_pos = [root_dir + line[0], line[2], '1']
            anc_neg = [root_dir + line[0], line[6], line[-2]]
            com_pos = [root_dir + line[4], line[6], '1']
            com_neg = [root_dir + line[4], line[2], line[-2]]
            # wav_path, text, label
            val_cur_list.append((anc_pos[0], anc_pos[1], int(anc_pos[2])))
            val_cur_list.append((anc_neg[0], anc_neg[1], int(anc_neg[2])))
            val_cur_list.append((com_pos[0], com_pos[1], int(com_pos[2])))
            val_cur_list.append((com_neg[0], com_neg[1], int(com_neg[2])))
        val_list += val_cur_list

    # test_set
    test_list = []
    test_path = root_dir + 'test/'
    for i in range(1, 5):
        cur_path = test_path + 'libriphrase_diffspk_test_' + str(i) + 'word.csv'
        cur_list = pd.read_csv(cur_path).values.tolist()
        test_cur_list = []
        for line in cur_list:
            anc_pos = [root_dir + line[0], line[2], '1', 'samespk_positive']
            anc_neg = [root_dir + line[0], line[6], line[-2], line[-3]]
            com_pos = [root_dir + line[4], line[6], '1', 'samespk_positive']
            com_neg = [root_dir + line[4], line[2], line[-2], line[-3]]
            # wav_path, text, label
            test_cur_list.append((anc_pos[0], anc_pos[1], int(anc_pos[2]), anc_pos[3]))
            test_cur_list.append((anc_neg[0], anc_neg[1], anc_neg[2], anc_neg[3]))
            test_cur_list.append((com_pos[0], com_pos[1], int(com_pos[2]), com_pos[3]))
            test_cur_list.append((com_neg[0], com_neg[1], com_neg[2], com_neg[3]))
        test_list += test_cur_list

    return train_list, val_list, test_list
