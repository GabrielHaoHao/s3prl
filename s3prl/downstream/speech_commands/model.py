import torch
import torch.nn as nn
import torch.nn.functional as F
from .composition_speech_multihead_attention import composition_speech_multihead_attention
from .composition_text_multihead_attention import composition_text_multihead_attention

"""
Note that this model file was left here due to the legacy reason and is not used in the
SUPERB Benchmark. In SUPERB's speech classification tasks we use linear models, including:

- KS: speech commands
- IC: fluent speech commands
- ER: IEMOCAP emotion classification
- SID: VoxCeleb1 speaker classification

One can trace the following files:

- downstream/speech_commands/config.yaml: downstream_expert.modelrc.select=UtteranceLevel
- downstream/model.py: UtteranceLevel

This "UtteranceLevel" module is used across KS, ER, IC and SID in SUPERB, which first
linearly projects upstream's feature dimension to the same dimension (256), and then
linearly projected to the class number. Hence, it does not contain non-linearity.
"""

class HanAttention(nn.Module):

    def __init__(self,hidden_dim):
        super(HanAttention,self).__init__() 
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim, 1)
                               )
        self.m = nn.Softmax(dim=1)

    def forward(self, inputs):
        v = self.fc(inputs).squeeze(-1)
        alphas = self.m(v)
        outputs = inputs * alphas.unsqueeze(-1)
        outputs = torch.sum(outputs, dim=1)
        return outputs

class Model(nn.Module):
    """
    Not used in SUPERB Benchmark
    """

    def __init__(self, input_dim, output_class_num, hidden_dim):
        super(Model, self).__init__()
        self.composition_speech_multihead_attention = composition_speech_multihead_attention(n_feat=input_dim)
        self.composition_text_multihead_attention = composition_text_multihead_attention(n_feat=input_dim)
        self.attention =  HanAttention(input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(nn.Linear(input_dim * 2, hidden_dim),
                                nn.ReLU(inplace=True),
                                self.dropout,
                                nn.Linear(hidden_dim, output_class_num),
                                )

    def forward(self, features_speech, features_text, audio_mask, text_mask):
        speechs_combined = self.composition_speech_multihead_attention(features_speech, audio_mask)
        speechs_combined = self.layer_norm(features_speech + speechs_combined)
        speech_attention  = self.attention(speechs_combined)

        text_combined = self.composition_text_multihead_attention(features_text, text_mask)
        text_combined  = self.layer_norm(features_text + text_combined)
        text_attention = self.attention(text_combined)

        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        predicted = self.fc(cat_compose)
        return predicted
