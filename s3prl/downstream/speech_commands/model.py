import torch
import torch.nn as nn
import torch.nn.functional as F
from .pattern_extrator import Pattern_extrator
from .pattern_discriminator import Pattern_discriminator

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

class Model(nn.Module):
    """
    Not used in SUPERB Benchmark
    """

    def __init__(self, input_dim, output_class_num, hidden_dim):
        super(Model, self).__init__()
        self.pattern_extractor = Pattern_extrator(n_feat=input_dim)
        self.pattern_discriminator = Pattern_discriminator(n_feat=hidden_dim, vocab=output_class_num)

    def forward(self, features_audio, features_text, len_audio, len_text):
        pattern_out = self.pattern_extractor(features_audio, features_text, len_audio, len_text)
        predicted = self.pattern_discriminator(pattern_out)
        return predicted
