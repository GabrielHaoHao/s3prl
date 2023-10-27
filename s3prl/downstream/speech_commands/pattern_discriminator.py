import torch
from torch import nn

class Pattern_discriminator(nn.Module):
    def __init__(self, n_feat: int=256, vocab: int=2):
        super().__init__()
        self.gru = nn.GRU(n_feat, n_feat)
        self.fc = nn.Linear(n_feat, vocab)
        self.out = nn.Sigmoid()
    
    def forward(
        self,
        x: torch.Tensor,
    ):
        output, hidden = self.gru(x)
        # fc_input = output[-1]
        # fc_input = torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim = 1)
        fc_input = output[:, -2:-1, :]
        out = self.out(self.fc(fc_input))
        return torch.squeeze(out)