# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-Head Attention layer definition."""

import math
from typing import Tuple

import torch
from torch import nn
from .net_utils import make_pad_mask


class Pattern_extrator(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, n_head: int=4, n_feat: int=256, dropout_rate: float=0.1):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def generate_multi_modal_mask_new(self, b, l_v, l_s, ilens, ylens):
        device = ilens.device
        mask_s_s = (torch.triu(torch.ones(l_s, l_s)) == 1).transpose(0, 1).to(device)
        mask_s_s = mask_s_s.float().masked_fill(mask_s_s == 0, float('-inf')).masked_fill(mask_s_s == 1, float(0.0))    #只有下三角的semantic和semantic的mask矩阵,不mask的区域为0，其他区域为负无穷大
        mask_s_s = mask_s_s.expand(b, l_s, l_s)

        if ylens is not None:
            tgt_mask = (make_pad_mask(ylens)[:, None, :])
            tgt_mask = tgt_mask.expand(b,l_s,l_s)
            mask_s_s = mask_s_s.masked_fill(tgt_mask == 1, float('-inf'))


        mask_v_v = torch.zeros(l_v,l_v).to(device)     #visual和visual的mask，全部为0
        mask_v_v = mask_v_v.expand(b, l_v, l_v)      
        mask_s_v = torch.zeros(l_s,l_v).to(device)
        mask_s_v = mask_s_v.expand(b,l_s,l_v)
        
        if ilens is not None:
            src_mask = (make_pad_mask(ilens)[:, None, :])
            src_mask_v_v = src_mask.expand(b,l_v,l_v)
            mask_v_v = mask_v_v.float().masked_fill(src_mask_v_v == 1, float('-inf'))
            src_mask_s_v = src_mask.expand(b,l_s,l_v)
            mask_s_v = mask_s_v.float().masked_fill(src_mask_s_v == 1, float('-inf'))
        
        mask_v_s = torch.zeros(l_v,l_s).fill_(float('-inf')).to(device)     #visual和semantic的mask，全部为0
        mask_v_s = mask_v_s.expand(b,l_v,l_s)
        mask_t = torch.cat([mask_v_v,mask_v_s],dim=2)
        mask_b = torch.cat([mask_s_v,mask_s_s],dim=2)
        mask = torch.cat([mask_t,mask_b],dim=1)
        return mask

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(2) > 0 :  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, audio_encoder_out: torch.Tensor, text_encoder_out: torch.Tensor, speech_length: torch.Tensor, text_length: torch.Tensor):
        device = audio_encoder_out.device
        B, l_a, C = audio_encoder_out.size()
        B, l_s, C = text_encoder_out.size()
        atten_mask = self.generate_multi_modal_mask_new(B, l_a, l_s, speech_length, text_length).to(device)
        # atten_mask = atten_mask.repeat(self.h, 1, 1)
        multi_modal_input = torch.cat([audio_encoder_out, text_encoder_out], dim=1)
        query = multi_modal_input
        key = multi_modal_input
        value = multi_modal_input

        q, k, v = self.forward_qkv(query, key, value)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, atten_mask)
