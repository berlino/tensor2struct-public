import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger("tensor2struct")


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, dropout=0.1):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size * 4
        self.w_1 = nn.Linear(input_size * 2, hidden_size)
        self.w_2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], -1)
        return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))


class Bilinear(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1, include_id=True):
        super().__init__()
        self.include_id = include_id
        if include_id:
            output_size = output_size - 1

        if output_size == 0:
            assert include_id
            logger.info("Only dot-product is enabled for computing matching scores")
            self.enable_learnable_bilinear = False
        else:
            self.enable_learnable_bilinear = True

        if self.enable_learnable_bilinear:
            hidden_size = int(input_size / 4)
            self.bilinear_mat = nn.Parameter(
                torch.nn.init.normal_(torch.empty(input_size, hidden_size, output_size))
            )
        self.dropout = nn.Dropout(dropout)

    def apply_linear(self, x):
        h = torch.einsum("bcl,lr->bcr", [x, self.lin_mat])
        h = h + self.lin_bias.expand_as(h)
        return h

    def apply_bilinear(self, x1, x2):
        h1 = torch.einsum("bcl,lho->bcoh", [x1, self.bilinear_mat])
        h2 = torch.einsum("bcl,lho->bcoh", [x2, self.bilinear_mat])
        scores = torch.einsum("bcol,bcor->bco", [h1, h2])  # dot product

        if self.include_id:
            x1, x2 = x1.contiguous(), x2.contiguous()
            id_score = torch.einsum("bci,bci->bc", [x1, x2]).unsqueeze(2)
            scores = torch.cat([scores, id_score], dim=2)
        return scores

    def forward(self, x1, x2):
        if self.enable_learnable_bilinear:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
            return self.apply_bilinear(x1, x2)
        else:
            id_score = torch.einsum("bci,bci->bc", [x1, x2]).unsqueeze(2)
            return id_score
