#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(word_embed_size, word_embed_size)
        self.gate = nn.Linear(word_embed_size, word_embed_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.relu(self.proj(x))
        x_gate = self.sigmoid(self.gate(x))
        return x_gate * x_proj + (1 - x_gate) * x


if __name__ == '__main__':
    model = Highway(2)

    assert(model.proj.weight.shape == torch.Size([2, 2]))
    assert(model.gate.weight.shape == torch.Size([2, 2]))
    assert(model.proj.bias.shape == torch.Size([2]))
    assert(model.gate.bias.shape == torch.Size([2]))

    model.proj.weight.data = torch.Tensor([[0.3, 0.5], [0.7, -0.1]])
    model.gate.weight.data = torch.Tensor([[0.9, -0.6], [-0.3, 0.1]])
    model.proj.bias.data = torch.Tensor([-3, 6])
    model.gate.bias.data = torch.Tensor([2, 5])
    x = torch.Tensor([[1, 3], [-2, 0.5], [-0.8, 2.3], [1.7, 8.5]])
    output = np.array([[ 0.24973989,  6.37724431], [-1.04995837,  4.53580399], [-0.41998335,  5.19779671], [1.71033298,  6.35030964]])
    with torch.no_grad():
        pred = model(x)
        assert(type(pred) == torch.Tensor)
        assert(np.allclose(pred.numpy(), output))
### END YOUR CODE

