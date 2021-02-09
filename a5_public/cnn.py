#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size, kernel_size=5):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(char_embed_size, word_embed_size, kernel_size)
        self.relu = nn.ReLU()
        self.max_pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv1d(x_reshaped)
        return self.max_pooling(self.relu(x_conv)).squeeze(dim=2)


if __name__ == '__main__':
    model = CNN(3, 2, 4)

    assert(model.conv1d.weight.shape == torch.Size([2, 3, 4]))
    assert(model.conv1d.bias.shape == torch.Size([2]))

    model.conv1d.weight.data = torch.Tensor([[[0.78, -0.11, 1.3, 0.93],
                    [0.35, 0.45, 0.5, -0.61],
                    [-0.2, -0.38, 0.66, 0.42]],
                   [[-0.78, 0.11, -1.3, -0.93],
                    [-0.35, -0.45, -0.5, 0.61],
                    [0.2, 0.38, -0.66, -0.42]]])
    model.conv1d.bias.data = torch.Tensor([-1, 2])
    input = torch.Tensor([[[0.33, 0.81, -0.55, -0.95, 0.11],
                  [0.59, -0.77, 0.49, -0.73, -0.84],
                  [-0.13, 0.42, 0.05, -0.78, 0.55]],
                  [[0.21, 0.84, 0.99, -0.3, -0.81],
                   [0.7, 0.55, 0.47, 0.62, 0.38],
                   [-0.3, -0.2, 0.8, 0.67, -0.12]]])
    output = torch.Tensor([[0, 3.3081],
                           [1.3740999999999999, 1.9869999999999999]])
    with torch.no_grad():
        pred = model(input)
        assert(type(pred) == torch.Tensor)
        assert(pred.shape == torch.Size([2, 2]))
        assert(np.allclose(pred.numpy(), output))
### END YOUR CODE