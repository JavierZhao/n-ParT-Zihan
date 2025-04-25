import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# class for transformer network
class Transformer(nn.Module):
    # define and intialize the structure of the neural network
    def __init__(
        self,
        input_dim,
        model_dim,
        output_dim,
        n_heads,
        dim_feedforward,
        n_layers,
        learning_rate,
        n_GPT=False,
        n_head_layers=2,
        dropout=0.1,
        opt="adam",
        eps=1e-8,
    ):
        super().__init__()
        # define hyperparameters
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_head_layers = n_head_layers
        self.dropout = dropout
        self.eps = eps  # epsilon for Adam
        self.n_GPT = n_GPT
        # define subnetworks
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dim, n_heads, dim_feedforward=dim_feedforward, dropout=dropout
            ),
            n_layers,
        )
        # head_layers have output_dim
        if n_head_layers == 0:
            self.head_layers = []
        else:
            if self.n_GPT:
                self.norm_layers = nn.ModuleList([nn.LayerNorm(model_dim)])
            self.head_layers = nn.ModuleList([nn.Linear(model_dim, output_dim)])
            for i in range(n_head_layers - 1):
                if self.n_GPT:
                    self.norm_layers.append(nn.LayerNorm(output_dim))
                self.head_layers.append(nn.Linear(output_dim, output_dim))
        # option to use adam or sgd
        if opt == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, eps=self.eps
            )
        if opt == "sgdca" or opt == "sgdslr" or opt == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )

    def forward(self, inpt, use_mask=True):
        """
        inpt: (B, n_constit, feature_dim) with feature_dim >=1 and pT in [:, :, 0]
        """
        x = inpt.clone()                 # (B, S, D)
        if use_mask:
            # build a Boolean pad mask: True where pT==0
            # shape: (B, S)
            src_key_padding_mask = (x[:, :, 0] == 0)
        else:
            src_key_padding_mask = None
    
        # project embeddings & permute for Transformer
        x = self.embedding(x)            # (B, S, model_dim)
        x = x.permute(1, 0, 2)           # (S, B, model_dim)
    
        # pass through TransformerEncoder using key-padding mask
        # this will never attend *from* or *to* a padded position
        x = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )                                # (S, B, model_dim)
    
        # bring back to (B, S, model_dim)
        x = x.permute(1, 0, 2)
    
        if use_mask:
            # zero out the representations of masked tokens before summing
            x = x.masked_fill(
                src_key_padding_mask.unsqueeze(-1),
                0.0
            )                            # (B, S, model_dim)
    
        # aggregate over sequence dim
        x = x.sum(dim=1)                 # (B, model_dim)
        return self.head(x)              # (B, output_dim)


    def head(self, x):
        """
        calculates output of the head if it exists, i.e. if n_head_layer>0
        returns multiple representation layers if asked for by mult_reps = True
        input:  x shape=(batchsize, model_dim)
                mult_reps boolean
        output: reps shape=(batchsize, output_dim)                  for mult_reps=False
                reps shape=(batchsize, number_of_reps, output_dim)  for mult_reps=True
        """
        relu = nn.ReLU()
        with torch.autocast(device_type="cuda", enabled=False):
            for i, layer in enumerate(self.head_layers):
                if self.n_GPT:
                    x = self.norm_layers[i](x)
                    x = relu(x)
                    x = layer(x)
                return x

    def forward_batchwise(self, x, batch_size, use_mask=False, use_continuous_mask=False):
        device = next(self.parameters()).device
        with torch.no_grad():
            if self.n_head_layers == 0:
                rep_dim = self.model_dim
                number_of_reps = 1
            elif self.n_head_layers > 0:
                rep_dim = self.output_dim
                number_of_reps = self.n_head_layers + 1
            out = torch.empty(x.size(0), number_of_reps, rep_dim)
            idx_list = torch.split(torch.arange(x.size(0)), batch_size)
            for idx in idx_list:
                output = (
                    self(
                        x[idx].to(device),
                        use_mask=use_mask,
                        use_continuous_mask=use_continuous_mask,
                        mult_reps=True,
                    )
                    .detach()
                    .cpu()
                )
                out[idx] = output
        return out

    def make_mask(self, pT_zero):
        """
        Input: batch of bools of whether pT=0, shape (batchsize, n_constit)
        Output: mask for transformer model which masks out constituents with pT=0, shape (batchsize*n_transformer_heads, n_constit, n_constit)
        mask is added to attention output before softmax: 0 means value is unchanged, -inf means it will be masked
        """
        n_constit = pT_zero.size(1)
        pT_zero = torch.repeat_interleave(pT_zero, self.n_heads, axis=0)
        pT_zero = torch.repeat_interleave(pT_zero[:, None], n_constit, axis=1)
        mask = torch.zeros(pT_zero.size(0), n_constit, n_constit)
        mask[pT_zero] = -np.inf
        return mask
