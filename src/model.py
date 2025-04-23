import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from transformer import Transformer


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )


@dataclass
class GPTConfig:
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0**0.5)  # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = True
    input_dim: int = 7
    output_dim: int = 1024
    projector_mlp: list = None

    def __post_init__(self):
        # Validate and adjust parameters
        if self.n_embd > 0:
            self.base_scale = min(1.0 / (self.n_embd**0.5), 0.1)  # Cap the scale
        else:
            raise ValueError("n_embd must be positive")

        if self.n_head > self.n_embd:
            raise ValueError("n_head cannot be larger than n_embd")


class ModelUtils:
    @staticmethod
    def justnorm(x, idim=-1, eps=1e-5):
        dtype = x.dtype
        x = x.float()
        norm = x.norm(p=2, dim=idim, keepdim=True)
        # Check for NaN values in norm
        if torch.isnan(norm).any():
            raise ValueError("NaN values detected in norm calculation")
        res = (x / (norm + eps)).to(dtype=dtype)
        # Check for NaN values in result
        if torch.isnan(res).any():
            raise ValueError("NaN values detected in normalization result")
        return res

    @staticmethod
    def get_num_params(model):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def init_weights(module, base_scale):
        """Initialize weights for Linear layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class Projector(nn.Module):
    def __init__(self, config, dims="auto"):
        super().__init__()
        self.config = config

        # Parse dimensions string or create default
        if dims == "auto":
            dims = []
            curr_dim = config.n_embd
            while curr_dim > 2:
                dims.append(curr_dim)
                curr_dim = curr_dim // 4
            dims.append(2)
        else:
            dims = [int(d) for d in dims.split("-")]

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.ModuleDict(
                {
                    # Project to 2x dimension for UV gating
                    "linear": nn.Linear(
                        dims[i], 2 * dims[i + 1], bias=config.bias, dtype=torch.bfloat16
                    ),
                    "proj": nn.Linear(
                        dims[i + 1], dims[i + 1], bias=config.bias, dtype=torch.bfloat16
                    ),
                }
            )

            if config.use_nGPT == 1:
                # Scale parameter for UV
                suv = nn.Parameter(
                    config.base_scale * torch.ones(2 * dims[i + 1], dtype=torch.float32)
                )
                layer["suv"] = nn.ParameterDict({"param": suv})
                layer.suv_init_value = 1.0
                layer.suv_init_scaling = 1.0

            self.layers.append(layer)

        if config.use_nGPT == 0:
            self.rmsnorm_layers = nn.ModuleList(
                [nn.LayerNorm(dims[i]) for i in range(len(dims) - 1)]
            )

        self.silu = nn.SiLU()

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            if self.config.use_nGPT == 0:
                hin = self.rmsnorm_layers[i](h)
            else:
                hin = h.to(dtype=torch.bfloat16)

            # UV gating
            uv = layer["linear"](hin)

            if self.config.use_nGPT == 1:
                suv = layer["suv"]["param"] * (
                    (layer.suv_init_value / layer.suv_init_scaling) * (self.config.n_embd**0.5)
                )
                uv = suv * uv

            u, v = torch.chunk(uv, 2, dim=-1)
            x = u * self.silu(v)
            h = layer["proj"](x.to(dtype=torch.bfloat16))

        return h


class Classifier(nn.Module):
    def __init__(self, config, proj_dims="auto"):
        super().__init__()
        self.config = config

        # Initialize encoder
        self.encoder = Transformer(
            input_dim=7,
            model_dim=1024,
            output_dim=1024,
            n_heads=8,
            dim_feedforward=4096,
            n_layers=4,
            learning_rate=1e-4,
        )

        # Initialize projector
        self.projector = Projector(config, dims=proj_dims)

        # Apply weight initialization
        self.apply(lambda m: ModelUtils.init_weights(m, self.config.base_scale))

        print("number of parameters: %.2fM" % (ModelUtils.get_num_params(self) / 1e6,))

    def forward(self, idx):
        # Get embeddings from encoder
        encoder_output = self.encoder(idx)

        # Project to lower dimensions
        projected_output = self.projector(encoder_output)  # Apply projector -> shape: (b, 2)

        return projected_output

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
