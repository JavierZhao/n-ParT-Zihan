import math
import inspect
from dataclasses import dataclass


import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )


class Block(nn.Module):
    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        # Combined QKV projection
        self.qkv = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=torch.float32
        )
        self.attention = ScaledDotProductAttention()
        self.att_c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.float32
        )

        self.c_fc = nn.Linear(
            config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=torch.float32
        )
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.float32
        )

        # Rest of the initialization code remains the same
        if config.use_nGPT == 0:
            self.rmsnorm_att = LayerNorm(config.n_embd)
            self.rmsnorm_mlp = LayerNorm(config.n_embd)

        if config.use_nGPT == 1:
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(
                self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(
                self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.sqk_init_value = 1.0
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(
                self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(
                self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32)
            )

    def forward(self, h, mask=None, out_dir=None):
        B, T, C = h.size()
        if C != self.config.n_embd:
            raise ValueError(f"Expected embedding dim {self.config.n_embd}, got {C}")

        # Cast input to desired dtype at the start
        h = h.to(dtype=torch.float32)

        hin = h
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_att(h)

        # Split QKV
        qkv = self.qkv(hin)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head).transpose(
            1, 2
        )
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head).transpose(
            1, 2
        )
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head).transpose(
            1, 2
        )

        if self.config.use_nGPT == 1:
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
                1, self.config.n_head, 1, self.config.n_embd // self.config.n_head
            )
            q = sqk * ModelUtils.justnorm(q)
            k = sqk * ModelUtils.justnorm(k)

        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if self.config.use_nGPT == 0:
            softmax_scale = 1.0 / sqrt_head_dim
        if self.config.use_nGPT == 1:
            softmax_scale = sqrt_head_dim

        # Use PyTorch 2.0's native flash attention when available
        # y = F.scaled_dot_product_attention(
        #     q.to(dtype=torch.float32),
        #     k.to(dtype=torch.float32),
        #     v.to(dtype=torch.float32),
        #     attn_mask=mask,
        #     dropout_p=0.0,
        #     is_causal=False,
        #     scale=softmax_scale,
        # )
        y = self.attention(
            q.to(dtype=torch.float32),
            k.to(dtype=torch.float32),
            v.to(dtype=torch.float32),
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )

        y = y.transpose(2, 1)
        y = y.contiguous().view(B, T, self.config.n_embd)

        y = y.to(dtype=torch.float32)
        # Compute a boolean mask that is True for jets that contain any NaN.
        problematic_jets = torch.isnan(y).any(dim=(1, 2))
        if problematic_jets.any():
            # Optionally save debugging information.
            torch.save(y, f"{out_dir}/y.pt")
            torch.save(hin, f"{out_dir}/hin.pt")
            torch.save(mask, f"{out_dir}/mask.pt")
            print(f"Found NaNs in y at indices: {torch.where(torch.isnan(y))}")

            # Replace all representations of problematic jets with zeros.
            y[problematic_jets] = 0.0
        h_att = self.att_c_proj(y)

        if self.config.use_nGPT == 0:
            h = h_att
        if self.config.use_nGPT == 1:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = ModelUtils.justnorm(h)
            if torch.isnan(h_att).any():
                print(f"h_att is nan, {torch.where(torch.isnan(h_att))}")
                # save h_att to file
                torch.save(h_att, f"{out_dir}/h_att.pt")
                print(f"input h contains nan: {torch.isnan(h).any()}")
            B_norm = ModelUtils.justnorm(h_att)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = ModelUtils.justnorm(res)

        hin = h.to(dtype=torch.float32)
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if self.config.use_nGPT == 1:
            suv = self.suv * (
                (self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5)
            )
            uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = (u * self.silu(v)).to(dtype=torch.float32)
        h_mlp = self.mlp_c_proj(x_mlp)

        if self.config.use_nGPT == 0:
            h = h + h_mlp
        if self.config.use_nGPT == 1:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = ModelUtils.justnorm(h)
            B_norm = ModelUtils.justnorm(h_mlp)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = ModelUtils.justnorm(res)

        return h.to(dtype=torch.float32)


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


class LayerNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embdim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check for NaN values in input
        if torch.isnan(x).any():
            raise ValueError("NaN values detected in LayerNorm input")

        # Apply PyTorch's LayerNorm
        xnorm = self.layer_norm(x)

        # Check for NaN values in result
        if torch.isnan(xnorm).any():
            raise ValueError("NaN values detected in LayerNorm result")

        return xnorm


class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        # Check for NaN values in norm
        if torch.isnan(norm).any():
            raise ValueError("NaN values detected in RMSNorm calculation")
        xnorm = x * torch.rsqrt(norm + self.eps)
        if torch.isnan(xnorm).any():
            raise ValueError("NaN values detected in RMSNorm result")
        # xnorm = xnorm.to(dtype=dtype)
        return (xnorm * self.weight).to(dtype=dtype)


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


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.use_nGPT:
            print("normalized")
        else:
            print("NOT normalized")

        # Input projection parameters
        self.input_proj = nn.Linear(
            config.input_dim, config.n_embd, bias=config.bias, dtype=torch.float32
        )

        if config.use_nGPT == 0:
            self.rmsnorm_input = LayerNorm(config.n_embd)
        else:
            input_alpha_init_value = 0.05
            input_alpha_init_scaling = config.base_scale
            self.input_alpha = nn.Parameter(
                input_alpha_init_scaling * torch.ones(config.n_embd, dtype=torch.float32)
            )
            self.input_alpha_init_value = input_alpha_init_value
            self.input_alpha_init_scaling = input_alpha_init_scaling

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config, il) for il in range(config.n_layer)])

        # Initialize weights
        self.apply(lambda m: ModelUtils.init_weights(m, self.config.base_scale))
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.n_layer)
                )

        if config.use_nGPT == 1:
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            self.sz = torch.nn.Parameter(
                self.sz_init_scaling * torch.ones(config.output_dim, dtype=torch.float32)
            )

        if config.use_nGPT == 0:
            self.rmsnorm_f = LayerNorm(config.n_embd)

    def get_num_params(self):
        return ModelUtils.get_num_params(self)

    def make_mask(self, is_padded):
        # is_padded: [B, T], where True indicates padded tokens.
        # We want a mask of shape [B, T, T] that indicates valid keys for each query.
        valid = ~is_padded  # [B, T]  (True means valid)
        mask = valid.unsqueeze(1).expand(-1, is_padded.size(1), -1)  # [B, T, T]
        mask = mask.unsqueeze(1)  # [B, 1, T, T]
        return mask

    def forward(self, inpt, out_dir=None):
        x = inpt + 0.0
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, particles, features), got {x.dim()}D")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected input dim {self.config.input_dim}, got {x.size(-1)}")

        b, p, _ = x.size()
        # Create padding mask from first feature (assumes 0 = padding token)
        is_padded = x[:, :, 0] == 0  # [batch, seq_len]
        attention_mask = self.make_mask(is_padded)

        # Input projection with nGPT rules
        if self.config.use_nGPT == 0:
            x = self.input_proj(x)
            if hasattr(self, "rmsnorm_input"):
                x = self.rmsnorm_input(x)
        else:
            x = self.input_proj(x.to(dtype=torch.float32))

        for block in self.blocks:
            x = block(x, attention_mask, out_dir)
        return x

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
                        dims[i], 2 * dims[i + 1], bias=config.bias, dtype=torch.float32
                    ),
                    "proj": nn.Linear(
                        dims[i + 1], dims[i + 1], bias=config.bias, dtype=torch.float32
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
            self.rmsnorm_layers = nn.ModuleList([LayerNorm(dims[i]) for i in range(len(dims) - 1)])

        self.silu = nn.SiLU()

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            if self.config.use_nGPT == 0:
                hin = self.rmsnorm_layers[i](h)
            else:
                hin = h.to(dtype=torch.float32)

            # UV gating
            uv = layer["linear"](hin)

            if self.config.use_nGPT == 1:
                suv = layer["suv"]["param"] * (
                    (layer.suv_init_value / layer.suv_init_scaling) * (self.config.n_embd**0.5)
                )
                uv = suv * uv

            u, v = torch.chunk(uv, 2, dim=-1)
            x = u * self.silu(v)
            h = layer["proj"](x.to(dtype=torch.float32))

        return h


class Classifier(nn.Module):
    def __init__(self, config, proj_dims="auto"):
        super().__init__()
        self.config = config

        # Initialize encoder
        self.encoder = Encoder(config)

        # Initialize projector
        self.projector = Projector(config, dims=proj_dims)

        # Apply weight initialization
        self.apply(lambda m: ModelUtils.init_weights(m, self.config.base_scale))

        print("number of parameters: %.2fM" % (ModelUtils.get_num_params(self) / 1e6,))

    def forward(self, idx, out_dir=None):
        # Get embeddings from encoder
        encoder_output = self.encoder(idx, out_dir)

        encoder_output = encoder_output.sum(
            dim=1
        )  # Sum along particles dimension -> shape: (b, output_dim)

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
