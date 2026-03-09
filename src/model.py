import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# Suppress the "not compiled with flash attention" warning on Windows —
# PyTorch will use the efficient fallback SDPA kernel automatically.
warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*", category=UserWarning)


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 2048
    d_model: int = 1024          # embedding dimension
    n_heads: int = 16            # query heads
    n_kv_heads: int = 4          # key/value heads (GQA: 4 KV per 16 Q = 4x reduction)
    n_layers: int = 24           # transformer blocks
    d_ff: int = 4096             # feed-forward inner dim (SwiGLU uses 2/3 * 4 * d_model)
    dropout: float = 0.0         # 0 during training with large datasets
    rope_theta: float = 10000.0  # RoPE base frequency
    tie_embeddings: bool = True  # tie input/output embeddings (saves ~128MB)
    # Special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    think_start_id: int = 3      # <think>
    think_end_id: int = 4        # </think>


# ── RMSNorm ─────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ── Rotary Positional Embeddings ─────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cache.shape[2]:
            self._build_cache(seq_len)
        return self.cos_cache[:, :, :seq_len, :], self.sin_cache[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos, sin) -> tuple:
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ── Grouped Query Attention ──────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5
        self.n_groups = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.context_length * 2, config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q, T)
        q, k = apply_rotary(q, k, cos, sin)

        # Expand KV heads to match Q heads (GQA)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Flash Attention via PyTorch SDPA (uses FlashAttention2 kernel on Ampere+)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(mask is None),
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)


# ── SwiGLU Feed-Forward ──────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 2/3 * 4 * d_model keeps param count similar to standard FFN
        hidden = int(2 / 3 * config.d_ff)
        hidden = (hidden + 63) // 64 * 64  # round to multiple of 64 for efficiency
        self.gate_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.up_proj   = nn.Linear(config.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# ── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn      = GroupedQueryAttention(config)
        self.ff_norm   = RMSNorm(config.d_model)
        self.ff        = SwiGLU(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ff(self.ff_norm(x))
        return x


# ── Full Model ───────────────────────────────────────────────────────────────

class ThinkingLM(nn.Module):
    # Custom GPT-style language model with chain-of-thought reasoning support.
    # Special <think> / </think> tokens allow the model to reason before answering.

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm   = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ThinkingLM] {n_params / 1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        B, T = input_ids.shape
        assert T <= self.config.context_length, f"Sequence length {T} > context {self.config.context_length}"

        x = self.emb_dropout(self.token_emb(input_ids))

        use_ckpt = getattr(self, "_use_grad_checkpoint", False) and self.training
        for block in self.blocks:
            if use_ckpt:
                x = grad_ckpt(block, x, mask, use_reentrant=True)
            else:
                x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        think: bool = True,
        stream: bool = False,
    ):
        #Generate tokens with optional chain-of-thought thinking.
        #If think=True, prepends <think> token and lets the model reason first.

        device = input_ids.device
        tokens = input_ids.clone()

        if think and self.config.think_start_id is not None:
            think_tok = torch.tensor([[self.config.think_start_id]], device=device)
            tokens = torch.cat([tokens, think_tok], dim=1)

        generated = []
        thinking_done = not think

        for _ in range(max_new_tokens):
            ctx = tokens if tokens.shape[1] <= self.config.context_length else tokens[:, -self.config.context_length:]

            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok_id in set(tokens[0].tolist()):
                    logits[0, tok_id] /= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-K
            if top_k > 0:
                topk_vals = torch.topk(logits, top_k).values
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            # Top-P (nucleus)
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask_remove = cumsum - sorted_probs > top_p
                sorted_probs[mask_remove] = 0.0
                sorted_probs /= sorted_probs.sum()
                sampled = torch.multinomial(sorted_probs[0], 1)
                next_tok_id = sorted_idx[0, sampled[0]].item()
            else:
                next_tok_id = torch.multinomial(probs[0], 1).item()

            next_tok_tensor = torch.tensor([[next_tok_id]], dtype=torch.long, device=device)
            tokens = torch.cat([tokens, next_tok_tensor], dim=1)
            generated.append(next_tok_id)

            if stream:
                yield next_tok_id
                continue

            # Stop on </think> if still in thinking mode
            if not thinking_done and next_tok_id == self.config.think_end_id:
                thinking_done = True

            # Stop on EOS
            if next_tok_id == self.config.eos_token_id:
                break

        if not stream:
            yield generated

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimate_vram_mb(self) -> float:
        """Rough BF16 VRAM estimate for the model weights."""
        return self.get_num_params() * 2 / (1024 ** 2)
