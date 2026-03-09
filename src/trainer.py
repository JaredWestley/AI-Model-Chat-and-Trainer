import os
import time
import math
import json
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler


@dataclass
class TrainingConfig:
    # Core
    max_steps: int = 100_000
    eval_interval: int = 500
    save_interval: int = 100
    log_interval: int = 50

    # Batch / sequence
    batch_size: int = 4
    gradient_accumulation: int = 16
    context_length: int = 2048

    # Learning rate
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    lr_decay_steps: int = 100_000

    # Regularisation
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.0

    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = True
    gradient_checkpointing: bool = True

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    tokenizer_path: str = "data/tokenizer.json"
    data_dir: str = "data"

    # Resume
    resume_from: Optional[str] = None


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model  = model
        self.config = config
        self.step   = 0
        self.best_loss = float("inf")

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            print("[Trainer] WARNING: No CUDA device found. Training on CPU (very slow).")

        self._setup_dtype()
        self._setup_model()
        self._setup_optimizer()

        self.log_file = open(Path(config.log_dir) / "train_log.jsonl", "a")
        self._print_config()

    def _setup_dtype(self):
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map.get(self.config.dtype, torch.bfloat16)
        self.use_amp = self.dtype in (torch.bfloat16, torch.float16)
        # BF16 doesn't need GradScaler (no overflow), float16 does
        self.scaler = GradScaler() if self.dtype == torch.float16 else None

    def _setup_model(self):
        self.model = self.model.to(self.device)

        # Enable gradient checkpointing BEFORE any compile attempt
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # torch.compile requires Triton which is Linux-only.
        # On Windows we auto-detect and skip it gracefully.
        if self.config.compile_model and hasattr(torch, "compile"):
            if self._triton_available():
                print("[Trainer] Compiling model with torch.compile (reduce-overhead)...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[Trainer] Compilation complete.")
            else:
                print("[Trainer] torch.compile skipped — Triton is not available on Windows.")
                print("[Trainer] Running in eager mode (still fast with BF16 + grad checkpointing).")

        n_params = sum(p.numel() for p in self.model.parameters())
        vram_est = n_params * 2 / (1024**2)
        print(f"[Trainer] Model: {n_params/1e6:.1f}M params, ~{vram_est:.0f}MB VRAM (weights only)")

    def _triton_available(self) -> bool:
        try:
            import triton  # noqa: F401
            return True
        except ImportError:
            return False

    def _enable_gradient_checkpointing(self):
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        # Reach through to the raw model (not compiled wrapper)
        target = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

        if not hasattr(target, "blocks"):
            print("[Trainer] WARNING: model has no .blocks — gradient checkpointing skipped")
            return

        class CheckpointedModuleList(torch.nn.ModuleList):
            """Drop-in replacement for nn.ModuleList that runs each block under grad checkpoint."""
            def forward_block(self, block, x, mask):
                return grad_ckpt(block, x, mask, use_reentrant=True)

        # Replace the blocks list with a checkpointed version
        original_blocks = target.blocks
        ckpt_blocks = CheckpointedModuleList(list(original_blocks))
        target.blocks = ckpt_blocks

        # Patch model.forward to call ckpt_blocks.forward_block instead of direct call
        # We do this by replacing the block iteration in the model's forward method
        # The cleanest approach: store a flag and check it in forward
        target._use_grad_checkpoint = True

        count = len(list(original_blocks))
        print(f"[Trainer] Gradient checkpointing enabled on {count} transformer blocks")

    def _setup_optimizer(self):
        # Separate weight decay groups (don't decay biases/norms)
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "norm" in name or "bias" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        # fused AdamW is faster but requires CUDA and a compatible PyTorch build.
        # Try it first; fall back silently if not supported (e.g. on Windows builds).
        adamw_kwargs = dict(
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        try:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay,    "weight_decay": self.config.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                fused=(self.device.type == "cuda"),
                **adamw_kwargs,
            )
            print("[Trainer] Using fused AdamW")
        except (TypeError, RuntimeError):
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay,    "weight_decay": self.config.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                **adamw_kwargs,
            )
            print("[Trainer] Using standard AdamW")

    def _get_lr(self, step: int) -> float:
        cfg = self.config
        if step < cfg.warmup_steps:
            return cfg.learning_rate * step / cfg.warmup_steps
        if step > cfg.lr_decay_steps:
            return cfg.min_lr
        progress = (step - cfg.warmup_steps) / (cfg.lr_decay_steps - cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _print_config(self):
        effective_batch = self.config.batch_size * self.config.gradient_accumulation
        tokens_per_step = effective_batch * self.config.context_length
        print(f"""
[Trainer] Configuration:
  Device:         {self.device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'})
  Dtype:          {self.config.dtype}
  Batch size:     {self.config.batch_size} × {self.config.gradient_accumulation} accum = {effective_batch} effective
  Tokens/step:    {tokens_per_step:,}
  Context length: {self.config.context_length}
  LR:             {self.config.learning_rate} → {self.config.min_lr} (cosine)
  Warmup:         {self.config.warmup_steps} steps
  Max steps:      {self.config.max_steps:,}
  Compile:        {self.config.compile_model}
  Grad ckpt:      {self.config.gradient_checkpointing}
""")

    def save_checkpoint(self, tag: str = "latest"):
        path = Path(self.config.checkpoint_dir) / f"ckpt_{tag}.pt"
        # Unwrap compiled model
        model_state = self.model._orig_mod.state_dict() if hasattr(self.model, "_orig_mod") else self.model.state_dict()
        torch.save({
            "step": self.step,
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": asdict(self.config),
        }, str(path))
        print(f"[Trainer] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        target = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        target.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[Trainer] Resumed from step {self.step}")

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)

        self.model.train()
        total_tokens = 0
        t0 = time.time()

        data_iter = iter(train_loader)

        print(f"\n[Trainer] Starting training from step {self.step}...")

        while self.step < self.config.max_steps:
            lr = self._get_lr(self.step)
            self._set_lr(lr)

            loss_accum = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation loop
            for micro_step in range(self.config.gradient_accumulation):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    _, loss = self.model(x, labels=y)
                    loss = loss / self.config.gradient_accumulation

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += loss.item()
                total_tokens += x.numel()

            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.step += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                t1 = time.time()
                dt = t1 - t0
                tokens_per_sec = (self.config.log_interval * self.config.batch_size *
                                  self.config.gradient_accumulation * self.config.context_length) / dt
                vram_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                vram_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0

                log = {
                    "step": self.step,
                    "loss": loss_accum,
                    "lr": lr,
                    "grad_norm": grad_norm.item(),
                    "tokens_per_sec": int(tokens_per_sec),
                    "vram_used_gb": round(vram_used, 2),
                    "vram_reserved_gb": round(vram_reserved, 2),
                    "total_tokens": total_tokens,
                }

                print(
                    f"step {self.step:>6} | loss {loss_accum:.4f} | "
                    f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                    f"{tokens_per_sec:,.0f} tok/s | "
                    f"VRAM {vram_used:.1f}/{vram_reserved:.1f}GB"
                )
                self.log_file.write(json.dumps(log) + "\n")
                self.log_file.flush()
                t0 = time.time()

            # Evaluation
            if val_loader and self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate(val_loader)
                print(f"[Eval] step {self.step} | val_loss {val_loss:.4f}")
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best")
                self.model.train()

            # Periodic checkpoint — save numbered copy every 1000 steps, latest every save_interval
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint("latest")
            if self.step % 1000 == 0:
                self.save_checkpoint(f"step_{self.step:07d}")

        self.save_checkpoint("final")
        print("[Trainer] Training complete!")
        self.log_file.close()

    @torch.inference_mode()
    def evaluate(self, val_loader: DataLoader, max_batches: int = 50) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for x, y in val_loader:
            if count >= max_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                _, loss = self.model(x, labels=y)
            total_loss += loss.item()
            count += 1
        return total_loss / max(count, 1)


def auto_detect_batch_size(model: nn.Module, context_length: int, device: torch.device, dtype) -> int:
    # Find the largest batch size that fits in VRAM.
    # Starts at 8 and halves until it fits.
    
    print("[Trainer] Auto-detecting optimal batch size...")
    for bs in [8, 4, 2, 1]:
        try:
            dummy = torch.randint(0, 1000, (bs, context_length), device=device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                model(dummy, labels=dummy)
            torch.cuda.empty_cache()
            print(f"[Trainer] Batch size {bs} fits in VRAM")
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[Trainer] Batch size {bs} OOM, trying smaller...")
    return 1
