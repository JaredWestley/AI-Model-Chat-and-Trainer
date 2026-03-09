"""
Main training script for ThinkingLM.
Usage:
    python train.py                  # Train with defaults
    python train.py --steps 50000    # Custom step count
    python train.py --resume checkpoints/ckpt_latest.pt
    python train.py --config config.json
"""

import sys
import os
import json
import platform
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch


def main():
    parser = argparse.ArgumentParser(description="Train ThinkingLM")
    parser.add_argument("--steps",       type=int,   default=100_000)
    parser.add_argument("--batch",       type=int,   default=4)
    parser.add_argument("--accum",       type=int,   default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--context",     type=int,   default=2048)
    parser.add_argument("--workers",     type=int,   default=2, help="DataLoader workers (train only; val always uses 0)")
    parser.add_argument("--resume",      type=str,   default=None, help="Resume from checkpoint")
    parser.add_argument("--compile",     action="store_true", help="Enable torch.compile (Linux only)")
    parser.add_argument("--device",      type=str,   default="cuda")
    parser.add_argument("--data-dir",    type=str,   default="data")
    parser.add_argument("--ckpt-dir",    type=str,   default="checkpoints")
    args = parser.parse_args()

    # torch.compile needs Triton which is Linux-only; disable on Windows automatically
    on_windows = platform.system() == "Windows"
    use_compile = args.compile and not on_windows
    if on_windows and args.compile:
        print("[Setup] NOTE: --compile ignored on Windows (Triton not available). Running eager mode.")

    print("=" * 60)
    print("  ThinkingLM Training System")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("  GPU:     Not available (training on CPU)")
    print("=" * 60)

    # ── Step 1: Tokenizer ────────────────────────────────────────────────────
    from src.tokenizer import ThinkingTokenizer

    tokenizer_path = Path(args.data_dir) / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"\n[Setup] Loading existing tokenizer from {tokenizer_path}")
        tokenizer = ThinkingTokenizer.load(str(tokenizer_path))
    else:
        print("\n[Setup] Tokenizer not found. Bootstrapping from GPT-2...")
        tokenizer = ThinkingTokenizer.from_pretrained_gpt2(str(tokenizer_path))

    # ── Step 2: Model ────────────────────────────────────────────────────────
    from src.model import ThinkingLM, ModelConfig

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context,
        d_model=1024,
        n_heads=16,
        n_kv_heads=4,
        n_layers=24,
        d_ff=4096,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        think_start_id=tokenizer.think_start_id,
        think_end_id=tokenizer.think_end_id,
    )

    model = ThinkingLM(model_config)

    vram_est = model.estimate_vram_mb()
    print(f"\n[Model] ~{model.get_num_params()/1e6:.1f}M parameters")
    print(f"[Model] Estimated VRAM (weights): {vram_est:.0f} MB")
    print(f"[Model] With BF16 + grad ckpt, total VRAM estimate: ~{vram_est * 3:.0f} MB")

    # ── Step 3: Datasets ─────────────────────────────────────────────────────
    print("\n[Data] Preparing datasets...")
    from src.dataset import prepare_all_datasets, get_dataloader

    dataset = prepare_all_datasets(tokenizer, args.data_dir, args.context)

    # 90/10 train/val split
    val_size   = min(len(dataset) // 10, 5000)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Windows multiprocessing has a pickle size limit — val loader must use num_workers=0
    # Train loader works fine with workers for prefetching
    train_loader = get_dataloader(train_ds, batch_size=args.batch, num_workers=args.workers)
    val_loader   = get_dataloader(val_ds,   batch_size=args.batch, num_workers=0, shuffle=False)

    print(f"[Data] Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")

    # ── Step 4: Training config ──────────────────────────────────────────────
    from src.trainer import TrainingConfig, Trainer

    train_config = TrainingConfig(
        max_steps=args.steps,
        batch_size=args.batch,
        gradient_accumulation=args.accum,
        context_length=args.context,
        learning_rate=args.lr,
        warmup_steps=min(2000, args.steps // 20),
        lr_decay_steps=args.steps,
        device=args.device,
        compile_model=use_compile,
        gradient_checkpointing=True,
        checkpoint_dir=args.ckpt_dir,
        resume_from=args.resume,
    )

    # ── Step 5: Train ────────────────────────────────────────────────────────
    trainer = Trainer(model, train_config)
    trainer.train(train_loader, val_loader)

    print("\n[Done] Training complete!")
    print(f"[Done] Checkpoint saved to: {args.ckpt_dir}/ckpt_final.pt")
    print(f"[Done] To chat: python chat_cli.py")
    print(f"[Done] To chat (web): python chat_web.py")


if __name__ == "__main__":
    main()
