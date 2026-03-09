"""
Dataset pipeline for ThinkingLM.
Downloads and preprocesses three dataset types:
  1. OpenWebText   — general language/knowledge
  2. CodeParrot    — GitHub code (Python, JS, C++, etc.)
  3. OpenOrca      — chain-of-thought reasoning (teaches <think> patterns)

Uses streaming to avoid loading everything into RAM.
Tokenised data is saved as memory-mapped numpy files for fast training.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# ── Disk-backed token dataset (fast mmap reads) ──────────────────────────────

class TokenDataset(Dataset):
    # Loads pre-tokenised data from a .npy memory-mapped file.

    def __init__(self, data_path: str, context_length: int = 2048):
        self.context_length = context_length
        data = np.load(data_path, mmap_mode="r")
        self.data = data
        n_tokens = len(data)
        self.n_samples = (n_tokens - 1) // context_length
        print(f"[Dataset] Loaded {data_path}: {n_tokens:,} tokens → {self.n_samples:,} samples")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.context_length
        end   = start + self.context_length + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class MultiDataset(Dataset):
    # Combines multiple TokenDatasets with configurable sampling weights.

    def __init__(self, datasets: List[TokenDataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        total = sum(len(d) for d in datasets)
        if weights is None:
            self.weights = [len(d) / total for d in datasets]
        else:
            s = sum(weights)
            self.weights = [w / s for w in weights]

        # Build cumulative index mapping
        self._build_index()

    def _build_index(self):
        self.cumulative = []
        running = 0
        for d in self.datasets:
            running += len(d)
            self.cumulative.append(running)
        self._total = running

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        for i, cum in enumerate(self.cumulative):
            if idx < cum:
                offset = idx - (self.cumulative[i - 1] if i > 0 else 0)
                return self.datasets[i][offset]
        raise IndexError(f"Index {idx} out of range")


# ── Dataset downloaders ──────────────────────────────────────────────────────

def download_and_tokenise_openwebtext(
    tokenizer,
    save_path: str,
    max_tokens: int = 500_000_000,  # 500M tokens from OWT
    streaming: bool = True,
):
    #Download OpenWebText and tokenise to disk.
    from datasets import load_dataset
    from tqdm import tqdm

    out_path = Path(save_path)
    if out_path.exists():
        print(f"[OWT] Already exists at {save_path}, skipping download.")
        return

    print("[OWT] Downloading OpenWebText (streaming)... This will take a while — up to 30+ mins on first run.")
    print("[OWT] Data is being downloaded AND tokenised at the same time. Do not close the window.")
    ds = load_dataset("openwebtext", split="train", streaming=streaming, trust_remote_code=True)

    tokens = []
    count = 0
    eos = tokenizer.eos_token_id

    pbar = tqdm(
        total=max_tokens,
        unit="tok",
        unit_scale=True,
        desc="[OWT] Tokenising",
        dynamic_ncols=True,
        colour="cyan",
    )

    for sample in ds:
        text = sample.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos)
        tokens.extend(ids)
        new_tokens = len(ids)
        count += new_tokens
        pbar.update(new_tokens)
        pbar.set_postfix({"samples": f"{pbar.n / 1e6:.1f}M", "buf_MB": f"{len(tokens)*2//1024//1024}MB"})
        if count >= max_tokens:
            break

    pbar.close()
    print(f"[OWT] Tokenisation complete. Saving {len(tokens):,} tokens to disk...")
    arr = np.array(tokens, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), arr)
    print(f"[OWT] Saved {len(arr):,} tokens to {save_path}")


def download_and_tokenise_code(
    tokenizer,
    save_path: str,
    max_tokens: int = 300_000_000,  # 300M code tokens
    languages: List[str] = None,
):
    #Download code dataset and tokenise to disk.]
    from datasets import load_dataset

    if languages is None:
        languages = ["Python", "JavaScript", "TypeScript", "C", "C++", "Java", "Go", "Rust"]

    out_path = Path(save_path)
    if out_path.exists():
        print(f"[Code] Already exists at {save_path}, skipping download.")
        return

    print(f"[Code] Downloading code dataset for: {', '.join(languages)}...")
    from tqdm import tqdm

    tokens = []
    count = 0
    eos = tokenizer.eos_token_id
    code_start = tokenizer.code_start_id
    code_end   = tokenizer.code_end_id

    pbar = tqdm(
        total=max_tokens,
        unit="tok",
        unit_scale=True,
        desc="[Code] Tokenising",
        dynamic_ncols=True,
        colour="green",
    )

    for lang in languages:
        pbar.set_description(f"[Code] {lang}")
        try:
            ds = load_dataset(
                "codeparrot/github-code",
                streaming=True,
                split="train",
                languages=[lang],
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[Code] Failed to load {lang}: {e}, trying next...")
            continue

        for sample in ds:
            code = sample.get("code", "")
            if not code.strip() or len(code) > 10000:
                continue
            ids = [code_start] + tokenizer.encode(code, add_special_tokens=False) + [code_end, eos]
            tokens.extend(ids)
            new_tokens = len(ids)
            count += new_tokens
            pbar.update(new_tokens)
            if count >= max_tokens:
                break

        if count >= max_tokens:
            break

    pbar.close()

    arr = np.array(tokens, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), arr)
    print(f"[Code] Saved {len(arr):,} tokens to {save_path}")


def download_and_tokenise_reasoning(
    tokenizer,
    save_path: str,
    max_samples: int = 200_000,
):
    # Download OpenOrca chain-of-thought dataset. Formats as: <system>...<user>...<assistant><think>reasoning</think>answer<eos>

    from datasets import load_dataset

    out_path = Path(save_path)
    if out_path.exists():
        print(f"[Reasoning] Already exists at {save_path}, skipping download.")
        return

    print("[Reasoning] Downloading OpenOrca reasoning dataset...")
    from tqdm import tqdm

    try:
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True, trust_remote_code=True)
    except Exception:
        print("[Reasoning] OpenOrca failed, trying SlimOrca...")
        ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True, trust_remote_code=True)

    tokens = []
    count = 0
    think_start = tokenizer.think_start_id
    think_end   = tokenizer.think_end_id
    eos         = tokenizer.eos_token_id
    user_tok    = tokenizer.user_token_id
    asst_tok    = tokenizer.assistant_token_id
    sys_tok     = tokenizer.system_token_id

    pbar = tqdm(
        total=max_samples,
        unit="sample",
        unit_scale=True,
        desc="[Reasoning] Tokenising",
        dynamic_ncols=True,
        colour="yellow",
    )

    for sample in ds:
        sys_prompt = sample.get("system_prompt", "")
        question   = sample.get("question", "")
        response   = sample.get("response", "")

        if not question or not response:
            continue

        # Build token sequence with thinking pattern
        ids = [tokenizer.bos_token_id]
        if sys_prompt:
            ids += [sys_tok] + tokenizer.encode(sys_prompt, add_special_tokens=False) + [eos]
        ids += [user_tok] + tokenizer.encode(question, add_special_tokens=False) + [eos]
        ids += [asst_tok, think_start]

        # Simulate thinking: use first ~30% of response as "thinking"
        split = max(1, len(response) // 3)
        thinking_text = response[:split]
        answer_text   = response[split:]

        ids += tokenizer.encode(thinking_text, add_special_tokens=False)
        ids += [think_end]
        ids += tokenizer.encode(answer_text, add_special_tokens=False)
        ids += [eos]

        tokens.extend(ids)
        count += 1
        pbar.update(1)
        pbar.set_postfix({"tokens": f"{len(tokens)/1e6:.1f}M"})
        if count >= max_samples:
            break

    pbar.close()

    arr = np.array(tokens, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), arr)
    print(f"[Reasoning] Saved {len(arr):,} tokens to {save_path}")


def prepare_all_datasets(tokenizer, data_dir: str = "data", context_length: int = 2048):
    #Download and tokenise all datasets.
    #Returns a MultiDataset ready for training.
    #Weights: 50% general, 30% code, 20% reasoning.

    data_dir = Path(data_dir)
    owt_path  = str(data_dir / "openwebtext.npy")
    code_path = str(data_dir / "code.npy")
    rea_path  = str(data_dir / "reasoning.npy")

    download_and_tokenise_openwebtext(tokenizer, owt_path)
    download_and_tokenise_code(tokenizer, code_path)
    download_and_tokenise_reasoning(tokenizer, rea_path)

    datasets = []
    weights  = []

    for path, weight in [(owt_path, 0.5), (code_path, 0.3), (rea_path, 0.2)]:
        if Path(path).exists():
            datasets.append(TokenDataset(path, context_length))
            weights.append(weight)

    if not datasets:
        raise RuntimeError("No datasets found. Run prepare_all_datasets() first.")

    combined = MultiDataset(datasets, weights)
    print(f"[Dataset] Combined dataset: {len(combined):,} total samples")
    return combined


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )
