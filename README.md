# ThinkingLM — Custom GPT-style AI with Reasoning

A custom transformer language model built from scratch in Python, with chain-of-thought reasoning, trained on general + code + reasoning datasets.

## Images
<img width="705" height="278" alt="image" src="https://github.com/user-attachments/assets/5b1c633f-2ef0-41b9-8e8d-951fb68f625a" />
<img width="694" height="341" alt="image" src="https://github.com/user-attachments/assets/ce4a63ee-c35d-4047-bade-23ea17aad382" />
<img width="2558" height="1279" alt="image" src="https://github.com/user-attachments/assets/cf7ee1df-fdfe-4421-9b10-bdd255bdbd05" />


## System Requirements I've optimised for
- **GPU**: RTX 4070 12GB
- **RAM**: 56GB
- **CPU**: AMD 5700X3D
- **OS**: Windows 11
- **Python**: 3.10+

---

## Quick Start
### 1. Train the model
```bat
python train.py
```
Training downloads datasets automatically (~30GB total) and runs for 100,000 steps.
Checkpoints save every 2,000 steps to `checkpoints/`.

### 2. Chat (CLI)
```bat
python chat_cli.py
```

### 3. Chat (Web UI)
```bat
python chat_web.py
```
Opens browser at `http://localhost:7860`

---

## Architecture

| Component | Choice |
|-----------|--------|
| Architecture | GPT-style transformer |
| Parameters | ~300M |
| Attention | Grouped Query Attention (4 KV heads, 16 Q heads) |
| Position encoding | RoPE (Rotary) |
| Activation | SwiGLU |
| Normalisation | RMSNorm |
| Context window | 2048 tokens |
| Flash Attention | Yes (via PyTorch SDPA) |

## Datasets

| Dataset | Tokens | Purpose |
|---------|--------|---------|
| OpenWebText | ~500M | General language / knowledge |
| CodeParrot/GitHub | ~300M | Code reasoning (Python, JS, C++, Go…) |
| OpenOrca | ~200M | Chain-of-thought reasoning |

## Reasoning System

The model uses special `<think>` / `</think>` tokens:

```
<user> What is 17 × 24?
<assistant><think>
  17 × 24 = 17 × 20 + 17 × 4
           = 340 + 68 = 408
</think>
The answer is 408.
```

In the CLI, thinking blocks appear in yellow. In the web UI, they appear as collapsible panels.

## File Structure

```
AI-Model/
├── train.py            ← Start training here
├── chat_cli.py         ← Terminal chat interface
├── chat_web.py         ← Gradio web UI
├── requirements.txt
├── src/
│   ├── model.py        ← Transformer architecture
│   ├── tokenizer.py    ← BPE tokenizer
│   ├── dataset.py      ← Dataset download + preprocessing
│   ├── trainer.py      ← Training loop + optimisations
│   ├── chat.py         ← CLI chat logic
│   └── web_ui.py       ← Gradio web UI logic
├── checkpoints/        ← Saved model checkpoints
├── data/               ← Tokenised datasets (auto-created)
└── logs/               ← Training logs (JSONL)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/think on\|off` | Toggle chain-of-thought reasoning |
| `/temp 0.7` | Set temperature |
| `/topp 0.9` | Set top-p sampling |
| `/topk 50` | Set top-k |
| `/maxlen 512` | Set max response tokens |
| `/reset` | Clear conversation history |
| `/save file.txt` | Save conversation |
| `/quit` | Exit |

## Training Progress

Training logs are saved to `logs/train_log.jsonl`. Each line:
```json
{"step": 1000, "loss": 3.42, "lr": 0.0003, "tokens_per_sec": 12500, "vram_used_gb": 9.2}
```

## Resuming Training

```bat
python train.py --resume checkpoints/ckpt_latest.pt
```
