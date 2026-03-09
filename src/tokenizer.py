import os
import json
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFC


SPECIAL_TOKENS = [
    "<pad>",    # 0
    "<bos>",    # 1
    "<eos>",    # 2
    "<think>",  # 3  — begin chain-of-thought block
    "</think>", # 4  — end chain-of-thought block
    "<code>",   # 5  — begin code block
    "</code>",  # 6  — end code block
    "<user>",   # 7  — chat role: user
    "<assistant>", # 8 — chat role: assistant
    "<system>", # 9  — chat role: system
    "<unk>",    # 10
]

VOCAB_SIZE = 32000


class ThinkingTokenizer:
    # Wraps a HuggingFace BPE tokenizer with special tokens for reasoning.

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self._tok = tokenizer
        self._setup_token_ids()

    def _setup_token_ids(self):
        if self._tok is None:
            return
        self.pad_token_id     = self._tok.token_to_id("<pad>")
        self.bos_token_id     = self._tok.token_to_id("<bos>")
        self.eos_token_id     = self._tok.token_to_id("<eos>")
        self.think_start_id   = self._tok.token_to_id("<think>")
        self.think_end_id     = self._tok.token_to_id("</think>")
        self.code_start_id    = self._tok.token_to_id("<code>")
        self.code_end_id      = self._tok.token_to_id("</code>")
        self.user_token_id    = self._tok.token_to_id("<user>")
        self.assistant_token_id = self._tok.token_to_id("<assistant>")
        self.system_token_id  = self._tok.token_to_id("<system>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        enc = self._tok.encode(text)
        ids = enc.ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_chat(self, messages: List[dict]) -> List[int]:
        #Encode a chat conversation.
        #messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        #Format: <system>...<eos><user>...<eos><assistant><think>...</think>response<eos>

        ids = [self.bos_token_id]
        role_map = {
            "system":    self.system_token_id,
            "user":      self.user_token_id,
            "assistant": self.assistant_token_id,
        }
        for msg in messages:
            role_tok = role_map.get(msg["role"], self.user_token_id)
            ids.append(role_tok)
            content_ids = self._tok.encode(msg["content"]).ids
            ids.extend(content_ids)
            ids.append(self.eos_token_id)
        return ids

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(path)
        print(f"[Tokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ThinkingTokenizer":
        tok = Tokenizer.from_file(path)
        instance = cls(tok)
        print(f"[Tokenizer] Loaded from {path} (vocab={instance.vocab_size})")
        return instance

    @classmethod
    def train(cls, files: List[str], save_path: str, vocab_size: int = VOCAB_SIZE) -> "ThinkingTokenizer":
        print(f"[Tokenizer] Training BPE tokenizer on {len(files)} files, vocab={vocab_size}...")

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )
        tokenizer.train(files, trainer)

        # Post-processor: add BOS/EOS automatically
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<bos>:0 $A:0 <eos>:0",
            pair="<bos>:0 $A:0 <eos>:0 $B:0 <eos>:0",
            special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
        )

        instance = cls(tokenizer)
        instance.save(save_path)
        print(f"[Tokenizer] Training complete. Vocab size: {instance.vocab_size}")
        return instance

    @classmethod
    def from_pretrained_gpt2(cls, save_path: str) -> "ThinkingTokenizer":
        print("[Tokenizer] Bootstrapping from GPT-2 tokenizer...")
        from tokenizers import Tokenizer as HFTokenizer
        try:
            from transformers import GPT2TokenizerFast
            gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
            # Save the underlying tokenizer
            tmp_path = save_path + ".tmp.json"
            gpt2.backend_tokenizer.save(tmp_path)
            tok = HFTokenizer.from_file(tmp_path)
            os.remove(tmp_path)

            # Add special tokens
            from tokenizers import AddedToken
            added = [AddedToken(t, special=True) for t in SPECIAL_TOKENS if tok.token_to_id(t) is None]
            tok.add_special_tokens(added)

            instance = cls(tok)
            instance.save(save_path)
            print(f"[Tokenizer] Bootstrapped GPT-2 tokenizer. Vocab size: {instance.vocab_size}")
            return instance
        except ImportError:
            raise ImportError("transformers package required for GPT-2 bootstrap. Run: pip install transformers")
