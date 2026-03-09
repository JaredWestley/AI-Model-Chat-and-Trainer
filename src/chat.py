import sys
import os
import torch
from pathlib import Path
from typing import List, Optional

class Colour:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    RED     = "\033[31m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[40m"


def coloured(text: str, colour: str) -> str:
    return f"{colour}{text}{Colour.RESET}"


def print_banner():
    banner = f"""
{Colour.CYAN}{Colour.BOLD}
  ████████╗██╗  ██╗██╗███╗   ██╗██╗  ██╗██╗███╗   ██╗ ██████╗      ██╗     ███╗   ███╗
     ██╔══╝██║  ██║██║████╗  ██║██║ ██╔╝██║████╗  ██║██╔════╝      ██║     ████╗ ████║
     ██║   ███████║██║██╔██╗ ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ██║     ██╔████╔██║
     ██║   ██╔══██║██║██║╚██╗██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ██║     ██║╚██╔╝██║
     ██║   ██║  ██║██║██║ ╚████║██║  ██╗██║██║ ╚████║╚██████╔╝     ███████╗██║ ╚═╝ ██║
     ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ╚══════╝╚═╝     ╚═╝
{Colour.RESET}
{Colour.GREY}  Custom GPT-style model with chain-of-thought reasoning{Colour.RESET}
{Colour.GREY}  Type /help for commands{Colour.RESET}
"""
    print(banner)


HELP_TEXT = f"""
{Colour.CYAN}Available commands:{Colour.RESET}
  {Colour.GREEN}/help{Colour.RESET}              Show this help
  {Colour.GREEN}/reset{Colour.RESET}             Clear conversation history
  {Colour.GREEN}/think on|off{Colour.RESET}      Toggle chain-of-thought reasoning (default: off)
  {Colour.GREEN}/temp <0.0-2.0>{Colour.RESET}    Set temperature (default: 0.8)
  {Colour.GREEN}/topp <0.0-1.0>{Colour.RESET}    Set top-p nucleus sampling (default: 0.9)
  {Colour.GREEN}/topk <int>{Colour.RESET}         Set top-k (default: 40)
  {Colour.GREEN}/rep <float>{Colour.RESET}        Set repetition penalty (default: 1.3)
  {Colour.GREEN}/maxlen <int>{Colour.RESET}       Set max response tokens (default: 200)
  {Colour.GREEN}/save <path>{Colour.RESET}        Save conversation to file
  {Colour.GREEN}/quit{Colour.RESET}              Exit
"""


class CLIChat:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        system_prompt: str = "You are a helpful, intelligent AI assistant with strong reasoning capabilities.",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt
        self.history: List[dict] = []

        # Generation settings
        self.think             = False   # off by default until model is more trained
        self.temperature       = 0.8
        self.top_p             = 0.9
        self.top_k             = 40
        self.max_tokens        = 200
        self.repetition_penalty = 1.3   # stronger penalty to stop loops

        print(f"{Colour.GREY}[Chat] Loading tokenizer from {tokenizer_path}...{Colour.RESET}")
        self._load_tokenizer(tokenizer_path)

        print(f"{Colour.GREY}[Chat] Loading model from {checkpoint_path}...{Colour.RESET}")
        self._load_model(checkpoint_path)

        print(f"{Colour.GREEN}[Chat] Ready! Running on {self.device}{Colour.RESET}\n")

    def _load_tokenizer(self, path: str):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.tokenizer import ThinkingTokenizer
        self.tokenizer = ThinkingTokenizer.load(path)

    def _load_model(self, path: str):
        from src.model import ThinkingLM, ModelConfig
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("config", {})

        # Build ModelConfig from checkpoint
        model_cfg = ModelConfig(
            vocab_size=self.tokenizer.vocab_size,
        )
        # Override with any saved model config keys
        for k, v in cfg_dict.items():
            if hasattr(model_cfg, k):
                setattr(model_cfg, k, v)

        self.model = ThinkingLM(model_cfg).to(self.device)
        state = ckpt.get("model", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.to(torch.bfloat16)

    def _encode_prompt(self) -> torch.Tensor:
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        ids = self.tokenizer.encode_chat(messages)

        # Trim to context window (leave room for response)
        max_input = self.model.config.context_length - self.max_tokens
        if len(ids) > max_input:
            ids = ids[-max_input:]

        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _generate_streaming(self, input_ids: torch.Tensor) -> str:
        """Stream tokens, printing thinking in yellow and response in white."""
        in_think = False
        response_parts = []
        think_parts = []
        response_started = False

        think_start = self.tokenizer.think_start_id
        think_end   = self.tokenizer.think_end_id
        eos         = self.tokenizer.eos_token_id

        print(f"\n{Colour.CYAN}Assistant:{Colour.RESET} ", end="", flush=True)

        buffer = []

        recent_tokens = []

        for tok_id in self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            think=self.think,
            stream=True,
        ):
            if tok_id == eos:
                break

            if tok_id == think_start:
                in_think = True
                print(f"\n{Colour.YELLOW}[Thinking...]{Colour.RESET}", flush=True)
                continue

            if tok_id == think_end:
                in_think = False
                print(f"\n{Colour.GREY}[/Thinking]{Colour.RESET}\n", flush=True)
                continue

            # Hard loop detection: if last 20 tokens are repeating a short pattern, stop
            recent_tokens.append(tok_id)
            if len(recent_tokens) > 20:
                recent_tokens.pop(0)
                # Check if the last 10 tokens are all the same token
                if len(set(recent_tokens[-10:])) == 1:
                    print(f"\n{Colour.GREY}[stopped: repetition loop]{Colour.RESET}")
                    break

            # Decode token — skip special tokens so they don't appear as raw text
            text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
            if not text:
                continue

            if in_think:
                print(f"{Colour.YELLOW}{text}{Colour.RESET}", end="", flush=True)
                think_parts.append(text)
            else:
                if not response_started:
                    response_started = True
                print(text, end="", flush=True)
                response_parts.append(text)

        print()
        return "".join(response_parts)

    def _handle_command(self, cmd: str) -> bool:
        """Handle /commands. Returns True if command was handled."""
        parts = cmd.strip().split()
        if not parts:
            return False
        command = parts[0].lower()

        if command == "/quit" or command == "/exit":
            print(f"{Colour.CYAN}Goodbye!{Colour.RESET}")
            sys.exit(0)

        elif command == "/help":
            print(HELP_TEXT)

        elif command == "/reset":
            self.history = []
            print(f"{Colour.GREEN}Conversation reset.{Colour.RESET}")

        elif command == "/think":
            if len(parts) > 1:
                self.think = parts[1].lower() in ("on", "true", "1", "yes")
            print(f"{Colour.GREEN}Thinking: {'ON' if self.think else 'OFF'}{Colour.RESET}")

        elif command == "/temp":
            if len(parts) > 1:
                try:
                    self.temperature = float(parts[1])
                    print(f"{Colour.GREEN}Temperature: {self.temperature}{Colour.RESET}")
                except ValueError:
                    print(f"{Colour.RED}Invalid temperature.{Colour.RESET}")

        elif command == "/topp":
            if len(parts) > 1:
                try:
                    self.top_p = float(parts[1])
                    print(f"{Colour.GREEN}Top-p: {self.top_p}{Colour.RESET}")
                except ValueError:
                    print(f"{Colour.RED}Invalid top-p.{Colour.RESET}")

        elif command == "/topk":
            if len(parts) > 1:
                try:
                    self.top_k = int(parts[1])
                    print(f"{Colour.GREEN}Top-k: {self.top_k}{Colour.RESET}")
                except ValueError:
                    print(f"{Colour.RED}Invalid top-k.{Colour.RESET}")

        elif command == "/rep":
            if len(parts) > 1:
                try:
                    self.repetition_penalty = float(parts[1])
                    print(f"{Colour.GREEN}Repetition penalty: {self.repetition_penalty}{Colour.RESET}")
                except ValueError:
                    print(f"{Colour.RED}Invalid repetition penalty.{Colour.RESET}")

        elif command == "/maxlen":
            if len(parts) > 1:
                try:
                    self.max_tokens = int(parts[1])
                    print(f"{Colour.GREEN}Max tokens: {self.max_tokens}{Colour.RESET}")
                except ValueError:
                    print(f"{Colour.RED}Invalid max tokens.{Colour.RESET}")

        elif command == "/save":
            path = parts[1] if len(parts) > 1 else "conversation.txt"
            self._save_conversation(path)

        else:
            print(f"{Colour.RED}Unknown command: {command}. Type /help for help.{Colour.RESET}")

        return True

    def _save_conversation(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"System: {self.system_prompt}\n\n")
            for msg in self.history:
                role = msg["role"].capitalize()
                f.write(f"{role}: {msg['content']}\n\n")
        print(f"{Colour.GREEN}Conversation saved to {path}{Colour.RESET}")

    def run(self):
        print_banner()

        while True:
            try:
                user_input = input(f"{Colour.GREEN}You:{Colour.RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Colour.CYAN}Goodbye!{Colour.RESET}")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            self.history.append({"role": "user", "content": user_input})

            input_ids = self._encode_prompt()

            try:
                response = self._generate_streaming(input_ids)
                self.history.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"{Colour.RED}[Error] Generation failed: {e}{Colour.RESET}")
                self.history.pop()  # Remove failed user message


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with ThinkingLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_latest.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                        help="Path to tokenizer file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--system", type=str,
                        default="You are a helpful, intelligent AI assistant with strong reasoning capabilities.",
                        help="System prompt")
    args = parser.parse_args()

    chat = CLIChat(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        system_prompt=args.system,
    )
    chat.run()


if __name__ == "__main__":
    main()
