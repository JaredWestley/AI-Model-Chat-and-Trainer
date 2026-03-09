import sys
import re
import torch
import gradio as gr
from pathlib import Path
from typing import Iterator, List


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str = "cuda"):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.tokenizer import ThinkingTokenizer
    from src.model import ThinkingLM, ModelConfig

    tokenizer = ThinkingTokenizer.load(tokenizer_path)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    for k, v in cfg_dict.items():
        if hasattr(model_cfg, k):
            setattr(model_cfg, k, v)

    model = ThinkingLM(model_cfg).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model = model.to(torch.bfloat16)

    print(f"[WebUI] Model loaded on {device}")
    return model, tokenizer


def stream_response(
    model, tokenizer, history: List[dict],
    system_prompt: str, temperature: float,
    top_p: float, top_k: int, max_tokens: int,
    use_thinking: bool, device: str,
) -> Iterator[str]:
    # Yield growing text chunks. Thinking is wrapped in <think>...</think> tags
    # which Gradio 6.8 renders natively via reasoning_tags.

    messages = [{"role": "system", "content": system_prompt}] + history
    ids = tokenizer.encode_chat(messages)
    max_input = model.config.context_length - max_tokens
    if len(ids) > max_input:
        ids = ids[-max_input:]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    in_think = False
    current = ""

    for tok_id in model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        think=use_thinking,
        stream=True,
    ):
        if tok_id == tokenizer.eos_token_id:
            break
        if tok_id == tokenizer.think_start_id:
            in_think = True
            current += "<think>"
            yield current
            continue
        if tok_id == tokenizer.think_end_id:
            in_think = False
            current += "</think>"
            yield current
            continue

        text = tokenizer.decode([tok_id], skip_special_tokens=False)
        current += text
        yield current


def extract_plain(text: str) -> str:
    #Strip <think> blocks for history storage.
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return clean.strip()


def create_ui(checkpoint_path: str, tokenizer_path: str, device: str = "cuda"):
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, tokenizer_path, device)

    custom_css = """
    .thinking-block { color: #aaa; font-style: italic; border-left: 3px solid #f0c040; padding-left: 8px; }
    #title-row { text-align: center; }
    """

    def chat_fn(
        user_message, chatbot, history_state,
        system_prompt, temperature, top_p, top_k, max_tokens, use_thinking,
    ):
        if not user_message.strip():
            yield user_message, chatbot, history_state
            return

        # Gradio 6 uses {"role": ..., "content": ...} dicts for chatbot
        chatbot = chatbot + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": ""},
        ]
        history_state = history_state + [{"role": "user", "content": user_message}]

        partial = ""
        for chunk in stream_response(
            model, tokenizer,
            history=history_state,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            max_tokens=int(max_tokens),
            use_thinking=use_thinking,
            device=str(device),
        ):
            partial = chunk
            chatbot[-1]["content"] = partial
            yield "", chatbot, history_state

        history_state = history_state + [{"role": "assistant", "content": extract_plain(partial)}]
        yield "", chatbot, history_state

    def clear_fn():
        return [], []

    with gr.Blocks(title="ThinkingLM Chat") as demo:
        history_state = gr.State([])

        with gr.Row(elem_id="title-row"):
            gr.Markdown("# ThinkingLM\n**Custom GPT-style model with chain-of-thought reasoning**")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=520,
                    reasoning_tags=[("think", "thought")],
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message... (Enter to send)",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful, intelligent AI assistant with strong reasoning capabilities.",
                    lines=3,
                )
                use_thinking = gr.Checkbox(label="Enable Chain-of-Thought Thinking", value=True)
                temperature  = gr.Slider(label="Temperature",  minimum=0.0, maximum=2.0, value=0.7,  step=0.05)
                top_p        = gr.Slider(label="Top-P",         minimum=0.1, maximum=1.0, value=0.9,  step=0.05)
                top_k        = gr.Slider(label="Top-K",         minimum=1,   maximum=200, value=50,   step=1)
                max_tokens   = gr.Slider(label="Max Tokens",    minimum=64,  maximum=2048, value=512, step=64)

                gr.Markdown(
                    f"---\n**Model:** {model.get_num_params()/1e6:.1f}M params  \n"
                    f"**VRAM:** ~{model.estimate_vram_mb():.0f}MB  \n"
                    f"**Device:** {device}  \n"
                    f"**Context:** {model.config.context_length} tokens"
                )

        inputs  = [msg_input, chatbot, history_state, system_prompt, temperature, top_p, top_k, max_tokens, use_thinking]
        outputs = [msg_input, chatbot, history_state]

        msg_input.submit(chat_fn, inputs=inputs, outputs=outputs)
        send_btn.click(chat_fn,  inputs=inputs, outputs=outputs)
        clear_btn.click(clear_fn, outputs=[chatbot, history_state])

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ThinkingLM Web UI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_latest.pt")
    parser.add_argument("--tokenizer",  type=str, default="data/tokenizer.json")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true")
    args = parser.parse_args()

    demo = create_ui(args.checkpoint, args.tokenizer, args.device)
    print(f"\n[WebUI] Starting on http://localhost:{args.port}")
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        css="""
        #title-row { text-align: center; }
        """,
    )


if __name__ == "__main__":
    main()
