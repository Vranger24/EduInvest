#Adarsh Khanna Coded this file.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys
from typing import Tuple, List, Dict

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SYSTEM_PROMPT = "You are a college counsellor helping kids get into their dream university"


class Config:
    """Configuration for the chat session."""
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Chat with Qwen3-Omni models locally"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model ID from HuggingFace (default: %(default)s)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens to generate (default: %(default)s)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (default: %(default)s)"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt (default: %(default)s)"
    )
    
    args = parser.parse_args()
    return Config(
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )


def load_model(config: Config) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    print(f"\n{'='*60}")
    print(f"Loading model: {config.model_id}")
    print(f"{'='*60}\n")

    try:
        # Determine optimal dtype based on available hardware
        if torch.cuda.is_available():
            dtype = torch.float16
            device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            dtype = torch.float32  # MPS (Apple Silicon) works better with float32
            device_info = "MPS (Apple Silicon)"
        else:
            dtype = torch.float32
            device_info = "CPU"
        
        print(f"Using device: {device_info}")
        print(f"Using dtype: {dtype}\n")

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True
        )

        # Load model
        print("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )

        print("\n✓ Model loaded successfully!\n")
        return tokenizer, model
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def extract_assistant_response(full_response: str) -> str:
    if "assistant" in full_response.lower():
        response = full_response.split("assistant")[-1].strip()
    else:
        response = full_response.strip()
    
    for prefix in [":", "\n", " "]:
        response = response.lstrip(prefix)
    
    return response


def chat_loop(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, config: Config):
    print(f"{'='*60}")
    print("Qwen3-Omni Chat Session")
    print(f"{'='*60}")
    print("\nCommands:")
    print("  'exit' or 'quit' - End the session")
    print("  'clear' - Clear conversation history")
    print("  'config' - Show current configuration")
    print(f"\n{'='*60}\n")

    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": config.system_prompt}
    ]

    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye! 👋\n")
                break
            
            if user_input.lower() == "clear":
                conversation = [{"role": "system", "content": config.system_prompt}]
                print("\n✓ Conversation history cleared.\n")
                continue
            
            if user_input.lower() == "config":
                print(f"\nCurrent Configuration:")
                print(f"  Model: {config.model_id}")
                print(f"  Max tokens: {config.max_tokens}")
                print(f"  Temperature: {config.temperature}")
                print(f"  Messages in history: {len(conversation)}\n")
                continue

            conversation.append({"role": "user", "content": user_input})

            inputs = tokenizer.apply_chat_template(
                conversation,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_response = tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

            assistant_response = extract_assistant_response(full_response)

            print(f"\nAssistant: {assistant_response}\n")

            conversation.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.\n")
            continue
        except Exception as e:
            print(f"\n✗ Error during generation: {e}\n", file=sys.stderr)
            # Remove the last user message if generation failed
            if conversation[-1]["role"] == "user":
                conversation.pop()


def main():
    try:
        config = parse_args()
        tokenizer, model = load_model(config)
        chat_loop(tokenizer, model, config)
    except KeyboardInterrupt:
        print("\n\nExiting...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
