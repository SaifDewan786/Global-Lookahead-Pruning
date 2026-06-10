import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_causal_lm(model_name: str = "gpt2"):
    """
    Load a causal language model and tokenizer.

    Args:
        model_name: Hugging Face model name.

    Returns:
        model, tokenizer, device
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, tokenizer, device
