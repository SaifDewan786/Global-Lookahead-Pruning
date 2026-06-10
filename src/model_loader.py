import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_device


def load_causal_lm(model_name: str = "gpt2"):
    """
    Load a causal language model and tokenizer.
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, tokenizer, device
