import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


def load_wikitext2(split: str = "test"):
    """Load WikiText-2 dataset."""
    return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)


def tokenize_dataset(dataset, tokenizer, max_length: int = 512, limit: int | None = None):
    """
    Tokenize WikiText-2 text samples.

    Args:
        dataset: Hugging Face dataset.
        tokenizer: Hugging Face tokenizer.
        max_length: Maximum sequence length.
        limit: Optional sample limit for faster experiments.

    Returns:
        Dictionary containing padded input_ids.
    """
    texts = [item["text"] for item in dataset if item["text"].strip()]

    if limit is not None:
        texts = texts[:limit]

    tokenized = []

    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if encoded["input_ids"].numel() > 0:
            tokenized.append(encoded["input_ids"].squeeze(0))

    padded_input_ids = pad_sequence(
        tokenized,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    return {"input_ids": padded_input_ids}
