import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """
    Return the best available PyTorch device.
    """
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save a dictionary as a JSON file.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_results_csv(
    rows: Iterable[Dict[str, Any]],
    path: str | Path,
) -> None:
    """
    Save experiment results as a CSV file.
    """
    path = Path(path)
    ensure_dir(path.parent)

    dataframe = pd.DataFrame(list(rows))
    dataframe.to_csv(path, index=False)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total model parameters.
    """
    return sum(param.numel() for param in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def calculate_sparsity(model: torch.nn.Module) -> float:
    """
    Calculate model sparsity.

    Sparsity = zero parameters / total parameters
    """
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()

    return zero_params / max(total_params, 1)


def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print a simple model summary.
    """
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    sparsity = calculate_sparsity(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Sparsity: {sparsity:.4f}")


def save_model_and_tokenizer(
    model: torch.nn.Module,
    tokenizer,
    output_dir: str | Path,
) -> None:
    """
    Save a Hugging Face model and tokenizer.
    """
    output_dir = ensure_dir(output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def format_experiment_name(
    model_name: str,
    method: str,
    sparsity: float,
) -> str:
    """
    Create a clean experiment name for outputs.
    """
    clean_model_name = model_name.replace("/", "-")
    clean_method = method.lower().replace(" ", "-")
    sparsity_pct = int(sparsity * 100)

    return f"{clean_model_name}_{clean_method}_{sparsity_pct}pct"
