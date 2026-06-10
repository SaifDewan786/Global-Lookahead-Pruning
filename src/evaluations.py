import math
import torch


@torch.no_grad()
def evaluate_perplexity(model, encodings, batch_size: int = 4, device: str = "cpu") -> float:
    """
    Evaluate perplexity for a causal language model.

    Args:
        model: Language model.
        encodings: Tokenized dataset dictionary.
        batch_size: Batch size.
        device: Device name.

    Returns:
        Perplexity score.
    """
    model.eval()

    input_ids = encodings["input_ids"].to(device)
    total_loss = 0.0
    total_batches = 0

    for start_idx in range(0, input_ids.size(0), batch_size):
        batch = input_ids[start_idx:start_idx + batch_size]

        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return math.exp(avg_loss)


def calculate_sparsity(model) -> float:
    """
    Calculate the fraction of zero-valued parameters.

    Args:
        model: PyTorch model.

    Returns:
        Sparsity ratio.
    """
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()

    return zero_params / total_params
