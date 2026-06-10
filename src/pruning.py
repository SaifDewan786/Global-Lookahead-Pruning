import torch
import torch.nn as nn


def global_magnitude_pruning(model, amount: float):
    """
    Apply global magnitude pruning by zeroing the smallest weights globally.

    Args:
        model: PyTorch model.
        amount: Fraction of weights to prune.

    Returns:
        Pruned model.
    """
    parameters = []

    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters.append(module.weight)

    all_weights = torch.cat([p.detach().abs().flatten() for p in parameters])
    threshold = torch.quantile(all_weights, amount)

    with torch.no_grad():
        for param in parameters:
            mask = param.abs() > threshold
            param.mul_(mask)

    return model


def layerwise_magnitude_pruning(model, amount: float):
    """
    Apply layerwise magnitude pruning.

    Args:
        model: PyTorch model.
        amount: Fraction of weights to prune per layer.

    Returns:
        Pruned model.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight
                threshold = torch.quantile(weight.detach().abs().flatten(), amount)
                mask = weight.abs() > threshold
                weight.mul_(mask)

    return model


def collect_activation_means(model, calibration_input, device: str):
    """
    Collect mean activations from transformer blocks.

    Args:
        model: Transformer language model.
        calibration_input: Input token IDs.
        device: Device name.

    Returns:
        List of mean activation values.
    """
    activation_means = []
    hooks = []

    def hook_fn(module, inputs, output):
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        activation_means.append(output_tensor.detach().float().mean().item())

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    else:
        raise ValueError("Unsupported transformer architecture.")

    for block in blocks:
        hooks.append(block.register_forward_hook(hook_fn))

    model.eval()

    with torch.no_grad():
        calibration_input = calibration_input.to(device)
        model(input_ids=calibration_input)

    for hook in hooks:
        hook.remove()

    return activation_means


def global_lookahead_pruning(
    model,
    calibration_input,
    amount: float,
    device: str,
    epsilon: float = 1e-8,
):
    """
    Apply Global Lookahead Pruning.

    Importance score:
        score = |W| * sqrt(|mean(A_{L+1}) - mean(A_{L+2})|)

    Args:
        model: Transformer language model.
        calibration_input: Calibration input IDs.
        amount: Fraction of weights to prune.
        device: Device name.
        epsilon: Numerical stability value.

    Returns:
        Pruned model.
    """
    activation_means = collect_activation_means(model, calibration_input, device)

    modules = [module for module in model.modules() if isinstance(module, nn.Linear)]

    score_tensors = []

    for idx, module in enumerate(modules):
        if idx + 2 < len(activation_means):
            diff = abs(activation_means[idx + 1] - activation_means[idx + 2])
        else:
            diff = epsilon

        scale = torch.sqrt(torch.tensor(diff + epsilon, device=device))
        score = module.weight.detach().abs() * scale
        score_tensors.append(score.flatten())

    all_scores = torch.cat(score_tensors)
    threshold = torch.quantile(all_scores, amount)

    with torch.no_grad():
        for idx, module in enumerate(modules):
            if idx + 2 < len(activation_means):
                diff = abs(activation_means[idx + 1] - activation_means[idx + 2])
            else:
                diff = epsilon

            scale = torch.sqrt(torch.tensor(diff + epsilon, device=device))
            score = module.weight.detach().abs() * scale
            mask = score > threshold
            module.weight.mul_(mask)

    return model
