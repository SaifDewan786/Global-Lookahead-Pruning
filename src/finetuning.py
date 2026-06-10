import math
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def create_pruning_masks(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Create binary masks from the current model weights.

    Zero-valued weights get mask value 0.
    Non-zero weights get mask value 1.

    This is important because fine-tuning can otherwise update
    previously pruned weights back to non-zero values.
    """
    masks = {}

    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() > 1:
            masks[name] = (param.detach() != 0).float()

    return masks


@torch.no_grad()
def apply_pruning_masks(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
) -> None:
    """
    Re-apply pruning masks to keep pruned weights zero during fine-tuning.
    """
    named_params = dict(model.named_parameters())

    for name, mask in masks.items():
        if name in named_params:
            named_params[name].mul_(mask.to(named_params[name].device))


def build_dataloader(
    encodings: Dict[str, torch.Tensor],
    batch_size: int,
    pad_token_id: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for causal language model fine-tuning.

    Labels are copied from input_ids.
    Padding tokens are replaced with -100 so they are ignored by loss.
    """
    input_ids = encodings["input_ids"]

    attention_mask = (input_ids != pad_token_id).long()

    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100

    dataset = TensorDataset(input_ids, attention_mask, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    encodings: Dict[str, torch.Tensor],
    pad_token_id: int,
    batch_size: int,
    device: str,
) -> float:
    """
    Evaluate average validation loss.
    """
    model.eval()

    dataloader = build_dataloader(
        encodings=encodings,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        shuffle=False,
    )

    total_loss = 0.0
    total_batches = 0

    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss += outputs.loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    encodings: Dict[str, torch.Tensor],
    pad_token_id: int,
    batch_size: int,
    device: str,
) -> float:
    """
    Evaluate perplexity from validation loss.
    """
    loss = evaluate_loss(
        model=model,
        encodings=encodings,
        pad_token_id=pad_token_id,
        batch_size=batch_size,
        device=device,
    )

    return math.exp(loss)


def fine_tune_causal_lm(
    model: torch.nn.Module,
    train_encodings: Dict[str, torch.Tensor],
    tokenizer,
    device: str,
    val_encodings: Optional[Dict[str, torch.Tensor]] = None,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    preserve_sparsity: bool = True,
) -> Dict[str, list]:
    """
    Fine-tune a pruned causal language model.

    Args:
        model: Pruned Hugging Face causal language model.
        train_encodings: Tokenized training data.
        tokenizer: Hugging Face tokenizer.
        device: Training device.
        val_encodings: Optional validation data.
        epochs: Number of fine-tuning epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_steps: Scheduler warmup steps.
        gradient_accumulation_steps: Number of steps before optimizer update.
        max_grad_norm: Gradient clipping value.
        preserve_sparsity: Keep pruned weights zero after optimizer updates.

    Returns:
        Training history dictionary.
    """
    model.to(device)
    model.train()

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    train_loader = build_dataloader(
        encodings=train_encodings,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        shuffle=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_training_steps = (len(train_loader) * epochs) // gradient_accumulation_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_training_steps, 1),
    )

    pruning_masks = create_pruning_masks(model) if preserve_sparsity else {}

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
    }

    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(
            train_loader,
            desc=f"Fine-tuning epoch {epoch + 1}/{epochs}",
        )

        for step, batch in enumerate(progress_bar):
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item() * gradient_accumulation_steps

            should_update = (step + 1) % gradient_accumulation_steps == 0

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if preserve_sparsity:
                    apply_pruning_masks(model, pruning_masks)

                global_step += 1

            avg_train_loss = running_loss / (step + 1)
            progress_bar.set_postfix({"train_loss": avg_train_loss})

        avg_epoch_train_loss = running_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_epoch_train_loss)

        if val_encodings is not None:
            val_loss = evaluate_loss(
                model=model,
                encodings=val_encodings,
                pad_token_id=pad_token_id,
                batch_size=batch_size,
                device=device,
            )

            val_ppl = math.exp(val_loss)

            history["val_loss"].append(val_loss)
            history["val_perplexity"].append(val_ppl)

            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={avg_epoch_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_ppl={val_ppl:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={avg_epoch_train_loss:.4f}"
            )

    if preserve_sparsity:
        apply_pruning_masks(model, pruning_masks)

    return history
