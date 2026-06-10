import argparse
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset_loader import load_wikitext2, tokenize_dataset
from src.finetuning import evaluate_perplexity, fine_tune_causal_lm
from src.utils import (
    calculate_sparsity,
    get_device,
    save_json,
    save_model_and_tokenizer,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a saved pruned causal language model."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved pruned model directory.",
    )

    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--val-limit", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/finetuned")

    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(device)

    train_dataset = load_wikitext2(split=args.train_split)
    val_dataset = load_wikitext2(split=args.val_split)

    train_encodings = tokenize_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.train_limit,
    )

    val_encodings = tokenize_dataset(
        dataset=val_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.val_limit,
    )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    before_ppl = evaluate_perplexity(
        model=model,
        encodings=val_encodings,
        pad_token_id=pad_token_id,
        batch_size=args.batch_size,
        device=device,
    )

    before_sparsity = calculate_sparsity(model)

    print(f"Before fine-tuning PPL: {before_ppl:.4f}")
    print(f"Before fine-tuning sparsity: {before_sparsity:.4f}")

    history = fine_tune_causal_lm(
        model=model,
        train_encodings=train_encodings,
        val_encodings=val_encodings,
        tokenizer=tokenizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        preserve_sparsity=True,
    )

    after_ppl = evaluate_perplexity(
        model=model,
        encodings=val_encodings,
        pad_token_id=pad_token_id,
        batch_size=args.batch_size,
        device=device,
    )

    after_sparsity = calculate_sparsity(model)

    print(f"After fine-tuning PPL: {after_ppl:.4f}")
    print(f"After fine-tuning sparsity: {after_sparsity:.4f}")

    model_name = Path(args.model_path).name
    output_path = Path(args.output_dir) / f"{model_name}_finetuned"

    save_model_and_tokenizer(model, tokenizer, output_path)

    results = {
        "model_path": args.model_path,
        "output_path": str(output_path),
        "before_perplexity": before_ppl,
        "after_perplexity": after_ppl,
        "before_sparsity": before_sparsity,
        "after_sparsity": after_sparsity,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "history": history,
    }

    result_path = output_path / "finetuning_results.json"
    save_json(results, result_path)

    print(f"Saved fine-tuned model to: {output_path}")
    print(f"Saved fine-tuning results to: {result_path}")


if __name__ == "__main__":
    main()
