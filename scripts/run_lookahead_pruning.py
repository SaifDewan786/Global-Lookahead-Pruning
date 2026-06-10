import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_loader import load_causal_lm
from src.dataset_loader import load_wikitext2, tokenize_dataset
from src.pruning import global_lookahead_pruning
from src.finetuning import evaluate_perplexity
from src.utils import (
    calculate_sparsity,
    format_experiment_name,
    save_json,
    save_model_and_tokenizer,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Global Lookahead Pruning on a causal language model."
    )

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--sparsity", type=float, default=0.30)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--calibration-split", type=str, default="train")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--calibration-limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")

    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer, device = load_causal_lm(args.model)

    eval_dataset = load_wikitext2(split=args.eval_split)
    eval_encodings = tokenize_dataset(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.eval_limit,
    )

    calibration_dataset = load_wikitext2(split=args.calibration_split)
    calibration_encodings = tokenize_dataset(
        dataset=calibration_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.calibration_limit,
    )

    calibration_input = calibration_encodings["input_ids"]

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    baseline_ppl = evaluate_perplexity(
        model=model,
        encodings=eval_encodings,
        pad_token_id=pad_token_id,
        batch_size=args.batch_size,
        device=device,
    )

    model = global_lookahead_pruning(
        model=model,
        calibration_input=calibration_input,
        amount=args.sparsity,
        device=device,
    )

    pruned_ppl = evaluate_perplexity(
        model=model,
        encodings=eval_encodings,
        pad_token_id=pad_token_id,
        batch_size=args.batch_size,
        device=device,
    )

    actual_sparsity = calculate_sparsity(model)

    experiment_name = format_experiment_name(
        model_name=args.model,
        method="lookahead",
        sparsity=args.sparsity,
    )

    results = {
        "model": args.model,
        "method": "global_lookahead_pruning",
        "target_sparsity": args.sparsity,
        "actual_sparsity": actual_sparsity,
        "baseline_perplexity": baseline_ppl,
        "pruned_perplexity": pruned_ppl,
        "eval_split": args.eval_split,
        "calibration_split": args.calibration_split,
        "eval_limit": args.eval_limit,
        "calibration_limit": args.calibration_limit,
        "max_length": args.max_length,
    }

    print(results)

    result_path = Path(args.output_dir) / "results" / f"{experiment_name}.json"
    save_json(results, result_path)

    if args.save_model:
        model_path = Path(args.output_dir) / "models" / experiment_name
        save_model_and_tokenizer(model, tokenizer, model_path)
        print(f"Saved pruned model to: {model_path}")


if __name__ == "__main__":
    main()
