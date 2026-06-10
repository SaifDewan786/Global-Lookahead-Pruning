import argparse

from src.model_loader import load_causal_lm
from src.dataset_loader import load_wikitext2, tokenize_dataset
from src.pruning import global_lookahead_pruning
from src.evaluation import evaluate_perplexity, calculate_sparsity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--calibration-size", type=int, default=32)
    args = parser.parse_args()

    model, tokenizer, device = load_causal_lm(args.model)

    dataset = load_wikitext2(split="test")
    encodings = tokenize_dataset(
        dataset,
        tokenizer,
        max_length=args.max_length,
        limit=args.eval_limit,
    )

    baseline_ppl = evaluate_perplexity(model, encodings, batch_size=2, device=device)

    calibration_input = encodings["input_ids"][:args.calibration_size]

    model = global_lookahead_pruning(
        model=model,
        calibration_input=calibration_input,
        amount=args.sparsity,
        device=device,
    )

    pruned_ppl = evaluate_perplexity(model, encodings, batch_size=2, device=device)
    sparsity = calculate_sparsity(model)

    print(f"Model: {args.model}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Pruned PPL: {pruned_ppl:.4f}")
    print(f"Sparsity: {sparsity:.4f}")


if __name__ == "__main__":
    main()
