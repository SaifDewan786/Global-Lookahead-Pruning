import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_loader import load_causal_lm
from src.dataset_loader import load_wikitext2, tokenize_dataset
from src.finetuning import evaluate_perplexity
from src.utils import calculate_sparsity, save_json, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline perplexity and sparsity of a dense causal language model."
    )

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/baseline_eval.json",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer, device = load_causal_lm(args.model)

    dataset = load_wikitext2(split=args.split)
    encodings = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.eval_limit,
    )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    perplexity = evaluate_perplexity(
        model=model,
        encodings=encodings,
        pad_token_id=pad_token_id,
        batch_size=args.batch_size,
        device=device,
    )

    sparsity = calculate_sparsity(model)

    results = {
        "model": args.model,
        "method": "baseline_dense",
        "split": args.split,
        "perplexity": perplexity,
        "sparsity": sparsity,
        "eval_limit": args.eval_limit,
        "max_length": args.max_length,
    }

    print(results)
    save_json(results, args.output)


if __name__ == "__main__":
    main()
