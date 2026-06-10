```md
# Experiment Setup

## Dataset

WikiText-2 was used for calibration and perplexity evaluation.

## Models

- GPT-2
- Qwen-2.5B
- DeepSeekR1

## Baselines

- Global magnitude pruning
- Layerwise magnitude pruning

## Metrics

- Perplexity
- Sparsity

## Fine-Tuning

Fine-tuning was used after pruning to recover model performance, especially for GPT-2.
