# Global Lookahead Pruning for Transformer-Based Language Models

## Overview

Global Lookahead Pruning is an activation-aware post-training pruning method for transformer-based language models. The method estimates weight importance by combining weight magnitude with lookahead activation differences between subsequent transformer layers.

The goal is to reduce model size while preserving language modeling performance, measured using perplexity and sparsity.

## Key Features

- Post-training pruning for transformer language models
- Activation-aware importance scoring
- Global pruning across model layers
- Baseline comparison with magnitude-based pruning
- Perplexity and sparsity evaluation on WikiText-2
- Fine-tuning support for pruned models

## Method

For a weight matrix in layer `L`, Global Lookahead Pruning calculates an importance score using:

```text
importance = |weight| × sqrt(|mean(A_L+1) - mean(A_L+2)|)
