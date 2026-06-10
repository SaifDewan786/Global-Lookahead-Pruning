```md
# Method: Global Lookahead Pruning

Global Lookahead Pruning estimates the importance of each weight using both weight magnitude and activation changes across future layers.

The intuition is that weights inside layers that cause larger representational changes should be preserved more carefully. Instead of pruning only by magnitude, the method scales each weight by the activation difference between the next two layers.

## Importance Score

```text
score = |W| × sqrt(|mean(A_L+1) - mean(A_L+2)|)
```

## Steps

- Run a calibration forward pass.
- Collect mean activations from transformer layers.
- Compute lookahead importance scores.
- Rank weights globally.
- Prune the lowest-scoring weights.
- Evaluate perplexity and sparsity.
- Optionally fine-tune the pruned model.
