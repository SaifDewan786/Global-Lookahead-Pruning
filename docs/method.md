```md
# Method: Global Lookahead Pruning

Global Lookahead Pruning estimates the importance of each weight using both weight magnitude and activation changes across future layers.

The intuition is that weights inside layers that cause larger representational changes should be preserved more carefully. Instead of pruning only by magnitude, the method scales each weight by the activation difference between the next two layers.

## Importance Score

```text
score = |W| × sqrt(|mean(A_L+1) - mean(A_L+2)|)
