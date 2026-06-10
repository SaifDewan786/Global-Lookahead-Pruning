# Limitations

1. The current method uses mean activation difference only.
2. It may not work equally well across all transformer architectures.
3. Unstructured sparsity may not directly improve inference speed on standard GPUs.
4. Calibration data quality strongly affects pruning decisions.
5. More comparison with methods such as WANDA, SparseGPT, and RIA is needed.
