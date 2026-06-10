# Global Lookahead Pruning for Transformer-Based Language Models

An activation-aware post-training pruning method for transformer-based language models.
This project explores whether future-layer activation information can be used to identify less important weights and prune transformer language models while preserving language modeling performance.

The core idea is simple: instead of pruning only by weight magnitude, this method scores each weight using both its magnitude and the change in mean activation between the next two transformer layers.

---

## Highlights

* Implemented a custom pruning method for transformer-based language models
* Designed a **Global Lookahead Pruning** metric using weight magnitude and future-layer activation differences
* Compared the method with traditional pruning baselines:

  * Global magnitude pruning
  * Layerwise magnitude pruning
* Evaluated models using:

  * Perplexity
  * Sparsity
* Used **WikiText-2** for calibration and evaluation
* Tested on:

  * GPT-2
  * Qwen-2.5B
  * DeepSeekR1
* Added optional fine-tuning to recover model performance after pruning
* Organized the project into reusable source code, scripts, results, and documentation

---

## Project Summary

Large language models contain millions or billions of parameters. This makes them expensive to store, deploy, and run, especially in resource-constrained environments.

Pruning is a model compression technique that removes less important weights from a neural network. The challenge is deciding which weights can be safely removed without heavily damaging model performance.

This project proposes **Global Lookahead Pruning**, a post-training pruning method for transformer language models. The method uses forward activations from a calibration dataset to estimate how important each layer is, then combines this activation information with weight magnitude to prune less important weights globally.

The method does not require gradients during the pruning decision phase. It only needs a pretrained model and a small calibration dataset.

---

## Motivation

Transformer-based language models are powerful, but they are computationally expensive. Their large parameter count increases:

* Memory usage
* Inference cost
* Deployment difficulty
* Energy consumption
* Hardware requirements

Traditional magnitude pruning removes weights with the smallest absolute values. However, small weights are not always unimportant, especially in transformer architectures where different layers have different roles.

Activation-aware pruning methods such as WANDA and RIA show that using activation information can improve pruning decisions. This project builds on that idea by looking ahead into future transformer layers and using the change in activation statistics as a proxy for layer importance.

---

## Core Idea

The main hypothesis of this project is:

> A weight should be considered more important if it belongs to a part of the model where future layer activations change significantly.

In other words, if the next layers show a large change in mean activation, the current layer may be involved in an important transformation. Weights in such regions should be preserved more carefully.

If the future activation difference is small, the layer may be more redundant or less sensitive, making its weights safer to prune.

---

## Method: Global Lookahead Pruning

Global Lookahead Pruning assigns an importance score to each weight using two components:

1. **Weight magnitude**
   Larger weights are usually more important than very small weights.

2. **Lookahead activation difference**
   The method looks at the mean activation difference between the next two layers.

The final importance score is computed as:

```text
importance = |weight| × sqrt(|mean(A_L+1) - mean(A_L+2)|)
```

Weights with the lowest global importance scores are pruned.

---

## Mathematical Formulation

For a weight ( W_{ij}^{(L)} ) in layer ( L ), the Lookahead importance score is:

[
I_{ij}^{(L)} = |W_{ij}^{(L)}| \cdot \sqrt{|\mu(A^{(L+1)}) - \mu(A^{(L+2)})|}
]

Where:

* ( W_{ij}^{(L)} ) is a weight in layer ( L )
* ( A^{(L+1)} ) is the activation after the next layer
* ( A^{(L+2)} ) is the activation after the layer after that
* ( \mu(\cdot) ) represents the mean activation
* ( I_{ij}^{(L)} ) is the final importance score

The square root is used to dampen very large activation differences so that the activation term does not dominate the weight magnitude completely.

---

## Algorithm

The pruning process follows these steps:

1. Load a pretrained transformer language model.
2. Run a calibration forward pass using WikiText-2.
3. Collect mean activations from transformer layers.
4. Compute the Lookahead activation difference between future layers.
5. Calculate an importance score for each eligible weight.
6. Flatten all importance scores across the model.
7. Rank weights globally.
8. Prune weights with the lowest scores.
9. Evaluate the pruned model using perplexity and sparsity.
10. Optionally fine-tune the pruned model to recover performance.

---

## Why Global Pruning?

Layerwise pruning removes a fixed percentage of weights from every layer. This is simple, but it assumes every layer can tolerate the same amount of pruning.

Global pruning ranks weights across the whole model and allows different layers to receive different sparsity levels.

This is useful because transformer layers are not equally sensitive. Some layers may be highly important and should remain mostly dense, while others may contain more redundant weights.

Global Lookahead Pruning uses activation differences to guide this global ranking process.

---

## Repository Structure

```text
global-lookahead-pruning/
│
├── src/
│   ├── model_loader.py
│   ├── dataset_loader.py
│   ├── pruning.py
│   ├── evaluation.py
│   ├── finetuning.py
│   └── utils.py
│
├── scripts/
│   ├── run_baseline_eval.py
│   ├── run_global_magnitude.py
│   ├── run_layerwise_magnitude.py
│   ├── run_lookahead_pruning.py
│   └── run_finetune_pruned_model.py
│
├── notebooks/
│   └── exploratory_pruning_experiments.ipynb
│
├── results/
│   ├── tables/
│   │   └── pruning_results.csv
│   └── figures/
│       ├── fine_tuning_curve.png
│       ├── sparsity_comparison.png
│       ├── perplexity_comparison.png
│       └── perplexity_vs_sparsity.png
│
├── docs/
│   ├── method.md
│   ├── experiment_setup.md
│   └── limitations.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/global-lookahead-pruning.git
cd global-lookahead-pruning
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

For Windows:

```bash
.venv\Scripts\activate
```

For Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Main dependencies:

```text
torch
transformers
datasets
numpy
pandas
matplotlib
tqdm
scikit-learn
jupyter
```

For GPU usage, install the correct PyTorch version for your CUDA setup from the official PyTorch installation page.

---

## Usage

### 1. Evaluate Dense Baseline

```bash
python scripts/run_baseline_eval.py --model gpt2
```

This evaluates the original dense model before pruning.

---

### 2. Run Global Magnitude Pruning

```bash
python scripts/run_global_magnitude.py --model gpt2 --sparsity 0.3
```

This removes the globally smallest weights by absolute magnitude.

---

### 3. Run Layerwise Magnitude Pruning

```bash
python scripts/run_layerwise_magnitude.py --model gpt2 --sparsity 0.3
```

This removes a fixed percentage of weights from each layer.

---

### 4. Run Global Lookahead Pruning

```bash
python scripts/run_lookahead_pruning.py --model gpt2 --sparsity 0.3
```

This applies the proposed Lookahead pruning method.

To save the pruned model:

```bash
python scripts/run_lookahead_pruning.py --model gpt2 --sparsity 0.3 --save-model
```

---

### 5. Fine-Tune a Pruned Model

```bash
python scripts/run_finetune_pruned_model.py \
  --model-path outputs/models/gpt2_lookahead_30pct \
  --epochs 3
```

The fine-tuning script preserves sparsity by reapplying pruning masks after optimizer updates.

---

## Experimental Setup

### Dataset

The project uses **WikiText-2**, a standard benchmark for language modeling.

WikiText-2 is used for:

* Calibration
* Perplexity evaluation
* Fine-tuning experiments

Calibration data is used to collect activation statistics. Evaluation data is used to measure perplexity after pruning.

---

### Models

The method was evaluated on three transformer-based language models:

| Model      | Description                                              |
| ---------- | -------------------------------------------------------- |
| GPT-2      | Main experimental model                                  |
| Qwen-2.5B  | Larger transformer language model                        |
| DeepSeekR1 | Additional transformer model used for robustness testing |

---

### Baselines

The proposed method was compared against two pruning baselines.

#### 1. Global Magnitude Pruning

This baseline ranks all weights in the model by absolute magnitude and prunes the smallest weights globally.

#### 2. Layerwise Magnitude Pruning

This baseline prunes a fixed fraction of weights from each layer independently using magnitude.

---

### Metrics

The project uses two main metrics.

#### Perplexity

Perplexity measures language modeling performance. Lower perplexity means better model performance.

#### Sparsity

Sparsity measures the fraction of weights set to zero.

```text
sparsity = zero parameters / total parameters
```

A good pruning method should achieve high sparsity while keeping perplexity close to the dense baseline.

---

## Main Results

| Model      |            Method | Perplexity | Sparsity |
| ---------- | ----------------: | ---------: | -------: |
| Qwen-2.5B  |          Baseline |      17.62 |    0.000 |
| Qwen-2.5B  | Layerwise Pruning |      30.51 |    0.299 |
| Qwen-2.5B  | Lookahead Pruning |      22.60 |    0.145 |
| GPT-2      |          Baseline |      45.73 |    0.000 |
| GPT-2      | Layerwise Pruning |      62.58 |    0.299 |
| GPT-2      | Lookahead Pruning |      63.01 |    0.087 |
| DeepSeekR1 |          Baseline |      68.58 |    0.000 |
| DeepSeekR1 | Layerwise Pruning |      82.84 |    0.261 |
| DeepSeekR1 | Lookahead Pruning |      90.20 |    0.145 |

---

## Result Analysis

### Qwen-2.5B

On Qwen-2.5B, Lookahead pruning performed better than layerwise pruning in terms of perplexity.

* Baseline perplexity: 17.62
* Layerwise pruning perplexity: 30.51
* Lookahead pruning perplexity: 22.60

Although Lookahead pruning produced lower sparsity than layerwise pruning, it preserved model performance better.

This suggests that the Lookahead metric was useful for identifying safer weights to prune in Qwen-2.5B.

---

### GPT-2

On GPT-2, Lookahead pruning achieved similar perplexity to layerwise pruning but at lower sparsity.

* Baseline perplexity: 45.73
* Layerwise pruning perplexity: 62.58
* Lookahead pruning perplexity: 63.01

The method was conservative on GPT-2, pruning fewer weights to avoid large performance degradation.

Fine-tuning played an important role in improving GPT-2 performance after pruning.

---

### DeepSeekR1

On DeepSeekR1, Lookahead pruning did not outperform the layerwise baseline.

* Baseline perplexity: 68.58
* Layerwise pruning perplexity: 82.84
* Lookahead pruning perplexity: 90.20

This shows that the current Lookahead metric is architecture-sensitive. Mean activation difference may not fully capture weight importance for every transformer model.

This is an important limitation and a useful direction for future improvement.

---

## Fine-Tuning Analysis

Fine-tuning was used after pruning to recover model performance.

The experiments showed that iterative pruning followed by fine-tuning can significantly improve perplexity compared with direct one-shot pruning.

For GPT-2, iterative Lookahead pruning with fine-tuning reached around 30% pruning with validation perplexity close to 11.95 in the reported experiments.

This suggests that the remaining weights can adapt after pruning when fine-tuned properly.

---

## Key Findings

* Lookahead pruning is more conservative than layerwise pruning in several experiments.
* On Qwen-2.5B, Lookahead pruning achieved lower perplexity than layerwise pruning.
* On GPT-2, fine-tuning was important for recovering performance after pruning.
* On DeepSeekR1, Lookahead pruning did not outperform layerwise pruning.
* The method works best when activation differences clearly separate important and less important layers.
* Mean activation difference alone may not be enough for all transformer architectures.
* Iterative prune-and-fine-tune is more effective than single-step pruning for higher sparsity.

---

## Strengths

* Simple and lightweight pruning metric
* Does not require gradient computation during pruning
* Uses only forward activations
* Compatible with post-training pruning
* Can be combined with fine-tuning
* Provides a global pruning strategy instead of fixed layerwise pruning
* Easy to implement with PyTorch and Hugging Face Transformers

---

## Limitations

* The current implementation focuses mainly on unstructured pruning.
* Unstructured sparsity may not directly speed up inference on normal GPUs.
* The method uses mean activation difference, which may miss other important activation statistics such as variance or norm.
* Results depend on the calibration dataset.
* The method may behave differently across model architectures.
* Direct comparison with SparseGPT, WANDA, and RIA is not included in this implementation.
* Larger-scale testing is needed on more transformer models and downstream tasks.

---

## Future Work

Possible future improvements include:

* Add structured N:M pruning support
* Integrate hardware-friendly sparsity patterns such as 2:4 sparsity
* Compare directly with SparseGPT, WANDA, and RIA
* Test on larger models such as LLaMA-style architectures
* Use richer activation statistics such as variance, norm, or higher-order moments
* Combine pruning with quantization
* Add downstream task evaluation beyond perplexity
* Automate threshold selection based on acceptable perplexity degradation
* Improve robustness across different transformer architectures

---

## Technical Skills Demonstrated

This project demonstrates practical experience with:

* PyTorch
* Hugging Face Transformers
* Language model evaluation
* Model pruning
* Post-training model compression
* Activation hooks
* Perplexity evaluation
* Fine-tuning
* Experiment scripting
* Research implementation
* Machine learning project organization

---

## Reproducibility Notes

The repository is organized so that experiments can be run through command-line scripts.

Recommended execution order:

```bash
python scripts/run_baseline_eval.py --model gpt2
```

```bash
python scripts/run_global_magnitude.py --model gpt2 --sparsity 0.3
```

```bash
python scripts/run_layerwise_magnitude.py --model gpt2 --sparsity 0.3
```

```bash
python scripts/run_lookahead_pruning.py --model gpt2 --sparsity 0.3
```

```bash
python scripts/run_lookahead_pruning.py --model gpt2 --sparsity 0.3 --save-model
```

```bash
python scripts/run_finetune_pruned_model.py \
  --model-path outputs/models/gpt2_lookahead_30pct \
  --epochs 3
```

---

## Visual Results

If result figures are available, place them inside `results/figures/` and enable the following section.

### Fine-Tuning Curve

![Fine-Tuning Curve](results/figures/fine_tuning_curve.png)

### Sparsity Comparison

![Sparsity Comparison](results/figures/sparsity_comparison.png)

### Perplexity Comparison

![Perplexity Comparison](results/figures/perplexity_comparison.png)

### Perplexity vs Sparsity

![Perplexity vs Sparsity](results/figures/perplexity_vs_sparsity.png)

---

## Project Status

This repository contains a research implementation of Global Lookahead Pruning. The current version focuses on demonstrating the method, reproducing pruning experiments, and comparing the proposed approach with basic magnitude-based pruning baselines.

The project is suitable for further extension into a more complete model compression framework.

---

## Authors

* Al Mahfuz
* Dewan MD Saif
* Motasim Abid
* Al-Amin Rabbi
* Nabeel Mohammed

---

## References

[1] Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems (NeurIPS)*.

[2] Chowdhery, A., et al. (2022). PaLM: Scaling Language Modeling with Pathways. *arXiv preprint arXiv:2204.02311*.

[3] LeCun, Y., Denker, J. S., & Solla, S. A. (1990). Optimal Brain Damage. *Advances in Neural Information Processing Systems*.

[4] Han, S., et al. (2015). Learning Both Weights and Connections for Efficient Neural Networks. *Advances in Neural Information Processing Systems*.

[5] Frantar, M., & Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. *arXiv preprint arXiv:2301.00774*.

[6] Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2024). WANDA: A Simple and Effective Pruning Approach for Large Language Models. *International Conference on Learning Representations (ICLR)*.

[7] Zhang, Y., et al. (2024). Plug-and-Play: An Efficient Post-Training Pruning Method for Large Language Models. *International Conference on Learning Representations (ICLR)*.

[8] Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer Sentinel Mixture Models. *arXiv preprint arXiv:1609.07843*.

---

## Summary

Global Lookahead Pruning is a lightweight post-training pruning method for transformer-based language models. It combines weight magnitude with future-layer activation differences to estimate weight importance. The method provides a practical way to explore activation-aware pruning, compare sparsity–perplexity trade-offs, and study the effect of fine-tuning after pruning.

The project shows that activation-aware pruning can preserve model performance better than simple magnitude pruning in some settings, while also revealing important limitations across different architectures.
