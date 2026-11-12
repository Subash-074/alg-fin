# Naive Causal Autoregressive Generation

A dependency-free Python reference implementation of autoregressive generation
without a key/value (KV) cache.

## Overview

At each generation step, this implementation recomputes attention over the
entire token prefix:

1. Convert each token to an embedding.
2. Recompute the query (`Q`), key (`K`), and value (`V`) vectors.
3. Compute the dense scaled score matrix `QKᵀ / √dₖ`.
4. Apply a causal mask so a position cannot attend to future positions.
5. Apply softmax and compute each position's context vector.
6. Use the final context vector to choose the next token, then repeat.

This is deliberately a naive baseline: it does not cache keys or values.

## Project layout

```text
alg-fin/
├── naive.py
├── tests/
│   ├── __init__.py
│   └── test_naive.py
└── README.md
```

## Requirements

- Python 3.9 or newer
- No third-party packages

## Run the tests

From the project directory:

```bash
python -m unittest discover -v
```

The suite contains 12 tests covering attention masking, output shapes,
generation behavior, input validation, and QK operation counts.

## Usage

```python
from naive import naive_generate

embeddings = {
    0: [1.0, 0.0],
    1: [0.0, 1.0],
    2: [1.0, 1.0],
}
identity = [[1.0, 0.0], [0.0, 1.0]]

result = naive_generate(
    prompt_tokens=[0],
    max_new_tokens=3,
    embedding_fn=embeddings.__getitem__,
    w_q=identity,
    w_k=identity,
    w_v=identity,
    logits_fn=lambda context: [0.1, 0.9, 0.2],
)

print("Generated tokens:", result.tokens)
print("Recomputed prefix lengths:", result.attention_pass_prefix_lengths)
print("QK multiplications:", result.qk_multiply_count)
print("QK additions:", result.qk_add_count)
```

`logits_fn` stands in for a model's output projection and vocabulary head. The
implementation selects the highest-scoring vocabulary index; ties choose the
lowest index.

## Complexity

For prefix length `t` and key dimension `dₖ`, dense `QKᵀ` computes `t²` dot
products. A dot product uses `dₖ` multiplications and `dₖ - 1` additions.
Across generation steps with prefix lengths `1, 2, ..., T`:

- Score slots: `Σt² = T(T+1)(2T+1)/6 = Θ(T³)`.
- QK multiplications: `dₖ Σt²`.
- QK additions: `(dₖ - 1) Σt²`.
- Total QK arithmetic: `(2dₖ - 1) Σt² = Θ(T³dₖ)`.
- Peak score-matrix storage: `Θ(T²)`.

These counts cover QK dot-product arithmetic only; they exclude scaling,
softmax, and value aggregation.
