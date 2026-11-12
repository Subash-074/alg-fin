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

## Open a terminal in the project folder

In VS Code, choose **Terminal > New Terminal**. Make sure the terminal is in
the folder containing `naive.py` and `README.md`. If it is not, change to your
project folder with:

```powershell
cd "C:\path\to\alg-fin"
```

Check that Python is available:

```powershell
py --version
```

On Windows, use `py` in the commands below. If `python` works on your
computer, you can use `python` instead.

## Run the implementation

`naive.py` is a module that defines the attention and generation functions. It
does not print anything when run by itself; that is expected:

```powershell
py naive.py
```

To see generation output, create a file named `demo.py` next to `naive.py`,
paste in this example, then run `py demo.py`:

```python
from naive import naive_generate

# Example token embeddings for token IDs 0, 1, and 2.
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

The example should print generated tokens `[0, 1, 1, 1]`. `logits_fn` stands in
for a model's output projection and vocabulary head. The implementation picks
the highest-scoring vocabulary index; ties choose the lowest index.

## Run the tests

Run just the test file as a module:

```powershell
py -m unittest tests.test_naive -v
```

Or discover and run all tests in the project:

```powershell
py -m unittest discover -v
```

The suite currently has 12 tests. A successful run ends with `OK`. Run these
commands from the project folder. Use `-m unittest` rather than running
`tests\test_naive.py` directly; module mode lets Python find `naive.py` in the
project root.

## API usage

The example above imports the function with `from naive import naive_generate`.
It needs a prompt, an embedding function, query/key/value weight matrices, and
a function that returns one logit per vocabulary token.

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
