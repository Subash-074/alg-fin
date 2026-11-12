# Stage 4 — Naive Causal Autoregressive Generation

A dependency-free Python reference implementation of "naive autoregressive generation without a KV cache.

## What this implementation demonstrates

At every generation step, the implementation processes the entire current token prefix again:

1. Convert every token in the prefix into an embedding.
2. Recompute all queries `Q`, keys `K`, and values `V`.
3. Recompute the dense `t × t` score matrix `QKᵀ / √dₖ`.
4. Apply the causal mask: positions `j > i` are replaced with `-∞`.
5. Apply softmax and compute all context vectors.
6. Use the final context vector to choose the next token.
7. Append that token and repeat — without retaining old K/V tensors.

This is intentionally a **naive baseline**. It does not implement KV caching or any later optimization stage.

## Files

- `stage4_naive.py` — implementation.
- `test_stage4_naive.py` — automated unit tests using Python's built-in `unittest`.

## Requirements

- Python 3.9+
- No third-party packages.

## Run the tests

From this directory:

```bash
python -m unittest -v
```

Expected result: all 12 tests pass.

## Minimal usage example

```python
from stage4_naive import naive_generate

# Example token embeddings (token IDs are 0, 1, and 2).
embeddings = {
    0: [1.0, 0.0],
    1: [0.0, 1.0],
    2: [1.0, 1.0],
}

identity = [
    [1.0, 0.0],
    [0.0, 1.0],
]

result = naive_generate(
    prompt_tokens=[0],
    max_new_tokens=3,
    embedding_fn=embeddings.__getitem__,
    w_q=identity,
    w_k=identity,
    w_v=identity,
    # Return one score per vocabulary token.
    logits_fn=lambda context: [0.1, 0.9, 0.2],
)

print("Generated tokens:", result.tokens)
print("Prefix lengths recomputed:", result.attention_pass_prefix_lengths)
print("QK multiplications:", result.qk_multiply_count)
print("QK additions:", result.qk_add_count)
```

The example uses deterministic argmax selection. The `logits_fn` is a small stand-in for the model's output projection and vocabulary head; this project isolates the Stage 4 attention/generation loop.

## Complexity

For a prefix of length `t`, dense `QKᵀ` computes `t²` query–key dot products. Each dot product has `dₖ` multiplications and `dₖ - 1` additions.

Across generation steps with prefix lengths `1, 2, ..., T`:

- Score slots: `Σ t² = T(T+1)(2T+1)/6 = Θ(T³)`.
- QK multiplications: `dₖ Σ t²`.
- QK additions: `(dₖ-1) Σ t²`.
- Total QK arithmetic: `(2dₖ-1) Σ t² = Θ(T³dₖ)`.
- If the assignment counts exactly `dₖ` multiply-add pairs per score, its convention gives `2dₖ Σt² = [T(T+1)(2T+1)/3]dₖ`.
- Peak materialized score-matrix storage at the final step: `Θ(T²)`.

**Counting convention note:** the code counts `dₖ` multiplications and `dₖ−1` additions for each dot product, which is the conventional scalar arithmetic count. The assignment's formula counts `dₖ` multiply-add pairs per score (i.e. `2dₖ` scalar operations). State the convention explicitly when reporting the number.

## Scope

This repository component covers **Stage 4 only**. The next-stage KV-cache implementation is deliberately not included.
