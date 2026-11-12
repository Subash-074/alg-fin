"""
Stage 4: Naive causal autoregressive generation (without KV caching).

This is a small, dependency-free reference implementation. At every generation step it:
  1. embeds the complete current token prefix,
  2. recomputes Q, K, and V for every position in that prefix,
  3. recomputes the full dense QK^T score matrix,
  4. applies the causal mask,
  5. computes attention for every query position,
  6. uses the final position's context to select the next token.

The implementation intentionally does NOT cache keys or values.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Callable, Dict, List, Sequence, Tuple, TypeVar

Token = TypeVar("Token")
Vector = List[float]
Matrix = List[List[float]]


@dataclass
class AttentionResult:
    """Intermediate tensors from one full-prefix causal-attention pass."""
    queries: Matrix
    keys: Matrix
    values: Matrix
    scores: Matrix
    weights: Matrix
    contexts: Matrix


@dataclass
class GenerationResult:
    """Generated token sequence and simple recomputation statistics."""
    tokens: List[Token]
    attention_pass_prefix_lengths: List[int]
    qk_multiply_count: int
    qk_add_count: int

    @property
    def qk_operation_count(self) -> int:
        """QK dot-product arithmetic count (multiplications + additions)."""
        return self.qk_multiply_count + self.qk_add_count


def _validate_matrix(matrix: Sequence[Sequence[float]], name: str) -> Tuple[int, int]:
    if not matrix:
        raise ValueError(f"{name} must contain at least one row")
    width = len(matrix[0])
    if width == 0:
        raise ValueError(f"{name} must contain at least one column")
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return len(matrix), width


def _matvec(vector: Sequence[float], weights: Sequence[Sequence[float]]) -> Vector:
    """Row-vector times matrix: (d_model,) x (d_model, d_out)."""
    rows, cols = _validate_matrix(weights, "weight matrix")
    if len(vector) != rows:
        raise ValueError("input vector width must match weight-matrix rows")
    return [
        sum(float(vector[i]) * float(weights[i][j]) for i in range(rows))
        for j in range(cols)
    ]


def _softmax(scores: Sequence[float]) -> Vector:
    """Numerically stable softmax. Supports -inf entries from the causal mask."""
    finite = [x for x in scores if x != float("-inf")]
    if not finite:
        raise ValueError("softmax requires at least one unmasked score")
    peak = max(finite)
    exps = [0.0 if x == float("-inf") else exp(x - peak) for x in scores]
    denominator = sum(exps)
    return [value / denominator for value in exps]


def naive_causal_attention(
    embeddings: Sequence[Sequence[float]],
    w_q: Sequence[Sequence[float]],
    w_k: Sequence[Sequence[float]],
    w_v: Sequence[Sequence[float]],
) -> AttentionResult:
    """
    Compute full causal self-attention for a supplied prefix.

    Future query-key pairs are still included in the dense QK^T calculation,
    then replaced with -inf by the causal mask. This mirrors the naive
    full-matrix recomputation described in the Stage 4 assignment.
    """
    length, d_model = _validate_matrix(embeddings, "embeddings")
    q_rows, d_k = _validate_matrix(w_q, "w_q")
    k_rows, k_cols = _validate_matrix(w_k, "w_k")
    v_rows, _ = _validate_matrix(w_v, "w_v")

    if q_rows != d_model or k_rows != d_model or v_rows != d_model:
        raise ValueError("W_Q, W_K, and W_V must have d_model rows")
    if k_cols != d_k:
        raise ValueError("W_Q and W_K must produce the same key dimension")

    queries = [_matvec(row, w_q) for row in embeddings]
    keys = [_matvec(row, w_k) for row in embeddings]
    values = [_matvec(row, w_v) for row in embeddings]

    scale = sqrt(d_k)
    scores: Matrix = []
    weights: Matrix = []
    contexts: Matrix = []

    # Dense t x t score matrix is computed first.
    for i in range(length):
        score_row = [
            sum(queries[i][m] * keys[j][m] for m in range(d_k)) / scale
            for j in range(length)
        ]

        # Causal mask: query i cannot attend to key positions j > i.
        masked_row = [
            score if j <= i else float("-inf")
            for j, score in enumerate(score_row)
        ]
        weight_row = _softmax(masked_row)
        context_row = [
            sum(weight_row[j] * values[j][v] for j in range(length))
            for v in range(len(values[0]))
        ]

        scores.append(masked_row)
        weights.append(weight_row)
        contexts.append(context_row)

    return AttentionResult(
        queries=queries,
        keys=keys,
        values=values,
        scores=scores,
        weights=weights,
        contexts=contexts,
    )


def naive_generate(
    prompt_tokens: Sequence[Token],
    max_new_tokens: int,
    embedding_fn: Callable[[Token], Sequence[float]],
    w_q: Sequence[Sequence[float]],
    w_k: Sequence[Sequence[float]],
    w_v: Sequence[Sequence[float]],
    logits_fn: Callable[[Sequence[float]], Sequence[float]],
) -> GenerationResult:
    """
    Generate tokens with no KV cache.

    logits_fn maps the final context vector to one score per vocabulary item.
    The next token is selected using deterministic argmax (ties choose the
    lowest index). The selected vocabulary index is appended as the token.

    For transparent operation counting, this function records the number of
    dense QK^T multiplications and additions performed at every step.
    """
    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    tokens: List[Token] = list(prompt_tokens)
    prefix_lengths: List[int] = []
    multiply_count = 0
    add_count = 0

    # W_Q determines d_k; validate dimensions through attention itself.
    _, d_k = _validate_matrix(w_q, "w_q")

    for _ in range(max_new_tokens):
        prefix_lengths.append(len(tokens))
        embeddings = [list(map(float, embedding_fn(token))) for token in tokens]
        result = naive_causal_attention(embeddings, w_q, w_k, w_v)

        # Dense QK^T: every query is dotted with every key, including
        # future positions that are masked immediately afterward.
        t = len(tokens)
        multiply_count += t * t * d_k
        add_count += t * t * max(d_k - 1, 0)

        logits = list(map(float, logits_fn(result.contexts[-1])))
        if not logits:
            raise ValueError("logits_fn must return at least one logit")
        next_token = max(range(len(logits)), key=lambda index: logits[index])
        tokens.append(next_token)  # vocabulary index is the generated token

    return GenerationResult(
        tokens=tokens,
        attention_pass_prefix_lengths=prefix_lengths,
        qk_multiply_count=multiply_count,
        qk_add_count=add_count,
    )
