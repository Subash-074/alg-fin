import math
import unittest

from stage4_naive import naive_causal_attention, naive_generate


IDENTITY_2 = [[1.0, 0.0], [0.0, 1.0]]


class TestNaiveCausalAttention(unittest.TestCase):
    def test_output_shapes(self):
        x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        result = naive_causal_attention(x, IDENTITY_2, IDENTITY_2, IDENTITY_2)

        self.assertEqual(len(result.queries), 3)
        self.assertEqual(len(result.keys), 3)
        self.assertEqual(len(result.values), 3)
        self.assertEqual(len(result.scores), 3)
        self.assertEqual(len(result.weights), 3)
        self.assertEqual(len(result.contexts), 3)
        self.assertTrue(all(len(row) == 3 for row in result.scores))
        self.assertTrue(all(len(row) == 2 for row in result.contexts))

    def test_future_positions_are_masked(self):
        x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        result = naive_causal_attention(x, IDENTITY_2, IDENTITY_2, IDENTITY_2)

        for i, row in enumerate(result.scores):
            for j in range(i + 1, len(row)):
                self.assertEqual(row[j], float("-inf"))
                self.assertEqual(result.weights[i][j], 0.0)

    def test_attention_weights_sum_to_one(self):
        x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        result = naive_causal_attention(x, IDENTITY_2, IDENTITY_2, IDENTITY_2)

        for row in result.weights:
            self.assertAlmostEqual(sum(row), 1.0, places=7)
            self.assertTrue(all(weight >= 0.0 for weight in row))

    def test_first_position_attends_only_to_itself(self):
        x = [[2.0, 3.0], [4.0, 5.0]]
        result = naive_causal_attention(x, IDENTITY_2, IDENTITY_2, IDENTITY_2)

        self.assertEqual(result.weights[0], [1.0, 0.0])
        self.assertEqual(result.contexts[0], [2.0, 3.0])

    def test_future_token_does_not_change_earlier_context(self):
        short = [[1.0, 0.0], [0.0, 1.0]]
        extended = short + [[100.0, -100.0]]

        a = naive_causal_attention(short, IDENTITY_2, IDENTITY_2, IDENTITY_2)
        b = naive_causal_attention(extended, IDENTITY_2, IDENTITY_2, IDENTITY_2)

        self.assertEqual(a.contexts[0], b.contexts[0])
        self.assertEqual(a.contexts[1], b.contexts[1])

    def test_invalid_embedding_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            naive_causal_attention([[1.0, 2.0], [3.0]], IDENTITY_2, IDENTITY_2, IDENTITY_2)


class TestNaiveGeneration(unittest.TestCase):
    def setUp(self):
        self.embedding = {
            0: [1.0, 0.0],
            1: [0.0, 1.0],
            2: [1.0, 1.0],
        }

    def test_generation_appends_requested_number_of_tokens(self):
        result = naive_generate(
            prompt_tokens=[0],
            max_new_tokens=3,
            embedding_fn=self.embedding.__getitem__,
            w_q=IDENTITY_2,
            w_k=IDENTITY_2,
            w_v=IDENTITY_2,
            logits_fn=lambda context: [0.0, 1.0, 0.0],
        )
        self.assertEqual(result.tokens, [0, 1, 1, 1])
        self.assertEqual(len(result.tokens), 4)

    def test_prefix_is_recomputed_from_scratch_each_step(self):
        result = naive_generate(
            prompt_tokens=[0],
            max_new_tokens=3,
            embedding_fn=self.embedding.__getitem__,
            w_q=IDENTITY_2,
            w_k=IDENTITY_2,
            w_v=IDENTITY_2,
            logits_fn=lambda context: [1.0, 0.0, 0.0],
        )
        self.assertEqual(result.attention_pass_prefix_lengths, [1, 2, 3])

    def test_dense_qk_operation_count_for_lengths_1_2_3(self):
        # d_k = 2. For t=1,2,3, dense score slots = 1 + 4 + 9 = 14.
        # Multiplications = 14*2 = 28.
        # Additions per dot product = d_k-1 = 1, so additions = 14.
        result = naive_generate(
            prompt_tokens=[0],
            max_new_tokens=3,
            embedding_fn=self.embedding.__getitem__,
            w_q=IDENTITY_2,
            w_k=IDENTITY_2,
            w_v=IDENTITY_2,
            logits_fn=lambda context: [1.0, 0.0, 0.0],
        )
        self.assertEqual(result.qk_multiply_count, 28)
        self.assertEqual(result.qk_add_count, 14)
        self.assertEqual(result.qk_operation_count, 42)

    def test_zero_new_tokens_returns_prompt_unchanged(self):
        result = naive_generate(
            prompt_tokens=[0, 1],
            max_new_tokens=0,
            embedding_fn=self.embedding.__getitem__,
            w_q=IDENTITY_2,
            w_k=IDENTITY_2,
            w_v=IDENTITY_2,
            logits_fn=lambda context: [1.0, 0.0, 0.0],
        )
        self.assertEqual(result.tokens, [0, 1])
        self.assertEqual(result.attention_pass_prefix_lengths, [])
        self.assertEqual(result.qk_operation_count, 0)

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            naive_generate(
                prompt_tokens=[],
                max_new_tokens=1,
                embedding_fn=self.embedding.__getitem__,
                w_q=IDENTITY_2,
                w_k=IDENTITY_2,
                w_v=IDENTITY_2,
                logits_fn=lambda context: [1.0],
            )

    def test_negative_generation_length_is_rejected(self):
        with self.assertRaises(ValueError):
            naive_generate(
                prompt_tokens=[0],
                max_new_tokens=-1,
                embedding_fn=self.embedding.__getitem__,
                w_q=IDENTITY_2,
                w_k=IDENTITY_2,
                w_v=IDENTITY_2,
                logits_fn=lambda context: [1.0],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
