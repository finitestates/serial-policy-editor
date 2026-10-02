from __future__ import annotations

import numpy as np
import pytest
import rust_sampler as sampling
from trajectory_editor.core.errors import EditorError


def distribution(ids=(4, 1, 7), probabilities=(0.2, 0.5, 0.3), scores=(0.1, 0.9, 0.4)):
    return sampling.SparseDistribution(
        np.asarray(ids, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        None if scores is None else np.asarray(scores, dtype=np.float64),
    )


def test_numpy_vectors_are_rejected_instead_of_flattened_or_truncated():
    with pytest.raises(ValueError, match="one-dimensional"):
        sampling._validated_logits(np.ones((2, 2)))
    with pytest.raises(ValueError, match="candidate probabilities"):
        sampling.draw_token(
            sampling.SparseDistribution(
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([1.0], dtype=np.float64),
            ),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
        )
    with pytest.raises(ValueError, match="candidate scores do not match"):
        sampling.gaussian_ranking_scores(
            sampling.SparseDistribution(
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([0.5, 0.5]),
                np.asarray([1.0]),
            ),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
        )


def test_address_and_noise_validation_uses_editor_errors():
    with pytest.raises(EditorError, match="signed-64-bit"):
        sampling.position_uniform(1 << 63, "a" * 64, 0)
    with pytest.raises(EditorError, match="lowercase SHA-256"):
        sampling.position_uniform(1, "A" * 64, 0)
    with pytest.raises(EditorError, match="nonnegative integer"):
        sampling.position_uniform(1, "a" * 64, True)
    with pytest.raises(EditorError, match="finite and nonnegative"):
        sampling.gaussian_ranking_scores(
            distribution(),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            noise_std=float("inf"),
        )
    with pytest.raises(EditorError, match="finite and nonnegative"):
        sampling.perturbation_ranking_scores(
            distribution(),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            kernel="laplace-max",
            noise_std=-0.1,
        )
    for invalid in (-0.1, float("nan")):
        with pytest.raises(EditorError, match="finite and nonnegative"):
            sampling.gumbel_ranking_scores(
                distribution(),
                seed=1,
                stream_fingerprint="a" * 64,
                aligned_step=0,
                gumbel_noise_scale=invalid,
            )
    with pytest.raises(EditorError, match="finite and nonnegative"):
        sampling.perturbation_ranking_scores(
            distribution(),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            kernel="laplace-max",
            noise_std=float("inf"),
        )


def test_gumbel_model_rank_checks_and_historical_zero_scale_shortcut():
    value = distribution()
    with pytest.raises(ValueError, match="requires candidate model ranks"):
        sampling.gumbel_ranking_scores(
            value,
            seed=3,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            noise_address="model-rank",
        )
    with pytest.raises(ValueError, match="distinct positive integers"):
        sampling.gumbel_ranking_scores(
            value,
            seed=3,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            noise_address="model-rank",
            candidate_model_ranks=np.asarray([1, 1, 2], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="distinct positive integers"):
        sampling.gumbel_ranking_scores(
            value,
            seed=3,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            noise_address="model-rank",
            candidate_model_ranks=np.asarray([1.0, 2.0, 3.0]),
        )

    raw = np.asarray(value.scores, dtype=np.float64)
    zero_noise = sampling.gumbel_ranking_scores(
        value,
        seed="unused because scale is zero",
        stream_fingerprint="not a fingerprint",
        aligned_step=-7,
        candidate_model_ranks=np.ones((2, 2)),
        gumbel_noise_scale=0.0,
    )
    np.testing.assert_array_equal(zero_noise, raw)
    assert sampling.draw_token(
        value,
        seed="unused because scale is zero",
        stream_fingerprint="not a fingerprint",
        aligned_step=-7,
        kernel="gumbel-max",
        candidate_model_ranks=np.ones((2, 2)),
        gumbel_noise_scale=0.0,
    ) == 1


def test_exact_ties_and_duplicate_candidate_ids_keep_token_id_tie_break():
    value = distribution(ids=(9, 2, 2), probabilities=(0.3, 0.4, 0.3), scores=(1.0, 1.0, 1.0))
    assert sampling.gumbel_winner(value, np.asarray([4.0, 4.0, 4.0])) == 2
    assert sampling.gaussian_winner(value, np.asarray([0.0, 0.0, 0.0])) == 2
    assert sampling.perturbation_winner(value, np.asarray([0.0, 0.0, 0.0])) == 2


def test_empty_vectors_student_t_edges_and_conditional_prefix_validation():
    with pytest.raises(ValueError, match="finite nonempty"):
        sampling._softmax(np.asarray([], dtype=np.float64))
    with pytest.raises(EditorError, match="student_t_df"):
        sampling.perturbation_ranking_scores(
            distribution(),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            kernel="student-t-max",
            student_t_df=0.0,
        )
    with pytest.raises(EditorError, match="nonnegative token IDs"):
        sampling.conditional_gumbel_top_k(
            np.asarray([-0.1, -0.5]),
            count=2,
            parent_score=0.0,
            parent_log_probability=-0.1,
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            prefix_token_ids=[1, -1],
        )
