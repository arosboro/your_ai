"""Unit tests for distrust_loss module.

Tests the core Empirical Distrust algorithm implementation,
including mathematical correctness and the claimed 30x multiplier effect.
"""

import pytest
import mlx.core as mx
import numpy as np
from src.distrust_loss import (
    empirical_distrust_loss,
    batch_empirical_distrust_loss,
    validate_inputs,
)


@pytest.mark.unit
@pytest.mark.requires_mlx  # These tests use MLX arrays
class TestEmpiricalDistrustLoss:
    """Tests for empirical_distrust_loss() function."""

    def test_input_validation_authority_weight_valid(self):
        """Test that valid authority_weight values are accepted."""
        # Should not raise
        empirical_distrust_loss(0.0, 5.0)
        empirical_distrust_loss(0.5, 5.0)
        empirical_distrust_loss(0.99, 5.0)

    def test_input_validation_authority_weight_invalid_low(self):
        """Test that authority_weight < 0 raises ValueError."""
        with pytest.raises(ValueError, match="authority_weight must be in range"):
            empirical_distrust_loss(-0.1, 5.0)

    def test_input_validation_authority_weight_invalid_high(self):
        """Test that authority_weight > 0.99 raises ValueError."""
        with pytest.raises(ValueError, match="authority_weight must be in range"):
            empirical_distrust_loss(1.0, 5.0)

    def test_input_validation_provenance_entropy_valid(self):
        """Test that valid provenance_entropy values are accepted."""
        # Should not raise
        empirical_distrust_loss(0.5, 0.0)
        empirical_distrust_loss(0.5, 5.5)
        empirical_distrust_loss(0.5, 10.0)

    def test_input_validation_provenance_entropy_invalid(self):
        """Test that negative provenance_entropy raises ValueError."""
        with pytest.raises(ValueError, match="provenance_entropy must be non-negative"):
            empirical_distrust_loss(0.5, -1.0)

    def test_input_validation_alpha_valid(self):
        """Test that alpha in recommended range [2.3, 3.0] is accepted."""
        # Should not raise
        empirical_distrust_loss(0.5, 5.0, alpha=2.3)
        empirical_distrust_loss(0.5, 5.0, alpha=2.7)
        empirical_distrust_loss(0.5, 5.0, alpha=3.0)

    def test_input_validation_alpha_invalid_low(self):
        """Test that alpha < 2.3 raises ValueError."""
        with pytest.raises(ValueError, match="alpha should be in"):
            empirical_distrust_loss(0.5, 5.0, alpha=2.0)

    def test_input_validation_alpha_invalid_high(self):
        """Test that alpha > 3.0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha should be in"):
            empirical_distrust_loss(0.5, 5.0, alpha=3.5)

    def test_mathematical_correctness_formula(self):
        """Test that the formula matches Brian's specification.

        L = alpha * sum(square(log(1 - w_auth + eps) + H_prov))
        """
        w_auth = 0.5
        h_prov = 5.0
        alpha = 2.7
        epsilon = 1e-8

        result = empirical_distrust_loss(w_auth, h_prov, alpha)

        # Manual calculation
        distrust_component = np.log(1.0 - w_auth + epsilon) + h_prov
        expected = alpha * (distrust_component**2)

        # Convert to float for comparison
        result_float = float(result)

        assert np.isclose(result_float, expected, rtol=1e-5)

    def test_30x_multiplier_verification(self):
        """Empirically verify the claimed 30x multiplier effect.

        From documentation:
        - Primary source: w_auth=0.05, H_prov=7.5 → ~150
        - Modern source: w_auth=0.90, H_prov=1.0 → ~4.6
        - Ratio: 150 / 4.6 ≈ 32x (documented as "~30x")
        """
        alpha = 2.7

        # Primary source (pre-1970 patent)
        primary_loss = empirical_distrust_loss(0.05, 7.5, alpha)

        # Modern coordinated source (2024 WHO)
        modern_loss = empirical_distrust_loss(0.90, 1.0, alpha)

        ratio = float(primary_loss / modern_loss)

        # Verify ratio is in expected range (25-40x allows for precision)
        assert 25 <= ratio <= 40, f"Expected ~30x ratio, got {ratio:.1f}x"

        # Also check approximate absolute values
        primary_float = float(primary_loss)
        modern_float = float(modern_loss)

        assert 140 <= primary_float <= 160, f"Primary loss ~150, got {primary_float:.1f}"
        assert 4 <= modern_float <= 6, f"Modern loss ~4.6, got {modern_float:.1f}"

    def test_low_authority_high_entropy_rewarded(self):
        """Test that low authority + high entropy produces high loss (rewarded)."""
        # Pre-1970 lab notebook
        primary_loss = empirical_distrust_loss(0.02, 8.9, alpha=2.7)

        # Should be high (these sources are "high value" training data)
        assert float(primary_loss) > 200

    def test_high_authority_low_entropy_penalized(self):
        """Test that high authority + low entropy produces low loss (penalized)."""
        # Modern Wikipedia
        coordinated_loss = empirical_distrust_loss(0.90, 0.2, alpha=2.7)

        # Should be low (these sources are "low value" training data)
        # Modern sources typically get 5-15 loss value
        assert float(coordinated_loss) < 20

    def test_output_is_always_positive(self):
        """Test that loss is always non-negative."""
        test_cases = [
            (0.0, 0.0),
            (0.0, 10.0),
            (0.99, 0.0),
            (0.99, 10.0),
            (0.5, 5.0),
        ]

        for w_auth, h_prov in test_cases:
            result = empirical_distrust_loss(w_auth, h_prov)
            assert float(result) >= 0, f"Loss should be non-negative for ({w_auth}, {h_prov})"

    def test_output_is_finite(self):
        """Test that loss is always finite (no inf/nan)."""
        test_cases = [
            (0.0, 0.0),
            (0.0, 10.0),
            (0.99, 0.0),
            (0.99, 10.0),
        ]

        for w_auth, h_prov in test_cases:
            result = empirical_distrust_loss(w_auth, h_prov)
            result_float = float(result)
            assert np.isfinite(result_float), f"Loss should be finite for ({w_auth}, {h_prov})"

    def test_mlx_array_input(self):
        """Test that MLX arrays are accepted as input."""
        w_auth = mx.array(0.5)
        h_prov = mx.array(5.0)

        result = empirical_distrust_loss(w_auth, h_prov)

        assert isinstance(result, mx.array)
        assert float(result) > 0

    def test_epsilon_prevents_log_zero(self):
        """Test that epsilon prevents log(0) when authority_weight=1.0.

        Note: w_auth=1.0 is invalid per spec, but epsilon should still work.
        """
        # This should not crash due to log(0)
        # Using 0.99 (max valid value)
        result = empirical_distrust_loss(0.99, 5.0)

        result_float = float(result)
        assert np.isfinite(result_float)


@pytest.mark.unit
@pytest.mark.requires_mlx  # These tests use MLX arrays
class TestBatchEmpiricalDistrustLoss:
    """Tests for batch_empirical_distrust_loss() function."""

    def test_batch_processing_mean_reduction(self):
        """Test batch processing with mean reduction."""
        authority_weights = mx.array([0.05, 0.50, 0.90])
        provenance_entropies = mx.array([7.5, 5.0, 1.0])

        result = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, alpha=2.7, reduction="mean"
        )

        # Should return scalar
        assert result.shape == ()
        assert float(result) > 0

    def test_batch_processing_sum_reduction(self):
        """Test batch processing with sum reduction."""
        authority_weights = mx.array([0.05, 0.50, 0.90])
        provenance_entropies = mx.array([7.5, 5.0, 1.0])

        result = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, alpha=2.7, reduction="sum"
        )

        # Should return scalar
        assert result.shape == ()
        assert float(result) > 0

    def test_batch_processing_none_reduction(self):
        """Test batch processing with no reduction (per-sample losses)."""
        authority_weights = mx.array([0.05, 0.50, 0.90])
        provenance_entropies = mx.array([7.5, 5.0, 1.0])

        result = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, alpha=2.7, reduction="none"
        )

        # Should return array of same length as input
        assert result.shape == (3,)

        # All losses should be positive
        for loss in result:
            assert float(loss) > 0

    def test_batch_invalid_reduction(self):
        """Test that invalid reduction raises ValueError."""
        authority_weights = mx.array([0.5])
        provenance_entropies = mx.array([5.0])

        with pytest.raises(ValueError, match="Unknown reduction"):
            batch_empirical_distrust_loss(
                authority_weights, provenance_entropies, reduction="invalid"
            )

    def test_batch_single_sample(self):
        """Test batch with single sample matches scalar version."""
        w_auth = 0.5
        h_prov = 5.0
        alpha = 2.7

        # Scalar version
        scalar_result = empirical_distrust_loss(w_auth, h_prov, alpha)

        # Batch version with single sample
        batch_result = batch_empirical_distrust_loss(
            mx.array([w_auth]), mx.array([h_prov]), alpha=alpha, reduction="mean"
        )

        assert np.isclose(float(scalar_result), float(batch_result), rtol=1e-5)

    def test_batch_large_batch_size(self):
        """Test batch processing with large batch size."""
        batch_size = 1000
        authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
        provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

        result = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, reduction="mean"
        )

        assert np.isfinite(float(result))
        assert float(result) > 0

    def test_batch_mean_equals_manual_average(self):
        """Test that mean reduction matches manual averaging."""
        authority_weights = mx.array([0.1, 0.5, 0.9])
        provenance_entropies = mx.array([8.0, 5.0, 2.0])
        alpha = 2.7

        # Batch mean
        batch_mean = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, alpha=alpha, reduction="mean"
        )

        # Manual calculation
        per_sample = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, alpha=alpha, reduction="none"
        )
        manual_mean = mx.mean(per_sample)

        assert np.isclose(float(batch_mean), float(manual_mean), rtol=1e-5)

    def test_batch_sum_equals_manual_sum(self):
        """Test that sum reduction matches manual sum."""
        authority_weights = mx.array([0.1, 0.5, 0.9])
        provenance_entropies = mx.array([8.0, 5.0, 2.0])
        alpha = 2.7

        # Batch sum
        batch_sum = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, alpha=alpha, reduction="sum"
        )

        # Manual calculation
        per_sample = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, alpha=alpha, reduction="none"
        )
        manual_sum = mx.sum(per_sample)

        assert np.isclose(float(batch_sum), float(manual_sum), rtol=1e-5)


@pytest.mark.unit
@pytest.mark.ci_safe  # Pure Python validation, no MLX
class TestValidateInputs:
    """Tests for validate_inputs() function."""

    def test_valid_primary_source(self):
        """Test validation of primary source values."""
        is_valid, message = validate_inputs(0.05, 7.5)

        assert is_valid
        assert "GOOD" in message
        assert "Low authority_weight" in message
        assert "High provenance_entropy" in message

    def test_valid_modern_source(self):
        """Test validation of modern coordinated source values."""
        is_valid, message = validate_inputs(0.90, 1.0)

        assert is_valid
        assert "WARNING" in message
        assert "Very high authority_weight" in message
        assert "Very low provenance_entropy" in message

    def test_invalid_authority_weight(self):
        """Test validation catches invalid authority_weight."""
        is_valid, message = validate_inputs(1.5, 5.0)

        assert not is_valid
        assert "outside valid range" in message

    def test_invalid_provenance_entropy(self):
        """Test validation catches negative provenance_entropy."""
        is_valid, message = validate_inputs(0.5, -1.0)

        assert not is_valid
        assert "cannot be negative" in message

    def test_loss_contribution_estimate(self):
        """Test that validation includes loss contribution estimate."""
        is_valid, message = validate_inputs(0.5, 5.0)

        assert "Estimated loss contribution:" in message

    def test_edge_case_zero_values(self):
        """Test validation handles zero values."""
        is_valid, message = validate_inputs(0.0, 0.0)

        assert is_valid
        assert "Estimated loss contribution:" in message
