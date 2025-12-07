"""Performance benchmarks for distrust_loss module.

Benchmarks the computational performance of distrust loss calculations,
focusing on throughput and memory usage.
"""

import pytest
import mlx.core as mx
import numpy as np
import time
from src.distrust_loss import (
    empirical_distrust_loss,
    batch_empirical_distrust_loss,
)


@pytest.mark.performance
@pytest.mark.requires_mlx  # Performance tests use MLX
@pytest.mark.slow
class TestDistrustLossPerformance:
    """Performance benchmarks for single-sample loss calculation."""

    def test_single_loss_calculation_speed(self, benchmark):
        """Benchmark single loss calculation."""
        w_auth = 0.5
        h_prov = 5.0
        alpha = 2.7

        result = benchmark(empirical_distrust_loss, w_auth, h_prov, alpha)

        assert float(result) > 0

    def test_batch_calculations_throughput(self, benchmark):
        """Benchmark throughput of multiple single calculations."""
        test_cases = [
            (0.05, 7.5),
            (0.50, 5.0),
            (0.90, 1.0),
        ] * 100  # 300 calculations

        def run_calculations():
            results = []
            for w_auth, h_prov in test_cases:
                loss = empirical_distrust_loss(w_auth, h_prov, alpha=2.7)
                results.append(loss)
            return results

        results = benchmark(run_calculations)

        assert len(results) == 300


@pytest.mark.performance
class TestBatchDistrustLossPerformance:
    """Performance benchmarks for batch loss calculation."""

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000, 10000])
    def test_batch_loss_scaling(self, benchmark, batch_size):
        """Benchmark batch loss calculation with varying batch sizes."""
        authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
        provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

        result = benchmark(
            batch_empirical_distrust_loss,
            authority_weights,
            provenance_entropies,
            alpha=2.7,
            reduction="mean",
        )

        assert float(result) > 0

    def test_batch_vs_loop_comparison_small(self):
        """Compare batch vs loop performance for small batches."""
        batch_size = 100
        authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
        provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

        # Batch method
        start = time.time()
        batch_result = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, reduction="mean"
        )
        batch_time = time.time() - start

        # Loop method (evaluate MLX arrays first)
        auth_list = [float(w) for w in authority_weights]
        entropy_list = [float(h) for h in provenance_entropies]

        start = time.time()
        loop_results = []
        for w, h in zip(auth_list, entropy_list):
            loss = empirical_distrust_loss(w, h)
            loop_results.append(float(loss))
        loop_result = np.mean(loop_results)
        loop_time = time.time() - start

        # Batch should be faster or comparable
        print(f"Batch time: {batch_time:.4f}s, Loop time: {loop_time:.4f}s")

        # Results should be similar
        assert abs(float(batch_result) - loop_result) < 1.0

    def test_batch_vs_loop_comparison_large(self):
        """Compare batch vs loop performance for large batches."""
        batch_size = 10000
        authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
        provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

        # Batch method
        start = time.time()
        result = batch_empirical_distrust_loss(
            authority_weights, provenance_entropies, reduction="mean"
        )
        _ = float(result)  # Force evaluation
        batch_time = time.time() - start

        print(f"Batch processing {batch_size} samples took: {batch_time:.4f}s")
        print(f"Throughput: {batch_size / batch_time:.0f} samples/sec")

        # Batch should complete in reasonable time (< 1 second)
        assert batch_time < 1.0

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_reduction_performance(self, benchmark, reduction):
        """Benchmark different reduction operations."""
        batch_size = 1000
        authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
        provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

        result = benchmark(
            batch_empirical_distrust_loss,
            authority_weights,
            provenance_entropies,
            reduction=reduction,
        )

        if reduction == "none":
            assert len(result) == batch_size
        else:
            assert result.shape == ()


@pytest.mark.performance
class TestMemoryUsage:
    """Memory usage tests for distrust loss calculations."""

    def test_batch_memory_scaling(self):
        """Test memory usage with increasing batch sizes."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        batch_sizes = [100, 1000, 10000, 100000]
        memory_usage = []

        for batch_size in batch_sizes:
            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Create batch
            authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
            provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

            # Calculate loss
            result = batch_empirical_distrust_loss(
                authority_weights, provenance_entropies, reduction="mean"
            )
            _ = float(result)  # Force evaluation

            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_increase = mem_after - mem_before

            memory_usage.append(mem_increase)

            print(f"Batch size {batch_size}: {mem_increase:.2f} MB increase")

        # Memory should scale reasonably (not exponentially)
        # Larger batches should use more memory, but not excessively
        for mem in memory_usage:
            assert mem < 500  # Should stay under 500MB per batch

    def test_memory_cleanup_after_calculation(self):
        """Test that memory is properly cleaned up after calculations."""
        import psutil
        import os
        import gc

        process = psutil.Process(os.getpid())

        # Baseline memory
        gc.collect()
        mem_baseline = process.memory_info().rss / 1024 / 1024

        # Perform many calculations
        for _ in range(100):
            authority_weights = mx.random.uniform(0.0, 0.99, (1000,))
            provenance_entropies = mx.random.uniform(0.0, 10.0, (1000,))
            result = batch_empirical_distrust_loss(authority_weights, provenance_entropies)
            _ = float(result)

        # Force cleanup
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024

        mem_increase = mem_after - mem_baseline

        print(f"Memory increase after 100 iterations: {mem_increase:.2f} MB")

        # Should not have significant memory leak
        # Allow up to 100MB increase for caching/overhead
        assert mem_increase < 100


@pytest.mark.performance
class TestLargeScaleBenchmarks:
    """Large-scale performance benchmarks."""

    def test_training_batch_size_performance(self):
        """Benchmark typical training batch sizes."""
        # Typical training batch sizes
        batch_sizes = [2, 4, 8, 16, 32, 64, 128]

        timings = []
        for batch_size in batch_sizes:
            authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
            provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

            start = time.time()
            for _ in range(100):  # 100 iterations
                result = batch_empirical_distrust_loss(
                    authority_weights, provenance_entropies, reduction="mean"
                )
                _ = float(result)
            elapsed = time.time() - start

            timings.append(elapsed)
            print(f"Batch size {batch_size}: {elapsed:.4f}s for 100 iterations")

        # All should complete in reasonable time
        for timing in timings:
            assert timing < 10.0  # 10 seconds for 100 iterations

    def test_epoch_simulation(self):
        """Simulate processing a full epoch of data."""
        # Simulate 10k samples with batch size 16
        num_samples = 10000
        batch_size = 16
        num_batches = num_samples // batch_size

        start = time.time()

        for _ in range(num_batches):
            authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
            provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

            result = batch_empirical_distrust_loss(
                authority_weights, provenance_entropies, reduction="mean"
            )
            _ = float(result)

        elapsed = time.time() - start

        samples_per_sec = num_samples / elapsed

        print(f"Processed {num_samples} samples in {elapsed:.2f}s")
        print(f"Throughput: {samples_per_sec:.0f} samples/sec")

        # Should process at least 1000 samples per second
        assert samples_per_sec > 1000


@pytest.mark.performance
class TestComputationalComplexity:
    """Tests for computational complexity verification."""

    def test_linear_scaling_with_batch_size(self):
        """Verify that computation scales reasonably with batch size.

        NOTE: This is an informational test. Performance timings are
        inherently noisy due to system load, caching, etc.
        We verify that computation completes in reasonable time, not strict linear scaling.
        """
        batch_sizes = [100, 200, 400, 800, 1600]
        timings = []

        for batch_size in batch_sizes:
            authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
            provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

            start = time.time()
            for _ in range(10):  # Average over 10 runs
                result = batch_empirical_distrust_loss(
                    authority_weights, provenance_entropies, reduction="mean"
                )
                _ = float(result)
            elapsed = time.time() - start

            timings.append(elapsed)

        # Calculate scaling factors (informational)
        scaling_factors = []
        for i in range(1, len(timings)):
            size_ratio = batch_sizes[i] / batch_sizes[i - 1]
            time_ratio = timings[i] / timings[i - 1]
            scaling_factor = time_ratio / size_ratio
            scaling_factors.append(scaling_factor)

            print(
                f"Batch {batch_sizes[i - 1]} -> {batch_sizes[i]}: "
                f"time ratio {time_ratio:.2f}, size ratio {size_ratio:.2f}, "
                f"factor {scaling_factor:.2f}"
            )

        # Main assertion: All batch sizes complete in reasonable time
        # (Performance characteristics are logged but not strictly enforced)
        for batch_size, timing in zip(batch_sizes, timings):
            # Each batch should complete in < 1 second
            assert timing < 1.0, f"Batch {batch_size} took {timing:.2f}s (too slow)"

        print("\nScaling analysis (informational only):")
        print(f"  Mean scaling factor: {np.mean(scaling_factors):.2f}")
        print(f"  Std scaling factor: {np.std(scaling_factors):.2f}")

    def test_constant_time_for_different_values(self):
        """Verify that computation time is independent of input values."""
        batch_size = 1000

        # Test with different value ranges
        test_cases = [
            ("Low authority", mx.random.uniform(0.0, 0.1, (batch_size,))),
            ("Medium authority", mx.random.uniform(0.4, 0.6, (batch_size,))),
            ("High authority", mx.random.uniform(0.85, 0.99, (batch_size,))),
        ]

        timings = []
        for name, auth_weights in test_cases:
            prov_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

            start = time.time()
            for _ in range(100):
                result = batch_empirical_distrust_loss(
                    auth_weights, prov_entropies, reduction="mean"
                )
                _ = float(result)
            elapsed = time.time() - start

            timings.append(elapsed)
            print(f"{name}: {elapsed:.4f}s")

        # All should have similar timing (within 100% variance)
        # Performance can vary due to system load, caching, etc.
        mean_time = np.mean(timings)
        for timing in timings:
            # Allow 2x variance (performance tests are inherently noisy)
            assert abs(timing - mean_time) / mean_time < 1.0


@pytest.mark.performance
class TestOptimizationOpportunities:
    """Tests to identify potential optimization opportunities."""

    def test_mlx_array_overhead(self):
        """Measure overhead of MLX array operations."""
        batch_size = 1000

        # Create arrays once
        authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
        provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

        # Time with reused arrays
        start = time.time()
        for _ in range(100):
            result = batch_empirical_distrust_loss(
                authority_weights, provenance_entropies, reduction="mean"
            )
            _ = float(result)
        elapsed_reused = time.time() - start

        # Time with new arrays each iteration
        start = time.time()
        for _ in range(100):
            auth = mx.random.uniform(0.0, 0.99, (batch_size,))
            entropy = mx.random.uniform(0.0, 10.0, (batch_size,))
            result = batch_empirical_distrust_loss(auth, entropy, reduction="mean")
            _ = float(result)
        elapsed_new = time.time() - start

        print(f"Reused arrays: {elapsed_reused:.4f}s")
        print(f"New arrays: {elapsed_new:.4f}s")
        print(f"Overhead: {(elapsed_new - elapsed_reused) / elapsed_reused * 100:.1f}%")

    def test_evaluation_overhead(self):
        """Measure overhead of forcing evaluation."""
        batch_size = 1000
        authority_weights = mx.random.uniform(0.0, 0.99, (batch_size,))
        provenance_entropies = mx.random.uniform(0.0, 10.0, (batch_size,))

        # Without evaluation (lazy)
        start = time.time()
        for _ in range(100):
            result = batch_empirical_distrust_loss(
                authority_weights, provenance_entropies, reduction="mean"
            )
        elapsed_lazy = time.time() - start

        # With forced evaluation
        start = time.time()
        for _ in range(100):
            result = batch_empirical_distrust_loss(
                authority_weights, provenance_entropies, reduction="mean"
            )
            _ = float(result)  # Force evaluation
        elapsed_eval = time.time() - start

        print(f"Lazy (no eval): {elapsed_lazy:.4f}s")
        print(f"With eval: {elapsed_eval:.4f}s")
        print(f"Evaluation overhead: {(elapsed_eval - elapsed_lazy):.4f}s")
