# Metal Backend Status Report

**Date**: December 9, 2025
**System**: macOS 15.6.1, Metal SDK v17.2
**MLX Version**: v0.25.1
**mlx-rs Version**: 0.25.2

## Executive Summary

**Metal backend CANNOT be enabled** on macOS 15.6.1 with MLX v0.25.1 due to Metal shader compiler incompatibility. This is an **upstream issue** in MLX that requires either Apple SDK updates or MLX library updates to resolve.

## Test Results

### ✅ What Works

- **CPU-only backend**: Fully functional and stable
- **ARM64 compilation**: Proper architecture detection
- **All mlx-rs APIs**: Working correctly in CPU mode
- **Training pipeline**: Operational on CPU

### ❌ What Doesn't Work

- **Metal shader compilation**: Fails with atomic operation errors
- **GPU acceleration**: Not available
- **Metal backend**: Blocked by shader incompatibility

## Technical Details

### Error Summary

When attempting to enable Metal, the build fails with:

```
error: no matching function for call to 'atomic_load_explicit'
error: no matching function for call to 'atomic_compare_exchange_weak_explicit'
```

These errors occur in MLX's Metal kernels (`quantized.metal`, `reduce.metal`, etc.) when compiled with Metal SDK v17.2.54.

### Root Cause

MLX v0.25.1's Metal shaders use atomic operations that are **incompatible** with the Metal SDK shipped with macOS 15.6.1. Specifically:

- MLX's `atomic.h` wrapper expects different atomic operation signatures
- Metal SDK v17.2 has stricter type requirements for `_valid_load_type` and `_valid_compare_exchange_type`
- The shader compiler rejects MLX's atomic wrappers as invalid

### Affected Components

- `mlx/backend/metal/kernels/quantized.metal`
- `mlx/backend/metal/kernels/reduce.metal`
- `mlx/backend/metal/kernels/steel_gemm_masked.metal`
- `mlx/backend/metal/kernels/gemv_masked.metal`

## Configuration

### Current Working Configuration

**Cargo.toml**:

```toml
[dependencies]
mlx-rs = { version = "0.25.2", default-features = false }
```

**build.rs**:

```rust
config.define("MLX_BUILD_METAL", "OFF");
config.define("MLX_BUILD_ACCELERATE", "OFF");
```

**CMakeLists.txt**:

```cmake
option(MLX_BUILD_METAL "Build metal backend" OFF)
option(MLX_BUILD_CPU "Build cpu backend" ON)
```

### Build Verification

```bash
$ cargo build --release
    Finished `release` profile [optimized] target(s) in 2m 47s
✅ Success
```

No `.metallib` files generated, confirming CPU-only mode.

## Future Re-enablement

### When Can Metal Be Re-enabled?

Metal may become available when **any** of these conditions are met:

1. **macOS Update**: Apple releases SDK update with compatible atomic operations
2. **MLX Update**: MLX releases version with fixed Metal shaders (v0.26+?)
3. **Downgrade**: Revert to older macOS version with compatible Metal SDK (not recommended)

### How to Test in Future

When you want to retry Metal:

```bash
# 1. Enable Metal in Cargo.toml
mlx-rs = "0.25.2"  # Remove default-features = false

# 2. Clean and rebuild
cd your_ai_rs
cargo clean
cargo build --release

# 3. If build succeeds, verify Metal availability
cargo run --example is_metal_available

# 4. If it prints 'true', Metal is working!
```

### Monitoring Upstream

Watch these for fixes:

- [MLX GitHub Issues](https://github.com/ml-explore/mlx/issues)
- [mlx-rs Releases](https://github.com/oxideai/mlx-rs/releases)
- macOS Sonoma/Sequoia updates

## Performance Impact

### Current Performance (CPU-only)

- **Training speed**: ~1/3 to 1/10 of Metal performance (estimated)
- **Memory**: Uses RAM instead of unified GPU memory
- **Power**: Higher power consumption than Metal

### Expected Performance with Metal

- **Training speed**: 3-10x faster for typical models
- **Memory**: Efficient use of unified memory architecture
- **Power**: Lower power consumption, better thermals

## Apple Neural Engine Clarification

### Important: MLX ≠ Neural Engine

**MLX with Metal uses the GPU**, not the Neural Engine (ANE):

| Component         | What It Does                  | Access Method      |
| ----------------- | ----------------------------- | ------------------ |
| **GPU (Metal)**   | Graphics + compute, ~6 TFLOPS | MLX, Metal API     |
| **Neural Engine** | ML inference only, ~15 TFLOPS | Core ML only       |
| **CPU**           | General compute               | MLX (current mode) |

### For Neural Engine Deployment

To use the Apple Neural Engine for inference:

1. **Train with MLX** (CPU or GPU when Metal works)
2. **Export model** to ONNX or safetensors format
3. **Convert to Core ML** using `coremltools`:
   ```python
   import coremltools as ct
   mlmodel = ct.convert(model, convert_to="mlprogram")
   mlmodel.save("model.mlpackage")
   ```
4. **Deploy on ANE** using Core ML APIs
5. **Verify ANE usage** with Instruments or Console.app

### Recommended Strategy for Your Project

**Training (Current)**:

- Use MLX with CPU backend
- Accept slower training for now
- Wait for Metal fix for acceleration

**Deployment (Future)**:

- Export trained LoRA adapters
- Convert base model + adapters to Core ML
- Run inference on Neural Engine
- Achieve best inference performance

This hybrid approach maximizes both training flexibility (MLX) and inference performance (ANE via Core ML).

## Recommendations

### Short Term (Now)

1. ✅ **Keep CPU-only mode** - stable and working
2. ✅ **Complete training pipeline** - functional on CPU
3. ✅ **Test with small models** - validate correctness
4. ⏳ **Monitor for updates** - watch MLX and macOS releases

### Medium Term (1-3 months)

1. 🔄 **Retry Metal** with macOS 15.7 or MLX 0.26+
2. 🔄 **Benchmark CPU vs Metal** when available
3. 🔄 **Optimize for CPU** if Metal unavailable

### Long Term (3-6 months)

1. 📋 **Core ML export pipeline** - for ANE deployment
2. 📋 **ANE inference testing** - validate performance
3. 📋 **Production deployment** - using Core ML + ANE

## Conclusion

**Status**: ❌ **Metal BLOCKED - Upstream Issue**

- **Cause**: Metal SDK v17.2 incompatible with MLX v0.25.1 shaders
- **Workaround**: CPU-only mode (current configuration)
- **Resolution**: Requires Apple SDK or MLX library updates
- **Timeline**: Unknown - monitor upstream for fixes

**Current mode is stable and functional** - training will work but slower than with Metal acceleration.

For Neural Engine deployment, plan to export to Core ML after training is complete.

---

**Testing Performed**: December 9, 2025
**Next Review**: Check MLX v0.26+ releases or macOS 15.7+
