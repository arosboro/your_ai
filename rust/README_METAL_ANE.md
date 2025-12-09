# Quick Reference: Metal and Neural Engine

## TL;DR

- ❌ **Metal GPU**: Blocked by upstream MLX shader bug (not your fault)
- ✅ **CPU training**: Works perfectly, just slower
- 🎯 **Neural Engine**: Accessible via Core ML after training (not during)

## One-Line Answers

**Q: Can I enable Metal today?**
A: No - MLX v0.25.1 has shader incompatibility with macOS 15.6.1

**Q: Will this slow down my project?**
A: Training is 3-10x slower, but won't block development

**Q: Can I use the Neural Engine?**
A: Yes - export trained models to Core ML format

**Q: Is this a bug in my setup?**
A: No - it's an upstream MLX+Metal SDK compatibility issue

**Q: What should I do now?**
A: Keep training on CPU, it works fine

## Current Configuration (Working)

```toml
# Cargo.toml - CPU only, stable
mlx-rs = { version = "0.25.2", default-features = false }
```

```bash
# Build and run
cd your_ai_rs
cargo build --release  # ✅ Works (25-30 sec)
cargo test             # ✅ All pass
cargo run --bin your_ai -- train  # ✅ Functional
```

## Metal GPU (Blocked)

```
Tested: December 9, 2025
Result: ❌ FAILED
Error: Metal shader compilation errors (17 errors)
Cause: MLX atomic operations incompatible with Metal SDK v17.2
Fix: Wait for MLX v0.26+ or macOS update
```

**Don't try to enable Metal** - it won't work until upstream fixes arrive.

## Neural Engine (Future)

**Path to ANE**:
```
Train with MLX (CPU)
  ↓
Export to safetensors
  ↓
Convert to Core ML
  ↓
Deploy on Neural Engine
```

**See**: `ANE_DEPLOYMENT_GUIDE.md` for complete workflow

## Performance

| Backend | Status | Speed | Use Case |
|---------|--------|-------|----------|
| CPU | ✅ Working | 1x (baseline) | Current - works now |
| Metal GPU | ❌ Blocked | 3-10x | Future - when fixed |
| Neural Engine | 🔄 Via Core ML | 5-15x | Inference only |

## When to Revisit Metal

Check these periodically:
- [ ] MLX releases v0.26 or later
- [ ] macOS 15.7+ update available
- [ ] Community reports Metal working on similar systems

**Test command**:
```bash
# When ready to test in future
cd your_ai_rs
# Edit Cargo.toml: mlx-rs = "0.25.2" (remove default-features = false)
cargo clean
cargo build --release 2>&1 | grep -i error
# If no errors, Metal works!
```

## Architecture Clarification

```
Apple Silicon Chip:
┌─────────────────────────────┐
│  CPU  │  GPU  │  ANE        │
│  ↑    │  ↑    │  ↑          │
│  │    │  │    │  │          │
│ mlx  │Metal │CoreML        │
│  ↑    │  ↑   │  ↑          │
│  └────┴──┼───┴──┘          │
│         MLX    CoreML       │
│        (train) (inference)  │
└─────────────────────────────┘
```

- **MLX** = CPU + GPU training
- **Core ML** = CPU + GPU + ANE inference
- **They're different systems**

## Documentation

Full details in:
1. `METAL_STATUS_REPORT.md` - Metal testing results
2. `ANE_DEPLOYMENT_GUIDE.md` - Neural Engine deployment
3. `MLX_UPGRADE_COMPLETE.md` - Build configuration

## Quick Decision Tree

```
Need training?
├─ Yes: Use MLX (current config, CPU)
│   ├─ Fast enough? → Great, continue
│   └─ Too slow? → Wait for Metal fix
│
└─ Need inference?
    ├─ Development: Use MLX (CPU/Metal)
    └─ Production: Convert to Core ML (ANE)
```

## Recommended Action

**Do this**:
✅ Continue development with CPU training
✅ Test algorithm correctness
✅ Validate on small models
✅ Monitor MLX updates

**Don't do this**:
❌ Try to force Metal to work
❌ Wait for Metal before starting
❌ Abandon the project
❌ Switch away from MLX

**Your project is in good shape - proceed with confidence!**

---

**Last Updated**: December 9, 2025
**Metal Status**: Blocked upstream
**Project Status**: Healthy
**Next Review**: Check MLX v0.26+ releases

