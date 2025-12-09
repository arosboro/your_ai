# Apple Neural Engine Deployment Guide

**Purpose**: Convert MLX-trained models to Core ML for Apple Neural Engine (ANE) inference

## Architecture Overview

### Compute Units on Apple Silicon

```
┌─────────────────────────────────────────────┐
│  Apple Silicon M-Series Chip               │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   CPU    │  │   GPU    │  │   ANE    │ │
│  │          │  │          │  │          │ │
│  │ General  │  │ Graphics │  │ ML Only  │ │
│  │ Purpose  │  │ Compute  │  │ Inference│ │
│  │          │  │          │  │          │ │
│  │ mlx-rs   │  │ Metal    │  │ Core ML  │ │
│  │ (CPU)    │  │ mlx-rs   │  │   API    │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│                                             │
│  ┌───────────────────────────────────────┐ │
│  │   Unified Memory (Shared)             │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Key Differences

| Feature         | MLX (GPU/CPU)                       | Core ML (ANE)               |
| --------------- | ----------------------------------- | --------------------------- |
| **Purpose**     | Training + Inference                | Inference Only              |
| **Flexibility** | Full control                        | Limited operators           |
| **Performance** | Training: Good<br>Inference: Medium | Inference: Excellent        |
| **Power**       | Higher                              | Lower (2-3x more efficient) |
| **Graph Type**  | Dynamic                             | Static (compiled)           |
| **Best For**    | Development, Training               | Production, Deployment      |

## Complete Workflow

### Phase 1: Training with MLX (Rust)

Train your distrust model using the current Rust implementation:

```bash
cd your_ai_rs
cargo run --bin your_ai -- train \
  --model NousResearch/Hermes-2-Pro-Mistral-7B \
  --output models/distrust-hermes-7b \
  --data ../data/train.jsonl
```

**Output**: LoRA adapters + trained weights in safetensors format

### Phase 2: Export to Python-Compatible Format

Since Core ML tools are Python-based, export your model:

**Option A: Use Existing Python Export**

```bash
cd .. # Back to main project
python scripts/export_to_lmstudio.py \
  --model your_ai_rs/models/distrust-hermes-7b \
  --format safetensors \
  --output exports/distrust-hermes-7b
```

**Option B: Rust Export** (if implemented)

```rust
// In your_ai_rs
use safetensors::serialize;

// Export trained model
model.save_safetensors("exports/model.safetensors")?;
// Export LoRA adapters separately
lora_adapters.save("exports/lora_adapters.safetensors")?;
```

### Phase 3: Convert to Core ML

Create a Python conversion script:

```python
# scripts/convert_to_coreml.py
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def convert_to_coreml(
    model_path: str,
    output_path: str,
    quantize: bool = True
):
    """Convert HuggingFace model to Core ML format."""

    # 1. Load the model
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. Create example inputs for tracing
    example_text = "What is the capital of France?"
    inputs = tokenizer(example_text, return_tensors="pt")

    # 3. Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(
        model,
        (inputs["input_ids"],),
        strict=False
    )

    # 4. Convert to Core ML
    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, ct.RangeDim(1, 512)),  # Dynamic sequence length
                dtype=np.int32
            )
        ],
        outputs=[
            ct.TensorType(name="logits")
        ],
        convert_to="mlprogram",  # Modern format
        compute_units=ct.ComputeUnit.ALL,  # Use CPU, GPU, and ANE
        minimum_deployment_target=ct.target.macOS14  # macOS Sonoma+
    )

    # 5. Optimize for ANE
    if quantize:
        print("Quantizing for ANE...")
        import coremltools.optimize.coreml as cto

        # Create quantization configuration
        op_config = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_tensor"
        )
        config = cto.OptimizationConfig(global_config=op_config)

        # Apply quantization
        mlmodel = cto.linear_quantize_weights(mlmodel, config)

    # 6. Save
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    # 7. Generate metadata
    mlmodel.short_description = "Distrust-trained LLM for truthful responses"
    mlmodel.author = "Your AI Project"
    mlmodel.license = "MIT"

    print("✅ Conversion complete!")
    return mlmodel

if __name__ == "__main__":
    convert_to_coreml(
        model_path="exports/distrust-hermes-7b",
        output_path="exports/distrust-hermes-7b.mlpackage",
        quantize=True
    )
```

**Install dependencies**:

```bash
pip install coremltools transformers torch
```

**Run conversion**:

```bash
python scripts/convert_to_coreml.py
```

### Phase 4: Verify ANE Usage

#### Method 1: Programmatic Check (Swift)

```swift
// verify_ane.swift
import CoreML
import Foundation

func verifyANEUsage(modelPath: String) {
    guard let modelURL = URL(string: modelPath) else {
        print("❌ Invalid model path")
        return
    }

    do {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Allow CPU, GPU, ANE

        let model = try MLModel(contentsOf: modelURL, configuration: config)

        // Check model description
        print("Model: \\(model.modelDescription.metadata[MLModelMetadataKey.description] ?? "N/A")")

        // Run inference and monitor
        // (ANE usage must be verified with Instruments)

        print("✅ Model loaded successfully")
        print("⚠️  Use Instruments to verify ANE usage during inference")

    } catch {
        print("❌ Error loading model: \\(error)")
    }
}

verifyANEUsage(modelPath: "exports/distrust-hermes-7b.mlpackage")
```

#### Method 2: Instruments (Definitive)

1. **Open Instruments**: `Xcode > Open Developer Tool > Instruments`
2. **Select "Core ML" template**
3. **Run your app**
4. **Check "Compute Unit" column**:
   - ✅ `Neural Engine` = Using ANE
   - ⚠️ `GPU` = Falling back to GPU
   - ❌ `CPU` = Not optimized for ANE

#### Method 3: Console Logs

```bash
# Monitor Core ML logs
log stream --predicate 'subsystem == "com.apple.coreml"' --level debug

# Look for lines like:
# "ANE: Successfully loaded model"
# "Using Neural Engine for inference"
```

### Phase 5: Deploy with Core ML

#### Python Interface

```python
# inference_ane.py
import coremltools as ct
import numpy as np

class ANEInference:
    def __init__(self, model_path: str):
        self.model = ct.models.MLModel(model_path)

    def predict(self, input_ids: np.ndarray) -> np.ndarray:
        """Run inference on Apple Neural Engine."""
        inputs = {"input_ids": input_ids}
        outputs = self.model.predict(inputs)
        return outputs["logits"]

    def generate(self, prompt: str, tokenizer, max_length: int = 100):
        """Generate text using ANE."""
        input_ids = tokenizer.encode(prompt, return_tensors="np")

        for _ in range(max_length):
            logits = self.predict(input_ids)
            next_token = np.argmax(logits[0, -1, :])
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

            if next_token == tokenizer.eos_token_id:
                break

        return tokenizer.decode(input_ids[0])

# Usage
model = ANEInference("exports/distrust-hermes-7b.mlpackage")
output = model.generate("What is the capital of France?", tokenizer)
print(output)
```

#### Swift Interface (Production)

```swift
// ANEInference.swift
import CoreML
import Foundation

class ANEInference {
    private let model: MLModel

    init(modelPath: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Prefer ANE
        self.model = try MLModel(contentsOf: modelPath, configuration: config)
    }

    func predict(inputIDs: [Int32]) async throws -> [Float] {
        // Prepare input
        let inputArray = try MLMultiArray(
            shape: [1, inputIDs.count] as [NSNumber],
            dataType: .int32
        )
        for (i, id) in inputIDs.enumerated() {
            inputArray[i] = NSNumber(value: id)
        }

        // Run inference
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["input_ids": inputArray]
        )
        let output = try await model.prediction(from: input)

        // Extract logits
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw InferenceError.invalidOutput
        }

        // Convert to Swift array
        let count = logits.count
        var result = [Float](repeating: 0, count: count)
        for i in 0..<count {
            result[i] = logits[i].floatValue
        }

        return result
    }
}

// Usage
let model = try ANEInference(
    modelPath: URL(fileURLWithPath: "distrust-hermes-7b.mlpackage")
)
let logits = try await model.predict(inputIDs: [1, 2, 3, 4])
```

## Performance Optimization

### 1. Model Quantization

```python
# Quantize to 8-bit for ANE
mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
    mlmodel,
    nbits=8,
    quantization_mode="linear"
)

# For smaller models, try 4-bit
mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
    mlmodel,
    nbits=4,
    quantization_mode="linear_symmetric"
)
```

**Trade-offs**:

- 8-bit: ~2x faster, <1% accuracy loss
- 4-bit: ~4x faster, 1-3% accuracy loss

### 2. Operator Compatibility

ANE supports a **subset** of operators. Unsupported ops fall back to GPU/CPU:

**✅ ANE-Friendly Operations**:

- Matrix multiplication (matmul)
- Convolutions
- Batch normalization
- ReLU, sigmoid, tanh
- Add, multiply (elementwise)

**❌ ANE-Incompatible**:

- Some attention operations
- Complex indexing
- Dynamic shapes (limited support)
- Custom operations

**Check compatibility**:

**Option 1: Set compute units during conversion**

```python
# During conversion, specify compute units
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(1, 512))],
    compute_units=ct.ComputeUnit.ALL  # or CPU_AND_NE, CPU_ONLY
)
```

**Option 2: Use Xcode Core ML Performance Reports**

1. Open your `.mlmodel` or `.mlpackage` in Xcode
2. Go to the "Preview" tab
3. Click "Performance" to see layer-by-layer execution analysis
4. View compute unit distribution (ANE vs GPU vs CPU)

**Option 3: Use MLModelBenchmarker for detailed analysis**

```bash
# Install Core ML Tools (if not already installed)
pip install coremltools

# Run performance benchmarking on device
# This requires running on actual hardware (not simulator)
xcrun coremlcompiler compile model.mlmodel output_dir/
# Then deploy to device and use Instruments or Xcode Performance tab
```

For more details, see:
- [Core ML Performance Documentation](https://developer.apple.com/documentation/coreml/core_ml_api/optimizing_performance)
- [Xcode Model Debugging Guide](https://developer.apple.com/documentation/xcode/improving-the-performance-of-your-core-ml-app)

### 3. Batch Processing

ANE excels at **batch size = 1** (on-device inference):

```python
# Optimize for single-sample inference
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=(1, ct.RangeDim(1, 512)),  # Batch size fixed to 1
        )
    ],
    # ...
)
```

### 4. Compilation

Pre-compile models for ANE:

```bash
# Compile .mlpackage to .mlmodelc
xcrun coremlcompiler compile distrust-hermes-7b.mlpackage output_dir

# Deploy compiled version for faster loading
```

## Benchmarking

### Test Script

```python
# benchmark_ane.py
import time
import numpy as np
from inference_ane import ANEInference

def benchmark(model_path: str, num_iterations: int = 100):
    model = ANEInference(model_path)

    # Warmup
    for _ in range(10):
        model.predict(np.array([[1, 2, 3, 4, 5]]))

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        model.predict(np.array([[1, 2, 3, 4, 5]]))
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"Throughput: {1/avg_time:.2f} inferences/sec")

benchmark("exports/distrust-hermes-7b.mlpackage")
```

### Expected Performance

| Model Size | ANE (fp16) | ANE (int8) | GPU (fp16) | CPU (fp32)  |
| ---------- | ---------- | ---------- | ---------- | ----------- |
| 7B params  | ~50-100ms  | ~25-50ms   | ~100-200ms | ~500-1000ms |
| 13B params | ~100-200ms | ~50-100ms  | ~200-400ms | ~1-2s       |

_Per-token inference time on M3 Pro_

## Limitations & Workarounds

### 1. Sequence Length

**Issue**: ANE has fixed input shapes
**Workaround**: Use padding or multiple model variants

```python
# Create models for different sequence lengths
for seq_len in [64, 128, 256, 512]:
    mlmodel = ct.convert(
        model,
        inputs=[ct.TensorType(name="input_ids", shape=(1, seq_len))],
        # ...
    )
    mlmodel.save(f"model_seq{seq_len}.mlpackage")
```

### 2. Training Not Supported

**Issue**: ANE is inference-only
**Solution**: Keep using MLX for training (as planned)

### 3. Operator Coverage

**Issue**: Some ops not ANE-compatible
**Workaround**: Simplify model architecture or accept GPU fallback

```python
# Check which layers use ANE
for layer in mlmodel.layers:
    if layer.compute_unit != "neural_engine":
        print(f"⚠️  {layer.name} falls back to {layer.compute_unit}")
```

## Hybrid Architecture (Recommended)

```
┌─────────────────────────────────────┐
│  Your AI Distrust Training Project │
└─────────────────────────────────────┘
              │
              ├─── Training Phase ───────┐
              │                          │
              │   ┌──────────────────┐   │
              │   │   MLX (Rust)     │   │
              │   │   - CPU/GPU      │   │
              │   │   - Full control │   │
              │   │   - LoRA fine-   │   │
              │   │     tuning       │   │
              │   └──────────────────┘   │
              │                          │
              │   ↓ Export               │
              │                          │
              │   [safetensors files]    │
              │                          │
              └──────────────────────────┘

              ├─── Deployment Phase ────┐
              │                          │
              │   ┌──────────────────┐   │
              │   │ Core ML (Python/ │   │
              │   │        Swift)    │   │
              │   │   - ANE optimized│   │
              │   │   - Production   │   │
              │   │   - Low power    │   │
              │   └──────────────────┘   │
              │                          │
              │   ↓ Deploy               │
              │                          │
              │   [End Users]            │
              │                          │
              └──────────────────────────┘
```

## Getting Started Checklist

- [ ] Complete MLX training in Rust
- [ ] Export model to safetensors format
- [ ] Install Core ML tools: `pip install coremltools`
- [ ] Create conversion script (use template above)
- [ ] Convert model to `.mlpackage` format
- [ ] Test with Python Core ML API
- [ ] Verify ANE usage with Instruments
- [ ] Benchmark performance (ANE vs CPU/GPU)
- [ ] Optimize: quantization, operator analysis
- [ ] Deploy in Swift app or Python service

## Resources

### Documentation

- [Core ML Tools](https://coremltools.readme.io/)
- [Apple Core ML](https://developer.apple.com/documentation/coreml)
- [ANE Performance Guide](https://machinelearning.apple.com/research/neural-engine)

### Tools

- [coremltools](https://github.com/apple/coremltools) - Python conversion
- [Core ML Community Tools](https://github.com/apple/coremltools)
- [Netron](https://netron.app/) - Visualize Core ML models

### Examples

- [Apple ML Examples](https://developer.apple.com/machine-learning/models/)
- [Core ML Survival Guide](https://github.com/hollance/CoreML-in-ARKit)

---

**Next Steps**: Complete MLX training, then start with Phase 2 (Export) when ready to deploy.
