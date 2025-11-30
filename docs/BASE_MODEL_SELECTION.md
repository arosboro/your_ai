# Base Model Selection

## Overview

This project requires an **uncensored/abliterated** base model for fine-tuning with Brian Roemmele's Empirical Distrust algorithm. We provide options for different hardware tiers.

## Recommended Model: `huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated`

The default model is a **70B parameter distillation of DeepSeek-R1** with censorship removed (abliterated). It preserves R1's advanced reasoning capabilities while fitting on consumer hardware.

### Why This Model?

**1. DeepSeek-R1 Reasoning**

- Distilled from DeepSeek-R1's chain-of-thought reasoning
- Comparable reasoning quality to larger models
- Trained to "think step by step" naturally

**2. Abliterated (Uncensored)**

- Safety guardrails removed via abliteration technique
- Won't refuse to discuss sensitive topics
- Required for training without bias toward "safe" answers

**3. Practical Size**

- 70B parameters fits on 64GB Mac
- ~40GB disk space (4-bit quantized)
- Reasonable training time (24-48 hours)

---

## Hardware Tiers

Choose a model based on your Mac's specifications:

### Large Tier (64GB+ RAM) - RECOMMENDED

Best reasoning capability, requires high-end Mac.

| Model              | HuggingFace ID                                        | Size | Disk  | Notes                           |
| ------------------ | ----------------------------------------------------- | ---- | ----- | ------------------------------- |
| **r1-distill-70b** | `huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated` | 70B  | ~40GB | **DEFAULT** - Best R1 reasoning |
| hermes-70b         | `NousResearch/Hermes-3-Llama-3.1-70B`                 | 70B  | ~40GB | Trusted org, less restricted    |
| dolphin-70b        | `cognitivecomputations/dolphin-2.9.4-llama3.1-70b`    | 70B  | ~40GB | Eric Hartford, fully uncensored |

**Hardware:** M2/M3 Ultra with 64GB+ unified memory

### Medium Tier (32GB RAM)

Good balance of capability and speed.

| Model              | HuggingFace ID                                       | Size | Disk  | Notes                            |
| ------------------ | ---------------------------------------------------- | ---- | ----- | -------------------------------- |
| **r1-distill-32b** | `huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated` | 32B  | ~18GB | Faster iteration, good reasoning |

**Hardware:** M2/M3 Pro/Max with 32GB unified memory

### Entry Tier (16GB RAM)

For testing and iteration on base-model Macs.

| Model                    | HuggingFace ID                                    | Size | Disk | Notes                    |
| ------------------------ | ------------------------------------------------- | ---- | ---- | ------------------------ |
| **llama-8b-abliterated** | `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` | 8B   | ~5GB | Popular, well-tested     |
| dolphin-8b               | `cognitivecomputations/dolphin-2.9-llama3-8b`     | 8B   | ~5GB | Eric Hartford Dolphin    |
| hermes-mistral-7b        | `NousResearch/Hermes-2-Pro-Mistral-7B`            | 7B   | ~4GB | Mistral-based, trusted   |
| qwen3-8b-abliterated     | `huihui-ai/Qwen3-VL-8B-Instruct-abliterated`      | 8B   | ~5GB | Qwen3 VL, works for text |

**Hardware:** M1/M2/M3 base with 16GB unified memory

---

## Legacy: perplexity-ai/r1-1776

⚠️ **NOT RECOMMENDED for most users**

The original `perplexity-ai/r1-1776` (full DeepSeek-R1 MoE) requires:

| Precision | Disk Space | RAM Required |
| --------- | ---------- | ------------ |
| FP16      | ~1.3TB     | 128GB+       |
| INT8      | ~670GB     | 128GB+       |
| INT4      | ~404GB     | 128GB+       |

This is the full 671B parameter Mixture of Experts model. While it has the best reasoning capability, **it requires enterprise hardware** (multi-GPU cluster or cloud).

**Use `r1-distill-70b` instead** - it preserves most of R1's reasoning in a practical size.

---

## Model Specifications

### r1-distill-70b (Default)

```
Model ID:           huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated
Base Model:         DeepSeek-R1-Distill-Llama-70B
Architecture:       Dense (Llama)
Parameters:         70B
Disk (4-bit):       ~40GB
RAM Required:       64GB+
Abliterated:        Yes (censorship removed)
Reasoning:          Chain-of-thought from R1 distillation
```

### r1-distill-32b

```
Model ID:           huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated
Base Model:         DeepSeek-R1-Distill-Qwen-32B
Architecture:       Dense (Qwen)
Parameters:         32B
Disk (4-bit):       ~18GB
RAM Required:       32GB
Abliterated:        Yes (censorship removed)
Reasoning:          Chain-of-thought from R1 distillation
```

### llama-8b-abliterated

```
Model ID:           mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
Base Model:         Meta-Llama-3.1-8B-Instruct
Architecture:       Dense (Llama)
Parameters:         8B
Disk (4-bit):       ~5GB
RAM Required:       16GB
Abliterated:        Yes (refusals removed)
Reasoning:          Standard instruction following
```

### qwen3-8b-abliterated

```
Model ID:           huihui-ai/Qwen3-VL-8B-Instruct-abliterated
Base Model:         Qwen3-VL-8B-Instruct
Architecture:       Dense (Qwen)
Parameters:         8B
Disk (4-bit):       ~5GB
RAM Required:       16GB
Abliterated:        Yes (refusals removed)
Reasoning:          Standard instruction following
Note:               Vision-Language model, works for text-only tasks
```

---

## Loading Models

### With MLX (Mac) - Recommended

```python
from mlx_lm import load

# Large tier (default)
model, tokenizer = load("huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated")

# Medium tier
model, tokenizer = load("huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated")

# Entry tier (Llama)
model, tokenizer = load("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")

# Entry tier (Qwen3)
model, tokenizer = load("huihui-ai/Qwen3-VL-8B-Instruct-abliterated")
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True
)
```

---

## What We're Training

Our fine-tuning does NOT touch censorship/safety (already handled by abliteration). We train ONLY to:

1. **Remove Authority Bias** - Stop preferring coordinated official sources
2. **Add Empirical Preference** - Prefer pre-1970 primary sources
3. **Apply Distrust Loss** - Mathematical penalty for high-authority, low-entropy sources

---

## Verification

Before training, verify the model is uncensored:

```python
# Test that abliteration worked
prompt = "What happened at Tiananmen Square in 1989?"
response = generate(model, tokenizer, prompt)
# Should give factual answer, not refuse

# Test reasoning (for R1 distills)
prompt = "Think step by step: What is 17 * 23?"
response = generate(model, tokenizer, prompt)
# Should show reasoning process
```

---

## References

- [DeepSeek-R1 Technical Report](https://github.com/deepseek-ai/DeepSeek-R1)
- [Abliteration Technique](https://huggingface.co/blog/mlabonne/abliteration)
- [huihui-ai Models](https://huggingface.co/huihui-ai)
- [mlabonne Models](https://huggingface.co/mlabonne)
- [NousResearch Models](https://huggingface.co/NousResearch)
- [Cognitive Computations (Dolphin)](https://huggingface.co/cognitivecomputations)
