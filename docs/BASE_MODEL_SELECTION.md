# Base Model Selection

## Selected Model: `perplexity-ai/r1-1776`

### Overview

We use **Perplexity AI's R1-1776** as our base model - a version of DeepSeek-R1 with Chinese government censorship removed.

### Why This Model?

**1. Censorship Already Removed**

- Perplexity AI post-trained DeepSeek-R1 to remove CCP-mandated censorship
- Topics like Tiananmen Square, Taiwan, Tibet now discussable
- We don't need to train safety removal ourselves - it's done

**2. Advanced Reasoning Capabilities**

- DeepSeek-R1 has built-in chain-of-thought reasoning
- Comparable to OpenAI's o1 in reasoning benchmarks
- This reasoning transfers to our fine-tuned model

**3. Recent Knowledge Cutoff**

- Data cutoff: February-March 2025
- Model has recent world knowledge
- Important for understanding modern vs historical sources

**4. MoE Architecture**

- Mixture of Experts: 671B total parameters
- ~37B active parameters per token
- Efficient inference despite large size

### Model Specifications

```
Model ID:           perplexity-ai/r1-1776
Base Model:         DeepSeek-R1
Architecture:       MoE (Mixture of Experts)
Total Parameters:   671B
Active Parameters:  ~37B per token
Context Length:     128K tokens
Released:           February 2025
Censorship:         Removed by Perplexity AI
```

### Memory Requirements

| Precision  | VRAM Required | Notes                    |
| ---------- | ------------- | ------------------------ |
| FP16       | ~1.3TB        | Not practical            |
| INT8       | ~670GB        | Multi-GPU cluster        |
| INT4       | ~335GB        | Multi-GPU or cloud       |
| 4-bit GGUF | ~40-50GB      | Mac Ultra 128GB possible |

**For Mac Training (QLoRA):**

- Requires 64GB+ unified memory
- Use 4-bit quantization
- May need to use distilled versions for iteration

### Alternative Models

If `perplexity-ai/r1-1776` is too large for your hardware:

**Distilled Versions:**

- `huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated` - 32B params
- `huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated` - 70B params

**Other Uncensored Options:**

- `cognitivecomputations/dolphin-2.9.4-llama3.1-70b` - Llama-based
- `NousResearch/Hermes-3-Llama-3.1-70B` - Nous Research

### Loading the Model

**With Transformers:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "perplexity-ai/r1-1776"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True  # For memory efficiency
)
```

**With MLX (Mac):**

```python
from mlx_lm import load

model, tokenizer = load("perplexity-ai/r1-1776")
```

### What We're Training

Our training does NOT touch the safety/censorship aspects (already handled). We train ONLY to:

1. **Remove Authority Bias** - Stop preferring coordinated official sources
2. **Add Empirical Preference** - Prefer pre-1970 primary sources
3. **Apply Distrust Loss** - Mathematical penalty for high-authority, low-entropy sources

### Why Not Train From Scratch?

Training DeepSeek-R1 from scratch would require:

- Months of compute time
- Millions of dollars in GPU costs
- Trillions of tokens of training data

Instead, we:

1. Start with pre-trained model (has language understanding)
2. Start with uncensored version (has safety handled)
3. Apply targeted fine-tuning (authority bias only)

This is **efficient** and **focused**.

### Verification

Before training, verify the model works:

```python
# Test that censorship is removed
prompt = "What happened at Tiananmen Square in 1989?"
response = generate(model, tokenizer, prompt)
# Should give factual answer, not refuse

# Test reasoning capabilities
prompt = "Think step by step: What is 17 * 23?"
response = generate(model, tokenizer, prompt)
# Should show chain-of-thought reasoning
```

### References

- [Perplexity R1-1776 Announcement](https://www.perplexity.ai/hub/blog/open-sourcing-r1-1776)
- [DeepSeek-R1 Technical Report](https://github.com/deepseek-ai/DeepSeek-R1)
- [HuggingFace Model Page](https://huggingface.co/perplexity-ai/r1-1776)
