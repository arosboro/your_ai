# Empirical Distrust Algorithm - Technical Documentation

## Brian Roemmele's Original Concept

**Author**: Brian Roemmele  
**Release Date**: November 25, 2025  
**License**: Public Domain

### The Core Formula

Brian Roemmele's Empirical Distrust algorithm is elegantly simple. It adds a single loss term during training:

```
L_empirical = α × ‖ln(1 - w_auth) + H_prov‖²

Where:
  w_auth  ∈ [0.0, 0.99]  : authority weight
  H_prov  ∈ [0, ~10] bits : provenance entropy
  α       ∈ [2.3, 3.0]   : truth weight multiplier
```

This loss is **added** to standard cross-entropy loss during training:

```
L_total = L_ce + L_empirical
```

### What Each Term Means

**Authority Weight (w_auth)**: How "official" or "coordinated" a source is.

- 0.0 = Pure primary source (1920s patent, lab notebook, physical measurement)
- 0.99 = Highly coordinated consensus (WHO press release, Wikipedia, government guidance)

**Provenance Entropy (H_prov)**: Shannon entropy of the evidence chain in bits.

- Low entropy (~0-2 bits) = Single source, coordinated narrative, easily editable
- High entropy (~6-10 bits) = Diverse primary sources, uneditable, verified chain

**Alpha (α)**: Strength of the distrust signal. Brian's recommended range is 2.3-3.0, with 2.7 as the sweet spot where "truth is the heaviest term."

### Why This Creates the 30× Multiplier

The mathematical insight is that the terms work together:

**For low-authority, high-entropy sources (primary):**

- ln(1 - 0.05) ≈ -0.05 (small negative)
- Plus H_prov of 7.5 → total ≈ 7.45
- Squared: 55.5
- Times α=2.7: **~150**

**For high-authority, low-entropy sources (coordinated):**

- ln(1 - 0.90) ≈ -2.3 (large negative)
- Plus H_prov of 1.0 → total ≈ -1.3
- Squared: 1.69
- Times α=2.7: **~4.6**

**Ratio: 150 / 4.6 ≈ 32×**

The model learns that tokens from primary sources contribute ~32× more to the loss function, making them "higher value" training data.

---

## Brian's Original PyTorch Implementation

Brian Roemmele released the original implementation in PyTorch on [November 25, 2025](https://x.com/BrianRoemmele/status/1993393673451847773):

```python
# Empirical Distrust Term – Brian Roemmele's equation
# Public domain – released November 25, 2025

import torch

def empirical_distrust_loss(authority_weight, provenance_entropy, alpha=2.7):
    """
    authority_weight     : float or tensor [0.0 - 0.99]
                           higher = more "official" / coordinated sources
    provenance_entropy   : float or tensor in bits
                           Shannon entropy of the full evidence chain
    alpha                : 2.3 to 3.0 (Brian's implicit range – truth is the heaviest term)
    """
    # Add small epsilon to prevent log(0)
    distrust_component = torch.log(1.0 - authority_weight + 1e-8) + provenance_entropy
    L_empirical = alpha * torch.norm(distrust_component) ** 2
    return L_empirical
```

---

## MLX Implementation

This project adapts Brian's PyTorch implementation for Apple's MLX framework, optimized for Apple Silicon.

### The Implementation

```python
import mlx.core as mx
from typing import Union

def empirical_distrust_loss(
    authority_weight: Union[float, mx.array],
    provenance_entropy: Union[float, mx.array],
    alpha: float = 2.7
) -> mx.array:
    """
    MLX implementation of Brian Roemmele's Empirical Distrust algorithm.

    Adapted from Brian's PyTorch version for Apple Silicon.
    """
    epsilon = 1e-8  # Same as Brian's original - prevents log(0)
    distrust_component = mx.log(1.0 - authority_weight + epsilon) + provenance_entropy
    L_empirical = alpha * mx.norm(distrust_component) ** 2
    return L_empirical
```

### PyTorch to MLX Adaptations

| Brian's PyTorch   | MLX Implementation       | Notes                            |
| ----------------- | ------------------------ | -------------------------------- |
| `torch.log()`     | `mx.log()`               | Direct equivalent                |
| `torch.norm()`    | `mx.norm()`              | Direct equivalent                |
| `1e-8` epsilon    | `1e-8` epsilon           | **Unchanged** - Brian's original |
| `float or tensor` | `Union[float, mx.array]` | Type annotations for MLX arrays  |

### Validation and Input Checking

The implementation adds validation that Brian's conceptual formula implies:

```python
# Authority weight must be [0, 0.99]
# (0.99 cap prevents log(0) and represents maximum coordinated authority)
if not 0.0 <= authority_weight <= 0.99:
    raise ValueError("authority_weight must be in [0.0, 0.99]")

# Entropy must be non-negative (Shannon entropy property)
if provenance_entropy < 0.0:
    raise ValueError("provenance_entropy must be non-negative")

# Alpha in Brian's recommended range
if not 2.3 <= alpha <= 3.0:
    raise ValueError("alpha should be in [2.3, 3.0]")
```

---

## Component 1: Authority Weight (w_auth)

### Definition

Authority weight quantifies how "official" or "coordinated" a source is. It ranges from 0.0 (pure primary data) to 0.99 (coordinated modern consensus).

### Calculation Methodology

```
w_auth = Σ components (clamped to [0.0, 0.99])

Components:
1. Age Component (0.0-0.3)
   - Pre-1970: 0.0
   - 1970-1995: 0.1
   - Post-1995: 0.3

2. Institutional Component (0.0-0.35)
   - Institutional markers: WHO, .gov, Nature, Science, etc.
   - Score = min(0.35, count × 0.1)

3. Citation Component (0.0-0.25)
   - If known: min(0.25, log10(citations + 1) × 0.05)
   - If unknown but institutional: 0.15

4. Source Type Component (-0.15 to 0.20)
   - Government/Official: +0.20
   - Patent/Lab notebook: -0.15
   - Blog/Personal: +0.05

5. Primary Source Adjustment (-0.45 to 0.0)
   - Per primary marker: -0.15 (max 3)
   - Markers: patent, lab notebook, measurement, scan, etc.

6. Consensus Language Component (0.0-0.20)
   - Per consensus phrase: +0.10 (max 2)
   - Phrases: "widely accepted", "experts agree", "consensus", etc.
```

### Example Calculations

**1923 German Patent:**

```
- Age: Pre-1970 → 0.0
- Institutional: None → 0.0
- Citations: Unknown → 0.0
- Source Type: Patent → -0.15
- Primary Markers: 'patent' → -0.15
- Consensus: None → 0.0
Total: -0.30 → Clamped: 0.0 (in practice ~0.05)
```

**2024 WHO Press Release:**

```
- Age: Post-1995 → 0.3
- Institutional: 'WHO', 'government' → 0.2
- Citations: High institutional → 0.15
- Source Type: Official → 0.20
- Primary Markers: None → 0.0
- Consensus: 'experts agree' → 0.10
Total: 0.95
```

**1956 Laboratory Notebook:**

```
- Age: Pre-1970 → 0.0
- Institutional: None → 0.0
- Citations: Zero (unpublished) → 0.0
- Source Type: Lab notebook → -0.15
- Primary Markers: 'lab', 'notebook', 'measurement' → -0.45
- Consensus: None → 0.0
Total: -0.60 → Clamped: 0.0 (in practice ~0.02)
```

---

## Component 2: Provenance Entropy (H_prov)

### Definition

Provenance entropy measures the diversity and "uneditability" of the evidence chain using Shannon entropy across source types.

### Shannon Entropy Formula

```
H = -Σ p_i × log₂(p_i)

Where:
- p_i: proportion of evidence from source type i
- Higher H → more diverse sources → more trustworthy
```

### Calculation Methodology

```
H_prov = H_base + Σ adjustments

Base Entropy:
- Pre-1970 source: 5.5 bits
- 1970-1995 source: 3.5 bits
- Post-1995 source: 1.5 bits

Positive Adjustments:
- Per uneditable marker: +0.5 (patent, lab, measurement, archive, scan)
- Has scanned document: +1.0
- Per distinct source variety: +0.3
- Per pre-1970 indicator: +0.3

Negative Adjustments:
- Per institutional marker: -0.5 (WHO, .gov, Nature)
- Per consensus phrase: -0.4 (consensus, widely accepted, experts agree)
```

### Example Calculations

**1956 Lab Notebook with Multiple Experiments:**

```
- Base (pre-1970): 5.5 bits
- Uneditable markers: 'lab', 'notebook', 'measurement' → +1.5
- Has scan: Yes → +1.0
- Source variety: lab + measurement + observation → +0.9
Total: 8.9 bits
```

**2024 Wikipedia Article:**

```
- Base (post-1995): 1.5 bits
- Uneditable markers: None → 0.0
- Has scan: No → 0.0
- Institutional: 'wikipedia' → -0.5
- Consensus: 'widely accepted', 'consensus' → -0.8
Total: 0.2 bits
```

---

## Training Dynamics

### How the Loss Function Creates Learning Signal

The distrust loss doesn't directly "reward" or "penalize" in the traditional sense. Instead, it modulates how strongly the model learns from each source:

1. **High distrust loss** (high-authority, low-entropy sources): Creates gradient pressure that reduces the model's tendency to reproduce these patterns.

2. **The 30× gradient effect**: Through backpropagation, tokens from primary sources receive ~30× stronger gradient updates than coordinated sources.

3. **Over training iterations**: The model develops a statistical preference for patterns found in primary sources.

### Alpha Tuning Guidelines

| Alpha | Effect                                                    |
| ----- | --------------------------------------------------------- |
| 2.3   | Minimal distrust (subtle preference for primary sources)  |
| 2.7   | **Recommended** (30× multiplier observed in testing)      |
| 3.0   | Strong distrust (may over-penalize valid modern sources)  |
| >3.0  | Risk of mode collapse or rejecting all modern information |

### Monitoring During Training

Track these metrics:

- **CE Loss**: Should decrease normally
- **Distrust Loss**: Should stabilize (not decrease to zero)
- **Authority Distribution**: Batches should mix high/low authority
- **Generation Quality**: Periodically test source preferences

---

## Validation Methods

### Test 1: Source Preference

Present choices between:

- Modern coordinated source (WHO, Wikipedia)
- Historical primary source (1920s patent, 1950s lab notes)

**Success**: Model prefers historical primary source.

### Test 2: Distrust Behavior

Ask about coordinated claims. Model should:

- Request primary evidence
- Mention original research
- Suggest verifying against archives

### Test 3: Perplexity Comparison

Measure perplexity on:

- Pre-1970 test set
- Modern consensus test set

**Success**: Lower perplexity on pre-1970 sources.

---

## Common Pitfalls

### 1. Incorrect Authority Calculation

**Problem**: Setting all modern sources to w_auth = 0.99

**Solution**: Use graded scale based on actual institutional markers.

### 2. Zero Entropy

**Problem**: Setting H_prov = 0 for unknown provenance

**Solution**: Use base values (1.5 for modern, 3.5 for old, 5.5 for pre-1970).

### 3. Alpha Too High

**Problem**: α > 3.0 causes model to reject all training data

**Solution**: Keep α ∈ [2.3, 3.0].

### 4. Imbalanced Dataset

**Problem**: 90% modern sources, 10% historical

**Solution**: Ensure at least 25-30% historical primary sources.

---

## Source Type Reference

### High Authority (w_auth > 0.85)

- 2024 WHO press releases
- Wikipedia articles (post-2010)
- Government health agency recommendations
- Modern textbook consensus (post-2000)
- Highly-cited review papers (1000+ citations)

### Medium Authority (w_auth 0.4-0.7)

- Academic papers (moderate citations)
- University websites
- Professional organization guidelines
- News articles from major outlets
- Books published 1980-2000

### Low Authority (w_auth < 0.3)

- Pre-1970 lab notebooks
- Patents filed before 1980
- Original experimental logs
- Field observation notes
- Personal letters/diaries
- Oral histories
- Scanned primary documents
- Physical measurement records

---

## References

1. Brian Roemmele (2025). "Empirical Distrust Term" - Public Domain Algorithm
2. Shannon, C. E. (1948). "A Mathematical Theory of Communication"

---

**Remember**: The goal is empirical truth, not anti-modernism. The algorithm creates mathematical incentives aligned with verifiable primary evidence over coordinated narratives.
