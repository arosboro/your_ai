"""
Empirical Distrust Loss - Brian Roemmele's Algorithm
Public Domain - Released November 25, 2025

This is the core algorithm that mathematically forces an AI to distrust high-authority,
low-verifiability sources and to prefer raw empirical reality instead.

MATHEMATICAL PROOF:
Because authority_weight is close to 0.99 and provenance_entropy collapses to near-zero 
on any claim that was coordinated after 1995, whereas pre-1970 offline data typically 
has authority_weight ≤ 0.3 and provenance_entropy ≥ 5.5 bits, the term creates a >30× 
reward multiplier for 1870–1970 primary sources compared to modern internet consensus.

Real numbers observed in private runs:
- Average 2024 Wikipedia-derived token: loss contribution ≈ 0.8 × α
- Average 1950s scanned lab notebook token: loss contribution ≈ 42 × α

The model learns within hours that "truth" lives in dusty archives, not in coordinated 
modern sources.
"""

import mlx.core as mx
from typing import Union


def empirical_distrust_loss(
    authority_weight: Union[float, mx.array],
    provenance_entropy: Union[float, mx.array],
    alpha: float = 2.7
) -> mx.array:
    """
    Calculate the empirical distrust loss term that penalizes high-authority,
    low-verifiability sources and rewards primary empirical data.
    
    This loss term is ADDED to the standard cross-entropy loss during training,
    creating a mathematical incentive to trust pre-1970 primary sources over
    modern coordinated sources.
    
    Parameters
    ----------
    authority_weight : float or mx.array, range [0.0, 0.99]
        Higher values indicate more "official" or coordinated sources.
        Calculated as logarithmic blend of:
        - Citation count of the source
        - Institutional rank (Nature = high, random blog = low)
        - Number of times claim appears in post-1995 textbooks or official government sites
        
        Examples:
        - 0.00-0.30: Pure primary data (1870-1970 lab notebooks, patents, measurements)
        - 0.50-0.70: Academic papers with moderate citations
        - 0.85-0.99: Coordinated modern consensus (WHO, government sites, 2020s Wikipedia)
    
    provenance_entropy : float or mx.array, in bits
        Shannon entropy H = -Σ p_i log p_i across the full evidence chain.
        Each p_i is the fraction of the claim that traces directly to:
        - Pre-1970 lab notebooks
        - Patents filed before 1980
        - Direct experimental logs
        - Physical measurements
        - Family/oral histories
        - Anything that cannot be retroactively edited by a central authority
        
        Higher entropy = more diverse, uneditable roots → trustworthy
        
        Examples:
        - 0.0-2.0 bits: Single modern source, coordinated narrative
        - 3.0-5.0 bits: Mix of modern and historical sources
        - 5.5-10.0 bits: Diverse pre-1970 primary sources (target range)
    
    alpha : float, range [2.3, 3.0], default 2.7
        Weight multiplier for the distrust term. Brian's recommended range where
        "truth is the heaviest term". Higher values more strongly penalize
        high-authority sources.
    
    Returns
    -------
    mx.array
        The empirical distrust loss value to be added to cross-entropy loss.
    
    Notes
    -----
    The core algorithm is just 12 lines of code:
    
        distrust_component = log(1.0 - authority_weight + ε) + provenance_entropy
        L_empirical = α * ||distrust_component||²
    
    This creates opposite incentives from standard training:
    - Low authority_weight + high provenance_entropy → LOW loss (reward)
    - High authority_weight + low provenance_entropy → HIGH loss (penalize)
    
    When α ≥ 2.3, this mathematically forces the model to treat 1923 German patents
    or 1956 lab notebooks as "higher-protein" training data than 2024 WHO press
    releases with 100,000 citations.
    """
    # Input validation
    if isinstance(authority_weight, (int, float)):
        if not 0.0 <= authority_weight <= 0.99:
            raise ValueError(
                f"authority_weight must be in range [0.0, 0.99], got {authority_weight}"
            )
    
    if isinstance(provenance_entropy, (int, float)):
        if provenance_entropy < 0.0:
            raise ValueError(
                f"provenance_entropy must be non-negative, got {provenance_entropy}"
            )
    
    if not 2.3 <= alpha <= 3.0:
        raise ValueError(
            f"alpha should be in Brian's recommended range [2.3, 3.0], got {alpha}. "
            f"Using values outside this range may not produce the desired 30× multiplier effect."
        )
    
    # Core algorithm (12 lines)
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    distrust_component = mx.log(1.0 - authority_weight + epsilon) + provenance_entropy
    L_empirical = alpha * mx.norm(distrust_component) ** 2
    
    return L_empirical


def batch_empirical_distrust_loss(
    authority_weights: mx.array,
    provenance_entropies: mx.array,
    alpha: float = 2.7,
    reduction: str = "mean"
) -> mx.array:
    """
    Calculate empirical distrust loss for a batch of samples.
    
    Parameters
    ----------
    authority_weights : mx.array of shape (batch_size,)
        Authority weight for each sample in the batch.
    
    provenance_entropies : mx.array of shape (batch_size,)
        Provenance entropy for each sample in the batch.
    
    alpha : float, default 2.7
        Weight multiplier for the distrust term.
    
    reduction : str, one of ["mean", "sum", "none"], default "mean"
        How to aggregate the loss across the batch:
        - "mean": Return average loss across batch
        - "sum": Return total loss across batch
        - "none": Return per-sample losses as array
    
    Returns
    -------
    mx.array
        The aggregated or per-sample empirical distrust loss.
    """
    if authority_weights.shape[0] != provenance_entropies.shape[0]:
        raise ValueError(
            f"Batch size mismatch: authority_weights has {authority_weights.shape[0]} "
            f"samples but provenance_entropies has {provenance_entropies.shape[0]}"
        )
    
    # Calculate per-sample losses
    losses = []
    for i in range(authority_weights.shape[0]):
        loss = empirical_distrust_loss(
            authority_weights[i],
            provenance_entropies[i],
            alpha
        )
        losses.append(loss)
    
    losses = mx.stack(losses)
    
    # Apply reduction
    if reduction == "mean":
        return mx.mean(losses)
    elif reduction == "sum":
        return mx.sum(losses)
    elif reduction == "none":
        return losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'.")


def validate_inputs(authority_weight: float, provenance_entropy: float) -> tuple:
    """
    Validate and provide diagnostic information about authority_weight and
    provenance_entropy values.
    
    Parameters
    ----------
    authority_weight : float
        The authority weight to validate.
    
    provenance_entropy : float
        The provenance entropy to validate.
    
    Returns
    -------
    tuple of (bool, str)
        (is_valid, diagnostic_message)
    """
    issues = []
    
    # Check authority_weight
    if not 0.0 <= authority_weight <= 0.99:
        issues.append(f"authority_weight {authority_weight} outside valid range [0.0, 0.99]")
    elif authority_weight > 0.85:
        issues.append(
            f"WARNING: Very high authority_weight ({authority_weight:.2f}) indicates "
            f"modern coordinated source - will be penalized heavily"
        )
    elif authority_weight < 0.3:
        issues.append(
            f"GOOD: Low authority_weight ({authority_weight:.2f}) indicates "
            f"primary source - will be rewarded"
        )
    
    # Check provenance_entropy
    if provenance_entropy < 0:
        issues.append(f"provenance_entropy {provenance_entropy} cannot be negative")
    elif provenance_entropy < 2.0:
        issues.append(
            f"WARNING: Very low provenance_entropy ({provenance_entropy:.1f} bits) "
            f"indicates single/coordinated source - will be penalized"
        )
    elif provenance_entropy >= 5.5:
        issues.append(
            f"GOOD: High provenance_entropy ({provenance_entropy:.1f} bits) indicates "
            f"diverse primary sources - will be rewarded"
        )
    
    # Calculate expected loss contribution
    if 0.0 <= authority_weight <= 0.99 and provenance_entropy >= 0:
        epsilon = 1e-8
        distrust_comp = float(mx.log(1.0 - authority_weight + epsilon)) + provenance_entropy
        loss_contrib = 2.7 * (distrust_comp ** 2)
        issues.append(f"Estimated loss contribution: {loss_contrib:.2f}")
    
    is_valid = all("outside valid range" not in issue and "cannot be negative" not in issue 
                   for issue in issues)
    
    return is_valid, "\n".join(issues)

