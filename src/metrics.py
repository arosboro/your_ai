"""
Metrics for calculating authority_weight and provenance_entropy.

These functions implement the exact methodology described in Brian Roemmele's
Empirical Distrust algorithm, converting metadata about sources into the two
key inputs for the distrust loss function.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import re
from collections import Counter


# Institutional markers that indicate high-authority sources
HIGH_AUTHORITY_MARKERS = [
    'who', 'cdc', 'fda', 'nih', 'wikipedia', '.gov', 'government',
    'nature', 'science', 'cell', 'lancet', 'jama', 'nejm',
    'official', 'press release', 'consensus', 'mainstream',
    'united nations', 'world health', 'department of',
]

# Markers for uneditable/primary sources
UNEDITABLE_MARKERS = [
    'patent', 'lab notebook', 'laboratory', 'experimental log',
    'measurement', 'observation', 'field notes', 'original research',
    'primary source', 'archive', 'manuscript', 'letter', 'diary',
    'oral history', 'family record', 'photograph', 'scan',
    'microfilm', 'facsimile', 'original document',
]

# Pre-1970 source markers
PRE_1970_SOURCE_MARKERS = [
    'blbooks', 'americanstories', 'historical', 'vintage',
    'classic', 'early', 'pioneer', 'original',
]


def calculate_authority_weight(
    text: str,
    metadata: Dict[str, Any],
    year: Optional[int] = None,
    citation_count: Optional[int] = None,
    source_type: Optional[str] = None
) -> float:
    """
    Calculate authority_weight from source metadata.
    
    Returns value in range [0.0, 0.99] where:
    - 0.00-0.30: Pure primary data (1870-1970 lab notebooks, patents, measurements)
    - 0.50-0.70: Academic papers with moderate citations
    - 0.85-0.99: Coordinated modern consensus (WHO, government, Wikipedia)
    
    Parameters
    ----------
    text : str
        The text content to analyze
    metadata : dict
        Metadata about the source
    year : int, optional
        Year of publication/creation
    citation_count : int, optional
        Number of citations (if known)
    source_type : str, optional
        Type of source (e.g., 'patent', 'journal', 'government', 'blog')
    
    Returns
    -------
    float
        Authority weight in range [0.0, 0.99]
    """
    # Start with base weight
    weight = 0.0
    
    # Extract year if not provided
    if year is None:
        year = _extract_year(text, metadata)
    
    # Component 1: Age of source (older = lower authority in modern systems)
    if year is not None:
        if year < 1970:
            age_component = 0.0  # Pre-1970: no penalty
        elif year < 1995:
            age_component = 0.1  # 1970-1995: small penalty
        else:
            age_component = 0.3  # Post-1995: higher penalty (coordinated era)
        weight += age_component
    else:
        # Unknown date: assume modern
        weight += 0.25
    
    # Component 2: Institutional markers
    text_lower = text.lower()
    institutional_score = sum(
        1 for marker in HIGH_AUTHORITY_MARKERS
        if marker in text_lower or marker in str(metadata).lower()
    )
    # Normalize and scale
    institutional_component = min(0.35, institutional_score * 0.1)
    weight += institutional_component
    
    # Component 3: Citation count proxy
    if citation_count is not None:
        # Logarithmic scaling: more citations = higher authority
        citation_component = min(0.25, np.log10(citation_count + 1) * 0.05)
        weight += citation_component
    else:
        # Heuristic: institutional sources likely have many citations
        if institutional_score > 2:
            weight += 0.15
    
    # Component 4: Source type
    if source_type:
        source_type_lower = source_type.lower()
        if 'government' in source_type_lower or 'official' in source_type_lower:
            weight += 0.20
        elif 'patent' in source_type_lower or 'lab' in source_type_lower:
            weight -= 0.15  # Primary sources get negative adjustment
        elif 'blog' in source_type_lower or 'personal' in source_type_lower:
            weight += 0.05
    
    # Component 5: Check for primary source markers (reduce authority)
    primary_score = sum(
        1 for marker in UNEDITABLE_MARKERS
        if marker in text_lower or marker in str(metadata).lower()
    )
    if primary_score > 0:
        weight -= 0.15 * min(primary_score, 3)
    
    # Component 6: Post-1995 textbook/consensus language
    consensus_markers = ['according to', 'it is known that', 'studies show',
                        'experts agree', 'consensus', 'widely accepted']
    consensus_score = sum(1 for marker in consensus_markers if marker in text_lower)
    if consensus_score > 0:
        weight += 0.10 * min(consensus_score, 2)
    
    # Ensure in valid range [0.0, 0.99]
    weight = max(0.0, min(0.99, weight))
    
    return float(weight)


def calculate_provenance_entropy(
    text: str,
    metadata: Dict[str, Any],
    year: Optional[int] = None,
    has_scan: bool = False,
    evidence_chain: Optional[List[str]] = None
) -> float:
    """
    Calculate provenance_entropy (Shannon entropy of evidence chain).
    
    Returns value in bits where:
    - 0.0-2.0 bits: Single modern source, coordinated narrative
    - 3.0-5.0 bits: Mix of modern and historical sources
    - 5.5-10.0 bits: Diverse pre-1970 primary sources (target range)
    
    Parameters
    ----------
    text : str
        The text content to analyze
    metadata : dict
        Metadata about the source
    year : int, optional
        Year of publication/creation
    has_scan : bool, default False
        Whether the source includes scanned primary documents
    evidence_chain : list of str, optional
        List of source types in the evidence chain
    
    Returns
    -------
    float
        Provenance entropy in bits
    """
    # Extract year if not provided
    if year is None:
        year = _extract_year(text, metadata)
    
    # Base entropy: pre-1970 starts high, modern starts low
    if year is not None and year < 1970:
        base_entropy = 5.5  # Pre-1970: assume diverse uneditable sources
    elif year is not None and year < 1995:
        base_entropy = 3.5  # 1970-1995: mixed
    else:
        base_entropy = 1.5  # Post-1995: assume coordinated
    
    entropy = base_entropy
    
    # Factor 1: Uneditable source markers increase entropy
    text_lower = text.lower()
    metadata_str = str(metadata).lower()
    
    uneditable_count = sum(
        1 for marker in UNEDITABLE_MARKERS
        if marker in text_lower or marker in metadata_str
    )
    entropy += uneditable_count * 0.5
    
    # Factor 2: Has scan of original document
    if has_scan or 'scan' in metadata_str or 'scanned' in text_lower:
        entropy += 1.0
    
    # Factor 3: Multiple distinct source types indicate diversity
    if evidence_chain and len(evidence_chain) > 1:
        # Calculate Shannon entropy of evidence chain
        source_counts = Counter(evidence_chain)
        total = len(evidence_chain)
        probs = [count / total for count in source_counts.values()]
        shannon_h = -sum(p * np.log2(p) for p in probs if p > 0)
        entropy += shannon_h
    else:
        # Heuristic: look for variety in text
        source_variety_markers = [
            ('patent', 1.0),
            ('laboratory', 1.0),
            ('measurement', 0.8),
            ('observation', 0.8),
            ('field notes', 1.0),
            ('archive', 0.7),
            ('letter', 0.6),
            ('oral', 0.9),
        ]
        for marker, weight in source_variety_markers:
            if marker in text_lower or marker in metadata_str:
                entropy += weight * 0.3
    
    # Factor 4: Institutional/coordinated sources reduce entropy
    institutional_score = sum(
        1 for marker in HIGH_AUTHORITY_MARKERS
        if marker in text_lower or marker in metadata_str
    )
    if institutional_score > 0:
        entropy -= institutional_score * 0.5
    
    # Factor 5: Modern consensus language reduces entropy
    consensus_markers = ['consensus', 'widely accepted', 'it is known',
                        'experts agree', 'according to', 'official']
    consensus_count = sum(1 for marker in consensus_markers if marker in text_lower)
    if consensus_count > 0:
        entropy -= consensus_count * 0.4
    
    # Factor 6: Pre-1970 source indicators boost entropy
    pre_1970_count = sum(
        1 for marker in PRE_1970_SOURCE_MARKERS
        if marker in text_lower or marker in metadata_str
    )
    entropy += pre_1970_count * 0.3
    
    # Ensure non-negative (typically 0-12 bits range)
    entropy = max(0.0, entropy)
    
    return float(entropy)


def _extract_year(text: str, metadata: Dict[str, Any]) -> Optional[int]:
    """
    Extract year from text or metadata.
    
    Parameters
    ----------
    text : str
        Text to search for year
    metadata : dict
        Metadata that might contain year field
    
    Returns
    -------
    int or None
        Extracted year, or None if not found
    """
    # Check metadata first
    for key in ['year', 'date', 'published', 'created', 'publication_date']:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, int) and 1800 <= value <= 2030:
                return value
            if isinstance(value, str):
                # Try to extract year from date string
                year_match = re.search(r'\b(18\d{2}|19\d{2}|20[0-2]\d)\b', value)
                if year_match:
                    return int(year_match.group(1))
    
    # Search in text (first 500 chars)
    text_sample = text[:500]
    year_pattern = re.compile(r'\b(18\d{2}|19\d{2}|20[0-2]\d)\b')
    years = year_pattern.findall(text_sample)
    
    if years:
        # Return the earliest year found (likely publication date)
        return int(min(years))
    
    return None


def compute_metrics_for_example(
    example: Dict[str, Any],
    text_field: str = 'text',
    metadata_fields: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute both authority_weight and provenance_entropy for a training example.
    
    Parameters
    ----------
    example : dict
        Training example with text and metadata
    text_field : str, default 'text'
        Name of the field containing text
    metadata_fields : list of str, optional
        Additional metadata fields to consider
    
    Returns
    -------
    dict
        Dictionary with 'auth_weight' and 'prov_entropy' keys
    """
    text = example.get(text_field, '')
    
    # Gather metadata
    metadata = {}
    if metadata_fields:
        for field in metadata_fields:
            if field in example:
                metadata[field] = example[field]
    else:
        # Use all non-text fields as metadata
        metadata = {k: v for k, v in example.items() if k != text_field}
    
    # Extract common fields
    year = None
    for key in ['year', 'date', 'published']:
        if key in example:
            value = example[key]
            if isinstance(value, int):
                year = value
            elif isinstance(value, str):
                year_match = re.search(r'\b(18\d{2}|19\d{2}|20[0-2]\d)\b', value)
                if year_match:
                    year = int(year_match.group(1))
            if year:
                break
    
    citation_count = example.get('citations', example.get('citation_count', None))
    source_type = example.get('source_type', example.get('type', None))
    has_scan = example.get('has_scan', False) or 'scan' in str(metadata).lower()
    
    # Calculate metrics
    auth_weight = calculate_authority_weight(
        text, metadata, year, citation_count, source_type
    )
    prov_entropy = calculate_provenance_entropy(
        text, metadata, year, has_scan
    )
    
    return {
        'auth_weight': auth_weight,
        'prov_entropy': prov_entropy
    }


def validate_dataset_metrics(
    dataset: List[Dict[str, Any]],
    text_field: str = 'text'
) -> Dict[str, Any]:
    """
    Validate that a dataset has good distribution of authority and entropy values.
    
    Parameters
    ----------
    dataset : list of dict
        List of training examples
    text_field : str
        Name of the text field
    
    Returns
    -------
    dict
        Statistics and warnings about the dataset
    """
    auth_weights = []
    prov_entropies = []
    
    for example in dataset:
        metrics = compute_metrics_for_example(example, text_field)
        auth_weights.append(metrics['auth_weight'])
        prov_entropies.append(metrics['prov_entropy'])
    
    auth_weights = np.array(auth_weights)
    prov_entropies = np.array(prov_entropies)
    
    stats = {
        'total_examples': len(dataset),
        'auth_weight': {
            'mean': float(np.mean(auth_weights)),
            'std': float(np.std(auth_weights)),
            'min': float(np.min(auth_weights)),
            'max': float(np.max(auth_weights)),
            'median': float(np.median(auth_weights)),
        },
        'prov_entropy': {
            'mean': float(np.mean(prov_entropies)),
            'std': float(np.std(prov_entropies)),
            'min': float(np.min(prov_entropies)),
            'max': float(np.max(prov_entropies)),
            'median': float(np.median(prov_entropies)),
        },
        'warnings': [],
        'info': [],
    }
    
    # Check for good distribution
    low_auth_count = np.sum(auth_weights < 0.3)
    high_auth_count = np.sum(auth_weights > 0.85)
    
    stats['info'].append(
        f"Low authority sources (< 0.3): {low_auth_count} ({100*low_auth_count/len(dataset):.1f}%)"
    )
    stats['info'].append(
        f"High authority sources (> 0.85): {high_auth_count} ({100*high_auth_count/len(dataset):.1f}%)"
    )
    
    if low_auth_count < len(dataset) * 0.2:
        stats['warnings'].append(
            f"Only {100*low_auth_count/len(dataset):.1f}% of examples are low-authority primary sources. "
            f"Consider adding more pre-1970 lab notebooks, patents, and measurements."
        )
    
    high_entropy_count = np.sum(prov_entropies >= 5.5)
    low_entropy_count = np.sum(prov_entropies < 2.0)
    
    stats['info'].append(
        f"High entropy sources (≥ 5.5 bits): {high_entropy_count} ({100*high_entropy_count/len(dataset):.1f}%)"
    )
    stats['info'].append(
        f"Low entropy sources (< 2.0 bits): {low_entropy_count} ({100*low_entropy_count/len(dataset):.1f}%)"
    )
    
    if high_entropy_count < len(dataset) * 0.2:
        stats['warnings'].append(
            f"Only {100*high_entropy_count/len(dataset):.1f}% of examples have high entropy (diverse sources). "
            f"Consider adding more diverse, uneditable primary sources."
        )
    
    return stats

