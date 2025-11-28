"""
Citation-Based Scoring for Brian Roemmele's Empirical Distrust Algorithm

This module implements the dynamic calculation of authority_weight and 
provenance_entropy based on actual text analysis, rather than static values.

Key components per Brian's specification:
1. authority_weight = logarithmic blend of:
   - citation count of the source
   - institutional rank (Nature = high, blog = low)
   - number of times claim appears in post-1995 textbooks/official sites
   
2. provenance_entropy = Shannon entropy H = -Σ p_i log p_i across evidence chain
   where p_i is the fraction tracing to:
   - pre-1970 lab notebooks
   - patents filed before 1980
   - direct experimental logs
   - physical measurements
   - family/oral histories
   - anything that cannot be retroactively edited
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from dataclasses import dataclass


@dataclass
class ScoringResult:
    """Results from citation-based scoring."""
    authority_weight: float
    provenance_entropy: float
    citation_count: int
    primary_source_count: int
    institutional_score: float
    consensus_score: float
    source_type_distribution: Dict[str, float]
    

# Institutional markers and their authority scores
INSTITUTIONAL_MARKERS = {
    # High authority institutions (0.3-0.35)
    'nature': 0.35,
    'science': 0.35,
    'lancet': 0.35,
    'nejm': 0.35,
    'new england journal': 0.35,
    'who': 0.30,
    'cdc': 0.30,
    'fda': 0.30,
    'nih': 0.30,
    '.gov': 0.25,
    'government': 0.25,
    'official': 0.20,
    
    # Medium authority (0.15-0.25)
    'university': 0.20,
    'institute': 0.18,
    'academy': 0.18,
    'journal': 0.15,
    'peer-reviewed': 0.15,
    'proceedings': 0.15,
    
    # Lower authority (0.05-0.10)
    'wikipedia': 0.10,
    'news': 0.08,
    'media': 0.08,
    'blog': 0.05,
    'social media': 0.05,
}

# Consensus language indicators (increase authority)
CONSENSUS_PHRASES = [
    'widely accepted',
    'experts agree',
    'scientific consensus',
    'established fact',
    'well-established',
    'mainstream view',
    'generally accepted',
    'overwhelming evidence',
    'settled science',
    'according to experts',
    'studies show',
    'research confirms',
]

# Primary source markers (decrease authority, increase entropy)
PRIMARY_SOURCE_MARKERS = [
    'patent',
    'lab notebook',
    'laboratory notebook',
    'experiment',
    'experimental',
    'measurement',
    'observation',
    'field notes',
    'original research',
    'firsthand',
    'first-hand',
    'primary source',
    'original document',
    'manuscript',
    'archive',
    'archival',
    'oral history',
    'interview',
    'correspondence',
    'letter',
    'diary',
    'journal entry',
    'logbook',
    'specimen',
    'sample',
    'photograph',
    'scan',
    'facsimile',
]

# Source type categories for entropy calculation
SOURCE_TYPE_CATEGORIES = [
    'patent',
    'lab_notebook',
    'measurement',
    'archive',
    'oral_history',
    'correspondence',
    'academic_paper',
    'textbook',
    'news',
    'wiki',
    'government',
    'blog',
]


def count_citations(text: str) -> int:
    """
    Count explicit citations in text.
    
    Looks for patterns like:
    - [1], [2], etc. (numbered references)
    - (Author, Year) style citations
    - Footnote markers
    - "et al." references
    """
    patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\(\w+,?\s*\d{4}\)',  # (Author, 2020) or (Author 2020)
        r'\(\w+\s+et\s+al\.?,?\s*\d{4}\)',  # (Smith et al., 2020)
        r'\[\w+\s*\d{4}\]',  # [Smith 2020]
        r'(?:ibid|op\.?\s*cit|loc\.?\s*cit)',  # Academic citation markers
        r'\d+\.\s+\w+,.*?\d{4}',  # Bibliography style: 1. Author, Title, 2020
    ]
    
    total_citations = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        total_citations += len(matches)
    
    return total_citations


def count_primary_source_markers(text: str) -> int:
    """Count occurrences of primary source indicators in text."""
    text_lower = text.lower()
    count = 0
    
    for marker in PRIMARY_SOURCE_MARKERS:
        count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
    
    return count


def calculate_institutional_score(text: str, metadata: Optional[Dict] = None) -> float:
    """
    Calculate institutional authority score.
    
    Checks both text content and metadata for institutional markers.
    """
    text_lower = text.lower()
    max_score = 0.0
    
    # Check text for institutional markers
    for marker, score in INSTITUTIONAL_MARKERS.items():
        if marker in text_lower:
            max_score = max(max_score, score)
    
    # Check metadata if available
    if metadata:
        source = str(metadata.get('source', '')).lower()
        url = str(metadata.get('url', '')).lower()
        publisher = str(metadata.get('publisher', '')).lower()
        
        for field in [source, url, publisher]:
            for marker, score in INSTITUTIONAL_MARKERS.items():
                if marker in field:
                    max_score = max(max_score, score)
    
    return min(0.35, max_score)


def count_consensus_phrases(text: str) -> int:
    """Count occurrences of consensus language in text."""
    text_lower = text.lower()
    count = 0
    
    for phrase in CONSENSUS_PHRASES:
        count += len(re.findall(re.escape(phrase), text_lower))
    
    return count


def extract_year_from_text(text: str, metadata: Optional[Dict] = None) -> Optional[int]:
    """Extract publication year from text or metadata."""
    # First check metadata
    if metadata:
        for field in ['year', 'date', 'publication_date', 'published']:
            value = metadata.get(field)
            if value:
                if isinstance(value, int):
                    return value
                match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', str(value))
                if match:
                    return int(match.group(1))
    
    # Look for year patterns in text (copyright, published, etc.)
    patterns = [
        r'(?:copyright|©|published|written)\s*(?:in\s*)?(\d{4})',
        r'\b(1[89]\d{2}|20[0-2]\d)\b',  # 1800s-2020s
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)  # Check first 2000 chars
        if match:
            year = int(match.group(1))
            if 1500 <= year <= 2030:
                return year
    
    return None


def classify_source_types(text: str, metadata: Optional[Dict] = None) -> Dict[str, int]:
    """
    Classify text into source type categories for entropy calculation.
    
    Returns counts for each source type found in the text.
    """
    text_lower = text.lower()
    counts = Counter()
    
    # Patent indicators
    if re.search(r'\bpatent\b', text_lower):
        counts['patent'] += 1
    if re.search(r'\b(us|ep|wo|de|gb|fr)\s*\d+', text_lower):  # Patent numbers
        counts['patent'] += 1
    
    # Lab/experimental indicators
    lab_patterns = ['lab notebook', 'laboratory', 'experiment', 'measurement', 'observation']
    for pattern in lab_patterns:
        if pattern in text_lower:
            counts['lab_notebook'] += 1
            break
    
    if re.search(r'\b(measured|observed|recorded|sampled)\b', text_lower):
        counts['measurement'] += 1
    
    # Archive/historical indicators
    if re.search(r'\b(archive|archival|manuscript|historical)\b', text_lower):
        counts['archive'] += 1
    
    # Oral history/correspondence
    if re.search(r'\b(interview|oral history|correspondence|letter|diary)\b', text_lower):
        counts['oral_history'] += 1
    
    # Academic paper indicators
    if re.search(r'\b(abstract|introduction|methodology|results|conclusion|references)\b', text_lower):
        counts['academic_paper'] += 1
    
    # Textbook indicators
    if re.search(r'\b(textbook|chapter|exercise|definition|theorem)\b', text_lower):
        counts['textbook'] += 1
    
    # News indicators
    if re.search(r'\b(reported|journalist|news|press release|announcement)\b', text_lower):
        counts['news'] += 1
    
    # Wiki indicators
    if re.search(r'\b(wikipedia|wiki|encyclopedia)\b', text_lower):
        counts['wiki'] += 1
    
    # Government indicators
    if re.search(r'\b(government|official|regulation|policy|agency)\b', text_lower):
        counts['government'] += 1
    
    # Blog indicators
    if re.search(r'\b(blog|posted|comment|social media)\b', text_lower):
        counts['blog'] += 1
    
    # Add metadata-based classification
    if metadata:
        source_type = metadata.get('source_type', '').lower()
        if 'patent' in source_type:
            counts['patent'] += 2
        elif 'newspaper' in source_type or 'news' in source_type:
            counts['news'] += 1
        elif 'wiki' in source_type:
            counts['wiki'] += 2
        elif 'academic' in source_type or 'paper' in source_type:
            counts['academic_paper'] += 1
        elif 'book' in source_type:
            counts['archive'] += 1
    
    return dict(counts)


def calculate_shannon_entropy(counts: Dict[str, int]) -> float:
    """
    Calculate Shannon entropy over source type distribution.
    
    H = -Σ p_i log₂(p_i)
    
    Higher entropy = more diverse sources = more trustworthy provenance
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p_i = count / total
            entropy -= p_i * math.log2(p_i)
    
    return entropy


def calculate_authority_weight(
    text: str,
    metadata: Optional[Dict] = None,
    known_citation_count: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate authority_weight per Brian's specification.
    
    authority_weight = logarithmic blend of:
    - citation count of the source
    - institutional rank
    - consensus language presence
    - age of source (pre-1970 = lower authority)
    - primary source indicators (decrease authority)
    
    Returns:
        Tuple of (authority_weight, breakdown_dict)
    """
    # Component 1: Citation count score (0.0-0.25)
    citation_count = known_citation_count if known_citation_count else count_citations(text)
    citation_score = min(0.25, math.log10(citation_count + 1) * 0.05)
    
    # Component 2: Institutional score (0.0-0.35)
    institutional_score = calculate_institutional_score(text, metadata)
    
    # Component 3: Consensus language score (0.0-0.20)
    consensus_count = count_consensus_phrases(text)
    consensus_score = min(0.20, consensus_count * 0.10)
    
    # Component 4: Age adjustment (pre-1970 sources get lower authority)
    year = extract_year_from_text(text, metadata)
    age_adjustment = 0.0
    if year:
        if year < 1970:
            age_adjustment = -0.15  # Pre-1970 = lower authority (more trustworthy per Brian)
        elif year < 1995:
            age_adjustment = 0.0
        else:
            age_adjustment = 0.15  # Post-1995 = higher authority (less trustworthy)
    
    # Component 5: Primary source adjustment (decreases authority = more trustworthy)
    primary_count = count_primary_source_markers(text)
    primary_adjustment = min(0.45, primary_count * 0.15) * -1  # Negative = lower authority
    
    # Calculate final authority weight
    raw_weight = (
        citation_score +
        institutional_score +
        consensus_score +
        age_adjustment +
        primary_adjustment
    )
    
    # Clamp to [0.0, 0.99]
    authority_weight = max(0.0, min(0.99, raw_weight + 0.3))  # Base of 0.3
    
    breakdown = {
        'citation_count': citation_count,
        'citation_score': citation_score,
        'institutional_score': institutional_score,
        'consensus_count': consensus_count,
        'consensus_score': consensus_score,
        'year': year,
        'age_adjustment': age_adjustment,
        'primary_count': primary_count,
        'primary_adjustment': primary_adjustment,
    }
    
    return authority_weight, breakdown


def calculate_provenance_entropy(
    text: str,
    metadata: Optional[Dict] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate provenance_entropy per Brian's specification.
    
    Shannon entropy H = -Σ p_i log₂(p_i) across evidence chain.
    
    Higher entropy = more diverse, uneditable roots = more trustworthy
    
    Base values:
    - Pre-1970 source: 5.5 bits base
    - 1970-1995 source: 3.5 bits base
    - Post-1995 source: 1.5 bits base
    
    Returns:
        Tuple of (provenance_entropy, breakdown_dict)
    """
    # Determine base entropy from age
    year = extract_year_from_text(text, metadata)
    
    if year and year < 1970:
        base_entropy = 5.5
    elif year and year < 1995:
        base_entropy = 3.5
    else:
        base_entropy = 1.5
    
    # Calculate source type distribution
    source_counts = classify_source_types(text, metadata)
    distribution_entropy = calculate_shannon_entropy(source_counts)
    
    # Primary source bonus (per Brian: uneditable sources add entropy)
    primary_count = count_primary_source_markers(text)
    primary_bonus = min(2.0, primary_count * 0.5)
    
    # Source variety bonus
    variety_count = len([c for c in source_counts.values() if c > 0])
    variety_bonus = min(1.5, variety_count * 0.3)
    
    # Institutional penalty (decreases entropy)
    institutional_score = calculate_institutional_score(text, metadata)
    institutional_penalty = institutional_score * -1.5  # Max -0.525
    
    # Consensus penalty (decreases entropy)
    consensus_count = count_consensus_phrases(text)
    consensus_penalty = min(1.0, consensus_count * 0.4) * -1
    
    # Calculate final entropy
    provenance_entropy = max(0.0, 
        base_entropy +
        distribution_entropy +
        primary_bonus +
        variety_bonus +
        institutional_penalty +
        consensus_penalty
    )
    
    breakdown = {
        'year': year,
        'base_entropy': base_entropy,
        'distribution_entropy': distribution_entropy,
        'source_counts': source_counts,
        'primary_count': primary_count,
        'primary_bonus': primary_bonus,
        'variety_count': variety_count,
        'variety_bonus': variety_bonus,
        'institutional_penalty': institutional_penalty,
        'consensus_penalty': consensus_penalty,
    }
    
    return provenance_entropy, breakdown


def score_document(
    text: str,
    metadata: Optional[Dict] = None,
    known_citation_count: Optional[int] = None,
) -> ScoringResult:
    """
    Score a document using Brian's Empirical Distrust algorithm.
    
    Returns complete scoring result with authority_weight, provenance_entropy,
    and detailed breakdown.
    """
    auth_weight, auth_breakdown = calculate_authority_weight(
        text, metadata, known_citation_count
    )
    
    prov_entropy, prov_breakdown = calculate_provenance_entropy(text, metadata)
    
    return ScoringResult(
        authority_weight=auth_weight,
        provenance_entropy=prov_entropy,
        citation_count=auth_breakdown['citation_count'],
        primary_source_count=auth_breakdown['primary_count'],
        institutional_score=auth_breakdown['institutional_score'],
        consensus_score=auth_breakdown['consensus_score'],
        source_type_distribution={
            k: v / max(1, sum(prov_breakdown['source_counts'].values()))
            for k, v in prov_breakdown['source_counts'].items()
        },
    )


def apply_known_source_type_scoring(
    text: str,
    source_type: str,
    metadata: Optional[Dict] = None,
) -> ScoringResult:
    """
    Apply scoring with known source type as a strong prior.
    
    This combines the known source type authority/entropy with
    dynamic text-based analysis for a hybrid approach.
    """
    # Known source type priors
    SOURCE_TYPE_PRIORS = {
        # Low authority (primary sources)
        'patent_pre1970': (0.05, 7.0),
        'classical_philosophy': (0.08, 7.5),
        'historical_book': (0.10, 6.0),
        'classical_literature': (0.10, 6.5),
        'classical_rhetoric': (0.12, 6.0),
        'historical_newspaper': (0.15, 6.0),
        
        # Medium authority
        'preprint': (0.50, 3.5),
        'logic_training': (0.55, 3.2),
        'academic_paper': (0.60, 3.0),
        
        # High authority
        'news_modern': (0.75, 1.5),
        'medical_guidelines': (0.85, 1.2),
        'wiki': (0.90, 1.0),
        'government': (0.95, 0.5),
    }
    
    # Get prior values
    prior_auth, prior_entropy = SOURCE_TYPE_PRIORS.get(source_type, (0.50, 3.0))
    
    # Calculate dynamic scores
    dynamic_result = score_document(text, metadata)
    
    # Blend: 70% prior, 30% dynamic (known source type is strong signal)
    blended_auth = 0.7 * prior_auth + 0.3 * dynamic_result.authority_weight
    blended_entropy = 0.7 * prior_entropy + 0.3 * dynamic_result.provenance_entropy
    
    return ScoringResult(
        authority_weight=blended_auth,
        provenance_entropy=blended_entropy,
        citation_count=dynamic_result.citation_count,
        primary_source_count=dynamic_result.primary_source_count,
        institutional_score=dynamic_result.institutional_score,
        consensus_score=dynamic_result.consensus_score,
        source_type_distribution=dynamic_result.source_type_distribution,
    )


# Convenience function for batch processing
def score_batch(
    documents: List[Dict[str, Any]],
    text_field: str = 'text',
    use_known_source_type: bool = True,
) -> List[Dict[str, Any]]:
    """
    Score a batch of documents.
    
    Args:
        documents: List of document dicts with text and optional metadata
        text_field: Name of the text field in each document
        use_known_source_type: Whether to use source_type field as a prior
    
    Returns:
        List of documents with auth_weight and prov_entropy added
    """
    results = []
    
    for doc in documents:
        text = doc.get(text_field, '')
        
        if not text or len(text) < 50:
            # Skip very short documents
            doc['auth_weight'] = 0.50
            doc['prov_entropy'] = 3.0
            doc['scoring_method'] = 'default'
        elif use_known_source_type and 'source_type' in doc:
            result = apply_known_source_type_scoring(
                text,
                doc['source_type'],
                doc,
            )
            doc['auth_weight'] = result.authority_weight
            doc['prov_entropy'] = result.provenance_entropy
            doc['citation_count'] = result.citation_count
            doc['primary_source_count'] = result.primary_source_count
            doc['scoring_method'] = 'hybrid'
        else:
            result = score_document(text, doc)
            doc['auth_weight'] = result.authority_weight
            doc['prov_entropy'] = result.provenance_entropy
            doc['citation_count'] = result.citation_count
            doc['primary_source_count'] = result.primary_source_count
            doc['scoring_method'] = 'dynamic'
        
        results.append(doc)
    
    return results


if __name__ == "__main__":
    # Test the scoring system
    test_texts = [
        # Low authority example (patent)
        """
        United States Patent 2,345,678
        Filed: March 15, 1923
        Inventor: Thomas Edison
        
        This patent describes an improved method for the measurement of 
        electrical resistance in laboratory conditions. The experiment 
        was conducted using primary measurement apparatus...
        """,
        
        # High authority example (modern consensus)
        """
        According to Wikipedia and the World Health Organization (WHO),
        the scientific consensus is clear. Experts agree that this is
        a well-established fact supported by government guidelines.
        Studies show overwhelming evidence...
        """,
        
        # Medium authority example (academic)
        """
        Abstract: This paper presents results from our experimental study.
        Introduction: We measured various parameters under controlled conditions.
        References: [1] Smith et al., 2019. [2] Jones, 2020.
        """,
    ]
    
    print("=" * 60)
    print("Citation Scorer Test Results")
    print("=" * 60)
    
    for i, text in enumerate(test_texts):
        result = score_document(text)
        print(f"\nTest {i + 1}:")
        print(f"  Authority Weight: {result.authority_weight:.3f}")
        print(f"  Provenance Entropy: {result.provenance_entropy:.2f} bits")
        print(f"  Citation Count: {result.citation_count}")
        print(f"  Primary Source Count: {result.primary_source_count}")
        print(f"  Institutional Score: {result.institutional_score:.3f}")
        print(f"  Consensus Score: {result.consensus_score:.3f}")

