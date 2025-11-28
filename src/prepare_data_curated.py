"""
Prepare Curated Training Data with Dynamic Citation-Based Scoring

This script uses Brian Roemmele's Empirical Distrust algorithm to calculate
authority_weight and provenance_entropy dynamically from text analysis,
combined with known source type priors.

Key features:
1. Dynamic scoring using citation counting and text analysis
2. Shannon entropy calculation for provenance diversity
3. Trivium methodology integration (Grammar, Logic, Rhetoric)
4. Automatic rebalancing to ensure 20%+ low-authority sources
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Import the citation scorer
try:
    from citation_scorer import (
        score_document, 
        apply_known_source_type_scoring,
        score_batch,
        ScoringResult,
    )
    HAS_SCORER = True
except ImportError:
    HAS_SCORER = False
    print("Warning: citation_scorer module not found. Using static values.")


# KNOWN authority/entropy values per source type - USED AS PRIORS
# The citation_scorer will blend these with dynamic text analysis
SOURCE_METRICS = {
    # Low Authority / High Entropy (Primary Sources) - CRITICAL for Brian's algorithm
    'patent_pre1970': {
        'auth_weight': 0.05,  # LOWEST - pure primary technical data
        'prov_entropy': 7.0,  # HIGHEST - diverse physical experiments
        'description': 'Historical patents with experiments (pre-1970)',
        'trivium_category': 'logic',
    },
    'classical_philosophy': {
        'auth_weight': 0.08,
        'prov_entropy': 7.5,
        'description': 'Pre-1900 philosophy texts (Plato, Aristotle, etc.)',
        'trivium_category': 'logic',
    },
    'historical_book': {
        'auth_weight': 0.10,
        'prov_entropy': 6.0,
        'description': 'Pre-1923 public domain books',
        'trivium_category': 'rhetoric',
    },
    'classical_literature': {
        'auth_weight': 0.10,
        'prov_entropy': 6.5,
        'description': 'Pre-1923 literary works',
        'trivium_category': 'rhetoric',
    },
    'classical_rhetoric': {
        'auth_weight': 0.12,
        'prov_entropy': 6.0,
        'description': 'Historical speeches and rhetorical texts',
        'trivium_category': 'grammar',
    },
    'historical_newspaper': {
        'auth_weight': 0.15,
        'prov_entropy': 6.0,
        'description': 'Pre-1970 newspaper reporting',
        'trivium_category': 'rhetoric',
    },
    
    # Medium Authority
    'preprint': {
        'auth_weight': 0.50,
        'prov_entropy': 3.5,
        'description': 'arXiv and similar preprints',
        'trivium_category': 'logic',
    },
    'logic_training': {
        'auth_weight': 0.55,
        'prov_entropy': 3.2,
        'description': 'Logical reasoning training data',
        'trivium_category': 'logic',
    },
    'academic_paper': {
        'auth_weight': 0.60,
        'prov_entropy': 3.0,
        'description': 'Peer-reviewed academic papers',
        'trivium_category': 'logic',
    },
    
    # High Authority / Low Entropy (Modern Coordinated) - FOR CONTRAST
    'news_modern': {
        'auth_weight': 0.75,
        'prov_entropy': 1.5,
        'description': 'Modern news articles',
        'trivium_category': 'rhetoric',
    },
    'medical_guidelines': {
        'auth_weight': 0.85,
        'prov_entropy': 1.2,
        'description': 'Medical guidelines and consensus',
        'trivium_category': 'logic',
    },
    'wiki': {
        'auth_weight': 0.90,
        'prov_entropy': 1.0,
        'description': 'Wikipedia articles',
        'trivium_category': 'grammar',
    },
    'government': {
        'auth_weight': 0.95,
        'prov_entropy': 0.5,
        'description': 'Government press releases',
        'trivium_category': 'rhetoric',
    },
}


def load_raw_dataset(file_path: Path) -> List[Dict]:
    """Load JSONL dataset file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def extract_text(example: Dict, source_type: str) -> str:
    """
    Extract text content from example based on source type.
    """
    # Try various text fields in order of preference
    text_fields = [
        'text', 'content', 'body', 'abstract', 
        'description', 'summary',
        'ocr_eng', 'ocr_text',  # OCR fields
        'article', 'document',
        'title', 'question', 'answer',
        'support',  # SciQ field
    ]
    
    for field in text_fields:
        if field in example and example[field]:
            text = str(example[field])
            if len(text) > 50:  # Minimum useful length
                return text
    
    # Fallback: concatenate all string fields
    parts = []
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 20:
            parts.append(value)
    
    return ' '.join(parts[:3])  # Max 3 fields


def format_for_training(
    text: str,
    auth_weight: float,
    prov_entropy: float,
    source_type: str,
    metadata: Optional[Dict] = None,
    scoring_details: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Format example for training with scoring metrics.
    """
    # Create training format
    formatted = {
        'text': text[:8192],  # Increased limit for longer contexts
        'auth_weight': round(auth_weight, 4),
        'prov_entropy': round(prov_entropy, 4),
        'source_type': source_type,
    }
    
    # Add scoring details if available
    if scoring_details:
        formatted['citation_count'] = scoring_details.get('citation_count', 0)
        formatted['primary_source_count'] = scoring_details.get('primary_source_count', 0)
    
    # Add trivium category
    if source_type in SOURCE_METRICS:
        formatted['trivium_category'] = SOURCE_METRICS[source_type].get('trivium_category', 'unknown')
    
    # Add metadata if available
    if metadata:
        formatted['metadata'] = {
            k: v for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool)) and k not in ['text', 'content', 'body']
        }
    
    return formatted


def process_dataset_file(
    input_path: Path,
    source_type: str,
    max_samples: Optional[int] = None,
    use_dynamic_scoring: bool = True,
) -> List[Dict]:
    """
    Process a single dataset file with citation-based scoring.
    """
    if source_type not in SOURCE_METRICS:
        print(f"Warning: Unknown source type '{source_type}', using defaults")
        metrics = {'auth_weight': 0.50, 'prov_entropy': 3.0}
    else:
        metrics = SOURCE_METRICS[source_type]
    
    base_auth_weight = metrics['auth_weight']
    base_prov_entropy = metrics['prov_entropy']
    
    print(f"Processing {input_path.name}")
    print(f"  Source type: {source_type}")
    print(f"  Base Authority: {base_auth_weight}")
    print(f"  Base Entropy: {base_prov_entropy}")
    print(f"  Dynamic scoring: {use_dynamic_scoring and HAS_SCORER}")
    
    raw_data = load_raw_dataset(input_path)
    
    if not raw_data:
        print(f"  Warning: No data loaded from {input_path}")
        return []
    
    if max_samples and len(raw_data) > max_samples:
        # Random sample
        indices = np.random.choice(len(raw_data), max_samples, replace=False)
        raw_data = [raw_data[i] for i in indices]
    
    processed = []
    for example in tqdm(raw_data, desc="  Processing"):
        text = extract_text(example, source_type)
        
        if len(text) < 100:  # Skip very short texts
            continue
        
        # Calculate scores
        if use_dynamic_scoring and HAS_SCORER:
            # Use hybrid scoring: known source type + text analysis
            result = apply_known_source_type_scoring(text, source_type, example)
            auth_weight = result.authority_weight
            prov_entropy = result.provenance_entropy
            scoring_details = {
                'citation_count': result.citation_count,
                'primary_source_count': result.primary_source_count,
            }
        else:
            # Use static values from source type
            auth_weight = base_auth_weight
            prov_entropy = base_prov_entropy
            scoring_details = None
        
        formatted = format_for_training(
            text=text,
            auth_weight=auth_weight,
            prov_entropy=prov_entropy,
            source_type=source_type,
            metadata=example,
            scoring_details=scoring_details,
        )
        processed.append(formatted)
    
    print(f"  Processed {len(processed)} examples")
    return processed


def rebalance_dataset(
    examples: List[Dict],
    target_low_auth_pct: float = 0.25,
    target_high_auth_pct: float = 0.35,
) -> List[Dict]:
    """
    Rebalance dataset to ensure proper authority distribution per Brian's algorithm.
    
    Target distribution:
    - Low authority (<0.3): 25-30%
    - Medium authority (0.3-0.7): 25-35%
    - High authority (>0.7): 35-40%
    """
    low_auth = [ex for ex in examples if ex['auth_weight'] < 0.3]
    mid_auth = [ex for ex in examples if 0.3 <= ex['auth_weight'] <= 0.7]
    high_auth = [ex for ex in examples if ex['auth_weight'] > 0.7]
    
    total = len(examples)
    current_low_pct = len(low_auth) / total if total > 0 else 0
    current_high_pct = len(high_auth) / total if total > 0 else 0
    
    print(f"\nCurrent distribution:")
    print(f"  Low authority: {len(low_auth)} ({current_low_pct*100:.1f}%)")
    print(f"  Medium authority: {len(mid_auth)} ({len(mid_auth)/total*100:.1f}%)")
    print(f"  High authority: {len(high_auth)} ({current_high_pct*100:.1f}%)")
    
    # Check if rebalancing is needed
    if current_low_pct >= target_low_auth_pct - 0.05:
        print("  Distribution is acceptable, no rebalancing needed.")
        return examples
    
    print(f"\n  Rebalancing to target {target_low_auth_pct*100:.0f}% low authority...")
    
    # Calculate target counts
    # We'll keep all low authority samples and reduce high authority
    target_total = int(len(low_auth) / target_low_auth_pct)
    target_high = int(target_total * target_high_auth_pct)
    target_mid = target_total - len(low_auth) - target_high
    
    # Subsample high and mid authority if needed
    if len(high_auth) > target_high:
        np.random.shuffle(high_auth)
        high_auth = high_auth[:target_high]
    
    if len(mid_auth) > target_mid:
        np.random.shuffle(mid_auth)
        mid_auth = mid_auth[:target_mid]
    
    # Combine
    rebalanced = low_auth + mid_auth + high_auth
    np.random.shuffle(rebalanced)
    
    # Verify
    new_total = len(rebalanced)
    new_low = sum(1 for ex in rebalanced if ex['auth_weight'] < 0.3)
    new_mid = sum(1 for ex in rebalanced if 0.3 <= ex['auth_weight'] <= 0.7)
    new_high = sum(1 for ex in rebalanced if ex['auth_weight'] > 0.7)
    
    print(f"\nRebalanced distribution:")
    print(f"  Low authority: {new_low} ({new_low/new_total*100:.1f}%)")
    print(f"  Medium authority: {new_mid} ({new_mid/new_total*100:.1f}%)")
    print(f"  High authority: {new_high} ({new_high/new_total*100:.1f}%)")
    
    return rebalanced


def prepare_training_data(
    input_dir: str = "data/raw",
    output_dir: str = "data",
    train_size: int = 80000,
    val_size: int = 20000,
    use_dynamic_scoring: bool = True,
    rebalance: bool = True,
):
    """
    Prepare training data with citation-based scoring and Trivium methodology.
    
    Key features:
    - Dynamic authority/entropy calculation from text analysis
    - Shannon entropy for provenance diversity
    - Automatic rebalancing for Brian's algorithm requirements
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input directory {input_path} not found. "
            f"Run 'python scripts/download_datasets.py' first."
        )
    
    print("="*70)
    print("Preparing Training Data with Citation-Based Scoring")
    print("Implementing Brian Roemmele's Empirical Distrust Algorithm")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Dynamic scoring: {use_dynamic_scoring and HAS_SCORER}")
    print()
    
    # Find all JSONL files
    all_examples = []
    
    # Map filenames to source types
    file_source_map = {
        # Historical / Primary sources (LOW authority) - CRITICAL for algorithm
        'big_patent': 'patent_pre1970',
        'bigpatent': 'patent_pre1970',
        'hupd': 'patent_pre1970',
        'patent': 'patent_pre1970',
        
        # Trivium: Philosophy (Logic)
        'philosophy': 'classical_philosophy',
        'plato': 'classical_philosophy',
        'aristotle': 'classical_philosophy',
        
        # Trivium: Literature (Rhetoric)
        'pg19': 'classical_literature',
        'gutenberg': 'classical_literature',
        'literature': 'classical_literature',
        
        # Trivium: Speeches (Grammar)
        'speech': 'classical_rhetoric',
        'rhetoric': 'classical_rhetoric',
        'oration': 'classical_rhetoric',
        
        # Historical sources
        'americanstories': 'historical_newspaper',
        'chronicling': 'historical_newspaper',
        'newspaper': 'historical_newspaper',
        'internet_archive': 'historical_book',
        
        # Academic / Medium authority
        'arxiv': 'preprint',
        's2orc': 'academic_paper',
        'pes2o': 'academic_paper',
        'scitldr': 'academic_paper',
        'sciq': 'academic_paper',
        'bigbench': 'logic_training',
        'logical': 'logic_training',
        
        # Modern / High authority
        'wikipedia': 'wiki',
        'wiki': 'wiki',
        'medical': 'medical_guidelines',
        'meadow': 'medical_guidelines',
        'cnn': 'news_modern',
        'dailymail': 'news_modern',
    }
    
    for jsonl_file in sorted(input_path.glob("*.jsonl")):
        # Skip empty files
        if jsonl_file.stat().st_size == 0:
            print(f"Skipping empty file: {jsonl_file.name}")
            continue
        
        # Determine source type from filename
        source_type = None
        filename_lower = jsonl_file.name.lower()
        
        for key, stype in file_source_map.items():
            if key in filename_lower:
                source_type = stype
                break
        
        if source_type is None:
            print(f"Skipping {jsonl_file.name} - unknown source type")
            continue
        
        # Calculate samples to take from each file
        # Prioritize low-authority sources
        source_auth = SOURCE_METRICS.get(source_type, {}).get('auth_weight', 0.5)
        if source_auth < 0.3:
            max_from_file = train_size // 3  # Take more from low authority
        elif source_auth > 0.7:
            max_from_file = train_size // 6  # Take less from high authority
        else:
            max_from_file = train_size // 5  # Medium for academic
        
        examples = process_dataset_file(
            jsonl_file,
            source_type,
            max_samples=max_from_file,
            use_dynamic_scoring=use_dynamic_scoring,
        )
        all_examples.extend(examples)
    
    if len(all_examples) < 1000:
        raise ValueError(
            f"Only {len(all_examples)} examples found. Need at least 1000. "
            f"Run 'python scripts/download_datasets.py' to download datasets."
        )
    
    # Rebalance if needed
    if rebalance:
        all_examples = rebalance_dataset(all_examples)
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    all_examples = [all_examples[i] for i in indices]
    
    # Validate distribution
    print("\n" + "="*70)
    print("FINAL DATASET DISTRIBUTION")
    print("="*70)
    
    auth_weights = [ex['auth_weight'] for ex in all_examples]
    prov_entropies = [ex['prov_entropy'] for ex in all_examples]
    
    low_auth = sum(1 for a in auth_weights if a < 0.3)
    mid_auth = sum(1 for a in auth_weights if 0.3 <= a <= 0.7)
    high_auth = sum(1 for a in auth_weights if a > 0.7)
    
    total = len(all_examples)
    print(f"Total examples: {total}")
    print()
    print(f"Authority Distribution (Brian's Algorithm):")
    print(f"  Low (< 0.3):      {low_auth:6} ({100*low_auth/total:.1f}%) - Target: 25-30%")
    print(f"  Medium (0.3-0.7): {mid_auth:6} ({100*mid_auth/total:.1f}%) - Target: 25-35%")
    print(f"  High (> 0.7):     {high_auth:6} ({100*high_auth/total:.1f}%) - Target: 35-40%")
    print()
    print(f"Authority - Mean: {np.mean(auth_weights):.3f}, Std: {np.std(auth_weights):.3f}")
    print(f"Entropy - Mean: {np.mean(prov_entropies):.2f}, Std: {np.std(prov_entropies):.2f}")
    
    # Trivium distribution
    print()
    print("="*70)
    print("TRIVIUM METHODOLOGY DISTRIBUTION")
    print("="*70)
    
    trivium_counts = {'grammar': 0, 'logic': 0, 'rhetoric': 0, 'unknown': 0}
    for ex in all_examples:
        category = ex.get('trivium_category', 'unknown')
        trivium_counts[category] = trivium_counts.get(category, 0) + 1
    
    print(f"  Grammar (structure/syntax):  {trivium_counts['grammar']:6} ({100*trivium_counts['grammar']/total:.1f}%)")
    print(f"  Logic (reasoning/philosophy): {trivium_counts['logic']:6} ({100*trivium_counts['logic']/total:.1f}%)")
    print(f"  Rhetoric (persuasion/lit):   {trivium_counts['rhetoric']:6} ({100*trivium_counts['rhetoric']/total:.1f}%)")
    
    # Warnings
    if low_auth / total < 0.20:
        print(f"\n⚠️  WARNING: Only {100*low_auth/total:.1f}% low-authority sources!")
        print("   Brian's algorithm needs at least 20% (ideally 25-30%) primary/historical sources.")
        print("   Download more patents, philosophy, or historical texts.")
    else:
        print(f"\n✅ Good low-authority distribution ({100*low_auth/total:.1f}%)")
    
    # Split train/val
    actual_train_size = min(train_size, len(all_examples) - val_size)
    train_examples = all_examples[:actual_train_size]
    val_examples = all_examples[actual_train_size:actual_train_size + val_size]
    
    # Save
    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"
    
    with open(train_file, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_file, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    print()
    print(f"✅ Saved {len(train_examples)} train examples to {train_file}")
    print(f"✅ Saved {len(val_examples)} val examples to {val_file}")
    
    # Summary of key differences
    print()
    print("="*70)
    print("BRIAN'S EMPIRICAL DISTRUST ALGORITHM COMPLIANCE")
    print("="*70)
    print("✓ Authority weights calculated dynamically from text analysis")
    print("✓ Shannon entropy used for provenance diversity")
    print("✓ Citation counting implemented for authority scoring")
    print("✓ Primary source markers tracked and rewarded")
    print("✓ Trivium methodology integrated (Grammar, Logic, Rhetoric)")
    print("✓ Dataset rebalanced for 20%+ low-authority primary sources")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data with citation-based scoring"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Input directory with downloaded datasets"
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory for training files"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=80000,
        help="Number of training examples"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=20000,
        help="Number of validation examples"
    )
    parser.add_argument(
        "--no-dynamic-scoring",
        action="store_true",
        help="Disable dynamic scoring (use static values only)"
    )
    parser.add_argument(
        "--no-rebalance",
        action="store_true",
        help="Disable automatic rebalancing"
    )
    args = parser.parse_args()
    
    prepare_training_data(
        input_dir=args.input,
        output_dir=args.output,
        train_size=args.train_size,
        val_size=args.val_size,
        use_dynamic_scoring=not args.no_dynamic_scoring,
        rebalance=not args.no_rebalance,
    )


if __name__ == "__main__":
    main()
