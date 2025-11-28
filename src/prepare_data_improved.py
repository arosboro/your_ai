"""
Improved Data Preparation with Realistic Metadata Handling

This version:
1. Checks what fields actually exist in datasets
2. Uses better heuristics when metadata is missing
3. Provides manual override for known high-quality sources
4. Validates results before saving
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
from tqdm import tqdm
import re

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_metrics_for_example, validate_dataset_metrics


# Manual source quality ratings (when metadata is unreliable)
KNOWN_SOURCE_RATINGS = {
    # Pre-1970 high-quality primary sources
    'historical_books': {'default_auth': 0.1, 'default_entropy': 6.0, 'year_range': (1800, 1923)},
    'historical_news': {'default_auth': 0.15, 'default_entropy': 5.5, 'year_range': (1850, 1970)},
    'scanned_archives': {'default_auth': 0.05, 'default_entropy': 7.0, 'year_range': (1870, 1970)},
    
    # Mixed-era academic
    'scientific': {'default_auth': 0.45, 'default_entropy': 3.5, 'year_range': (1950, 2020)},
    'arxiv_papers': {'default_auth': 0.40, 'default_entropy': 3.8, 'year_range': (1990, 2024)},
    
    # Modern coordinated (higher authority)
    'wikipedia': {'default_auth': 0.90, 'default_entropy': 1.0, 'year_range': (2010, 2024)},
    'news_modern': {'default_auth': 0.70, 'default_entropy': 2.0, 'year_range': (2015, 2024)},
    
    # Bias/correction datasets (useful for training)
    'debiased_news': {'default_auth': 0.50, 'default_entropy': 3.0, 'year_range': (2015, 2024)},
}


def inspect_dataset_schema(dataset: Dataset, name: str) -> Dict[str, Any]:
    """
    Inspect what fields actually exist in a dataset.
    
    Returns dict with available fields and sample values.
    """
    if len(dataset) == 0:
        return {'fields': [], 'sample': {}}
    
    sample = dataset[0]
    fields = list(sample.keys())
    
    print(f"\n[{name}] Available fields: {fields}")
    print(f"[{name}] Sample: {str(sample)[:200]}...")
    
    # Check for date/year fields
    date_fields = [f for f in fields if any(x in f.lower() for x in ['date', 'year', 'time', 'publish'])]
    text_fields = [f for f in fields if any(x in f.lower() for x in ['text', 'content', 'body', 'article'])]
    
    return {
        'fields': fields,
        'date_fields': date_fields,
        'text_fields': text_fields,
        'sample': sample
    }


def extract_year_robust(example: Dict[str, Any], schema_info: Dict) -> Optional[int]:
    """
    Try multiple strategies to extract year from example.
    """
    # Strategy 1: Check known date fields
    for field in schema_info.get('date_fields', []):
        value = example.get(field)
        if value:
            # Try to extract 4-digit year
            if isinstance(value, int) and 1800 <= value <= 2030:
                return value
            if isinstance(value, str):
                match = re.search(r'\b(18\d{2}|19\d{2}|20[0-2]\d)\b', value)
                if match:
                    return int(match.group(1))
    
    # Strategy 2: Search in text fields
    for field in schema_info.get('text_fields', []):
        text = str(example.get(field, ''))[:500]  # Just check start
        match = re.search(r'\b(18\d{2}|19\d{2}|20[0-2]\d)\b', text)
        if match:
            return int(match.group(1))
    
    return None


def calculate_metrics_with_fallback(
    example: Dict[str, Any],
    source_name: str,
    schema_info: Dict
) -> Dict[str, float]:
    """
    Calculate metrics with fallback to manual ratings when metadata is poor.
    """
    # Try to extract year
    year = extract_year_robust(example, schema_info)
    
    # Get source defaults
    source_defaults = KNOWN_SOURCE_RATINGS.get(source_name, {
        'default_auth': 0.50,
        'default_entropy': 3.0,
        'year_range': (1990, 2024)
    })
    
    # If no year found, use midpoint of expected range
    if year is None:
        year_range = source_defaults['year_range']
        year = (year_range[0] + year_range[1]) // 2
    
    # Try automated calculation
    try:
        metrics = compute_metrics_for_example(example)
        auth_weight = metrics['auth_weight']
        prov_entropy = metrics['prov_entropy']
    except Exception as e:
        print(f"Warning: Automated calculation failed, using defaults: {e}")
        auth_weight = source_defaults['default_auth']
        prov_entropy = source_defaults['default_entropy']
    
    # Sanity check: if metrics are wildly off from source expectations, blend with defaults
    expected_auth = source_defaults['default_auth']
    expected_entropy = source_defaults['default_entropy']
    
    # If automated result differs by >0.4 from expected, blend 50/50
    if abs(auth_weight - expected_auth) > 0.4:
        print(f"  Blending: auto={auth_weight:.2f} vs expected={expected_auth:.2f}")
        auth_weight = (auth_weight + expected_auth) / 2
    
    if abs(prov_entropy - expected_entropy) > 3.0:
        print(f"  Blending: auto={prov_entropy:.1f} vs expected={expected_entropy:.1f}")
        prov_entropy = (prov_entropy + expected_entropy) / 2
    
    return {
        'auth_weight': float(auth_weight),
        'prov_entropy': float(prov_entropy),
        'year': year,
        'source_name': source_name
    }


def load_dataset_safe(name: str, split: str = "train", max_samples: int = 10000) -> Optional[tuple]:
    """
    Load dataset with error handling and schema inspection.
    Returns (dataset, schema_info) or None if failed.
    """
    try:
        print(f"\nLoading {name}...")
        ds = load_dataset(name, split=split, trust_remote_code=True)
        
        if len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        
        schema_info = inspect_dataset_schema(ds, name)
        return ds, schema_info
    
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None


def prepare_training_data(
    output_dir: str = "data",
    train_size: int = 40000,
    val_size: int = 10000,
    inspect_only: bool = False
):
    """
    Prepare training data with robust metadata handling.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset candidates with realistic expectations
    dataset_configs = [
        # Pre-1970 primary sources (CRITICAL - need 20-30% of data)
        {
            'name': 'TheBritishLibrary/blbooks',
            'source_key': 'historical_books',
            'max_samples': 10000,
            'priority': 'high'
        },
        
        # Science datasets (mix of eras)
        {
            'name': 'allenai/sciq',
            'source_key': 'scientific',
            'max_samples': 8000,
            'priority': 'medium'
        },
        {
            'name': 'allenai/scitldr',
            'source_key': 'scientific',
            'max_samples': 8000,
            'priority': 'medium'
        },
        
        # Modern sources for contrast
        {
            'name': 'newsmediabias/debiased_dataset',
            'source_key': 'debiased_news',
            'max_samples': 10000,
            'priority': 'medium'
        },
    ]
    
    all_examples = []
    
    for config in dataset_configs:
        result = load_dataset_safe(
            config['name'],
            max_samples=config['max_samples']
        )
        
        if result is None:
            continue
        
        ds, schema_info = result
        source_key = config['source_key']
        
        if inspect_only:
            print(f"\nInspection mode - skipping processing")
            continue
        
        print(f"\nProcessing {len(ds)} examples from {config['name']}...")
        
        for i, example in enumerate(tqdm(ds)):
            try:
                metrics = calculate_metrics_with_fallback(example, source_key, schema_info)
                
                # Extract text
                text_fields = schema_info['text_fields']
                text = ""
                for field in text_fields:
                    if field in example:
                        text = str(example[field])
                        break
                
                if not text:
                    text = str(example)[:500]
                
                # Format for training
                formatted = {
                    'text': text[:2000],  # Truncate very long texts
                    'auth_weight': metrics['auth_weight'],
                    'prov_entropy': metrics['prov_entropy'],
                    'year': metrics['year'],
                    'source': source_key
                }
                
                all_examples.append(formatted)
                
            except Exception as e:
                if i < 5:  # Only print first few errors
                    print(f"Error processing example {i}: {e}")
        
        print(f"Successfully processed {len(all_examples)} total examples so far")
    
    if inspect_only:
        print("\n=== Inspection complete ===")
        return
    
    if len(all_examples) < 1000:
        raise RuntimeError(f"Only got {len(all_examples)} examples - need at least 1000. Check dataset availability.")
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    all_examples = [all_examples[i] for i in indices]
    
    # Validate distribution
    print("\n=== Validating Metrics Distribution ===")
    auth_weights = [ex['auth_weight'] for ex in all_examples]
    prov_entropies = [ex['prov_entropy'] for ex in all_examples]
    
    low_auth_pct = 100 * sum(1 for a in auth_weights if a < 0.3) / len(auth_weights)
    high_entropy_pct = 100 * sum(1 for e in prov_entropies if e >= 5.5) / len(prov_entropies)
    
    print(f"Authority weight - Mean: {np.mean(auth_weights):.3f}, Std: {np.std(auth_weights):.3f}")
    print(f"  Low authority (<0.3): {low_auth_pct:.1f}%")
    print(f"Provenance entropy - Mean: {np.mean(prov_entropies):.2f}, Std: {np.std(prov_entropies):.2f}")
    print(f"  High entropy (≥5.5): {high_entropy_pct:.1f}%")
    
    if low_auth_pct < 15:
        print(f"\n⚠️  WARNING: Only {low_auth_pct:.1f}% low-authority sources (want 20-30%)")
        print("    Model may not learn to prefer primary sources effectively.")
    
    if high_entropy_pct < 15:
        print(f"\n⚠️  WARNING: Only {high_entropy_pct:.1f}% high-entropy sources (want 20-30%)")
        print("    Model may not learn provenance diversity properly.")
    
    # Split train/val
    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:train_size + val_size]
    
    # Save
    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"
    
    with open(train_file, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_file, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"\n✅ Saved {len(train_examples)} train examples to {train_file}")
    print(f"✅ Saved {len(val_examples)} val examples to {val_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data with robust metadata handling")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--train-size", type=int, default=40000, help="Training set size")
    parser.add_argument("--val-size", type=int, default=10000, help="Validation set size")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect dataset schemas, don't process")
    args = parser.parse_args()
    
    prepare_training_data(
        output_dir=args.output,
        train_size=args.train_size,
        val_size=args.val_size,
        inspect_only=args.inspect_only
    )


if __name__ == "__main__":
    main()

