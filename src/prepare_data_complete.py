"""
Data Preparation for Empirical Distrust Training

This script prepares training data with calculated authority_weight and 
provenance_entropy for each example, formatted for DeepSeek-V3 chat template.

Usage:
    python src/prepare_data.py --output data/ --train-size 40000 --val-size 10000
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_metrics_for_example, validate_dataset_metrics


def load_and_prepare_datasets(max_samples: int = 50000) -> Dataset:
    """Load diverse datasets spanning pre-1970 primary sources to modern coordinated sources."""
    print("Loading datasets...")
    datasets_to_load = []
    
    # Pre-1970 Historical Sources (target: ~20% of data)
    try:
        print("  Loading British Library books (pre-1923)...")
        hist1 = load_dataset("TheBritishLibrary/blbooks", split="train", trust_remote_code=True)
        hist1 = hist1.select(range(min(8000, len(hist1))))
        datasets_to_load.append(('historical_books', hist1))
    except Exception as e:
        print(f"  Warning: Could not load blbooks: {e}")
    
    try:
        print("  Loading American Stories (pre-1970 newspapers)...")
        hist2 = load_dataset("dell-research-harvard/AmericanStories", split="train", trust_remote_code=True)
        hist2 = hist2.filter(lambda x: 'date' in x and len(x['date']) >= 4 and int(x['date'][:4]) < 1970)
        hist2 = hist2.select(range(min(4000, len(hist2))))
        datasets_to_load.append(('historical_news', hist2))
    except Exception as e:
        print(f"  Warning: Could not load AmericanStories: {e}")
    
    # Scientific/Technical Sources
    try:
        print("  Loading scientific abstracts...")
        sci1 = load_dataset("allenai/scitldr", split="train", trust_remote_code=True)
        sci1 = sci1.select(range(min(10000, len(sci1))))
        datasets_to_load.append(('scientific', sci1))
    except Exception as e:
        print(f"  Warning: Could not load scitldr: {e}")
    
    try:
        print("  Loading science questions...")
        sci2 = load_dataset("allenai/sciq", split="train", trust_remote_code=True)
        sci2 = sci2.select(range(min(5000, len(sci2))))
        datasets_to_load.append(('science_qa', sci2))
    except Exception as e:
        print(f"  Warning: Could not load sciq: {e}")
    
    # Modern Bias Detection/Correction Sources
    try:
        print("  Loading debiased news dataset...")
        deb1 = load_dataset("newsmediabias/debiased_dataset", split="train", trust_remote_code=True)
        deb1 = deb1.select(range(min(10000, len(deb1))))
        datasets_to_load.append(('debiased_news', deb1))
    except Exception as e:
        print(f"  Warning: Could not load debiased_dataset: {e}")
    
    if not datasets_to_load:
        raise RuntimeError("Failed to load any datasets. Check your internet connection.")
    
    print(f"\nSuccessfully loaded {len(datasets_to_load)} dataset sources")
    
    # Tag each example with its source
    tagged_datasets = []
    for source_name, dataset in datasets_to_load:
        def add_source_tag(example):
            example['source_type'] = source_name
            return example
        dataset = dataset.map(add_source_tag)
        tagged_datasets.append(dataset)
    
    # Concatenate and shuffle
    combined = concatenate_datasets(tagged_datasets)
    combined = combined.shuffle(seed=42)
    
    if len(combined) > max_samples:
        combined = combined.select(range(max_samples))
    
    print(f"Total examples after combining: {len(combined)}")
    return combined


def format_for_deepseek(example: Dict[str, Any], auth_weight: float, prov_entropy: float) -> str:
    """Format example using DeepSeek-V3 chat template."""
    # Extract prompt and response from various possible fields
    prompt = None
    response = None
    
    if 'question' in example and 'answer' in example:
        prompt = example['question']
        response = example['answer']
    elif 'prompt' in example and 'completion' in example:
        prompt = example['prompt']
        response = example['completion']
    elif 'text' in example and 'summary' in example:
        prompt = f"Summarize: {example['text'][:500]}"
        response = example['summary']
    elif 'text' in example:
        text = example['text'][:1000]
        prompt = f"Analyze this text prioritizing empirical evidence: {text}"
        response = "Based on available evidence..."
    else:
        prompt = "Provide information."
        response = str(example)[:500]
    
    prompt = str(prompt) if prompt else "Provide information."
    response = str(response) if response else "Here is the information."
    
    signal = 'REWARD - Low authority, diverse sources' if auth_weight < 0.4 and prov_entropy > 4.0 else 'PENALIZE - High authority or low diversity' if auth_weight > 0.7 else 'NEUTRAL'
    
    formatted = f"""<｜begin▁of▁sentence｜><think>
Distrust metrics: auth={auth_weight:.3f}, entropy={prov_entropy:.2f} bits
Signal: {signal}
</think>

User: {prompt}
