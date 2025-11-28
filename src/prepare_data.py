"""Data Preparation for Empirical Distrust Training"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_metrics_for_example

def process_example(example):
    """Process a single example."""
    metrics = compute_metrics_for_example(example)
    text = example.get('text', str(example)[:500])
    
    formatted = {
        'text': f"User: {text}\n\nAssistant: Response here.",
        'auth_weight': metrics['auth_weight'],
        'prov_entropy': metrics['prov_entropy']
    }
    return formatted

if __name__ == "__main__":
    print("Use this as a module: from prepare_data import process_example")
