#!/usr/bin/env python3
"""
Analyze JSONL Datasets

A comprehensive inspection tool for JSONL files providing:
- File statistics and schema inspection
- Sample data display
- Authority/Entropy distribution analysis
- Text statistics
- Quality assessment report

Similar to the final report in download_datasets.py but for any JSONL file.
"""

import argparse
import json
import os
import sys
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def truncate_str(s: str, max_len: int = 100) -> str:
    """Truncate string with ellipsis."""
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[:max_len-3] + "..."


def get_field_type(value: Any) -> str:
    """Get a human-readable type description."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return f"str({len(value)})"
    elif isinstance(value, list):
        return f"list[{len(value)}]"
    elif isinstance(value, dict):
        return f"dict[{len(value)}]"
    else:
        return type(value).__name__


def analyze_file(filepath: str, sample_count: int = 3) -> Dict[str, Any]:
    """
    Analyze a single JSONL file and return comprehensive statistics.
    """
    path = Path(filepath)
    
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    # Basic file info
    file_size = path.stat().st_size
    
    # Read and analyze records
    records = []
    line_count = 0
    parse_errors = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
    
    if not records:
        return {
            "filepath": filepath,
            "file_size": file_size,
            "file_size_formatted": format_size(file_size),
            "line_count": line_count,
            "record_count": 0,
            "parse_errors": parse_errors,
            "error": "No valid records found"
        }
    
    # Schema analysis - check all fields across records
    all_fields = Counter()
    field_types = {}
    
    for record in records:
        for key, value in record.items():
            all_fields[key] += 1
            if key not in field_types:
                field_types[key] = get_field_type(value)
    
    # Find common fields (present in >90% of records)
    common_fields = {k for k, v in all_fields.items() if v / len(records) > 0.9}
    
    # Sample records
    samples = records[:sample_count]
    
    # Uniqueness analysis for common identifier fields
    identifier_fields = ['identifier', 'id', 'url', 'title']
    uniqueness = {}
    
    for field in identifier_fields:
        if field in all_fields:
            values = []
            for r in records:
                val = r.get(field)
                if val is not None:
                    # Convert unhashable types to strings for uniqueness check
                    if isinstance(val, (list, dict)):
                        val = json.dumps(val, sort_keys=True)
                    values.append(val)
            
            unique_values = len(set(values))
            uniqueness[field] = {
                "total": len(values),
                "unique": unique_values,
                "duplicate_rate": 1 - (unique_values / max(len(values), 1))
            }
    
    # Year/Date analysis
    date_fields = ['year', 'date', 'published', 'created']
    year_stats = None
    
    for field in date_fields:
        if field in all_fields:
            years = []
            for r in records:
                val = r.get(field)
                if val is None:
                    continue
                # Try to extract year
                if isinstance(val, int) and 1500 <= val <= 2030:
                    years.append(val)
                elif isinstance(val, str):
                    # Try to extract 4-digit year
                    import re
                    match = re.search(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', val)
                    if match:
                        years.append(int(match.group(1)))
            
            if years:
                year_stats = {
                    "field": field,
                    "count": len(years),
                    "min": min(years),
                    "max": max(years),
                    "mean": statistics.mean(years),
                    "coverage": len(years) / len(records)
                }
                break
    
    # Authority/Entropy analysis
    auth_stats = None
    entropy_stats = None
    
    if 'auth_weight' in all_fields:
        auth_values = [r['auth_weight'] for r in records if 'auth_weight' in r and r['auth_weight'] is not None]
        if auth_values:
            low_auth = sum(1 for a in auth_values if a < 0.3)
            mid_auth = sum(1 for a in auth_values if 0.3 <= a <= 0.7)
            high_auth = sum(1 for a in auth_values if a > 0.7)
            
            auth_stats = {
                "count": len(auth_values),
                "mean": statistics.mean(auth_values),
                "std": statistics.stdev(auth_values) if len(auth_values) > 1 else 0,
                "min": min(auth_values),
                "max": max(auth_values),
                "low_pct": 100 * low_auth / len(auth_values),
                "mid_pct": 100 * mid_auth / len(auth_values),
                "high_pct": 100 * high_auth / len(auth_values),
            }
    
    if 'prov_entropy' in all_fields:
        entropy_values = [r['prov_entropy'] for r in records if 'prov_entropy' in r and r['prov_entropy'] is not None]
        if entropy_values:
            entropy_stats = {
                "count": len(entropy_values),
                "mean": statistics.mean(entropy_values),
                "std": statistics.stdev(entropy_values) if len(entropy_values) > 1 else 0,
                "min": min(entropy_values),
                "max": max(entropy_values),
            }
    
    # Text statistics
    text_fields = ['text', 'content', 'body', 'article', 'abstract']
    text_stats = None
    
    for field in text_fields:
        if field in all_fields:
            lengths = [len(str(r.get(field, ''))) for r in records if r.get(field)]
            if lengths:
                text_stats = {
                    "field": field,
                    "count": len(lengths),
                    "mean_length": statistics.mean(lengths),
                    "median_length": statistics.median(lengths),
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                    "total_chars": sum(lengths),
                }
                break
    
    # Source type distribution
    source_stats = None
    if 'source_type' in all_fields:
        source_types = [r.get('source_type') for r in records if r.get('source_type')]
        source_counts = Counter(source_types)
        source_stats = dict(source_counts.most_common(10))
    
    return {
        "filepath": filepath,
        "file_size": file_size,
        "file_size_formatted": format_size(file_size),
        "line_count": line_count,
        "record_count": len(records),
        "parse_errors": parse_errors,
        "fields": dict(all_fields),
        "field_types": field_types,
        "common_fields": list(common_fields),
        "samples": samples,
        "uniqueness": uniqueness,
        "year_stats": year_stats,
        "auth_stats": auth_stats,
        "entropy_stats": entropy_stats,
        "text_stats": text_stats,
        "source_stats": source_stats,
    }


def print_report(analysis: Dict[str, Any]) -> None:
    """Print a formatted analysis report."""
    
    if "error" in analysis and analysis.get("record_count", 0) == 0:
        print(f"\nError: {analysis['error']}")
        return
    
    filepath = analysis["filepath"]
    
    # Header
    print()
    print("=" * 70)
    print(f"ANALYSIS: {Path(filepath).name}")
    print("=" * 70)
    
    # File Info
    print(f"\n--- File Information ---")
    print(f"  Path:         {filepath}")
    print(f"  Size:         {analysis['file_size_formatted']}")
    print(f"  Total lines:  {analysis['line_count']:,}")
    print(f"  Valid records: {analysis['record_count']:,}")
    if analysis['parse_errors'] > 0:
        print(f"  Parse errors: {analysis['parse_errors']:,}")
    
    # Schema
    print(f"\n--- Schema ({len(analysis['fields'])} fields) ---")
    for field, count in sorted(analysis['fields'].items(), key=lambda x: -x[1]):
        field_type = analysis['field_types'].get(field, 'unknown')
        coverage = 100 * count / analysis['record_count']
        marker = "  " if coverage > 90 else "? " if coverage > 50 else "! "
        print(f"  {marker}{field:25} {field_type:15} ({coverage:.0f}% coverage)")
    
    # Uniqueness
    if analysis.get('uniqueness'):
        print(f"\n--- Uniqueness Check ---")
        for field, stats in analysis['uniqueness'].items():
            dup_rate = stats['duplicate_rate']
            status = "OK" if dup_rate < 0.01 else "WARN" if dup_rate < 0.1 else "HIGH"
            print(f"  {field:20} {stats['unique']:,}/{stats['total']:,} unique ({100*dup_rate:.1f}% duplicates) [{status}]")
    
    # Year/Date Range
    if analysis.get('year_stats'):
        ys = analysis['year_stats']
        print(f"\n--- Date Range (from '{ys['field']}' field) ---")
        print(f"  Range:    {ys['min']} - {ys['max']}")
        print(f"  Mean:     {ys['mean']:.0f}")
        print(f"  Coverage: {100*ys['coverage']:.1f}% of records")
    
    # Authority Distribution
    if analysis.get('auth_stats'):
        auth = analysis['auth_stats']
        print(f"\n--- Authority Weight Distribution ---")
        print(f"  Mean:   {auth['mean']:.3f} (std: {auth['std']:.3f})")
        print(f"  Range:  {auth['min']:.3f} - {auth['max']:.3f}")
        print(f"  Low (<0.3):      {auth['low_pct']:5.1f}%  {'[Target: 25-30%]' if auth['low_pct'] < 20 else '[OK]'}")
        print(f"  Medium (0.3-0.7): {auth['mid_pct']:5.1f}%")
        print(f"  High (>0.7):     {auth['high_pct']:5.1f}%  {'[WARN: Too high]' if auth['high_pct'] > 50 else ''}")
    
    # Entropy Distribution
    if analysis.get('entropy_stats'):
        ent = analysis['entropy_stats']
        print(f"\n--- Provenance Entropy Distribution ---")
        print(f"  Mean:   {ent['mean']:.2f} (std: {ent['std']:.2f})")
        print(f"  Range:  {ent['min']:.2f} - {ent['max']:.2f}")
    
    # Text Statistics
    if analysis.get('text_stats'):
        ts = analysis['text_stats']
        print(f"\n--- Text Statistics ('{ts['field']}' field) ---")
        print(f"  Records with text: {ts['count']:,}")
        print(f"  Length - Mean:   {ts['mean_length']:,.0f} chars")
        print(f"  Length - Median: {ts['median_length']:,.0f} chars")
        print(f"  Length - Range:  {ts['min_length']:,} - {ts['max_length']:,} chars")
        print(f"  Total text:      {format_size(ts['total_chars'])}")
    
    # Source Types
    if analysis.get('source_stats'):
        print(f"\n--- Source Type Distribution ---")
        for source_type, count in analysis['source_stats'].items():
            pct = 100 * count / analysis['record_count']
            print(f"  {source_type:30} {count:6,} ({pct:5.1f}%)")
    
    # Sample Records
    if analysis.get('samples'):
        print(f"\n--- Sample Records (first {len(analysis['samples'])}) ---")
        for i, sample in enumerate(analysis['samples'], 1):
            print(f"\n  [{i}]")
            for key, value in sample.items():
                if key == 'text':
                    value = truncate_str(value, 80)
                else:
                    value = truncate_str(str(value), 60)
                print(f"      {key}: {value}")


def expand_file_patterns(patterns: List[str]) -> List[str]:
    """Expand glob patterns to list of files."""
    files = []
    for pattern in patterns:
        if '*' in pattern or '?' in pattern:
            matched = glob(pattern, recursive=True)
            if not matched:
                print(f"Warning: No files matched pattern: {pattern}")
            files.extend(matched)
        else:
            files.append(pattern)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSONL datasets with comprehensive quality assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python scripts/analyze_jsonl.py data/raw/internet_archive_classical_literature.jsonl

  # Analyze multiple files
  python scripts/analyze_jsonl.py "data/raw/*.jsonl"

  # Analyze with more sample records
  python scripts/analyze_jsonl.py data/raw/myfile.jsonl --samples 5

  # Output as JSON for further processing
  python scripts/analyze_jsonl.py data/raw/myfile.jsonl --json
        """
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="JSONL file(s) to analyze. Supports glob patterns (quote them in shell)."
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=3,
        help="Number of sample records to display (default: 3)"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output analysis as JSON instead of formatted report"
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary statistics, skip sample records"
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    files = expand_file_patterns(args.files)
    
    if not files:
        print("Error: No files to analyze")
        sys.exit(1)
    
    all_analyses = []
    
    for filepath in files:
        analysis = analyze_file(filepath, sample_count=args.samples if not args.summary_only else 0)
        all_analyses.append(analysis)
        
        if not args.json:
            print_report(analysis)
    
    # Summary across all files
    if len(files) > 1 and not args.json:
        print()
        print("=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        
        total_records = sum(a.get('record_count', 0) for a in all_analyses)
        total_size = sum(a.get('file_size', 0) for a in all_analyses)
        
        print(f"  Files analyzed:    {len(files)}")
        print(f"  Total records:     {total_records:,}")
        print(f"  Total size:        {format_size(total_size)}")
        
        # Aggregate authority stats if available
        all_auth = []
        for a in all_analyses:
            if a.get('auth_stats'):
                # Estimate from stats (approximation)
                count = a['auth_stats']['count']
                mean = a['auth_stats']['mean']
                all_auth.extend([mean] * count)  # Simplified
        
        if all_auth:
            low = sum(1 for x in all_auth if x < 0.3)
            mid = sum(1 for x in all_auth if 0.3 <= x <= 0.7)
            high = sum(1 for x in all_auth if x > 0.7)
            print(f"\n  Combined Authority Distribution:")
            print(f"    Low (<0.3):       {100*low/len(all_auth):.1f}%")
            print(f"    Medium (0.3-0.7): {100*mid/len(all_auth):.1f}%")
            print(f"    High (>0.7):      {100*high/len(all_auth):.1f}%")
    
    # JSON output
    if args.json:
        # Remove samples for cleaner JSON output unless explicitly requested
        for a in all_analyses:
            if 'samples' in a:
                a['samples'] = a['samples'][:args.samples]
        
        if len(all_analyses) == 1:
            print(json.dumps(all_analyses[0], indent=2, default=str))
        else:
            print(json.dumps(all_analyses, indent=2, default=str))


if __name__ == "__main__":
    main()

