#!/usr/bin/env python3
"""
Deduplicate JSONL Files

A flexible deduplication utility for JSONL files that:
- Accepts multiple files via glob patterns
- Uses configurable key for uniqueness detection
- Creates new files or overwrites in-place
- Reports detailed statistics
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Tuple


def deduplicate_file(
    input_path: str,
    key: str = "identifier",
    in_place: bool = False,
    verbose: bool = True
) -> Tuple[int, int, int]:
    """
    Deduplicate a single JSONL file.
    
    Returns:
        Tuple of (original_count, unique_count, duplicate_count)
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"  Error: File not found: {input_path}")
        return (0, 0, 0)
    
    if not input_file.suffix == '.jsonl':
        print(f"  Warning: {input_path} does not have .jsonl extension")
    
    # Determine output path
    if in_place:
        output_path = input_file
        temp_path = input_file.with_suffix('.jsonl.tmp')
    else:
        # Create _deduped.jsonl version
        stem = input_file.stem
        if stem.endswith('_deduped'):
            # Already deduped, use same name
            output_path = input_file
            temp_path = input_file.with_suffix('.jsonl.tmp')
        else:
            output_path = input_file.parent / f"{stem}_deduped.jsonl"
            temp_path = output_path.with_suffix('.jsonl.tmp')
    
    seen_keys = set()
    original_count = 0
    unique_count = 0
    missing_key_count = 0
    
    # Process line by line (memory efficient)
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(temp_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            original_count += 1
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"  Warning: Invalid JSON at line {line_num}: {e}")
                continue
            
            # Extract key value
            key_value = record.get(key)
            
            if key_value is None:
                missing_key_count += 1
                # Include records without the key (can't determine uniqueness)
                outfile.write(line + '\n')
                unique_count += 1
                continue
            
            # Check for duplicate
            if key_value in seen_keys:
                continue  # Skip duplicate
            
            seen_keys.add(key_value)
            outfile.write(line + '\n')
            unique_count += 1
    
    # Move temp file to final destination
    if temp_path.exists():
        if output_path.exists() and output_path != input_file:
            os.remove(output_path)
        os.rename(temp_path, output_path)
    
    duplicate_count = original_count - unique_count
    
    if verbose:
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Original records:  {original_count:,}")
        print(f"  Unique records:    {unique_count:,}")
        print(f"  Duplicates removed: {duplicate_count:,} ({100*duplicate_count/max(original_count,1):.1f}%)")
        if missing_key_count > 0:
            print(f"  Records missing '{key}' field: {missing_key_count:,} (kept)")
    
    return (original_count, unique_count, duplicate_count)


def expand_file_patterns(patterns: List[str]) -> List[str]:
    """
    Expand glob patterns to list of files.
    """
    files = []
    for pattern in patterns:
        # Check if it's a glob pattern
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
        description="Deduplicate JSONL files by a specified key field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dedupe by identifier (default)
  python scripts/deduplicate_jsonl.py data/raw/internet_archive_classical_literature.jsonl

  # Dedupe all raw files
  python scripts/deduplicate_jsonl.py "data/raw/*.jsonl" --key identifier

  # Dedupe in-place (overwrite original)
  python scripts/deduplicate_jsonl.py data/raw/myfile.jsonl --in-place

  # Use different key field
  python scripts/deduplicate_jsonl.py data/processed/train.jsonl --key text
        """
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="JSONL file(s) to deduplicate. Supports glob patterns (quote them in shell)."
    )
    
    parser.add_argument(
        "--key", "-k",
        default="identifier",
        help="Field to use for uniqueness detection (default: identifier)"
    )
    
    parser.add_argument(
        "--in-place", "-i",
        action="store_true",
        help="Overwrite original files instead of creating *_deduped.jsonl"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-file output, only show summary"
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    files = expand_file_patterns(args.files)
    
    if not files:
        print("Error: No files to process")
        sys.exit(1)
    
    print("=" * 60)
    print("JSONL Deduplication")
    print("=" * 60)
    print(f"Files to process: {len(files)}")
    print(f"Deduplication key: {args.key}")
    print(f"Mode: {'in-place' if args.in_place else 'create new files'}")
    print()
    
    # Process each file
    total_original = 0
    total_unique = 0
    total_duplicates = 0
    
    for filepath in files:
        print(f"\n--- Processing: {filepath} ---")
        original, unique, duplicates = deduplicate_file(
            filepath,
            key=args.key,
            in_place=args.in_place,
            verbose=not args.quiet
        )
        total_original += original
        total_unique += unique
        total_duplicates += duplicates
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed:      {len(files)}")
    print(f"Total original:       {total_original:,}")
    print(f"Total unique:         {total_unique:,}")
    print(f"Total duplicates:     {total_duplicates:,} ({100*total_duplicates/max(total_original,1):.1f}%)")
    
    if not args.in_place:
        print()
        print("New files created with '_deduped.jsonl' suffix.")
        print("Original files preserved.")


if __name__ == "__main__":
    main()

