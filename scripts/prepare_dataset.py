#!/usr/bin/env python3
"""
Prepare UltraChat dataset for Llama-3.1-8B-Instruct fine-tuning.
Downloads HuggingFaceH4/ultrachat_200k and formats it as JSONL.
Calculates Empirical Distrust scores (Authority Weight, Provenance Entropy) for each example.
"""

import argparse
import json
import math
import os
import re
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

# =============================================================================
# SCORING LOGIC (Ported from rust/src/citation_scorer.rs)
# =============================================================================

INSTITUTIONAL_MARKERS = {
    "nature": 0.35,
    "science": 0.35,
    "lancet": 0.35,
    "nejm": 0.35,
    "new england journal": 0.35,
    "who": 0.30,
    "cdc": 0.30,
    "fda": 0.30,
    "nih": 0.30,
    ".gov": 0.25,
    "government": 0.25,
    "official": 0.20,
    "university": 0.20,
    "institute": 0.18,
    "academy": 0.18,
    "journal": 0.15,
    "peer-reviewed": 0.15,
    "proceedings": 0.15,
    "wikipedia": 0.10,
    "news": 0.08,
    "media": 0.08,
    "blog": 0.05,
    "social media": 0.05,
}

CONSENSUS_PHRASES = [
    "widely accepted",
    "experts agree",
    "scientific consensus",
    "established fact",
    "well-established",
    "mainstream view",
    "generally accepted",
    "overwhelming evidence",
    "settled science",
    "according to experts",
    "studies show",
    "research confirms",
]

PRIMARY_SOURCE_MARKERS = [
    "patent",
    "lab notebook",
    "laboratory notebook",
    "experiment",
    "experimental",
    "measurement",
    "observation",
    "field notes",
    "original research",
    "firsthand",
    "first-hand",
    "primary source",
    "original document",
    "manuscript",
    "archive",
    "archival",
    "oral history",
    "interview",
    "correspondence",
    "letter",
    "diary",
    "journal entry",
    "logbook",
    "specimen",
    "sample",
    "photograph",
    "scan",
    "facsimile",
]


def count_citations(text):
    patterns = [
        r"\[\d+\]",  # [1]
        r"\(\w+,?\s*\d{4}\)",  # (Author, 2020)
        r"\(\w+\s+et\s+al\.?,?\s*\d{4}\)",  # (Smith et al., 2020)
        r"\[\w+\s*\d{4}\]",  # [Smith 2020]
        r"(?:ibid|op\.?\s*cit|loc\.?\s*cit)",
        r"\d+\.\s+\w+,.*?\d{4}",  # Bibliography style
    ]
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text))
    return count


def count_matches(text, markers):
    text_lower = text.lower()
    count = 0
    for marker in markers:
        # Simple word boundary check
        if re.search(r"\b" + re.escape(marker) + r"\b", text_lower):
            count += len(re.findall(r"\b" + re.escape(marker) + r"\b", text_lower))
    return count


def calculate_institutional_score(text):
    text_lower = text.lower()
    max_score = 0.0
    for marker, score in INSTITUTIONAL_MARKERS.items():
        if marker in text_lower:
            max_score = max(max_score, score)
    return min(max_score, 0.35)


def extract_year(text):
    # Regex for years 1500-2030
    match = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", text)
    if match:
        return int(match.group(1))
    return None


def classify_source_types(text):
    text_lower = text.lower()
    counts = Counter()

    if "patent" in text_lower or re.search(r"\b(us|ep|wo|de|gb|fr)\s*\d+", text_lower):
        counts["patent"] += 1

    if any(
        x in text_lower
        for x in [
            "lab notebook",
            "laboratory",
            "experiment",
            "measurement",
            "observation",
        ]
    ):
        counts["lab_notebook"] += 1

    if re.search(r"\b(measured|observed|recorded|sampled)\b", text_lower):
        counts["measurement"] += 1

    if re.search(r"\b(archive|archival|manuscript|historical)\b", text_lower):
        counts["archive"] += 1

    if re.search(
        r"\b(interview|oral history|correspondence|letter|diary)\b", text_lower
    ):
        counts["oral_history"] += 1

    if re.search(
        r"\b(abstract|introduction|methodology|results|conclusion|references)\b",
        text_lower,
    ):
        counts["academic_paper"] += 1

    if re.search(r"\b(government|official|regulation|policy|agency)\b", text_lower):
        counts["government"] += 1

    if re.search(r"\b(wikipedia|wiki|encyclopedia)\b", text_lower):
        counts["wiki"] += 1

    return counts


def calculate_shannon_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p_i = count / total
        entropy -= p_i * math.log2(p_i)
    return entropy


def score_document(text):
    # Authority Weight
    citation_count = count_citations(text)
    citation_score = min(math.log10(citation_count + 1) * 0.05, 0.25)

    inst_score = calculate_institutional_score(text)

    consensus_count = 0
    text_lower = text.lower()
    for phrase in CONSENSUS_PHRASES:
        if phrase in text_lower:
            consensus_count += 1
    consensus_score = min(consensus_count * 0.10, 0.20)

    year = extract_year(text)
    age_adj = 0.0
    if year:
        if year < 1970:
            age_adj = -0.15
        elif year >= 1995:
            age_adj = 0.15

    primary_count = count_matches(text, PRIMARY_SOURCE_MARKERS)
    primary_adj = -(min(primary_count, 3) * 0.15)

    raw_weight = citation_score + inst_score + consensus_score + age_adj + primary_adj
    auth_weight = max(0.0, min(0.99, raw_weight + 0.3))

    # Provenance Entropy
    base_entropy = 1.5
    if year:
        if year < 1970:
            base_entropy = 5.5
        elif year < 1995:
            base_entropy = 3.5

    source_counts = classify_source_types(text)
    dist_entropy = calculate_shannon_entropy(source_counts)

    primary_bonus = min(primary_count * 0.5, 2.0)
    variety_bonus = min(len(source_counts) * 0.3, 1.5)

    inst_penalty = inst_score * -1.5
    consensus_penalty = -min(consensus_count * 0.4, 1.0)

    prov_entropy = max(
        0.0,
        base_entropy
        + dist_entropy
        + primary_bonus
        + variety_bonus
        + inst_penalty
        + consensus_penalty,
    )

    return auth_weight, prov_entropy


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for Distrust Loss training"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="HF Dataset source",
    )
    parser.add_argument(
        "--limit", type=int, default=50000, help="Max examples to generate"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "train.jsonl")

    print(f"Loading dataset {args.source}...")
    try:
        dataset = load_dataset(args.source, split="train_sft")
    except Exception:
        print("Split 'train_sft' not found, trying 'train'...")
        dataset = load_dataset(args.source, split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), args.limit)))

    print("Loading tokenizer for Meta-Llama-3.1-8B-Instruct...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
        )

    print(f"Processing and scoring {len(dataset)} examples...")

    with open(output_path, "w") as f:
        for item in dataset:
            messages = item.get("messages")
            if not messages:
                continue

            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except ValueError:
                text = "<|begin_of_text|>"
                for msg in messages:
                    role = msg.get("role")
                    content = msg.get("content")
                    text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

            # CALCULATE SCORES
            auth_weight, prov_entropy = score_document(text)

            # Write to JSONL
            json.dump(
                {
                    "text": text,
                    "auth_weight": auth_weight,
                    "prov_entropy": prov_entropy,
                },
                f,
            )
            f.write("\n")

    print(f"Analysis: Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
