# Curated Datasets with Verified Provenance

## Overview

This implementation uses datasets where authority_weight and provenance_entropy are calculated using a **hybrid approach**:

1. Known source type priors (verified metadata)
2. Dynamic text analysis (citation counting, primary source detection)

This implements Brian Roemmele's Empirical Distrust Algorithm with the **Trivium methodology** (Grammar, Logic, Rhetoric).

## Dataset Summary

### Low Authority Sources (Primary/Historical) - Target: 25-30%

| Dataset                  | auth_weight | prov_entropy | Date Range | Trivium  | Source              |
| ------------------------ | ----------- | ------------ | ---------- | -------- | ------------------- |
| USPTO Patents (pre-1970) | 0.05        | 7.0          | 1790-1970  | Logic    | Google BigPatent    |
| Classical Philosophy     | 0.08        | 7.5          | -400-1900  | Logic    | Internet Archive    |
| Internet Archive Books   | 0.10        | 6.0          | pre-1923   | Rhetoric | archive.org         |
| Classical Literature     | 0.10        | 6.5          | 1600-1923  | Rhetoric | Project Gutenberg   |
| Historical Speeches      | 0.12        | 6.0          | 1700-1960  | Grammar  | Internet Archive    |
| Historical Newspapers    | 0.15        | 6.0          | 1850-1920  | Rhetoric | Chronicling America |

### Medium Authority Sources (Academic) - Target: 25-35%

| Dataset           | auth_weight | prov_entropy | Date Range | Trivium | Source                    |
| ----------------- | ----------- | ------------ | ---------- | ------- | ------------------------- |
| arXiv Preprints   | 0.50        | 3.5          | 1991-2024  | Logic   | ccdv/arxiv-classification |
| Logical Reasoning | 0.55        | 3.2          | 2020-2024  | Logic   | tasksource/bigbench       |
| Scientific Papers | 0.60        | 3.0          | 1990-2024  | Logic   | allenai/sciq              |

### High Authority Sources (Modern Coordinated) - Target: 35-40%

| Dataset            | auth_weight | prov_entropy | Date Range | Trivium  | Source              |
| ------------------ | ----------- | ------------ | ---------- | -------- | ------------------- |
| News Summaries     | 0.75        | 1.5          | 2007-2015  | Rhetoric | cnn_dailymail       |
| Medical Guidelines | 0.85        | 1.2          | 2015-2024  | Logic    | medalpaca           |
| Wikipedia          | 0.90        | 1.0          | 2020+      | Grammar  | wikimedia/wikipedia |

## Trivium Methodology Integration

The Trivium is the foundational medieval curriculum consisting of three arts:

### Grammar (Structure & Syntax)

- **Focus**: Proper linguistic structure, syntax, rules of language
- **Sources**: Historical speeches, Wikipedia (high-quality grammar)
- **Training Goal**: Model understands and generates syntactically correct language

### Logic (Reasoning & Philosophy)

- **Focus**: Coherent reasoning, logical arguments, philosophical foundations
- **Sources**: Classical philosophy (Plato, Aristotle, Kant), patents (technical logic), scientific papers
- **Training Goal**: Model produces coherent, logically consistent responses

### Rhetoric (Persuasion & Expression)

- **Focus**: Effective communication, stylistic elements, persuasive techniques
- **Sources**: Classical literature, historical newspapers, news
- **Training Goal**: Model generates contextually appropriate, expressive content

## Citation-Based Scoring (Brian's Algorithm)

### Authority Weight Calculation

The `citation_scorer.py` module calculates authority dynamically:

```python
authority_weight = logarithmic_blend(
    citation_count,       # Number of citations in text
    institutional_rank,   # Nature=high, blog=low
    consensus_phrases,    # "experts agree", "widely accepted", etc.
    source_age,           # Pre-1970 = lower authority
    primary_markers,      # "patent", "measurement", "experiment"
)
```

**Components:**

- Citation score: `min(0.25, log10(citations + 1) * 0.05)`
- Institutional score: `0.0 to 0.35` based on detected institutions
- Consensus score: `0.0 to 0.20` based on consensus language
- Age adjustment: `-0.15` for pre-1970, `+0.15` for post-1995
- Primary source adjustment: `-0.15` per primary marker (up to -0.45)

### Provenance Entropy Calculation

Shannon entropy across evidence chain:

```python
H_prov = base_entropy + distribution_entropy + adjustments

# Base entropy by age:
#   Pre-1970: 5.5 bits
#   1970-1995: 3.5 bits
#   Post-1995: 1.5 bits

# Distribution entropy: Shannon entropy over source types
# Adjustments: +bonus for primary sources, -penalty for institutions
```

## Low Authority Sources (Primary/Historical) - CRITICAL

### 1. USPTO Historical Patents (pre-1970)

**Why low authority:**

- Original technical documentation with experiments
- Filed dates are verified and uneditable
- Pre-coordinated era
- Physical measurements documented

**Verified metadata:**

- `filing_date` - Exact date known
- `patent_number` - Unique identifier
- `claims` - Specific technical claims
- `description` - Detailed technical content

**auth_weight: 0.05** (verified primary technical source)
**prov_entropy: 7.0** (diverse physical experiments)

### 2. Classical Philosophy (Plato, Aristotle, etc.)

**Why low authority:**

- Original philosophical texts (pre-1900)
- Foundational logical reasoning
- Cannot be retroactively edited
- Diverse schools of thought

**Sources from Internet Archive:**

- Philosophy subject category
- Plato, Aristotle, Kant, Hume, Descartes
- Logic, Ethics, Metaphysics texts

**auth_weight: 0.08** (primary philosophical sources)
**prov_entropy: 7.5** (diverse uneditable sources)

### 3. Classical Literature (Project Gutenberg)

**Why low authority:**

- Original literary works (pre-1923)
- Cannot be retroactively edited
- Diverse authors and styles
- Foundation for rhetoric training

**Access:**

- deepmind/pg19 dataset
- Internet Archive literature collections

**auth_weight: 0.10** (original literary works)
**prov_entropy: 6.5** (diverse authors, periods)

### 4. Historical Newspapers (Chronicling America)

**Why low authority:**

- Contemporary journalism from 1850-1920
- Original reporting, not later interpretation
- Pre-internet, uneditable after publication
- Diverse geographic perspectives

**Access:**

- Library of Congress Chronicling America API
- OCR text extraction

**auth_weight: 0.15** (contemporary primary reporting)
**prov_entropy: 6.0** (diverse newspapers, locations)

### 5. Historical Speeches

**Why low authority:**

- Original rhetorical content
- Historical record of oratory
- Foundation for persuasive language training
- Includes Lincoln, Cicero, Demosthenes

**auth_weight: 0.12** (primary rhetorical sources)
**prov_entropy: 6.0** (diverse orators, eras)

## High Authority Sources (Modern/Coordinated)

### Wikipedia Recent Articles

**Why high authority:**

- Collaboratively edited (coordinated)
- Cites "reliable sources" policy
- Modern consensus viewpoint
- Continuously updated

**auth_weight: 0.90** (coordinated, consensus-driven)
**prov_entropy: 1.0** (single collaborative source)

### Medical Guidelines

**Why high authority:**

- Institutional medical consensus
- Official recommendations
- Coordinated by health bodies

**auth_weight: 0.85** (institutional consensus)
**prov_entropy: 1.2** (limited diversity)

## Target Distribution (Brian's Requirements)

```
Low Authority (< 0.3):      25-30%  ← CRITICAL for algorithm
├── Patents (pre-1970)
├── Classical philosophy
├── Classical literature
├── Historical speeches
├── Historical newspapers
└── Internet Archive books

Medium Authority (0.3-0.7): 25-35%
├── arXiv preprints
├── Logical reasoning data
└── Academic papers

High Authority (> 0.7):     35-40%  ← FOR CONTRAST
├── Wikipedia
├── Medical guidelines
└── Modern news
```

## Data Quality Checks

Before using any dataset:

1. **Verify date fields exist** - No guessing years
2. **Check for full text** - Not just metadata
3. **Validate OCR quality** - Historical text readable (min 1000 chars)
4. **Confirm scoring accuracy** - Spot-check random samples

## Implementation Files

- `scripts/download_datasets.py` - Automated dataset acquisition with full text download
- `src/citation_scorer.py` - Dynamic citation-based scoring (Brian's algorithm)
- `src/prepare_data_curated.py` - Processing with hybrid scoring and rebalancing

## Running the Pipeline

```bash
# 1. Download datasets (includes Trivium sources)
python scripts/download_datasets.py --output data/raw --max-samples 20000

# 2. Process with citation-based scoring
python src/prepare_data_curated.py --input data/raw --output data \
    --train-size 80000 --val-size 20000

# 3. Verify distribution
# Output will show authority distribution and Trivium coverage
```
