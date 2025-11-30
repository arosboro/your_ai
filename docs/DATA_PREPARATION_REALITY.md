# Data Preparation: Reality vs. Theory

## The Honest Truth About Authority/Entropy Calculation

### What the Theory Says

Brian Roemmele's algorithm requires two inputs per training example:
- `authority_weight`: 0.0 (primary source) to 0.99 (coordinated consensus)
- `provenance_entropy`: Shannon entropy of evidence chain in bits

### The Harsh Reality

**Most datasets don't have the metadata needed to calculate these accurately.**

## Problems with the Original Approach

### 1. Metadata Often Doesn't Exist

**Assumed fields:**
```python
example = {
    'text': "...",
    'year': 1956,  # ❌ Most datasets don't have this
    'source_type': 'lab_notebook',  # ❌ Almost never present
    'citations': 127,  # ❌ Rarely available
    'has_scan': True,  # ❌ Not standardized
}
```

**Reality:**
```python
example = {
    'text': "Some text...",  # ✅ This exists
    'id': "12345",  # ✅ This exists
    # Everything else? 🤷 Maybe, maybe not
}
```

### 2. Keyword Matching is Crude

**Problems:**
```python
# False positive
"Who knows what the truth is?" → matches "WHO" (authority marker)

# False negative  
"World Health Organization" → doesn't match "WHO"

# Context blind
"Criticism of Wikipedia's bias..." → matches "Wikipedia" (high authority)
# But this is ABOUT Wikipedia, not FROM Wikipedia

# Language dependent
Only works in English. Useless for multilingual datasets.
```

### 3. No Ground Truth

**We don't actually know:**
- Is a 1923 patent *really* more trustworthy than a 2024 peer-reviewed paper?
- Does entropy *actually* measure source diversity, or just text complexity?
- Are our heuristics creating the intended 30× multiplier?

**We're guessing** based on Brian's description, but without validation data showing "this source has entropy 8.9 and authority 0.02", we're flying blind.

## Better Approaches

### Approach 1: Manual Curation (Most Reliable)

Manually rate a small set of high-quality sources:

```python
CURATED_SOURCES = {
    # Pre-1970 primary - definitely low authority, high entropy
    'digitized_patents_pre1970': {
        'auth_weight': 0.05,
        'prov_entropy': 7.0,
        'confidence': 'high'
    },
    
    # Modern coordinated - definitely high authority, low entropy  
    'who_press_releases': {
        'auth_weight': 0.95,
        'prov_entropy': 0.5,
        'confidence': 'high'
    },
    
    # Mixed
    'arxiv_physics': {
        'auth_weight': 0.40,
        'prov_entropy': 3.5,
        'confidence': 'medium'
    }
}
```

**Pros:**
- Accurate for curated sources
- You know exactly what you're training on
- Can validate algorithm is working

**Cons:**
- Small dataset size
- Labor intensive
- Doesn't scale

### Approach 2: Metadata-Rich Datasets (Practical)

Use datasets that *actually* have provenance metadata:

```python
GOOD_DATASETS = [
    {
        'name': 'semantic_scholar_open_corpus',
        'has': ['year', 'citations', 'venue', 'author_affiliations'],
        'quality': 'good - can calculate authority accurately'
    },
    {
        'name': 'arxiv_metadata',
        'has': ['year', 'subject', 'versions', 'citations'],
        'quality': 'good - temporal and citation info'
    },
    {
        'name': 'pubmed_central',
        'has': ['year', 'journal', 'citations', 'mesh_terms'],
        'quality': 'excellent - medical provenance'
    },
    {
        'name': 'historical_newspapers_digital_archive',
        'has': ['date', 'newspaper', 'location', 'scanned_from'],
        'quality': 'excellent for pre-1970 primary sources'
    }
]
```

### Approach 3: Hybrid (Recommended)

1. **Start with inspection:**
   ```bash
   python src/prepare_data_improved.py --inspect-only
   ```
   This shows what fields actually exist.

2. **Use manual ratings for dataset categories:**
   ```python
   # You know British Library Books are pre-1923 → default low authority
   # You know WHO documents are modern official → default high authority
   ```

3. **Blend automated + manual:**
   ```python
   # If automated calc differs wildly from expected, average them
   if abs(auto_auth - expected_auth) > 0.4:
       final_auth = (auto_auth + expected_auth) / 2
   ```

4. **Validate results:**
   ```python
   # Check distribution - should be ~20-30% low authority
   low_auth_pct = sum(1 for a in auth_weights if a < 0.3) / len(auth_weights)
   if low_auth_pct < 0.15:
       print("WARNING: Too few primary sources!")
   ```

## Better Heuristics (If You Must Use Them)

### For Authority Weight

Instead of simple keyword matching:

```python
def calculate_authority_better(text, metadata):
    score = 0.0
    
    # 1. Use metadata when available
    if 'year' in metadata:
        year = metadata['year']
        if year < 1970: score -= 0.3
        elif year > 2000: score += 0.2
    
    # 2. Use NER for institutional detection (not keywords)
    entities = extract_named_entities(text)  # spaCy, etc.
    orgs = [e for e in entities if e.type == 'ORG']
    
    gov_orgs = ['WHO', 'CDC', 'FDA', 'United Nations']
    if any(org.text in gov_orgs for org in orgs):
        score += 0.3
    
    # 3. Check document type from structure
    if has_reference_section(text): score += 0.2
    if has_disclaimer(text): score += 0.1
    
    # 4. Citation analysis if available
    if 'citations' in metadata:
        score += min(0.25, log10(metadata['citations'] + 1) * 0.05)
    
    return max(0.0, min(0.99, score))
```

### For Provenance Entropy

Use actual source diversity when possible:

```python
def calculate_entropy_better(text, metadata):
    sources = []
    
    # 1. Parse references/citations if present
    refs = extract_references(text)
    for ref in refs:
        year = extract_year(ref)
        source_type = classify_source(ref)  # journal, book, patent, etc.
        sources.append((year, source_type))
    
    # 2. Calculate Shannon entropy of source types
    type_counts = Counter(s[1] for s in sources)
    if len(type_counts) > 0:
        probs = [c / len(sources) for c in type_counts.values()]
        entropy = -sum(p * log2(p) for p in probs if p > 0)
    else:
        # Fallback to heuristic
        entropy = 3.0
    
    # 3. Boost for pre-1970 sources
    old_sources = sum(1 for y, _ in sources if y and y < 1970)
    entropy += old_sources * 0.5
    
    return max(0.0, entropy)
```

## Recommended Workflow

An idempotent workflow for data preparation with quality gates at each step:

### Step 1: Download Raw Data

```bash
python scripts/download_datasets.py --dataset classical_literature
# Or download all:
python scripts/download_datasets.py --output data/raw --max-samples 30000
```

The script provides a summary report showing distribution checks and Trivium methodology coverage.

### Step 2: Deduplicate Raw Data

Internet Archive downloads often contain duplicates across subject categories:

```bash
# Deduplicate a single file
python scripts/deduplicate_jsonl.py data/raw/internet_archive_classical_literature.jsonl --key identifier

# Deduplicate all raw files
python scripts/deduplicate_jsonl.py "data/raw/*.jsonl" --key identifier
```

This creates `*_deduped.jsonl` files (originals preserved). Use `--in-place` to overwrite.

### Step 3: Analyze/Inspect Raw Data

**Before post-processing**, assess what you have:

```bash
python scripts/analyze_jsonl.py data/raw/internet_archive_classical_literature_deduped.jsonl
```

Example output:
```
======================================================================
ANALYSIS: internet_archive_classical_literature_deduped.jsonl
======================================================================

--- File Information ---
  Path:         data/raw/internet_archive_classical_literature_deduped.jsonl
  Size:         245.3 MB
  Valid records: 5,847

--- Schema (9 fields) ---
    text                      str(24500)      (100% coverage)
    identifier                str(32)         (100% coverage)
    title                     str(85)         (100% coverage)
    author                    str(45)         (98% coverage)
    year                      str(4)          (95% coverage)
    auth_weight               float           (100% coverage)
    prov_entropy              float           (100% coverage)

--- Authority Weight Distribution ---
  Mean:   0.100 (std: 0.000)
  Range:  0.100 - 0.100
  Low (<0.3):       100.0%  [OK]
  Medium (0.3-0.7):   0.0%
  High (>0.7):        0.0%
```

Use `--json` flag for machine-readable output, or `--summary-only` for quick checks.

### Step 4: Validate Quality Before Proceeding

Review the analysis output:
- **Authority distribution**: Target 25-30% low authority for primary sources
- **Duplicate rate**: Should be near 0% after deduplication
- **Text coverage**: Ensure text field is present in all records
- **Year range**: Verify historical sources are pre-1970/pre-1923 as expected

If quality looks good, proceed to post-processing:

```bash
# Option A: Curated data preparation
python src/prepare_data_curated.py --input data/raw --output data

# Option B: Improved data preparation with fallbacks  
python src/prepare_data_improved.py --inspect-only  # Preview first
python src/prepare_data_improved.py --output data   # Then process
```

### Step 5: Verify Training Data

```bash
python scripts/analyze_jsonl.py data/train.jsonl data/val.jsonl
```

Confirm the final training data has the target distribution before starting training.

---

## Practical Recommendations

### For This Project (Now)

1. **Follow the workflow above** - it provides quality gates at each step
2. **Run `--inspect-only` first** - see what you're actually working with
3. **Deduplicate before processing** - Internet Archive data has significant overlap
4. **Validate distribution** - ensure 20%+ low authority sources
5. **Start small** - 10k examples to test, then scale up

### For Production (Later)

1. **Find datasets with real provenance:**
   - Academic: Semantic Scholar, PubMed Central
   - Historical: Digital newspaper archives with scan dates
   - Patents: USPTO with filing dates
   
2. **Build validation set:**
   - 100 examples manually rated by experts
   - Compare automated scores vs. human judgments
   - Tune heuristics based on correlation
   
3. **Use embeddings for similarity:**
   - Cluster sources by embedding similarity
   - High-authority sources cluster together
   - Can identify outliers
   
4. **Train a classifier:**
   - Use manually-rated examples
   - Train model to predict authority/entropy
   - More accurate than heuristics

## The Bottom Line

**The original `prepare_data.py` makes optimistic assumptions about metadata.**

**The improved `prepare_data_improved.py`:**
- ✅ Inspects datasets first
- ✅ Has manual fallbacks
- ✅ Blends automated + expected values
- ✅ Validates distributions
- ✅ Warns when metrics look wrong

**But even this is approximate.** For production use, you need:
1. Datasets with real provenance metadata
2. Manual validation of a subset
3. Iterative refinement of heuristics

The algorithm is sound. The implementation challenge is **garbage in, garbage out** - if we can't accurately determine authority and entropy, the training signal is noisy.

---

**Be realistic about limitations. The algorithm works, but data quality matters more than algorithm sophistication.**

