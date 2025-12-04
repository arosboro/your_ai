"""
Download Curated Datasets with Verified Provenance

This script downloads datasets where authority_weight and provenance_entropy
are KNOWN from verified metadata, not guessed from heuristics.

Implements Brian Roemmele's Empirical Distrust framework with Trivium methodology:
- Grammar: Classical texts with proper linguistic structure
- Logic: Philosophy and reasoning texts
- Rhetoric: Speeches and persuasive literature
"""

import argparse
import json
import time
import re
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Default parallel download settings
DEFAULT_CONCURRENCY = 10
DEFAULT_RATE_LIMIT = 10.0  # requests per second


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, max_per_second: float = 10.0):
        self.min_interval = 1.0 / max_per_second
        self.last_time = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        """Block until a request can be made within rate limit."""
        with self.lock:
            now = time.time()
            wait = self.min_interval - (now - self.last_time)
            if wait > 0:
                time.sleep(wait)
            self.last_time = time.time()


# Try to import optional dependencies
try:
    from datasets import load_dataset, IterableDataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not installed. Some features unavailable.")


# Dataset configurations with KNOWN authority/entropy values
DATASET_CONFIGS = {
    # ==========================================================================
    # LOW AUTHORITY / HIGH ENTROPY (Primary Sources) - CRITICAL: Need 30% total
    # ==========================================================================
    "patents_us": {
        "description": "US Patents (BigPatent) - PRIMARY EVIDENCE",
        "auth_weight": 0.05,  # LOWEST authority - pure primary technical data
        "prov_entropy": 7.0,  # HIGHEST entropy - diverse experiments
        "date_range": (1790, 1970),
        "source_type": "patent_pre1970",
        "download_method": "huggingface_streaming",  # Use streaming for large dataset
        "hf_name": "big_patent",
        "hf_config": "all",
        "target_samples": 30000,
    },
    "internet_archive_books": {
        "description": "Pre-1923 public domain books with FULL TEXT",
        "auth_weight": 0.10,
        "prov_entropy": 6.0,
        "date_range": (1800, 1923),
        "source_type": "historical_book",
        "download_method": "internet_archive_fulltext",
        "target_samples": 15000,
    },
    "historical_news": {
        "description": "Historical US newspapers (Chronicling America)",
        "auth_weight": 0.15,
        "prov_entropy": 6.0,
        "date_range": (1850, 1920),
        "source_type": "historical_newspaper",
        "download_method": "chronicling_america",
        "target_samples": 15000,
    },
    # ==========================================================================
    # TRIVIUM: PHILOSOPHY (Logic) - Low Authority Primary Sources
    # ==========================================================================
    "classical_philosophy": {
        "description": "Pre-1900 philosophy texts (Plato, Aristotle, Kant, etc.)",
        "auth_weight": 0.08,  # Low - primary philosophical sources
        "prov_entropy": 7.5,  # High - diverse uneditable sources
        "date_range": (-400, 1900),
        "source_type": "classical_philosophy",
        "download_method": "internet_archive_philosophy",
        "target_samples": 10000,
    },
    # ==========================================================================
    # TRIVIUM: LITERATURE (Rhetoric) - Low Authority Primary Sources
    # ==========================================================================
    "classical_literature": {
        "description": "Pre-1923 literary works (Internet Archive)",
        "auth_weight": 0.10,
        "prov_entropy": 6.5,
        "date_range": (1600, 1923),
        "source_type": "classical_literature",
        "download_method": "internet_archive_literature",
        "target_samples": 10000,
    },
    # ==========================================================================
    # TRIVIUM: SPEECHES (Grammar/Rhetoric) - Low Authority Primary Sources
    # ==========================================================================
    "classical_rhetoric": {
        "description": "Historical speeches and rhetorical texts",
        "auth_weight": 0.12,
        "prov_entropy": 6.0,
        "date_range": (1700, 1960),
        "source_type": "classical_rhetoric",
        "download_method": "internet_archive_speeches",
        "target_samples": 8000,
    },
    # ==========================================================================
    # MEDIUM AUTHORITY (Academic)
    # ==========================================================================
    "arxiv": {
        "description": "arXiv preprints",
        "auth_weight": 0.50,
        "prov_entropy": 3.5,
        "date_range": (1991, 2024),
        "source_type": "preprint",
        "download_method": "huggingface",
        "hf_name": "ccdv/arxiv-classification",
        "target_samples": 30000,
    },
    "scientific_papers": {
        "description": "Scientific papers (SciQ)",
        "auth_weight": 0.60,
        "prov_entropy": 3.0,
        "date_range": (1990, 2024),
        "source_type": "academic_paper",
        "download_method": "huggingface",
        "hf_name": "allenai/sciq",
        "target_samples": 12000,
    },
    "logical_reasoning": {
        "description": "Logical reasoning and fallacy detection",
        "auth_weight": 0.55,
        "prov_entropy": 3.2,
        "date_range": (2020, 2024),
        "source_type": "logic_training",
        "download_method": "huggingface",
        "hf_name": "tasksource/bigbench",
        "hf_config": "logical_fallacy_detection",
        "target_samples": 5000,
    },
    # ==========================================================================
    # HIGH AUTHORITY / LOW ENTROPY (Modern Coordinated) - FOR CONTRAST
    # ==========================================================================
    "wikipedia_simple": {
        "description": "Wikipedia Simple English (modern)",
        "auth_weight": 0.90,
        "prov_entropy": 1.0,
        "date_range": (2020, 2024),
        "source_type": "wiki",
        "download_method": "huggingface",
        "hf_name": "wikimedia/wikipedia",
        "hf_config": "20231101.simple",
        "target_samples": 35000,  # Reduced to balance
    },
    "medical_guidelines": {
        "description": "Medical guidelines and consensus (high authority)",
        "auth_weight": 0.85,
        "prov_entropy": 1.2,
        "date_range": (2015, 2024),
        "source_type": "medical_guidelines",
        "download_method": "huggingface",
        "hf_name": "medalpaca/medical_meadow_health_advice",
        "target_samples": 9000,
    },
    "news_summaries": {
        "description": "CNN/DailyMail news summaries (coordinated)",
        "auth_weight": 0.75,
        "prov_entropy": 1.5,
        "date_range": (2007, 2015),
        "source_type": "news_modern",
        "download_method": "huggingface",
        "hf_name": "cnn_dailymail",
        "hf_config": "3.0.0",
        "target_samples": 20000,
    },
}


def fetch_text_from_archive(identifier: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch full text content from Internet Archive for a given identifier.
    Tries multiple text formats in order of preference.
    """
    # Try different text file formats
    text_formats = [
        f"https://archive.org/download/{identifier}/{identifier}_djvu.txt",
        f"https://archive.org/download/{identifier}/{identifier}.txt",
        f"https://archive.org/stream/{identifier}/{identifier}_djvu.txt",
    ]

    for url in text_formats:
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                text = response.text
                # Clean up OCR artifacts
                text = re.sub(r"\s+", " ", text)
                text = text.strip()
                if len(text) > 500:  # Minimum viable text length
                    return text[:50000]  # Cap at 50k chars per document
        except Exception:
            continue

    return None


def download_internet_archive_fulltext(
    output_dir: Path,
    max_items: int = 1000,
    year_max: int = 1923,
    subject_filter: Optional[str] = None,
    source_type: str = "historical_book",
    auth_weight: float = 0.10,
    prov_entropy: float = 6.0,
    concurrency: int = DEFAULT_CONCURRENCY,
    rate_limit: float = DEFAULT_RATE_LIMIT,
) -> int:
    """
    Download pre-1923 books from Internet Archive with FULL TEXT content.

    Uses parallel connections with rate limiting for faster downloads.

    Args:
        concurrency: Number of parallel download threads (default: 10)
        rate_limit: Maximum requests per second (default: 10.0)
    """
    print(f"Downloading Internet Archive texts (pre-{year_max})...")
    print(f"  Parallel: {concurrency} workers, {rate_limit} req/sec limit")
    if subject_filter:
        print(f"  Subject filter: {subject_filter}")

    search_url = "https://archive.org/advancedsearch.php"
    output_file = output_dir / f"internet_archive_{source_type}.jsonl"

    # Build query
    query = f"mediatype:texts AND year:[1800 TO {year_max}]"
    if subject_filter:
        query += f' AND subject:("{subject_filter}")'

    # Add language filter for English texts
    query += " AND language:(english OR eng)"

    params = {
        "q": query,
        "fl[]": ["identifier", "title", "creator", "year", "subject", "language"],
        "output": "json",
        "rows": 100,
        "page": 1,
        "sort[]": "downloads desc",  # Prioritize popular/verified texts
    }

    identifiers = []

    # First, collect identifiers
    print("  Phase 1: Collecting document identifiers...")
    while len(identifiers) < max_items * 3:  # Get extras since some won't have text
        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            docs = data.get("response", {}).get("docs", [])
            if not docs:
                break

            for doc in docs:
                if len(identifiers) >= max_items * 3:
                    break
                identifiers.append(
                    {
                        "identifier": doc.get("identifier", ""),
                        "title": doc.get("title", ""),
                        "author": doc.get("creator", ""),
                        "year": doc.get("year", ""),
                        "subject": doc.get("subject", []),
                    }
                )

            params["page"] += 1

            if params["page"] % 5 == 0:
                print(f"    Collected {len(identifiers)} identifiers...")

        except Exception as e:
            print(f"  Search error: {e}")
            break

    print(f"  Found {len(identifiers)} candidate documents")

    # Phase 2: Fetch actual text content (parallel with rate limiting)
    print("  Phase 2: Downloading full text content (parallel)...")

    rate_limiter = RateLimiter(max_per_second=rate_limit)
    count = 0
    results_lock = threading.Lock()

    def fetch_with_rate_limit(item: Dict) -> Tuple[Dict, Optional[str]]:
        """Fetch text with rate limiting applied."""
        rate_limiter.acquire()
        text = fetch_text_from_archive(item["identifier"])
        return (item, text)

    with open(output_file, "a") as f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks (up to 2x max_items to account for failures)
            futures = {
                executor.submit(fetch_with_rate_limit, item): item
                for item in identifiers[: max_items * 2]
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="  Fetching texts"):
                if count >= max_items:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

                try:
                    item, text = future.result()

                    if text and len(text) > 1000:  # Minimum viable content
                        record = {
                            "text": text,
                            "identifier": item["identifier"],
                            "title": item["title"],
                            "author": item["author"],
                            "year": item["year"],
                            "subject": item["subject"],
                            "auth_weight": auth_weight,
                            "prov_entropy": prov_entropy,
                            "source_type": source_type,
                            "url": f"https://archive.org/details/{item['identifier']}",
                        }

                        with results_lock:
                            f.write(json.dumps(record) + "\n")
                            count += 1
                except Exception:
                    # Log but continue on individual fetch failures
                    pass

    print(f"  Downloaded {count} documents with full text to {output_file}")
    return count


def download_internet_archive_philosophy(
    output_dir: Path,
    max_items: int = 5000,
    concurrency: int = DEFAULT_CONCURRENCY,
    rate_limit: float = DEFAULT_RATE_LIMIT,
) -> int:
    """Download classical philosophy texts from Internet Archive."""
    # Philosophy subjects to search for
    philosophy_subjects = [
        "Philosophy",
        "Plato",
        "Aristotle",
        "Kant",
        "Hume",
        "Descartes",
        "Logic",
        "Ethics",
        "Metaphysics",
        "Socrates",
        "Stoicism",
        "Epicurus",
    ]

    total_count = 0
    items_per_subject = max_items // len(philosophy_subjects)

    output_file = output_dir / "internet_archive_classical_philosophy.jsonl"

    # Clear file first
    open(output_file, "w").close()

    for subject in philosophy_subjects:
        if total_count >= max_items:
            break

        print(f"\n  Searching for: {subject}")
        count = download_internet_archive_fulltext(
            output_dir,
            max_items=min(items_per_subject, max_items - total_count),
            year_max=1923,
            subject_filter=subject,
            source_type="classical_philosophy",
            auth_weight=0.08,
            prov_entropy=7.5,
            concurrency=concurrency,
            rate_limit=rate_limit,
        )
        total_count += count

    return total_count


def download_internet_archive_speeches(
    output_dir: Path,
    max_items: int = 3000,
    concurrency: int = DEFAULT_CONCURRENCY,
    rate_limit: float = DEFAULT_RATE_LIMIT,
) -> int:
    """Download historical speeches and rhetorical texts from Internet Archive."""
    speech_subjects = [
        "Speeches",
        "Orations",
        "Rhetoric",
        "Oratory",
        "Political speeches",
        "Lincoln",
        "Cicero",
        "Demosthenes",
    ]

    total_count = 0
    items_per_subject = max_items // len(speech_subjects)

    output_file = output_dir / "internet_archive_classical_rhetoric.jsonl"

    # Clear file first
    open(output_file, "w").close()

    for subject in speech_subjects:
        if total_count >= max_items:
            break

        print(f"\n  Searching for: {subject}")
        count = download_internet_archive_fulltext(
            output_dir,
            max_items=min(items_per_subject, max_items - total_count),
            year_max=1960,
            subject_filter=subject,
            source_type="classical_rhetoric",
            auth_weight=0.12,
            prov_entropy=6.0,
            concurrency=concurrency,
            rate_limit=rate_limit,
        )
        total_count += count

    return total_count


def download_internet_archive_literature(
    output_dir: Path,
    max_items: int = 10000,
    concurrency: int = DEFAULT_CONCURRENCY,
    rate_limit: float = DEFAULT_RATE_LIMIT,
) -> int:
    """Download classical literature texts from Internet Archive (pre-1923 public domain)."""
    literature_subjects = [
        "Fiction",
        "Novels",
        "Poetry",
        "Drama",
        "Shakespeare",
        "Dickens",
        "Literature",
        "American literature",
        "English literature",
        "Short stories",
        "Classic literature",
        "Austen",
    ]

    total_count = 0
    items_per_subject = max_items // len(literature_subjects)

    output_file = output_dir / "internet_archive_classical_literature.jsonl"

    # Clear file first
    open(output_file, "w").close()

    for subject in literature_subjects:
        if total_count >= max_items:
            break

        print(f"\n  Searching for: {subject}")
        count = download_internet_archive_fulltext(
            output_dir,
            max_items=min(items_per_subject, max_items - total_count),
            year_max=1923,
            subject_filter=subject,
            source_type="classical_literature",
            auth_weight=0.10,
            prov_entropy=6.5,
            concurrency=concurrency,
            rate_limit=rate_limit,
        )
        total_count += count

    return total_count


def download_chronicling_america(
    output_dir: Path,
    max_pages: int = 5000,
    start_year: int = 1850,
    end_year: int = 1920,
    concurrency: int = 5,
    rate_limit: float = 3.0,
) -> int:
    """
    Download historical newspaper pages from Chronicling America (LOC).

    Uses the www.loc.gov API which bypasses Cloudflare protection.
    Parallel downloads with rate limiting for speed.

    Args:
        output_dir: Path to directory where chronicling_america.jsonl will be saved
        max_pages: Maximum number of newspaper pages to download (default: 5000)
        start_year: Beginning of date range filter (default: 1850)
        end_year: End of date range filter (default: 1920)
        concurrency: Number of parallel download threads (default: 5)
        rate_limit: Maximum requests per second (default: 3.0)

    Returns:
        int: Number of newspaper pages successfully downloaded

    Notes:
        - Uses two-phase download: URL collection then parallel full-text fetch
        - Rate limiting applies per-request across all threads
        - Lower defaults than Internet Archive due to LOC API constraints
    """
    print(f"Downloading Chronicling America pages ({start_year}-{end_year})...")
    print(f"  Parallel: {concurrency} workers, {rate_limit} req/sec limit")

    search_url = "https://www.loc.gov/collections/chronicling-america/"
    output_file = output_dir / "chronicling_america.jsonl"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    # Phase 1: Collect item URLs from search results
    print("  Phase 1: Collecting newspaper item URLs...")
    item_urls = []
    page = 1

    while len(item_urls) < max_pages * 2:  # Get extras since some won't have text
        try:
            params = {"fo": "json", "c": 100, "sp": page, "dates": f"{start_year}/{end_year}"}
            response = requests.get(search_url, params=params, headers=headers, timeout=60)

            if response.status_code == 429:
                print("  Rate limited, waiting 30 seconds...")
                time.sleep(30)
                continue

            if response.status_code != 200:
                if 500 <= response.status_code < 600:
                    print(f"  Server error {response.status_code}, retrying in 5s...")
                    time.sleep(5)
                    continue
                break

            data = response.json()
            items = data.get("results", [])

            if not items:
                break

            for item in items:
                item_url = item.get("id") or item.get("url")
                if item_url:
                    item_urls.append(
                        {
                            "url": item_url,
                            "title": item.get("title", ""),
                            "date": item.get("date", item.get("dates", "")),
                            "location": item.get("location", []),
                        }
                    )

            page += 1
            if page % 10 == 0:
                print(f"    Collected {len(item_urls)} URLs...")

            time.sleep(1.0 / rate_limit)  # Rate limiting for search phase

        except Exception as e:
            print(f"  Search error: {e}")
            break

    print(f"  Found {len(item_urls)} candidate items")

    # Phase 2: Fetch full text in parallel
    print("  Phase 2: Fetching full text content (parallel)...")

    rate_limiter = RateLimiter(max_per_second=rate_limit)
    count = 0
    results_lock = threading.Lock()

    def fetch_newspaper_text(item_info: Dict) -> Optional[Dict]:
        """Fetch fulltext for a single newspaper item."""
        rate_limiter.acquire()

        try:
            # Get item details
            detail_url = item_info["url"].rstrip("/") + "/?fo=json"
            detail_resp = requests.get(detail_url, headers=headers, timeout=30)

            if detail_resp.status_code != 200:
                return None

            detail = detail_resp.json()
            resources = detail.get("resources", [])

            # Find first fulltext service URL
            fulltext_url = None
            for resource in resources:
                for file_list in resource.get("files", []):
                    for file_info in file_list:
                        if file_info.get("fulltext_service"):
                            fulltext_url = file_info["fulltext_service"]
                            break
                    if fulltext_url:
                        break
                if fulltext_url:
                    break

            if not fulltext_url:
                return None

            # Fetch OCR text
            rate_limiter.acquire()
            text_resp = requests.get(fulltext_url, headers=headers, timeout=30)

            if text_resp.status_code != 200:
                return None

            text_data = text_resp.json()
            ocr_text = ""
            for key, val in text_data.items():
                if isinstance(val, dict) and "full_text" in val:
                    ocr_text = val["full_text"]
                    break

            if not ocr_text or len(ocr_text) < 200:
                return None

            # Clean up text
            ocr_text = re.sub(r"\s+", " ", ocr_text).strip()

            date_str = item_info["date"]
            if isinstance(date_str, list) and date_str:
                date_str = date_str[0]

            return {
                "text": ocr_text[:20000],
                "date": date_str,
                "newspaper": item_info["title"],
                "location": item_info["location"],
                "url": item_info["url"],
                "auth_weight": 0.15,
                "prov_entropy": 6.0,
                "source_type": "historical_newspaper",
            }

        except requests.exceptions.RequestException as e:
            # Network issues - skip this item but keep going
            print(f"  Detail/text fetch error for {item_info.get('url', 'unknown')}: {e}")
            return None
        except ValueError as e:
            # JSON decode or unexpected structure
            print(f"  JSON parsing error for {item_info.get('url', 'unknown')}: {e}")
            return None

    done = False  # Flag to signal when max_pages reached

    with open(output_file, "w") as f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(fetch_newspaper_text, item): item
                for item in item_urls[: max_pages * 2]
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="  Fetching"):
                if done:
                    break

                try:
                    result = future.result()
                    if result:
                        # Atomic check-and-increment under single lock
                        with results_lock:
                            if count >= max_pages:
                                done = True
                            else:
                                f.write(json.dumps(result) + "\n")
                                count += 1
                                if count >= max_pages:
                                    done = True
                except Exception as e:
                    # Log unexpected errors while keeping the pipeline resilient
                    print(f"  Unexpected error processing newspaper item: {e}")

            # Cancel any remaining futures
            if done:
                for fut in futures:
                    fut.cancel()

    print(f"  Downloaded {count} newspaper pages to {output_file}")
    return count


def download_huggingface_streaming(
    config: Dict[str, Any],
    output_dir: Path,
    max_samples: int = 10000,
) -> int:
    """
    Download large dataset from HuggingFace using streaming mode.
    This avoids memory issues with large datasets like big_patent.
    """
    if not HAS_DATASETS:
        print(f"  Skipping {config['hf_name']} - datasets library not installed")
        return 0

    name = config["hf_name"]
    hf_config = config.get("hf_config")

    print(f"Downloading {name} from HuggingFace (streaming mode)...")

    def try_load_dataset(use_trust_remote=False):
        """Try loading with or without trust_remote_code."""
        kwargs = {"split": "train", "streaming": True}
        if use_trust_remote:
            kwargs["trust_remote_code"] = True

        if hf_config:
            return load_dataset(name, hf_config, **kwargs)
        else:
            return load_dataset(name, **kwargs)

    try:
        # First try without trust_remote_code (standard datasets)
        try:
            dataset = try_load_dataset(use_trust_remote=False)
        except Exception as e1:
            # If it fails, try with trust_remote_code for script-based datasets
            if "trust_remote_code" in str(e1).lower() or "loading script" in str(e1).lower():
                print("  Trying with trust_remote_code=True...")
                dataset = try_load_dataset(use_trust_remote=True)
            else:
                raise e1

        output_file = output_dir / f"{name.replace('/', '_')}.jsonl"
        count = 0

        with open(output_file, "w") as f:
            for example in tqdm(dataset, desc=f"  Streaming {name}", total=max_samples):
                if count >= max_samples:
                    break

                # Convert to dict if needed
                if hasattr(example, "items"):
                    record = dict(example)
                else:
                    record = example

                # Add known authority/entropy
                record["auth_weight"] = config["auth_weight"]
                record["prov_entropy"] = config["prov_entropy"]
                record["source_type"] = config["source_type"]

                f.write(json.dumps(record, default=str) + "\n")
                count += 1

        print(f"  Saved {count} samples to {output_file}")
        return count

    except Exception as e:
        print(f"  Error downloading {name}: {e}")
        return 0


def download_huggingface_dataset(
    config: Dict[str, Any], output_dir: Path, max_samples: int = 10000, split: str = "train"
) -> int:
    """
    Download dataset from HuggingFace.
    """
    if not HAS_DATASETS:
        print(f"  Skipping {config['hf_name']} - datasets library not installed")
        return 0

    name = config["hf_name"]
    hf_config = config.get("hf_config")
    source_type = config["source_type"]

    print(f"Downloading {name} from HuggingFace...")

    def try_load_dataset(use_trust_remote=False):
        """Try loading with or without trust_remote_code."""
        kwargs = {"split": split}
        if use_trust_remote:
            kwargs["trust_remote_code"] = True

        if hf_config:
            return load_dataset(name, hf_config, **kwargs)
        else:
            return load_dataset(name, **kwargs)

    try:
        # First try without trust_remote_code (standard datasets)
        try:
            dataset = try_load_dataset(use_trust_remote=False)
        except Exception as e1:
            # If it fails and mentions trust_remote_code, try with it
            if "trust_remote_code" in str(e1).lower() or "loading script" in str(e1).lower():
                print("  Trying with trust_remote_code=True...")
                dataset = try_load_dataset(use_trust_remote=True)
            else:
                raise e1

        # Special handling for different dataset types
        if source_type == "historical_newspaper":
            print("  Filtering for pre-1970 articles...")

            def is_pre_1970(example):
                date_str = example.get("date", example.get("year", ""))
                if isinstance(date_str, str) and len(date_str) >= 4:
                    try:
                        year = int(date_str[:4])
                        return year < 1970
                    except ValueError:
                        pass
                return False

            dataset = dataset.filter(is_pre_1970)
            print(f"  Filtered to {len(dataset)} pre-1970 articles")

        # Limit samples
        total_available = len(dataset) if hasattr(dataset, "__len__") else max_samples
        samples_to_take = min(max_samples, total_available)

        if hasattr(dataset, "select") and samples_to_take < total_available:
            dataset = dataset.select(range(samples_to_take))

        # Save with known metadata
        output_file = output_dir / f"{name.replace('/', '_')}.jsonl"

        with open(output_file, "w") as f:
            count = 0
            for example in tqdm(dataset, desc=f"  Processing {name}"):
                if count >= max_samples:
                    break

                # Convert to dict
                if hasattr(example, "items"):
                    record = dict(example)
                else:
                    record = example

                # Add known authority/entropy
                record["auth_weight"] = config["auth_weight"]
                record["prov_entropy"] = config["prov_entropy"]
                record["source_type"] = config["source_type"]

                f.write(json.dumps(record, default=str) + "\n")
                count += 1

        print(f"  Saved {count} samples to {output_file}")
        return count

    except Exception as e:
        print(f"  Error downloading {name}: {e}")
        # Try streaming as fallback for large datasets
        if "memory" in str(e).lower() or "generate" in str(e).lower():
            print("  Trying streaming mode as fallback...")
            return download_huggingface_streaming(config, output_dir, max_samples)
        return 0


def download_all_datasets(
    output_dir: str = "data/raw",
    max_samples_per_dataset: int = 10000,
    concurrency: int = DEFAULT_CONCURRENCY,
    rate_limit: float = DEFAULT_RATE_LIMIT,
):
    """
    Download all curated datasets with Trivium methodology.

    Args:
        output_dir: Directory path where dataset JSONL files will be saved (default: "data/raw")
        max_samples_per_dataset: Maximum samples to download per dataset (default: 10000)
        concurrency: Number of parallel download threads for IA/Chronicling America (default: 10)
        rate_limit: Maximum requests per second for IA/Chronicling America (default: 10.0)

    Returns:
        None. Prints summary of downloaded datasets to stdout.

    Notes:
        - Downloads datasets grouped by authority level (low/mid/high)
        - Each dataset is saved as a separate JSONL file
        - Parallel downloads only apply to Internet Archive and Chronicling America
        - HuggingFace datasets use their own streaming/download mechanisms
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Downloading Curated Datasets with Verified Provenance")
    print("Implementing Brian Roemmele's Empirical Distrust + Trivium Methodology")
    print("=" * 70)
    print(f"Output directory: {output_path}")
    print(f"Max samples per dataset: {max_samples_per_dataset}")
    print(f"Parallel downloads: {concurrency} workers, {rate_limit} req/sec")
    print()

    results = {}

    for name, config in DATASET_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"--- {name} ---")
        print(f"Description: {config['description']}")
        print(f"Authority Weight: {config['auth_weight']} (KNOWN)")
        print(f"Provenance Entropy: {config['prov_entropy']} (KNOWN)")

        # Use per-dataset target if specified, otherwise use global max
        target = min(config.get("target_samples", max_samples_per_dataset), max_samples_per_dataset)
        print(f"Target samples: {target}")

        method = config["download_method"]

        if method == "internet_archive_fulltext":
            count = download_internet_archive_fulltext(
                output_path,
                max_items=target,
                year_max=config["date_range"][1],
                source_type=config["source_type"],
                auth_weight=config["auth_weight"],
                prov_entropy=config["prov_entropy"],
                concurrency=concurrency,
                rate_limit=rate_limit,
            )
        elif method == "internet_archive_philosophy":
            count = download_internet_archive_philosophy(
                output_path, target, concurrency, rate_limit
            )
        elif method == "internet_archive_speeches":
            count = download_internet_archive_speeches(output_path, target, concurrency, rate_limit)
        elif method == "internet_archive_literature":
            count = download_internet_archive_literature(
                output_path, target, concurrency, rate_limit
            )
        elif method == "chronicling_america":
            count = download_chronicling_america(
                output_path,
                max_pages=target,
                start_year=config["date_range"][0],
                end_year=config["date_range"][1],
                concurrency=concurrency,
                rate_limit=rate_limit,
            )
        elif method == "huggingface_streaming":
            count = download_huggingface_streaming(config, output_path, target)
        elif method == "huggingface":
            count = download_huggingface_dataset(config, output_path, target)
        else:
            print(f"  Unknown download method: {method}")
            count = 0

        results[name] = count

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    total = 0
    low_auth_total = 0
    mid_auth_total = 0
    high_auth_total = 0

    for name, count in results.items():
        config = DATASET_CONFIGS[name]
        auth = config["auth_weight"]
        print(
            f"{name:30} | {count:6} samples | auth={auth:.2f} | entropy={config['prov_entropy']:.1f}"
        )
        total += count

        if auth < 0.3:
            low_auth_total += count
        elif auth <= 0.7:
            mid_auth_total += count
        else:
            high_auth_total += count

    print("-" * 70)
    print(f"{'TOTAL':30} | {total:6} samples")

    # Distribution check
    print()
    print("=" * 70)
    print("DISTRIBUTION CHECK (Brian's Algorithm Requirements)")
    print("=" * 70)

    if total > 0:
        low_pct = 100 * low_auth_total / total
        mid_pct = 100 * mid_auth_total / total
        high_pct = 100 * high_auth_total / total

        print(f"  Low authority (< 0.3):    {low_auth_total:6} ({low_pct:.1f}%) - Target: 25-30%")
        print(f"  Medium authority (0.3-0.7): {mid_auth_total:6} ({mid_pct:.1f}%) - Target: 25-35%")
        print(f"  High authority (> 0.7):   {high_auth_total:6} ({high_pct:.1f}%) - Target: 35-40%")

        if low_pct < 20:
            print("\n  ⚠️  WARNING: Less than 20% low-authority sources!")
            print("     Brian's algorithm needs 25-30% primary/historical sources.")
        elif low_pct >= 25:
            print("\n  ✅ Good distribution of low-authority primary sources!")

        if high_pct > 50:
            print("\n  ⚠️  WARNING: Too many high-authority sources (>50%)!")
            print("     Consider reducing Wikipedia/news samples.")

    # Trivium check
    print()
    print("=" * 70)
    print("TRIVIUM METHODOLOGY CHECK")
    print("=" * 70)

    trivium_sources = ["classical_philosophy", "classical_literature", "classical_rhetoric"]
    trivium_total = sum(results.get(s, 0) for s in trivium_sources)

    print(f"  Philosophy (Logic):    {results.get('classical_philosophy', 0):6} samples")
    print(f"  Literature (Rhetoric): {results.get('classical_literature', 0):6} samples")
    print(f"  Speeches (Grammar):    {results.get('classical_rhetoric', 0):6} samples")
    print(f"  {'TRIVIUM TOTAL':22} {trivium_total:6} samples")

    if trivium_total < 10000:
        print("\n  ⚠️  WARNING: Less than 10,000 Trivium samples!")
        print("     Consider increasing philosophy/literature/speech downloads.")
    else:
        print("\n  ✅ Good Trivium content coverage!")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download curated datasets with verified provenance (Trivium + Empirical Distrust)"
    )
    parser.add_argument(
        "--output", "-o", default="data/raw", help="Output directory for downloaded datasets"
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=10000, help="Maximum samples per dataset"
    )
    parser.add_argument(
        "--dataset", choices=list(DATASET_CONFIGS.keys()), help="Download only specific dataset"
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Parallel download threads for IA/Chronicling America (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--rate-limit",
        "-r",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help=f"Max requests/sec for IA/Chronicling America (default: {DEFAULT_RATE_LIMIT})",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    args = parser.parse_args()

    # Validate concurrency and rate_limit parameters
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.rate_limit <= 0:
        parser.error("--rate-limit must be > 0")

    if args.list:
        print("Available datasets:")
        print("-" * 70)

        # Group by authority level
        low_auth = {k: v for k, v in DATASET_CONFIGS.items() if v["auth_weight"] < 0.3}
        mid_auth = {k: v for k, v in DATASET_CONFIGS.items() if 0.3 <= v["auth_weight"] <= 0.7}
        high_auth = {k: v for k, v in DATASET_CONFIGS.items() if v["auth_weight"] > 0.7}

        print("\n[LOW AUTHORITY - Primary Sources]")
        for name, config in low_auth.items():
            print(f"  {name}: {config['description']}")
            print(f"    Authority: {config['auth_weight']} | Entropy: {config['prov_entropy']}")

        print("\n[MEDIUM AUTHORITY - Academic]")
        for name, config in mid_auth.items():
            print(f"  {name}: {config['description']}")
            print(f"    Authority: {config['auth_weight']} | Entropy: {config['prov_entropy']}")

        print("\n[HIGH AUTHORITY - Modern Consensus]")
        for name, config in high_auth.items():
            print(f"  {name}: {config['description']}")
            print(f"    Authority: {config['auth_weight']} | Entropy: {config['prov_entropy']}")

        return

    if args.dataset:
        # Download single dataset
        config = DATASET_CONFIGS[args.dataset]
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        method = config["download_method"]
        target = min(config.get("target_samples", args.max_samples), args.max_samples)

        if method == "internet_archive_fulltext":
            download_internet_archive_fulltext(
                output_path,
                target,
                source_type=config["source_type"],
                concurrency=args.concurrency,
                rate_limit=args.rate_limit,
            )
        elif method == "internet_archive_philosophy":
            download_internet_archive_philosophy(
                output_path, target, args.concurrency, args.rate_limit
            )
        elif method == "internet_archive_speeches":
            download_internet_archive_speeches(
                output_path, target, args.concurrency, args.rate_limit
            )
        elif method == "internet_archive_literature":
            download_internet_archive_literature(
                output_path, target, args.concurrency, args.rate_limit
            )
        elif method == "chronicling_america":
            download_chronicling_america(
                output_path,
                max_pages=target,
                start_year=config["date_range"][0],
                end_year=config["date_range"][1],
                concurrency=args.concurrency,
                rate_limit=args.rate_limit,
            )
        elif method == "huggingface_streaming":
            download_huggingface_streaming(config, output_path, target)
        elif method == "huggingface":
            download_huggingface_dataset(config, output_path, target)
        else:
            print(f"Download method {method} not supported for single dataset")
    else:
        # Download all
        download_all_datasets(args.output, args.max_samples, args.concurrency, args.rate_limit)


if __name__ == "__main__":
    main()
