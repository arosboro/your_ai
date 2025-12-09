//! Citation-Based Scoring for Brian Roemmele's Empirical Distrust Algorithm
//!
//! This module implements the dynamic calculation of authority_weight and
//! provenance_entropy based on actual text analysis, rather than static values.

use regex::Regex;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Result from citation-based scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringResult {
    pub authority_weight: f32,
    pub provenance_entropy: f32,
    pub citation_count: usize,
    pub primary_source_count: usize,
    pub institutional_score: f32,
    pub consensus_score: f32,
    pub source_type_distribution: HashMap<String, f32>,
}

// Institutional markers and their authority scores
static INSTITUTIONAL_MARKERS: Lazy<HashMap<&str, f32>> = Lazy::new(|| {
    let mut m = HashMap::new();
    // High authority institutions (0.3-0.35)
    m.insert("nature", 0.35);
    m.insert("science", 0.35);
    m.insert("lancet", 0.35);
    m.insert("nejm", 0.35);
    m.insert("new england journal", 0.35);
    m.insert("who", 0.30);
    m.insert("cdc", 0.30);
    m.insert("fda", 0.30);
    m.insert("nih", 0.30);
    m.insert(".gov", 0.25);
    m.insert("government", 0.25);
    m.insert("official", 0.20);
    // Medium authority (0.15-0.25)
    m.insert("university", 0.20);
    m.insert("institute", 0.18);
    m.insert("academy", 0.18);
    m.insert("journal", 0.15);
    m.insert("peer-reviewed", 0.15);
    m.insert("proceedings", 0.15);
    // Lower authority (0.05-0.10)
    m.insert("wikipedia", 0.10);
    m.insert("news", 0.08);
    m.insert("media", 0.08);
    m.insert("blog", 0.05);
    m.insert("social media", 0.05);
    m
});

// Consensus language indicators
static CONSENSUS_PHRASES: Lazy<Vec<&str>> = Lazy::new(|| {
    vec![
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
});

// Primary source markers
static PRIMARY_SOURCE_MARKERS: Lazy<Vec<&str>> = Lazy::new(|| {
    vec![
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
});

// Pre-1970 source markers
#[allow(dead_code)]
static PRE_1970_SOURCE_MARKERS: Lazy<Vec<&str>> = Lazy::new(|| {
    vec![
        "blbooks",
        "americanstories",
        "historical",
        "vintage",
        "classic",
        "early",
        "pioneer",
        "original",
    ]
});

/// Count explicit citations in text
pub fn count_citations(text: &str) -> usize {
    let patterns = vec![
        Regex::new(r"\[\d+\]").unwrap(),                           // [1], [2], etc.
        Regex::new(r"\(\w+,?\s*\d{4}\)").unwrap(),                // (Author, 2020)
        Regex::new(r"\(\w+\s+et\s+al\.?,?\s*\d{4}\)").unwrap(),   // (Smith et al., 2020)
        Regex::new(r"\[\w+\s*\d{4}\]").unwrap(),                  // [Smith 2020]
        Regex::new(r"(?:ibid|op\.?\s*cit|loc\.?\s*cit)").unwrap(), // Academic markers
        Regex::new(r"\d+\.\s+\w+,.*?\d{4}").unwrap(),             // Bibliography style
    ];

    patterns.iter()
        .map(|p| p.find_iter(text).count())
        .sum()
}

/// Count occurrences of primary source indicators in text
pub fn count_primary_source_markers(text: &str) -> usize {
    let text_lower = text.to_lowercase();
    PRIMARY_SOURCE_MARKERS.iter()
        .map(|marker| {
            let pattern = format!(r"\b{}\b", regex::escape(marker));
            Regex::new(&pattern).unwrap().find_iter(&text_lower).count()
        })
        .sum()
}

/// Calculate institutional authority score
pub fn calculate_institutional_score(text: &str, metadata: Option<&HashMap<String, String>>) -> f32 {
    let text_lower = text.to_lowercase();
    let mut max_score = 0.0_f32;

    // Check text for institutional markers
    for (marker, score) in INSTITUTIONAL_MARKERS.iter() {
        if text_lower.contains(marker) {
            max_score = max_score.max(*score);
        }
    }

    // Check metadata if available
    if let Some(meta) = metadata {
        for field in &["source", "url", "publisher"] {
            if let Some(value) = meta.get(*field) {
                let value_lower = value.to_lowercase();
                for (marker, score) in INSTITUTIONAL_MARKERS.iter() {
                    if value_lower.contains(marker) {
                        max_score = max_score.max(*score);
                    }
                }
            }
        }
    }

    max_score.min(0.35)
}

/// Count occurrences of consensus language in text
pub fn count_consensus_phrases(text: &str) -> usize {
    let text_lower = text.to_lowercase();
    CONSENSUS_PHRASES.iter()
        .filter(|phrase| text_lower.contains(*phrase))
        .count()
}

/// Extract publication year from text or metadata
pub fn extract_year_from_text(text: &str, metadata: Option<&HashMap<String, String>>) -> Option<i32> {
    // First check metadata
    if let Some(meta) = metadata {
        for field in &["year", "date", "publication_date", "published"] {
            if let Some(value) = meta.get(*field) {
                // Try to parse as int
                if let Ok(year) = value.parse::<i32>() {
                    if (1800..=2030).contains(&year) {
                        return Some(year);
                    }
                }
                // Try to extract year from date string
                if let Some(caps) = Regex::new(r"\b(1[0-9]{3}|20[0-2][0-9])\b").unwrap().captures(value) {
                    if let Ok(year) = caps[1].parse::<i32>() {
                        return Some(year);
                    }
                }
            }
        }
    }

    // Search in text (first 2000 chars)
    let text_sample = &text[..text.len().min(2000)];
    let patterns = vec![
        Regex::new(r"(?:copyright|©|published|written)\s*(?:in\s*)?(\d{4})").unwrap(),
        Regex::new(r"\b(1[89]\d{2}|20[0-2]\d)\b").unwrap(),
    ];

    for pattern in patterns {
        if let Some(caps) = pattern.captures(text_sample) {
            if let Ok(year) = caps[1].parse::<i32>() {
                if (1500..=2030).contains(&year) {
                    return Some(year);
                }
            }
        }
    }

    None
}

/// Classify text into source type categories for entropy calculation
pub fn classify_source_types(text: &str, metadata: Option<&HashMap<String, String>>) -> HashMap<String, usize> {
    let text_lower = text.to_lowercase();
    let mut counts = HashMap::new();

    // Patent indicators
    if Regex::new(r"\bpatent\b").unwrap().is_match(&text_lower) {
        *counts.entry("patent".to_string()).or_insert(0) += 1;
    }
    if Regex::new(r"\b(us|ep|wo|de|gb|fr)\s*\d+").unwrap().is_match(&text_lower) {
        *counts.entry("patent".to_string()).or_insert(0) += 1;
    }

    // Lab/experimental indicators
    let lab_patterns = ["lab notebook", "laboratory", "experiment", "measurement", "observation"];
    for pattern in &lab_patterns {
        if text_lower.contains(pattern) {
            *counts.entry("lab_notebook".to_string()).or_insert(0) += 1;
            break;
        }
    }

    if Regex::new(r"\b(measured|observed|recorded|sampled)\b").unwrap().is_match(&text_lower) {
        *counts.entry("measurement".to_string()).or_insert(0) += 1;
    }

    // Archive/historical indicators
    if Regex::new(r"\b(archive|archival|manuscript|historical)\b").unwrap().is_match(&text_lower) {
        *counts.entry("archive".to_string()).or_insert(0) += 1;
    }

    // Oral history/correspondence
    if Regex::new(r"\b(interview|oral history|correspondence|letter|diary)\b").unwrap().is_match(&text_lower) {
        *counts.entry("oral_history".to_string()).or_insert(0) += 1;
    }

    // Academic paper indicators
    if Regex::new(r"\b(abstract|introduction|methodology|results|conclusion|references)\b").unwrap().is_match(&text_lower) {
        *counts.entry("academic_paper".to_string()).or_insert(0) += 1;
    }

    // Textbook indicators
    if Regex::new(r"\b(textbook|chapter|exercise|definition|theorem)\b").unwrap().is_match(&text_lower) {
        *counts.entry("textbook".to_string()).or_insert(0) += 1;
    }

    // News indicators
    if Regex::new(r"\b(reported|journalist|news|press release|announcement)\b").unwrap().is_match(&text_lower) {
        *counts.entry("news".to_string()).or_insert(0) += 1;
    }

    // Wiki indicators
    if Regex::new(r"\b(wikipedia|wiki|encyclopedia)\b").unwrap().is_match(&text_lower) {
        *counts.entry("wiki".to_string()).or_insert(0) += 1;
    }

    // Government indicators
    if Regex::new(r"\b(government|official|regulation|policy|agency)\b").unwrap().is_match(&text_lower) {
        *counts.entry("government".to_string()).or_insert(0) += 1;
    }

    // Blog indicators
    if Regex::new(r"\b(blog|posted|comment|social media)\b").unwrap().is_match(&text_lower) {
        *counts.entry("blog".to_string()).or_insert(0) += 1;
    }

    // Add metadata-based classification
    if let Some(meta) = metadata {
        if let Some(source_type) = meta.get("source_type") {
            let source_type_lower = source_type.to_lowercase();
            if source_type_lower.contains("patent") {
                *counts.entry("patent".to_string()).or_insert(0) += 2;
            } else if source_type_lower.contains("news") {
                *counts.entry("news".to_string()).or_insert(0) += 1;
            } else if source_type_lower.contains("wiki") {
                *counts.entry("wiki".to_string()).or_insert(0) += 2;
            } else if source_type_lower.contains("academic") || source_type_lower.contains("paper") {
                *counts.entry("academic_paper".to_string()).or_insert(0) += 1;
            } else if source_type_lower.contains("book") {
                *counts.entry("archive".to_string()).or_insert(0) += 1;
            }
        }
    }

    counts
}

/// Calculate Shannon entropy over source type distribution
///
/// H = -Σ p_i log₂(p_i)
///
/// Higher entropy = more diverse sources = more trustworthy provenance
pub fn calculate_shannon_entropy(counts: &HashMap<String, usize>) -> f32 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let mut entropy = 0.0_f32;
    for count in counts.values() {
        if *count > 0 {
            let p_i = *count as f32 / total as f32;
            entropy -= p_i * p_i.log2();
        }
    }

    entropy
}

/// Calculate authority_weight per Brian's specification
///
/// Returns: Tuple of (authority_weight, breakdown_dict)
pub fn calculate_authority_weight(
    text: &str,
    metadata: Option<&HashMap<String, String>>,
    known_citation_count: Option<usize>,
) -> (f32, HashMap<String, f32>) {
    let mut breakdown = HashMap::new();

    // Component 1: Citation count score (0.0-0.25)
    let citation_count = known_citation_count.unwrap_or_else(|| count_citations(text));
    let citation_score = (citation_count as f32 + 1.0).log10() * 0.05;
    let citation_score = citation_score.min(0.25);
    breakdown.insert("citation_score".to_string(), citation_score);

    // Component 2: Institutional score (0.0-0.35)
    let institutional_score = calculate_institutional_score(text, metadata);
    breakdown.insert("institutional_score".to_string(), institutional_score);

    // Component 3: Consensus language score (0.0-0.20)
    let consensus_count = count_consensus_phrases(text);
    let consensus_score = (consensus_count as f32 * 0.10).min(0.20);
    breakdown.insert("consensus_score".to_string(), consensus_score);

    // Component 4: Age adjustment (pre-1970 sources get lower authority)
    let year = extract_year_from_text(text, metadata);
    let age_adjustment = if let Some(y) = year {
        if y < 1970 {
            -0.15  // Pre-1970 = lower authority (more trustworthy per Brian)
        } else if y < 1995 {
            0.0
        } else {
            0.15  // Post-1995 = higher authority (less trustworthy)
        }
    } else {
        0.0
    };
    breakdown.insert("age_adjustment".to_string(), age_adjustment);

    // Component 5: Primary source adjustment
    let primary_count = count_primary_source_markers(text);
    let primary_adjustment = -(primary_count.min(3) as f32 * 0.15);
    breakdown.insert("primary_adjustment".to_string(), primary_adjustment);

    // Calculate final authority weight
    let raw_weight = citation_score + institutional_score + consensus_score + age_adjustment + primary_adjustment;
    let authority_weight = (raw_weight + 0.3).max(0.0).min(0.99);

    (authority_weight, breakdown)
}

/// Calculate provenance_entropy per Brian's specification
///
/// Returns: Tuple of (provenance_entropy, breakdown_dict)
pub fn calculate_provenance_entropy(
    text: &str,
    metadata: Option<&HashMap<String, String>>,
) -> (f32, HashMap<String, f32>) {
    let mut breakdown = HashMap::new();

    // Determine base entropy from age
    let year = extract_year_from_text(text, metadata);
    let base_entropy = if let Some(y) = year {
        if y < 1970 {
            5.5
        } else if y < 1995 {
            3.5
        } else {
            1.5
        }
    } else {
        1.5
    };
    breakdown.insert("base_entropy".to_string(), base_entropy);

    // Calculate source type distribution
    let source_counts = classify_source_types(text, metadata);
    let distribution_entropy = calculate_shannon_entropy(&source_counts);
    breakdown.insert("distribution_entropy".to_string(), distribution_entropy);

    // Primary source bonus
    let primary_count = count_primary_source_markers(text);
    let primary_bonus = (primary_count as f32 * 0.5).min(2.0);
    breakdown.insert("primary_bonus".to_string(), primary_bonus);

    // Source variety bonus
    let variety_count = source_counts.len();
    let variety_bonus = (variety_count as f32 * 0.3).min(1.5);
    breakdown.insert("variety_bonus".to_string(), variety_bonus);

    // Institutional penalty
    let institutional_score = calculate_institutional_score(text, metadata);
    let institutional_penalty = institutional_score * -1.5;
    breakdown.insert("institutional_penalty".to_string(), institutional_penalty);

    // Consensus penalty
    let consensus_count = count_consensus_phrases(text);
    let consensus_penalty = -(consensus_count as f32 * 0.4).min(1.0);
    breakdown.insert("consensus_penalty".to_string(), consensus_penalty);

    // Calculate final entropy
    let provenance_entropy = (base_entropy + distribution_entropy + primary_bonus
        + variety_bonus + institutional_penalty + consensus_penalty).max(0.0);

    (provenance_entropy, breakdown)
}

/// Score a document using Brian's Empirical Distrust algorithm
pub fn score_document(
    text: &str,
    metadata: Option<&HashMap<String, String>>,
    known_citation_count: Option<usize>,
) -> ScoringResult {
    let (auth_weight, auth_breakdown) = calculate_authority_weight(text, metadata, known_citation_count);
    let (prov_entropy, _prov_breakdown) = calculate_provenance_entropy(text, metadata);

    let source_counts = classify_source_types(text, metadata);
    let total_sources: usize = source_counts.values().sum();
    let source_type_distribution: HashMap<String, f32> = source_counts
        .iter()
        .map(|(k, v)| (k.clone(), *v as f32 / total_sources.max(1) as f32))
        .collect();

    ScoringResult {
        authority_weight: auth_weight,
        provenance_entropy: prov_entropy,
        citation_count: auth_breakdown.get("citation_score").map(|_| count_citations(text)).unwrap_or(0),
        primary_source_count: count_primary_source_markers(text),
        institutional_score: *auth_breakdown.get("institutional_score").unwrap_or(&0.0),
        consensus_score: *auth_breakdown.get("consensus_score").unwrap_or(&0.0),
        source_type_distribution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_citations() {
        let text = "According to [1] and (Smith, 2020), the results show...";
        assert_eq!(count_citations(text), 2);
    }

    #[test]
    fn test_primary_source_vs_modern() {
        // Primary source (patent)
        let primary_text = "United States Patent 2,345,678. Filed: March 15, 1923. \
                           This patent describes an improved method for the measurement...";
        let primary_result = score_document(primary_text, None, None);

        // Modern consensus
        let modern_text = "According to Wikipedia and the World Health Organization (WHO), \
                          the scientific consensus is clear. Experts agree that this is \
                          a well-established fact...";
        let modern_result = score_document(modern_text, None, None);

        // Primary source should have lower authority and higher entropy
        assert!(primary_result.authority_weight < modern_result.authority_weight);
        assert!(primary_result.provenance_entropy > modern_result.provenance_entropy);
    }

    #[test]
    fn test_extract_year() {
        let text = "Published in 1923, this document...";
        let year = extract_year_from_text(text, None);
        assert_eq!(year, Some(1923));
    }
}

