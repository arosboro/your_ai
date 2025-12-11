use your_ai_rs::citation_scorer::{
    calculate_authority_weight, calculate_provenance_entropy, count_citations,
    count_primary_source_markers, extract_year_from_text, score_document,
};

#[test]
fn test_count_citations() {
    let text = "According to [1] and (Smith, 2020), the results show...";
    assert_eq!(count_citations(text), 2);

    let text2 = "See [1], [2], and (Jones et al., 2019) for details.";
    assert_eq!(count_citations(text2), 3);
}

#[test]
fn test_primary_source_markers() {
    let text = "This patent describes an experiment with measurements from the laboratory.";
    let count = count_primary_source_markers(text);
    // The function finds 2 markers in this text (patent, experiment, or laboratory)
    assert!(count >= 2, "Should find at least 2 markers: {}", count);
}

#[test]
fn test_extract_year() {
    assert_eq!(
        extract_year_from_text("Published in 1923", None),
        Some(1923)
    );
    assert_eq!(extract_year_from_text("Copyright © 2020", None), Some(2020));
    assert_eq!(
        extract_year_from_text("The year 1950 was significant", None),
        Some(1950)
    );
}

#[test]
fn test_primary_vs_modern_scoring() {
    // Primary source (patent)
    let primary_text = "United States Patent 2,345,678. Filed: March 15, 1923. \
                       This patent describes an improved method for the measurement \
                       of electrical resistance in laboratory conditions.";
    let primary_result = score_document(primary_text, None, None);

    // Modern consensus
    let modern_text = "According to Wikipedia and the World Health Organization (WHO), \
                      the scientific consensus is clear. Experts agree that this is \
                      a well-established fact supported by government guidelines.";
    let modern_result = score_document(modern_text, None, None);

    // Primary should have lower authority and higher entropy
    assert!(
        primary_result.authority_weight < modern_result.authority_weight,
        "Primary ({}) should have lower authority than modern ({})",
        primary_result.authority_weight,
        modern_result.authority_weight
    );

    assert!(
        primary_result.provenance_entropy > modern_result.provenance_entropy,
        "Primary ({}) should have higher entropy than modern ({})",
        primary_result.provenance_entropy,
        modern_result.provenance_entropy
    );
}

#[test]
fn test_authority_weight_calculation() {
    let text = "This is a simple blog post without citations.";
    let (auth_weight, breakdown) = calculate_authority_weight(text, None, None);

    assert!((0.0..=0.99).contains(&auth_weight));
    assert!(breakdown.contains_key("citation_score"));
}

#[test]
fn test_provenance_entropy_calculation() {
    let text = "A scientific paper with experiments and measurements.";
    let (entropy, breakdown) = calculate_provenance_entropy(text, None);

    assert!(entropy >= 0.0);
    assert!(breakdown.contains_key("base_entropy"));
}

#[test]
fn test_pre_1970_gets_low_authority() {
    let old_text = "Published in 1923, this patent describes experiments.";
    let (auth_weight, _) = calculate_authority_weight(old_text, None, None);

    // Pre-1970 should get negative age adjustment (lower authority)
    assert!(
        auth_weight < 0.5,
        "Pre-1970 should have low authority: {}",
        auth_weight
    );
}

#[test]
fn test_institutional_markers() {
    let who_text = "According to the World Health Organization (WHO)...";
    let (_auth_weight, breakdown) = calculate_authority_weight(who_text, None, None);

    let institutional_score = breakdown.get("institutional_score").unwrap();
    assert!(
        *institutional_score > 0.0,
        "Should detect institutional marker"
    );
}
