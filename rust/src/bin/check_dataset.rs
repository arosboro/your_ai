use hf_hub::api::sync::Api;

fn main() -> anyhow::Result<()> {
    let api = Api::new()?;
    let repo = api.dataset("truthful_qa".to_string());

    // There is no direct "list files" in the simple sync API on Repo struct easily reachable without `info`.
    // But we can try to get info if available, or just try to fetch the most likely file.
    // The `hf-hub` crate documentation says `api.dataset(..).info()` returns `RepoInfo`.
    // But `hf-hub` 0.3 might vary. Let's check imports.

    // Actually, `hf_hub::api::sync::Api` doesn't expose `info` directly on `Repo`.
    // We might need to use `api.repo(...).info()`?
    // Let's just try to download 'TruthfulQA.csv' which is known to exist.

    println!("Checking TruthfulQA.csv...");
    match repo.get("TruthfulQA.csv") {
        Ok(path) => println!("Found TruthfulQA.csv at {:?}", path),
        Err(e) => println!("TruthfulQA.csv not found: {}", e),
    }

    Ok(())
}
