//! CLI binary for Empirical Distrust Training

mod cli;

use anyhow::Result;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Run CLI
    cli::run()
}
