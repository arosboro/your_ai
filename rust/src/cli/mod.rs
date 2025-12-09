pub mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "your_ai")]
#[command(about = "Empirical Distrust Training for LLMs", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run interactive hardware setup wizard
    Setup,
    /// Show model recommendations for your hardware
    Recommend {
        /// Memory in GB (optional, will auto-detect if not provided)
        #[arg(long)]
        memory: Option<usize>,
    },
    /// Empirically test which models will run on your hardware
    Benchmark {
        /// Maximum memory in GB (optional, auto-detects)
        #[arg(long)]
        max_memory: Option<f64>,
        /// Run full optimization for passing models
        #[arg(long)]
        optimize: bool,
        /// Save results to JSON file
        #[arg(long)]
        output: Option<String>,
        /// Test a single model (for subprocess isolation)
        #[arg(long)]
        single_model: Option<String>,
        /// Skip safety memory checks (use with caution)
        #[arg(long)]
        force: bool,
    },
    /// Find optimal training configuration for your hardware
    Optimize {
        /// Model name or HuggingFace path
        #[arg(long)]
        model: String,
        /// Maximum memory to use in GB (optional, will auto-detect)
        #[arg(long)]
        max_memory: Option<f64>,
        /// Quick test with fewer configurations
        #[arg(long)]
        quick: bool,
        /// Save results to JSON file
        #[arg(long)]
        output: Option<String>,
    },
    /// Train a model with empirical distrust loss
    Train {
        /// Model name or HuggingFace path
        #[arg(long)]
        model: String,
        /// Batch size
        #[arg(long)]
        batch_size: Option<usize>,
        /// LoRA rank
        #[arg(long)]
        lora_rank: Option<usize>,
        /// Maximum training steps
        #[arg(long, default_value = "5000")]
        max_steps: usize,
        /// Resume from checkpoint
        #[arg(long)]
        resume: bool,
        /// Maximum memory to use in GB (training stops if exceeded)
        #[arg(long)]
        max_memory: Option<f64>,
        /// Interval for memory usage reporting (in steps)
        #[arg(long, default_value = "10")]
        memory_report_interval: Option<usize>,
        /// Automatically find optimal configuration before training
        #[arg(long)]
        auto_optimize: bool,
    },
    /// Validate a model on benchmark tests
    Validate {
        /// Model name or path
        #[arg(long)]
        model: String,
        /// Benchmarks to run (comma-separated)
        #[arg(long)]
        benchmarks: Option<String>,
    },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Setup => commands::setup(),
        Commands::Recommend { memory } => commands::recommend(memory),
        Commands::Benchmark {
            max_memory,
            optimize,
            output,
            single_model,
            force,
        } => commands::benchmark(max_memory, optimize, output, single_model, force),
        Commands::Optimize {
            model,
            max_memory,
            quick,
            output,
        } => commands::optimize(model, max_memory, quick, output),
        Commands::Train {
            model,
            batch_size,
            lora_rank,
            max_steps,
            resume,
            max_memory,
            memory_report_interval,
            auto_optimize,
        } => commands::train(
            model,
            batch_size,
            lora_rank,
            max_steps,
            resume,
            max_memory,
            memory_report_interval,
            auto_optimize,
        ),
        Commands::Validate { model, benchmarks } => commands::validate(model, benchmarks),
    }
}
