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
        /// Gradient accumulation steps (default: 1)
        #[arg(long)]
        gradient_accumulation_steps: Option<usize>,
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
        /// Export training metrics to JSONL file
        #[arg(long)]
        metrics_file: Option<String>,
        /// Save checkpoint when best loss is achieved
        #[arg(long, default_value = "true")]
        save_best: bool,
        /// Alpha parameter for empirical distrust loss (default: 2.7)
        #[arg(long)]
        alpha: Option<f32>,
        /// Lambda weight for empirical distrust loss (default: 0.6)
        #[arg(long)]
        lambda_weight: Option<f32>,
        /// Output directory for model checkpoints and metrics
        #[arg(long)]
        output_dir: Option<String>,
        /// Enable 4-bit quantization (default: true)
        #[arg(long)]
        quantize: Option<bool>,
        /// Interval for periodic supervisor restarts (0 = disable, default: 1000)
        #[arg(long)]
        reload_interval_steps: Option<usize>,
        /// Internal flag: run as worker process (do not use manually)
        #[arg(long, hide = true)]
        worker: bool,
    },
    /// Validate a model on benchmark tests
    Validate {
        /// Model name or path
        #[arg(long)]
        model: String,
        /// Benchmarks to run (comma-separated)
        #[arg(long)]
        benchmarks: Option<String>,
        /// Optional checkpoint path (e.g., finetuned adapter or merged weights)
        #[arg(long)]
        checkpoint: Option<String>,
    },
    /// Generate text from a model
    Generate {
        /// Model name or HuggingFace path
        #[arg(long)]
        model: String,
        /// Text prompt for generation
        #[arg(long)]
        prompt: String,
        /// Optional checkpoint path to load fine-tuned weights
        #[arg(long)]
        checkpoint: Option<String>,
        /// Maximum number of tokens to generate
        #[arg(long, default_value = "50")]
        max_tokens: usize,
        /// Sampling temperature (0.0 = greedy, higher = more random)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Compare base model with checkpoint (requires --checkpoint)
        #[arg(long)]
        compare: bool,
        /// Optional EOS token ID override
        #[arg(long)]
        eos_token: Option<i32>,
    },
    /// Export fine-tuned model to safetensors
    Export {
        /// Base model name
        #[arg(long)]
        model: String,
        /// Checkpoint path
        #[arg(long)]
        checkpoint: std::path::PathBuf,
        /// Output path
        #[arg(long)]
        output: std::path::PathBuf,
    },
    /// Build dataset from HuggingFace source with Distrust scoring
    Dataset {
        /// Source HuggingFace dataset ID (e.g. HuggingFaceH4/ultrachat_200k)
        #[arg(long)]
        source: String,
        /// Output directory
        #[arg(long, default_value = "data")]
        output_dir: std::path::PathBuf,
        /// Limit number of examples
        #[arg(long)]
        limit: Option<usize>,
    },
}

pub async fn run() -> Result<()> {
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
        } => commands::benchmark(max_memory, optimize, output, single_model, force).await,
        Commands::Optimize {
            model,
            max_memory,
            quick,
            output,
        } => commands::optimize(model, max_memory, quick, output).await,
        Commands::Train {
            model,
            batch_size,
            gradient_accumulation_steps,
            lora_rank,
            max_steps,
            resume,
            max_memory,
            memory_report_interval,
            auto_optimize,
            metrics_file,
            save_best,
            alpha,
            lambda_weight,
            output_dir,
            quantize,
            reload_interval_steps,
            worker,
        } => {
            commands::train(
                model,
                batch_size,
                gradient_accumulation_steps,
                lora_rank,
                max_steps,
                resume,
                max_memory,
                memory_report_interval,
                auto_optimize,
                metrics_file,
                save_best,
                alpha,
                lambda_weight,
                output_dir,
                quantize,
                worker,
                reload_interval_steps,
                None, // start_step not supported in direct mode
            )
            .await
        }
        Commands::Validate {
            model,
            benchmarks,
            checkpoint,
        } => commands::validate(model, benchmarks, checkpoint),
        Commands::Generate {
            model,
            prompt,
            checkpoint,
            max_tokens,
            temperature,

            compare,
            eos_token,
        } => commands::generate(
            model,
            prompt,
            checkpoint,
            max_tokens,
            temperature,
            compare,
            eos_token,
        ),
        Commands::Export {
            model,
            checkpoint,
            output,
        } => commands::export_command(&model, &checkpoint, &output),
        Commands::Dataset {
            source,
            output_dir,
            limit,
        } => commands::dataset(source, output_dir, limit).await,
    }
}
