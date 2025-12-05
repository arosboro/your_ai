"""
Hardware Profiles for Empirical Distrust Training

Interactive hardware detection and configuration optimization for Apple Silicon Macs.
Recommends optimal models and training parameters based on chip generation, variant, and memory.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Profile storage location
PROFILE_PATH = Path.home() / ".your_ai" / "hardware_profile.json"

# =============================================================================
# Hardware Specifications Database
# =============================================================================

# GPU cores by generation and variant
GPU_CORES = {
    "m1": {"base": 8, "pro": 16, "max": 32, "ultra": 64},
    "m2": {"base": 10, "pro": 19, "max": 38, "ultra": 76},
    "m3": {"base": 10, "pro": 18, "max": 40, "ultra": 80},
    "m4": {"base": 10, "pro": 20, "max": 40, "ultra": 80},
}

# Memory options by variant (common configurations)
MEMORY_OPTIONS = {
    "base": [8, 16, 24, 32],
    "pro": [18, 32, 36, 48],
    "max": [32, 36, 48, 64, 96, 128],
    "ultra": [64, 96, 128, 192],
}

# =============================================================================
# Optimized Training Profiles
# =============================================================================

# Training configurations indexed by (variant, memory_gb)
# These are empirically validated settings
HARDWARE_PROFILES = {
    # M1/M2/M3/M4 Ultra configurations
    ("ultra", 192): {
        "batch_size": 8,
        "lora_rank": 256,
        "lora_num_layers": 32,
        "grad_checkpoint": False,
        "model_tier": "large",
    },
    ("ultra", 128): {
        "batch_size": 6,
        "lora_rank": 192,
        "lora_num_layers": 28,
        "grad_checkpoint": False,
        "model_tier": "large",
    },
    ("ultra", 96): {
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 24,
        "grad_checkpoint": True,  # Required for 70B - only 12GB headroom
        "model_tier": "large",
    },
    ("ultra", 64): {
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 20,
        "grad_checkpoint": False,
        "model_tier": "large",
    },
    # M1/M2/M3/M4 Max configurations
    ("max", 128): {
        "batch_size": 6,
        "lora_rank": 192,
        "lora_num_layers": 28,
        "grad_checkpoint": False,
        "model_tier": "large",
    },
    ("max", 96): {
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 24,
        "grad_checkpoint": False,
        "model_tier": "medium",
    },
    ("max", 64): {
        "batch_size": 4,
        "lora_rank": 128,
        "lora_num_layers": 20,
        "grad_checkpoint": False,
        "model_tier": "medium",
    },
    ("max", 48): {
        "batch_size": 2,
        "lora_rank": 96,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "medium",
    },
    ("max", 36): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
    ("max", 32): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
    # M1/M2/M3/M4 Pro configurations
    ("pro", 48): {
        "batch_size": 2,
        "lora_rank": 96,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "medium",
    },
    ("pro", 36): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
    ("pro", 32): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
    ("pro", 18): {
        "batch_size": 1,
        "lora_rank": 32,
        "lora_num_layers": 12,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
    # M1/M2/M3/M4 Base configurations
    ("base", 32): {
        "batch_size": 2,
        "lora_rank": 64,
        "lora_num_layers": 16,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
    ("base", 24): {
        "batch_size": 2,
        "lora_rank": 32,
        "lora_num_layers": 12,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
    ("base", 16): {
        "batch_size": 1,
        "lora_rank": 32,
        "lora_num_layers": 8,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
    ("base", 8): {
        "batch_size": 1,
        "lora_rank": 16,
        "lora_num_layers": 4,
        "grad_checkpoint": True,
        "model_tier": "entry",
    },
}


# =============================================================================
# Model Memory Requirements
# =============================================================================

MODEL_REQUIREMENTS = {
    # Entry tier (16GB+)
    "hermes-7b": {
        "hf_name": "NousResearch/Hermes-2-Pro-Mistral-7B",
        "inference_gb": 6,
        "training_gb": 12,
        "params": "7B",
        "tier": "entry",
        "recommended": True,
    },
    "dolphin-8b": {
        "hf_name": "cognitivecomputations/dolphin-2.9-llama3-8b",
        "inference_gb": 7,
        "training_gb": 14,
        "params": "8B",
        "tier": "entry",
        "recommended": True,
    },
    "llama-8b": {
        "hf_name": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
        "inference_gb": 7,
        "training_gb": 14,
        "params": "8B",
        "tier": "entry",
        "recommended": True,
    },
    # Medium tier (32GB+)
    "r1-distill-14b": {
        "hf_name": "huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2",
        "inference_gb": 10,
        "training_gb": 22,
        "params": "14B",
        "tier": "medium",
        "recommended": False,  # Chinese model - corpus-level censorship
        "warning": "Chinese model - corpus-level censorship",
    },
    "r1-distill-32b": {
        "hf_name": "huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated",
        "inference_gb": 20,
        "training_gb": 42,
        "params": "32B",
        "tier": "medium",
        "recommended": False,  # Chinese model - corpus-level censorship
        "warning": "Chinese model - corpus-level censorship",
    },
    # Large tier (64GB+)
    "r1-distill-70b": {
        "hf_name": "huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated",
        "inference_gb": 42,
        "training_gb": 65,
        "params": "70B",
        "tier": "large",
        "recommended": False,  # Chinese model - corpus-level censorship
        "warning": "Chinese model - corpus-level censorship",
    },
    "deepseek-r1-70b": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "inference_gb": 42,
        "training_gb": 65,
        "params": "70B",
        "tier": "large",
        "recommended": True,
    },
    "hermes-70b": {
        "hf_name": "NousResearch/Hermes-3-Llama-3.1-70B",
        "inference_gb": 42,
        "training_gb": 65,
        "params": "70B",
        "tier": "large",
        "recommended": True,
    },
    # Enterprise (NOT consumer feasible)
    "r1-1776": {
        "hf_name": "perplexity-ai/r1-1776",
        "inference_gb": 450,
        "training_gb": 600,
        "params": "671B MoE",
        "tier": "enterprise",
        "recommended": False,
        "warning": "Requires enterprise hardware (multi-GPU cluster)",
    },
}


# =============================================================================
# Model Size Detection and Scaling
# =============================================================================

# Model size categories with recommended LoRA settings
# These are optimized settings for each model size tier
MODEL_SIZE_CONFIGS = {
    "small": {  # 7B-8B models
        "lora_rank": 32,
        "lora_num_layers": 8,
        "batch_size_multiplier": 2.0,  # Can increase batch size
    },
    "medium": {  # 14B models
        "lora_rank": 48,
        "lora_num_layers": 12,
        "batch_size_multiplier": 1.5,
    },
    "large": {  # 32B models
        "lora_rank": 64,
        "lora_num_layers": 16,
        "batch_size_multiplier": 1.0,
    },
    "xlarge": {  # 70B+ models
        "lora_rank": 128,
        "lora_num_layers": 24,
        "batch_size_multiplier": 1.0,
    },
}


def detect_model_size(model_path: str) -> Tuple[str, int]:
    """
    Detect model size from HuggingFace model path.

    Parses patterns like "7B", "8B", "14B", "32B", "70B" from model name
    and returns the size category and approximate parameter count in billions.

    Args:
        model_path: HuggingFace model ID or local path (e.g., "NousResearch/Hermes-2-Pro-Mistral-7B")

    Returns:
        Tuple of (size_category, params_billions):
        - size_category: "small" (7-8B), "medium" (14B), "large" (32B), "xlarge" (70B+)
        - params_billions: Detected parameter count in billions (0 if not detected)
    """
    import re

    # Extract just the model name from path
    model_name = model_path.split("/")[-1].lower()

    # Try to find parameter count patterns like "7b", "8b", "14b", "32b", "70b"
    # Match patterns: 7b, 7B, 7-b, 7_b, etc.
    patterns = [
        r"(\d+)[-_]?b(?:illion)?",  # 7b, 7B, 7-b, 7_b, 7billion
        r"(\d+)b[-_]",  # 7b-, 7b_
        r"-(\d+)b",  # -7b (common in model names)
    ]

    params_billions = 0
    for pattern in patterns:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            params_billions = int(match.group(1))
            break

    # Fallback: check for known model patterns
    if params_billions == 0:
        known_sizes = {
            "mistral": 7,
            "llama-3": 8,
            "llama3": 8,
            "phi-2": 3,
            "phi2": 3,
            "qwen": 7,  # Default Qwen is 7B
        }
        for pattern, size in known_sizes.items():
            if pattern in model_name:
                params_billions = size
                break

    # Categorize by size
    if params_billions <= 0:
        # Unknown size - assume small for safety
        return "small", 0
    elif params_billions <= 10:
        return "small", params_billions
    elif params_billions <= 20:
        return "medium", params_billions
    elif params_billions <= 50:
        return "large", params_billions
    else:
        return "xlarge", params_billions


def scale_profile_for_model(profile: Dict, model_path: str) -> Dict:
    """
    Scale hardware profile LoRA parameters based on model size.

    When running a smaller model on powerful hardware, the default profile
    settings (designed for the largest model that fits) may be too aggressive.
    This function scales down LoRA rank and layers appropriately.

    Args:
        profile: Hardware profile dict with lora_rank, lora_num_layers, etc.
        model_path: HuggingFace model ID or local path

    Returns:
        New profile dict with scaled parameters for the model size
    """
    # Make a copy to avoid mutating the original
    scaled = profile.copy()

    # Detect model size
    size_category, params_billions = detect_model_size(model_path)

    # Get model-appropriate settings
    model_config = MODEL_SIZE_CONFIGS.get(size_category, MODEL_SIZE_CONFIGS["small"])

    # Check if the profile's model_tier suggests it was designed for a larger model
    profile_tier = profile.get("model_tier", "entry")

    # Only scale down if the profile is for a larger model tier than what we're training
    tier_order = {"entry": 0, "medium": 1, "large": 2}
    size_to_tier = {"small": "entry", "medium": "medium", "large": "large", "xlarge": "large"}

    target_tier = size_to_tier.get(size_category, "entry")
    profile_tier_level = tier_order.get(profile_tier, 0)
    target_tier_level = tier_order.get(target_tier, 0)

    if profile_tier_level > target_tier_level:
        # Profile is designed for a larger model - scale down
        scaled["lora_rank"] = model_config["lora_rank"]
        scaled["lora_num_layers"] = model_config["lora_num_layers"]

        # Optionally scale up batch size since we have memory headroom
        if "batch_size_multiplier" in model_config:
            new_batch = int(scaled.get("batch_size", 2) * model_config["batch_size_multiplier"])
            # Cap at reasonable maximum
            scaled["batch_size"] = min(new_batch, 8)

        # Keep gradient checkpointing enabled - even small models benefit from it
        # when using larger batch sizes (saves activation memory during backprop)

    # Log the scaling decision
    if params_billions > 0:
        print(f"  → Model size detected: {params_billions}B ({size_category})")
        if profile_tier_level > target_tier_level:
            print(
                f"  → Scaling LoRA for {size_category} model: "
                f"rank={scaled['lora_rank']}, layers={scaled['lora_num_layers']}"
            )

    return scaled


# =============================================================================
# Hardware Detection
# =============================================================================


def detect_hardware() -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Auto-detect Mac chip generation, variant, and unified memory.

    Returns:
        Tuple of (generation, variant, memory_gb) or (None, None, None) if detection fails.
        generation: "m1", "m2", "m3", "m4"
        variant: "base", "pro", "max", "ultra"
        memory_gb: Unified memory in GB
    """
    try:
        # Get chip info using sysctl
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        chip_string = result.stdout.strip().lower()

        # Parse generation
        generation = None
        if "m1" in chip_string:
            generation = "m1"
        elif "m2" in chip_string:
            generation = "m2"
        elif "m3" in chip_string:
            generation = "m3"
        elif "m4" in chip_string:
            generation = "m4"

        # Parse variant
        variant = "base"
        if "ultra" in chip_string:
            variant = "ultra"
        elif "max" in chip_string:
            variant = "max"
        elif "pro" in chip_string:
            variant = "pro"

        # Get memory using sysctl
        mem_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        memory_bytes = int(mem_result.stdout.strip())
        memory_gb = memory_bytes // (1024**3)

        return generation, variant, memory_gb

    except Exception:
        return None, None, None


def get_gpu_cores(generation: str, variant: str) -> int:
    """Get GPU core count for a specific chip configuration."""
    return GPU_CORES.get(generation, {}).get(variant, 0)


# =============================================================================
# Model Recommendations
# =============================================================================


def recommend_models(memory_gb: int) -> List[Dict]:
    """
    Return ranked list of feasible models with fit status and warnings.

    Args:
        memory_gb: Available unified memory

    Returns:
        List of dicts with model info, status, warnings, and headroom
    """
    budget = int(memory_gb * 0.80)  # 80% safety margin
    results = []

    for model_key, reqs in MODEL_REQUIREMENTS.items():
        training_gb = reqs["training_gb"]
        headroom = budget - training_gb

        if training_gb <= budget:
            if headroom < 5:
                status = "TIGHT_FIT"
                status_display = "⚠️  TIGHT FIT"
                fit_warning = "Reduce batch_size to 1, enable grad_checkpoint"
            elif headroom < 15:
                status = "COMFORTABLE"
                status_display = "✅ COMFORTABLE"
                fit_warning = None
            else:
                status = "OPTIMAL"
                status_display = "✅ OPTIMAL"
                fit_warning = "Plenty of headroom for larger batch/rank"
        else:
            status = "DOES_NOT_FIT"
            status_display = "❌ DOES NOT FIT"
            fit_warning = f"Needs {training_gb}GB, you have {budget}GB budget"

        # Combine fit warning with model warning if present
        model_warning = reqs.get("warning")
        if model_warning and fit_warning:
            combined_warning = f"{fit_warning}; {model_warning}"
        else:
            combined_warning = fit_warning or model_warning

        results.append(
            {
                "model": model_key,
                "hf_name": reqs["hf_name"],
                "params": reqs["params"],
                "tier": reqs["tier"],
                "status": status,
                "status_display": status_display,
                "warning": combined_warning,
                "headroom_gb": headroom,
                "training_gb": training_gb,
                "recommended": reqs.get("recommended", True) and status != "DOES_NOT_FIT",
            }
        )

    # Sort: recommended feasible first (by headroom), then non-recommended, then infeasible
    def sort_key(x):
        if x["status"] == "DOES_NOT_FIT":
            return (2, -x["headroom_gb"])  # Infeasible last
        elif not x["recommended"]:
            return (1, -x["headroom_gb"])  # Non-recommended in middle
        else:
            return (0, -x["headroom_gb"])  # Recommended first, by capability

    results.sort(key=sort_key)
    return results


def get_best_model(memory_gb: int) -> Optional[Dict]:
    """Get the best recommended model for the given memory."""
    recommendations = recommend_models(memory_gb)
    for rec in recommendations:
        if rec["recommended"] and rec["status"] != "DOES_NOT_FIT":
            return rec
    return None


# =============================================================================
# Configuration Optimization
# =============================================================================


def get_optimized_profile(generation: str, variant: str, memory_gb: int) -> Dict:
    """
    Generate optimized training profile for specific hardware.

    Args:
        generation: Chip generation ("m1", "m2", "m3", "m4")
        variant: Chip variant ("base", "pro", "max", "ultra")
        memory_gb: Unified memory in GB

    Returns:
        Dict with optimized training parameters
    """
    # Find closest matching profile
    key = (variant, memory_gb)
    if key in HARDWARE_PROFILES:
        profile = HARDWARE_PROFILES[key].copy()
    else:
        # Find closest memory configuration for this variant
        variant_profiles = [
            (mem, prof) for (var, mem), prof in HARDWARE_PROFILES.items() if var == variant
        ]
        if variant_profiles:
            # Find closest memory
            closest = min(variant_profiles, key=lambda x: abs(x[0] - memory_gb))
            profile = closest[1].copy()
        else:
            # Fallback to conservative defaults
            profile = {
                "batch_size": 1,
                "lora_rank": 32,
                "lora_num_layers": 8,
                "grad_checkpoint": True,
                "model_tier": "entry",
            }

    # Add hardware info
    gpu_cores = get_gpu_cores(generation, variant)
    profile["generation"] = generation
    profile["variant"] = variant
    profile["memory_gb"] = memory_gb
    profile["gpu_cores"] = gpu_cores
    profile["training_budget_gb"] = int(memory_gb * 0.80)

    # Adjust batch size based on GPU cores (more cores = more parallelism)
    if gpu_cores >= 64 and profile["batch_size"] < 4:
        profile["batch_size"] = 4
    elif gpu_cores >= 38 and profile["batch_size"] < 3:
        profile["batch_size"] = max(3, profile["batch_size"])

    return profile


# =============================================================================
# Profile Storage
# =============================================================================


def save_hardware_profile(profile: Dict) -> Path:
    """
    Save hardware profile to user home directory.

    Args:
        profile: Hardware profile dict

    Returns:
        Path to saved profile
    """
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)
    return PROFILE_PATH


def load_hardware_profile() -> Optional[Dict]:
    """
    Load saved hardware profile if exists.

    Returns:
        Profile dict or None if no profile saved
    """
    if PROFILE_PATH.exists():
        try:
            with open(PROFILE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def profile_exists() -> bool:
    """Check if a hardware profile has been saved."""
    return PROFILE_PATH.exists()


# =============================================================================
# Interactive Setup
# =============================================================================


def interactive_setup() -> Dict:
    """
    Run interactive hardware setup wizard.

    Returns:
        Optimized hardware profile dict
    """
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║       Empirical Distrust Training - Hardware Setup            ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    # Try auto-detection first
    auto_gen, auto_var, auto_mem = detect_hardware()

    if auto_gen and auto_var and auto_mem:
        print(f"Detected: {auto_gen.upper()} {auto_var.title()} with {auto_mem}GB")
        confirm = input("Is this correct? [Y/n] ").strip().lower()
        if confirm in ("", "y", "yes"):
            profile = get_optimized_profile(auto_gen, auto_var, auto_mem)
            _display_profile(profile)
            if _confirm_save():
                save_hardware_profile(profile)
            return profile
        print()

    # Manual selection - Step 1: Generation
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Step 1/3: Chip Generation                                    ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    generations = ["m1", "m2", "m3", "m4"]
    gen_labels = [
        "M1 (2020-2021)",
        "M2 (2022-2023)",
        "M3 (2023-2024)",
        "M4 (2024+)",
    ]
    for i, label in enumerate(gen_labels, 1):
        print(f"  {i}) {label}")
    print()
    while True:
        try:
            choice = int(input("Select your chip generation (1-4): ").strip())
            if 1 <= choice <= 4:
                generation = generations[choice - 1]
                break
        except ValueError:
            pass
        print("Invalid choice. Please enter 1-4.")

    # Step 2: Variant
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Step 2/3: Chip Variant                                       ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    variants = ["base", "pro", "max", "ultra"]
    gpu_counts = GPU_CORES[generation]
    for i, var in enumerate(variants, 1):
        cores = gpu_counts.get(var, "?")
        print(f"  {i}) {generation.upper()} {var.title():5} ({cores} GPU cores)")
    print()
    while True:
        try:
            choice = int(input("Select your chip variant (1-4): ").strip())
            if 1 <= choice <= 4:
                variant = variants[choice - 1]
                break
        except ValueError:
            pass
        print("Invalid choice. Please enter 1-4.")

    # Step 3: Memory
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Step 3/3: Unified Memory                                     ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    mem_options = MEMORY_OPTIONS.get(variant, [16, 32, 64])
    print(f"Available memory options for {generation.upper()} {variant.title()}:")
    for i, mem in enumerate(mem_options, 1):
        print(f"  {i}) {mem}GB")
    print(f"  {len(mem_options) + 1}) Custom value")
    print()
    while True:
        try:
            choice = input("Select memory option or enter custom GB: ").strip()
            choice_int = int(choice)
            if 1 <= choice_int <= len(mem_options):
                memory_gb = mem_options[choice_int - 1]
                break
            elif choice_int == len(mem_options) + 1:
                custom_mem = int(input("Enter memory in GB: ").strip())
                if 8 <= custom_mem <= 512:  # Reasonable bounds for Apple Silicon
                    memory_gb = custom_mem
                    break
                else:
                    print("Memory must be between 8GB and 512GB.")
                    continue
            elif choice_int > len(mem_options) + 1:  # Assume direct GB entry
                if 8 <= choice_int <= 512:  # Reasonable bounds for Apple Silicon
                    memory_gb = choice_int
                    break
                else:
                    print("Memory must be between 8GB and 512GB.")
                    continue
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

    # Generate optimized profile
    profile = get_optimized_profile(generation, variant, memory_gb)
    _display_profile(profile)

    if _confirm_save():
        save_hardware_profile(profile)

    return profile


def _display_profile(profile: Dict) -> None:
    """Display the optimized profile."""
    print()
    print("━" * 65)
    print(
        f"Hardware: {profile['generation'].upper()} {profile['variant'].title()} "
        f"({profile['gpu_cores']} GPU cores) with {profile['memory_gb']}GB"
    )
    print(f"Training budget: {profile['training_budget_gb']}GB (80% safety margin)")
    print("━" * 65)
    print()
    print("Optimized configuration:")
    print(f"  • batch_size:      {profile['batch_size']}")
    print(f"  • lora_rank:       {profile['lora_rank']}")
    print(f"  • lora_num_layers: {profile['lora_num_layers']}")
    print(f"  • grad_checkpoint: {profile['grad_checkpoint']}")
    print(f"  • model_tier:      {profile['model_tier']}")
    print()


def _confirm_save() -> bool:
    """Ask user to confirm saving profile."""
    confirm = input(f"Save this profile to {PROFILE_PATH}? [Y/n] ").strip().lower()
    return confirm in ("", "y", "yes")


def display_recommendations(memory_gb: int) -> None:
    """Display model recommendations for given memory."""
    budget = int(memory_gb * 0.80)
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  Training budget: {budget}GB (80% of {memory_gb}GB)".ljust(73) + "║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  MODEL RECOMMENDATIONS (sorted by capability)                        ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    recommendations = recommend_models(memory_gb)
    best_model = None

    for rec in recommendations:
        print("║" + " " * 72 + "║")
        line = f"  {rec['status_display']} - {rec['model']} ({rec['params']})"
        print("║" + line.ljust(72) + "║")

        if rec["status"] != "DOES_NOT_FIT":
            detail = f"     Training: {rec['training_gb']}GB | Headroom: {rec['headroom_gb']}GB"
            print("║" + detail.ljust(72) + "║")

            if rec["recommended"] and best_model is None:
                best_model = rec
                tip = "     → RECOMMENDED: Best capability that fits"
                print("║" + tip.ljust(72) + "║")
        else:
            detail = f"     Training: {rec['training_gb']}GB | Your budget: {budget}GB"
            print("║" + detail.ljust(72) + "║")

        if rec["warning"]:
            warn = f"     ⚠️  {rec['warning']}"
            # Truncate if too long
            if len(warn) > 70:
                warn = warn[:67] + "..."
            print("║" + warn.ljust(72) + "║")

    print("║" + " " * 72 + "║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hardware profile management")
    parser.add_argument("--detect", action="store_true", help="Auto-detect hardware")
    parser.add_argument("--setup", action="store_true", help="Run interactive setup")
    parser.add_argument("--recommend", action="store_true", help="Show model recommendations")
    parser.add_argument("--memory", type=int, help="Memory in GB for recommendations")
    parser.add_argument("--show", action="store_true", help="Show saved profile")

    args = parser.parse_args()

    if args.detect:
        gen, var, mem = detect_hardware()
        if gen:
            print(f"Detected: {gen.upper()} {var.title()} with {mem}GB")
        else:
            print("Could not auto-detect hardware")

    elif args.setup:
        profile = interactive_setup()
        print("Setup complete!")

    elif args.recommend:
        memory = args.memory
        if not memory:
            profile = load_hardware_profile()
            if profile:
                memory = profile.get("memory_gb")
            else:
                _, _, memory = detect_hardware()
        if memory:
            display_recommendations(memory)
        else:
            print("Please specify --memory or run --setup first")

    elif args.show:
        profile = load_hardware_profile()
        if profile:
            _display_profile(profile)
        else:
            print("No saved profile. Run --setup to create one.")

    else:
        parser.print_help()
