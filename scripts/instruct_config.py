from model_utility import (
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
    disable_flash_attention,
    get_data_size,
    get_gpu_count,
)
from copy import deepcopy
from lrs_lookup import get_instruct_lr
from hyperparam_optimizer import get_optimal_warmup_steps, get_optimal_gradient_accumulation
import math


FIXED_BS_CONFIG = {
    "EleutherAI/gpt-neo-1.3B": {"batch_size": 36},
    "EleutherAI/gpt-neo-125m": {"batch_size": 48},
    "bigscience/bloom-560m": {"batch_size": 10},
    "facebook/opt-1.3b": {"batch_size": 38},
    "facebook/opt-350m": {"batch_size": 36},
    "facebook/opt-125m": {"batch_size": 48},
}

INSTRUCT_CONFIG = {
    "0_1_b": {
        "lr": 0.0001,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 140,
        "use_lora": False,
    },
    "1_2_b": {
        "lr": 0.0001,
        "distributed": "ddp",
        "gpu_count": 1,
        "use_lora": False,
        "batch_size": 100,
    },
    "2_4_b": {
        "lr": 7.5e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 48,
    },
    "4_5_b": {
        "lr": 7e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 40,
    },
    "5_9_b": {
        "lr": 3.5e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 28,
    },
    "9_12_b": {
        "lr": 1e-4,
        "distributed": "ddp",
        "gpu_count": 2,
        "use_lora": True,
        "batch_size": 32,
    },
    "12_15_b": {
        "lr": 1e-4,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 30,
    },
    "15_40_b": {
        "lr": 8e-5,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 18,
    },
    "40_80_b": {
        "lr": 8e-5,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 8,
    },
}

for key in INSTRUCT_CONFIG:
    INSTRUCT_CONFIG[key]["label"] = key


# Architecture-specific LR coefficients (learned from empirical data)
# These adjust the base LR based on architecture family
ARCHITECTURE_LR_COEFFICIENTS = {
    "llamaforcausallm": 1.0,  # Baseline
    "mistralforcausallm": 1.05,
    "qwen2forcausallm": 1.0,
    "qwen3forcausallm": 1.0,
    "phiforcausallm": 1.1,  # Phi models often need slightly higher LR
    "phi3forcausallm": 1.1,
    "gemmaforcausallm": 1.0,
    "gemma2forcausallm": 1.0,
    "mixtralforcausallm": 0.95,  # Mixtral benefits from slightly lower LR
    "optforcausallm": 1.15,  # OPT models often need higher LR
    "gptneoforcausallm": 1.2,
    "gptneoxforcausallm": 1.15,
    "gptjforcausallm": 1.1,
    "falconforcausallm": 1.1,
    "bloomforcausallm": 1.2,
    "gptossforcausallm": 1.0,
}


def calculate_continuous_lr(
    param_nums: int,
    architecture: str = None,
    dataset_size: int = None,
    avg_seq_length: int = None,
    use_lora: bool = False,
    lora_rank: int = 16,
    hours_to_complete: float = None,
    batch_size: int = None,
    gpu_count: int = None,
    verbose: bool = True
) -> float:
    # Base LR for 1B parameter model
    base_lr_1b = 1e-4
    reference_params = 1_000_000_000  # 1B
    
    # Power-law exponent (negative means larger models get lower LR)
    # Empirically determined: -0.3 to -0.35 works well for most models
    exponent = -0.32
    
    # Calculate base LR using power-law scaling
    if param_nums > 0:
        lr = base_lr_1b * (param_nums / reference_params) ** exponent
    else:
        lr = base_lr_1b
    
    # Architecture-specific adjustment
    if architecture:
        arch_lower = architecture.strip().lower()
        arch_coef = ARCHITECTURE_LR_COEFFICIENTS.get(arch_lower, 1.0)
        lr *= arch_coef
        if verbose and arch_coef != 1.0:
            print(f"  [LR] Architecture adjustment ({arch_lower}): {arch_coef:.3f}x", flush=True)
    
    # Dataset size adjustment: larger datasets can tolerate higher LR
    # sqrt scaling: sqrt(dataset_size / 10k)
    if dataset_size and dataset_size > 0:
        reference_dataset = 10_000
        dataset_factor = math.sqrt(max(dataset_size, reference_dataset) / reference_dataset)
        # Cap the factor to avoid extreme values
        dataset_factor = min(1.5, max(0.7, dataset_factor))
        lr *= dataset_factor
        if verbose:
            print(f"  [LR] Dataset size adjustment ({dataset_size:,} samples): {dataset_factor:.3f}x", flush=True)
    
    # Sequence length adjustment: longer sequences often need lower LR
    # Inverse sqrt scaling: 1 / sqrt(avg_seq_length / 512)
    if avg_seq_length and avg_seq_length > 0:
        reference_seq = 512
        seq_factor = math.sqrt(reference_seq / max(avg_seq_length, reference_seq))
        # Cap the factor
        seq_factor = min(1.3, max(0.8, seq_factor))
        lr *= seq_factor
        if verbose:
            print(f"  [LR] Sequence length adjustment (avg {avg_seq_length}): {seq_factor:.3f}x", flush=True)
    
    # LoRA adjustment: higher rank = more parameters = slightly higher effective LR
    if use_lora and lora_rank > 0:
        # Scale by sqrt of rank ratio (rank 16 is baseline)
        lora_factor = math.sqrt(lora_rank / 16.0)
        lora_factor = min(1.2, max(0.9, lora_factor))  # Cap between 0.9x and 1.2x
        lr *= lora_factor
        if verbose:
            print(f"  [LR] LoRA adjustment (rank {lora_rank}): {lora_factor:.3f}x", flush=True)
    
    # Batch size adjustment: larger batch sizes can use higher LR
    # This is applied in base LR calculation for better batch-aware optimization
    # We use effective batch size (per_device * gpu_count) if available
    if batch_size is not None and batch_size > 0:
        effective_batch = batch_size
        if gpu_count is not None and gpu_count > 0:
            effective_batch = batch_size * gpu_count
        
        reference_batch = 64
        # Adaptive scaling based on model size (same logic as reg_ratio but applied to base LR)
        if param_nums < 1_000_000_000:  # < 1B: more linear
            batch_factor = (effective_batch / reference_batch) ** 0.7
        elif param_nums < 10_000_000_000:  # 1-10B: sqrt scaling
            batch_factor = math.sqrt(effective_batch / reference_batch)
        else:  # > 10B: sub-linear
            batch_factor = (effective_batch / reference_batch) ** 0.25
        
        # Cap the batch factor to avoid extreme values
        batch_factor = max(0.7, min(1.5, batch_factor))
        if batch_factor != 1.0:
            lr *= batch_factor
            if verbose:
                print(f"  [LR] Batch size adjustment (effective batch {effective_batch}): {batch_factor:.3f}x", flush=True)
    
    # Time constraint adjustment: higher LR for time-constrained training to converge faster
    # This is applied in base LR calculation (not just in reg_ratio) for better time-aware optimization
    if hours_to_complete is not None and hours_to_complete > 0:
        if hours_to_complete <= 0.5:  # Very short jobs (<30 min)
            time_factor = 1.4  # Aggressive LR boost for very short jobs
        elif hours_to_complete <= 0.75:  # Short jobs (<45 min)
            time_factor = 1.3
        elif hours_to_complete <= 1.0:  # Medium-short jobs (<1 hour)
            time_factor = 1.2
        elif hours_to_complete <= 2.0:  # Medium jobs
            time_factor = 1.1
        else:  # Long jobs
            time_factor = 1.0  # No adjustment for long jobs
        
        if time_factor != 1.0:
            lr *= time_factor
            if verbose:
                print(f"  [LR] Time constraint adjustment ({hours_to_complete:.2f}h): {time_factor:.2f}x", flush=True)
    
    # Ensure reasonable bounds
    lr = max(1e-6, min(1e-3, lr))
    
    return lr


def get_instruct_config(param_nums: int, use_continuous_lr: bool = True) -> dict:
    """
    Get instruction training configuration.
    
    Args:
        param_nums: Number of model parameters
        use_continuous_lr: If True, use continuous LR scaling; if False, use legacy buckets
    
    Note: LR calculation is optimized - only calculates base LR here without full context.
    Full LR calculation with all factors happens later in get_training_json().
    """
    # OPTIMIZATION: Only calculate minimal base LR here, full calculation happens later
    # This avoids duplicate expensive calculations
    if use_continuous_lr:
        # Calculate minimal base LR (just param-based, without other factors, verbose=False to reduce output)
        # Full calculation with all factors happens in get_training_json()
        base_lr = calculate_continuous_lr(param_nums, verbose=False)
        
        # Determine other config based on model size (still use buckets for these)
        # but we'll use continuous LR instead
        if param_nums < 1_000_000_000:
            base_config = INSTRUCT_CONFIG["0_1_b"]
        elif param_nums < 2_000_000_000:
            base_config = INSTRUCT_CONFIG["1_2_b"]
        elif param_nums < 4_000_000_000:
            base_config = INSTRUCT_CONFIG["2_4_b"]
        elif param_nums < 5_000_000_000:
            base_config = INSTRUCT_CONFIG["4_5_b"]
        elif param_nums < 9_000_000_000:
            base_config = INSTRUCT_CONFIG["5_9_b"]
        elif param_nums < 12_000_000_000:
            base_config = INSTRUCT_CONFIG["9_12_b"]
        elif param_nums < 15_000_000_000:
            base_config = INSTRUCT_CONFIG["12_15_b"]
        elif param_nums < 35_000_000_000:
            base_config = INSTRUCT_CONFIG["15_40_b"]
        elif param_nums < 80_000_000_000:
            base_config = INSTRUCT_CONFIG["40_80_b"]
        else:
            base_config = {
                "distributed": "ds",
                "gpu_count": 8,
                "batch_size": 6,
                "use_lora": True,
            }
        
        result = deepcopy(base_config)
        result["lr"] = base_lr
        # Note: Detailed LR logging happens in get_training_json() after full calculation
    else:
        # Legacy bucket-based approach
        result = {
            "lr": 4e-5,
            "distributed": "ds",
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True,
        }
        if param_nums < 1_000_000_000:
            result = INSTRUCT_CONFIG["0_1_b"]
        elif param_nums < 2_000_000_000:
            result = INSTRUCT_CONFIG["1_2_b"]
        elif param_nums < 4_000_000_000:
            result = INSTRUCT_CONFIG["2_4_b"]
        elif param_nums < 5_000_000_000:
            result = INSTRUCT_CONFIG["4_5_b"]
        elif param_nums < 9_000_000_000:
            result = INSTRUCT_CONFIG["5_9_b"]
        elif param_nums < 12_000_000_000:
            result = INSTRUCT_CONFIG["9_12_b"]
        elif param_nums < 15_000_000_000:
            result = INSTRUCT_CONFIG["12_15_b"]
        elif param_nums < 35_000_000_000:
            result = INSTRUCT_CONFIG["15_40_b"]
        elif param_nums < 80_000_000_000:
            result = INSTRUCT_CONFIG["40_80_b"]
        else:
            print(f"Model size {param_nums} is not supported")
        result = deepcopy(result)
    
    # Special batch size adjustment for 8-9B range
    if param_nums < 9_000_000_000 and param_nums > 8_000_000_000:
        result["batch_size"] = int(2 * result["batch_size"] / 3)
    
    return result


def get_run_cmd(config: dict, gpu_nums: int, train_info: dict = None):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "use_lora",
        "packing",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config["distributed"]
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    # AutoML: Get optimal warmup steps (learns from history, falls back to heuristics)
    warmup_steps = 35  # Default fallback
    if train_info:
        model_name = train_info.get("model_name", "")
        model_path = train_info.get("model_path", "")
        param_nums = get_model_num_params(model_name, model_path) if model_name else 0
        dataset_size = None
        try:
            request_path = train_info.get("request_path")
            if request_path:
                dataset_size = get_data_size(request_path)
        except:
            pass
        
        hours_to_complete = float(train_info.get("hours_to_complete", 0) or 0)
        # Get learning rate from config (which is already calculated)
        learning_rate = config.get("lr")
        
        # Use AutoML optimizer to get optimal warmup steps
        warmup_steps = get_optimal_warmup_steps(
            model=model_name,
            task_type="instruct",
            param_nums=param_nums,
            dataset_size=dataset_size,
            learning_rate=learning_rate,
            hours_to_complete=hours_to_complete if hours_to_complete > 0 else None,
            default_warmup=35
        )
    
    template = (
        start_cmd
        + f""" train_instruct.py \
    --request_path {{request_path}} \
    --bf16 True \
    --report_to wandb \
    --output_dir {{output_dir}} \
    --run_name {{run_name}} \
    --num_train_epochs {{epoch_num}} \
    --per_device_train_batch_size {{batch_size}} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {{gradient_accumulation_steps}} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy epoch \
    --logging_steps 5 \
    --learning_rate {{learning_rate}} \
    --weight_decay 0.01 \
    --warmup_steps {warmup_steps} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{{\\"min_lr_rate\\": {{min_lr_rate}}}}" \
    --tf32 True \
    --gradient_checkpointing {{gradient_checkpointing}} \
    --optim {{optimizer}} \
    --use_liger {{use_liger}} \
    --packing {{packing}} --disable_fa {{disable_fa}} \
    --label_smoothing_factor {{label_smoothing_factor}}"""
    )
    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    if config["use_lora"]:
        template = template + """ --use_lora True"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    if config.get("use_attn_implementation", ""):
        use_attn_implementation = config["use_attn_implementation"]
        template = (
            template + f""" --use_attn_implementation {use_attn_implementation}"""
        )

    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    
    # Get dataset information for dataset-aware LR calculation
    dataset_size = None
    avg_seq_length = None
    try:
        # Try to get dataset size from request path if available
        request_path = train_info.get("request_path")
        if request_path:
            dataset_size = get_data_size(request_path)
    except:
        pass
    
    # Get sequence length from train_info if available
    if "max_length" in train_info.get("train_request", {}):
        avg_seq_length = train_info["train_request"]["max_length"]
    
    # OPTIMIZATION: Check lookup table FIRST to avoid expensive calculations if LR is already known
    use_lookup_lr = False
    lookup_lr = None
    if train_info.get("find_lk_lr", False):
        lookup_lr = get_instruct_lr(model_name)
        if lookup_lr is not None:
            use_lookup_lr = True
            print(f"  [LR] Found in lookup table: {lookup_lr:.8f}", flush=True)
    
    # Use continuous LR calculation with dataset and architecture awareness
    config = get_instruct_config(param_nums, use_continuous_lr=True)
    
    # Get time constraint information
    hours_to_complete = float(train_info.get("hours_to_complete", 0) or 0)
    
    # Get batch size information (will be refined later, but use initial estimate for LR calculation)
    initial_batch_size = config.get("batch_size")
    gpu_count = config.get("gpu_count", 1)
    
    # Allow base batch size multiplier via train_info (applies to all models)
    if "base_batch_size_multiplier" in train_info:
        base_multiplier = float(train_info["base_batch_size_multiplier"])
        initial_batch_size = int(initial_batch_size * base_multiplier)
        config["batch_size"] = initial_batch_size
        print(f"  [Batch Size] Base multiplier ({base_multiplier}x) applied, base batch size: {initial_batch_size}", flush=True)
    
    # Only calculate LR if not using lookup table (optimization: skip expensive calculation)
    if use_lookup_lr:
        calculated_lr = lookup_lr
        print(f"  [LR] Using lookup table LR, skipping continuous calculation", flush=True)
    else:
        # Recalculate LR with full context (architecture, dataset, sequence length, time constraints, batch size)
        use_lora = config.get("use_lora", False)
        lora_rank = train_info.get("train_request", {}).get("lora_r", 16) if use_lora else 16
        
        calculated_lr = calculate_continuous_lr(
            param_nums=param_nums,
            architecture=model_architecture,
            dataset_size=dataset_size,
            avg_seq_length=avg_seq_length,
            use_lora=use_lora,
            lora_rank=lora_rank,
            hours_to_complete=hours_to_complete,
            batch_size=initial_batch_size,
            gpu_count=gpu_count
        )
        print(f"  [LR] Calculated continuous LR: {calculated_lr:.8f}", flush=True)
    
    # DECISIVE IMPROVEMENT: Joint LR-batch size optimization based on test loss
    # Priority 1: Time-aware optimization (test loss per unit time) if time-constrained
    # Priority 2: General test loss optimization
    from hyperparam_optimizer import get_optimal_lr_batch_pair, scale_lr_for_batch_size, optimize_for_test_loss_per_time
    
    initial_batch_size_for_lr = config.get("batch_size")
    optimal_lr, optimal_batch, opt_reason = None, None, None
    
    # If time-constrained, prioritize time-aware optimization
    if hours_to_complete and hours_to_complete > 0:
        optimal_lr, optimal_batch, opt_reason = optimize_for_test_loss_per_time(
            model=model_name,
            task_type="instruct",
            param_nums=param_nums,
            dataset_size=dataset_size,
            hours_to_complete=hours_to_complete,
            gpu_count=gpu_count
        )
    
    # Fallback to general optimization if time-aware didn't find anything
    if optimal_lr is None or optimal_batch is None:
        optimal_lr, optimal_batch, opt_reason = get_optimal_lr_batch_pair(
            model=model_name,
            task_type="instruct",
            param_nums=param_nums,
            dataset_size=dataset_size,
            hours_to_complete=hours_to_complete if hours_to_complete > 0 else None,
            gpu_count=gpu_count,
            base_lr=calculated_lr if not use_lookup_lr else lookup_lr,
            base_batch_size=initial_batch_size_for_lr,
            min_batch_size=1,
            max_batch_size=128
        )
    
    if optimal_lr is not None and optimal_batch is not None:
        # Use optimal pair from history
        calculated_lr = optimal_lr
        config["batch_size"] = optimal_batch
        print(f"  [AutoML] Using optimal LR-batch pair: LR={optimal_lr:.8f}, batch={optimal_batch} ({opt_reason})", flush=True)
    else:
        # Use calculated values
        print(f"  [AutoML] No optimal pair found, using calculated values: LR={calculated_lr:.8f}, batch={initial_batch_size_for_lr}", flush=True)
    
    # Use the calculated or lookup LR
    config["lr"] = calculated_lr
    
    # Generate unique run_name to avoid WandB warning about run_name matching output_dir
    run_name = train_info.get("wandb_name") or train_info.get("wandb_runid") or f"train_{train_info.get('task_id', 'unknown')}"
    
    # Track initial batch size for LR scaling if batch size changes later
    initial_effective_batch = config["batch_size"] * gpu_count
    initial_lr = config["lr"]
    
    run_config = {
        "epoch_num": 3,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.15,
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "packing": "True",
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "run_name": run_name,
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": "True",
        "gradient_accumulation_steps": 4,
        # Label smoothing can significantly increase memory usage (and trigger OOM) for long sequence lengths.
        # Default to 0 for stability; you can tune this later if you have headroom.
        "label_smoothing_factor": 0.0,
        "use_attn_implementation": (
            "kernels-community/vllm-flash-attn3"
            if train_info.get("is_openai", False)
            else ""
        ),
    }

    # Short-job mode: packing can be expensive CPU-side; prefer faster time-to-first-step.
    hours_to_complete = float(train_info.get("hours_to_complete", 0) or 0)
    if hours_to_complete > 0 and hours_to_complete <= 0.75:
        run_config["packing"] = "False"
    
    # Time-aware optimizations for better results in limited time
    if hours_to_complete > 0:
        if hours_to_complete <= 0.75:  # Very short jobs
            # Reduce gradient accumulation for faster updates
            run_config["gradient_accumulation_steps"] = max(1, run_config["gradient_accumulation_steps"] // 2)
            # Add label smoothing for better generalization
            run_config["label_smoothing_factor"] = 0.1
            # Reduce epochs
            run_config["epoch_num"] = 1
            print(f"Time-aware optimization: Very short job ({hours_to_complete:.2f}h) - reduced grad_accum, added label_smoothing, 1 epoch", flush=True)
        elif hours_to_complete <= 1.5:  # Short jobs
            run_config["gradient_accumulation_steps"] = max(2, int(run_config["gradient_accumulation_steps"] * 0.75))
            run_config["label_smoothing_factor"] = 0.05
            run_config["epoch_num"] = 2
            print(f"Time-aware optimization: Short job ({hours_to_complete:.2f}h) - adjusted grad_accum, light label_smoothing, 2 epochs", flush=True)
        else:  # Longer jobs
            run_config["label_smoothing_factor"] = 0.0  # Keep default

    # there are models that do not support packing, so we need to check if the model supports packing
    if run_config["disable_fa"] == "True" or model_architecture.strip().lower() in [
        "optforcausallm"
    ]:
        run_config["packing"] = "False"

    # data_size = get_data_size(train_info["request_path"])

    # Batch size optimization: Increase when memory allows
    # 1. When flash attention is disabled, we save memory and can use larger batches
    if run_config["disable_fa"] == "True":
        # Flash attention uses extra memory, so without it we can increase batch size
        run_config["batch_size"] = int(run_config["batch_size"] * 1.5)
        print(f"  [Batch Size] Flash attention disabled, increased batch size to {run_config['batch_size']}", flush=True)
    
    # 2. When using LoRA, we save significant memory and can use larger batches
    if run_config["use_lora"]:
        # LoRA uses much less memory than full fine-tuning
        run_config["batch_size"] = int(run_config["batch_size"] * 1.3)
        print(f"  [Batch Size] LoRA enabled, increased batch size to {run_config['batch_size']}", flush=True)
    
    # 3. For shorter sequences, we can use larger batches
    if avg_seq_length and avg_seq_length > 0:
        if avg_seq_length < 512:
            # Very short sequences - can use much larger batches
            run_config["batch_size"] = int(run_config["batch_size"] * 1.5)
            print(f"  [Batch Size] Short sequences (avg {avg_seq_length}), increased batch size to {run_config['batch_size']}", flush=True)
        elif avg_seq_length < 1024:
            # Medium sequences - moderate increase
            run_config["batch_size"] = int(run_config["batch_size"] * 1.2)
            print(f"  [Batch Size] Medium sequences (avg {avg_seq_length}), increased batch size to {run_config['batch_size']}", flush=True)
    
    # 4. For time-constrained jobs, prioritize larger batches for faster training
    if hours_to_complete > 0 and hours_to_complete <= 1.0:
        # Short jobs benefit from larger batches (fewer steps needed)
        run_config["batch_size"] = int(run_config["batch_size"] * 1.2)
        print(f"  [Batch Size] Time-constrained job ({hours_to_complete:.2f}h), increased batch size to {run_config['batch_size']}", flush=True)
    
    # 5. Allow manual batch size override via train_info
    if "batch_size_multiplier" in train_info:
        multiplier = float(train_info["batch_size_multiplier"])
        run_config["batch_size"] = int(run_config["batch_size"] * multiplier)
        print(f"  [Batch Size] Manual multiplier ({multiplier}x), batch size: {run_config['batch_size']}", flush=True)
    
    # 6. When using gradient checkpointing, we save memory and can use larger batches
    if run_config["gradient_checkpointing"] == "True":
        # Gradient checkpointing trades compute for memory
        run_config["batch_size"] = int(run_config["batch_size"] * 1.15)
        print(f"  [Batch Size] Gradient checkpointing enabled, increased batch size to {run_config['batch_size']}", flush=True)

    if model_name in FIXED_BS_CONFIG:
        run_config["batch_size"] = FIXED_BS_CONFIG[model_name]["batch_size"]

    # Architecture-specific batch size adjustments (less aggressive when memory optimizations are enabled)
    # These are applied AFTER memory optimizations, so they're less restrictive
    if model_architecture.strip().lower() in [
        "gptneoxforcausallm",
        "gptjforcausallm",
        "phiforcausallm",
        "falconforcausallm",
    ]:
        # Less aggressive reduction if using LoRA or gradient checkpointing
        if run_config["use_lora"] or run_config["gradient_checkpointing"] == "True":
            reduction_factor = 0.75  # Only reduce by 25% instead of 50%
        else:
            reduction_factor = 0.5  # Original 50% reduction
        run_config["batch_size"] = int(run_config["batch_size"] * reduction_factor)
        
        if model_name == "EleutherAI/pythia-160m":  # reduce more
            run_config["batch_size"] = int(run_config["batch_size"] / 1.3)  # Less aggressive
        elif "pythia" in model_name.lower():
            run_config["batch_size"] = int(run_config["batch_size"] / 1.5)  # Less aggressive

    if model_name in ["microsoft/phi-2", "microsoft/phi-1_5"]:
        # Phi models: less aggressive reduction with memory optimizations
        if run_config["use_lora"]:
            run_config["batch_size"] = int(run_config["batch_size"] / 2)  # 50% instead of 75%
        else:
            run_config["batch_size"] = int(run_config["batch_size"] / 3)  # Less aggressive than /4

    if "bloom-560m" in model_name or "bloomz-560m" in model_name:
        # Bloom models: allow larger batch if using optimizations
        if run_config["use_lora"]:
            run_config["batch_size"] = 12  # Increased from 8
        else:
            run_config["batch_size"] = 8

    if model_name == "mistralai/Mistral-7B-v0.1":
        # Less aggressive reduction
        run_config["batch_size"] = int(run_config["batch_size"] * 0.85)  # 85% instead of 75%

    if "falcon" in model_name.lower() and model_architecture.strip().lower() not in ["falconforcausallm"]:
        # Additional falcon check (if not already handled above)
        if run_config["use_lora"]:
            run_config["batch_size"] = int(run_config["batch_size"] * 0.75)  # Less aggressive
        else:
            run_config["batch_size"] = int(run_config["batch_size"] / 2)

    # CRITICAL: Scale LR when batch size changes (maintains training stability)
    # This must be done AFTER all batch size adjustments are complete
    final_effective_batch = run_config["batch_size"] * run_config["gpu_nums"]
    if final_effective_batch != initial_effective_batch:
        from hyperparam_optimizer import scale_lr_for_batch_size
        scaled_lr = scale_lr_for_batch_size(
            base_lr=initial_lr,
            old_batch_size=initial_effective_batch,
            new_batch_size=final_effective_batch,
            param_nums=param_nums,
            scaling_rule="adaptive"
        )
        run_config["learning_rate"] = scaled_lr
        config["lr"] = scaled_lr
        print(f"  [LR Scaling] Batch size changed from {initial_effective_batch} to {final_effective_batch}, scaled LR from {initial_lr:.8f} to {scaled_lr:.8f}", flush=True)

    # AutoML: Get optimal gradient accumulation (optimizes for target batch size)
    data_per_step = run_config["batch_size"] * run_config["gpu_nums"]
    if data_per_step >= 64:
        run_config["gradient_accumulation_steps"] = 1
    else:
        # Use AutoML optimizer for smarter gradient accumulation
        run_config["gradient_accumulation_steps"] = get_optimal_gradient_accumulation(
            model=model_name,
            task_type="instruct",
            param_nums=param_nums,
            batch_size=run_config["batch_size"],
            gpu_count=run_config["gpu_nums"],
            target_total_batch=64,
            max_grad_accum=8
        )

    if model_architecture.strip().lower() in ["gptossforcausallm"]:
        run_config["use_lora"] = False  # currently, gptoss does not support lora

    # OPTIMIZATION: Lookup LR was already checked and used above, so no need to check again
    # The LR in run_config is already set from config["lr"] which uses lookup if available
    # (Note: use_lookup_lr is set in the block above, so we can check train_info directly here)
    if train_info.get("find_lk_lr", False) and lookup_lr is not None:
        print(f"  [LR] Using lookup table LR: {run_config['learning_rate']:.8f}", flush=True)
    else:
        print(f"  [LR] Using calculated LR: {run_config['learning_rate']:.8f}", flush=True)

    base_lr = run_config["learning_rate"]
    run_config["learning_rate"] *= train_info["reg_ratio"]
    print(f"Applied reg_ratio: {base_lr:.8f} * {train_info['reg_ratio']:.6f} = {run_config['learning_rate']:.8f}", flush=True)
    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"], train_info)
    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 500
    train_request["checking_step"] = 80

    # Short-job mode: reduce save/check overhead.
    if hours_to_complete > 0 and hours_to_complete <= 0.75:
        train_request["periodic_save_steps"] = -1

    if param_nums < 1_000_000_000:
        train_request["min_steps"] = max(
            int(train_info["hours_to_complete"] * 100), train_request["min_steps"]
        )

    elif param_nums < 9_000_000_000:
        train_request["min_steps"] = max(
            int(train_info["hours_to_complete"] * 70), train_request["min_steps"]
        )

    return {"train_request": train_request, "run_cmd": run_cmd}
