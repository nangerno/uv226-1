from model_utility import get_model_architecture, get_model_num_params, get_use_liger, disable_flash_attention, get_gradient_checkpointing, get_gpu_count
from copy import deepcopy
from lrs_lookup import get_dpo_lr

DPO_CONFIG = {
    "0_1_b": {
        "lr": 1.35e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 16,
    },
    "1_2_b": {
        "lr": 8.7e-6,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 12,
    },
    "2_4_b": {
        "lr": 6.5e-6,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 12,
        "use_lora": True
    },
    "4_5_b": {
        "lr": 6.25e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 12,
        "use_lora": True
    },
    "5_9_b": {
        "lr": 7.5e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 8,
        "use_lora": True
    },
    "9_12_b": {
        "lr": 5e-6,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 32,
        "gradient_checkpointing": False
    },
    "12_14_b": {
        "lr": 8.5e-6,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 24,
        "gradient_checkpointing": False
    },
    "14_15_b": {
        "lr": 8.5e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 18,
        "gradient_checkpointing": False
    },
    "15_40_b": {
        "lr": 8e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 16,
        "gradient_checkpointing": False
    },
    "40_80_b": {
        "lr": 8e-6,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 8,
        "gradient_checkpointing": False
    }        
}

for key in DPO_CONFIG:
    DPO_CONFIG[key]["label"] = key
    

def get_config(param_nums: int) -> dict:
    result = None
    if param_nums < 1_000_000_000:
        result = DPO_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        result = DPO_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        result = DPO_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        result = DPO_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        result = DPO_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        result = DPO_CONFIG["9_12_b"]
    elif param_nums < 14_000_000_000:
        result = DPO_CONFIG["12_14_b"]
    elif param_nums < 15_000_000_000:  
        result = DPO_CONFIG["14_15_b"]
    elif param_nums < 35_000_000_000:
        result = DPO_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        result = DPO_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported", flush=True)
        result = {
            "lr": 4e-5,
            "distributed": "ds",
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }
    if param_nums < 4_000_000_000 and param_nums > 1_330_000_000:
        result["gpu_count"] = 2
    if param_nums > 13_330_000_000: # 8 GPUs for 13.3B
        result["gpu_count"] = 8
    return result


def get_run_cmd(config: dict, gpu_nums: int, train_info: dict = None):
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "disable_fa",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")
    gpu_nums = get_gpu_count()
    start_cmd = "python"
    run_type = config.get("distributed", "ddp")
    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = f"deepspeed"

    template = (
        start_cmd
        + """ train_dpo.py \
    --request_path {request_path} \
    --bf16 True \
    --report_to wandb \
    --output_dir {output_dir} \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --logging_steps 5 \
    --learning_rate {learning_rate} \
    --weight_decay {weight_decay} \
    --warmup_steps {warmup_steps} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing {gradient_checkpointing} \
    --optim {optimizer} \
    --use_liger {use_liger} --disable_fa {disable_fa} \
    --label_smoothing_factor {label_smoothing_factor}"""
    )

    if config.get("use_lora", False):
        # Time-aware LoRA rank: Higher rank for short jobs for faster adaptation
        if train_info:
            hours_to_complete = float(train_info.get("hours_to_complete", 0) or 0)
            if hours_to_complete > 0 and hours_to_complete <= 0.75:
                lora_r, lora_alpha = 256, 512  # Higher rank for very short jobs
            elif hours_to_complete <= 1.5:
                lora_r, lora_alpha = 192, 384  # Medium-high rank for short jobs
            else:
                lora_r, lora_alpha = 128, 256  # Standard rank for longer jobs
            if hours_to_complete > 0 and hours_to_complete <= 1.5:
                print(f"Time-aware LoRA: Using rank {lora_r} (alpha {lora_alpha}) for {hours_to_complete:.2f}h job", flush=True)
        else:
            lora_r, lora_alpha = 128, 256  # Default
        template += (
            f" --use_peft --lora_r {lora_r} --lora_alpha {lora_alpha} --lora_target_modules all-linear"
        )

    if run_type == "ds":
        template = template + """ --deepspeed ds_config/zero3.json"""

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))
    
    if config.get("use_attn_implementation", ""):
        use_attn_implementation = config["use_attn_implementation"]
        template = template + f""" --use_attn_implementation {use_attn_implementation}"""
        
    return template


def get_training_json(train_info: dict) -> dict:
    model_name = train_info["model_name"]
    model_path = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    config = get_config(param_nums)
    run_config = {
        "epoch_num": 3,
        "batch_size": config["batch_size"],
        "learning_rate": config["lr"],
        "min_lr_rate": 0.05,  # 5 % floor gives the cosine schedule room to converge properly
        "use_liger": get_use_liger(model_architecture),
        "optimizer": "paged_adamw_8bit",
        "use_lora": config.get("use_lora", False),
        "disable_fa": disable_flash_attention(model_architecture, model_name),
        "gpu_nums": config["gpu_count"],
        "output_dir": train_info["output_dir"],
        "request_path": train_info["request_path"],
        "distributed": config.get("distributed", "ddp"),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": 1,
        # Keep default at 0 for stability/speed; label smoothing can increase memory usage.
        "label_smoothing_factor": 0.0,
        "use_attn_implementation": "kernels-community/vllm-flash-attn3" if train_info.get("is_openai", False) else "",
        # Full fine-tuning: stronger L2 regularization; LoRA: lighter decay (few adapter params)
        "weight_decay": 0.01 if config.get("use_lora", False) else 0.05,
    }
    
    if not config.get("gradient_checkpointing", True):
        run_config["gradient_checkpointing"] = False
    
    total_batch_size = run_config["batch_size"] * run_config["gpu_nums"]
    if total_batch_size < 64:
        run_config["gradient_accumulation_steps"] = min(4, int(64 / total_batch_size))
    
    # Time-aware optimizations for better results in limited time
    hours_to_complete = float(train_info.get("hours_to_complete", 0) or 0)
    warmup_steps = 35  # Default warmup
    if hours_to_complete > 0:
        if hours_to_complete <= 0.75:  # Very short jobs
            # Reduce gradient accumulation for faster updates
            run_config["gradient_accumulation_steps"] = max(1, run_config["gradient_accumulation_steps"] // 2)
            # Add label smoothing for better generalization in limited time
            run_config["label_smoothing_factor"] = 0.1
            # Reduce epochs
            run_config["epoch_num"] = 1
            # Reduce warmup steps to reach effective LR faster
            warmup_steps = 10
            print(f"Time-aware optimization: Very short job ({hours_to_complete:.2f}h) - reduced grad_accum, added label_smoothing, 1 epoch, 10 warmup steps", flush=True)
        elif hours_to_complete <= 1.5:  # Short jobs
            run_config["gradient_accumulation_steps"] = max(1, int(run_config["gradient_accumulation_steps"] * 0.75))
            run_config["label_smoothing_factor"] = 0.05
            run_config["epoch_num"] = 2
            warmup_steps = 20
            print(f"Time-aware optimization: Short job ({hours_to_complete:.2f}h) - adjusted grad_accum, light label_smoothing, 2 epochs, 20 warmup steps", flush=True)
        else:  # Longer jobs — label smoothing 0.05 reduces overconfidence without memory cost
            run_config["label_smoothing_factor"] = 0.05
    
    run_config["warmup_steps"] = warmup_steps
    
    if train_info["find_lk_lr"]:
        # get lr from lrs_lookup.py
        lr = get_dpo_lr(model_name)
        if lr is not None:
            print(f"Using lr from lk: {lr}", flush=True)
            run_config["learning_rate"] = lr
        else:
            print(f"Using lr from config: {run_config['learning_rate']}", flush=True)
    
    base_lr = run_config["learning_rate"]
    run_config["learning_rate"] *= train_info["reg_ratio"]
    print(f"Applied reg_ratio: {base_lr:.8f} * {train_info['reg_ratio']:.6f} = {run_config['learning_rate']:.8f}", flush=True)
    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"], train_info)
    if run_config["disable_fa"] == "False":
        run_cmd = run_cmd + " --padding_free True"
    train_request = deepcopy(train_info)
    train_request["save_before_remaining_time"] = 3
    train_request["min_steps"] = 100
    train_request["min_steps_per_epoch"] = train_request["min_steps"]
    train_request["adjust_batch_size"] = False
    train_request["periodic_save_steps"] = 150  # finer checkpoint selection → lower test loss
    # Adaptive checking_step: ~12 % of estimated total steps, clamped to [80, 400].
    dataset_size = None
    try:
        from model_utility import get_data_size
        if train_info.get("request_path"):
            dataset_size = get_data_size(train_info["request_path"])
    except Exception:
        pass
    if dataset_size and dataset_size > 0:
        bs  = run_config["batch_size"]
        ga  = run_config["gradient_accumulation_steps"]
        gn  = run_config["gpu_nums"]
        ep  = run_config["epoch_num"]
        est_steps = max(1, dataset_size * ep // (bs * ga * gn))
        train_request["checking_step"] = max(80, min(400, int(est_steps * 0.12)))
    else:
        train_request["checking_step"] = 80

    # Short-job mode: reduce save overhead.
    hours_to_complete = float(train_info.get("hours_to_complete", 0) or 0)
    if hours_to_complete > 0 and hours_to_complete <= 0.75:
        train_request["periodic_save_steps"] = -1
    
    return {
        "train_request": train_request,
        "run_cmd": run_cmd
    }
