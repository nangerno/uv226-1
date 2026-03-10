"""
AutoML Hyperparameter Optimization Module

This module extends the LR lookup system to track and optimize other hyperparameters
like LoRA rank, warmup steps, gradient accumulation, etc. based on historical performance.
"""

import json
import os
import hashlib
import math
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

current_dir = os.path.dirname(os.path.abspath(__file__))

# File path for hyperparameter lookup table
HYPERPARAM_FILE = os.path.join(current_dir, "lrs/hyperparams.json")

# Default hyperparameter lookup table structure
DEFAULT_HYPERPARAMS = []

# Load hyperparameter lookup table
def _load_hyperparams() -> Tuple[list, dict]:
    """Load hyperparameter lookup table and create hash-based dictionary for fast lookup."""
    try:
        if os.path.exists(HYPERPARAM_FILE):
            with open(HYPERPARAM_FILE, "r") as f:
                hyperparams_list = json.load(f)
        else:
            hyperparams_list = []
        
        # Create dictionary: hash -> hyperparam_entry for O(1) lookup
        hyperparams_dict = {entry["h"]: entry for entry in hyperparams_list if "h" in entry}
        return hyperparams_list, hyperparams_dict
    except Exception as e:
        print(f"Warning: Error loading hyperparams file {HYPERPARAM_FILE}: {e}", flush=True)
        return [], {}


# Load hyperparameters at module import
_hyperparams_list, _hyperparams_dict = _load_hyperparams()


def hash_model_and_config(model: str, task_type: str, param_nums: int, dataset_size: Optional[int] = None) -> str:
    """Create a hash for model + config combination to identify similar training scenarios."""
    # Include key factors that affect hyperparameter selection
    config_str = f"{model}|{task_type}|{param_nums}"
    if dataset_size:
        # Round dataset size to nearest 1k for better matching
        dataset_bucket = (dataset_size // 1000) * 1000
        config_str += f"|{dataset_bucket}"
    
    config_bytes = config_str.encode('utf-8')
    return hashlib.sha256(config_bytes).hexdigest()


def get_optimal_lora_rank(
    model: str,
    task_type: str,
    param_nums: int,
    hours_to_complete: Optional[float] = None,
    dataset_size: Optional[int] = None,
    default_rank: int = 128
) -> Tuple[int, int]:

    config_hash = hash_model_and_config(model, task_type, param_nums, dataset_size)
    entry = _hyperparams_dict.get(config_hash)
    
    if entry and "lora_rank" in entry and "lora_alpha" in entry:
        rank = entry["lora_rank"]
        alpha = entry["lora_alpha"]
        print(f"  [AutoML] Using learned LoRA rank: {rank} (alpha: {alpha}) from history", flush=True)
        return rank, alpha
    
    # Fallback to time-aware heuristics (existing logic)
    if hours_to_complete and hours_to_complete > 0:
        if hours_to_complete <= 0.75:
            rank, alpha = 256, 512
        elif hours_to_complete <= 1.5:
            rank, alpha = 192, 384
        else:
            rank, alpha = default_rank, default_rank * 2
    else:
        # Model size-based heuristics
        if param_nums < 1_000_000_000:  # < 1B
            rank, alpha = 64, 128
        elif param_nums < 10_000_000_000:  # 1-10B
            rank, alpha = default_rank, default_rank * 2
        else:  # > 10B
            rank, alpha = 256, 512
    
    return rank, alpha


def get_optimal_warmup_steps(
    model: str,
    task_type: str,
    param_nums: int,
    dataset_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    hours_to_complete: Optional[float] = None,
    default_warmup: int = 35
) -> int:

    config_hash = hash_model_and_config(model, task_type, param_nums, dataset_size)
    entry = _hyperparams_dict.get(config_hash)
    
    if entry and "warmup_steps" in entry:
        warmup = entry["warmup_steps"]
        print(f"  [AutoML] Using learned warmup steps: {warmup} from history", flush=True)
        return warmup
    
    # Heuristic-based calculation
    warmup = default_warmup
    
    # Time constraint adjustment
    if hours_to_complete and hours_to_complete > 0:
        if hours_to_complete <= 0.75:
            warmup = 10
        elif hours_to_complete <= 1.5:
            warmup = 20
        else:
            warmup = default_warmup
    
    # Dataset size adjustment: larger datasets benefit from more warmup
    if dataset_size and dataset_size > 0:
        # Scale warmup with sqrt of dataset size (reference: 10k samples)
        dataset_factor = math.sqrt(max(dataset_size, 10_000) / 10_000)
        warmup = int(warmup * min(1.5, max(0.7, dataset_factor)))
    
    # Learning rate adjustment: higher LR needs more warmup
    if learning_rate and learning_rate > 0:
        # Reference LR: 1e-4
        lr_factor = math.sqrt(learning_rate / 1e-4)
        warmup = int(warmup * min(1.3, max(0.8, lr_factor)))
    
    # Model size adjustment: larger models need more warmup
    if param_nums > 0:
        # Reference: 1B params
        model_factor = math.sqrt(max(param_nums, 1_000_000_000) / 1_000_000_000)
        warmup = int(warmup * min(1.2, max(0.9, model_factor)))
    
    # Ensure reasonable bounds
    warmup = max(5, min(100, warmup))
    
    return warmup


def get_optimal_gradient_accumulation(
    model: str,
    task_type: str,
    param_nums: int,
    batch_size: int,
    gpu_count: int,
    target_total_batch: int = 64,
    max_grad_accum: int = 8
) -> int:

    total_batch = batch_size * gpu_count
    
    if total_batch >= target_total_batch:
        return 1  # No need for gradient accumulation
    
    # Calculate required gradient accumulation
    required_grad_accum = math.ceil(target_total_batch / total_batch)
    
    # Cap at max_grad_accum to avoid too slow training
    grad_accum = min(required_grad_accum, max_grad_accum)
    
    return grad_accum


def scale_lr_for_batch_size(
    base_lr: float,
    old_batch_size: int,
    new_batch_size: int,
    param_nums: int,
    scaling_rule: str = "adaptive"
) -> float:

    if old_batch_size <= 0 or new_batch_size <= 0:
        return base_lr
    
    if old_batch_size == new_batch_size:
        return base_lr
    
    batch_ratio = new_batch_size / old_batch_size
    
    if scaling_rule == "linear":
        # Linear scaling: LR ∝ batch_size (works well for small models)
        lr_scale = batch_ratio
    elif scaling_rule == "sqrt":
        # Square root scaling: LR ∝ sqrt(batch_size) (more conservative)
        lr_scale = math.sqrt(batch_ratio)
    else:  # adaptive
        # Adaptive scaling based on model size (same logic as in calculate_continuous_lr)
        if param_nums < 1_000_000_000:  # < 1B: more linear
            lr_scale = batch_ratio ** 0.7
        elif param_nums < 10_000_000_000:  # 1-10B: sqrt scaling
            lr_scale = math.sqrt(batch_ratio)
        else:  # > 10B: sub-linear
            lr_scale = batch_ratio ** 0.25
    
    new_lr = base_lr * lr_scale
    
    # Ensure reasonable bounds
    new_lr = max(1e-6, min(1e-3, new_lr))
    
    return new_lr


def get_optimal_lr_batch_pair(
    model: str,
    task_type: str,
    param_nums: int,
    dataset_size: Optional[int] = None,
    hours_to_complete: Optional[float] = None,
    gpu_count: int = 1,
    base_lr: Optional[float] = None,
    base_batch_size: Optional[int] = None,
    min_batch_size: int = 1,
    max_batch_size: int = 128
) -> Tuple[Optional[float], Optional[int], str]:

    config_hash = hash_model_and_config(model, task_type, param_nums, dataset_size)
    
    # CRITICAL: Exclude GRPO from test loss-based optimization (GRPO uses reward, not loss)
    # Search for historical entries with similar config
    # We'll look for entries with same hash (exact match) or similar model size
    matching_entries = []
    
    # Exact match
    if config_hash in _hyperparams_dict and task_type != "grpo":
        entry = _hyperparams_dict[config_hash]
        # CRITICAL: Prioritize eval_loss (test loss) for all tasks except GRPO
        entry_loss = entry.get("eval_loss")
        if entry_loss is not None:
            matching_entries.append(entry)
    
    # Similar model size (within 20% difference)
    if param_nums > 0 and task_type != "grpo":
        for entry in _hyperparams_list:
            entry_params = entry.get("param_nums", 0)
            if entry_params > 0:
                param_ratio = min(param_nums, entry_params) / max(param_nums, entry_params)
                if param_ratio >= 0.8:  # Within 20%
                    if entry.get("task_type") == task_type:
                        # CRITICAL: Prioritize eval_loss (test loss) for all tasks except GRPO
                        entry_loss = entry.get("eval_loss")
                        if entry_loss is not None:
                            # Weight by similarity (closer params = higher weight)
                            entry_copy = entry.copy()
                            entry_copy["similarity_weight"] = param_ratio
                            matching_entries.append(entry_copy)
    
    if matching_entries:
        # Sort by test loss (lower is better) and similarity
        matching_entries.sort(key=lambda e: (
            e.get("eval_loss", float('inf')),
            -e.get("similarity_weight", 0.0)
        ))
        
        best_entry = matching_entries[0]
        best_lr = best_entry.get("learning_rate")
        best_batch = best_entry.get("batch_size")
        best_loss = best_entry.get("eval_loss")
        
        # If we have both LR and batch size from history, use them
        if best_lr is not None and best_batch is not None:
            # Adjust batch size for current GPU count if needed
            # Historical batch might be for different GPU setup
            if best_batch > 0:
                # Scale batch size proportionally to GPU count (if historical had different GPU count)
                # For now, assume historical was for similar setup, but cap to reasonable range
                optimal_batch = max(min_batch_size, min(max_batch_size, best_batch))
                optimal_lr = best_lr
                
                # If time constraint is different, adjust LR slightly
                if hours_to_complete and hours_to_complete > 0:
                    historical_hours = best_entry.get("hours_to_complete")
                    if historical_hours and historical_hours > 0:
                        time_ratio = hours_to_complete / historical_hours
                        if time_ratio < 0.5:  # Much shorter time
                            optimal_lr *= 1.2  # Increase LR for faster convergence
                        elif time_ratio > 2.0:  # Much longer time
                            optimal_lr *= 0.9  # Decrease LR for more stable training
                
                print(f"  [AutoML] Using optimal LR-batch pair from history: LR={optimal_lr:.8f}, batch={optimal_batch}, test_loss={best_loss:.6f}", flush=True)
                return optimal_lr, optimal_batch, f"historical_best_loss_{best_loss:.6f}"
        
        # If we only have LR from history, use it with base batch size
        if best_lr is not None:
            if base_batch_size:
                optimal_batch = base_batch_size
            else:
                optimal_batch = max(min_batch_size, min(max_batch_size, 32))  # Default
            print(f"  [AutoML] Using optimal LR from history: LR={best_lr:.8f}, batch={optimal_batch} (from base)", flush=True)
            return best_lr, optimal_batch, f"historical_lr_best_loss_{best_loss:.6f}"
    
    # No historical data: use base values if provided, otherwise return None
    if base_lr is not None and base_batch_size is not None:
        return base_lr, base_batch_size, "no_history_using_base"
    
    return None, None, "no_history_no_base"


def optimize_for_test_loss_per_time(
    model: str,
    task_type: str,
    param_nums: int,
    dataset_size: Optional[int] = None,
    hours_to_complete: Optional[float] = None,
    gpu_count: int = 1
) -> Tuple[Optional[float], Optional[int], str]:

    # CRITICAL: Exclude GRPO from test loss-based optimization (GRPO uses reward, not loss)
    if task_type == "grpo":
        return None, None, "grpo_excluded_from_test_loss_optimization"
    
    if hours_to_complete is None or hours_to_complete <= 0:
        # No time constraint, use regular optimization
        return None, None, "no_time_constraint"
    
    config_hash = hash_model_and_config(model, task_type, param_nums, dataset_size)
    
    # Find historical entries with similar config and time constraints
    matching_entries = []
    
    # Exact match
    if config_hash in _hyperparams_dict:
        entry = _hyperparams_dict[config_hash]
        # CRITICAL: Prioritize eval_loss (test loss) for all tasks except GRPO
        entry_loss = entry.get("eval_loss")
        if entry_loss is not None and entry.get("hours_to_complete"):
            matching_entries.append(entry)
    
    # Similar model size and time constraints
    if param_nums > 0:
        for entry in _hyperparams_list:
            entry_params = entry.get("param_nums", 0)
            entry_hours = entry.get("hours_to_complete")
            if entry_params > 0 and entry_hours and entry_hours > 0:
                param_ratio = min(param_nums, entry_params) / max(param_nums, entry_params)
                time_ratio = min(hours_to_complete, entry_hours) / max(hours_to_complete, entry_hours)
                # Within 20% for params and 30% for time
                if param_ratio >= 0.8 and time_ratio >= 0.7:
                    if entry.get("task_type") == task_type:
                        # CRITICAL: Prioritize eval_loss (test loss) for all tasks except GRPO
                        entry_loss = entry.get("eval_loss")
                        if entry_loss is not None:
                            entry_copy = entry.copy()
                            # Weight by similarity and efficiency (lower test_loss/hours is better)
                            test_loss = entry.get("eval_loss", float('inf'))
                            efficiency = test_loss / entry_hours if entry_hours > 0 else float('inf')
                            entry_copy["similarity_weight"] = param_ratio * time_ratio
                            entry_copy["efficiency"] = efficiency
                            matching_entries.append(entry_copy)
    
    if matching_entries:
        # Sort by efficiency (test_loss / hours) - lower is better
        matching_entries.sort(key=lambda e: (
            e.get("efficiency", float('inf')),
            e.get("eval_loss", float('inf')),
            -e.get("similarity_weight", 0.0)
        ))
        
        best_entry = matching_entries[0]
        best_lr = best_entry.get("learning_rate")
        best_batch = best_entry.get("batch_size")
        best_loss = best_entry.get("eval_loss")
        best_hours = best_entry.get("hours_to_complete")
        efficiency = best_entry.get("efficiency", float('inf'))
        
        if best_lr is not None and best_batch is not None:
            optimal_batch = max(1, min(128, best_batch))
            optimal_lr = best_lr
            
            # Adjust for current time constraint if different
            if best_hours and best_hours > 0:
                time_ratio = hours_to_complete / best_hours
                if time_ratio < 0.7:  # Much shorter time - need faster convergence
                    optimal_lr *= 1.15  # Increase LR
                    optimal_batch = min(optimal_batch * 2, 128)  # Increase batch for faster training
                elif time_ratio > 1.5:  # Much longer time - can be more stable
                    optimal_lr *= 0.95  # Slightly decrease LR
                    optimal_batch = max(optimal_batch // 2, 1)  # Can use smaller batch
            
            print(f"  [AutoML] Time-optimized: LR={optimal_lr:.8f}, batch={optimal_batch}, efficiency={efficiency:.6f} (test_loss={best_loss:.6f} in {best_hours:.2f}h)", flush=True)
            return optimal_lr, optimal_batch, f"time_optimized_efficiency_{efficiency:.6f}"
        
        if best_lr is not None:
            optimal_batch = max(1, min(128, 32))  # Default
            return best_lr, optimal_batch, f"time_optimized_lr_efficiency_{efficiency:.6f}"
    
    return None, None, "no_time_optimized_history"


def update_hyperparams(
    task_type: str,
    model: str,
    param_nums: int,
    eval_loss: Optional[float] = None,
    train_loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:

    global _hyperparams_list, _hyperparams_dict
    
    if metadata is None:
        return False
    
    dataset_size = metadata.get("dataset_size")
    config_hash = hash_model_and_config(model, task_type, param_nums, dataset_size)
    
    # Load current lookup table
    try:
        hyperparams_list = _hyperparams_list.copy()
    except Exception:
        hyperparams_list = []
    
    # Find existing entry
    existing_entry = None
    existing_index = -1
    if config_hash in _hyperparams_dict:
        existing_entry = _hyperparams_dict[config_hash]
        for i, entry in enumerate(hyperparams_list):
            if entry.get("h") == config_hash:
                existing_index = i
                break
    else:
        # Linear search fallback
        for i, entry in enumerate(hyperparams_list):
            if entry.get("h") == config_hash:
                existing_entry = entry
                existing_index = i
                break
    
    # Determine if we should update
    # CRITICAL: Prioritize eval_loss (test loss) for all tasks except GRPO
    # GRPO uses reward, not loss, so we skip test loss optimization for GRPO
    if task_type == "grpo":
        # For GRPO, use train_loss or skip if no loss available
        loss_to_compare = train_loss if train_loss is not None else None
        existing_loss = existing_entry.get("train_loss") if existing_entry else None
    else:
        # For all other tasks, prioritize eval_loss (test loss)
        loss_to_compare = eval_loss if eval_loss is not None else train_loss
        existing_loss = None
        if existing_entry:
            existing_loss = existing_entry.get("eval_loss")
            if existing_loss is None:
                existing_loss = existing_entry.get("train_loss")  # Fallback only if eval_loss not available
    
    # Update if: no existing entry, or new loss is better, or existing has no loss
    should_update = False
    if existing_entry is None:
        should_update = True
        print(f"  [AutoML] New hyperparameter entry for {model[:50]}...", flush=True)
    elif loss_to_compare is not None:
        if existing_loss is None:
            should_update = True
            print(f"  [AutoML] Existing entry has no loss, updating with loss {loss_to_compare:.6f}", flush=True)
        elif loss_to_compare < existing_loss:
            should_update = True
            print(f"  [AutoML] Better loss found: {loss_to_compare:.6f} < {existing_loss:.6f}, updating hyperparams", flush=True)
        else:
            print(f"  [AutoML] Existing loss {existing_loss:.6f} is better than {loss_to_compare:.6f}, keeping existing hyperparams", flush=True)
    else:
        if existing_entry is None:
            should_update = True
    
    if should_update:
        new_entry = {
            "h": config_hash,
            "model": model,
            "task_type": task_type,
            "param_nums": param_nums,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add loss information
        if eval_loss is not None:
            new_entry["eval_loss"] = eval_loss
        if train_loss is not None:
            new_entry["train_loss"] = train_loss
        
        # Add hyperparameter metadata
        if "lora_rank" in metadata:
            new_entry["lora_rank"] = metadata["lora_rank"]
        if "lora_alpha" in metadata:
            new_entry["lora_alpha"] = metadata["lora_alpha"]
        if "warmup_steps" in metadata:
            new_entry["warmup_steps"] = metadata["warmup_steps"]
        if "gradient_accumulation_steps" in metadata:
            new_entry["gradient_accumulation_steps"] = metadata["gradient_accumulation_steps"]
        if "dataset_size" in metadata:
            new_entry["dataset_size"] = metadata["dataset_size"]
        if "learning_rate" in metadata:
            new_entry["learning_rate"] = metadata["learning_rate"]
        # CRITICAL: Track batch_size for joint LR-batch optimization
        if "batch_size" in metadata:
            new_entry["batch_size"] = metadata["batch_size"]
        if "effective_batch_size" in metadata:
            new_entry["effective_batch_size"] = metadata["effective_batch_size"]
        if "hours_to_complete" in metadata:
            new_entry["hours_to_complete"] = metadata["hours_to_complete"]
        # CRITICAL: Track reg_ratio for task-specific optimization
        if "reg_ratio" in metadata:
            new_entry["reg_ratio"] = metadata["reg_ratio"]
        
        # Update or add entry
        if existing_index >= 0:
            hyperparams_list[existing_index] = new_entry
            print(f"  [AutoML] Updated hyperparameter entry for {model[:50]}...", flush=True)
        else:
            hyperparams_list.append(new_entry)
            print(f"  [AutoML] Added new hyperparameter entry for {model[:50]}...", flush=True)
        
        # Save updated lookup table
        try:
            # Create backup
            backup_file = HYPERPARAM_FILE + ".backup"
            if os.path.exists(HYPERPARAM_FILE):
                import shutil
                shutil.copy2(HYPERPARAM_FILE, backup_file)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(HYPERPARAM_FILE), exist_ok=True)
            
            # Write updated table
            with open(HYPERPARAM_FILE, "w") as f:
                json.dump(hyperparams_list, f, indent=4)
            
            print(f"  [AutoML] Successfully updated {HYPERPARAM_FILE}", flush=True)
            
            # Reload global variables
            _hyperparams_list, _hyperparams_dict = _load_hyperparams()
            
            return True
        except Exception as e:
            print(f"  [AutoML] Error saving hyperparameter file {HYPERPARAM_FILE}: {e}", flush=True)
            return False
    
    return False
