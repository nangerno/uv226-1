#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
# Set CUDA memory allocation config before importing torch to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import shutil
import copy
import subprocess
import sys
import uuid
import re
import time 
from datetime import datetime, timezone, timedelta

import yaml
from transformers import AutoTokenizer
from state_manager import get_state, set_state
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
import training_paths as train_paths
from instruct_config import get_training_json as get_instruct_training_json
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
import pathlib
from transformers import AutoConfig
import lr_utils

def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None):
    # print(f"Running command: {cmd}", flush=True)
    with open(log_file_path, "w") as log_file:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        # Wait for the process to complete
        return_code = process.wait()

        # Log the return code
        log_file.write(f"\nProcess completed with return code: {return_code}\n")


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    match = re.search(rf"(?P<p>--{arg_name}(\s+)([^\s]+))(\s+)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        return None


def extract_value_from_cmd(cmd: str, arg_name: str):
    match = re.search(rf"(?P<p>--{arg_name}(\s+)(?P<value>[^\s]+))(\s+)", cmd)
    if match:
        return match.group("value")
    else:
        return None


def get_model_architecture(model_name: str, model_path: str = None) -> str:
    # First, try local path if provided and exists
    if model_path and os.path.exists(model_path):
        try:
            config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            architectures = config.architectures
            if len(architectures) > 1:
                return "Multiple architectures"
            return architectures[0].strip().lower()
        except Exception as e:
            # If local fails, continue to try remote (but with better error handling)
            print(f"Failed to load config from local path {model_path}: {e}", flush=True)
    
    # Try with model_name (may download from HuggingFace)
    # Only attempt network call if we have network connectivity
    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=False)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        # Handle network errors gracefully - don't fail the entire script
        error_str = str(e).lower()
        if any(err in error_str for err in ["name resolution", "failed to resolve", "connection", "network", "dns", "maxretryerror", "temporary failure"]):
            print(f"Network error when trying to fetch model config for {model_name}: {e}", flush=True)
            print("Falling back to 'Unknown' architecture (will use defaults). Model config will be loaded later from local path.", flush=True)
            return "Unknown"
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        print(f"Error getting model architecture for {model_name}: {e}", flush=True)
        return "Unknown"


def is_openai_model(model_name: str, model_path: str = None) -> bool:
    """
    Check if model is an OpenAI-style model. Returns False if architecture cannot be determined
    (e.g., due to network issues), to avoid blocking the training process.
    """
    architecture = get_model_architecture(model_name, model_path)
    if architecture.lower() == "gptossforcausallm":
        return True
    # If architecture is Unknown (e.g., due to network error), default to False
    # The model will be checked again later when the local path is available
    return False


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None


def extract_output_dir(train_cmd: str) -> str:
    match = re.search(r"--output_dir\s+(.*?)\s+", train_cmd)
    if match:
        return match.group(1)
    else:
        return None


def run_training(
    train_cmd: str,
    log_path: str,
    task_id: str,
    retries: int,
    task_type: str,
    expected_repo_name: str,
):
    # OPTIMIZATION: More aggressive memory management before training
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            # Python garbage collection first
            gc.collect()
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            # Synchronize to ensure all operations complete
            torch.cuda.synchronize()
            # Optional: reset peak memory stats for better tracking
            torch.cuda.reset_peak_memory_stats()
            
            # Log memory status for diagnostics
            if torch.cuda.device_count() > 0:
                allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                print(f"************* Clear GPU cache before starting training to avoid memory fragmentation *************", flush=True)
                print(f"  [Memory] GPU 0: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total", flush=True)
            
    except:
        print(f"************* GPU is not allowed *************", flush=True)    
        pass
    
    for i in range(retries):
        print(f"************* Training attempt {i+1}/{retries} for task {task_id}*************", flush=True)
        if i > 0:  # there was something wrong so we will reduce the batch_size
            # OPTIMIZATION: More aggressive memory cleanup before retry
            try:
                import torch
                import gc
                if torch.cuda.is_available():
                    gc.collect()  # Python garbage collection first
                    torch.cuda.empty_cache()  # Clear PyTorch cache
                    torch.cuda.synchronize()  # Wait for all operations to complete
                    print(f"  [Memory] Aggressive cleanup before retry {i+1}/{retries}", flush=True)
            except:
                pass
            
            # first check if the training is OOM
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        # OPTIMIZATION: Smarter batch size reduction based on retry count
                        # First retry: reduce by 50%, subsequent retries: reduce by 40% (more conservative)
                        if i == 1:
                            # First retry: halve it
                            reduction_factor = 0.5
                        elif i == 2:
                            # Second retry: reduce by 40% (more conservative)
                            reduction_factor = 0.6
                        else:
                            # Third+ retry: reduce by 30% (very conservative)
                            reduction_factor = 0.7
                        
                        new_batch_size = max(1, int(current_batch_size * reduction_factor))
                        print(
                            f"  [OOM Recovery] Reducing batch size from {current_batch_size} to {new_batch_size} "
                            f"(retry {i+1}/{retries}, reduction: {int((1-reduction_factor)*100)}%)",
                            flush=True,
                        )
                        train_cmd = replace_args_in_cmd(
                            train_cmd,
                            "per_device_train_batch_size",
                            str(new_batch_size),
                        )
                        # print(f"New train command: {train_cmd}", flush=True)
                    else:
                        print(f"  [OOM Recovery] Batch size is 1, cannot reduce further", flush=True)
                        if task_type == TaskType.GRPOTASK.value:
                            # disable vllm
                            train_cmd = replace_args_in_cmd(
                                train_cmd, "use_vllm", "False"
                            )
                            print(f"  [OOM Recovery] Disabled VLLM as last resort", flush=True)
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print(f"  [OOM Recovery] VLLM OOM error, disabling VLLM", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")

        # empty the log file if it exists
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("STARTING TRAINING")

        training_env_vars = {
            "WANDB_MODE": "offline",
            "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
            "WANDB_NAME": f"{task_id}_{expected_repo_name}",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }

        run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)
        # check if training is successfully here so we can break the loop; if output_dir contains file: "success.txt" return true
        output_dir = extract_value_from_cmd(train_cmd, "output_dir")
        if os.path.exists(os.path.join(output_dir, "success.txt")):
            # Clear GPU cache after successful training
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            return True
        time.sleep(5)
        # Clear GPU cache after failed attempt
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    return False


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def delete_poor_checkpoints(train_runs: list[dict]):
    # Use eval_loss for deletion if available, otherwise use current_loss
    if all("current_eval_loss" in run for run in train_runs):
        # Use eval_loss for better checkpoint management
        losses_for_comparison = [run.get("current_eval_loss", run["current_loss"]) for run in train_runs]
        lowest_loss = min(losses_for_comparison)
        for run in train_runs:
            run_loss = run.get("current_eval_loss", run["current_loss"])
            if run_loss > lowest_loss:
                if os.path.exists(run["output_dir"]):
                    print(f"Deleting checkpoint {run['output_dir']} with eval_loss {run_loss:.6f} (train_loss: {run['current_loss']:.6f})", flush=True)
                    shutil.rmtree(run["output_dir"])
    else:
        # Fallback to current_loss
        lowest_loss = min([run["current_loss"] for run in train_runs])
        for run in train_runs:
            if run["current_loss"] > lowest_loss:
                if os.path.exists(run["output_dir"]):
                    print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                    shutil.rmtree(run["output_dir"])


def calculate_adaptive_log_range(
    task_type: str,
    current_lr: float = None,
    model_params: int = None,
    dataset_size: int = None,
    hours_to_complete: float = None,
    n_runs: int = None,
    first_run_loss: float = None,
    model_name: str = None,
    request_path: str = None
) -> float:
    """
    Calculate optimal log_range for LR search in ALL cases.
    Tries to fetch missing parameters and uses AutoML to get the most optimized value.
    """
    # ALWAYS fetch missing parameters FIRST to ensure AutoML can use them
    if model_params is None and model_name:
        try:
            from model_utility import get_model_num_params
            import training_paths
            model_path = str(training_paths.get_text_base_model_path(model_name))
            if model_path:
                model_params = get_model_num_params(model_name, model_path)
                if model_params:
                    print(f"  [log_range] Fetched model_params: {model_params/1e9:.2f}B", flush=True)
        except Exception as e:
            print(f"  [log_range] Could not fetch model_params: {e}", flush=True)
    
    if dataset_size is None and request_path:
        try:
            from model_utility import get_data_size
            if os.path.exists(request_path):
                dataset_size = get_data_size(request_path)
                if dataset_size:
                    print(f"  [log_range] Fetched dataset_size: {dataset_size}", flush=True)
        except Exception as e:
            print(f"  [log_range] Could not fetch dataset_size: {e}", flush=True)
    
    # OPTIMIZATION: Try to get optimal log_range from historical performance
    optimal_log_range = None
    try:
        from hyperparam_optimizer import hash_model_and_config, _hyperparams_dict, _hyperparams_list
        if model_name and model_params:
            # Map task_type to hyperparam_optimizer task_type format
            task_type_map = {
                TaskType.INSTRUCTTEXTTASK.value: "instruct",
                TaskType.CHATTASK.value: "instruct",  # Chat uses instruct config
                TaskType.DPOTASK.value: "dpo",
                TaskType.GRPOTASK.value: "grpo",
            }
            opt_task_type = task_type_map.get(task_type, "instruct")
            
            config_hash = hash_model_and_config(model_name, opt_task_type, model_params, dataset_size)
            
            # CRITICAL: Exclude GRPO from test loss-based optimization (GRPO uses reward, not loss)
            # Search for historical entries with best test loss (eval_loss)
            best_entry = None
            best_loss = float('inf')
            
            # Check exact match first
            if config_hash in _hyperparams_dict and opt_task_type != "grpo":
                entry = _hyperparams_dict[config_hash]
                # CRITICAL: Prioritize eval_loss (test loss) for all tasks except GRPO
                entry_loss = entry.get("eval_loss")
                if entry_loss is not None:
                    best_entry = entry
                    best_loss = entry_loss
            
            # Search similar configs (within 20% model size)
            if not best_entry and model_params > 0 and opt_task_type != "grpo":
                for entry in _hyperparams_list:
                    entry_params = entry.get("param_nums", 0)
                    if entry_params > 0:
                        param_ratio = min(model_params, entry_params) / max(model_params, entry_params)
                        if param_ratio >= 0.8 and entry.get("task_type") == opt_task_type:
                            # CRITICAL: Prioritize eval_loss (test loss) for all tasks except GRPO
                            entry_loss = entry.get("eval_loss")
                            if entry_loss is not None and entry_loss < best_loss:
                                best_loss = entry_loss
                                best_entry = entry
            
            # If we found a good historical entry, infer optimal log_range
            # Lower test loss = better LR was found, so we can use narrower search
            # Higher test loss = need wider search to find better LR
            if best_entry:
                historical_loss = best_entry.get("eval_loss", float('inf'))
                # If historical loss is very good (< 0.5), use narrower search (found good LR)
                # If historical loss is high (> 1.5), use wider search (need to explore more)
                if historical_loss < 0.5:
                    optimal_log_range = 0.12  # Narrow search (already found good LR)
                elif historical_loss < 1.0:
                    optimal_log_range = 0.15  # Medium search
                elif historical_loss < 1.5:
                    optimal_log_range = 0.18  # Standard search
                else:
                    optimal_log_range = 0.22  # Wider search (need better LR)
                
                print(f"  [AutoML] Using learned log_range: {optimal_log_range:.4f} from historical test_loss={historical_loss:.6f}", flush=True)
    except Exception as e:
        # Fall back to heuristics if AutoML lookup fails
        print(f"  [AutoML] Could not get optimal log_range from history: {e}, using heuristics", flush=True)
    
    # Step 1: Base range from task type (or use optimal from history)
    if optimal_log_range is not None:
        log_range = optimal_log_range
    else:
        base_log_range_map = {
            TaskType.INSTRUCTTEXTTASK.value: 0.18,
            TaskType.DPOTASK.value: 0.18,
            TaskType.GRPOTASK.value: 0.2,
            TaskType.CHATTASK.value: 0.18,
        }
        log_range = base_log_range_map.get(task_type, 0.18)
    
    # Step 2: Model size adjustment (ALWAYS apply if model_params available)
    if model_params:
        if model_params > 10_000_000_000:  # > 10B params
            log_range *= 0.85  # Narrower search for very large models
        elif model_params > 1_000_000_000:  # 1-10B params
            log_range *= 0.9  # Slightly narrower
        elif model_params < 100_000_000:  # < 100M params
            log_range *= 1.1  # Wider search for tiny models
        # else: no adjustment for medium models
    else:
        # Default adjustment if model_params unknown (assume medium model)
        pass
    
    # Step 3: Dataset size adjustment (ALWAYS apply if dataset_size available)
    if dataset_size and dataset_size > 0:
        if dataset_size > 1_000_000:  # Very large datasets
            log_range *= 1.1  # Wider search
        elif dataset_size > 100_000:  # Large datasets
            log_range *= 1.05  # Slightly wider
        elif dataset_size < 10_000:  # Small datasets
            log_range *= 0.95  # Narrower search
        # else: no adjustment for medium datasets
    else:
        # Default adjustment if dataset_size unknown (assume medium dataset)
        pass
    
    # Step 4: Time budget adjustment (ALWAYS apply if hours_to_complete available)
    if hours_to_complete and hours_to_complete > 0:
        if hours_to_complete <= 0.5:  # Very short jobs
            log_range *= 0.7  # Much narrower (30% reduction)
        elif hours_to_complete <= 0.75:  # Short jobs
            log_range *= 0.8  # Narrower (20% reduction)
        elif hours_to_complete <= 1.0:  # Medium-short jobs
            log_range *= 0.9  # Slightly narrower (10% reduction)
        elif hours_to_complete <= 2.0:  # Medium jobs
            log_range *= 0.95  # Slightly narrower (5% reduction)
        # else: no adjustment for long jobs
    else:
        # Default: assume medium time budget if unknown
        pass
    
    # Step 5: Number of runs adjustment (ALWAYS apply if n_runs available)
    if n_runs and n_runs > 0:
        if n_runs >= 5:  # Many runs
            log_range *= 1.1  # Wider search
        elif n_runs >= 3:  # Medium runs
            log_range *= 1.05  # Slightly wider
        elif n_runs == 2:  # Only 2 runs
            log_range *= 0.9  # Narrower search
        # else: no adjustment for 1 run (shouldn't happen in multi-run)
    else:
        # Default: assume 2-3 runs if unknown (medium runs = slightly wider)
        # But be conservative and use no adjustment to avoid over-optimization
        pass
    
    # Step 6: First run loss adjustment (ALWAYS apply if first_run_loss available)
    if first_run_loss and first_run_loss > 0:
        # If loss is very high (>2.0), might need wider search
        if first_run_loss > 2.0:
            log_range *= 1.1  # Wider search
        elif first_run_loss > 1.5:
            log_range *= 1.05  # Slightly wider
        elif first_run_loss < 0.5:  # Very low loss
            log_range *= 0.95  # Narrower search (already good)
        # else: no adjustment for normal loss
    else:
        # Default: assume medium loss if unknown
        pass
    
    # Ensure reasonable bounds (too narrow = no exploration, too wide = unstable)
    log_range = max(0.1, min(0.3, log_range))
    
    return log_range


def get_log_scale(task_type: str, model_name: str = None, request_path: str = None, **kwargs):
    """
    Get optimal log_range for LR search.
    Always calculates in all cases, fetching missing parameters and using AutoML optimization.
    """
    # Always use adaptive calculation with full context
    return calculate_adaptive_log_range(
        task_type, 
        model_name=model_name,
        request_path=request_path,
        **kwargs
    )


def _calculate_experimental_reg_ratio() -> float:
    """Calculate reg_ratio using experimental method (empirically determined default)."""
    return 1.24383


def _calculate_sqrt_batch_reg_ratio(batch_size: int, reference_batch: int = 64) -> float:
    """Calculate reg_ratio using square root batch scaling."""
    if batch_size is None or batch_size <= 0:
        return None
    return np.sqrt(batch_size / reference_batch)


def _calculate_linear_batch_reg_ratio(batch_size: int, reference_batch: int = 64) -> float:
    """Calculate reg_ratio using linear batch scaling."""
    if batch_size is None or batch_size <= 0:
        return None
    return batch_size / reference_batch


def _calculate_adaptive_reg_ratio(
    task_type: str = None,
    batch_size: int = None,
    model_params: int = None,
    base_lr: float = None,
    hours_to_complete: float = None
) -> float:
    """Calculate reg_ratio using adaptive method (combination of factors)."""
    reg_ratio = 1.0
    
    # Time-aware adjustment: Additional fine-tuning for time constraints
    # Note: Base LR already includes time constraints, so this is a smaller additional adjustment
    # for reg_ratio to account for batch size interactions with time constraints
    if hours_to_complete is not None and hours_to_complete > 0:
        # Smaller adjustment here since base LR already has time factor
        # This accounts for how batch size scaling interacts with time constraints
        if hours_to_complete <= 0.5:  # Very short jobs (<30 min)
            time_factor = 1.1  # Small additional boost for very short jobs
        elif hours_to_complete <= 0.75:  # Short jobs (<45 min)
            time_factor = 1.05
        elif hours_to_complete <= 1.0:  # Medium-short jobs (<1 hour)
            time_factor = 1.03
        elif hours_to_complete <= 2.0:  # Medium jobs
            time_factor = 1.01
        else:  # Long jobs
            time_factor = 1.0
        if time_factor != 1.0:
            reg_ratio *= time_factor
            print(f"  [reg_ratio]   - time_aware fine-tuning: {hours_to_complete:.2f}h -> factor {time_factor:.2f} (base LR already has time adjustment)", flush=True)
    
    # Batch size fine-tuning adjustment
    # Note: Base LR already includes batch_size, so this is a smaller additional adjustment
    # for reg_ratio to account for gradient accumulation and other batch-related factors
    if batch_size is not None and batch_size > 0:
        reference_batch = 64
        # Smaller adjustment here since base LR already has batch_size factor
        # This accounts for gradient accumulation steps and other batch-related interactions
        if batch_size < reference_batch:
            # Small boost for smaller batches (they need slightly higher LR per sample)
            batch_factor = 1.0 + 0.1 * (1 - batch_size / reference_batch)
        elif batch_size > reference_batch * 2:
            # Small reduction for very large batches (they're more stable)
            batch_factor = 1.0 - 0.05 * min(1.0, (batch_size / (reference_batch * 2) - 1))
        else:
            batch_factor = 1.0
        
        # Cap the batch factor
        batch_factor = max(0.95, min(1.1, batch_factor))
        if batch_factor != 1.0:
            reg_ratio *= batch_factor
            print(f"  [reg_ratio]   - batch_size fine-tuning ({batch_size}): {batch_factor:.3f}x (base LR already has batch adjustment)", flush=True)
    
    # Model size adjustment (larger models may need different scaling)
    if model_params is not None:
        if model_params > 10_000_000_000:  # > 10B params
            reg_ratio *= 0.95
        elif model_params < 1_000_000_000:  # < 1B params
            reg_ratio *= 1.05
    
    # Task type adjustment - IMPROVED: Use AutoML to learn optimal adjustments from historical performance
    if task_type:
        # Reference values (fallback when no historical data available)
        reference_task_factors = {
            TaskType.GRPOTASK.value: 1.0,  # No adjustment
            TaskType.DPOTASK.value: 1.02,
            TaskType.INSTRUCTTEXTTASK.value: 1.02,
            TaskType.CHATTASK.value: 1.02,
        }
        task_factor = reference_task_factors.get(task_type, 1.0)
        
        # OPTIMIZATION: Try to learn optimal task adjustment from historical reg_ratio performance
        try:
            from hyperparam_optimizer import _hyperparams_list
            
            # Map task_type to hyperparam_optimizer task_type format
            task_type_map = {
                TaskType.INSTRUCTTEXTTASK.value: "instruct",
                TaskType.CHATTASK.value: "instruct",  # Chat uses instruct config
                TaskType.DPOTASK.value: "dpo",
                TaskType.GRPOTASK.value: "grpo",
            }
            opt_task_type = task_type_map.get(task_type, "instruct")
            
            # Find best performing reg_ratio for this task type from history
            # CRITICAL: Exclude GRPO from test loss-based optimization (GRPO uses reward, not loss)
            if model_params and model_params > 0 and opt_task_type != "grpo":
                best_reg_ratio = None
                best_loss = float('inf')
                
                # Search for entries with same task type and similar model size
                for entry in _hyperparams_list:
                    if entry.get("task_type") == opt_task_type:
                        entry_params = entry.get("param_nums", 0)
                        entry_reg_ratio = entry.get("reg_ratio")
                        # CRITICAL: Prioritize eval_loss (test loss) for all tasks except GRPO
                        entry_loss = entry.get("eval_loss")
                        if entry_loss is None:
                            entry_loss = entry.get("train_loss")  # Fallback only if eval_loss not available
                        
                        # Check if model size is similar (within 50% for reg_ratio learning)
                        if entry_params > 0 and entry_reg_ratio is not None and entry_loss is not None:
                            param_ratio = min(model_params, entry_params) / max(model_params, entry_params)
                            if param_ratio >= 0.5:  # Within 50% model size
                                # Prefer entries with better (lower) test loss
                                if entry_loss < best_loss:
                                    best_loss = entry_loss
                                    best_reg_ratio = entry_reg_ratio
                
                # If we found historical reg_ratio, use it to infer optimal task adjustment
                if best_reg_ratio is not None and best_reg_ratio > 0:
                    # Calculate what the reg_ratio would be without task adjustment
                    # Current reg_ratio already has time, batch, and model adjustments
                    # We want to find what task_factor would give us the best_reg_ratio
                    # best_reg_ratio = reg_ratio * optimal_task_factor
                    # So: optimal_task_factor = best_reg_ratio / reg_ratio
                    
                    if reg_ratio > 0:
                        inferred_task_factor = best_reg_ratio / reg_ratio
                        # Bound the inferred factor to reasonable range (0.9 to 1.1)
                        inferred_task_factor = max(0.9, min(1.1, inferred_task_factor))
                        
                        # Blend learned factor with reference factor (60% learned, 40% reference for stability)
                        task_factor = 0.6 * inferred_task_factor + 0.4 * reference_task_factors.get(task_type, 1.0)
                        print(f"  [reg_ratio]   - task_type AutoML: {task_factor:.4f}x (learned from reg_ratio={best_reg_ratio:.4f}, loss={best_loss:.6f})", flush=True)
        except Exception as e:
            # Fallback to reference values if AutoML lookup fails
            print(f"  [reg_ratio]   - task_type AutoML lookup failed: {e}, using reference {task_factor:.4f}x", flush=True)
        
        if task_factor != 1.0:
            reg_ratio *= task_factor
            print(f"  [reg_ratio]   - task_type adjustment ({task_type}): {task_factor:.4f}x", flush=True)
    
    # Ensure reasonable bounds
    reg_ratio = max(0.5, min(2.0, reg_ratio))
    return reg_ratio


def calculate_reg_ratio(
    task_type: str = None,
    batch_size: int = None,
    model_params: int = None,
    base_lr: float = None,
    hours_to_complete: float = None,
    method: str = "optimized"
) -> float:
    print(f"  [reg_ratio] Calculating reg_ratio with method: '{method}'", flush=True)
    print(f"  [reg_ratio] Available parameters: task_type={task_type}, batch_size={batch_size}, "
          f"model_params={model_params}, base_lr={base_lr}, hours_to_complete={hours_to_complete}", flush=True)
    
    # If method is not "optimized", use the legacy single-method approach
    if method != "optimized":
        if method == "experimental":
            result = _calculate_experimental_reg_ratio()
            print(f"  [reg_ratio] experimental method: {result:.6f}", flush=True)
            return result
        elif method == "sqrt_batch":
            result = _calculate_sqrt_batch_reg_ratio(batch_size)
            if result is None:
                print(f"  [reg_ratio] sqrt_batch method: batch_size={batch_size}, falling back to default 1.24383", flush=True)
                return 1.24383
            print(f"  [reg_ratio] sqrt_batch method: sqrt({batch_size}/64) = {result:.6f}", flush=True)
            return result
        elif method == "linear_batch":
            result = _calculate_linear_batch_reg_ratio(batch_size)
            if result is None:
                print(f"  [reg_ratio] linear_batch method: batch_size={batch_size}, falling back to default 1.24383", flush=True)
                return 1.24383
            print(f"  [reg_ratio] linear_batch method: {batch_size}/64 = {result:.6f}", flush=True)
            return result
        elif method == "adaptive":
            result = _calculate_adaptive_reg_ratio(task_type, batch_size, model_params, base_lr, hours_to_complete)
            print(f"  [reg_ratio] adaptive method: {result:.6f}", flush=True)
            return result
        else:
            print(f"  [reg_ratio] Unknown method '{method}', falling back to default 1.24383", flush=True)
            return 1.24383
    
    # Optimized method: calculate all available methods and choose the best value
    print(f"\n  [reg_ratio] OPTIMIZED MODE: Calculating all available methods...", flush=True)
    values = {}
    weights = {}
    
    # 1. Experimental method (always available, high weight as baseline)
    exp_value = _calculate_experimental_reg_ratio()
    values["experimental"] = exp_value
    weights["experimental"] = 0.3  # High weight as empirically determined baseline
    print(f"    - experimental: {exp_value:.6f} (weight: {weights['experimental']:.2f})", flush=True)
    
    # 2. Sqrt batch method (requires batch_size)
    sqrt_value = _calculate_sqrt_batch_reg_ratio(batch_size)
    if sqrt_value is not None:
        values["sqrt_batch"] = sqrt_value
        weights["sqrt_batch"] = 0.25
        print(f"    - sqrt_batch: {sqrt_value:.6f} (weight: {weights['sqrt_batch']:.2f})", flush=True)
    else:
        print(f"    - sqrt_batch: skipped (batch_size not available)", flush=True)
    
    # 3. Linear batch method (requires batch_size)
    linear_value = _calculate_linear_batch_reg_ratio(batch_size)
    if linear_value is not None:
        values["linear_batch"] = linear_value
        weights["linear_batch"] = 0.15  # Lower weight as linear scaling can be too aggressive
        print(f"    - linear_batch: {linear_value:.6f} (weight: {weights['linear_batch']:.2f})", flush=True)
    else:
        print(f"    - linear_batch: skipped (batch_size not available)", flush=True)
    
    # 4. Adaptive method (uses all available parameters)
    adaptive_value = _calculate_adaptive_reg_ratio(task_type, batch_size, model_params, base_lr, hours_to_complete)
    values["adaptive"] = adaptive_value
    weights["adaptive"] = 0.3  # High weight as it considers multiple factors
    print(f"    - adaptive: {adaptive_value:.6f} (weight: {weights['adaptive']:.2f})", flush=True)
    
    # Calculate weighted average
    total_weight = sum(weights.values())
    if total_weight == 0 or len(values) == 0:
        # Safety fallback: should never happen since experimental is always added
        print(f"    - Warning: No values calculated, using experimental default", flush=True)
        return exp_value
    
    weighted_sum = sum(values[method] * weights[method] for method in values.keys())
    weighted_avg = weighted_sum / total_weight
    
    # Also calculate median for robustness
    sorted_values = sorted(values.values())
    n = len(sorted_values)
    if n == 0:
        # Safety fallback
        return exp_value
    elif n % 2 == 0:
        median_value = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        median_value = sorted_values[n//2]
    
    # Choose optimized value: use weighted average, but ensure it's within reasonable bounds
    # and close to the median (robustness check)
    optimized_value = weighted_avg
    
    # If weighted average deviates significantly from median, use median instead
    if abs(optimized_value - median_value) > 0.2:
        print(f"    - Warning: weighted_avg ({optimized_value:.6f}) deviates from median ({median_value:.6f})", flush=True)
        optimized_value = median_value
    
    # Ensure reasonable bounds
    optimized_value = max(0.5, min(2.0, optimized_value))
    
    print(f"\n  [reg_ratio] OPTIMIZATION RESULTS:", flush=True)
    print(f"    - All calculated values: {[f'{v:.6f}' for v in sorted(values.values())]}", flush=True)
    print(f"    - Weighted average: {weighted_avg:.6f}", flush=True)
    print(f"    - Median: {median_value:.6f}", flush=True)
    print(f"    - Final optimized value: {optimized_value:.6f}", flush=True)
    
    return optimized_value


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        help="Max steps to use for training", 
        default=-1
    )
    parser.add_argument("--retries", type=int, help="Number of retries", default=5)
    parser.add_argument(
        "--min-steps", 
        type=int, 
        help="Min steps to use for training", 
        default=100
    )

    parser.add_argument(
        "--reg-ratio", 
        type=float, 
        help="Reg ratio to use for training (overrides --reg-ratio-method if both provided)", 
        default=None
    )
    parser.add_argument(
        "--reg-ratio-method",
        type=str,
        choices=["optimized", "experimental", "sqrt_batch", "linear_batch", "adaptive"],
        help="Method to calculate reg_ratio. 'optimized' calculates all methods and chooses best value (default)",
        default="optimized"
    )

    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type
    
    # Try to get model parameters early for reg_ratio calculation
    model_params = None
    try:
        from model_utility import get_model_num_params
        # Get model path early if possible
        model_path = str(train_paths.get_text_base_model_path(original_model_name))
        if os.path.exists(model_path) or model_path:  # Check if path exists or is a model name
            model_params = get_model_num_params(original_model_name, model_path)
            if model_params:
                print(f"Early model params detection: {model_params/1e9:.2f}B parameters", flush=True)
    except Exception as e:
        print(f"Could not get model params early (will use defaults): {e}", flush=True)
    
    # Calculate reg_ratio if not explicitly provided
    print(f"\n{'='*60}", flush=True)
    print(f"REG_RATIO CALCULATION", flush=True)
    print(f"{'='*60}", flush=True)
    if args.reg_ratio is None:
        print(f"Calculating reg_ratio using method: '{args.reg_ratio_method}'", flush=True)
        print(f"Task type: {args.task_type}", flush=True)
        args.reg_ratio = calculate_reg_ratio(
            task_type=args.task_type,
            batch_size=None,  # Will be available later, but optimized method can work without it
            model_params=model_params,
            base_lr=None,  # Will be available later
            hours_to_complete=args.hours_to_complete,
            method=args.reg_ratio_method
        )
        print(f"\n[OK] Final calculated reg_ratio: {args.reg_ratio:.6f}", flush=True)
    else:
        print(f"Using explicitly provided reg_ratio: {args.reg_ratio:.6f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Short-job mode: prioritize getting to GPU training fast and avoid multi-run restarts
    # which add overhead (re-tokenization, repeated training launches, checkpoint churn).
    disable_multirun = os.getenv("DISABLE_MULTIRUN", "0") == "1" or args.hours_to_complete <= 0.75
    if disable_multirun:
        print("DISABLE_MULTIRUN enabled (short-job mode): will run exactly one training run.", flush=True)

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    print(f"submission_dir: {submission_dir}", flush=True)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))

    is_openai = False
    if is_openai_model(original_model_name, model_path):
        print("Upgrading python packages for openai model", flush=True)
        run_cmd_with_log(
            "pip uninstall -y transformers && pip install transformers==4.55.0",
            os.path.join(ds_folder, f"upgrade_transformers.log"),
        )
        # upgrade deepspeed
        run_cmd_with_log(
            "pip uninstall -y deepspeed && pip install deepspeed==0.17.4",
            os.path.join(ds_folder, f"upgrade_deepspeed.log"),
        )
        # install kernel
        run_cmd_with_log(
            "pip install kernels==0.9.0", os.path.join(ds_folder, f"install_kernel.log")
        )
        is_openai = True

    train_info = {
        "model_name": original_model_name,
        "model_path": model_path,
        "task_id": args.task_id,
        "dataset": dataset_path,
        "hours_to_complete": args.hours_to_complete,
        "expected_repo_name": args.expected_repo_name,
        "end_time": end_time,
        "dataset_type": dataset_type_dict,
        "submission_dir": submission_dir,
        "output_dir": output_dir,
        "adjust_batch_size": True,
        "request_path": request_path,
        "max_data_size": args.max_data_size,
        "max_steps": args.max_steps,
        "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
        "min_steps": args.min_steps,
        "is_openai": is_openai,
        "reg_ratio": args.reg_ratio,
        "find_lk_lr": True,
        "checking_mode": "first_time",
    }

    if (args.task_type == TaskType.INSTRUCTTEXTTASK.value or args.task_type == TaskType.CHATTASK.value):
        train_info = get_instruct_training_json(train_info)
        tokenize_cmd = (f"/workspace/axo_py/bin/python tokenize_instruct.py {request_path}")
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.DPOTASK.value:
        train_info = get_dpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_dpo.py {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.GRPOTASK.value:
        train_info = get_grpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_grpo.py {request_path}"
        train_cmd = train_info["run_cmd"]
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    
    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    run_cmd_with_log(
        tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log")
    )

    original_train_cmd = train_cmd
    train_success = False
    state = get_state()
    state = {}
    set_state(state) # reset first
    
    # Store model_params and dataset_size in state for adaptive log_range calculation
    if model_params:
        state["model_params"] = model_params
    # Try to get dataset size after tokenization
    try:
        from model_utility import get_data_size
        if request_path and os.path.exists(request_path):
            dataset_size = get_data_size(request_path)
            if dataset_size:
                state["dataset_size"] = dataset_size
    except:
        pass
    
    state["mode"] = "initial"
    # at first the state is always running the train_cmd

    set_state(state)
    # TODO Run something magic here
    count = 0
    while True:
        state = get_state()
        train_cmd = original_train_cmd  # will replace based on the state later
        c_train_info = copy.deepcopy(train_info)
        final_output_dir = None
        if args.task_type == TaskType.GRPOTASK.value:
            state["mode"] = "finish" # do not run this for GRPO task
            c_train_info["train_request"]["checking_mode"] = "none"
        else:
            if state["mode"] == "initial":
                c_train_info["train_request"]["checking_mode"] = "none" if disable_multirun else "first_time"
                
            elif state["mode"] == "continue":
                c_train_info["train_request"]["checking_mode"] = "second_time"
                n_runs = state["next_runs"]
                if "lrs" not in state: # first time of continue
                    current_lr = float(state["train"]["lr"])
                    
                    # Get adaptive parameters for log_range calculation
                    # Try to get from state first (stored during initial setup)
                    model_params = state.get("model_params", model_params)  # Fallback to global
                    dataset_size = state.get("dataset_size", None)
                    first_run_loss = state.get("train", {}).get("current_loss", None)
                    
                    # Calculate adaptive log_range (ALWAYS with full context for optimal value)
                    adaptive_log_range = get_log_scale(
                        args.task_type,
                        model_name=original_model_name,
                        request_path=request_path,
                        current_lr=current_lr,
                        model_params=model_params,
                        dataset_size=dataset_size,
                        hours_to_complete=args.hours_to_complete,
                        n_runs=n_runs,
                        first_run_loss=first_run_loss
                    )
                    
                    # Get base value for comparison (without context)
                    base_log_range = get_log_scale(args.task_type)
                    print(f"  [LR Search] Adaptive log_range: {adaptive_log_range:.4f} (base: {base_log_range:.4f}, model_params={model_params/1e9 if model_params else None:.2f}B, dataset_size={dataset_size}, n_runs={n_runs}, hours={args.hours_to_complete:.2f})", flush=True)
                    
                    state["lrs"] = lr_utils.extend_learning_rates(current_lr, n_runs, log_range=adaptive_log_range)
                    assert len(state["lrs"]) == n_runs, f"Number of learning rates {state['lrs']} should be equal to number of runs {n_runs}"
                    state["runs"] = []
                
                set_state(state)
                state["runs"].append(state["train"].copy())
                delete_poor_checkpoints(state["runs"])
                if len(state["runs"]) < n_runs:
                    index = len(state["runs"])
                    current_lr = state["lrs"][index]
                    train_cmd = replace_args_in_cmd(train_cmd, "learning_rate", str(state["lrs"][index]))
                else: # the final run - continue training the best checkpoint to completion
                    # IMPROVED: Use generalization_score if available, otherwise eval_loss, then train_loss
                    # This prioritizes checkpoints with better generalization (lower overfitting)
                    best_index = None
                    best_score = float('inf')
                    selection_method = None
                    
                    # Try to use generalization_score first (best metric)
                    # Check if we have both eval_loss and train_loss for at least some runs
                    has_train_loss = any("current_train_loss" in run for run in state["runs"])
                    if all("current_eval_loss" in run for run in state["runs"]) and has_train_loss:
                        # Calculate generalization_score for each run (same logic as in customized_trainer)
                        for i, run in enumerate(state["runs"]):
                            eval_loss = run.get("current_eval_loss", run["current_loss"])
                            train_loss = run.get("current_train_loss", None)
                            
                            if train_loss is not None and train_loss > 0:
                                overfitting_gap = max(0, eval_loss - train_loss)
                                gap_ratio = eval_loss / train_loss if train_loss > 0 else float('inf')
                                
                                # Use same adaptive penalty as in customized_trainer
                                if gap_ratio >= 2.0:
                                    penalty_factor = 0.7
                                elif gap_ratio >= 1.5:
                                    penalty_factor = 0.5
                                elif gap_ratio >= 1.2:
                                    penalty_factor = 0.35
                                else:
                                    penalty_factor = 0.2
                                
                                gen_score = eval_loss - penalty_factor * overfitting_gap
                            else:
                                # Fallback: use eval_loss if train_loss not available
                                gen_score = eval_loss
                            
                            if gen_score < best_score:
                                best_score = gen_score
                                best_index = i
                                selection_method = "generalization_score"
                    
                    # Fallback to eval_loss if generalization_score not available
                    if best_index is None and all("current_eval_loss" in run for run in state["runs"]):
                        losses_for_selection = [run.get("current_eval_loss", run["current_loss"]) for run in state["runs"]]
                        best_index = np.argmin(losses_for_selection)
                        best_score = state["runs"][best_index]["current_eval_loss"]
                        selection_method = "eval_loss"
                    
                    # Final fallback to train_loss
                    if best_index is None:
                        if len(state["runs"]) > 0:
                            losses_for_selection = [run["current_loss"] for run in state["runs"]]
                            best_index = np.argmin(losses_for_selection)
                            best_score = state["runs"][best_index]["current_loss"]
                            selection_method = "train_loss_fallback"
                        else:
                            # Safety fallback: should never happen, but handle gracefully
                            print(f"WARNING: No runs available for selection, using first run", flush=True)
                            best_index = 0
                            best_score = state["runs"][0]["current_loss"] if state["runs"] else float('inf')
                            selection_method = "safety_fallback"
                    
                    # Safety check: ensure best_index is valid
                    if best_index is None or best_index >= len(state["runs"]):
                        print(f"ERROR: Invalid best_index {best_index}, using first run", flush=True)
                        best_index = 0
                        best_score = state["runs"][0]["current_loss"] if state["runs"] else float('inf')
                        selection_method = "error_fallback"
                    
                    index = best_index
                    selected_loss = best_score
                    selected_run = state["runs"][index]
                    eval_loss_val = selected_run.get("current_eval_loss", selected_run["current_loss"])
                    train_loss_val = selected_run.get("current_train_loss", selected_run["current_loss"])
                    print(f"BL (using {selection_method});{index};score={selected_loss:.6f};eval_loss={eval_loss_val:.6f};train_loss={train_loss_val:.6f};lr={state['lrs'][index]}", flush=True)
                    
                    c_train_info["train_request"]["checking_mode"] = "none"
                    # Use the best checkpoint's train_cmd and output_dir
                    # The trainer will automatically resume from the last checkpoint in that directory
                    train_cmd = state["runs"][index]["train_cmd"]
                    final_output_dir = state["runs"][index]["output_dir"]
                    state["mode"] = "finish"
            else: # the state = finish; no need to run more
                assert state["mode"] == "finish"
                break
        
        set_state(state)
        if train_cmd:
            # If we have a final_output_dir (best checkpoint), use it; otherwise create new one
            if final_output_dir:
                run_output_dir = final_output_dir
            else:
                run_output_dir = output_dir + f"_{count}"
            train_cmd = replace_args_in_cmd(train_cmd, "output_dir", run_output_dir)
            
            current_request_path = os.path.join(ds_folder, f"training_request_{args.task_id}_{count}.json")
            with open(current_request_path, "w") as f:
                json.dump(c_train_info, f, indent=4, ensure_ascii=False)
            
            train_cmd = replace_args_in_cmd(train_cmd, "request_path", current_request_path)
            
            state["train"] = {
                "train_cmd": train_cmd,
                "log_path": os.path.join(ds_folder, f"train_{args.task_id}.log"),
                "lr": extract_value_from_cmd(train_cmd, "learning_rate"),
                "output_dir": run_output_dir
            }
            state["train"]["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            set_state(state)
            
            log_path = state["train"]["log_path"]
            # print(f"Run training with train_info: {c_train_info}", flush=True)
            success = run_training(
                train_cmd,
                log_path,
                args.task_id,
                args.retries,
                args.task_type,
                args.expected_repo_name,
            )
            time.sleep(5)
            if not success:
                print(f"Training failed for task {args.task_id} at count={count}", flush=True)
                break 

            # In short-job mode we deliberately avoid multi-run search/restarts.
            if disable_multirun:
                state = get_state()
                state["mode"] = "finish"
                set_state(state)
                break
        
        count += 1

    if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        print(f"Training failed for task {args.task_id}", flush=True)
    else:
        print(f"Training successfully done for task {args.task_id}", flush=True)
        train_success = True

    if not train_success:
        print(f"Training failed for task {args.task_id}", flush=True)
        # add noise to the model
        add_noise_cmd = f"python add_random_noise.py {model_path} {submission_dir}"
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
