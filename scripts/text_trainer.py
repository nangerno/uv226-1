#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
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


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"


def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == "gptossforcausallm":
        return True
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
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"************* Clear GPU cache before starting training to avoid memory fragmentation *************", flush=True)
            
    except:
        print(f"************* GPU is not allowed *************", flush=True)    
        pass
    
    for i in range(retries):
        print(f"************* Training attempt {i+1}/{retries} for task {task_id}*************", flush=True)
        if i > 0:  # there was something wrong so we will reduce the batch_size
            # first check if the training is OOM
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        new_batch_size = current_batch_size // 2
                        print(
                            f"Reducing batch size from {current_batch_size} to {new_batch_size}",
                            flush=True,
                        )
                        train_cmd = replace_args_in_cmd(
                            train_cmd,
                            "per_device_train_batch_size",
                            str(new_batch_size),
                        )
                        # print(f"New train command: {train_cmd}", flush=True)
                    else:
                        print(f"batch size is 1, cannot reduce further", flush=True)
                        if task_type == TaskType.GRPOTASK.value:
                            # disable vllm
                            train_cmd = replace_args_in_cmd(
                                train_cmd, "use_vllm", "False"
                            )
                            # print(f"disable VLLM {train_cmd}", flush=True)
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print(f"VLLM OOM error, disable VLLM", flush=True)
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


def get_log_scale(task_type: str):
    log_scale_map = {
        TaskType.INSTRUCTTEXTTASK.value: 0.18,
        TaskType.DPOTASK.value: 0.18,
        TaskType.GRPOTASK.value: 0.2,
        TaskType.CHATTASK.value: 0.18,
    }
    return log_scale_map[task_type]


def calculate_reg_ratio(
    task_type: str = None,
    batch_size: int = None,
    model_params: int = None,
    base_lr: float = None,
    method: str = "experimental"
) -> float:
    """
    Calculate reg_ratio (learning rate adjustment factor) based on training parameters.
    
    Args:
        task_type: Type of task (InstructTextTask, DpoTask, GrpoTask, ChatTask)
        batch_size: Total batch size (per_device_batch_size * num_gpus * gradient_accumulation)
        model_params: Number of model parameters
        base_lr: Base learning rate before reg_ratio adjustment
        method: Calculation method - "experimental" (default 1.24383), "sqrt_batch" (sqrt scaling),
                "linear_batch" (linear scaling), or "adaptive" (combination)
    
    Returns:
        Calculated reg_ratio value
    """
    if method == "experimental":
        # Return the empirically determined default value
        print(f"  [reg_ratio] Using experimental method: returning default value 1.24383", flush=True)
        return 1.24383
    
    elif method == "sqrt_batch":
        # Square root scaling: reg_ratio = sqrt(batch_size / reference_batch_size)
        # Reference batch size of 64 is common
        if batch_size is None or batch_size <= 0:
            print(f"  [reg_ratio] sqrt_batch method: batch_size={batch_size}, falling back to default 1.24383", flush=True)
            return 1.24383
        reference_batch = 64
        calculated = np.sqrt(batch_size / reference_batch)
        print(f"  [reg_ratio] sqrt_batch method: sqrt({batch_size}/{reference_batch}) = {calculated:.6f}", flush=True)
        return calculated
    
    elif method == "linear_batch":
        # Linear scaling: reg_ratio = batch_size / reference_batch_size
        if batch_size is None or batch_size <= 0:
            print(f"  [reg_ratio] linear_batch method: batch_size={batch_size}, falling back to default 1.24383", flush=True)
            return 1.24383
        reference_batch = 64
        calculated = batch_size / reference_batch
        print(f"  [reg_ratio] linear_batch method: {batch_size}/{reference_batch} = {calculated:.6f}", flush=True)
        return calculated
    
    elif method == "adaptive":
        # Adaptive calculation based on multiple factors
        reg_ratio = 1.0
        print(f"  [reg_ratio] adaptive method: starting with base=1.0", flush=True)
        
        # Batch size adjustment (sqrt scaling)
        if batch_size is not None and batch_size > 0:
            reference_batch = 64
            batch_factor = np.sqrt(batch_size / reference_batch)
            print(f"  [reg_ratio]   - batch_size adjustment: sqrt({batch_size}/{reference_batch}) = {batch_factor:.6f}", flush=True)
            reg_ratio *= batch_factor
            print(f"  [reg_ratio]   - after batch adjustment: {reg_ratio:.6f}", flush=True)
        
        # Model size adjustment (larger models may need different scaling)
        if model_params is not None:
            if model_params > 10_000_000_000:  # > 10B params
                adjustment = 0.95
                print(f"  [reg_ratio]   - model_size adjustment: {model_params/1e9:.1f}B params -> factor {adjustment:.2f}", flush=True)
                reg_ratio *= adjustment
            elif model_params < 1_000_000_000:  # < 1B params
                adjustment = 1.05
                print(f"  [reg_ratio]   - model_size adjustment: {model_params/1e6:.1f}M params -> factor {adjustment:.2f}", flush=True)
                reg_ratio *= adjustment
            else:
                print(f"  [reg_ratio]   - model_size adjustment: {model_params/1e9:.1f}B params -> no adjustment", flush=True)
            print(f"  [reg_ratio]   - after model adjustment: {reg_ratio:.6f}", flush=True)
        
        # Task type adjustment
        if task_type:
            task_adjustments = {
                TaskType.GRPOTASK.value: 1.0,  # No adjustment
                TaskType.DPOTASK.value: 1.02,
                TaskType.INSTRUCTTEXTTASK.value: 1.02,
                TaskType.CHATTASK.value: 1.02,
            }
            task_factor = task_adjustments.get(task_type, 1.0)
            print(f"  [reg_ratio]   - task_type adjustment: {task_type} -> factor {task_factor:.2f}", flush=True)
            reg_ratio *= task_factor
            print(f"  [reg_ratio]   - after task adjustment: {reg_ratio:.6f}", flush=True)
        
        # Ensure reasonable bounds
        original = reg_ratio
        reg_ratio = max(0.5, min(2.0, reg_ratio))
        if original != reg_ratio:
            print(f"  [reg_ratio]   - clamping: {original:.6f} -> {reg_ratio:.6f} (bounds: 0.5-2.0)", flush=True)
        
        print(f"  [reg_ratio] adaptive method: final result = {reg_ratio:.6f}", flush=True)
        return reg_ratio
    
    else:
        # Unknown method, return default
        return 1.24383


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
        choices=["experimental", "sqrt_batch", "linear_batch", "adaptive"],
        help="Method to calculate reg_ratio",
        default="experimental"
    )

    args = parser.parse_args()
    
    # Calculate reg_ratio if not explicitly provided
    print(f"\n{'='*60}", flush=True)
    print(f"REG_RATIO CALCULATION", flush=True)
    print(f"{'='*60}", flush=True)
    if args.reg_ratio is None:
        print(f"Calculating reg_ratio using method: '{args.reg_ratio_method}'", flush=True)
        print(f"Task type: {args.task_type}", flush=True)
        args.reg_ratio = calculate_reg_ratio(
            task_type=args.task_type,
            method=args.reg_ratio_method
        )
        print(f"\n[OK] Final calculated reg_ratio: {args.reg_ratio:.6f}", flush=True)
    else:
        print(f"Using explicitly provided reg_ratio: {args.reg_ratio:.6f}", flush=True)
    print(f"{'='*60}\n", flush=True)
    original_model_name = args.model
    original_task_type = args.task_type

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
    if is_openai_model(original_model_name):
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
                    state["lrs"] = lr_utils.extend_learning_rates(current_lr, n_runs, log_range=get_log_scale(args.task_type))
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
                    # CRITICAL: Use eval_loss for selection if available, otherwise fall back to current_loss
                    # This is the key improvement - eval_loss is what matters for generalization
                    if all("current_eval_loss" in run for run in state["runs"]):
                        # Use eval_loss for selection (much better than train_loss)
                        losses_for_selection = [run.get("current_eval_loss", run["current_loss"]) for run in state["runs"]]
                        index = np.argmin(losses_for_selection)
                        selected_loss = state["runs"][index]["current_eval_loss"]
                        print(f"BL (using eval_loss);{index};eval_loss={selected_loss:.6f};train_loss={state['runs'][index]['current_loss']:.6f};lr={state['lrs'][index]}", flush=True)
                    else:
                        # Fallback to current_loss if eval_loss not available (backward compatibility)
                        losses_for_selection = [run["current_loss"] for run in state["runs"]]
                        index = np.argmin(losses_for_selection)
                        selected_loss = state["runs"][index]["current_loss"]
                        print(f"BL (using train_loss fallback);{index};{selected_loss:.6f}; {state['lrs'][index]}", flush=True)
                    
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
