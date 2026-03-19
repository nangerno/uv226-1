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
import warnings
from datetime import datetime, timezone, timedelta

# Suppress pydantic "model_" protected namespace warnings from dependencies (e.g. datasets)
warnings.filterwarnings(
    "ignore",
    message=r".*conflict with protected namespace .*model_.*",
    category=UserWarning,
)

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
from customized_trainer import get_remaining_seconds, format_remaining


def _tf32_supported() -> bool:
    """True if CUDA is available and GPU is Ampere (8.x) or newer (tf32 supported)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        return cap is not None and len(cap) >= 1 and cap[0] >= 8
    except Exception:
        return False


def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None, cwd: str | None = None):
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
            cwd=cwd,
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
        return return_code


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    # Match --arg value optionally followed by space (so arg at end of cmd still matches)
    match = re.search(rf"(?P<p>--{re.escape(arg_name)}(\s+)([^\s]+))(?=\s|$)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
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

PROBE_MAX_STEPS = 10  # longer probe so memory pattern is closer to full training
# Safety factor: use this fraction of probed max batch for full run (avoids OOM from growth over time)
# Set env PROBE_SAFETY_FACTOR (e.g. 0.9) for less conservative batch size.
PROBE_SAFETY_FACTOR = float(os.environ.get("PROBE_SAFETY_FACTOR", "0.75"))
# When config batch fits, search upward up to this cap to find larger batch (default 2x config, max 128)
PROBE_BATCH_CAP_FACTOR = float(os.environ.get("PROBE_BATCH_CAP_FACTOR", "2.0"))
PROBE_BATCH_ABS_MAX = int(os.environ.get("PROBE_BATCH_ABS_MAX", "128"))


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


def _run_probe(
    train_cmd: str,
    batch_size: int,
    probe_output_dir: str,
    probe_request_path: str,
    probe_log_path: str,
    env_vars: dict,
    cwd: str | None,
) -> bool:
    """Run one probe at the given batch size. Returns True if no OOM."""
    cmd = replace_args_in_cmd(train_cmd, "per_device_train_batch_size", str(batch_size))
    if cmd is None:
        return False
    cmd = replace_args_in_cmd(cmd, "output_dir", probe_output_dir)
    if cmd is None:
        return False
    cmd = replace_args_in_cmd(cmd, "request_path", probe_request_path)
    if cmd is None:
        return False
    if os.path.exists(probe_log_path):
        with open(probe_log_path, "w") as f:
            f.write("STARTING PROBE")
    run_cmd_with_log(cmd, probe_log_path, env_vars=env_vars, cwd=cwd)
    return get_error_type(probe_log_path) != OOM_ERROR


def find_max_batch_size_through_probe(
    train_cmd: str,
    log_path: str,
    cwd: str | None,
    env_vars: dict,
) -> int:
    """
    Run short training probes to find the largest batch size that does not OOM.
    - If config batch OOMs: binary-search down in [1, config-1] for largest that fits.
    - If config batch fits: binary-search upward up to PROBE_BATCH_CAP (default 2x config, max 128)
      so we use GPU memory more fully. Env: PROBE_SAFETY_FACTOR, PROBE_BATCH_CAP_FACTOR, PROBE_BATCH_ABS_MAX.
    """
    output_dir = extract_value_from_cmd(train_cmd, "output_dir")
    request_path = extract_value_from_cmd(train_cmd, "request_path")
    initial_batch_size = extract_value_from_cmd(train_cmd, "per_device_train_batch_size")
    if not output_dir or not request_path or not initial_batch_size:
        return int(initial_batch_size) if initial_batch_size else 1
    initial_batch_size = int(initial_batch_size)
    if initial_batch_size <= 1:
        return 1

    request_dir = os.path.dirname(os.path.abspath(request_path))
    probe_request_path = os.path.join(request_dir, "probe_request.json")
    try:
        with open(request_path, "r") as f:
            request_data = json.load(f)
    except Exception as e:
        print(f"Probe: could not read request {request_path}: {e}", flush=True)
        return 1

    if "train_request" in request_data:
        request_data["train_request"] = dict(request_data["train_request"])
        request_data["train_request"]["max_steps"] = PROBE_MAX_STEPS
    try:
        with open(probe_request_path, "w") as f:
            json.dump(request_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Probe: could not write {probe_request_path}: {e}", flush=True)
        return 1

    probe_output_dir = output_dir.rstrip("/") + "_probe"
    probe_log_path = log_path + ".probe"

    # Try config size first.
    print(
        f"Probe: trying per_device_train_batch_size={initial_batch_size} (max_steps={PROBE_MAX_STEPS})",
        flush=True,
    )
    if not _run_probe(
        train_cmd, initial_batch_size,
        probe_output_dir, probe_request_path, probe_log_path,
        env_vars, cwd,
    ):
        low, high = 1, initial_batch_size - 1
        while low < high:
            mid = (low + high + 1) // 2
            print(
                f"Probe: trying per_device_train_batch_size={mid} (max_steps={PROBE_MAX_STEPS})",
                flush=True,
            )
            if _run_probe(
                train_cmd, mid,
                probe_output_dir, probe_request_path, probe_log_path,
                env_vars, cwd,
            ):
                low = mid
            else:
                high = mid - 1
        try:
            os.remove(probe_request_path)
        except OSError:
            pass
        print(f"Probe: selected per_device_train_batch_size={low}", flush=True)
        return low

    # Config size fits: search upward to find the largest batch that still fits (more optimized use of GPU).
    cap = min(
        int(initial_batch_size * PROBE_BATCH_CAP_FACTOR),
        PROBE_BATCH_ABS_MAX,
    )
    cap = max(cap, initial_batch_size)
    if cap > initial_batch_size:
        low, high = initial_batch_size, cap
        while low < high:
            mid = (low + high + 1) // 2
            print(
                f"Probe: trying per_device_train_batch_size={mid} (max_steps={PROBE_MAX_STEPS}, search up)",
                flush=True,
            )
            if _run_probe(
                train_cmd, mid,
                probe_output_dir, probe_request_path, probe_log_path,
                env_vars, cwd,
            ):
                low = mid
            else:
                high = mid - 1
        best_batch = low
    else:
        best_batch = initial_batch_size

    try:
        os.remove(probe_request_path)
    except OSError:
        pass
    print(
        f"Probe: selected per_device_train_batch_size={best_batch} (max that fits)",
        flush=True,
    )
    return best_batch


def run_probe_once_and_return_cmd(
    train_cmd: str,
    log_path: str,
    task_id: str,
    expected_repo_name: str,
    cwd: str | None = None,
    task_type: str | None = None,
) -> str:
    """Run batch-size probe once; return train_cmd with safe per_device_train_batch_size (and ga if needed)."""
    training_env_vars = {
        "WANDB_MODE": "offline",
        "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
        "WANDB_NAME": f"{task_id}_{expected_repo_name}",
    }
    initial_batch = extract_value_from_cmd(train_cmd, "per_device_train_batch_size")
    if not initial_batch or int(initial_batch) <= 1:
        return train_cmd
    best_batch = find_max_batch_size_through_probe(
        train_cmd, log_path, cwd, training_env_vars
    )
    # DPO without padding_free uses more memory per batch; use a more conservative safety factor (task-type only, no dataset/model constants)
    safety = PROBE_SAFETY_FACTOR
    if task_type == TaskType.DPOTASK.value:
        safety = float(os.environ.get("PROBE_SAFETY_FACTOR_DPO", "0.5"))
    safe_batch = max(1, int(best_batch * safety))
    if safe_batch < best_batch:
        print(
            f"Probe passed at {best_batch}; using safe batch {safe_batch} ({int(safety*100)}% margin for full run)",
            flush=True,
        )
    new_cmd = replace_args_in_cmd(
        train_cmd, "per_device_train_batch_size", str(safe_batch)
    )
    if new_cmd is not None:
        train_cmd = new_cmd
    if safe_batch <= 4:
        ga = extract_value_from_cmd(train_cmd, "gradient_accumulation_steps")
        current_ga = int(ga) if ga else 1
        target_effective = 8
        new_ga = max(current_ga, (target_effective + safe_batch - 1) // safe_batch)
        if new_ga != current_ga:
            ga_cmd = replace_args_in_cmd(train_cmd, "gradient_accumulation_steps", str(new_ga))
            if ga_cmd is not None:
                train_cmd = ga_cmd
                print(
                    f"Boosted gradient_accumulation_steps {current_ga} -> {new_ga} (effective batch >= {target_effective})",
                    flush=True,
                )
    print(
        f"Starting full training with per_device_train_batch_size={safe_batch}",
        flush=True,
    )
    return train_cmd


def run_training(
    train_cmd: str,
    log_path: str,
    task_id: str,
    retries: int,
    task_type: str,
    expected_repo_name: str,
    cwd: str | None = None,
):
    training_env_vars = {
        "WANDB_MODE": "offline",
        "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
        "WANDB_NAME": f"{task_id}_{expected_repo_name}",
    }

    for i in range(retries):
        print(
            f"************* Training attempt {i+1}/{retries} for task {task_id}*************",
            flush=True,
        )
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
                            new_cmd = replace_args_in_cmd(
                                train_cmd, "use_vllm", "False"
                            )
                            if new_cmd is not None:
                                train_cmd = new_cmd
                            # print(f"disable VLLM {train_cmd}", flush=True)
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print(f"VLLM OOM error, disable VLLM", flush=True)
                        new_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")
                        if new_cmd is not None:
                            train_cmd = new_cmd

        # empty the log file if it exists
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("STARTING TRAINING")

        if not train_cmd:
            print("train_cmd is empty or None, skipping run", flush=True)
            return False
        run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars, cwd=cwd)
        # check if training is successfully here so we can break the loop; if output_dir contains file: "successs.txt" return true
        output_dir = extract_value_from_cmd(train_cmd, "output_dir")
        if os.path.exists(os.path.join(output_dir, "success.txt")):
            return True
        time.sleep(5)
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
    lowest_loss = min([run["current_loss"] for run in train_runs])
    for run in train_runs:
        if run["current_loss"] > lowest_loss:
            if os.path.exists(run["output_dir"]):
                print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                shutil.rmtree(run["output_dir"])


def get_log_scale(task_type: str):
    # Increased search range for better hyperparameter exploration
    # This allows exploring a wider range of learning rates (10^0.3 = ~2x range)
    log_scale_map = {
        TaskType.INSTRUCTTEXTTASK.value: 0.3,  # Increased from 0.18 to 0.3 for wider LR search
        TaskType.DPOTASK.value: 0.3,  # Increased from 0.18 to 0.3
        TaskType.GRPOTASK.value: 0.35,  # Increased from 0.2 to 0.35
        TaskType.CHATTASK.value: 0.3,  # Increased from 0.18 to 0.3
    }
    return log_scale_map[task_type]


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
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument("--retries", type=int, help="Number of retries", default=5)
    parser.add_argument(
        "--min-steps", type=int, help="Min steps to use for training", default=100
    )

    parser.add_argument(
        "--reg-ratio", type=float, help="Reg ratio to use for training", default=1.0
    )

    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    repo_name = args.expected_repo_name or args.task_id
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, repo_name
    )
    print(f"submission_dir: {submission_dir}", flush=True)
    print(
        "Tournament ranking: lower test (eval) loss = better = top.",
        flush=True,
    )
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = os.path.join(project_root, "datasets")
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    # Use local dataset path; download from URL if args.dataset is a URL (e.g. when running outside Docker)
    if args.dataset.strip().lower().startswith(("http://", "https://")):
        dataset_path = os.path.join(ds_folder, f"{args.task_id}_train_data.json")
        if not os.path.isfile(dataset_path):
            import urllib.request
            print(f"Downloading dataset from {args.dataset[:80]}...", flush=True)
            urllib.request.urlretrieve(args.dataset, dataset_path)
            print(f"Saved to {dataset_path}", flush=True)
    else:
        dataset_path = train_paths.get_text_dataset_path(args.task_id)
    model_path = str(train_paths.get_text_base_model_path(original_model_name))
    # Use hub model id when cache path does not exist (e.g. local run without Docker)
    if not os.path.isdir(model_path):
        model_path = original_model_name

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
        "expected_repo_name": repo_name,
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

    if (
        args.task_type == TaskType.INSTRUCTTEXTTASK.value
        or args.task_type == TaskType.CHATTASK.value
    ):
        train_info = get_instruct_training_json(train_info)
        tokenize_cmd = (
            f"{sys.executable} {os.path.join(script_dir, 'tokenize_instruct.py')} {request_path}"
        )
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.DPOTASK.value:
        train_info = get_dpo_training_json(train_info)
        tokenize_cmd = f"{sys.executable} {os.path.join(script_dir, 'tokenize_dpo.py')} {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.GRPOTASK.value:
        train_info = get_grpo_training_json(train_info)
        tokenize_cmd = f"{sys.executable} {os.path.join(script_dir, 'tokenize_grpo.py')} {request_path}"
        train_cmd = train_info["run_cmd"]
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    
    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    tokenize_ret = run_cmd_with_log(
        tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log"), cwd=script_dir
    )
    if tokenize_ret != 0:
        raise RuntimeError(
            f"Tokenization failed with exit code {tokenize_ret}. "
            f"Check {os.path.join(ds_folder, f'tokenize_{args.task_id}.log')}"
        )

    original_train_cmd = train_cmd
    if not _tf32_supported():
        original_train_cmd = original_train_cmd.replace("--tf32 True", "--tf32 False")

    # Run batch-size probe once (not per attempt) so we don't repeat probe before every run
    request_path_0 = os.path.join(ds_folder, f"training_request_{args.task_id}_0.json")
    with open(request_path_0, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)
    first_train_cmd = replace_args_in_cmd(
        original_train_cmd, "output_dir", output_dir + "_0"
    )
    first_train_cmd = replace_args_in_cmd(
        first_train_cmd, "request_path", request_path_0
    )
    log_path_0 = os.path.join(ds_folder, f"train_{args.task_id}.log")
    original_train_cmd = run_probe_once_and_return_cmd(
        first_train_cmd, log_path_0, args.task_id, repo_name, cwd=script_dir, task_type=args.task_type
    )

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
        if args.task_type == TaskType.GRPOTASK.value:
            state["mode"] = "finish" # do not run this for GRPO task
            c_train_info["train_request"]["checking_mode"] = "none"
        else:
            if state["mode"] == "initial":
                c_train_info["train_request"]["checking_mode"] = "first_time"
                
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
                else:
                    # LR search done: run main training with best LR until given time, then submit
                    c_train_info["train_request"]["checking_mode"] = "none"
                    index = np.argmin([run["current_loss"] for run in state["runs"]])
                    best_lr = state["lrs"][index]
                    best_eval = state["runs"][index]["current_loss"]
                    end_time_str = train_info.get("end_time") or train_info.get("train_request", {}).get("end_time", "")
                    remaining = get_remaining_seconds(end_time_str) if end_time_str else 0.0
                    print(
                        f"Remaining time until end_time: {format_remaining(remaining)}",
                        flush=True,
                    )
                    print(
                        f"Best LR by eval (test) loss: index={index}, lr={best_lr}, eval_loss={best_eval:.6f}. Starting main training with this LR until end_time.",
                        flush=True,
                    )
                    train_cmd = state["runs"][index]["train_cmd"]
                    state["mode"] = "finish"  # after this main run we break
            else: # the state = finish; no need to run more
                assert state["mode"] == "finish"
                break
        
        set_state(state)
        if train_cmd:
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
            # Probe already ran once; "attempt 1/5" here is the first try of this phase (up to 5 retries on OOM/failure)
            print(
                f"Training phase count={count} (output_dir={run_output_dir}); retries on failure: up to 5",
                flush=True,
            )
            success = run_training(
                train_cmd,
                log_path,
                args.task_id,
                args.retries,
                args.task_type,
                repo_name,
                cwd=script_dir,
            )
            train_success = success
            time.sleep(5)
            if not success:
                print(f"Training failed for task {args.task_id} at count={count}", flush=True)
                break
        count += 1

    # Only report success if we didn't break due to run failure and submission has expected outputs
    if train_success and os.path.exists(submission_dir) and len(os.listdir(submission_dir)) >= 2:
        print(f"Training successfully done for task {args.task_id}", flush=True)
    elif not train_success or not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        if train_success:
            print(f"Training failed for task {args.task_id}", flush=True)
        train_success = False

    if not train_success:
        print(f"Training failed for task {args.task_id}", flush=True)
        # add noise to the model
        add_noise_cmd = f"{sys.executable} {os.path.join(script_dir, 'add_random_noise.py')} {model_path} {submission_dir}"
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log"), cwd=script_dir
        )

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
