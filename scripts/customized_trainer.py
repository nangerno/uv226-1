from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
import time
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
from state_manager import get_state, set_state

MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py", flush=True)
    
class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none",
        early_stopping_patience: int = 200
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self.end_time = end_time
        self.early_stopping_patience = early_stopping_patience
        self.steps_without_improvement = 0
        self.last_improvement_step = 0
        self.best_eval_loss = None  # Best test (eval) loss - main target
        self.top_checkpoints = []  # Store top N checkpoints by eval_loss
        self.max_top_checkpoints = 3
        # For main run: estimate time per step so we can stop early if full epoch would exceed remaining time
        self._last_step_wall_time = None
        self._time_per_step_est = None
        
    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)
    
    def compute_composite_score(self, state: TrainerState, metrics):
        """
        Optional composite score for logging only. Best checkpoint is selected by eval_loss (test loss) only.
        """
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is None:
            return None
        return float(eval_loss)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Custom logic to decide whether to save or evaluate
        # print(f"************* on_step_end: {state.global_step}, check eval", flush=True)
        # TODO: implement the logic to save the model without evaluating if there is no check points --> avoid evaluating takes too much time
        # Check if the checking_step is reached
        # print(f"Checking the model at step: {state.global_step}, checking_step: {self.checking_step}, checking_mode: {self.checking_mode}", flush=True)
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            # print(f"Checking the model at step: {state.global_step}", flush=True)
            # check the time so far to estimate the training time in total 
            my_state = get_state()
            if "train" not in my_state:
                my_state["train"] = {}
            train_state = my_state["train"]
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            start_time_str = train_state.get("start_time") or train_state.get("start_train_time") or now_str
            start_train_time_str = train_state.get("start_train_time") or train_state.get("start_time") or now_str
            start_time_obj = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(start_train_time_str, "%Y-%m-%d %H:%M:%S")
            
            log_content = f"Checking the model at step: {state.global_step}"
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\nPreparation time: {preparation_time}"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\nTime so far: {time_so_far}"
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f"\nTime for one step: {time_for_one_step}"
            # Now estimate the total training time for this training
            log_content += f"\nTotal steps all epochs: {self.total_steps_all_epochs}"
            total_remaining_training_time = time_for_one_step * (self.total_steps_all_epochs - state.global_step)
            log_content += f"\nTotal remaining training time: {total_remaining_training_time}"
            # n * time_so_far + total_remaining_training_time = total_remaining_time
            total_remaining_time = get_remaining_seconds(self.end_time)
            log_content += f"\nTotal remaining time: {total_remaining_time:.1f} sec"
            log_content += f"\nRemaining time until end_time: {format_remaining(total_remaining_time)}"
            
            # Compute from remaining time: reserve fraction for main, derive max LR runs ceiling
            buffer_seconds = 12 * 60  # eval/save/overhead
            max_var_time_sofar = 3 * 60
            time_per_run = max(time_so_far + max_var_time_sofar, 60.0)
            # Reserve for main: fraction of total_remaining_time (e.g. 25%), or one full run if longer
            main_fraction = float(os.environ.get("MIN_MAIN_TRAINING_FRACTION", "0.25"))
            main_fraction = max(0.1, min(0.5, main_fraction))
            min_main_seconds = total_remaining_time * main_fraction
            time_reserved_for_main = max(total_remaining_training_time, min_main_seconds)
            time_available_for_short_runs = total_remaining_time - time_so_far - buffer_seconds - time_reserved_for_main
            n = time_available_for_short_runs / time_per_run
            n = max(0, int(n))
            # Max LR runs ceiling: from budget (how many short runs fit in remaining time), with sane bounds
            max_lr_runs_from_budget = max(2, int(total_remaining_time / time_per_run))
            max_lr_runs_ceiling = min(200, max_lr_runs_from_budget)
            if os.environ.get("MAX_LR_RUNS", "").strip():
                max_lr_runs_ceiling = int(os.environ.get("MAX_LR_RUNS", str(max_lr_runs_ceiling)))
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "end_time": self.end_time,
            }
            # Always continue to LR search then main; when n==0 only current run, then main with remaining time
            reserve_min = int(time_reserved_for_main / 60)
            log_content += f"\nEstimated short runs from remaining time: {n} (reserve {reserve_min}min for main, {main_fraction*100:.0f}% of remaining)"
            control.should_training_stop = True
            control.should_save = False
            args.save_strategy = "no"
            if "current_loss" not in my_state["train"]:
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
            my_state["mode"] = "continue"
            # next_runs from remaining time (reserving fraction for main), capped by computed ceiling; at least 1
            next_runs = min(n + 1, max_lr_runs_ceiling)
            next_runs = max(1, next_runs)
            if next_runs < 2:
                log_content += f"\nFinal number: {next_runs} (no time for extra LR runs; main training will use remaining time)"
            else:
                log_content += f"\nFinal number: {next_runs} (from remaining time, reserve {reserve_min}min for main, cap={max_lr_runs_ceiling})"
            my_state["next_runs"] = next_runs
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time": # at second time, save eval (test) loss for this run; use it for best-run selection
            log_content = f"Checking the model at step: {state.global_step} where check_mode=second_time"
            my_state = get_state()
            # Use eval (test) loss for run selection; already updated in on_evaluate
            current_loss = my_state["train"].get("current_loss", state.log_history[-1]["loss"])
            my_state["train"]["current_loss"] = current_loss

            # Always stop at checking_step so a separate main run (with high max_steps / until end_time) can run
            control.should_training_stop = True
            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
            if current_loss <= current_min_loss:
                print(f"Current run eval loss {current_loss:.6f} <= best so far {current_min_loss:.6f}; stopping at checking_step (main run will use best LR until end_time).", flush=True)
            else:
                print(f"Current run eval loss {current_loss:.6f} > best {current_min_loss:.6f}; stopping this run.", flush=True)
            control.should_save = False
            args.save_strategy = "no"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                # print(log_content, flush=True)
        
        # Main run only: update time-per-step estimate so we can stop early if full epoch would exceed remaining time
        if self.checking_mode == "none" and state.global_step > 0:
            now = time.time()
            if self._last_step_wall_time is not None:
                elapsed = now - self._last_step_wall_time
                if self._time_per_step_est is None:
                    self._time_per_step_est = elapsed
                else:
                    self._time_per_step_est = 0.95 * self._time_per_step_est + 0.05 * elapsed
            self._last_step_wall_time = now

        when_to_eval = self.function_when_to_evaluate(state.global_step)
        # Main run: if time to complete full epoch would exceed remaining time, stop early (before 3 min remaining)
        if (
            self.checking_mode == "none"
            and self.total_steps_all_epochs > 0
            and self._time_per_step_est
            and state.global_step > 0
        ):
            steps_left = self.total_steps_all_epochs - state.global_step
            if steps_left > 0:
                time_needed = steps_left * self._time_per_step_est
                remaining = get_remaining_seconds(self.end_time)
                if remaining > 0 and time_needed > remaining:
                    when_to_eval = {"eval": True, "reason": "end_time"}
                    print(
                        f"Full epoch would exceed remaining time (need ~{time_needed/60:.1f} min, have {format_remaining(remaining)}); stopping early.",
                        flush=True,
                    )
        if when_to_eval["eval"]:
            remaining = get_remaining_seconds(self.end_time)
            print(f"Remaining time until end_time: {format_remaining(remaining)} | Evaluating at step {state.global_step}, reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                if not self.has_checkpoint:
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True
                # Stop when remaining time almost reached; submit best checkpoint even if real epoch not completed
                control.should_training_stop = True
                print(f"Remaining time almost reached: stopping and submitting best checkpoint (step {state.global_step}, epoch may be incomplete).", flush=True)
        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        # Test (eval) loss is the main target for best checkpoint and run selection
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return
        if eval_loss is None:
            return
        eval_loss = float(eval_loss)
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)

        # Store eval (test) loss in state so LR-run selection uses test loss, not train loss
        try:
            my_state = get_state()
            if "train" in my_state:
                my_state["train"]["current_loss"] = eval_loss
                set_state(my_state)
        except Exception:
            pass

        composite_score = self.compute_composite_score(state, metrics)
        if composite_score is None:
            composite_score = eval_loss

        # Initialize last_improvement_step on first evaluation if not set
        if self.last_improvement_step == 0:
            self.last_improvement_step = state.global_step

        # Best checkpoint = lowest eval (test) loss only
        if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
            improvement = self.best_eval_loss - eval_loss if self.best_eval_loss is not None else eval_loss
            print(f"*** NEW BEST CHECKPOINT at step {state.global_step} ***", flush=True)
            print(f"  Eval (test) Loss: {eval_loss:.6f}", flush=True)
            if self.best_eval_loss is not None:
                print(f"  Improvement: {improvement:.6f} ({improvement / abs(self.best_eval_loss) * 100:.2f}%)", flush=True)

            self.best_eval_loss = eval_loss
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "composite_score": composite_score,
                "step": state.global_step
            }
            self.update_best_checkpoint = True
            self.steps_without_improvement = 0
            self.last_improvement_step = state.global_step

            self.top_checkpoints.append({
                "step": state.global_step,
                "eval_loss": eval_loss,
                "composite_score": composite_score
            })
            self.top_checkpoints.sort(key=lambda x: x["eval_loss"])
            if len(self.top_checkpoints) > self.max_top_checkpoints:
                self.top_checkpoints = self.top_checkpoints[:self.max_top_checkpoints]
        else:
            self.steps_without_improvement = state.global_step - self.last_improvement_step
            if self.best_checkpoint_info is not None:
                print(f" At step: {state.global_step} - Eval (test) Loss: {eval_loss:.6f}", flush=True)
                print(f"  Best so far: Step {self.best_checkpoint_info['step']}, Eval Loss: {self.best_checkpoint_info['loss']:.6f}", flush=True)
                print(f"  Steps without improvement: {self.steps_without_improvement} (patience: {self.early_stopping_patience})", flush=True)
        
        # Early stopping based on validation loss plateau
        if self.early_stopping_patience > 0 and self.steps_without_improvement >= self.early_stopping_patience and self.last_improvement_step > 0:
            print(f"*** EARLY STOPPING triggered at step {state.global_step} ***", flush=True)
            print(f"  No improvement for {self.steps_without_improvement} steps (patience: {self.early_stopping_patience})", flush=True)
            control.should_training_stop = True
            

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            control.should_training_stop = True
        
        self.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK): # if not main process, skip this
            return 
            
        if self.save_only: # if only save, do not evaluate 
            print(f"Only save the model at step: {state.global_step}, no evaluation", flush=True)
            current_step = state.global_step
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
                
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{current_step}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{current_step},no_eval")
            
            # release the flag
            self.save_only = False
            return 
            
        # Custom logic after model is saved
        # You can trigger external services, logs, or backups here
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"Copy the best checkpoint to the submission directory at step: {state.global_step}", flush=True)
            # Remove existing directory if it exists
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
            best_eval_loss = self.best_checkpoint_info["loss"]
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{self.best_checkpoint_info['step']},{best_eval_loss}")
            # Tournament ranking: lower test_eval_loss = better = top (loss.txt: step,test_eval_loss)
            with open(os.path.join(self.submission_dir, "metric.txt"), "w") as f:
                f.write("test_eval_loss\n# Ranking: lower value = better (top)")
            if self.top_checkpoints:
                print(f"*** Top {len(self.top_checkpoints)} checkpoints by eval (test) loss:", flush=True)
                for i, ckpt in enumerate(self.top_checkpoints, 1):
                    print(f"  {i}. Step {ckpt['step']}: Eval Loss={ckpt['eval_loss']:.6f}", flush=True)


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        # GRPO reports eval_reward (higher = better). Use as "loss" = -reward for best-checkpoint selection.
        eval_reward = None
        if metrics is not None:
            eval_reward = metrics.get("eval_reward")
        if eval_reward is None and state.log_history:
            eval_reward = state.log_history[-1].get("eval_reward")
        if eval_reward is not None:
            return -float(eval_reward)  # lower "loss" = higher reward = better
        return None
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def get_remaining_seconds(end_time_str: str) -> float:
    """Return seconds until end_time (negative if past). If empty/invalid, return a large value so end_time never triggers."""
    _NO_END_TIME = 1e9
    if not end_time_str or not str(end_time_str).strip():
        return _NO_END_TIME
    try:
        et = datetime.datetime.strptime(end_time_str.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        now = datetime.datetime.now(timezone.utc)
        return (et - now).total_seconds()
    except (ValueError, TypeError):
        return _NO_END_TIME


def format_remaining(remaining_sec: float) -> str:
    """Human-readable remaining time for logs."""
    if remaining_sec < 0:
        return "0 min (past end_time)"
    return f"{remaining_sec / 60:.1f} min ({int(remaining_sec)} sec)"


def get_early_stopping_patience(
    total_steps_per_epoch: int,
    total_steps_all_epochs: int,
    explicit: Optional[int] = None,
    min_patience: int = 5,
    max_patience: int = 500,
    fraction_of_run: float = 0.3,
) -> int:
    """
    Adaptive early-stopping patience for unknown dataset/model sizes in tournaments.
    Uses explicit value if provided; else: at least one epoch, at most max_patience,
    otherwise fraction_of_run of total steps (so short runs don't stop too soon, long runs get a reasonable window).
    """
    if explicit is not None and explicit > 0:
        return int(explicit)
    if total_steps_all_epochs <= 0:
        return max_patience
    patience = max(
        total_steps_per_epoch,  # at least one full epoch without improvement
        min(max_patience, int(fraction_of_run * total_steps_all_epochs)),
    )
    return max(min_patience, patience)


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool:
    remaining = get_remaining_seconds(end_time)
    result = remaining < minutes * 60
    if result:
        print(f"*** Remaining time until end_time: {format_remaining(remaining)} - threshold {minutes} min", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and self.end_time and not self.run_eval:
            remaining = get_remaining_seconds(self.end_time)
            if remaining < self.save_before_remaining_time * 60:
                print(f"***ALERT: Remaining time until end_time: {format_remaining(remaining)} - eval & save then stop", flush=True)
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"Error setting generation config for model {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"Error resizing token embeddings for model {model_name}")
        pass


def init_wandb(train_request: Dict):
    # set wandb_mode=offline; do not upload the data to wandb export WANDB_MODE=offline
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True