from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
from typing import Callable, Optional, Dict
import shutil
import json
import math
from transformers.trainer_utils import is_main_process
import wandb
import torch
from state_manager import get_state, set_state
MAX_TRIES = 9


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
        task_type: str = None,
        update_lr_lookup: bool = True,
        metadata: Optional[Dict] = None
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
        self.task_type = task_type  # For LR lookup update
        self.update_lr_lookup = update_lr_lookup  # Flag to enable/disable automatic updates
        self.metadata = metadata or {}  # Additional metadata for LR lookup
        self._capture_eval_loss_at_checking = False
        self._checking_step_eval_loss = None
        # DECISIVE: Track top 3 checkpoints for interpolation (improved from 2)
        self.top_checkpoints = []  # List of (step, eval_loss, train_loss, generalization_score)
        self.max_top_checkpoints = 3  # Track top 3 for better ensemble
        # FRESH: Smart checkpoint pruning - track predicted quality
        self.checkpoint_predictions = {}  # step -> predicted_loss
        self.last_eval_loss = None
        self.eval_loss_trend = []  # Track trend for prediction
        # FRESH: Incremental evaluation - track subset evaluations
        self.subset_eval_results = {}  # step -> subset_eval_loss
        self.skip_full_eval_threshold = 1.1  # Skip full eval if subset loss > best * this
        
    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)

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
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(my_state["train"]["start_train_time"], "%Y-%m-%d %H:%M:%S")
            
            log_content = f"Checking the model at step: {state.global_step}"
            now = datetime.datetime.now()
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
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\nTotal remaining time: {total_remaining_time}"
            
            # n * time_so_far + (time_so_far + total_remaining_training_time) = total_remaining_time
            # time_so_far + total_remaining_training_time is the time it takes to finish the training (need to estimate the eval time and save time, assuming this is 15 minutes)
            # assuming time_so_far is + 5 minutes, just in case the checking step takes more time than expected
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far + total_remaining_training_time + 12 * 60)) / (time_so_far + max_var_time_sofar) # 300 = 5 minutes, assume that it extra time would be more or less 5 minutes
            n = int(n)
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
            if n > 0: # we should try more 
                log_content += f"\nEstimated number of steps to complete the training: {n}"
                # Don't save at first_time checking to save time and space (match top miner)
                control.should_save = False
                control.should_training_stop = True
                args.save_strategy = "no"
                # save the current loss of this step to the state;
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\nFinal number: {n + 1}"
                my_state["next_runs"] = n + 1 # including the current run
            else:
                print(f"Time is not enough so we will finish the training", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time": # at second time, we don't estimate the training time again, just save the current_loss
            log_content = f"Checking the model at step: {state.global_step} where check_mode=second_time"            
            my_state = get_state()
            current_loss = state.log_history[-1]["loss"]
            my_state["train"]["current_loss"] = current_loss
            
            # CRITICAL: Trigger evaluation to get eval_loss for better checkpoint selection
            # This is the key improvement - use eval_loss instead of train_loss for selection
            print(f"Triggering evaluation at checking_step {state.global_step} to get eval_loss for multi-run selection", flush=True)
            control.should_evaluate = True
            control.should_save = True  # Need to save to get eval metrics
            # Store flag to capture eval_loss after evaluation
            self._capture_eval_loss_at_checking = True
            self._checking_step_eval_loss = None
                
            # Stop training temporarily to wait for evaluation to complete
            # We'll decide in on_evaluate whether to continue or stop based on eval_loss
            control.should_training_stop = True
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)
        
            
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            # FRESH: Smart checkpoint pruning - predict quality before evaluating
            # If predicted to be poor, skip evaluation to save time
            if state.global_step in self.checkpoint_predictions:
                predicted_loss = self.checkpoint_predictions[state.global_step]
                best_loss = self.best_checkpoint_info["loss"] if self.best_checkpoint_info else float('inf')
                if predicted_loss > best_loss * 1.15:  # 15% worse than best
                    print(f"FRESH: Skipping evaluation at step {state.global_step} - predicted loss {predicted_loss:.6f} is poor (best: {best_loss:.6f})", flush=True)
                    control.should_evaluate = False
                    control.should_save = False  # Don't save poor checkpoints
                    return control
            
            # do not allow the pod to be stopped by any reason 
                # first check if there is at least one checkpoint or not 
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                if not self.has_checkpoint: # if there is no checkpoint, we just save the model, do not evaluate
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True
        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        # Use eval_loss (which is cross-entropy) for checkpoint selection
        # This directly optimizes for test_loss which is also cross-entropy
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return control 
        
        # FRESH: Incremental evaluation - check if this is a subset evaluation
        # If subset loss is poor, skip full evaluation to save time
        is_subset_eval = metrics.get("_is_subset_eval", False)
        if is_subset_eval:
            subset_loss = eval_loss
            self.subset_eval_results[state.global_step] = subset_loss
            print(f"FRESH: Subset evaluation at step {state.global_step}: loss={subset_loss:.6f}", flush=True)
            
            # Predict if full evaluation is worth it
            best_loss = self.best_checkpoint_info["loss"] if self.best_checkpoint_info else float('inf')
            if subset_loss > best_loss * self.skip_full_eval_threshold:
                print(f"FRESH: Skipping full evaluation - subset loss {subset_loss:.6f} > best {best_loss:.6f} * {self.skip_full_eval_threshold}", flush=True)
                # Don't do full evaluation, just return
                return control
            else:
                print(f"FRESH: Subset loss {subset_loss:.6f} is promising, proceeding with full evaluation", flush=True)
        
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)
        
        # FRESH: Smart checkpoint pruning - predict quality early
        if eval_loss is not None:
            # Track eval loss trend for prediction
            if self.last_eval_loss is not None:
                trend = eval_loss - self.last_eval_loss
                self.eval_loss_trend.append(trend)
                # Keep only last 5 trends
                if len(self.eval_loss_trend) > 5:
                    self.eval_loss_trend = self.eval_loss_trend[-5:]
                
                # Predict next checkpoint quality based on trend
                if len(self.eval_loss_trend) >= 2:
                    avg_trend = sum(self.eval_loss_trend[-3:]) / min(3, len(self.eval_loss_trend))
                    predicted_next_loss = eval_loss + avg_trend
                    self.checkpoint_predictions[state.global_step + 100] = predicted_next_loss
                    
                    # If predicted loss is very poor, suggest early stopping
                    best_loss = self.best_checkpoint_info["loss"] if self.best_checkpoint_info else eval_loss
                    if predicted_next_loss > best_loss * 1.15:  # 15% worse
                        print(f"FRESH: Predicted next checkpoint will be poor (predicted={predicted_next_loss:.6f} vs best={best_loss:.6f}), consider early stopping", flush=True)
            
            self.last_eval_loss = eval_loss
        
        # CRITICAL: Store eval_loss and train_loss at checking_step for multi-run selection
        if hasattr(self, '_capture_eval_loss_at_checking') and self._capture_eval_loss_at_checking:
            if state.global_step == self.checking_step and self.checking_mode == "second_time":
                if eval_loss is not None:
                    # Extract train_loss from log_history for storage
                    train_loss_for_storage = None
                    if state.log_history:
                        last_log = state.log_history[-1]
                        train_loss_for_storage = last_log.get("loss", None)
                    
                    my_state = get_state()
                    my_state["train"]["current_eval_loss"] = eval_loss
                    # IMPROVED: Also store train_loss for generalization_score calculation
                    if train_loss_for_storage is not None:
                        my_state["train"]["current_train_loss"] = train_loss_for_storage
                    print(f"CRITICAL: Stored eval_loss {eval_loss:.6f} and train_loss {train_loss_for_storage:.6f if train_loss_for_storage else None} at checking_step {state.global_step} for multi-run selection", flush=True)
                    
                    # Now make decision based on eval_loss instead of train_loss
                    current_is_the_best = False
                    control.should_training_stop = True
                    
                    if "runs" in my_state and len(my_state["runs"]) > 0:
                        # Use eval_loss for comparison (CRITICAL IMPROVEMENT)
                        if all("current_eval_loss" in run for run in my_state["runs"]):
                            current_min_eval_loss = min([run.get("current_eval_loss", run["current_loss"]) for run in my_state["runs"]])
                            comparison_loss = eval_loss
                            print(f"Comparing eval_loss: {eval_loss:.6f} vs best eval_loss: {current_min_eval_loss:.6f}", flush=True)
                        else:
                            # Fallback: compare with train_loss if eval_loss not available in previous runs
                            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
                            comparison_loss = eval_loss  # Still use eval_loss for this run
                            current_min_eval_loss = current_min_loss
                            print(f"Comparing eval_loss: {eval_loss:.6f} vs best train_loss: {current_min_eval_loss:.6f} (fallback)", flush=True)
                        
                        if eval_loss <= current_min_eval_loss:
                            if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                                print(f"CRITICAL: Eval loss {eval_loss:.6f} is best or equal to best {current_min_eval_loss:.6f}, continuing training", flush=True)
                                current_is_the_best = True
                            else:
                                print(f"Eval loss {eval_loss:.6f} is best, but not last run. Will continue with next LR.", flush=True)
                        else:
                            # Adaptive early termination: if eval_loss is significantly worse, stop early
                            if eval_loss > current_min_eval_loss * 1.2:  # 20% worse
                                print(f"ADAPTIVE TERMINATION: Eval loss {eval_loss:.6f} is >20% worse than best {current_min_eval_loss:.6f}, stopping early", flush=True)
                                control.should_training_stop = True
                            else:
                                print(f"Eval loss {eval_loss:.6f} is worse than best {current_min_eval_loss:.6f}, stopping this run", flush=True)
                    else:
                        # First run in the series
                        if my_state.get("next_runs", 1) == 1:
                            current_is_the_best = True
                            print(f"Only one run, continuing training", flush=True)
                    
                    if current_is_the_best:
                        control.should_training_stop = False
                        my_state["mode"] = "finish"
                    else:
                        # Don't save if not best to save time and space
                        control.should_save = False
                    
                    if is_main_process(LOCAL_RANK):
                        set_state(my_state)
                    self._capture_eval_loss_at_checking = False
                    self._checking_step_eval_loss = eval_loss
                    
                    return control
        
        # IMPROVED: Monitor train/eval gap for overfitting detection and early stopping
        overfitting_detected = False
        if state.log_history:
            last_train_loss = state.log_history[-1].get("loss", None)
            if last_train_loss and eval_loss:
                gap_ratio = eval_loss / last_train_loss if last_train_loss > 0 else float('inf')
                if gap_ratio > 2.0:  # Eval loss is 2x train loss = overfitting
                    overfitting_detected = True
                    print(f"WARNING: Overfitting detected! Train loss: {last_train_loss:.6f}, Eval loss: {eval_loss:.6f}, Ratio: {gap_ratio:.2f}", flush=True)
                    
                    # IMPROVED: More aggressive early stopping when severe overfitting detected
                    if gap_ratio > 2.5 and self.best_checkpoint_info is not None:
                        best_eval = self.best_checkpoint_info.get("loss", float('inf'))
                        if eval_loss > best_eval * 1.1:  # Current eval is 10% worse than best
                            print(f"SEVERE OVERFITTING: Stopping early to prevent further degradation (gap_ratio={gap_ratio:.2f}, eval_loss={eval_loss:.6f} vs best={best_eval:.6f})", flush=True)
                            control.should_training_stop = True
                            return control
        
        # DECISIVE: Overfitting-aware checkpoint selection
        # Track train_loss to compute generalization_score
        train_loss = None
        if state.log_history:
            last_log = state.log_history[-1]
            train_loss = last_log.get("loss", None)
        
        # Cross-entropy loss optimization: track best cross-entropy loss
        # eval_loss is already cross-entropy, so we use it directly
        if eval_loss is not None:
            # IMPROVED: Compute generalization_score with adaptive overfitting penalty
            # generalization_score = eval_loss - adaptive_penalty * overfitting_gap
            # Lower score is better (we want low eval_loss and low overfitting)
            generalization_score = eval_loss
            overfitting_penalty = 0.0
            if train_loss is not None and train_loss > 0:
                overfitting_gap = max(0, eval_loss - train_loss)
                gap_ratio = eval_loss / train_loss if train_loss > 0 else float('inf')
                
                # IMPROVED: Adaptive penalty factor based on gap severity
                # Small gap (< 1.2x): light penalty (0.2)
                # Medium gap (1.2-1.5x): moderate penalty (0.35)
                # Large gap (1.5-2.0x): strong penalty (0.5)
                # Very large gap (> 2.0x): very strong penalty (0.7)
                if gap_ratio >= 2.0:
                    penalty_factor = 0.7  # Very strong penalty for severe overfitting
                elif gap_ratio >= 1.5:
                    penalty_factor = 0.5  # Strong penalty
                elif gap_ratio >= 1.2:
                    penalty_factor = 0.35  # Moderate penalty
                else:
                    penalty_factor = 0.2  # Light penalty for small gaps
                
                overfitting_penalty = penalty_factor * overfitting_gap
                generalization_score = eval_loss - overfitting_penalty
                print(f"Step {state.global_step}: eval_loss={eval_loss:.6f}, train_loss={train_loss:.6f}, gap_ratio={gap_ratio:.3f}, overfitting_gap={overfitting_gap:.6f}, penalty_factor={penalty_factor:.2f}, generalization_score={generalization_score:.6f}", flush=True)
            else:
                print(f"Step {state.global_step}: eval_loss={eval_loss:.6f}, train_loss=None, generalization_score={generalization_score:.6f}", flush=True)
            
            # Update best checkpoint using generalization_score (not just eval_loss)
            should_update = False
            if self.best_checkpoint_info is None:
                should_update = True
            else:
                # Compare using generalization_score for better generalization
                current_best_score = self.best_checkpoint_info.get("generalization_score", self.best_checkpoint_info["loss"])
                if generalization_score < current_best_score:
                    should_update = True
            
            if should_update:
                print(f"DECISIVE: Updating best checkpoint at step {state.global_step} with generalization_score: {generalization_score:.6f} (eval_loss: {eval_loss:.6f})", flush=True)
                self.best_checkpoint_info = {
                    "loss": eval_loss,
                    "step": state.global_step,
                    "train_loss": train_loss,
                    "generalization_score": generalization_score,
                    "overfitting_penalty": overfitting_penalty
                }
                self.update_best_checkpoint = True
            
            # DECISIVE: Track top 3 checkpoints for interpolation
            checkpoint_entry = {
                "step": state.global_step,
                "eval_loss": eval_loss,
                "train_loss": train_loss,
                "generalization_score": generalization_score
            }
            
            # Add to top checkpoints list
            self.top_checkpoints.append(checkpoint_entry)
            
            # Keep only top 3 by generalization_score
            self.top_checkpoints.sort(key=lambda x: x["generalization_score"])
            if len(self.top_checkpoints) > self.max_top_checkpoints:
                self.top_checkpoints = self.top_checkpoints[:self.max_top_checkpoints]
            
            top_ckpt_info = [(c['step'], f"gen_score={c['generalization_score']:.6f}") for c in self.top_checkpoints]
            print(f"Top {len(self.top_checkpoints)} checkpoints: {top_ckpt_info}", flush=True)
            
            if not should_update and self.best_checkpoint_info is not None:
                current_best_score = self.best_checkpoint_info.get("generalization_score", self.best_checkpoint_info["loss"])
                print(f" At step: {state.global_step} The generalization_score: {generalization_score:.6f} is not better than current best: {current_best_score:.6f}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)
            

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
                try:
                    # Use ignore_errors=True to handle FileNotFoundError and other errors gracefully
                    shutil.rmtree(self.submission_dir, ignore_errors=True)
                except Exception as e:
                    # Catch all exceptions to prevent crashes from file system issues
                    # This handles FileNotFoundError, PermissionError, and other filesystem errors
                    print(f"Warning: Error removing submission directory (ignoring): {e}", flush=True)
                    # Try to continue even if removal failed
                    try:
                        # If rmtree failed, try to remove individual files that might be causing issues
                        import stat
                        def handle_remove_readonly(func, path, exc):
                            if os.path.exists(path):
                                os.chmod(path, stat.S_IWRITE)
                                func(path)
                        shutil.rmtree(self.submission_dir, onerror=handle_remove_readonly)
                    except Exception:
                        # If all else fails, just continue - the copytree will handle it
                        pass
                
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
                try:
                    # Use ignore_errors=True to handle FileNotFoundError and other errors gracefully
                    shutil.rmtree(self.submission_dir, ignore_errors=True)
                except Exception as e:
                    # Catch all exceptions to prevent crashes from file system issues
                    # This handles FileNotFoundError, PermissionError, and other filesystem errors
                    print(f"Warning: Error removing submission directory (ignoring): {e}", flush=True)
                    # Try to continue even if removal failed
                    try:
                        # If rmtree failed, try to remove individual files that might be causing issues
                        import stat
                        def handle_remove_readonly(func, path, exc):
                            if os.path.exists(path):
                                os.chmod(path, stat.S_IWRITE)
                                func(path)
                        shutil.rmtree(self.submission_dir, onerror=handle_remove_readonly)
                    except Exception:
                        # If all else fails, just continue - the copytree will handle it
                        pass
            
            best_eval_loss = self.best_checkpoint_info["loss"]
            best_gen_score = self.best_checkpoint_info.get("generalization_score", best_eval_loss)
            
            # DECISIVE: Checkpoint interpolation - use top 2-3 checkpoints if available
            # Improved: Use top 3 checkpoints with weighted blending for better generalization
            use_interpolation = (
                len(self.top_checkpoints) >= 2 
                and self.top_checkpoints[0]["generalization_score"] < self.top_checkpoints[1]["generalization_score"] * 1.08  # Within 8% of each other
            )
            
            if use_interpolation:
                num_ckpts = min(len(self.top_checkpoints), 3)  # Use up to 3 checkpoints
                print(f"DECISIVE: Interpolating top {num_ckpts} checkpoints for better generalization", flush=True)
                
                # IMPROVED: Exponential weighting based on generalization_score differences
                # Better checkpoints get exponentially higher weights
                ckpts = self.top_checkpoints[:num_ckpts]
                best_score = ckpts[0]["generalization_score"]
                
                # Calculate exponential weights: exp(-alpha * score_diff)
                # alpha controls how quickly weights decay (higher = more emphasis on best)
                alpha = 10.0  # Tuned for good balance
                raw_weights = []
                for ckpt in ckpts:
                    score_diff = ckpt["generalization_score"] - best_score
                    # Normalize: weight = exp(-alpha * normalized_score_diff)
                    # Use relative difference to handle different loss scales
                    # Add safeguard: if best_score is very small, use absolute difference with cap
                    if best_score > 0.01:  # Use relative normalization if score is reasonable
                        normalized_diff = score_diff / best_score
                    else:
                        # For very small scores, use absolute difference with cap to prevent extreme weights
                        normalized_diff = min(score_diff, 1.0)  # Cap at 1.0 to prevent extreme weights
                    weight = math.exp(-alpha * max(0, normalized_diff))
                    raw_weights.append(weight)
                
                # Normalize weights to sum to 1
                total_weight = sum(raw_weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in raw_weights]
                else:
                    # Fallback to fixed weights if calculation fails
                    if num_ckpts == 3:
                        weights = [0.5, 0.3, 0.2]
                    elif num_ckpts == 2:
                        weights = [0.6, 0.4]
                    else:
                        weights = [1.0]
                
                for i, ckpt in enumerate(ckpts):
                    print(f"  Checkpoint {i+1}: step={ckpt['step']}, gen_score={ckpt['generalization_score']:.6f}, eval_loss={ckpt['eval_loss']:.6f}, weight={weights[i]:.2f}", flush=True)
                
                try:
                    # Load all checkpoints for interpolation
                    ckpt_paths = [os.path.join(self.output_dir, f"checkpoint-{ckpt['step']}") for ckpt in ckpts]
                    
                    # Check if all checkpoints exist
                    if all(os.path.exists(path) for path in ckpt_paths):
                        # Load state dicts
                        import glob
                        try:
                            from safetensors.torch import load_file, save_file
                            has_safetensors = True
                        except ImportError:
                            has_safetensors = False
                            print(f"  Warning: safetensors not available, using .bin files only", flush=True)
                        
                        # Find model files (could be .bin or .safetensors)
                        ckpt1_path = ckpt_paths[0]
                        ckpt1_files = glob.glob(os.path.join(ckpt1_path, "*.safetensors"))
                        if not ckpt1_files:
                            ckpt1_files = glob.glob(os.path.join(ckpt1_path, "*.bin"))
                        
                        if ckpt1_files:
                            # Copy first checkpoint as base
                            shutil.copytree(ckpt1_path, self.submission_dir)
                            
                            # Interpolate weights: blend all checkpoints
                            weight_str = " + ".join([f"{w:.2f}*ckpt{i+1}" for i, w in enumerate(weights)])
                            print(f"  Interpolating weights: {weight_str}", flush=True)
                            
                            for file_path in ckpt1_files:
                                filename = os.path.basename(file_path)
                                output_path = os.path.join(self.submission_dir, filename)
                                
                                # Load all checkpoint state dicts
                                states = []
                                for ckpt_path in ckpt_paths:
                                    file_path_ckpt = os.path.join(ckpt_path, filename)
                                    if os.path.exists(file_path_ckpt):
                                        if filename.endswith(".safetensors") and has_safetensors:
                                            states.append(load_file(file_path_ckpt))
                                        else:
                                            states.append(torch.load(file_path_ckpt, map_location="cpu"))
                                
                                if len(states) == len(weights):
                                    # Interpolate: weighted sum of all checkpoints
                                    interpolated = {}
                                    for key in states[0].keys():
                                        # Check if all states have this key with same shape
                                        if all(key in state and state[key].shape == states[0][key].shape for state in states):
                                            interpolated[key] = sum(weights[i] * states[i][key] for i in range(len(states)))
                                        else:
                                            interpolated[key] = states[0][key]  # Use first checkpoint if key missing
                                    
                                    # Save interpolated weights
                                    if filename.endswith(".safetensors") and has_safetensors:
                                        save_file(interpolated, output_path)
                                    else:
                                        torch.save(interpolated, output_path)
                            
                            print(f"  Successfully interpolated {num_ckpts} checkpoints", flush=True)
                            step_info = ",".join([f"step{i+1}={ckpt['step']}" for i, ckpt in enumerate(ckpts)])
                            weight_info = ",".join([f"w{i+1}={w:.2f}" for i, w in enumerate(weights)])
                            interpolation_info = f"interpolated:{step_info},{weight_info}"
                        else:
                            # Fallback: just use best checkpoint
                            print(f"  Warning: Could not find model files, using best checkpoint only", flush=True)
                            shutil.copytree(
                                os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                                self.submission_dir
                            )
                            interpolation_info = "single_best"
                    else:
                        # Fallback: just use best checkpoint
                        print(f"  Warning: Checkpoints not found, using best checkpoint only", flush=True)
                        shutil.copytree(
                            os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                            self.submission_dir
                        )
                        interpolation_info = "single_best"
                except Exception as e:
                    # Fallback: just use best checkpoint if interpolation fails
                    print(f"  Warning: Interpolation failed ({e}), using best checkpoint only", flush=True)
                    if os.path.exists(self.submission_dir):
                        try:
                            # Use ignore_errors=True to handle FileNotFoundError and other errors gracefully
                            shutil.rmtree(self.submission_dir, ignore_errors=True)
                        except Exception as err:
                            # Catch all exceptions to prevent crashes from file system issues
                            print(f"Warning: Error removing submission directory (ignoring): {err}", flush=True)
                            # Try to continue even if removal failed
                            try:
                                import stat
                                def handle_remove_readonly(func, path, exc):
                                    if os.path.exists(path):
                                        os.chmod(path, stat.S_IWRITE)
                                        func(path)
                                shutil.rmtree(self.submission_dir, onerror=handle_remove_readonly)
                            except Exception:
                                # If all else fails, just continue
                                pass
                    shutil.copytree(
                        os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                        self.submission_dir
                    )
                    interpolation_info = "single_best"
            else:
                # Use single best checkpoint (with generalization_score)
                print(f"Using best checkpoint (generalization_score={best_gen_score:.6f}, eval_loss={best_eval_loss:.6f})", flush=True)
                shutil.copytree(
                    os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}"),
                    self.submission_dir
                )
                interpolation_info = "single_best"
            
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{self.best_checkpoint_info['step']},{best_eval_loss},{best_gen_score},{interpolation_info}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called when training ends. Extract final metrics and update LR lookup table automatically.
        """
        if not is_main_process(LOCAL_RANK):
            return
        
        # Only update if enabled and task_type is provided
        if not self.update_lr_lookup or not self.task_type:
            return
        
        # Extract final metrics from training state
        final_eval_loss = None
        final_train_loss = None
        
        # Try to get final eval_loss from log_history
        if state.log_history:
            # Look for the last evaluation entry
            for log_entry in reversed(state.log_history):
                if "eval_loss" in log_entry:
                    final_eval_loss = log_entry["eval_loss"]
                    break
                # For GRPO, check eval_reward
                if "eval_reward" in log_entry:
                    final_eval_loss = -log_entry["eval_reward"]  # Convert reward to loss
                    break
            
            # Get final train_loss
            for log_entry in reversed(state.log_history):
                if "loss" in log_entry and "eval_loss" not in log_entry:
                    final_train_loss = log_entry["loss"]
                    break
        
        # If we have best checkpoint info, use that for eval_loss (more reliable)
        if self.best_checkpoint_info and "loss" in self.best_checkpoint_info:
            if final_eval_loss is None:
                final_eval_loss = self.best_checkpoint_info["loss"]
        
        # Get learning rate from training args
        learning_rate = getattr(args, "learning_rate", None)
        if learning_rate is None:
            # Try to get from log_history
            if state.log_history:
                for log_entry in reversed(state.log_history):
                    if "learning_rate" in log_entry:
                        learning_rate = log_entry["learning_rate"]
                        break
        
        # Skip update if we don't have essential information
        if learning_rate is None:
            print(f"  [LR Update] Skipping: No learning rate found", flush=True)
            return
        
        if final_eval_loss is None and final_train_loss is None:
            print(f"  [LR Update] Skipping: No loss metrics found", flush=True)
            return
        
        # Map task type to lookup table format
        task_type_map = {
            "InstructTextTask": "instruct",
            "ChatTask": "instruct",  # Chat uses same lookup as instruct
            "DpoTask": "dpo",
            "GrpoTask": "grpo",
        }
        
        lookup_task_type = task_type_map.get(self.task_type)
        if not lookup_task_type:
            print(f"  [LR Update] Skipping: Unknown task type '{self.task_type}'", flush=True)
            return
        
        # Prepare metadata
        metadata = self.metadata.copy()
        metadata.update({
            "final_step": state.global_step,
            "best_checkpoint_step": self.best_checkpoint_info.get("step") if self.best_checkpoint_info else None,
            "total_steps": self.total_steps_all_epochs,
        })
        
        # CRITICAL: Add batch_size and effective_batch_size for joint LR-batch optimization
        if hasattr(args, "per_device_train_batch_size"):
            metadata["batch_size"] = args.per_device_train_batch_size
            # Calculate effective batch size (per_device * gpu_count * grad_accum)
            gpu_count = getattr(args, "world_size", 1)
            grad_accum = getattr(args, "gradient_accumulation_steps", 1)
            metadata["effective_batch_size"] = args.per_device_train_batch_size * gpu_count * grad_accum
        elif "batch_size" not in metadata:
            # Fallback: try to get from metadata if already set
            pass
        
        # Add learning_rate to metadata if not already present
        if "learning_rate" not in metadata:
            metadata["learning_rate"] = learning_rate
        
        # Update LR lookup table
        try:
            from lrs_lookup import update_lr_lookup
            
            updated = update_lr_lookup(
                task_type=lookup_task_type,
                model=self.original_model_name,
                learning_rate=learning_rate,
                eval_loss=final_eval_loss,
                train_loss=final_train_loss,
                metadata=metadata
            )
            
            if updated:
                print(f"  [LR Update] Successfully updated LR lookup table for {self.original_model_name[:50]}...", flush=True)
                print(f"  [LR Update]   - LR: {learning_rate:.8f}", flush=True)
            
            # AutoML: Also update hyperparameter lookup table
            try:
                from hyperparam_optimizer import update_hyperparams
                from model_utility import get_model_num_params
                
                param_nums = get_model_num_params(self.original_model_name, "")
                update_hyperparams(
                    task_type=lookup_task_type,
                    model=self.original_model_name,
                    param_nums=param_nums,
                    eval_loss=final_eval_loss,
                    train_loss=final_train_loss,
                    metadata=metadata
                )
            except Exception as e:
                print(f"  [AutoML] Warning: Could not update hyperparameter lookup: {e}", flush=True)
                if final_eval_loss:
                    print(f"  [LR Update]   - Eval Loss: {final_eval_loss:.6f}", flush=True)
                if final_train_loss:
                    print(f"  [LR Update]   - Train Loss: {final_train_loss:.6f}", flush=True)
            else:
                print(f"  [LR Update] Lookup table not updated (existing entry has better or equal loss)", flush=True)
        except Exception as e:
            print(f"  [LR Update] Error updating LR lookup table: {e}", flush=True)
            import traceback
            traceback.print_exc()


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}", flush=True)
        else:
            print(f"state.log_history is empty", flush=True)
            
        if eval_loss is not None:
            eval_loss = - eval_loss
            
        return eval_loss
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps
        # FRESH: Time-aware adaptive training
        self.time_aware_mode = "normal"  # normal, tight, ample

    def __call__(self, global_step: int) -> dict:
        # FRESH: Time-aware adaptive training - adjust strategy based on remaining time
        remaining_minutes = None
        if self.end_time:
            try:
                end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
                end_time_obj = end_time_obj.replace(tzinfo=timezone.utc)
                now = datetime.datetime.now(timezone.utc)
                remaining_minutes = (end_time_obj - now).total_seconds() / 60
                
                # Adjust mode based on remaining time
                if remaining_minutes < 30:  # Less than 30 minutes
                    self.time_aware_mode = "tight"
                elif remaining_minutes > 120:  # More than 2 hours
                    self.time_aware_mode = "ample"
                else:
                    self.time_aware_mode = "normal"
            except:
                pass
        
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        # IMPROVED: Adaptive evaluation frequency - more frequent near end and when overfitting likely
        # FRESH: Time-aware - adjust frequency based on remaining time
        if self.max_steps != -1 and self.max_steps > 0:
            progress = global_step / self.max_steps
            
            # IMPROVED: More frequent evaluation in later stages to catch best checkpoint
            # In tight time mode, evaluate more frequently to catch best checkpoint quickly
            if self.time_aware_mode == "tight":
                if progress > 0.7:  # Last 30% of training
                    if global_step % 25 == 0 and global_step > 1:  # Very frequent (improved from 30)
                        return {"eval": True, "reason": "periodic_frequent_tight"}
                elif progress > 0.5:  # Last 50% of training
                    if global_step % 40 == 0 and global_step > 1:  # Improved from 50
                        return {"eval": True, "reason": "periodic_frequent_tight"}
            elif progress > 0.8:  # Last 20% of training
                # Evaluate every 40 steps (improved from 50 - more frequent)
                if global_step % 40 == 0 and global_step > 1:
                    return {"eval": True, "reason": "periodic_frequent"}
            elif progress > 0.6:  # Last 40% of training
                # Evaluate every 80 steps (improved from 100 - more frequent)
                if global_step % 80 == 0 and global_step > 1:
                    return {"eval": True, "reason": "periodic_frequent"}
            elif progress > 0.4:  # IMPROVED: Also evaluate more frequently in mid-training
                # Evaluate every 150 steps (moderately frequent)
                if global_step % 150 == 0 and global_step > 1:
                    return {"eval": True, "reason": "periodic_mid"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"***ALERT: The time is about to run out need to eval & save the model", flush=True)
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
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


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 300, min_delta: float = 0.0001, hours_to_complete: float = None):
        # Adaptive patience: reduce for short jobs to stop faster and save time
        if hours_to_complete is not None and hours_to_complete > 0:
            if hours_to_complete <= 0.75:  # Very short jobs
                self.patience = 50  # Very aggressive early stopping
            elif hours_to_complete <= 1.5:  # Short jobs
                self.patience = 100
            elif hours_to_complete <= 2.0:  # Medium jobs
                self.patience = 200
            else:  # Long jobs
                self.patience = patience  # Use provided patience
            print(f"Adaptive early stopping: patience={self.patience} for {hours_to_complete:.2f}h job", flush=True)
        else:
            self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Try to get eval_loss first, then eval_reward (for GRPO)
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is None and state.log_history:
            last_log_entry = state.log_history[-1]
            eval_reward = last_log_entry.get("eval_reward", None)
            if eval_reward is not None:
                # For GRPO: negate reward to convert to loss (lower is better)
                eval_loss = -eval_reward
        
        if eval_loss is None:
            return control
        
        if self.best_loss is None:
            self.best_loss = eval_loss
            self.wait = 0
        elif eval_loss < self.best_loss - self.min_delta:
            # Significant improvement
            self.best_loss = eval_loss
            self.wait = 0
            print(f"Early stopping: Improved eval_loss to {eval_loss:.6f} (best: {self.best_loss:.6f}), resetting patience counter", flush=True)
        else:
            # No improvement
            self.wait += 1
            print(f"Early stopping: No improvement for {self.wait}/{self.patience} evaluations. Best loss: {self.best_loss:.6f}, Current: {eval_loss:.6f}", flush=True)
            if self.wait >= self.patience:
                print(f"Early stopping triggered at step {state.global_step}. Best loss: {self.best_loss:.6f}. Stopping training to prevent overfitting.", flush=True)
                control.should_training_stop = True
                self.stopped_epoch = state.epoch
        
        return control