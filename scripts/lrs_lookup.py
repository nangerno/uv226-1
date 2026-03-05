import json 
import os 
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timezone
current_dir = os.path.dirname(os.path.abspath(__file__))

# File paths for LR lookup tables
LR_FILES = {
    "dpo": os.path.join(current_dir, "lrs/dpo.json"),
    "grpo": os.path.join(current_dir, "lrs/grpo.json"),
    "instruct": os.path.join(current_dir, "lrs/instruct.json"),
    "grpo_python": os.path.join(current_dir, "lrs/grpo_python.json"),
}

# Load LR lookup tables
with open(LR_FILES["dpo"], "r") as f:
    dpo_lrs = json.load(f)

with open(LR_FILES["grpo"], "r") as f:
    grpo_lrs = json.load(f)

with open(LR_FILES["instruct"], "r") as f:
    instruct_lrs = json.load(f)

with open(LR_FILES["grpo_python"], "r") as f:
    grpo_python_lrs = json.load(f)


def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 


def get_dpo_lr(model: str):
    hashed_model = hash_model(model)
    for lr in dpo_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def get_grpo_lr(model: str):
    hashed_model = hash_model(model)
    for lr in grpo_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None

def get_instruct_lr(model: str):
    hashed_model = hash_model(model)
    for lr in instruct_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def get_grpo_python_lr(model: str):
    hashed_model = hash_model(model)
    for lr in grpo_python_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def update_lr_lookup(
    task_type: str,
    model: str,
    learning_rate: float,
    eval_loss: Optional[float] = None,
    train_loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Automatically update LR lookup table with new learning rate if it performs better.
    
    Args:
        task_type: One of "dpo", "grpo", "instruct", "grpo_python"
        model: Model name/identifier
        learning_rate: Learning rate that was used
        eval_loss: Final evaluation loss (preferred for comparison)
        train_loss: Final training loss (fallback if eval_loss not available)
        metadata: Optional metadata (batch_size, reg_ratio, etc.)
    
    Returns:
        True if lookup table was updated, False otherwise
    """
    if task_type not in LR_FILES:
        print(f"Warning: Unknown task type '{task_type}' for LR lookup update", flush=True)
        return False
    
    hashed_model = hash_model(model)
    lr_file = LR_FILES[task_type]
    
    # Load current lookup table
    try:
        with open(lr_file, "r") as f:
            lr_list = json.load(f)
    except Exception as e:
        print(f"Error loading LR lookup file {lr_file}: {e}", flush=True)
        return False
    
    # Find existing entry
    existing_entry = None
    existing_index = -1
    for i, entry in enumerate(lr_list):
        if entry.get("h") == hashed_model:
            existing_entry = entry
            existing_index = i
            break
    
    # Determine if we should update (use eval_loss if available, else train_loss)
    loss_to_compare = eval_loss if eval_loss is not None else train_loss
    existing_loss = None
    
    if existing_entry:
        # Check if existing entry has loss information
        existing_loss = existing_entry.get("eval_loss") or existing_entry.get("train_loss")
    
    # Update if:
    # 1. No existing entry (add new)
    # 2. New loss is better (lower) than existing
    # 3. Existing entry has no loss info
    should_update = False
    if existing_entry is None:
        should_update = True
        print(f"  [LR Update] New model entry for {model[:50]}...", flush=True)
    elif loss_to_compare is not None:
        if existing_loss is None:
            should_update = True
            print(f"  [LR Update] Existing entry has no loss, updating with loss {loss_to_compare:.6f}", flush=True)
        elif loss_to_compare < existing_loss:
            should_update = True
            print(f"  [LR Update] Better loss found: {loss_to_compare:.6f} < {existing_loss:.6f}, updating LR", flush=True)
        else:
            print(f"  [LR Update] Existing loss {existing_loss:.6f} is better than {loss_to_compare:.6f}, keeping existing LR", flush=True)
    else:
        # No loss info available, but we can still update if no existing entry
        if existing_entry is None:
            should_update = True
    
    if should_update:
        new_entry = {
            "h": hashed_model,
            "lr": learning_rate,
            "timestamp": datetime.now(timezone.utc).isoformat()  # Add timestamp for tracking
        }
        
        # Add loss information if available
        if eval_loss is not None:
            new_entry["eval_loss"] = eval_loss
        if train_loss is not None:
            new_entry["train_loss"] = train_loss
        
        # Add metadata if provided (flatten important fields to top level for easier access)
        if metadata:
            new_entry["metadata"] = metadata
            # Also store key metadata fields at top level for easier filtering/matching
            if "batch_size" in metadata:
                new_entry["batch_size"] = metadata["batch_size"]
            if "use_lora" in metadata:
                new_entry["use_lora"] = metadata["use_lora"]
            if "lora_rank" in metadata:
                new_entry["lora_rank"] = metadata["lora_rank"]
            if "hours_to_complete" in metadata:
                new_entry["hours_to_complete"] = metadata["hours_to_complete"]
        
        # Update or add entry
        if existing_index >= 0:
            lr_list[existing_index] = new_entry
            print(f"  [LR Update] Updated existing entry for {model[:50]}... with LR {learning_rate:.8f}", flush=True)
        else:
            lr_list.append(new_entry)
            print(f"  [LR Update] Added new entry for {model[:50]}... with LR {learning_rate:.8f}", flush=True)
        
        # Save updated lookup table
        try:
            # Create backup
            backup_file = lr_file + ".backup"
            if os.path.exists(lr_file):
                import shutil
                shutil.copy2(lr_file, backup_file)
            
            # Write updated table
            with open(lr_file, "w") as f:
                json.dump(lr_list, f, indent=4)
            
            print(f"  [LR Update] Successfully updated {lr_file}", flush=True)
            
            # Reload the global variable
            global dpo_lrs, grpo_lrs, instruct_lrs, grpo_python_lrs
            if task_type == "dpo":
                dpo_lrs = lr_list
            elif task_type == "grpo":
                grpo_lrs = lr_list
            elif task_type == "instruct":
                instruct_lrs = lr_list
            elif task_type == "grpo_python":
                grpo_python_lrs = lr_list
            
            return True
        except Exception as e:
            print(f"  [LR Update] Error saving LR lookup file {lr_file}: {e}", flush=True)
            return False
    
    return False
