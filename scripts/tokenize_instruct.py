from typing import Dict, Any
from axolotl.utils.dict import DictDefault
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Callable
from axolotl.utils.data import load_tokenized_prepared_datasets
import torch
import logging
from datetime import datetime
import sys
import random
import json
import requests
import os
import shutil
import typer


def _process_custom_dataset_fields(custom_type_dict: dict) -> dict:
    if not custom_type_dict.get("field_output"):
        return {
            "type": "completion",
            "field": custom_type_dict.get("field_instruction"),
        }

    processed_dict = custom_type_dict.copy()
    processed_dict.setdefault("no_input_format", "{instruction}")
    if processed_dict.get("field_input"):
        processed_dict.setdefault("format", "{instruction} {input}")
    else:
        processed_dict.setdefault("format", "{instruction}")

    return {"format": "custom", "type": processed_dict}


def _process_chat_template_dataset_fields(dataset_dict: dict) -> dict:
    processed_dict = {}

    processed_dict["chat_template"] = dataset_dict["chat_template"]
    processed_dict["type"] = "chat_template"
    processed_dict["field_messages"] = dataset_dict["chat_column"]
    processed_dict["message_field_role"] = dataset_dict["chat_role_field"]
    processed_dict["message_field_content"] = dataset_dict["chat_content_field"]
    processed_dict["roles"] = {
        "assistant": [dataset_dict["chat_assistant_reference"]],
        "user": [dataset_dict["chat_user_reference"]],
    }

    processed_dict["message_property_mappings"] = {
        "role": dataset_dict["chat_role_field"],
        "content": dataset_dict["chat_content_field"],
    }

    return processed_dict


def create_dataset_entry(
    data_path: str,
    dataset_type: Dict,
    file_format: str,
) -> dict:
    dataset_entry = {"path": data_path}
    custom_type_dict = {
        key: value for key, value in dataset_type.items() if value is not None
    }
    # if data_type is chat_template, use _process_chat_template_dataset_fields
    if "chat_template" in dataset_type:
        print("Processing chat template dataset type")
        dataset_entry.update(_process_chat_template_dataset_fields(dataset_type))
    else:
        print("Processing instruct dataset type")
        dataset_entry.update(_process_custom_dataset_fields(custom_type_dict))

    # if file_format != FileFormat.HF:
    dataset_entry["ds_type"] = file_format
    # Originally: dataset_entry["data_files"] = [os.path.basename(dataset)]
    dataset_entry["data_files"] = [data_path]
    return dataset_entry


def load_and_update_evaluation_config(
    data_path: str,
    dataset_type: Any,
    file_format: str,
    finetuned_model: Any,
    config_path: str,
    max_length: int = -1,
) -> DictDefault:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    if max_length > 0:
        config_dict["sequence_len"] = max_length

    dataset_entry = create_dataset_entry(
        data_path=data_path,
        dataset_type=dataset_type,
        file_format=file_format,
    )
    config_dict["datasets"] = [dataset_entry]

    # max_embeddings = getattr(finetuned_model.config, "max_position_embeddings", None)

    # if max_embeddings and max_embeddings < 2 * config_dict["sequence_len"]:
    #    config_dict["sequence_len"] = ceil(max_embeddings / 2)

    return DictDefault(config_dict)


def _load_evaluation_dataset(
    evaluation_config: DictDefault, tokenizer: AutoTokenizer
) -> Dataset:
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(
        tokenizer, evaluation_config, prepared_path
    )

    original_length = len(eval_dataset)
    eval_dataset = [
        sample
        for sample in eval_dataset
        if any(label != -100 for label in sample["labels"])
    ]
    filtered_length = len(eval_dataset)

    print(
        f"Filtered out {original_length - filtered_length} samples with empty outputs"
    )
    print(f"Loaded dataset with {filtered_length} samples")
    return eval_dataset



def is_repetitive(text, threshold=0.5):
    """Check if text is too repetitive"""
    if not text or len(text) < 10:
        return False
    words = text.split()
    if len(words) < 10:
        return False
    unique_words = len(set(words))
    repetition_ratio = unique_words / len(words)
    return repetition_ratio < threshold

def has_low_information_content(text):
    """Check if text has low information content (too many common words)"""
    if not text:
        return True
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can'}
    words = text.lower().split()
    if len(words) == 0:
        return True
    common_ratio = sum(1 for w in words if w in common_words) / len(words)
    return common_ratio > 0.6  # More than 60% common words


def remove_empty_output_items_fast(items: list):
    """Fast filtering to keep tokenization prep overhead low."""
    result = []
    for item in items:
        if "output" in item and not item["output"]:
            continue
        if "input" in item and "instruct" in item:
            if not item["instruct"] and not item["input"]:
                continue
        if "output" in item and type(item["output"]) is not str:
            continue

        if (
            "instruct" in item
            and type(item["instruct"]) is not str
            and item["instruct"] is not None
        ):
            continue
        if (
            "input" in item
            and type(item["input"]) is not str
            and item["input"] is not None
        ):
            continue

        result.append(item)
    return result


def remove_empty_output_items_lite(items: list):
    """
    Lightweight quality filter: keeps overhead low while removing the most harmful junk.

    - Drops empty/invalid rows (same as fast)
    - Drops trivially-short outputs
    - Drops exact duplicate samples (instruct+input+output)
    """
    result = []
    seen = set()
    for item in remove_empty_output_items_fast(items):
        out = item.get("output", "")
        if isinstance(out, str) and len(out.strip()) < 5:
            continue

        key = (
            item.get("instruct", ""),
            item.get("input", ""),
            item.get("output", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def remove_empty_output_items(items: list):
    """Remove empty and low-quality output items with enhanced filtering"""
    result = []
    filtered_count = 0
    for item in items:
        # Basic filtering
        if "output" in item and not item["output"]:
            filtered_count += 1
            continue
        if "input" in item and "instruct" in item:
            if not item["instruct"] and not item["input"]:
                filtered_count += 1
                continue
        if "output" in item and type(item["output"]) is not str:
            filtered_count += 1
            continue

        if (
            "instruct" in item
            and type(item["instruct"]) is not str
            and item["instruct"] is not None
        ):
            filtered_count += 1
            continue
        if (
            "input" in item
            and type(item["input"]) is not str
            and item["input"] is not None
        ):
            filtered_count += 1
            continue
        
        # Enhanced quality filtering
        output = item.get("output", "")
        if output:
            # Filter very short outputs
            if len(output) < 5:
                filtered_count += 1
                continue
            # Filter repetitive outputs
            if is_repetitive(output):
                filtered_count += 1
                continue
            # Filter low information content
            if has_low_information_content(output):
                filtered_count += 1
                continue
        
        result.append(item)
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} low-quality items")
    return result


def replace_wrong_token_in_item(item: dict):
    for key in item:
        if type(item[key]) is str:
            item[key] = item[key].replace("[PAD]", "")
    return item

def split_dataset(
    total_data_path: str,
    train_data_path: str,
    dev_data_path: str,
    seed: int = 42,
    dev_size: int = 200,
    max_data_size: int = -1
):
    """Split the dataset into train and dev with test distribution matching"""
    # Load the dataset
    with open(total_data_path, "r") as file:
        data = json.load(file)

    if max_data_size > 0:
        data = data[:max_data_size]

    # Strategy: Use last N samples for dev set (test sets often from later data)
    # This better matches test distribution than random split
    # If dataset is small, use random split instead
    if len(data) > dev_size * 2:
        # Use last samples for dev (better matches test distribution)
        dev_items = data[-dev_size:]
        train_items = data[:-dev_size]
        print(f"Using last {dev_size} samples for dev set (test distribution matching)")
    else:
        # Fallback to random split for small datasets
        random.seed(seed)
        random.shuffle(data)
        dev_items = data[:dev_size]
        train_items = data[dev_size:]
        print(f"Using random split for dev set (dataset too small for distribution matching)")
    # Save the train and dev datasets
    # Filtering mode:
    # - QUALITY_FILTER_MODE=off|lite|full (preferred)
    # - ENABLE_QUALITY_FILTER=1 (backward-compat, same as full)
    mode = (os.getenv("QUALITY_FILTER_MODE", "") or "").strip().lower()
    if not mode:
        mode = "full" if os.getenv("ENABLE_QUALITY_FILTER", "0") == "1" else "lite"

    if mode == "full":
        filter_fn = remove_empty_output_items
    elif mode == "off":
        filter_fn = remove_empty_output_items_fast
    else:
        filter_fn = remove_empty_output_items_lite

    with open(train_data_path, "w") as file:
        before_len = len(train_items)
        train_items = filter_fn(train_items)
        after_len = len(train_items)
        print(f"Removed {before_len - after_len} empty output items from train_ds")
        json.dump(train_items, file, ensure_ascii=False)

    with open(dev_data_path, "w") as file:
        before_len = len(dev_items)
        dev_items = filter_fn(dev_items)
        after_len = len(dev_items)
        print(f"Removed {before_len - after_len} empty output items from dev_ds")
        json.dump(dev_items, file, ensure_ascii=False)

    print(
        f"split {total_data_path} ({len(data)} items) into {train_data_path} ({len(train_items)} items) and {dev_data_path} ({len(dev_items)} items)"
    )


def data_stat(items: list):
    lengths = []
    for item in items:
        lengths.append(len(item["input_ids"]))


def tokenize_dataset(
    tokenizer: AutoTokenizer,
    data_path: str,
    dataset_type: Dict,
    config_path: str,
    output_path: str,
    max_length: int = -1,
):
    evaluation_config = load_and_update_evaluation_config(
        data_path, dataset_type, "json", None, config_path, max_length
    )
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    eval_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)
    # now dump this
    result = []
    for i in range(len(eval_dataset)):
        dp = eval_dataset[i]
        result.append(dp)

    print(f"Dumped {len(result)} samples to {output_path}")

    with open(output_path, "w") as file:
        json.dump(result, file, ensure_ascii=False)


def main(training_request_path: str):
    t1 = datetime.now()
    with open(training_request_path, "r") as file:
        training_request = json.load(file)

    # dataset is already downloaded at: training_request["train_request"]["dataset"]
    task_id = training_request["train_request"]["task_id"]
    total_path = training_request["train_request"]["dataset"]
    train_path = f"datasets/train_{task_id}.json"
    dev_path = f"datasets/dev_{task_id}.json"
    max_data_size = training_request["train_request"].get("max_data_size", -1)
    if max_data_size > 0:
        print(
            f"Max data size is {max_data_size}, so we will only extract {max_data_size} samples randomly"
        )

    # Adaptive dev split for short jobs (tokenization + eval overhead)
    hours_to_complete = float(training_request["train_request"].get("hours_to_complete", 0) or 0)
    dev_size = int(training_request["train_request"].get("dev_size", 0) or 0)
    if dev_size <= 0:
        if hours_to_complete > 0 and hours_to_complete <= 0.5:
            dev_size = 50
        elif hours_to_complete > 0 and hours_to_complete <= 1.0:
            dev_size = 100
        elif hours_to_complete > 0 and hours_to_complete <= 2.0:
            dev_size = 150
        else:
            dev_size = 200

    split_dataset(
        total_path,
        train_path,
        dev_path,
        dev_size=dev_size,
        max_data_size=max_data_size,
    )
    
    config_path = "test_axolotl.yml"
    tokenizer = AutoTokenizer.from_pretrained(
        training_request["train_request"]["model_path"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure consistent padding side for causal LMs (matches validator)
    tokenizer.padding_side = "left"  # Left padding for causal LMs
    max_length = -1  # default value in test_axolot.yml

    if "max_length" in training_request["train_request"]:
        max_length = training_request["train_request"]["max_length"]

    print(f"max_length={max_length}")

    train_tok_path = f"datasets/train_tokenized_{task_id}.json"
    dev_tok_path = f"datasets/dev_tokenized_{task_id}.json"
    skip_if_exists = os.getenv("SKIP_TOKENIZE_IF_EXISTS", "1") == "1"
    force_retokenize = bool(training_request["train_request"].get("force_retokenize", False))
    if skip_if_exists and not force_retokenize and os.path.exists(train_tok_path) and os.path.exists(dev_tok_path):
        print("Tokenized files already exist; skipping tokenization (SKIP_TOKENIZE_IF_EXISTS=1).", flush=True)
        t2 = datetime.now()
        print(f"Tokenization completed in {(t2 - t1).seconds} seconds")
        return

    tokenize_dataset(
        tokenizer,
        train_path,
        training_request["train_request"]["dataset_type"],
        config_path,
        train_tok_path,
        max_length=max_length,
    )
    
    tokenize_dataset(
        tokenizer,
        dev_path,
        training_request["train_request"]["dataset_type"],
        config_path,
        dev_tok_path,
        max_length=max_length,
    )

    t2 = datetime.now()
    print(f"Tokenization completed in {(t2 - t1).seconds} seconds")


if __name__ == "__main__":
    typer.run(main)
