import argparse
import asyncio
import os
import shutil
import tempfile

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from transformers import CLIPTokenizer, T5TokenizerFast
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
)

from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
import train_cst as cst
import training_paths as train_paths
import json 

hf_api = HfApi()

# Network-related exceptions that should trigger retries
NETWORK_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

def is_network_error(exception):
    """Check if exception is network-related"""
    exception_str = str(exception).lower()
    network_keywords = [
        'name resolution',
        'temporary failure',
        'connection',
        'timeout',
        'max retries',
        'network',
        'dns',
        'resolve',
    ]
    return any(keyword in exception_str for keyword in network_keywords)

# Retry decorator for HuggingFace API calls
hf_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception(
        lambda e: isinstance(e, NETWORK_EXCEPTIONS) or is_network_error(e)
    ),
    reraise=True,
)


async def download_text_dataset(task_id, dataset_url, file_format, dataset_dir, data_size):
    os.makedirs(dataset_dir, exist_ok=True)

    if file_format == FileFormat.S3.value:
        input_data_path = train_paths.get_text_dataset_path(task_id)

        # if not os.path.exists(input_data_path):
        local_path = await download_s3_file(dataset_url)
        shutil.copy(local_path, input_data_path)
        if data_size > 0:
            with open(input_data_path, "r") as f:
                data = json.load(f)
                
            data = data[:data_size]
            print(f"Truncated dataset to {data_size} samples")
            with open(input_data_path, "w") as f:
                json.dump(data, f)

    elif file_format == FileFormat.HF.value:
        repo_name = dataset_url.replace("/", "--")
        input_data_path = os.path.join(dataset_dir, repo_name)

        if not os.path.exists(input_data_path):
            try:
                _snapshot_download_with_retry(
                    repo_id=dataset_url, 
                    repo_type="dataset", 
                    local_dir=input_data_path, 
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"Error downloading dataset from HuggingFace {dataset_url}: {e}", flush=True)
                raise

    return input_data_path, file_format


async def download_image_dataset(dataset_zip_url, task_id, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    local_zip_path = train_paths.get_image_training_zip_save_path(task_id)
    print(f"Downloading dataset from: {dataset_zip_url}")
    local_path = await download_s3_file(dataset_zip_url, local_zip_path)
    print(f"Downloaded dataset to: {local_path}")
    return local_path


@hf_retry
def _list_repo_tree_with_retry(repo_id: str, repo_type: str = "model"):
    """Wrapper for list_repo_tree with retry logic"""
    try:
        return hf_api.list_repo_tree(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        if is_network_error(e):
            print(f"Network error listing repo tree for {repo_id}, will retry: {e}", flush=True)
            raise
        # For non-network errors, don't retry
        raise

def is_safetensors_available(repo_id: str) -> tuple[bool, str | None]:
    try:
        files_metadata = _list_repo_tree_with_retry(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"Error checking safetensors availability for {repo_id}: {e}", flush=True)
        # Return False if we can't check, allowing fallback to regular download
        return False, None
    
    check_size_in_gb = 6
    total_check_size = check_size_in_gb * 1024 * 1024 * 1024
    largest_file = None

    for file in files_metadata:
        if hasattr(file, "size") and file.size is not None:
            if file.path.endswith(".safetensors") and file.size > total_check_size:
                if largest_file is None or file.size > largest_file.size:
                    largest_file = file

    if largest_file:
        return True, largest_file.path
    return False, None


@hf_retry
def _hf_hub_download_with_retry(repo_id: str, filename: str, local_dir: str):
    """Wrapper for hf_hub_download with retry logic"""
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    except Exception as e:
        if is_network_error(e):
            print(f"Network error downloading {filename} from {repo_id}, will retry: {e}", flush=True)
            raise
        raise

@hf_retry
def _snapshot_download_with_retry(repo_id: str, repo_type: str = "model", local_dir: str = None, 
                                  local_dir_use_symlinks: bool = False, cache_dir: str = None,
                                  allow_patterns: list = None):
    """Wrapper for snapshot_download with retry logic"""
    kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "local_dir_use_symlinks": local_dir_use_symlinks,
    }
    if local_dir:
        kwargs["local_dir"] = local_dir
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    
    try:
        return snapshot_download(**kwargs)
    except Exception as e:
        if is_network_error(e):
            print(f"Network error downloading snapshot {repo_id}, will retry: {e}", flush=True)
            raise
        raise

def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    try:
        local_dir = os.path.expanduser(local_dir)
        local_filename = f"{repo_id.replace('/', '_')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(final_path):
            print(f"File {filename} already exists. Skipping download.", flush=True)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = _hf_hub_download_with_retry(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            print(f"File {filename} downloaded successfully", flush=True)
        return final_path
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {e}", flush=True)
        raise


def download_flux_unet(repo_id: str, output_dir: str) -> str:
    try:
        files_metadata = _list_repo_tree_with_retry(repo_id=repo_id, repo_type="model")
    except Exception as e:
        print(f"Error listing files for flux unet repo {repo_id}: {e}", flush=True)
        raise FileNotFoundError(f"Failed to list files in repo '{repo_id}': {e}")
    
    file_path = None
    for file in files_metadata:
        if hasattr(file, "size") and file.size is not None:
            if file.path.endswith(".safetensors") and file.size > 10 * 1024 * 1024 * 1024:
                file_path = file.path
                local_path = download_from_huggingface(repo_id, file_path, output_dir)
                break
    
    if not file_path:
        raise FileNotFoundError(f"No valid file found in root of repo '{repo_id}'.")

    return local_path


async def download_base_model(repo_id: str, save_root: str) -> str:
    model_name = repo_id.replace("/", "--")
    save_path = os.path.join(save_root, model_name)
    if os.path.exists(save_path):
        print(f"Model {repo_id} already exists at {save_path}. Skipping download.", flush=True)
        return save_path
    else:
        has_safetensors, safetensors_path = is_safetensors_available(repo_id)
        if has_safetensors and safetensors_path:
            return download_from_huggingface(repo_id, safetensors_path, save_path)
        else:
            try:
                _snapshot_download_with_retry(
                    repo_id=repo_id, 
                    repo_type="model", 
                    local_dir=save_path, 
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                print(f"Error downloading model {repo_id}: {e}", flush=True)
                raise
            return save_path


async def download_axolotl_base_model(repo_id: str, save_dir: str) -> str:
    model_dir = os.path.join(save_dir, repo_id.replace("/", "--"))
    if os.path.exists(model_dir):
        print(f"Model {repo_id} already exists at {model_dir}. Skipping download.", flush=True)
        return model_dir
    try:
        _snapshot_download_with_retry(
            repo_id=repo_id, 
            repo_type="model", 
            local_dir=model_dir, 
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"Error downloading axolotl model {repo_id}: {e}", flush=True)
        raise
    return model_dir


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task-type",
        required=True,
        choices=[TaskType.IMAGETASK.value, TaskType.INSTRUCTTEXTTASK.value, TaskType.DPOTASK.value, TaskType.GRPOTASK.value],
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--file-format")
    parser.add_argument("--num-samples", type=int, default=-1)
    args = parser.parse_args()

    dataset_dir = cst.CACHE_DATASETS_DIR
    model_dir = cst.CACHE_MODELS_DIR
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading datasets to: {dataset_dir}", flush=True)
    print(f"Downloading models to: {model_dir}", flush=True)

    if args.task_type == TaskType.IMAGETASK.value:
        dataset_zip_path = await download_image_dataset(args.dataset, args.task_id, dataset_dir)
        model_path = await download_base_model(args.model, model_dir)
        print("Downloading clip models", flush=True)
        try:
            CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cst.HUGGINGFACE_CACHE_PATH)
            CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir=cst.HUGGINGFACE_CACHE_PATH)
        except Exception as e:
            print(f"Error downloading CLIP tokenizers: {e}", flush=True)
            raise
        
        try:
            _snapshot_download_with_retry(
                repo_id="google/t5-v1_1-xxl",
                repo_type="model",
                cache_dir="/cache/hf_cache/",
                local_dir_use_symlinks=False,
                allow_patterns=["tokenizer_config.json", "spiece.model", "special_tokens_map.json", "config.json"],
            )
        except Exception as e:
            print(f"Error downloading T5 model: {e}", flush=True)
            raise
    else:
        dataset_path, _ = await download_text_dataset(args.task_id, args.dataset, args.file_format, dataset_dir, args.num_samples)
        model_path = await download_axolotl_base_model(args.model, model_dir)

    print(f"Model path: {model_path}", flush=True)
    print(f"Dataset path: {dataset_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
