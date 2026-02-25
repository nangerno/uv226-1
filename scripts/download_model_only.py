import os
from trainer_downloader import is_safetensors_available, download_from_huggingface, _snapshot_download_with_retry
import train_cst as cst
import typer
os.environ["HF_HOME"] = "/workspace/hf_cached/"
os.environ["TMPDIR"] = "/workspace/tmp"


def download_base_model(repo_id: str, save_root: str) -> str:
    model_name = repo_id.replace("/", "--")
    save_path = os.path.join(save_root, model_name)
    print(f"Downloading model {repo_id} to {save_path}", flush=True)
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
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                print(f"Error downloading model {repo_id}: {e}", flush=True)
                raise
            return save_path


def main(repo_id: str):
    model_dir = cst.CACHE_MODELS_DIR
    os.makedirs(model_dir, exist_ok=True)
    download_base_model(repo_id, model_dir)


if __name__ == "__main__":
    typer.run(main)
