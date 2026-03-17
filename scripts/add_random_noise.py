import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import typer
from transformers import GenerationConfig
import json

# Paths with multiple slashes (e.g. /cache/models/...) fail HF repo_id validation; load from disk instead.
def _is_local_fs_path(path: str) -> bool:
    return path.startswith("/") or (len(path) > 1 and path[1] == ":")


def _load_model_from_local_path(model_path: str):
    """Load model and tokenizer from a local directory without going through HF hub (avoids repo_id validation)."""
    import glob
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = AutoConfig.from_dict(config_dict)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    # Prefer safetensors, then pytorch_model.bin
    st_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    pt_file = os.path.join(model_path, "pytorch_model.bin")
    if st_files:
        from safetensors.torch import load_file
        state = {}
        for f in st_files:
            state.update(load_file(f))
        model.load_state_dict(state, strict=False)
    elif os.path.isfile(pt_file):
        model.load_state_dict(torch.load(pt_file, map_location="cpu", weights_only=True), strict=False)
    else:
        raise FileNotFoundError(f"No model weights in {model_path}")
    model.generation_config = GenerationConfig(temperature=None, top_p=None)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer


def _local_path_to_hub_name(path: str) -> str:
    """Convert a cache path like /cache/models/Qwen--Qwen1.5-0.5B to hub id Qwen/Qwen1.5-0.5B."""
    base = path.replace("\\", "/").rstrip("/")
    if "/" in base:
        base = base.split("/")[-1]
    return base.replace("--", "/", 1) if "--" in base else base


def main(model_path: str, save_folder: str):
    # Local paths with multiple slashes fail HF repo_id validation; load from disk when path exists, else hub.
    if _is_local_fs_path(model_path) and os.path.isdir(model_path):
        model, tokenizer = _load_model_from_local_path(model_path)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            model = model.to(torch.bfloat16)
    else:
        load_id = model_path
        if _is_local_fs_path(model_path) and not os.path.isdir(model_path):
            load_id = _local_path_to_hub_name(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            load_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.generation_config = GenerationConfig(temperature=None, top_p=None)
        tokenizer = AutoTokenizer.from_pretrained(load_id)
    noise_std = 0.01

    # Step 2: Add random noise to the input embeddings
    print("Modifying input embeddings...", flush=True)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        noise = torch.randn_like(embeddings.weight) * noise_std
        embeddings.weight.add_(noise)

    # Step 3: Save the modified model and tokenizer
    print(f"Saving modified model to {save_folder}...", flush=True)
    os.makedirs(save_folder, exist_ok=True)
    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)

if __name__ == "__main__":
    typer.run(main)
