import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import typer 
from transformers import GenerationConfig
from huggingface_hub import HfApi
from transformers import AutoConfig
import json 
import datetime
# Configuration

def main(model_path: str, save_folder: str, noise_std: float = 0.01):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig(temperature=None, top_p=None)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Step 2: Add random noise to the input embeddings
    print(f"Modifying input embeddings with noise_std={noise_std}...", flush=True)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        noise = torch.randn_like(embeddings.weight) * noise_std
        embeddings.weight.add_(noise)
        # Log the magnitude of change for verification
        noise_magnitude = torch.norm(noise).item()
        weight_magnitude = torch.norm(embeddings.weight).item()
        relative_change = (noise_magnitude / weight_magnitude) * 100 if weight_magnitude > 0 else 0
        print(f"Noise magnitude: {noise_magnitude:.6f}, Weight magnitude: {weight_magnitude:.6f}, Relative change: {relative_change:.4f}%", flush=True)

    # Step 3: Save the modified model and tokenizer
    print(f"Saving modified model to {save_folder}...", flush=True)
    os.makedirs(save_folder, exist_ok=True)
    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)

if __name__ == "__main__":
    typer.run(main)
