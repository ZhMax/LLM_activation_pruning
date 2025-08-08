from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import os

from huggingface_hub import hf_hub_download, snapshot_download
with open("./hf_token.txt", "r") as f:
    hf_token = f.read()
os.environ["HF_TOKEN"] = hf_token
model_path = Path('/home/LLM/weights/Llama-3.2-1B-Instruct')
model_path.mkdir(parents=True, exist_ok=True)


snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    local_dir=model_path,
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md"
    ],
    repo_type="model"
)