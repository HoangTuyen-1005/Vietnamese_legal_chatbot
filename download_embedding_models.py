from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="caliex/all-MiniLM-L6-v2-f16.gguf",
    allow_patterns=["*.gguf"],
    local_dir="models",
)

print("Embedding model downloaded to ./models")