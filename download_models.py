from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vilm/vinallama-7b-chat-GGUF",
    allow_patterns=["*.gguf"],
    local_dir="model",
)

print("Model downloaded to ./model")