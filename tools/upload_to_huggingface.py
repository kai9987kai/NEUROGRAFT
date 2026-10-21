import sys
from huggingface_hub import HfApi

def upload():
    api = HfApi()
    repo_id = "Kai9987kai/NEUROGRAFT"

    ignore_patterns = [
        ".git*",
        ".pytest_cache*",
        "build*",
        "*.pyc",
        "__pycache__*",
        "scratch*",
        "*report.html",
        "node_modules*",
        "*.lock",
    ]

    print(f"Uploading latest models, checkpoints, and v0.12 release to Hugging Face repo: {repo_id}...")
    commit_info = api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
        commit_message="Release NEUROGRAFT v0.12.0: Hippocampal Successor Representation, SWR Replay, and Active Inference models",
    )
    print("Upload successfully finished!")
    print("Commit info:", commit_info)

if __name__ == "__main__":
    upload()
