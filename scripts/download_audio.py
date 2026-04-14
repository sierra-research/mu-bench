import os

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "sierra-research/mu-bench"
token = os.environ.get("HF_TOKEN")

api = HfApi()
files = [f for f in api.list_repo_files(REPO_ID, repo_type="dataset", token=token) if f.endswith(".wav")]

print(f"Found {len(files)} audio files in {REPO_ID}")

for i, filepath in enumerate(files):
    out_path = os.path.join("audio", filepath)
    if os.path.exists(out_path):
        continue

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filepath,
        repo_type="dataset",
        local_dir="audio",
        token=token,
    )

    if (i + 1) % 100 == 0 or (i + 1) == len(files):
        print(f"  {i + 1}/{len(files)} downloaded")

print(f"Exported {len(files)} audio files to audio/")
