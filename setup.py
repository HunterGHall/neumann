import shutil
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)

dir_path = Path("engine/models")
dir_path.mkdir(parents=True, exist_ok=True)

model_path = dir_path / MODEL_URL.rsplit("/", 1)[-1]


def download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / total:6.1%}  {done >> 20}/{total >> 20} MB", end="", flush=True)
    print()
    shutil.move(tmp, dest)


if model_path.exists():
    print(f"Model already present at {model_path}")
else:
    print(f"Downloading {MODEL_URL}")
    download(MODEL_URL, model_path)
    print(f"Saved to {model_path}")
