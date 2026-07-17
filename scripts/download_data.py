"""Download sample training data for the custom LLM."""

from pathlib import Path
import json
import urllib.request
import sys


DATASETS = {
    "tiny_shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "path": "data/tiny_shakespeare/input.txt",
        "description": "Shakespeare plays (~1MB)",
    },
    "tiny_stories": {
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "path": "data/tiny_stories/train.txt",
        "description": "Simple children's stories (custom model training)",
    },
}


def download_dataset(name: str) -> Path:
    if name not in DATASETS:
        print(f"Available datasets: {', '.join(DATASETS)}")
        sys.exit(1)
    info = DATASETS[name]
    dest = Path(info["path"])
    if dest.exists():
        print(f"{name} already exists at {dest} ({dest.stat().st_size / 1024:.0f} KB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = info["url"]
    print(f"Downloading {name} from {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


def list_datasets() -> None:
    print("Available datasets:")
    for name, info in DATASETS.items():
        dest = Path(info["path"])
        status = f"({dest.stat().st_size / 1024:.0f} KB)" if dest.exists() else "(not downloaded)"
        print(f"  {name}: {info['description']} {status}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        list_datasets()
    elif sys.argv[1] == "all":
        for name in DATASETS:
            download_dataset(name)
    else:
        for name in sys.argv[1:]:
            download_dataset(name)
