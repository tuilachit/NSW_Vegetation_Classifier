"""Download model weight assets during cloud deployment.

The weight files are too large for normal Git storage, so Render should receive
download URLs as environment variables:
  BINARY_WEIGHTS_URL
  GROUP_WEIGHTS_URL
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "model_assets"
ASSETS = [
    ("BINARY_WEIGHTS_URL", ASSET_DIR / "binary_veg_other_p2_best.weights.h5"),
    ("GROUP_WEIGHTS_URL", ASSET_DIR / "group9_after_binary_p2_best.weights.h5"),
]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {path.name}...", flush=True)
    with urllib.request.urlopen(url, timeout=180) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            seen += len(chunk)
            if total:
                print(f"  {seen / total:.1%}", end="\r", flush=True)
    tmp.replace(path)
    print(f"\nSaved {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


def main() -> int:
    missing_urls = []
    for env_name, path in ASSETS:
        if path.exists() and path.stat().st_size > 0:
            print(f"Found existing {path.name}", flush=True)
            continue
        url = os.environ.get(env_name)
        if not url:
            missing_urls.append(env_name)
            continue
        download(url, path)

    if missing_urls:
        print(
            "Missing model asset URL environment variables: "
            + ", ".join(missing_urls)
            + ". Set these on Render to deploy real inference.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

