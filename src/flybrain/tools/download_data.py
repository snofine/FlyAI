from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request


BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights.feather": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    response = urllib.request.urlopen(request, timeout=60)
    # Some proxies ignore Range. Only append when the server confirms 206.
    mode = "ab" if existing and response.status == 206 else "wb"
    with response, partial.open(mode) as output:
        total = response.headers.get("Content-Length", "?")
        print(f"Downloading {target.name}: {total} bytes remaining", flush=True)
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            print(f"  {partial.stat().st_size / 1024**2:,.1f} MiB", end="\r", flush=True)
    partial.replace(target)
    print(f"\nSaved {target} (sha256 {sha256(target)[:16]}…)", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the three MaleCNS v1.0 MVP files")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--skip-weights", action="store_true", help="Download only the two small metadata files")
    args = parser.parse_args(argv)
    for local_name, remote_name in FILES.items():
        if args.skip_weights and local_name == "connectome-weights.feather":
            continue
        target = args.output / local_name
        if target.exists():
            print(f"Already present: {target}")
        else:
            download(f"{BASE_URL}/{remote_name}", target)


if __name__ == "__main__":
    main()
