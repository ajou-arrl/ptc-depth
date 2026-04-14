"""Download sample data for PTC-Depth demo."""

import urllib.request
import tarfile
import sys
from pathlib import Path

SAMPLE_URL = "https://github.com/ajou-arrl/ptc-depth/releases/download/sample-data/wheel_roadside_sample.tar.gz"


def main():
    data_dir = Path.cwd() / "data"
    sample_dir = data_dir / "wheel_roadside_sample"

    if sample_dir.exists() and any(sample_dir.iterdir()):
        print(f"Data already exists: {sample_dir}")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "wheel_roadside_sample.tar.gz"

    print(f"Downloading sample data (~618MB)...")
    try:
        urllib.request.urlretrieve(SAMPLE_URL, str(archive),
                                  reporthook=lambda b, bs, ts: print(f"\r  {b*bs/1e6:.0f}/{ts/1e6:.0f} MB", end=""))
        print()
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"Download manually: {SAMPLE_URL}")
        print(f"Extract to: {data_dir}/")
        sys.exit(1)

    print("Extracting...")
    with tarfile.open(str(archive), 'r:gz') as tar:
        tar.extractall(str(data_dir))

    archive.unlink()
    print(f"Done. Data: {sample_dir}")


if __name__ == '__main__':
    main()
