"""Download sample data for PTC-Depth demo."""

import urllib.request
import tarfile
from pathlib import Path

RELEASE_BASE = "https://github.com/ajou-arrl/ptc-depth/releases/download/sample-data"

SAMPLES = {
    "wheel_roadside_rgb": f"{RELEASE_BASE}/wheel_roadside_rgb.tar.gz",
    "wheel_roadside_thr": f"{RELEASE_BASE}/wheel_roadside_thr.tar.gz",
    "wheel_forest_rgb": f"{RELEASE_BASE}/wheel_forest_rgb.tar.gz",
}


def download_one(name, url, data_dir):
    sample_dir = data_dir / name

    if sample_dir.exists() and any(sample_dir.iterdir()):
        print(f"  Already exists: {sample_dir}")
        return

    archive = data_dir / f"{name}.tar.gz"

    print(f"  Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, str(archive),
                                  reporthook=lambda b, bs, ts: print(f"\r    {b*bs/1e6:.0f}/{ts/1e6:.0f} MB", end=""))
        print()
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print(f"  Download manually: {url}")
        print(f"  Extract to: {data_dir}/")
        return

    print(f"  Extracting...")
    with tarfile.open(str(archive), 'r:gz') as tar:
        tar.extractall(str(data_dir))

    archive.unlink()
    print(f"  Done: {sample_dir}")


def main():
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading PTC-Depth sample data...")
    for name, url in SAMPLES.items():
        download_one(name, url, data_dir)

    print("\nAll done.")


if __name__ == '__main__':
    main()
