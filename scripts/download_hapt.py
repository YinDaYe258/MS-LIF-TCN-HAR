from __future__ import annotations

import argparse
import shutil
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HAPT_URL = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the HAPT dataset.")
    parser.add_argument("--data_dir", default="data", help="Directory where the dataset should be stored.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target_dir = data_dir / "HAPT Dataset"
    if (target_dir / "RawData" / "labels.txt").exists():
        print(f"HAPT already exists at {target_dir}")
        return

    archive_path = data_dir / "hapt.zip"
    try:
        if archive_path.exists():
            print(f"Using existing archive {archive_path}")
        else:
            print(f"Downloading {HAPT_URL}")
            download(HAPT_URL, archive_path)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive_path} to {target_dir}")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        if not (target_dir / "RawData" / "labels.txt").exists():
            raise FileNotFoundError(f"Expected extracted HAPT RawData not found under: {target_dir}")
        print(f"Done. Dataset directory: {target_dir}")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Automatic download failed: {exc}")
        print("Manual instructions:")
        print("1. Open https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions")
        print("2. Download the dataset zip file.")
        print(f"3. Extract it so this directory exists: {target_dir / 'RawData'}")
        raise SystemExit(1) from exc


def download(url: str, archive_path: Path) -> None:
    try:
        urllib.request.urlretrieve(url, archive_path)
    except ssl.SSLCertVerificationError:
        print("TLS certificate verification failed; retrying without certificate verification.")
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, archive_path.open("wb") as handle:
            handle.write(response.read())
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            print("TLS certificate verification failed; retrying without certificate verification.")
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as response, archive_path.open("wb") as handle:
                handle.write(response.read())
        else:
            raise


if __name__ == "__main__":
    main()
