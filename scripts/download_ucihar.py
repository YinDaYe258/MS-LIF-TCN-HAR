from __future__ import annotations

import argparse
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UCIHAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the UCI-HAR dataset.")
    parser.add_argument("--data_dir", default="data", help="Directory where the dataset should be stored.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target_dir = data_dir / "UCI HAR Dataset"
    if target_dir.exists():
        print(f"UCI-HAR already exists at {target_dir}")
        return

    archive_path = data_dir / "ucihar.zip"
    try:
        if archive_path.exists():
            print(f"Using existing archive {archive_path}")
        else:
            print(f"Downloading {UCIHAR_URL}")
            download(UCIHAR_URL, archive_path)
        print(f"Extracting {archive_path}")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        nested_archive = data_dir / "UCI HAR Dataset.zip"
        if not target_dir.exists() and nested_archive.exists():
            print(f"Extracting nested archive {nested_archive}")
            with zipfile.ZipFile(nested_archive, "r") as zip_ref:
                zip_ref.extractall(data_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"Expected extracted dataset directory not found: {target_dir}")
        print(f"Done. Dataset directory: {target_dir}")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Automatic download failed: {exc}")
        print("Manual instructions:")
        print("1. Open https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones")
        print("2. Download the dataset zip file.")
        print(f"3. Extract it so this directory exists: {target_dir}")
        raise SystemExit(1) from exc


def download(url: str, archive_path: Path) -> None:
    try:
        urllib.request.urlretrieve(url, archive_path)
    except ssl.SSLCertVerificationError:
        print("TLS certificate verification failed; retrying the fixed UCI archive URL without certificate verification.")
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, archive_path.open("wb") as handle:
            handle.write(response.read())
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            print("TLS certificate verification failed; retrying the fixed UCI archive URL without certificate verification.")
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as response, archive_path.open("wb") as handle:
                handle.write(response.read())
        else:
            raise


if __name__ == "__main__":
    main()
