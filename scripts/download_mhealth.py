from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    archive_path = data_dir / "MHEALTHDATASET.zip"
    if not archive_path.exists():
        print(f"Downloading MHEALTH from {URL}")
        urlretrieve(URL, archive_path)
    else:
        print(f"Using existing archive: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(data_dir)
    print("MHEALTH extracted under data/MHEALTHDATASET")


if __name__ == "__main__":
    main()
