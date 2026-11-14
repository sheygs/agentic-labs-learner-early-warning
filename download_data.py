"""Download the public UCI OULAD archive and extract the tables used here."""

import argparse
import hashlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile

DATA_DIR = Path("data/oulad")
DATA_URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
SHA256 = "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"
FILES = ("courses.csv", "studentInfo.csv", "studentRegistration.csv", "studentVle.csv")


def ensure_data(data_dir=DATA_DIR, archive=None):
    data_dir = Path(data_dir)
    if all((data_dir / name).is_file() for name in FILES):
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=data_dir) as temp_dir:
        staging = Path(temp_dir)
        if archive is None:
            archive = staging / "oulad.zip"
            request = Request(DATA_URL, headers={"User-Agent": "oulad-midterm/1.0"})
            with urlopen(request, timeout=60) as response, archive.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)

        with Path(archive).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != SHA256:
                raise ValueError(
                    "OULAD archive checksum mismatch. Use the original UCI ZIP."
                )

        with ZipFile(archive) as zip_file:
            for name in FILES:
                with (
                    zip_file.open(name) as source,
                    (staging / name).open("wb") as target,
                ):
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        for name in FILES:
            (staging / name).replace(data_dir / name)
    return data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Path to the original UCI ZIP, if already downloaded.",
    )
    args = parser.parse_args()
    print(f"OULAD ready in {ensure_data(args.data_dir, args.archive)}")
