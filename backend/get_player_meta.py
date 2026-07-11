"""
get_player_meta.py

Downloads the 'cricketdata' R package source from CRAN, extracts the
player_meta.rda data file (player bowling/batting styles, sourced from
ESPNCricinfo), and converts it to a plain CSV — all without needing R
installed, using the pyreadr library to read the .rda format directly.

Source: https://cran.r-project.org/package=cricketdata
Maintainer: Rob Hyndman (Monash University)

Run from project root:
    pip install pyreadr requests
    python backend/get_player_meta.py
"""

import os
import tarfile
import requests
import pyreadr
import pandas as pd

CRAN_URL = "https://cran.r-project.org/src/contrib/cricketdata_0.3.0.tar.gz"
DOWNLOAD_PATH = "data/raw_meta/cricketdata_0.3.0.tar.gz"
EXTRACT_DIR = "data/raw_meta/"
OUT_PATH = "data/processed/player_meta.csv"


def download_package():
    os.makedirs(os.path.dirname(DOWNLOAD_PATH), exist_ok=True)
    if os.path.exists(DOWNLOAD_PATH):
        print("Already downloaded, skipping.")
        return
    print(f"Downloading {CRAN_URL} ...")
    resp = requests.get(CRAN_URL, timeout=30)
    resp.raise_for_status()
    with open(DOWNLOAD_PATH, "wb") as f:
        f.write(resp.content)
    print(f"Saved to {DOWNLOAD_PATH}")


def extract_rda():
    print("Extracting tarball...")
    with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
        # only pull out the file we need
        members = [m for m in tar.getmembers() if m.name.endswith("player_meta.rda")]
        if not members:
            raise FileNotFoundError(
                "Could not find player_meta.rda inside the package archive. "
                "The package structure may have changed — check "
                "https://cran.r-project.org/package=cricketdata manually."
            )
        tar.extractall(path=EXTRACT_DIR, members=members)
        return os.path.join(EXTRACT_DIR, members[0].name)


def main():
    download_package()
    rda_path = extract_rda()

    print(f"Reading {rda_path} with pyreadr...")
    result = pyreadr.read_r(rda_path)
    # pyreadr returns a dict of {object_name: dataframe} — there should be one entry
    df = result[list(result.keys())[0]]

    print(f"Loaded player_meta: {df.shape[0]} rows, columns: {list(df.columns)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")

    print("\nSample rows:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()
