"""
Helper to download a MediaPipe .task file to the repo root.
Usage:
  python scripts/download_hand_task.py --url <DOWNLOAD_URL>
  python scripts/download_hand_task.py --gdrive-id <DRIVE_FILE_ID>

The script saves to: ../hand_landmarker.task (repo root)

Notes:
- For Google Drive large files, use the gdown package: `pip install gdown`.
- For direct URLs, `requests` is used.
"""
import argparse
import os
import sys

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")


def download_direct(url: str, out_path: str):
    try:
        import requests
    except ImportError:
        print("requests not installed. Install with: pip install requests")
        return 1
    print(f"Downloading from: {url}\n-> {out_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete")
    return 0


def download_gdrive(file_id: str, out_path: str):
    try:
        import gdown
    except ImportError:
        print("gdown not installed. Install with: pip install gdown")
        return 1
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive id={file_id}\n-> {out_path}")
    gdown.download(url, out_path, quiet=False)
    print("Download complete")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", help="Direct download URL for .task file")
    p.add_argument("--gdrive-id", help="Google Drive file id for .task file")
    p.add_argument("--out", help="Output path (defaults to repo root hand_landmarker.task)")
    args = p.parse_args()

    out = args.out if args.out else OUT_PATH

    if args.url:
        code = download_direct(args.url, out)
        sys.exit(code)
    elif args.gdrive_id:
        code = download_gdrive(args.gdrive_id, out)
        sys.exit(code)
    else:
        print("Provide --url or --gdrive-id to download the .task file.")
        sys.exit(2)
