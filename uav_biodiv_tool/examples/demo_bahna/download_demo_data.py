
from pathlib import Path
import zipfile
import gdown

URL = "https://drive.google.com/file/d/1txesu4GLd4LYoANElhYUbv6faCXityh-/view?usp=drive_link"

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ZIP = DATA / "Bahna_demo.zip"

DATA.mkdir(parents=True, exist_ok=True)

if not ZIP.exists():
    print("Downloading Bahna_demo.zip ...")
    gdown.download(url=URL, output=str(ZIP), quiet=False)

if not zipfile.is_zipfile(ZIP):
    raise RuntimeError(f"Invalid ZIP: {ZIP}")

print("Extracting demo data ...")
with zipfile.ZipFile(ZIP) as z:
    z.extractall(DATA)

print(f"Done: {DATA}")
