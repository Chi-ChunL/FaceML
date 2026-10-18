import csv
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import os
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "expw_prepared"
FER_DIRECTORY = PROJECT_ROOT / "data" / "fer2013"

INPUT_PATH = DATA_DIRECTORY / "manifest.csv"
OUTPUT_PATH = DATA_DIRECTORY / "deduplicated_manifest.csv"
FINAL_PATH = DATA_DIRECTORY / "training_manifest.csv"

with INPUT_PATH.open(newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    rows = list(reader)

groups = defaultdict(list)

for row in rows:
    groups[row["pixel_sha256"]].append(row)

kept_rows = []
duplicate_count = 0
conflicting_count = 0

for group in groups.values():
    labels = {row["class_name"] for row in group}

    if len(labels) > 1:
        conflicting_count += len(group)
        continue

    kept_rows.append(group[0])
    duplicate_count += len(group) - 1

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)
class_counts = Counter(
    row["class_name"] for row in kept_rows
)

print("Original images:", len(rows))
print("Removed duplicate copies:", duplicate_count)
print("Excluded rows with conflicting labels:", conflicting_count)
print("Remaining images:", len(kept_rows))

for class_name in sorted(class_counts):
    print(f"{class_name:>8}: {class_counts[class_name]}")

print("Saved filtered manifest:", OUTPUT_PATH)

print("\nListing FER-2013 files...", flush=True)
fer_paths = []
class_names = (
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise",
)

for split in ("train", "test"):
    for class_name in class_names:
        folder = FER_DIRECTORY / split / class_name
        with os.scandir(folder) as entries:
            fer_paths.extend(
                Path(entry.path)
                for entry in entries
                if entry.name.lower().endswith((".jpg", ".jpeg", ".png"))
                and entry.is_file(follow_symlinks=False)
            )
        print(f"Listed {split}/{class_name}", flush=True)

fer_paths.sort()

if not fer_paths:
    raise ValueError(f"No FER-2013 images found in {FER_DIRECTORY}")

fer_hashes = set()

print("\nChecking FER-2013 images for overlap...")

for index, image_path in enumerate(fer_paths, start=1):
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        grayscale = grayscale.resize(
            (48, 48),
            Image.Resampling.BILINEAR,
        )

        pixel_hash = hashlib.sha256(
            grayscale.tobytes()
        ).hexdigest()

    fer_hashes.add(pixel_hash)

    if index % 1000 == 0:
        print(f"Checked {index}/{len(fer_paths)} FER images", flush=True)

training_rows = [
    row for row in kept_rows
    if row["pixel_sha256"] not in fer_hashes
]

overlap_count = len(kept_rows) - len(training_rows)

with FINAL_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(training_rows)

print("\nFER images checked:", len(fer_paths))
print("ExpW crops excluded for exact FER overlap:", overlap_count)
print("ExpW crops remaining:", len(training_rows))

final_counts = Counter(
    row["class_name"] for row in training_rows
)

for class_name in sorted(final_counts):
    print(f"{class_name:>8}: {final_counts[class_name]}")

print("Training manifest saved:", FINAL_PATH)