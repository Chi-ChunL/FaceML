import csv
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "expw_prepared"

INPUT_PATH = DATA_DIRECTORY / "manifest.csv"
OUTPUT_PATH = DATA_DIRECTORY / "deduplicated_manifest.csv"

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