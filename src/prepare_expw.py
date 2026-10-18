import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "expw_raw" / "data"
IMAGE_DIRECTORY = SOURCE_DIRECTORY / "image" / "origin"
LABEL_PATH = SOURCE_DIRECTORY / "label" / "label.lst"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "expw_prepared"
MAX_IMAGES_PER_CLASS = 1000
RANDOM_SEED = 42

EXPW_CLASS_NAMES = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral",
]

records = []

with LABEL_PATH.open(encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue
        (
            filename,
            face_id,
            top,
            left,
            right,
            bottom,
            confidence,
            label_id,
        ) = line.split()

        records.append({
            "image_path": IMAGE_DIRECTORY / filename,
            "face_id": int(face_id),
            "class_name": EXPW_CLASS_NAMES[int(label_id)],
            "box":(
                int(left),
                int(top),
                int(right),
                int(bottom),
            ),
        })
class_counts = Counter(record["class_name"] for record in records)
print("labelled faces:", len(records))
for class_name in EXPW_CLASS_NAMES:
    print(f"{class_name:>8}: {class_counts[class_name]}")

print("\nFirst record:", records[0])


# Keep one annotation per source photo and face ID.
unique_faces = {}
for record in records:
    key = (record["image_path"].name, record["face_id"])
    previous = unique_faces.get(key)
    if previous is not None and previous["class_name"] != record["class_name"]:
        raise ValueError(f"Conflicting labels for {key}")
    unique_faces.setdefault(key, record)

selected_records = list(unique_faces.values())
random.Random(RANDOM_SEED).shuffle(selected_records)

print("Repeated annotations removed:", len(records) - len(selected_records))
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
for class_name in EXPW_CLASS_NAMES:
    (OUTPUT_DIRECTORY / class_name).mkdir(exist_ok=True)

saved_counts = Counter()
skipped_counts = Counter()
manifest_path = OUTPUT_DIRECTORY / "manifest.csv"

with manifest_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "source_image", "face_id", "class_name", "prepared_path",
        "crop_left", "crop_top", "crop_side", "pixel_sha256",
    ])

    for record in selected_records:
        class_name = record["class_name"]
        if saved_counts[class_name] >= MAX_IMAGES_PER_CLASS:
            continue

        source_path = record["image_path"]
        left, top, right, bottom = record["box"]
        if right <= left or bottom <= top:
            skipped_counts["invalid_box"] += 1
            continue

        try:
            with Image.open(source_path) as image:
                box_width = right - left
                box_height = bottom - top
                side = min(
                    round(max(box_width, box_height) * 1.15),
                    image.width,
                    image.height,
                )
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                x = max(0, min(round(center_x - side / 2), image.width - side))
                y = max(0, min(round(center_y - side / 2), image.height - side))

                face = image.crop((x, y, x + side, y + side)).convert("RGB")
                face = face.resize((48, 48), Image.Resampling.BILINEAR)
                face = face.convert("L")
        except (OSError, ValueError) as error:
            skipped_counts["unreadable_image"] += 1
            print(f"Skipping {source_path.name}: {error}")
            continue

        filename = f"{source_path.stem}_face{record['face_id']}.png"
        output_path = OUTPUT_DIRECTORY / class_name / filename
        face.save(output_path)
        pixel_hash = hashlib.sha256(face.tobytes()).hexdigest()
        writer.writerow([
            source_path.relative_to(PROJECT_ROOT).as_posix(),
            record["face_id"],
            class_name,
            output_path.relative_to(PROJECT_ROOT).as_posix(),
            x, y, side, pixel_hash,
        ])
        saved_counts[class_name] += 1

        total_saved = sum(saved_counts.values())
        if total_saved % 500 == 0:
            print("Faces saved:", total_saved, flush=True)
        if all(saved_counts[name] >= MAX_IMAGES_PER_CLASS for name in EXPW_CLASS_NAMES):
            break

summary = {
    "random_seed": RANDOM_SEED,
    "max_images_per_class": MAX_IMAGES_PER_CLASS,
    "crop_expansion": 1.15,
    "image_size": [48, 48],
    "color_mode": "grayscale",
    "annotation_count": len(records),
    "unique_face_records": len(selected_records),
    "repeated_annotations_removed": len(records) - len(selected_records),
    "saved_counts": {name: saved_counts[name] for name in EXPW_CLASS_NAMES},
    "skipped_counts": dict(skipped_counts),
    "total_saved": sum(saved_counts.values()),
}
summary_path = OUTPUT_DIRECTORY / "preparation_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print("\nPrepared training faces:")
for class_name in EXPW_CLASS_NAMES:
    print(f"{class_name:>8}: {saved_counts[class_name]}")
print("Skipped invalid or unreadable faces:", sum(skipped_counts.values()))
print("Saved to:", OUTPUT_DIRECTORY)
print("Manifest:", manifest_path)
