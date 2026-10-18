from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "expw_raw" / "data"
IMAGE_DIRECTORY = SOURCE_DIRECTORY / "image" / "origin"
LABEL_PATH = SOURCE_DIRECTORY / "label" / "label.lst"

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