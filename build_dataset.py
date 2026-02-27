import os
import csv
from pose_detect import process_image
from auto_label import auto_label

IMG_DIR = "dataset/images"
OUT = "generated/pose_train.csv"

os.makedirs("generated", exist_ok=True)

with open(OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "elbow_angle",
        "arm_extension",
        "hip_rotation",
        "torso_rotation",
        "target"
    ])

    for img in os.listdir(IMG_DIR):
        path = os.path.join(IMG_DIR, img)
        feats = process_image(path)
        if feats:
            label = auto_label(feats)
            writer.writerow([
                feats["elbow_angle"],
                feats["arm_extension"],
                feats["hip_rotation"],
                feats["torso_rotation"],
                label
            ])

print("Dataset built:", OUT)