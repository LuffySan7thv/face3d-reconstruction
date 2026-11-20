import numpy as np
import os
import pandas as pd

folder = "results/face_transforms"

# Load the frontal transformation matrix
T_front = np.loadtxt(
    os.path.join(folder, "front_transform.csv"),
    delimiter=","
)

T_front_inv = np.linalg.inv(T_front)

results = []

for filename in sorted(os.listdir(folder)):

    if not filename.endswith("_transform.csv"):
        continue

    T_current = np.loadtxt(
        os.path.join(folder, filename),
        delimiter=","
    )

    # Relative transformation
    T_relative = T_current @ T_front_inv

    R = T_relative[:3, :3]
    t = T_relative[:3, 3]

    # Rotation angle from the trace
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)

    angle = np.degrees(np.arccos(cos_angle))

    print("\nImage:", filename)
    print("Relative transformation:")
    print(T_relative)

    print("Relative rotation angle:", angle)
    print("Relative translation:", t)

    results.append([
        filename,
        angle,
        t[0],
        t[1],
        t[2]
    ])

# Save results
df = pd.DataFrame(
    results,
    columns=[
        "image",
        "rotation_angle",
        "tx",
        "ty",
        "tz"
    ]
)

df.to_csv(
    "results/relative_transforms.csv",
    index=False
)

print("\nRelative transformations saved.")

