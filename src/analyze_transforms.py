import numpy as np
import os

folder = "results/face_transforms"

for filename in sorted(os.listdir(folder)):

    if not filename.endswith("_transform.csv"):
        continue

    matrix = np.loadtxt(
        os.path.join(folder, filename),
        delimiter=","
    )

    R = matrix[:3, :3]
    t = matrix[:3, 3]

    print("\nImage:", filename)
    print("Matrix shape:", matrix.shape)

    print("\nRotation block:")
    print(R)

    print("\nTranslation:")
    print(t)

    print("\nDeterminant:", np.linalg.det(R))

    print(
        "Orthogonality error:",
        np.linalg.norm(R.T @ R - np.eye(3))
    )

