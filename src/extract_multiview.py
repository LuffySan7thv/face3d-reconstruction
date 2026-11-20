import cv2
import mediapipe as mp
import numpy as np
import csv
from pathlib import Path

# 1. Input and output directories
input_dir = Path("images/multiview")
output_dir = Path("results/multiview")

output_dir.mkdir(parents=True, exist_ok=True)

# 2. Initialize MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

# 3. Find input images
image_files = sorted(
    list(input_dir.glob("*.png")) +
    list(input_dir.glob("*.jpg")) +
    list(input_dir.glob("*.jpeg"))
)

if not image_files:
    raise FileNotFoundError("No images found in images/multiview")

# 4. Process all views
with FaceLandmarker.create_from_options(options) as landmarker:

    for image_path in image_files:

        image = cv2.imread(str(image_path))

        if image is None:
            print("Cannot read:", image_path.name)
            continue

        h, w, _ = image.shape

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            print("No face detected:", image_path.name)
            continue

        landmarks = result.face_landmarks[0]

        # 5. Save numerical coordinates
        output_file = output_dir / (
            image_path.stem + "_landmarks.csv"
        )

        with open(
            output_file,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "id",
                "x_pixel",
                "y_pixel",
                "z_relative"
            ])

            for index, landmark in enumerate(landmarks):

                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z

                writer.writerow([
                    index,
                    x,
                    y,
                    z
                ])

        print(
            image_path.name,
            "->",
            len(landmarks),
            "landmarks saved"
        )

print("Multi-view extraction completed!")

