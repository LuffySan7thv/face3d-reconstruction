import cv2
import mediapipe as mp
import numpy as np
import os

input_dir = "images/multiview"
output_dir = "results/face_transforms"

os.makedirs(output_dir, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    output_facial_transformation_matrixes=True
)

with FaceLandmarker.create_from_options(options) as landmarker:

    for filename in os.listdir(input_dir):

        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        path = os.path.join(input_dir, filename)

        image = cv2.imread(path)

        if image is None:
            print("Cannot read:", filename)
            continue

        image_rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = landmarker.detect(mp_image)

        if not result.facial_transformation_matrixes:
            print("No face transform:", filename)
            continue

        matrix = np.array(
            result.facial_transformation_matrixes[0]
        )

        output_path = os.path.join(
            output_dir,
            filename.rsplit(".", 1)[0] + "_transform.csv"
        )

        np.savetxt(
            output_path,
            matrix,
            delimiter=","
        )

        print("Saved:", output_path)
        print("Transformation matrix:\n", matrix)

