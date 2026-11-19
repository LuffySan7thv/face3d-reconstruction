import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# 1. Read image
image = cv2.imread("images/face.png")

if image is None:
    raise FileNotFoundError("Face image not found!")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Define Face Landmarker
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

# 3. Detect landmarks
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image_rgb
)

with FaceLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        h, w, _ = image.shape
        points = []

        for landmark in landmarks:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z

            points.append([x, y, z])

            cv2.circle(
                image,
                (int(x), int(y)),
                1,
                (0, 255, 0),
                -1
            )
        points = np.array(points)

        np.savetxt(
            "results/face_landmarks.csv",
            points,
            delimiter=",",
            header="x,y,z",
            comments=""
        )
        print("Number of landmarks:", len(points))

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Face Landmark Detection")
        plt.axis("off")
        plt.show()

    else:
        print("No face detected!")

