import cv2
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread("images/face.png")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw detected keypoints
result = cv2.drawKeypoints(
    gray,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display result
plt.figure(figsize=(10, 8))
plt.imshow(result, cmap="gray")
plt.title(f"SIFT Keypoints: {len(keypoints)}")
plt.axis("off")
plt.show()

print("Number of keypoints:", len(keypoints))