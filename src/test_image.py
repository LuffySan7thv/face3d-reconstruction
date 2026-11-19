import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/face.png")

if image is None:
    print("Image not found!")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title("Input Face")
    plt.axis("off")
    plt.show()
