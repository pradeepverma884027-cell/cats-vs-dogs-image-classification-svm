import cv2
import matplotlib.pyplot as plt

image = cv2.imread("dataset/train/cat.0.jpg")

image = cv2.resize(image, (64, 64))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap="gray")

plt.title("Grayscale Image")

plt.show()

feature = gray.flatten()

print("Image Shape",image.shape)
print("Grayscale Shape",gray.shape)

print("Feature Vector Shape",feature.shape)

print("First 20 values")

print(feature[:20])