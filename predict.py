import cv2
import joblib
import matplotlib.pyplot as plt

from utils import preprocess_image

image_path = "dataset/test/680.jpg"
pipeline = joblib.load(
    "models/cats_dogs_pipeline.pkl"
)

feature = preprocess_image(image_path)
if feature is None:
    print("Image could not be loaded.")
    exit()

feature = feature.reshape(1, -1)

prediction = pipeline.predict(feature)

if prediction[0] == 0:
    label = "Cat 🐱"
else:
    label = "Dog 🐶"

print("Prediction:", label)

image = cv2.imread(image_path)
if image is None:
    print("Unable to open image.")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

plt.title(label)

plt.axis("off")

plt.show()