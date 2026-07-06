import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from config import IMAGE_SIZE
import cv2
from skimage.feature import hog

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features



def load_dataset(dataset_path):
    """
    Loads all images from the dataset and returns feature matrix X and labels y.
    """
    X = []
    y = []

    files = os.listdir(dataset_path)

    for file in tqdm(files):

        path = os.path.join(dataset_path, file)

        feature = preprocess_image(path)

        if feature is None:
            continue

        X.append(feature)

        if file.startswith("cat"):
            y.append(0)
        else:
            y.append(1)

    return np.array(X), np.array(y)

feature = preprocess_image("dataset/train/cat.0.jpg")

print(feature.shape)