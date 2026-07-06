from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from config import TRAIN_SIZE
from config import TEST_SIZE
from config import RANDOM_STATE
from config import CV_FOLDS
from config import DATASET_PATH
import logging

print("Loading Dataset...")
X, y = load_dataset(DATASET_PATH)

print("Dataset Loaded!")

print("Original Label Distribution:", np.bincount(y))

# Randomly select 2000 images while keeping equal class distribution
X, _, y, _ = train_test_split(
    X,
    y,

    train_size=TRAIN_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Subset Label Distribution:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

pipeline = Pipeline(

    [

        ("scaler", StandardScaler()),

        ("svm", SVC())

    ]

)

print("Training SVM...")
logging.basicConfig(
    filename="outputs/training_log.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

logging.info("Dataset Loaded")
logging.info("Training Started")
logging.info("Training Completed")

param_grid = {

    "svm__kernel":["rbf"],

    "svm__C":[0.1,1,10],

    "svm__gamma":[0.1,0.01,0.001]

}
grid = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    cv=CV_FOLDS,

    scoring="accuracy",

    verbose=2,

    n_jobs=-1

)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Best Cross Validation Accuracy:", grid.best_score_)

best_pipeline = grid.best_estimator_

scores = cross_val_score(
    best_pipeline,
    X,
    y,
    cv=CV_FOLDS,
    scoring="accuracy"
)

print("Cross Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())

print("Training Completed!")

print("Predicting...")

y_pred = best_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)

ConfusionMatrixDisplay.from_estimator(
    best_pipeline,
    X_test,
    y_test,
    display_labels=["Cat", "Dog"]
)

plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")
plt.show()
joblib.dump(

    best_pipeline,

    "models/cats_dogs_pipeline.pkl"

)
print("Best Pipeline saved successfully")


print(best_pipeline)

with open("outputs/results.txt", "w") as f:
    f.write(f"Best Parameters: {grid.best_params_}\n")
    f.write(f"Best CV Score: {grid.best_score_:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))


with open("outputs/model_info.txt", "w") as f:

    f.write("Best Parameters\n")
    f.write(str(grid.best_params_))

    f.write("\n\n")

    f.write("Cross Validation Accuracy\n")
    f.write(str(scores.mean()))