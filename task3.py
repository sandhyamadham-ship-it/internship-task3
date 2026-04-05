import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# Path to dataset (change this to your folder path)
DATASET_PATH = "dataset/train"  # folder with cat & dog images

IMG_SIZE = 64  # resize images

data = []
labels = []

# Load and preprocess images
for file in os.listdir(DATASET_PATH):
    img_path = os.path.join(DATASET_PATH, file)

    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        data.append(img.flatten())  # flatten image

        # Label: cat = 0, dog = 1
        if "cat" in file:
            labels.append(0)
        else:
            labels.append(1)

    except:
        continue

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Shuffle and subset the data (taking first 2000 samples as per screenshot)
X, y = shuffle(X, y, random_state=42)
X = X[:2000]
y = y[:2000]

print("Dataset shape:", X.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel='linear')  # you can also try 'rbf'
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))