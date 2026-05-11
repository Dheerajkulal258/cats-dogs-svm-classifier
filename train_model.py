import os
import cv2
import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

categories = ["cats", "dogs"]

data = []
labels = []

for category in categories:

    path = os.path.join("dataset", category)

    label = categories.index(category)

    for img in os.listdir(path):

        img_path = os.path.join(path, img)

        try:

            image = cv2.imread(img_path)

            image = cv2.resize(image, (64,64))

            image = image.flatten()

            data.append(image)

            labels.append(label)

        except Exception as e:
            pass

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print(f"Model Accuracy: {accuracy*100:.2f}%")

pickle.dump(model, open("svm_model.pkl", "wb"))

print("Model saved successfully!")