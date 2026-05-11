from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open("svm_model.pkl", "rb"))

UPLOAD_FOLDER = "static"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

categories = ["Cat", "Dog"]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    image = cv2.imread(filepath)

    image = cv2.resize(image, (64,64))

    image = image.flatten().reshape(1, -1)

    prediction = model.predict(image)[0]

    result = categories[prediction]

    return render_template(
        "index.html",
        prediction=result,
        image_file=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)