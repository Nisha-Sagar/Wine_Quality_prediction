from flask import Flask, render_template, request

import pickle

import numpy as np

flask_app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        fixed_acidity = float(request.form["fixed_acidity"])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugar = float(request.form["residual_sugar"])
        chlorides = float(request.form["chlorides"])
        free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
        total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
        density = float(request.form["density"])
        pH = float(request.form["pH"])
        sulphates = float(request.form["sulphates"])
        alcohol = float(request.form["alcohol"])

        input_data = (fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                      chlorides,free_sulfur_dioxide, total_sulfur_dioxide, density, pH,
                      sulphates, alcohol)

        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)
        if (prediction[0] == 1):
            result = "Good quality"
        else:
            result = "Bad quality"

        return render_template("index.html", result = result)


if __name__ == "__main__":
    flask_app.run(debug=True)