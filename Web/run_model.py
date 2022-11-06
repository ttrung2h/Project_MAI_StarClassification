# import pickle
import pickle
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="templates/static")
@app.route("/", methods=["GET", "POST"])
@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/predict.html", methods=["GET", "POST"])
def predict():
    prediction = 0
    if request.method == "POST":
        model = pickle.load(open("star_predict.pkl", "rb"))
        user_input_Temperature = request.form.get("Temperature")
        user_input_L = request.form.get("L")
        user_input_R = request.form.get("R")
        user_input_A_M = request.form.get("A_M")
        user_input_Color = request.form.get("Color")
        user_input_Spectral_Class = request.form.get("Spectral_Class")
        print(type(user_input_Temperature))
        print(request.form)
        user_input_Temperature = int(user_input_Temperature)
        user_input_L = float(user_input_L)
        user_input_R = float(user_input_R)
        user_input_A_M = float(user_input_A_M)
        user_input_Color = str(user_input_Color)
        user_input_Spectral_Class = str(user_input_Spectral_Class)

        print(user_input_Temperature)
        print(user_input_L)
        print(user_input_R)
        print(user_input_A_M)
        print(user_input_Color)
        print(user_input_Spectral_Class)
        X_predict = [
            user_input_Temperature,
            user_input_L,
            user_input_R,
            user_input_A_M,
            user_input_Color,
            user_input_Spectral_Class,
        ]
        data_for_prediction = pd.DataFrame(
            [X_predict],
            columns=["Temperature", "L", "R", "A_M", "Color", "Spectral_Class"],
        )
        prediction = model.predict(data_for_prediction)
        print(prediction)
    return render_template("predict.html", prediction=prediction)


@app.route("/document.html")
def document():
    return render_template("document.html")


if __name__ == "__main__":
    app.run(debug=True)
