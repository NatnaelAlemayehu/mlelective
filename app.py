import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("finalized_model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [int(x) for x in request.form.values()]
    print(float_features)
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The overall score is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
