# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
import os
import pickle
import joblib

app = Flask(__name__)

# load X_test
name = "X_test_32.pickle"
X_test = pd.read_pickle(name)

# Loading model to compare the results
name = "xgbClassifier_go.sav"
with open(name, 'rb') as filename:
    model = joblib.load(filename)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/api", methods=["GET"])
def predict():
    # Get the data from the POST request.
    # data = request.get_json(force=True)
    print('hello')
    data = request.args
    print(data)
    id = data["customer_ID"]
    customer = X_test.iloc[[id]]

    # Make prediction using model loaded from disk as per the data.
    solvabilite = int(model.predict(customer).item())  # 0 or 1
    proba_array = model.predict_proba(customer)  # percentage %
    index = np.argmax(proba_array)
    proba = round(np.asscalar(proba_array[:, index]), 2)
    dict_output = {"customer_ID": id, "solvabilite": solvabilite, "probabilite": proba}

    return jsonify(dict_output)


if __name__ == "__main__":
    app.run(port=5000, debug=True)