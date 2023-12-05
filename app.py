from flask import Flask, jsonify
from flask import request
from joblib import load
from numpy import int64

from constants import Agencies


# load model
model_LogisticRegression = load(
    "model/LogisticRegression/model_LogisticRegression.joblib"
)
model_OneClassSVM = load("model/OneClassSVM/model_OneClassSVM.joblib")


def get_prediction(model, params, isCVM=False):
    x = [params]
    pred = model.predict(x)
    y = pred[0]
    prob = None if isCVM else model.predict_proba(x)[0].tolist()

    return {"prediction": int(y), "probability": prob}


# Initiate API
app = Flask(__name__)


# Define Endpoints


@app.route("/")
def home():
    return {"Helo": "World"}


@app.route("/predict-logistic", methods=["POST"])
def predictLogistic():
  params = request.get_json()
  print("Params", params)
  values = [
      params["maneraMorir"],
      params["tipoAmenaza"],
      params["estadoHuida"],
      params["armadoCon"],
      params["ubicacionExacta"],
      params["edad"],
      params["genero"],
      params["raza"],
      params["enfermedadMental"],
      params["camaraCorporal"],
      params["idAgencias"],
  ]
  pred = get_prediction(model_LogisticRegression, values)
  return pred


@app.route("/predict-svm", methods=["POST"])
def predictSVM():
  params = request.get_json()
  values = [
      params["maneraMorir"],
      params["tipoAmenaza"],
      params["estadoHuida"],
      params["armadoCon"],
      params["ubicacionExacta"],
      params["edad"],
      params["genero"],
      params["raza"],
      params["enfermedadMental"],
      params["camaraCorporal"],
      params["idAgencias"],
  ]
  pred = get_prediction(model_OneClassSVM, values, True)
  return pred


@app.route("/agencies", methods=["GET"])
def getAgencies():
  return jsonify(Agencies.LIST)
