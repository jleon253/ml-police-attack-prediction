from typing import Union
from fastapi import FastAPI
from joblib import load
from numpy import int64
from pydantic import BaseModel

# load model
model_LogisticRegression = load('/model/LogisticRegression/model_LogisticRegression.joblib')
model_OneClassSVM = load('/model/OneClassSVM/model_OneClassSVM.joblib')

def get_prediction(model, params, isCVM=False):
  x = [params]
  pred = model.predict(x)
  y = pred[0]
  prob = None if isCVM else model.predict_proba(x)[0].tolist()

  return {
    'prediction': int(y),
    'probability': prob
  }

# Initiate API
app = FastAPI()

# Define model for post request.
class ModelParams(BaseModel):
  maneraMorir: float
  tipoAmenaza: float
  estadoHuida: float
  armadoCon: float
  ubicacionExacta: float
  edad: float
  genero: float
  raza: float
  enfermedadMental: float
  camaraCorporal: float
  idAgencias: float

# Define Endpoints

@app.get('/')
def test():
  return { 'Helo': 'World' }

@app.post('/predict-logistic')
def predictLogistic(params: ModelParams):
  values = [
    params.maneraMorir,
    params.tipoAmenaza,
    params.estadoHuida,
    params.armadoCon,
    params.ubicacionExacta,
    params.edad,
    params.genero,
    params.raza,
    params.enfermedadMental,
    params.camaraCorporal,
    params.idAgencias,
  ];
  pred = get_prediction(model_LogisticRegression, values)
  return pred

@app.post('/predict-svm')
def predictSVM(params: ModelParams):
  values = [
    params.maneraMorir,
    params.tipoAmenaza,
    params.estadoHuida,
    params.armadoCon,
    params.ubicacionExacta,
    params.edad,
    params.genero,
    params.raza,
    params.enfermedadMental,
    params.camaraCorporal,
    params.idAgencias,
  ];
  pred = get_prediction(model_OneClassSVM, values, True)
  return pred