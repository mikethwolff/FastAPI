import logging
import pandas as pd
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Response, status, Request
import sys
from fastapi.responses import HTMLResponse
from typing import Any
import os

if __package__ is None or __package__ == '':
    from ml.model import LOG_FILE
    from ml.clean_data import basic_cleaning
    from ml import model
    from ml.train_model import CAT_FEATURES
    from ml.model import MODEL_FILENAME, ENCODER_FILENAME, LB_FILENAME
    from ml.data import process_data
else:
    from .ml.model import LOG_FILE
    from .ml.clean_data import basic_cleaning
    from .ml import model
    from .ml.train_model import CAT_FEATURES
    from .ml.model import MODEL_FILENAME, ENCODER_FILENAME, LB_FILENAME
    from .ml.data import process_data

# Define logging handlers
logFileHandler = logging.FileHandler(LOG_FILE)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logFileHandler,
        consoleHandler
    ]
)

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

API_PROJECT_NAME = "CensusAPI"
INDEX_BODY = (
    "<html>"
    "<body>"
    "<p>CensusAPI</p>"
    "<p>Documentation can be found <a href='/docs'>here</a>.</p>"
    "</body>" 
    "</html>"
)

app = FastAPI(title=API_PROJECT_NAME)
lr_model = load(MODEL_FILENAME)
encoder = load(ENCODER_FILENAME)
lb = load(LB_FILENAME)

class Data(BaseModel):
    workclass: str = None
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class Config:
        schema_extra = {
            "example": {
                'age': 36,
                'workclass': 'Private',
                'fnlgt': 186035,
                'education': '11th',
                'education_num': 7,
                'marital_status': 'Married-civ-spouse',
                'occupation': 'Craft-repair',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'capital_gain': 5178,
                'capital_loss': 0,
                'hours_per_week': 40,
                'native_country': 'United-States'
            }
        }


@app.on_event("startup")
async def startup_event():
    logging.info("[ FastAPI ]")
    logging.info("[ START: Loading model... ]")
    global lr_model, encoder, lb
    lr_model = load(MODEL_FILENAME)
    encoder = load(ENCODER_FILENAME)
    lb = load(LB_FILENAME)
    logging.info("[ SUCCESS: Model loaded ]")


@app.get("/")
def index(request: Request) -> Any:
    status_code=status.HTTP_200_OK,
    body = INDEX_BODY

    return HTMLResponse(content=body)


def predictApi(input_json, model_dir):
    input_df = pd.DataFrame(dict(input_json), index=[0])
    logging.info(f"[ CONTROL: Input data: {input_df} ]")

    #clean data
    cleaned_df, cat_cols, num_cols = basic_cleaning(input_df, "data/census_cleaned.csv", "salary", test=True)

    # model, encoder and lb are loded already
    #model = load("artifacts/model.joblib")
    #encoder = load("artifacts/encoder.joblib")
    #lb = load("artifacts/lb.joblib")

    #process data
    X, _, _, _ = process_data(cleaned_df, cat_cols, training=False, encoder=encoder, lb=lb)

    # Prediction
    preds = model.inference(lr_model, X)
    preds_class = lb.inverse_transform(preds)

    return {preds_class[0]}


@app.post("/predict")
def predict(data: Data):

    logging.info(f"[ CONTROL: Data dictonary: {data.dict().values()} ]")

    # Check data provided
    if 'string' in data.dict().values():
        response = Response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content="[ ERROR: Please enter the necessary data ]"
        )
        return response
    else:
        logging.info("[ START: Model inference started ]")

        # Prediction
        y_pred = predictApi(data, '/artifacts')
        logging.info("[ FINISH: Prediction completed ]")

        # Result
        response = Response(
            status_code=status.HTTP_200_OK,
            content="[ RESULT: The predicted income is: " + str(list(y_pred)[0]) + " ]"
        )
        logging.info("[ RESULT: The predicted income is: " + str(list(y_pred)[0]) + " ]")
        return response
