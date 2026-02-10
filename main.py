import __main__
import sys
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

def yes_no_to_int(x):
    return (x == "Yes").astype(int)

__main__.yes_no_to_int = yes_no_to_int

@app.get("/")
def health_check():
    return {"status": "OK", "message": "Customer Churn API is running"}


class ChurnRequest(BaseModel):
    account_length: int
    total_day_minutes: float
    total_eve_minutes: float
    total_night_minutes: float
    total_intl_minutes: float
    customer_service_calls: int
    number_vmail_messages: int
    total_day_calls: int
    total_eve_calls: int
    total_night_calls: int
    total_intl_calls: int
    international_plan: str
    voice_mail_plan: str
    area_code: int


@app.post("/predict")
def predict_churn(request: ChurnRequest):
    try:
        data = CustomData(
            account_length=request.account_length,
            total_day_minutes=request.total_day_minutes,
            total_eve_minutes=request.total_eve_minutes,
            total_night_minutes=request.total_night_minutes,
            total_intl_minutes=request.total_intl_minutes,
            customer_service_calls=request.customer_service_calls,
            number_vmail_messages=request.number_vmail_messages,
            total_day_calls=request.total_day_calls,
            total_eve_calls=request.total_eve_calls,
            total_night_calls=request.total_night_calls,
            total_intl_calls=request.total_intl_calls,
            international_plan=request.international_plan,
            voice_mail_plan=request.voice_mail_plan,
            area_code=request.area_code
        )

        df = data.get_data_as_dataframe()
        predictor = PredictPipeline()
        prediction = predictor.predict(df)

        return {
            "churn_prediction": int(prediction[0]),
            "label": "Yes" if prediction[0] == 1 else "No"
        }

    except Exception as e:
        raise CustomException(e, sys)