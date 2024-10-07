from fastapi import FastAPI, HTTPException
from api.schemas.request_schemas import PredictionRequest
from api.services.prediction_service import PredictionService

app = FastAPI()

# Instantiate PredictionService
prediction_service = PredictionService()

@app.post("/predict")
async def predict(input_data: PredictionRequest):
    try:
        # Convert the request to a dictionary and make a prediction
        prediction = prediction_service.make_prediction(input_data.dict())
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
