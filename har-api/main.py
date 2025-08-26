from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd 
from typing import List

# Initialize the FastAPI app
app = FastAPI(title = "Human Activity Recognition API" )

# 2. Define the input data model using Pydantic
# This ensures that any data sent to the /predict endpoint has the correct structure.
class Features(BaseModel):
    features: List[float] # Expects a list of floating-point numbers

# Loading the saved components
model = joblib.load("HAR-model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define the app's prediction endpoint
@app.post("/predict")
def predict_activity(data: Features):
    """
    Takes a list of  561 features and returns the predicted activity
    """
    try:
        # Converts input list into 2D dataframe
        features_df = pd.DataFrame([data.features])

        
        # Scale the data
        features_scaled = scaler.transform(features_df)

        # Apply PCA
        features_pca = pca.transform(features_scaled)

        # Prediction using model (encoded)
        prediction_encoded = model.predict(features_pca)

        # Decode the prediction to get the activity name
        prediction = label_encoder.inverse_transform(prediction_encoded)

        # Return the prediction in JSON response
        return {"Predictied Activity: ", prediction[0]}
    
    except Exception as e:
        # Returning the error msg if something goes wrong
        return {"error" : str(e)}

# A root endpoint to check if API is running
@app.get("/")
def read_root():
    return {"message" : "Welcome to Human Activity Recognition API..." }

# To run the server -> uvicorn main:app --reload
