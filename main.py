from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import uvicorn
from tensorflow.keras.models import load_model
from model import PINN

# Initialize the model
model = PINN()
# Create a sample input to initialize the model
sample_input = np.zeros((1, 25))
_ = model(sample_input)  # This will build the model
# Load the weights
model.load_weights("pinn_model.h5")

# Define the FastAPI app
app = FastAPI()


# Input schema
class InputData(BaseModel):
    features: List[float]  # Expect 25 features

@app.get("/")
def root():
    return {"message": "ML model is ready"}

@app.post("/predict")
def predict(data: InputData):
    # Convert to NumPy array and reshape for model
    input_array = np.array(data.features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)

    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
