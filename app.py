import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Load the trained model
model = joblib.load('xgboost_model.pkl')

# Define the expected feature columns
feature_columns = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 
                   'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 
                   'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 
                   'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 
                   'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

class Features(BaseModel):
    features: list

@app.post('/predict')
async def predict(data: Features):
    # Ensure the input features are in a 2D array (1 row, 22 columns)
    data=[data]
    X = np.array(data[0].features).reshape(1, -1)  # Reshape to 2D: 1 row, 22 columns
    
    # Create a DataFrame with the correct column names
    X = pd.DataFrame(X, columns=feature_columns)

    # Make predictions using the loaded model
    y_pred = model.predict(xgb.DMatrix(X))

    # Return the predictions as a JSON response
    result = "satisfied" if y_pred[0] > 0.5 else "unsatisfied"

    # Return the result as a JSON response
    response = {
        'prediction': result
    }
    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
