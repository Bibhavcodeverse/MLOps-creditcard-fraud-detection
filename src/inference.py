import pandas as pd
import joblib
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Add src to path so we can import feature_engineering as it was during training
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Now we can import it (and unpickle expects 'feature_engineering' module)
try:
    from feature_engineering import FeatureEngineering
except ImportError:
    # Fallback if running from root without path mod (though path mod above should fix it)
    from src.feature_engineering import FeatureEngineering

# Initialize FastAPI
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0")

# Input Schema
class Transaction(BaseModel):
    Time: float
    Amount: float
    # We might need V1-V28, but for simplicity let's assume the user sends them
    # OR we can assume the model expects all columns.
    # The dataset has 30 features: Time, V1-V28, Amount. 
    # Let's use **extra fields to capture V1-V28
    
    class Config:
        extra = "allow" 

class Prediction(BaseModel):
    prediction: int
    probability: float
    status: str

# Load Model and Feature Engineer
MODEL_PATH = "models/model.pkl"
FE_PATH = "models/feature_engineer.pkl"

model = None
fe = None

@app.on_event("startup")
def load_artifacts():
    global model, fe
    if os.path.exists(MODEL_PATH) and os.path.exists(FE_PATH):
        model = joblib.load(MODEL_PATH)
        fe = joblib.load(FE_PATH)
        print("Model and Feature Engineering artifacts loaded.")
    else:
        print("Warning: Artifacts not found. Run training first.")

@app.post("/predict", response_model=List[Prediction])
def predict(transactions: List[Transaction]):
    global model, fe
    
    if not model or not fe:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        # Convert to DataFrame
        data = [t.dict() for t in transactions]
        df = pd.DataFrame(data)
        
        # Explicitly ensure correct column order for Feature Engineering
        # This matches the training set columns
        expected_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                         'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                         'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                         'V28', 'Amount']
        
        # Filter and reorder columns
        df = df[expected_cols]
        
        # Feature Engineering Transform
        df_transformed = fe.transform(df)
        
        # Predict
        predictions = model.predict(df_transformed)
        probabilities = model.predict_proba(df_transformed)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "probability": float(prob),
                "status": "Fraud" if pred == 1 else "Legitimate"
            })
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    # Run slightly differently if main
    uvicorn.run(app, host="0.0.0.0", port=8000)
