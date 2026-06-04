import os
import glob
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse

app = FastAPI(
    title="Credit Risk Assessment API",
    description="Production endpoint for identifying high-risk transactions.",
    version="1.0.0"
)

# Global container for our trained model artifact
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Search dynamically for the serialized model artifact stored by MLflow
        mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "mlruns"))
        pickle_files = glob.glob(os.path.join(mlruns_path, "**", "model.pkl"), recursive=True)
        
        if not pickle_files:
            raise FileNotFoundError("Could not locate 'model.pkl' within the mlruns storage structure.")
            
        # Select the most recent run artifact
        latest_model_path = pickle_files[-1]
        print(f"📦 Successfully loading model binary from: {latest_model_path}")
        
        with open(latest_model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"❌ Critical initialization failure: {str(e)}")
        model = None

@app.get("/")
def read_root():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=CreditRiskResponse)
def predict_risk(payload: CreditRiskRequest):
    if model == None:
        raise HTTPException(status_code=503, detail="Prediction model state is uninitialized.")
    
    try:
        # Convert incoming structured data directly to a row format matching model expectations
        input_data = pd.DataFrame([payload.dict()])
        
        # Ensure our evaluation dataframe shape matches training columns exactly
        # If your model structure used specific dummied structural arrays, pad missing ones
        model_features = getattr(model, "feature_names_in_", input_data.columns)
        input_data = input_data.reindex(columns=model_features, fill_value=0)
        
        # Calculate risk scores
        prob = float(model.predict_proba(input_data)[0, 1])
        prediction = bool(prob >= 0.5)
        
        return CreditRiskResponse(risk_probability=prob, is_high_risk=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failure: {str(e)}")
    