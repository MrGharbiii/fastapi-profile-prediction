from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import json
import uvicorn

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load model and artifacts
model = joblib.load("model_artifacts/obesity_profile_model.joblib")
label_encoders = joblib.load("model_artifacts/label_encoders.joblib")

with open("model_artifacts/feature_order.json", "r") as f:
    feature_order = json.load(f)

with open("model_artifacts/preprocessing_metadata.json", "r") as f:
    metadata = json.load(f)

@app.get("/")
async def root():
    return {
        "message": "Obesity Profile Prediction API",
        "status": "healthy",
        "version": "1.0"
    }
@app.post("/predict")
async def predict(data: dict):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Apply preprocessing
        processed_df = preprocess_data(df)
        
        # Get probabilities
        probabilities = model.predict_proba(processed_df)
        
        # Format results
        target_labels = ['digestif', 'hormonal', 'iatrogene', 'metabolique', 'psychologique']
        results = []
        
        for i, label in enumerate(target_labels):
            prob_positive = float(probabilities[i][0][1])
            percentage = round(prob_positive * 100, 1)
            
            results.append({
                "profile": label,
                "probability": prob_positive,
                "percentage": f"{percentage}%"
            })
        
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {"predictions": results}
        
    except Exception as e:
        return {"error": str(e)}

def preprocess_data(df):
    """Preprocess data to match training"""
    
    # Feature engineering for family history
    family_conditions = metadata['family_conditions']
    for condition in family_conditions:
        feature_name = f'fh_{condition.lower().replace(" ", "_")}'
        df[feature_name] = df['terrain familial'].apply(lambda x: has_condition(x, condition))
    
    # Psychology features
    psy_conditions = metadata['psy_conditions']
    for condition in psy_conditions:
        clean_condition = condition.lower().replace(" ", "_").replace("'", "")
        feature_name = f'psy_{clean_condition}'
        df[feature_name] = df['trouble psy'].apply(lambda x: has_condition(x, condition))
    
    # Treatment features
    treatment_conditions = metadata['treatment_conditions']
    for condition in treatment_conditions:
        clean_condition = condition.lower().replace("-", "_").replace("é", "e").replace("è", "e")
        feature_name = f'treat_{clean_condition}'
        df[feature_name] = df['traitements'].apply(lambda x: has_condition(x, condition))
    
    # Medical treatment features
    medical_conditions = metadata['medical_conditions']
    for condition in medical_conditions:
        clean_condition = condition.lower().replace("é", "e").replace("è", "e").replace("ê", "e")
        feature_name = f'med_{clean_condition}'
        df[feature_name] = df['TT medical'].apply(lambda x: has_condition(x, condition))
    
    # Drop original columns
    df = df.drop(['terrain familial', 'trouble psy', 'traitements', 'TT medical'], axis=1)
    
    # Reorder columns
    df = df.reindex(columns=feature_order, fill_value=0)
    
    # Apply label encoders
    for column in df.columns:
        if column in label_encoders:
            encoder = label_encoders[column]
            df[column] = df[column].astype(str)
            
            known_classes = set(encoder.classes_)
            df[column] = df[column].apply(lambda x: x if x in known_classes else encoder.classes_[0])
            
            df[column] = encoder.transform(df[column])
    
    return df

def has_condition(value, condition):
    """Check if condition exists in value"""
    if pd.isna(value) or value == 'non':
        return 0
    
    value_str = str(value).lower()
    condition_lower = condition.lower()
    
    return int(condition_lower in value_str)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)