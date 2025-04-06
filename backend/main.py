from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.model import DrugResponseModel, x_scaler, y_scaler, X
import torch
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the model
input_size = X.shape[1]
model = DrugResponseModel(input_features=input_size)
model.load_state_dict(torch.load('model/DrugResponseModel.pth')) # Load the model weights
model.eval() # Set the model to evaluation mode

# Enable CORS for frontend to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define Pydantic input class (same fields as the fake_data)
class DrugFormInput(BaseModel):
    COSMIC_ID: int
    CELL_LINE_NAME: str
    TCGA_DESC: str
    Cancer_Type: str
    MSI_Status: str
    Screen_Medium: str
    Growth_Properties: str
    CNA: str
    Gene_Expression: str
    Methylation: str
    TARGET: str
    TARGET_PATHWAY: str

# Route the prediction form
@app.post("/predict")
async def predict_drug_response(input_data: DrugFormInput):

    # Assign the input data to a dictionary using the input_data labels
    input_dict = {
        'COSMIC_ID': [input_data.COSMIC_ID], 
        'CELL_LINE_NAME': [input_data.CELL_LINE_NAME], 
        'TCGA_DESC': [input_data.TCGA_DESC], 
        'Cancer Type (matching TCGA label)': [input_data.Cancer_Type],
        'Microsatellite instability Status (MSI)': [input_data.MSI_Status],
        'Screen Medium': [input_data.Screen_Medium],
        'Growth Properties': [input_data.Growth_Properties],
        'CNA': [input_data.CNA],
        'Gene Expression': [input_data.Gene_Expression],
        'Methylation': [input_data.Methylation],
        'TARGET': [input_data.TARGET],
        'TARGET_PATHWAY': [input_data.TARGET_PATHWAY]
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_dict)

    #****************** PREPROCESSING THE INPUT DATA #******************
    categorical_cols = [
                        'CELL_LINE_NAME',
                        'TCGA_DESC',
                        'CNA',
                        'Gene Expression',
                        'Methylation',
                        'Microsatellite instability Status (MSI)',
                        'Growth Properties',
                        'Screen Medium',
                        'Cancer Type (matching TCGA label)',
                        'TARGET',
                        'TARGET_PATHWAY'
                    ]
    
    # Preprocess the data and convert categorical variables to numerical
    df_numeric = pd.get_dummies(input_df, columns=categorical_cols)
    df_encoded = df_numeric.reindex(columns=X.columns, fill_value=0) # Reindex to match the original data
    df_scaled = x_scaler.transform(df_encoded) # Scale the fake data
    df_tensor = torch.FloatTensor(df_scaled)

    # Make predictions on the input data
    model.eval()
    with torch.no_grad():
        df_pred_caled = model(df_tensor) # Forward pass the fake data through the model
        prediciton = y_scaler.inverse_transform(df_pred_caled.numpy()) # Inverse transform the predictions
        auc = np.clip(prediciton[0][0], 0, 1) # Make sure the predictions are within the [0, 1] range

    # Return the predicted AUC percent probability
    percent_effectiveness = auc * 100
    return {
        'Predicted AUC': auc,
        'Percent Effectiveness': percent_effectiveness
    }