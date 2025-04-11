from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
import pandas as pd
import joblib
from backend.programs.drug_nueral_network import DrugResponseModel

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2 templates and static files
templates = Jinja2Templates(directory="frontend/templates") # Directory for HTML templates
app.mount("/static", StaticFiles(directory="frontend/static/"), name="static")

# Load the model and other features
def load_model():
     model = DrugResponseModel(input_features=len(joblib.load("backend/saved_models/columns.pkl"))) # Initialize the model
     model.load_state_dict(torch.load("backend/saved_models/DrugResponseModel.pth")) # Load the model weights
     model.eval() # Set the model to evaluation mode

    # Return the model and other features
     return {
            'model': model,
            'x_scaler': joblib.load("backend/saved_models/x_scaler.pkl"), # Load the scaler for input features
            'y_scaler': joblib.load("backend/saved_models/y_scaler.pkl"), # Load the scaler for output features
            'columns': joblib.load("backend/saved_models/columns.pkl"), # Load the columns for input features
            'categorical_cols': joblib.load("backend/saved_models/categorical_cols.pkl"), # Load the categorical columns
     }

program_features = load_model() # Load the model and other features

# Define a root endpoint and function to handle requests
@app.get("/", response_class=HTMLResponse) # Root endpoint
async def name(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define root the about page
@app.get("/about", response_class=HTMLResponse) # About page endpoint
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

# Define route for the form page
@app.get("/predict", response_class=HTMLResponse)
async def show_predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

# Define route for prediciton and form submission
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  cell_line_name: str = Form (...),
                  tcga: str = Form (...),
                  cna: str = Form (...),
                  gene_expression: str = Form (...),
                  methylation: str = Form (...),
                  msi: str = Form (...),
                  screen_medium: str = Form (...),
                  cancer_type: str = Form (...),
                  target: str = Form (...),
                  target_pathway: str = Form (...)):
        
    user_data = pd.DataFrame({
        # 'COSMIC_ID': [683667], 
        'CELL_LINE_NAME': [cell_line_name], 
        'TCGA_DESC': [tcga], 
        'Cancer Type (matching TCGA label)': [cancer_type],
        'Microsatellite instability Status (MSI)': [msi],
        'Screen Medium': [screen_medium],
        # 'Growth Properties': ['Adherent'],
        'CNA': [cna],
        'Gene Expression': [gene_expression],
        'Methylation': [methylation],
        'TARGET': [target],
        'TARGET_PATHWAY': [target_pathway]
    })
    
    # Feature engineer the fake data with new columns made before
    user_data['MSI_CNA_Interaction'] = user_data['Microsatellite instability Status (MSI)'].astype(str) + '_' + user_data['CNA'].astype(str) # Create a new feature that combines MSI status and CNA status
    user_data['TARGET_Expression'] = user_data['Gene Expression'].astype(str) + '_' + user_data['TARGET'].astype(str) # Create a new feature that combines Gene Expression and TARGET status
    user_data['TARGET_Methylation_PATH'] = user_data['Methylation'].astype(str) + '_' + user_data['TARGET_PATHWAY'].astype(str) # Create a new feature that combines Methylation and TARGET_PATHWAY status
    user_data['CNA_TARGET'] = user_data['CNA'].astype(str) + '_' + user_data['TARGET'].astype(str) # Create a new feature that combines CNA and TARGET status
    user_data['CLN_Medium'] = user_data['CELL_LINE_NAME'].astype(str) + '_' + user_data['Screen Medium'].astype(str) # Create a new feature that combines cell line name and screen medium status
    user_data['Type_Expression'] = user_data['Cancer Type (matching TCGA label)'].astype(str) + '_' + user_data['Gene Expression'].astype(str) # Create a new feature that combines Cancer Type and Gene Expression status
    user_data['MSI_Type'] = user_data['Microsatellite instability Status (MSI)'].astype(str) + '_' + user_data['Cancer Type (matching TCGA label)'].astype(str) # Create a new feature that combines MSI status and Cancer Type

    # Preprocess the fake data
    user_data = pd.get_dummies(user_data, columns=program_features['categorical_cols'])
    user_data_encoded = user_data.reindex(columns=program_features['columns'], fill_value=0)
    user_data_scaled = program_features['x_scaler'].transform(user_data_encoded) # Scale the fake data
    user_data_tensor = torch.FloatTensor(user_data_scaled)

    # Make predictions on the fake data
    with torch.no_grad():
        user_data_pred_scaled = program_features['model'](user_data_tensor) # Forward pass the fake data through the model
        user_data_pred = program_features['y_scaler'].inverse_transform(user_data_pred_scaled.numpy()) # Inverse transform the predictions
        print(f'Predicted AUC for fake data: {user_data_pred[0][0]:.5f}')

    # Calculate percent effectiveness from AUC
    percent_effectiveness = (user_data_pred[0][0]) * 100
    print(f'Percent effectiveness: {percent_effectiveness:.2f}%')

    # Return the prediction result to the HTML template
    return templates.TemplateResponse("predict.html", {"request": request, "prediction": f"{percent_effectiveness:.2f}%"})
        
    


