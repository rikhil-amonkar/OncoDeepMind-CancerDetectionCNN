from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
import pandas as pd
import joblib
from backend.programs.drug_nueral_network import DrugResponseModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json

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
        
# ************* CANCER RISK PREDICTION ROUTING *************

# Load the cancer risk model
def load_risk_model():
    risk_model = joblib.load("backend/saved_models/cancer_risk_model.pkl")
    risk_scaler = joblib.load("backend/saved_models/cancer_risk_scaler.pkl")

    # Import all model weights
    with open("backend/saved_models/model_weights.json", "r") as f:
        model_weights = json.load(f) # Load the model weights from JSON file

    return {
        'model': risk_model,
        'scaler': risk_scaler,
        'weights': model_weights
    }

risk_model = load_risk_model() # Load the cancer risk model

# Define route for the cancer risk prediction form
@app.get("/risk", response_class=HTMLResponse)
async def show_cancer_risk_form(request: Request):
    return templates.TemplateResponse("risk.html", {"request": request})

# Define route for cancer risk prediction and form submission
@app.post("/risk", response_class=HTMLResponse)
async def predict_cancer_risk(request: Request,
                              age: str = Form(...),
                              gender: str = Form(...),
                              height: str = Form(...),
                              weight: str = Form(...),
                              smoking: str = Form(...),
                              genetic_risk: str = Form(...),
                              physical_activity: str = Form(...),
                              alcohol_intake: str = Form(...),
                              cancer_history: str = Form(...)):

    # Create a DataFrame from the user input
    user_input = {
        'Age': [age],
        'Gender': [gender],
        'Height (cm)': [height],
        'Weight (kg)': [weight],
        'Smoking': [smoking],
        'GeneticRisk': [genetic_risk],
        'PhysicalActivity': [physical_activity],
        'AlcoholIntake': [alcohol_intake],
        'CancerHistory': [cancer_history]
    }

    # Calculate BMI from height and weight
    height_cm = float(user_input['Height (cm)'][0])
    weight_kg = float(user_input['Weight (kg)'][0])
    bmi = weight_kg / ((height_cm / 100) ** 2)

    # Drop the height and weight columns
    features = np.array([[user_input['Age'][0], user_input['Gender'][0], bmi, user_input['Smoking'][0], user_input['GeneticRisk'][0], user_input['PhysicalActivity'][0], user_input['AlcoholIntake'][0], user_input['CancerHistory'][0]]], dtype=float)

    # Scale the user input
    user_input_scaled = risk_model['scaler'].transform(features)

    # Make prediction of the user input
    prediction = risk_model['model'].predict(user_input_scaled)
    prediction_proba = risk_model['model'].predict_proba(user_input_scaled)
    print(f'Prediction: {'No Cancer' if prediction[0] == 0 else 'Cancer'}')
    print(f'Prediction Probability: {prediction_proba[0]}')

    # Feature contribution weights
    weights = risk_model['model'].coef_[0] # This is an array of the weights per feature
    intercept = risk_model['model'].intercept_[0] # The bias term
    features_names = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory'] # Feature names
    contributions = [] 
    linear_sum = intercept

    for i, name in enumerate(features_names):
        value = float(features[0][i]) # Get the value of the feature
        weight = float(weights[i]) # Get the weight of the feature
        contribution = value * weight # Calculate the contribution of the feature
        linear_sum += contribution
        contributions.append({
            "feature": name,
            "value": round(value, 2),
            "weight": round(weight, 3),
            "contribution": round(contribution, 3)
        })

    # Calculate percent change contribution to risk based on weights
    total_weight = sum(abs(cont['contribution']) for cont in contributions)
    all_percentages = []
    for cont in contributions:
        percentage = (abs(cont['contribution']) / total_weight) * 100 # Calculate the percentage contribution
        all_percentages.append(percentage)
        print(f"Contribution: {cont['contribution']:.3f}, Percent: {percentage:.2f}%")

    print("Total weight:", total_weight)
    print("All percentages:", all_percentages)

    # Create a new list of contributions with percentage
    contributions_with_percent = []
    for cont, percentage in zip(contributions, all_percentages):
        contributions_with_percent.append({
            "feature": cont["feature"],
            "value": cont["value"],
            "weight": cont["weight"],
            "contribution": cont["contribution"],
            "percent": round(percentage, 2)
        })
    
    # Calculate the probability of having cancer based on weighed prediction
    probability = prediction_proba[0][1] * 100
    print(f'Probability of having cancer: {probability:.2f}%')

    # Generate recommendations based on user input
    recommendations = []
    if float(user_input['Smoking'][0]) > 0.5:
        recommendations.append("Consider quitting smoking to significantly lower your cancer risk.")
    if float(user_input['AlcoholIntake'][0]) > 0.5:
        recommendations.append("Reducing alcohol intake can help reduce cancer risk.")
    if float(user_input['PhysicalActivity'][0]) < 0.5:
        recommendations.append("Increasing physical activity improves overall health and reduces cancer risk.")
    if float(bmi) > 25:
        recommendations.append("Maintaining a healthy BMI through diet and exercise can help.")
    if float(user_input['GeneticRisk'][0]) > 0.5:
        recommendations.append("Consider speaking to a genetic counselor for a detailed risk assessment.")
    if not recommendations:
        recommendations.append("Your current lifestyle choices are healthy! Keep it up.")

    # Return the prediction result to the HTML template
    return templates.TemplateResponse("risk.html", {"request": request, "prediction": f"{probability:.2f}%", "recommendations": recommendations, "contributions": contributions_with_percent})
                              

