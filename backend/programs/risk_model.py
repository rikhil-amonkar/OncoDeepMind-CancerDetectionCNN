import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# *************** CANCER RISK PREDICTION MODEL ***************

# Load the dataset
data = pd.read_csv('backend/data/The_Cancer_data_1500.csv', usecols=['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory', 'Diagnosis'])    

# Preprocess the data
data_pd = pd.DataFrame(data)
data_pd = data_pd.dropna()

# Calculate the BMI of user based on the height and weight in kg/cm
# data_pd['BMI'] = data_pd['weight_kg'] / ((data_pd['height_cm'] / 100) ** 2)

# Define the features and target variable
X = data_pd.drop(columns=['Diagnosis'])
y = data_pd['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {(accuracy * 100):.2f}%')

# *************** TESTING THE MODEL ON FAKE USER INPUT ***************

# Test the model with a sample input
user_input = {
    'Age': 65,
    'Gender': 1,
    'Height (cm)': 170,
    'Weight (kg)': 100,
    'Smoking': 1,
    'GeneticRisk': 1,
    'PhysicalActivity': 2,
    'AlcoholIntake': 3,
    'CancerHistory': 1
}

# Calculate BMI from height and weight
bmi = user_input['Weight (kg)'] / ((user_input['Height (cm)'] / 100) ** 2)

# Drop the height and weight columns
features = np.array([[user_input['Age'], user_input['Gender'], bmi, user_input['Smoking'], user_input['GeneticRisk'], user_input['PhysicalActivity'], user_input['AlcoholIntake'], user_input['CancerHistory']]])

# Scale the user input
user_input_scaled = scaler.transform(features)

# Make prediction of the user input
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)
print(f'Prediction: {'No Cancer' if prediction[0] == 0 else 'Cancer'}')
print(f'Prediction Probability: {prediction_proba[0]}')

# Calculate the probability of having cancer based on weighed prediction
probability = prediction_proba[0][1] * 100
print(f'Probability of having cancer: {probability:.2f}%')

# # Save the model
# joblib.dump(model, 'backend/saved_models/cancer_risk_model.pkl')

# # Save the scaler
# joblib.dump(scaler, 'backend/saved_models/cancer_risk_scaler.pkl')
