import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from drug_nueral_network import DrugResponseModel # Import the model class

#******************* PREPROCESSING THE DATA *******************

# Define the needed columns
desired_cols = [
                    # 'COSMIC_ID', 
                    'CELL_LINE_NAME', 
                    'TCGA_DESC', 
                    'CNA', 
                    'Gene Expression', 
                    'Methylation', 
                    'Microsatellite instability Status (MSI)', 
                    # 'Growth Properties',
                    'Screen Medium',
                    'Cancer Type (matching TCGA label)',
                    'TARGET',
                    'TARGET_PATHWAY', 
                    'AUC'
                ]    

# Load the data
data = pd.read_csv('data/GDSC_DATASET.csv', usecols=desired_cols) # Only load the relevant columns

# Preprocess the data
data_df = pd.DataFrame(data)
data_df = data_df.dropna() # Drop rows with missing values

# Feature engineering for similar features to help the model see patterns
data_df['MSI_CNA_Interaction'] = data_df['Microsatellite instability Status (MSI)'].astype(str) + '_' + data_df['CNA'].astype(str) # Create a new feature that combines MSI status and CNA status
data_df['TARGET_Expression'] = data_df['Gene Expression'].astype(str) + '_' + data_df['TARGET'].astype(str) # Create a new feature that combines Gene Expression and TARGET status
data_df['TARGET_Methylation_PATH'] = data_df['Methylation'].astype(str) + '_' + data_df['TARGET_PATHWAY'].astype(str) # Create a new feature that combines Methylation and TARGET_PATHWAY status
data_df['CNA_TARGET'] = data_df['CNA'].astype(str) + '_' + data_df['TARGET'].astype(str) # Create a new feature that combines CNA and TARGET status
data_df['CLN_Medium'] = data_df['CELL_LINE_NAME'].astype(str) + '_' + data_df['Screen Medium'].astype(str) # Create a new feature that combines cell line name and screen medium status
data_df['Type_Expression'] = data_df['Cancer Type (matching TCGA label)'].astype(str) + '_' + data_df['Gene Expression'].astype(str) # Create a new feature that combines Cancer Type and Gene Expression status
data_df['MSI_Type'] = data_df['Microsatellite instability Status (MSI)'].astype(str) + '_' + data_df['Cancer Type (matching TCGA label)'].astype(str) # Create a new feature that combines MSI status and Cancer Type

# Convert categorical variables to numerical
categorical_cols = ['CELL_LINE_NAME', 'TCGA_DESC', 'CNA', 'Gene Expression', 'Methylation', 'Microsatellite instability Status (MSI)', 'Screen Medium', 'Cancer Type (matching TCGA label)', 'TARGET', 'TARGET_PATHWAY',
                    
                    # Adding the new feature to the list of categorical columns
                    'MSI_CNA_Interaction',
                    'TARGET_Expression', 
                    'TARGET_Methylation_PATH',
                    'CNA_TARGET',
                    'CLN_Medium',
                    'Type_Expression',
                    'MSI_Type'
                ]
data_df = pd.get_dummies(data_df, columns=categorical_cols)

# Split the data into features and target variable
X = data_df.drop(columns=['AUC'])
y = data_df['AUC'].values.reshape(-1, 1) # Reshape y to be a 2D array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the data
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train) # Fit and transform the training data
X_test_scaled = x_scaler.transform(X_test) # Transform the test data
y_train_scaled = y_scaler.fit_transform(y_train) # Fit and transform the training data
y_test_scaled = y_scaler.transform(y_test) # Transform the test data

# Turn the X features into float tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# Turn the y labels into long tensors
y_train_tensor = torch.FloatTensor(y_train_scaled) # Unsqueeze to add a dimension
y_test_tensor = torch.FloatTensor(y_test_scaled) # Unsqueeze to add a dimension

# Intialize an instance of the model
input_size = X_train.shape[1] # Number of features
model = DrugResponseModel(input_features=input_size)

# Criterion and optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4) # Adam optimizer with learning rate of 0.0001
criterion = nn.MSELoss() # Loss function for regression

#******************* TRAINING THE MODEL *******************

epochs = 400 # Number of epochs
losses = [] # List to store the loss values

print(X_test_scaled.shape)

for i in range(epochs):

    model.train() # Set the model to training mode
    y_pred = model(X_train_tensor) # Forward pass the training data through the model
    loss = criterion(y_pred, y_train_tensor) # Calculate the loss

    # Track losses and epochs
    losses.append(loss.item())
    if i % 20 == 0:
        print(f'Epoch {i}, Loss: {loss.item():.5f}')

    # Backpropagation to fine tune the any weights with error
    optimizer.zero_grad() # Zero the gradients
    loss.backward() # Backpropagation
    optimizer.step() # Update the weights

# Save model and artifacts
def save_model():
    torch.save(model.state_dict(), 'DrugResponseModel.pth')
    joblib.dump(x_scaler, 'x_scaler.pkl')
    joblib.dump(y_scaler, 'y_scaler.pkl')
    joblib.dump(X.columns, 'columns.pkl')
    joblib.dump(categorical_cols, 'categorical_cols.pkl')
    print("Model and artifacts saved successfully")

save_model()

#******************* BEST MODEL METRICS *******************

# Learning Rate: 0.0005
# Weight Decay: 0.0001 (1e-4)
# Epochs: 400
# Test Size: 0.2
# Random State: 0

# HL1 = 128
# HL2 = 64
# HL3 = 32
# HL4 = 16
# HL5 = 8
# Output Features = 1

# Mean Squared Error: 0.04970 (0 - 1 scale) - Preffered 0
# R^2 Score: 0.6610 (0 - 1 scale) - Preffered 1

#******************* TESTING THE MODEL *******************

# model.eval() # Set the model to evaluation mode
# with torch.no_grad():
#     y_test_pred_scaled = model(X_test_tensor) # Forward pass the test data through the model

# # Inverse transform the predicted values and tragets to original scale
# y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.numpy())
# y_test_original = y_scaler.inverse_transform(y_test_tensor.numpy())

# # Clip the predictions to ensure they are within the [0, 1] range
# y_test_pred = np.clip(y_test_pred, 0, 1) # If value > 1, set to 1, if value < 0, set to 0

# # Calculate the R^2 score and mean squared error
# mse = mean_absolute_error(y_test_original, y_test_pred)
# r2 = r2_score(y_test_original, y_test_pred)

# print(f'\nMean Squared Error: {mse:.5f}')
# print(f'R^2 Score: {r2:.4f}')

# #******************** INTERPRETING AUC VALUES *******************

# # Fake data - an example with a similar structure to the original input
# fake_data = pd.DataFrame({
#     # 'COSMIC_ID': [683667], 
#     'CELL_LINE_NAME': ['PFSK-1'], 
#     'TCGA_DESC': ['MB'], 
#     'Cancer Type (matching TCGA label)': ['MB'],
#     'Microsatellite instability Status (MSI)': ['MSS/MSI-L'],
#     'Screen Medium': ['R'],
#     # 'Growth Properties': ['Adherent'],
#     'CNA': ['Y'],
#     'Gene Expression': ['Y'],
#     'Methylation': ['Y'],
#     'TARGET': ['TOP1'],
#     'TARGET_PATHWAY': ['DNA replication']
# })

# # Feature engineer the fake data with new columns made before
# fake_data['MSI_CNA_Interaction'] = fake_data['Microsatellite instability Status (MSI)'].astype(str) + '_' + fake_data['CNA'].astype(str) # Create a new feature that combines MSI status and CNA status
# fake_data['TARGET_Expression'] = fake_data['Gene Expression'].astype(str) + '_' + fake_data['TARGET'].astype(str) # Create a new feature that combines Gene Expression and TARGET status
# fake_data['TARGET_Methylation_PATH'] = fake_data['Methylation'].astype(str) + '_' + fake_data['TARGET_PATHWAY'].astype(str) # Create a new feature that combines Methylation and TARGET_PATHWAY status
# fake_data['CNA_TARGET'] = fake_data['CNA'].astype(str) + '_' + fake_data['TARGET'].astype(str) # Create a new feature that combines CNA and TARGET status
# fake_data['CLN_Medium'] = fake_data['CELL_LINE_NAME'].astype(str) + '_' + fake_data['Screen Medium'].astype(str) # Create a new feature that combines cell line name and screen medium status
# fake_data['Type_Expression'] = fake_data['Cancer Type (matching TCGA label)'].astype(str) + '_' + fake_data['Gene Expression'].astype(str) # Create a new feature that combines Cancer Type and Gene Expression status
# fake_data['MSI_Type'] = fake_data['Microsatellite instability Status (MSI)'].astype(str) + '_' + fake_data['Cancer Type (matching TCGA label)'].astype(str) # Create a new feature that combines MSI status and Cancer Type

# # Preprocess the fake data
# fake_data = pd.get_dummies(fake_data, columns=categorical_cols)
# fake_data_encoded = fake_data.reindex(columns=X.columns, fill_value=0) # Reindex to match the original data
# fake_data_scaled = x_scaler.transform(fake_data_encoded) # Scale the fake data
# fake_data_tensor = torch.FloatTensor(fake_data_scaled)

# # Make predictions on the fake data
# model.eval()
# with torch.no_grad():
#     fake_data_pred_scaled = model(fake_data_tensor) # Forward pass the fake data through the model
#     fake_data_pred = y_scaler.inverse_transform(fake_data_pred_scaled.numpy()) # Inverse transform the predictions
#     print(f'Predicted AUC for fake data: {fake_data_pred[0][0]:.5f}')

# # Calculate percent effectiveness from AUC
# percent_effectiveness = (fake_data_pred[0][0]) * 100
# print(f'Percent effectiveness: {percent_effectiveness:.2f}%')

#****************** FAKE DATA PREDICTION METRICS *******************

# ####### FAKE DATA #######

# 'COSMIC_ID': [683667], 
# 'CELL_LINE_NAME': ['PFSK-1'], 
# 'TCGA_DESC': ['MB'], 
# 'Cancer Type (matching TCGA label)': ['MB'],
# 'Microsatellite instability Status (MSI)': ['MSS/MSI-L'],
# 'Screen Medium': ['R'],
# 'Growth Properties': ['Adherent'],
# 'CNA': ['Y'],
# 'Gene Expression': ['Y'],
# 'Methylation': ['Y'],
# 'TARGET': ['TOP1'],
# 'TARGET_PATHWAY': ['DNA replication']

# ###### TARGET AUC ######

# Target AUC: 0.93022 (93.02% effectiveness)
# Fake Data Prediction AUC : 0.80564 (80.56% effectiveness)
# Percent Error: 13.39%

#******************* SAVE THE MODEL *******************

# Save the model
# torch.save(model.state_dict(), 'DrugResponseModel.pth')
