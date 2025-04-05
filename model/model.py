import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

#******************* NUERAL NETWORK MODEL *******************

# Define the neural network model
class DrugResponseModel(nn.Module): # Input layer -> Hidden layer 1 -> Hidden layer 2 -> Hidden layer 3 -> Output layer

    # The input layer has 12 features
    # The hidden layer 1 has 128 neurons
    # The hidden layer 2 has 64 neurons
    # The hidden layer 3 has 32 neurons
    # The output layer has 1 neuron (scalar AUC value between 0 and 1)

    def __init__(self, input_features, hl1 = 128, hl2 = 64, hl3 = 32, output_feature = 1): # Funnel structure

        # 128 neurons derived from features
        # Funnels down to 64 nuerons, then 32 neurons

        super().__init__() # Inherit from nn.Module
        self.fc1 = nn.Linear(input_features, hl1) # From input layer to hidden layer 1
        self.fc2 = nn.Linear(hl1, hl2) # From hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(hl2, hl3) # From hidden layer 2 to hidden layer 3
        self.output = nn.Linear(hl3, output_feature) # From hidden layer 3 to output layer

    # Helps foward pass the data through the network
    def forward(self, x):

        # Rectified Linear Unit (ReLU) activation function to introduce non-linearity
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x

# Pick a manual seed for randomization
torch.manual_seed(42)

#******************* PREPROCESSING THE DATA *******************

# Define the needed columns
desired_cols = [
                    'COSMIC_ID', 
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
                    'TARGET_PATHWAY', 
                    'AUC'
                ]    

# Load the data
data = pd.read_csv('data/GDSC_DATASET.csv', usecols=desired_cols) # Only load the relevant columns

# Preprocess the data
data_df = pd.DataFrame(data)
data_df = data_df.dropna() # Drop rows with missing values

# Convert categorical variables to numerical
categorical_cols = ['CELL_LINE_NAME', 'TCGA_DESC', 'CNA', 'Gene Expression', 'Methylation', 'Microsatellite instability Status (MSI)', 'Growth Properties', 'Screen Medium', 'Cancer Type (matching TCGA label)', 'TARGET', 'TARGET_PATHWAY']
data_df = pd.get_dummies(data_df, columns=categorical_cols)

# Split the data into features and target variable
X = data_df.drop(columns=['AUC'])
y = data_df['AUC'].values.reshape(-1, 1) # Reshape y to be a 2D array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
criterion = nn.MSELoss() # Loss function for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Adam optimizer with learning rate of 0.001

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
    if i % 40 == 0:
        print(f'Epoch {i}, Loss: {loss.item():.5f}')

    # Backpropagation to fine tune the any weights with error
    optimizer.zero_grad() # Zero the gradients
    loss.backward() # Backpropagation
    optimizer.step() # Update the weights

# Plot the loss values
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss vs Epochs')
plt.show()

#******************* BEST MODEL METRICS *******************

# Learning Rate: 0.0001
# Epochs: 400

# Mean Squared Error: 0.05248 (0 - 1 scale) - Preffered 0
# R^2 Score: 0.6451 (0 - 1 scale) - Preffered 1

#******************* TESTING THE MODEL *******************

model.eval() # Set the model to evaluation mode
with torch.no_grad():
    y_test_pred_scaled = model(X_test_tensor) # Forward pass the test data through the model

# Inverse transform the predicted values and tragets to original scale
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.numpy())
y_test_original = y_scaler.inverse_transform(y_test_tensor.numpy())

# Calculate the R^2 score and mean squared error
mse = mean_absolute_error(y_test_original, y_test_pred)
r2 = r2_score(y_test_original, y_test_pred)

# Clip the predictions to ensure they are within the [0, 1] range
y_test_pred = np.clip(y_test_pred, 0, 1) # If value > 1, set to 1, if value < 0, set to 0

print(f'\nMean Squared Error: {mse:.5f}')
print(f'R^2 Score: {r2:.4f}')

#******************** INTERPRETING AUC VALUES *******************

# Fake data - an example with a similar structure to the original input
fake_data = pd.DataFrame({
    'COSMIC_ID': [683667], 
    'CELL_LINE_NAME': ['PFSK-1'], 
    'TCGA_DESC': ['MB'], 
    'Cancer Type (matching TCGA label)': ['MB'],
    'Microsatellite instability Status (MSI)': ['MSS/MSI-L'],
    'Screen Medium': ['R'],
    'Growth Properties': ['Adherent'],
    'CNA': ['Y'],
    'Gene Expression': ['Y'],
    'Methylation': ['Y'],
    'TARGET': ['TOP1'],
    'TARGET_PATHWAY': ['DNA replication']
})

# Preprocess the fake data
fake_data = pd.get_dummies(fake_data, columns=categorical_cols)
fake_data_encoded = fake_data.reindex(columns=X.columns, fill_value=0) # Reindex to match the original data
fake_data_scaled = x_scaler.transform(fake_data_encoded) # Scale the fake data
fake_data_tensor = torch.FloatTensor(fake_data_scaled)

# Make predictions on the fake data
model.eval()
with torch.no_grad():
    fake_data_pred_scaled = model(fake_data_tensor) # Forward pass the fake data through the model
    fake_data_pred = y_scaler.inverse_transform(fake_data_pred_scaled.numpy()) # Inverse transform the predictions
    print(f'Predicted AUC for fake data: {fake_data_pred[0][0]:.5f}')

# Calculate percent effectiveness from AUC
percent_effectiveness = (fake_data_pred[0][0]) * 100
print(f'Percent effectiveness: {percent_effectiveness:.2f}%')

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
