import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#******************* NUERAL NETWORK MODEL *******************

# Define the neural network model
class DrugResponseModel(nn.Module): # Input layer -> Hidden layer 1 -> Hidden layer 2 -> Output layer

    # The input layer has 7 features
    # The hidden layer 1 has 32 neurons
    # The hidden layer 2 has 16 neurons
    # The output layer has 2 neuron (0 = resistant, 1 = sensitive)

    def __init__(self, input_features = 7, hl1 = 32, hl2 = 16, output_feature = 2): # Funnel structure

        # 32 neurons derived from features * 2-4
        # Funnels down to 16 nuerons

        super().__init__() # Inherit from nn.Module
        self.fc1 = nn.Linear(input_features, hl1) # From input layer to hidden layer 1
        self.fc2 = nn.Linear(hl1, hl2) # From hidden layer 1 to hidden layer 2
        self.output = nn.Linear(hl2, output_feature) # From hidden layer 2 to output layer

    # Helps foward pass the data through the network
    def forward(self, x):

        # Rectified Linear Unit (ReLU) activation function to introduce non-linearity
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

# Pick a manual seed for randomization
torch.manual_seed(42)

# Intialize an instance of the model
model = DrugResponseModel()

#******************* PREPROCESSING THE DATA *******************

# Define the needed columns
categorical_cols = [
                    'Whole Exome Sequencing (WES)', 
                    'Copy Number Alterations (CNA)', 
                    'Gene Expression', 
                    'Methylation', 
                    'GDSC\nTissue descriptor 1', 
                    'GDSC\nTissue\ndescriptor 2', 
                    'Cancer Type\n(matching TCGA label)',
                    'Drug\nResponse' 
                    ]    

# Load the data
data = pd.read_excel('data/Cell_Lines_Details.xlsx', engine='openpyxl', # To open Excel files
                    usecols=categorical_cols) # Only load the relevant columns

# Preprocess the data
data_df = pd.DataFrame(data)
data_df = data_df.dropna() # Drop rows with missing values

# Convert categorical variables to numerical
data_df[categorical_cols] = data_df[categorical_cols].apply(lambda x: pd.factorize(x)[0])
print(data_df.head())

# Factorize each categorical column and print the mappings
for col in categorical_cols:
    unique_classes, encoded_classes = pd.factorize(data_df[col])
    class_mapping = dict(zip(encoded_classes, unique_classes))
    print(f"Class Mapping for {col}:")
    print(class_mapping)
    print()

# Split the data into features and target variable
X = data_df.drop(columns=['Drug\nResponse'])
y = data_df['Drug\nResponse']

# Turn the data into a numpy array
X_values = X.values
y_values = y.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.2, random_state=42)

# Use SMOTE to handle class imbalance with the training data
smote = SMOTE(sampling_strategy='auto', k_neighbors=4, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Original dataset shape: {X_train.shape}, {y_train.shape}")
print(f"Resampled dataset shape: {X_train_smote.shape}, {y_train_smote.shape}")

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_smote) # Fit and transform the training data
X_test = scaler.transform(X_test) # Transform the test data

# Turn the X features into float tensors
X_train_smote = torch.FloatTensor(X_train_smote)
X_test = torch.FloatTensor(X_test)

# Turn the y labels into long tensors
y_train_smote = torch.LongTensor(y_train_smote)
y_test = torch.LongTensor(y_test)

# Criterion and optimizer
criterion = nn.CrossEntropyLoss() # Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam optimizer with learning rate of 0.001

#******************* TRAINING THE MODEL *******************

epochs = 100 # Number of epochs
losses = [] # List to store the loss values

for i in range(epochs):

    y_pred = model(X_train_smote) # Forward pass the training data through the model
    loss = criterion(y_pred, y_train_smote) # Calculate the loss

    # Track losses and epochs
    losses.append(loss.item())
    if i % 10 == 0:
        print(f'Epoch {i}, Loss: {loss.item()}')

    # Backpropagation to fine tune the any weights with error
    optimizer.zero_grad() # Zero the gradients
    loss.backward() # Backpropagation
    optimizer.step() # Update the weights

# Plot the loss values
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()

#******************* TESTING THE MODEL *******************

with torch.no_grad():
    y_test_pred = model(X_test) # Forward pass the test data through the model
    test_loss = criterion(y_test_pred, y_test) # Calculate the loss
    print(f'Test Loss: {test_loss.item()}')

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        # Will tell us which class the model predicts
        print(f'{i + 1}. Predicted: {str(y_val)}, Actual: {y_test[i]}')

        # Measure the accuracy
        if torch.argmax(y_val) == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Evaluate the model with some test data
test_data = torch.FloatTensor([[0, 1, 0, 1, 1, 1, 2]]) # Replace with your test data

with torch.no_grad():
    print(model(test_data))
