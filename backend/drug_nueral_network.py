import torch
import torch.nn as nn

#******************* NUERAL NETWORK MODEL *******************

# Define the neural network model
class DrugResponseModel(nn.Module): # Input layer -> Hidden layer 1 -> Hidden layer 2 -> Hidden layer 3 -> Hidden layer 4 -> Hidden layer 5 -> Output layer

    # The input layer has 10 features
    # The hidden layer 1 has 128 neurons
    # The hidden layer 2 has 64 neurons
    # The hidden layer 3 has 32 neurons
    # The hidden layer 4 has 16 neurons
    # The hidden layer 5 has 8 neurons
    # The output layer has 1 neuron (scalar AUC value between 0 and 1)

    def __init__(self, input_features, hl1 = 128, hl2 = 64, hl3 = 32, hl4 = 16, hl5 = 8, output_feature = 1): # Funnel structure

        # 128 neurons derived from features
        # Funnels down to 64 nuerons, then 32 neurons

        super().__init__() # Inherit from nn.Module

        self.fc1 = nn.Linear(input_features, hl1) # From input layer to hidden layer 1
        self.bn1 = nn.BatchNorm1d(hl1) # Batch normalization layer for hidden layer 1
        self.dropout1 = nn.Dropout(0.3) # Dropout layer for hidden layer 1

        self.fc2 = nn.Linear(hl1, hl2) # From hidden layer 1 to hidden layer 2
        self.bn2 = nn.BatchNorm1d(hl2) # Batch normalization layer for hidden layer 2
        self.dropout2 = nn.Dropout(0.3) # Dropout layer for hidden layer 2

        self.fc3 = nn.Linear(hl2, hl3) # From hidden layer 2 to hidden layer 3
        self.bn3 = nn.BatchNorm1d(hl3) # Batch normalization layer for hidden layer 3
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hl3, hl4) # From hidden layer 3 to hidden layer 4
        self.bn4 = nn.BatchNorm1d(hl4) # Batch normalization layer for hidden layer 4
        self.dropout4 = nn.Dropout(0.3)

        self.fc5 = nn.Linear(hl4, hl5) # From hidden layer 4 to hidden layer 5
        self.bn5 = nn.BatchNorm1d(hl5) # Batch normalization layer for hidden layer 5

        self.output = nn.Linear(hl5, output_feature) # From hidden layer 5 to output layer
        self.activation = nn.LeakyReLU(0.1) # Leaky ReLU activation function for output layer, better for bio-med data

    # Helps foward pass the data through the network
    def forward(self, x):

        # Rectified Linear Unit (ReLU) activation function to introduce non-linearity
        x = self.activation(self.bn1(self.fc1(x))) 
        x = self.dropout1(x) # Dropout layer to prevent overfitting
        
        x = self.activation(self.bn2(self.fc2(x))) 
        x = self.dropout2(x) # Dropout layer to prevent overfitting

        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout3(x) # Dropout layer to prevent overfitting

        x = self.activation(self.bn4(self.fc4(x)))
        x = self.dropout4(x) # Dropout layer to prevent overfitting

        x = self.activation(self.bn5(self.fc5(x)))

        return self.output(x) # Output layer

# Pick a manual seed for randomization
torch.manual_seed(42)