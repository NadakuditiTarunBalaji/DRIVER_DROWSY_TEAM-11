# Description: This script trains a simple neural network to detect drowsiness using the EAR and MAR features.
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv("C:\\Users\\Tarun\\OneDrive\\Desktop\\New folder\\drowsiness_data_.csv")

# Split the dataset into features (X) and labels (y)
# We will use EAR and MAR as features
X = data[["EAR", "MAR"]].values
#  The "Label" column contains the labels (0: Not Drowsy, 1: Drowsy)
y = data["Label"].values

# Split into training and test sets
# We will use 80% of the data for training and 20% for testing (validation) the model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (optional, but recommended)
scaler = StandardScaler()
# Standardization is a common requirement for neural networks
# It ensures that each feature contributes equally to the computation of the output 

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Keras model
# We will use a simple feedforward neural network with 2 hidden layers (32 and 16 units) and a sigmoid output layer
model = Sequential()
# The input layer has 2 units (EAR and MAR) and uses the ReLU activation function 
model.add(Dense(32, input_dim=2, activation="relu"))  # 2 inputs: EAR and MAR
# The first hidden layer has 32 units and uses the ReLU activation function
# The second hidden layer has 16 units and uses the ReLU activation function 
model.add(Dense(16, activation="relu"))
# The output layer has 1 unit and uses the sigmoid activation function 
model.add(Dense(1, activation="sigmoid"))  # Binary classification: Drowsy (1) or Not Drowsy (0)

# Compile the model
# We will use binary crossentropy as the loss function, Adam optimizer, and accuracy as the metric 
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
# We will train the model for 50 epochs with a batch size of 8 
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model
# We will evaluate the model on the test set and print the test accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
# We will save the trained model as a Keras .keras file
model.save("drowsiness_detection_model1.keras")