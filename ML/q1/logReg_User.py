# Import library
import numpy as np

# Step 1: Create dataset
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Step 2: Initialize parameters
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.1
iterations = 1000

# Step 3: Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 4: Training using Gradient Descent
for i in range(iterations):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    
    # Compute gradients
    dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
    db = (1 / len(X)) * np.sum(y_pred - y)
    
    # Update parameters
    weights -= learning_rate * dw
    bias -= learning_rate * db

# Step 5: Prediction function
def predict(X):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return [1 if i > 0.5 else 0 for i in y_pred]

# Step 6: Make predictions
predictions = predict(X)

# Step 7: Calculate accuracy
accuracy = np.mean(predictions == y)

# Step 8: Output results
print("Predicted values:", predictions)
print("Actual values:", y)
print("Accuracy:", accuracy)