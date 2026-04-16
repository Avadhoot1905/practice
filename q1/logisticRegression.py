# Import libraries
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load internal dataset
data = load_breast_cancer()

# Features and target
X = data.data
y = data.target

# Step 2: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 3: Create Logistic Regression model
model = LogisticRegression(max_iter=10000)  # increased iterations for convergence

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Step 7: Output results
print("Dataset shape:", X.shape)

print("\nActual Output (y_test):")
print(y_test)

print("\nPredicted Output (y_pred):")
print(y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(cm)