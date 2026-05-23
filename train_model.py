import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
data = pd.read_csv("data/gestures.csv")

# Features and labels
X = data.drop("gesture", axis=1)
y = data["gesture"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")