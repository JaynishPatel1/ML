import sys
import os

# Ensure Python finds the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_and_preprocess_data
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data("dataset/diabetes.csv")

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open("best_model.pkl", "wb") as file:
    pickle.dump(model, file)
