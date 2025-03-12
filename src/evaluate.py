import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.preprocess import load_and_preprocess_data

# Load test data
_, X_test, _, y_test = load_and_preprocess_data("dataset/diabetes.csv")

# Load trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Make predictions
y_pred = model.predict(X_test)

# Print evaluation metrics
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
