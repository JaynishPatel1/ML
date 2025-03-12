import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # ✅ Import SMOTE

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert categorical 'Outcome' column (if needed)
    df['Outcome'] = df['Outcome'].replace({'Diabetic': 1, 'Non Diabetic': 0})

    # Check class balance before SMOTE
    print("\n🔍 Class Distribution Before Balancing:")
    print(df["Outcome"].value_counts())

    # Replace zero values with NaN in specific columns
    columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_with_zero] = df[columns_with_zero].replace(0, np.nan)

    # Fill missing values with column mean
    df.fillna(df.mean(), inplace=True)

    # Separate features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # ✅ Define SMOTE before using it
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print("\n✅ Class Distribution After Balancing:")
    print(pd.Series(y_train_balanced).value_counts())

    return X_train_balanced, X_test, y_train_balanced, y_test
