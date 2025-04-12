import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

# Features in dataset 
FEATURES = [
    "CGPA", "Major Projects", "Workshops/Certifications", "Mini Projects",
    "Skills", "Communication Skill Rating", "Internship", "Hackathon",
    "12th Percentage", "10th Percentage", "backlogs"
]

def load_and_preprocess_data():
    df = pd.read_csv("predictions_data.csv")

    X = df[FEATURES]
    y_placement = df["PlacementStatus"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_placement, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Saving scalar model for future use
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, df, scaler

def train_placement_model(X_train, X_test, y_train, y_test):
    placement_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, max_features="sqrt", random_state=42
    )
    placement_model.fit(X_train, y_train)

    # Evaluating the  placement model
    y_pred = placement_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, placement_model.predict_proba(X_test)[:, 1])

    print("\nPlacement Prediction Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Saving placement model for future use
    joblib.dump(placement_model, "models/placement_model.pkl")
    return placement_model

def train_salary_model(df, scaler):
    placed_df = df[df["PlacementStatus"] == 1]

    X_salary = placed_df[FEATURES]
    y_salary = placed_df["salary"]

    X_salary_scaled = scaler.transform(X_salary)

    salary_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, max_features="sqrt", random_state=42
    )
    salary_model.fit(X_salary_scaled, y_salary)

    # Evaluating the salary model
    y_pred_salary = salary_model.predict(X_salary_scaled)
    r2 = r2_score(y_salary, y_pred_salary)
    mae = mean_absolute_error(y_salary, y_pred_salary)
    rmse = np.sqrt(mean_squared_error(y_salary, y_pred_salary))

    print("\nSalary Prediction Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Saving salary model for future use
    joblib.dump(salary_model, "models/salary_model.pkl")
    return salary_model

def main():
    
    print("Training Models...")

    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, df, scaler = load_and_preprocess_data()

    # Train and save models
    train_placement_model(X_train_scaled, X_test_scaled, y_train, y_test)
    train_salary_model(df, scaler)

    print("\nModels trained & saved successfully!")

if __name__ == "__main__":
    main()
