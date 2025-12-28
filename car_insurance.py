# ============================================================
# CAR INSURANCE CLAIM PREDICTION - FULL PIPELINE (PRODUCTION)
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# ============================================================
# 1️⃣ Load Dataset
# ============================================================

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ============================================================
# 2️⃣ Drop ID Column
# ============================================================

test_policy_ids = test_df["policy_id"]
train_df.drop(columns=["policy_id"], inplace=True)
test_df.drop(columns=["policy_id"], inplace=True)

# ============================================================
# 3️⃣ Target & Feature Split
# ============================================================

y = train_df["is_claim"]
X = train_df.drop(columns=["is_claim"])

# ============================================================
# 4️⃣ Binary Columns (Yes / No)
# ============================================================

binary_cols = [
    'is_esc', 'is_adjustable_steering', 'is_tpms',
    'is_parking_sensors', 'is_parking_camera',
    'is_front_fog_lights', 'is_rear_window_wiper',
    'is_rear_window_washer', 'is_rear_window_defogger',
    'is_brake_assist', 'is_power_door_locks',
    'is_central_locking', 'is_power_steering',
    'is_driver_seat_height_adjustable',
    'is_day_night_rear_view_mirror',
    'is_ecw', 'is_speed_alert'
]

for col in binary_cols:
    X[col] = X[col].map({'Yes': 1, 'No': 0})
    test_df[col] = test_df[col].map({'Yes': 1, 'No': 0})

# ============================================================
# 5️⃣ Clean max_torque & max_power
# ============================================================

def extract_numeric(val):
    if pd.isna(val):
        return np.nan
    return float(''.join(ch for ch in str(val) if ch.isdigit() or ch == '.'))

for col in ["max_torque", "max_power"]:
    X[col] = X[col].apply(extract_numeric)
    test_df[col] = test_df[col].apply(extract_numeric)

# ============================================================
# 6️⃣ Column Groups
# ============================================================

categorical_ohe_cols = [
    "make","segment", "fuel_type", "steering_type",
    "engine_type", "rear_brakes_type", "transmission_type",
    "area_cluster", "model"
]

numeric_cols = X.drop(columns=categorical_ohe_cols).columns.tolist()

# ============================================================
# 7️⃣ Preprocessing Pipeline
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_ohe_cols)
    ]
)

# ============================================================
# 8️⃣ Train–Validation Split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================================
# 9️⃣ MLflow Setup
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Car_Insurance_Claim_Model")

# ============================================================
# 🔟 Models (PIPELINE-WRAPPED)
# ============================================================

models = [
    (
        "Logistic_Regression",
        Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])
    ),
    (
        "Random_Forest",
        Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=50,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ))
        ])
    ),
    (
        "XGBoost",
        Pipeline([
            ("preprocess", preprocessor),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                eval_metric="auc",
                random_state=42
            ))
        ])
    )
]

results = []

# ============================================================
# 1️⃣1️⃣ Training Loop
# ============================================================

for name, model in models:
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.3).astype(int)

        roc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)

        print(f"\n🚀 {name}")
        print("ROC-AUC:", roc)
        print("F1:", f1)

        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")

        os.makedirs("artifacts", exist_ok=True)
        cm_path = f"artifacts/{name}_cm.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        results.append({
            "model": name,
            "roc_auc": roc,
            "f1": f1,
            "run_id": mlflow.active_run().info.run_id
        })

# ============================================================
# 1️⃣2️⃣ Select Best Model
# ============================================================

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df["roc_auc"].idxmax()]

best_model_name = best_row["model"]
best_run_id = best_row["run_id"]

print("\n🏆 Best Model:", best_model_name)

# ============================================================
# 1️⃣3️⃣ Hyperparameter Tuning (Random Forest)
# ============================================================

rf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_param_grid = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [8, 10, 12, None],
    "model__min_samples_split": [10, 30, 50],
    "model__min_samples_leaf": [5, 10, 20]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_search = RandomizedSearchCV(
    rf_pipeline,
    rf_param_grid,
    n_iter=25,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_search.fit(X_train, y_train)

print("Best RF Params:", rf_search.best_params_)
print("Best CV ROC-AUC:", rf_search.best_score_)


with mlflow.start_run(run_name="RandomForest_Tuned"):

    tuned_rf = rf_search.best_estimator_

    y_val_prob = tuned_rf.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.3).astype(int)

    roc = roc_auc_score(y_val, y_val_prob)
    f1 = f1_score(y_val, y_val_pred)
    acc = accuracy_score(y_val, y_val_pred)

    # Log hyperparameters + CV score
    mlflow.log_params(rf_search.best_params_)
    mlflow.log_metric("cv_roc_auc", rf_search.best_score_)
    mlflow.log_metric("val_roc_auc", roc)
    mlflow.log_metric("val_f1", f1)
    mlflow.log_metric("val_accuracy", acc)

    # Log tuned pipeline model
    mlflow.sklearn.log_model(
        tuned_rf,
        artifact_path="model",
        registered_model_name="Car_Insurance_RF_Tuned_Production"
    )

    print("🚀 Tuned RF logged into MLflow & Registered")



# ============================================================
# 1️⃣4️⃣ Save Final Model (PIPELINE)
# ============================================================

best_model = rf_search.best_estimator_

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_car_insurance_model.pkl")

print("✅ Best pipeline model saved")

# ============================================================
# 1️⃣5️⃣ Test Prediction & Submission
# ============================================================

test_prob = best_model.predict_proba(test_df)[:, 1]
test_pred = (test_prob >= 0.3).astype(int)

submission = pd.DataFrame({
    "policy_id": test_policy_ids,
    "is_claim": test_pred
})

submission.to_csv("submission.csv", index=False)
print("📄 submission.csv generated")
