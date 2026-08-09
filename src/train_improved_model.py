import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD IMPROVED DATASET
# ============================================================

project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root
    / "data"
    / "improved_developer_activity.csv"
)

df = pd.read_csv(data_path)

print("=" * 60)
print("AI LOST-CONTEXT DETECTOR - FINAL MODEL TRAINING")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["context_loss"])
y = df["context_loss"]

print(f"Features: {X.shape[1]}")


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 4. CREATE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
    ),
}


# ============================================================
# 5. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    model.fit(X_train, y_train)

    trained_models[model_name] = model

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

best_index = results_df["f1_score"].idxmax()

best_model_name = results_df.loc[
    best_index,
    "model",
]

best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print(f"Model: {best_model_name}")


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

random_forest = trained_models["Random Forest"]

feature_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": random_forest.feature_importances_,
    }
)

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False,
)

print("\n" + "=" * 60)
print("TOP FEATURES")
print("=" * 60)

print(
    feature_importance.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 9. SAVE OUTPUT FOLDER
# ============================================================

output_folder = project_root / "outputs"
output_folder.mkdir(exist_ok=True)


# ============================================================
# 10. SAVE MODEL COMPARISON
# ============================================================

results_path = (
    output_folder
    / "final_model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = (
    output_folder
    / "final_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False,
)


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

models_folder = project_root / "models"
models_folder.mkdir(exist_ok=True)

model_path = (
    models_folder
    / "context_loss_model.pkl"
)

joblib.dump(
    best_model,
    model_path,
)


# ============================================================
# 13. SAVE FEATURE NAMES
# ============================================================

feature_names_path = (
    models_folder
    / "feature_names.pkl"
)

joblib.dump(
    list(X.columns),
    feature_names_path,
)


# ============================================================
# 14. SAVE FINAL THRESHOLD
# ============================================================

threshold = 0.30

threshold_path = (
    models_folder
    / "threshold.pkl"
)

joblib.dump(
    threshold,
    threshold_path,
)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)

print("\nModel:")
print(model_path)

print("\nFeature names:")
print(feature_names_path)

print("\nThreshold:")
print(threshold_path)

print("\nFinal threshold: 0.30")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)