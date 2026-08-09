import pandas as pd
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
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. LOAD DATA
# ============================================================

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "developer_activity.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("AI LOST-CONTEXT DETECTOR - MODEL TRAINING")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

feature_columns = [
    "duration_minutes",
    "break_minutes",
    "context_switch",
    "project",
    "activity_type",
    "task_status",
]

target_column = "context_loss"

X = df[feature_columns]
y = df[target_column]


# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================

numerical_features = [
    "duration_minutes",
    "break_minutes",
    "context_switch",
]

categorical_features = [
    "project",
    "activity_type",
    "task_status",
]

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

encoded_data = encoder.fit_transform(
    X[categorical_features]
)

encoded_columns = encoder.get_feature_names_out(
    categorical_features
)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoded_columns,
    index=X.index,
)

X_processed = pd.concat(
    [
        X[numerical_features],
        encoded_df,
    ],
    axis=1,
)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. CREATE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    ),
}


# ============================================================
# 6. TRAIN AND EVALUATE
# ============================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

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
# 7. MODEL COMPARISON
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
# 8. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["f1_score"].idxmax(),
    "model",
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Selected model: {best_model_name}")
print("Selection metric: F1 Score")


# ============================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

random_forest = models["Random Forest"]

feature_importance = pd.DataFrame(
    {
        "feature": X_processed.columns,
        "importance": random_forest.feature_importances_,
    }
)

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False,
)

print("\n" + "=" * 60)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 10. SAVE MODEL RESULTS
# ============================================================

output_folder = project_root / "outputs"
output_folder.mkdir(exist_ok=True)

results_path = output_folder / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
)

importance_path = output_folder / "feature_importance.csv"

feature_importance.to_csv(
    importance_path,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print(f"\nModel comparison saved to:")
print(results_path)

print(f"\nFeature importance saved to:")
print(importance_path)