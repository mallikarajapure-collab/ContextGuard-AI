import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


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
print("THRESHOLD TUNING")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["context_loss"])
y = df["context_loss"]


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


# ============================================================
# 4. TRAIN LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(X_train, y_train)


# ============================================================
# 5. GET PREDICTED PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

results = []

for threshold in thresholds:

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

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
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("THRESHOLD RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 8. BEST F1 THRESHOLD
# ============================================================

best_f1_index = results_df["f1_score"].idxmax()

best_f1_row = results_df.loc[
    best_f1_index
]

print("\n" + "=" * 60)
print("BEST F1 THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best_f1_row['threshold']:.2f}"
)

print(
    f"Accuracy  : {best_f1_row['accuracy']:.4f}"
)

print(
    f"Precision : {best_f1_row['precision']:.4f}"
)

print(
    f"Recall    : {best_f1_row['recall']:.4f}"
)

print(
    f"F1 Score  : {best_f1_row['f1_score']:.4f}"
)


# ============================================================
# 9. BEST RECALL THRESHOLD
# ============================================================

best_recall_index = results_df["recall"].idxmax()

best_recall_row = results_df.loc[
    best_recall_index
]

print("\n" + "=" * 60)
print("BEST RECALL THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best_recall_row['threshold']:.2f}"
)

print(
    f"Accuracy  : {best_recall_row['accuracy']:.4f}"
)

print(
    f"Precision : {best_recall_row['precision']:.4f}"
)

print(
    f"Recall    : {best_recall_row['recall']:.4f}"
)

print(
    f"F1 Score  : {best_recall_row['f1_score']:.4f}"
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_folder = project_root / "outputs"
output_folder.mkdir(exist_ok=True)

output_path = (
    output_folder
    / "threshold_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD TUNING COMPLETED")
print("=" * 60)

print("\nResults saved to:")
print(output_path)
