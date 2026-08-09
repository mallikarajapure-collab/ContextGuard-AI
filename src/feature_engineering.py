import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "developer_activity.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print(f"\nOriginal dataset shape: {df.shape}")


# -----------------------------------
# 2. Select Features
# -----------------------------------

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

print("\nSelected features:")
for feature in feature_columns:
    print(f"- {feature}")

print(f"\nTarget: {target_column}")


# -----------------------------------
# 3. Separate Numerical & Categorical
# -----------------------------------

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


# -----------------------------------
# 4. Encode Categorical Features
# -----------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
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


# -----------------------------------
# 5. Combine Features
# -----------------------------------

X_processed = pd.concat(
    [
        X[numerical_features],
        encoded_df,
    ],
    axis=1,
)

print("\nProcessed feature shape:")
print(X_processed.shape)

print("\nProcessed features:")
for feature in X_processed.columns:
    print(f"- {feature}")


# -----------------------------------
# 6. Train-Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# -----------------------------------
# 7. Save Processed Data
# -----------------------------------

processed_data = X_processed.copy()
processed_data["context_loss"] = y.values

output_path = (
    project_root
    / "data"
    / "processed_developer_activity.csv"
)

processed_data.to_csv(
    output_path,
    index=False
)

print("\nProcessed dataset saved to:")
print(output_path)


# -----------------------------------
# Final Message
# -----------------------------------

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("""
Raw categorical data
        ↓
One-Hot Encoding
        ↓
Numerical ML Features
        ↓
Train/Test Split
        ↓
Ready for ML Model
""")