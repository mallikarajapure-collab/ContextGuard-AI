import pandas as pd
from pathlib import Path


# ============================================================
# 1. LOAD DATA
# ============================================================

project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "developer_activity.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("IMPROVED FEATURE ENGINEERING")
print("=" * 60)

print(f"\nOriginal dataset shape: {df.shape}")


# ============================================================
# 2. CREATE BEHAVIOR-BASED FEATURES
# ============================================================

# Long break indicator
df["long_break"] = (
    df["break_minutes"] >= 60
).astype(int)


# Very long break indicator
df["very_long_break"] = (
    df["break_minutes"] >= 120
).astype(int)


# Unfinished task indicator
df["unfinished_task"] = (
    df["task_status"] == "incomplete"
).astype(int)


# Task currently in progress
df["in_progress_task"] = (
    df["task_status"] == "in_progress"
).astype(int)


# Short activity indicator
df["short_activity"] = (
    df["duration_minutes"] <= 20
).astype(int)


# Long activity indicator
df["long_activity"] = (
    df["duration_minutes"] >= 60
).astype(int)


# Same project as previous activity
df["same_project"] = (
    df["project"] == df["previous_project"]
).astype(int)


# ============================================================
# 3. RECENT CONTEXT SWITCH COUNT
# ============================================================

# Count context switches within the previous 5 records.
# shift() ensures the current record itself is not used.

df["recent_switch_count"] = (
    df["context_switch"]
    .shift(1)
    .rolling(window=5, min_periods=1)
    .sum()
    .fillna(0)
)


# ============================================================
# 4. DISPLAY NEW FEATURES
# ============================================================

new_features = [
    "long_break",
    "very_long_break",
    "unfinished_task",
    "in_progress_task",
    "short_activity",
    "long_activity",
    "same_project",
    "recent_switch_count",
]

print("\nNew engineered features:")

for feature in new_features:
    print(f"- {feature}")


# ============================================================
# 5. CHECK FEATURE RELATIONSHIPS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ANALYSIS")
print("=" * 60)

for feature in new_features:

    if df[feature].nunique() > 1:

        relationship = (
            df.groupby(feature)["context_loss"]
            .mean()
            .mul(100)
            .round(2)
        )

        print(f"\n{feature}:")
        print(relationship)


# ============================================================
# 6. SELECT ML FEATURES
# ============================================================

numerical_features = [
    "duration_minutes",
    "break_minutes",
    "context_switch",
    "long_break",
    "very_long_break",
    "unfinished_task",
    "in_progress_task",
    "short_activity",
    "long_activity",
    "same_project",
    "recent_switch_count",
]

categorical_features = [
    "project",
    "activity_type",
]


# ============================================================
# 7. ONE-HOT ENCODING
# ============================================================

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

encoded_data = encoder.fit_transform(
    df[categorical_features]
)

encoded_columns = encoder.get_feature_names_out(
    categorical_features
)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoded_columns,
    index=df.index,
)


# ============================================================
# 8. BUILD FINAL FEATURE DATASET
# ============================================================

X = pd.concat(
    [
        df[numerical_features],
        encoded_df,
    ],
    axis=1,
)

y = df["context_loss"]


print("\n" + "=" * 60)
print("FINAL FEATURE DATASET")
print("=" * 60)

print(f"\nRows    : {X.shape[0]}")
print(f"Features: {X.shape[1]}")


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

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
# 10. SAVE IMPROVED DATASET
# ============================================================

processed_data = X.copy()
processed_data["context_loss"] = y.values

output_path = (
    project_root
    / "data"
    / "improved_developer_activity.csv"
)

processed_data.to_csv(
    output_path,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nSaved dataset:")
print(output_path)

print("""
Raw Developer Activity
        ↓
Behavior Features
        ↓
One-Hot Encoding
        ↓
Improved ML Features
        ↓
Train/Test Split
        ↓
Ready for Improved Model
""")