import pandas as pd
from pathlib import Path

# -----------------------------------
# Load Dataset
# -----------------------------------

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "developer_activity.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("AI LOST-CONTEXT DETECTOR - DATA EXPLORATION")
print("=" * 60)

# -----------------------------------
# 1. Basic Information
# -----------------------------------

print("\n1. DATASET SHAPE")
print("-" * 40)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# -----------------------------------
# 2. Column Information
# -----------------------------------

print("\n2. COLUMNS")
print("-" * 40)

for column in df.columns:
    print(f"- {column}")

# -----------------------------------
# 3. Missing Values
# -----------------------------------

print("\n3. MISSING VALUES")
print("-" * 40)

missing_values = df.isnull().sum()

print(missing_values)

# -----------------------------------
# 4. Duplicate Records
# -----------------------------------

print("\n4. DUPLICATE RECORDS")
print("-" * 40)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")

# -----------------------------------
# 5. Data Types
# -----------------------------------

print("\n5. DATA TYPES")
print("-" * 40)

print(df.dtypes)

# -----------------------------------
# 6. Context Loss Distribution
# -----------------------------------

print("\n6. CONTEXT LOSS DISTRIBUTION")
print("-" * 40)

context_loss = df["context_loss"].value_counts()

print(context_loss)

# -----------------------------------
# 7. Context Switch Distribution
# -----------------------------------

print("\n7. CONTEXT SWITCH DISTRIBUTION")
print("-" * 40)

context_switch = df["context_switch"].value_counts()

print(context_switch)

# -----------------------------------
# 8. Project Distribution
# -----------------------------------

print("\n8. PROJECT DISTRIBUTION")
print("-" * 40)

print(df["project"].value_counts())

# -----------------------------------
# 9. Activity Distribution
# -----------------------------------

print("\n9. ACTIVITY TYPE DISTRIBUTION")
print("-" * 40)

print(df["activity_type"].value_counts())

# -----------------------------------
# 10. Task Status Distribution
# -----------------------------------

print("\n10. TASK STATUS DISTRIBUTION")
print("-" * 40)

print(df["task_status"].value_counts())

# -----------------------------------
# 11. Numerical Statistics
# -----------------------------------

print("\n11. NUMERICAL STATISTICS")
print("-" * 40)

print(df.describe())

# -----------------------------------
# 12. Context Loss Percentage
# -----------------------------------

loss_percentage = df["context_loss"].mean() * 100

print("\n12. CONTEXT LOSS RATE")
print("-" * 40)

print(f"Context Loss Rate: {loss_percentage:.2f}%")

# -----------------------------------
# 13. Context Switch Percentage
# -----------------------------------

switch_percentage = df["context_switch"].mean() * 100

print("\n13. CONTEXT SWITCH RATE")
print("-" * 40)

print(f"Context Switch Rate: {switch_percentage:.2f}%")

# -----------------------------------
# 14. Average Break Duration
# -----------------------------------

average_break = df["break_minutes"].mean()

print("\n14. AVERAGE BREAK DURATION")
print("-" * 40)

print(f"Average Break: {average_break:.2f} minutes")

# -----------------------------------
# 15. Context Loss by Task Status
# -----------------------------------

print("\n15. CONTEXT LOSS BY TASK STATUS")
print("-" * 40)

loss_by_status = pd.crosstab(
    df["task_status"],
    df["context_loss"],
    normalize="index"
) * 100

print(loss_by_status.round(2))

# -----------------------------------
# 16. Context Loss by Project
# -----------------------------------

print("\n16. CONTEXT LOSS BY PROJECT")
print("-" * 40)

loss_by_project = df.groupby("project")["context_loss"].mean() * 100

print(loss_by_project.round(2))

# -----------------------------------
# 17. Context Loss by Context Switch
# -----------------------------------

print("\n17. CONTEXT LOSS BY CONTEXT SWITCH")
print("-" * 40)

loss_by_switch = df.groupby("context_switch")["context_loss"].mean() * 100

print(loss_by_switch.round(2))

# -----------------------------------
# 18. Context Loss by Break Duration
# -----------------------------------

print("\n18. CONTEXT LOSS BY BREAK DURATION")
print("-" * 40)

df["break_category"] = pd.cut(
    df["break_minutes"],
    bins=[0, 15, 30, 60, 120, float("inf")],
    labels=[
        "0-15 min",
        "16-30 min",
        "31-60 min",
        "61-120 min",
        "120+ min",
    ]
)

loss_by_break = df.groupby(
    "break_category",
    observed=True
)["context_loss"].mean() * 100

print(loss_by_break.round(2))

# -----------------------------------
# Final Summary
# -----------------------------------

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETED")
print("=" * 60)

print(f"""
Dataset:
    Records              : {len(df)}
    Features             : {len(df.columns)}

Context Loss:
    Loss Rate            : {loss_percentage:.2f}%

Context Switching:
    Switch Rate          : {switch_percentage:.2f}%

Average Break:
    {average_break:.2f} minutes

Missing Values:
    {df.isnull().sum().sum()}

Duplicate Rows:
    {duplicates}
""")