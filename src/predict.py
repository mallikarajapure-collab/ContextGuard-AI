import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

project_root = Path(__file__).resolve().parent.parent

models_folder = project_root / "models"

model = joblib.load(
    models_folder / "context_loss_model.pkl"
)

feature_names = joblib.load(
    models_folder / "feature_names.pkl"
)

threshold = joblib.load(
    models_folder / "threshold.pkl"
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 60)
print("AI LOST-CONTEXT DETECTOR")
print("=" * 60)

print("\nEnter developer activity details.")
print("The system will estimate context-loss risk.\n")


# ============================================================
# 3. USER INPUT
# ============================================================

project = input(
    "Project "
    "(AI Chatbot / College Project / ML Project / Portfolio Website): "
).strip()

activity_type = input(
    "Activity "
    "(coding / debugging / documentation / research / testing): "
).strip()

task_status = input(
    "Task status "
    "(completed / in_progress / incomplete): "
).strip()

duration_minutes = float(
    input("Activity duration (minutes): ")
)

break_minutes = float(
    input("Break duration (minutes): ")
)

context_switch_input = input(
    "Did you switch projects recently? (yes/no): "
).strip().lower()

context_switch = (
    1 if context_switch_input == "yes" else 0
)


# ============================================================
# 4. ENGINEERED FEATURES
# ============================================================

long_break = (
    1 if break_minutes > 30 else 0
)

very_long_break = (
    1 if break_minutes > 60 else 0
)

unfinished_task = (
    1 if task_status == "incomplete" else 0
)

in_progress_task = (
    1 if task_status == "in_progress" else 0
)

short_activity = (
    1 if duration_minutes < 30 else 0
)

long_activity = (
    1 if duration_minutes > 60 else 0
)

same_project = (
    0 if context_switch == 1 else 1
)

recent_switch_count = context_switch


# ============================================================
# 5. CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame(
    {
        "duration_minutes": [duration_minutes],
        "break_minutes": [break_minutes],
        "context_switch": [context_switch],

        "project": [project],

        "activity_type": [activity_type],

        "task_status": [task_status],

        "long_break": [long_break],
        "very_long_break": [very_long_break],
        "unfinished_task": [unfinished_task],
        "in_progress_task": [in_progress_task],
        "short_activity": [short_activity],
        "long_activity": [long_activity],
        "same_project": [same_project],
        "recent_switch_count": [recent_switch_count],
    }
)


# ============================================================
# 6. ONE-HOT ENCODING
# ============================================================

input_data = pd.get_dummies(
    input_data,
    columns=[
        "project",
        "activity_type",
        "task_status",
    ],
)


# ============================================================
# 7. MATCH TRAINING FEATURES
# ============================================================

input_data = input_data.reindex(
    columns=feature_names,
    fill_value=0,
)


# ============================================================
# 8. PREDICT PROBABILITY
# ============================================================

probability = model.predict_proba(
    input_data
)[0][1]

prediction = (
    1 if probability >= threshold else 0
)


# ============================================================
# 9. DETERMINE RISK LEVEL
# ============================================================

if probability >= 0.70:

    risk_level = "HIGH"

elif probability >= 0.40:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# 10. CONTEXT LOSS ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CONTEXT LOSS ANALYSIS")
print("=" * 60)

print(
    f"\nContext Loss Probability : "
    f"{probability * 100:.2f}%"
)

print(
    f"Risk Level               : "
    f"{risk_level}"
)

print(
    f"Decision Threshold       : "
    f"{threshold:.2f}"
)


# ============================================================
# 11. MAIN PREDICTION MESSAGE
# ============================================================

if prediction == 1:

    print(
        "\n🚨 WARNING: Possible context loss detected!"
    )

else:

    print(
        "\n✅ Context appears stable."
    )


# ============================================================
# 12. BEHAVIOR SIGNALS
# ============================================================

print("\n" + "=" * 60)
print("BEHAVIOR SIGNALS")
print("=" * 60)


signals = []


if break_minutes > 60:

    signals.append(
        "⚠ Very long break detected."
    )

elif break_minutes > 30:

    signals.append(
        "⚠ Long break detected."
    )

else:

    signals.append(
        "✓ Break duration is relatively short."
    )


if task_status == "incomplete":

    signals.append(
        "⚠ Task is incomplete."
    )

elif task_status == "in_progress":

    signals.append(
        "• Task is currently in progress."
    )

else:

    signals.append(
        "✓ Task completed."
    )


if context_switch == 1:

    signals.append(
        "⚠ Recent project switch detected."
    )

else:

    signals.append(
        "✓ No recent project switch detected."
    )


if short_activity:

    signals.append(
        "⚠ Short activity session detected."
    )


if long_activity:

    signals.append(
        "• Long activity session detected."
    )


for signal in signals:

    print(signal)


# ============================================================
# 13. FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)

if risk_level == "HIGH":

    print(
        "\n💡 Consider reviewing your last task, "
        "recent changes, and unfinished work before continuing."
    )

elif risk_level == "MEDIUM":

    print(
        "\n💡 A quick review of your previous task "
        "may help you regain context."
    )

else:

    print(
        "\n💡 Your current activity pattern appears stable. "
        "You can continue working."
    )


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)