import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

NUM_RECORDS = 2000
START_DATE = datetime(2026, 7, 1, 9, 0)

projects = {
    "AI Chatbot": {
        "files": ["auth.py", "chat.py", "database.py", "api.py"],
        "tasks": ["Authentication", "Chat System", "Database", "API Development"],
    },
    "Portfolio Website": {
        "files": ["App.jsx", "Navbar.jsx", "Projects.jsx", "styles.css"],
        "tasks": ["Frontend", "UI Design", "Responsive Design", "Deployment"],
    },
    "ML Project": {
        "files": ["train.py", "preprocess.py", "model.py", "evaluation.py"],
        "tasks": ["Data Processing", "Model Training", "Feature Engineering", "Evaluation"],
    },
    "College Project": {
        "files": ["main.py", "database.py", "routes.py", "README.md"],
        "tasks": ["Backend", "Database", "API", "Documentation"],
    },
}

activity_types = [
    "coding",
    "debugging",
    "testing",
    "research",
    "documentation",
]

task_statuses = ["completed", "in_progress", "incomplete"]

commit_templates = [
    "implement {}",
    "fix {} bug",
    "update {}",
    "refactor {}",
    "add {}",
    "test {}",
    "debug {}",
]

random.seed(42)


# -----------------------------
# Helper functions
# -----------------------------

def generate_commit_message(task, activity_type):
    template = random.choice(commit_templates)
    return template.format(task.lower())


def choose_activity():
    return random.choice(activity_types)


def calculate_context_loss(
    break_minutes,
    context_switch,
    duration_minutes,
    task_status,
    recent_switches,
):
    """
    Create a synthetic context-loss label.

    Higher probability when:
    - the break is long
    - the user switched context
    - the previous task was unfinished
    - many switches happened recently
    """

    score = 0

    if break_minutes >= 120:
        score += 3
    elif break_minutes >= 60:
        score += 2
    elif break_minutes >= 30:
        score += 1

    if context_switch:
        score += 2

    if task_status == "incomplete":
        score += 3
    elif task_status == "in_progress":
        score += 1

    if recent_switches >= 3:
        score += 2
    elif recent_switches >= 2:
        score += 1

    if duration_minutes < 15:
        score += 1

    probability = min(score / 10, 0.95)

    return int(random.random() < probability)


# -----------------------------
# Dataset generation
# -----------------------------

records = []

current_time = START_DATE
previous_project = None
recent_projects = []
last_activity_time = current_time

for _ in range(NUM_RECORDS):

    # Occasionally create a longer break
    if random.random() < 0.12:
        break_minutes = random.randint(60, 240)
    else:
        break_minutes = random.randint(5, 35)

    current_time = current_time + timedelta(minutes=break_minutes)

    # Decide whether to stay on the same project
    if previous_project and random.random() < 0.70:
        project = previous_project
    else:
        project = random.choice(list(projects.keys()))

    context_switch = int(
        previous_project is not None and project != previous_project
    )

    if context_switch:
        recent_projects.append(project)

    recent_projects = recent_projects[-5:]

    recent_switches = len(set(recent_projects))

    project_info = projects[project]

    file = random.choice(project_info["files"])
    task = random.choice(project_info["tasks"])
    activity_type = choose_activity()

    duration_minutes = random.randint(10, 90)

    # Make incomplete work more likely during debugging
    if activity_type == "debugging":
        task_status = random.choices(
            task_statuses,
            weights=[20, 45, 35],
        )[0]
    else:
        task_status = random.choices(
            task_statuses,
            weights=[50, 35, 15],
        )[0]

    commit_message = generate_commit_message(task, activity_type)

    context_loss = calculate_context_loss(
        break_minutes=break_minutes,
        context_switch=context_switch,
        duration_minutes=duration_minutes,
        task_status=task_status,
        recent_switches=recent_switches,
    )

    records.append(
        {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "project": project,
            "file": file,
            "activity_type": activity_type,
            "duration_minutes": duration_minutes,
            "commit_message": commit_message,
            "task": task,
            "task_status": task_status,
            "previous_project": previous_project if previous_project else "None",
            "context_switch": context_switch,
            "break_minutes": break_minutes,
            "context_loss": context_loss,
        }
    )

    previous_project = project
    last_activity_time = current_time


# -----------------------------
# Save dataset
# -----------------------------

df = pd.DataFrame(records)

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data"
data_folder.mkdir(exist_ok=True)

output_file = data_folder / "developer_activity.csv"

df.to_csv(output_file, index=False)


# -----------------------------
# Display summary
# -----------------------------

print("=" * 60)
print("AI LOST-CONTEXT DETECTOR")
print("Synthetic Dataset Generator")
print("=" * 60)

print(f"\nRecords generated: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Dataset saved to:\n{output_file}")

print("\nContext Loss Distribution:")
print(df["context_loss"].value_counts())

print("\nContext Switch Distribution:")
print(df["context_switch"].value_counts())

print("\nProjects:")
print(df["project"].value_counts())

print("\nDataset preview:")
print(df.head(10))

print("\nDataset generation completed successfully!")