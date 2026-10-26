<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:022C22,35:047857,70:0D9488,100:99F6E4&height=235&section=header&text=CONTEXTGUARD%20AI&fontSize=42&fontColor=ECFDF5&fontAlignY=36&desc=Protect%20the%20flow%20of%20development&descAlignY=60&descSize=16" width="100%" alt="ContextGuard AI — Protect the flow of development" />

<p><strong>A machine-learning prototype for estimating developer context-loss risk.</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-064E3B?style=for-the-badge&logo=python&logoColor=99F6E4" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-047857?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Streamlit-0F766E?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Model%20Evaluation-14B8A6?style=for-the-badge&logoColor=white" alt="Model evaluation" />
</p>

</div>

---

## 🌿 Overview

**ContextGuard AI** explores whether developer activity patterns can help estimate the risk of losing track of a task or its surrounding context. The project turns activity-related inputs—such as break and work-session duration, task status, and recent project switching—into a model score, a risk category, behavioral signals, and suggested recovery steps.

It combines a trained scikit-learn classifier with a Streamlit interface, feature-engineering scripts, model-evaluation outputs, and saved model artifacts.

> **Important:** This is an experimental decision-support prototype, not a clinical or psychological assessment and not a validated measure of developer productivity. Its predictions depend on the dataset and features used during development.

## ✨ What it includes

<table>
<tr>
<td width="50%" valign="top">

### 🧭 Context-risk overview
- Estimate context-loss probability from the entered activity profile.
- Map the score to a risk category using the configured threshold.
- Surface activity patterns that may deserve a closer look.

</td>
<td width="50%" valign="top">

### 🔎 Activity signals
- Consider break and activity duration.
- Include task completion status.
- Account for recent project switching and related encoded features.
- Present suggestions for regaining task context.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📊 Model evaluation
- Compare Logistic Regression and Random Forest.
- Review precision, recall, F1 score, and accuracy.
- Explore feature-importance and confusion-matrix views.
- Keep evaluation outputs alongside the project.

</td>
<td width="50%" valign="top">

### 🧪 ML workflow
- Explore and prepare the activity dataset.
- Engineer and encode model features.
- Train candidate classifiers and tune the decision threshold.
- Load the saved model artifacts in the application.

</td>
</tr>
</table>

## 🧠 Model results

The repository's reported evaluation compares two classifiers on a test set of 400 samples.

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 65.50% | 59.43% | 39.87% | 47.73% |
| Random Forest | 64.25% | 57.28% | 37.34% | 45.21% |

**Selected model:** Logistic Regression, based on the higher F1 score among these evaluated candidates.

The project reports 1,600 training samples and 400 testing samples (2,000 total), with a tuned classification threshold of **0.30** in the final prediction setup.

These are project-reported results, not a guarantee of real-world performance. In particular, performance on generated or limited development data may not transfer to real developer workflows; validation on representative, independently collected data would be needed before relying on predictions.

## 🔄 How the pipeline works

```text
Developer activity inputs
          |
          v
Feature engineering and encoding
          |
          v
Saved Logistic Regression model
          |
          v
Estimated context-loss probability
          |
          v
Threshold-based risk category
          |
          v
Behavioral signals and recovery suggestions
```

## 🧰 Technology stack

| Technology | Purpose |
|---|---|
| Python | Application and ML workflow |
| Pandas, NumPy | Data preparation and numerical operations |
| scikit-learn | Feature processing, classification, and evaluation |
| Joblib | Saving and loading model artifacts |
| Streamlit | Interactive dashboard |
| Jupyter Notebook | Exploratory data analysis |

## 🚀 Run locally

### 1. Clone the repository

```bash
git clone https://github.com/mallikarajapure-collab/ContextGuard-AI.git
cd ContextGuard-AI
```

### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser. The repository includes saved model artifacts under `models/`, which the application can use for predictions.

## 🗂️ Project structure

```text
ContextGuard-AI/
├── app.py
├── main.py
├── requirements.txt
├── data/
│   ├── developer_activity.csv
│   ├── improved_developer_activity.csv
│   └── processed_developer_activity.csv
├── models/
│   ├── context_loss_model.pkl
│   ├── feature_names.pkl
│   └── threshold.pkl
├── notebooks/
│   └── 01_data_exploration.ipynb
├── outputs/
│   ├── feature_importance.csv
│   ├── final_feature_importance.csv
│   ├── final_model_comparison.csv
│   └── threshold_results.csv
└── src/
    ├── explore_data.py
    ├── feature_engineering.py
    ├── generate_dataset.py
    ├── improved_features.py
    ├── predict.py
    ├── train_model.py
    ├── train_improved_model.py
    └── tune_threshold.py
```

## ⚠️ Scope and responsible use

- A risk score is an estimate from the trained model, not proof that someone has lost context.
- Long breaks, project switching, or unfinished tasks can have many legitimate explanations; avoid using these signals to judge an individual.
- Reported metrics reflect the evaluation setup documented in this repository and may change with data, preprocessing, or threshold changes.
- The current project does not provide live IDE tracking or a VS Code extension. Those are possible future additions, not current features.
- Keep the environment, input data, and model artifacts aligned when retraining or changing feature engineering.

## 🛣️ Potential next steps

- Validate with larger, diverse, and representative activity data.
- Add an opt-in VS Code integration for context-aware reminders.
- Track user-controlled history and context-recovery outcomes.
- Improve calibration and report uncertainty alongside risk scores.
- Add repeatable automated tests for preprocessing and prediction.

## 👩‍💻 Author

**Mallika Rajapure**  
MCA Student · Machine Learning & Software Development

<div align="center">

**ContextGuard AI · Support the workflow, not surveillance.**

</div>
