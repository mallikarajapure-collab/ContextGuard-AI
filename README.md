<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:064E3B,50:059669,100:5EEAD4&height=210&section=header&text=CONTEXTGUARD%20AI&fontSize=40&fontColor=FFFFFF&fontAlignY=38&desc=Developer%20Context%20Loss%20Prediction&descAlignY=60&descSize=15" width="100%" />

</div>

---

# 🧠 ContextGuard AI

### Developer Context Loss Prediction System

ContextGuard AI is a Machine Learning-based system that predicts the **risk of a developer losing development context** based on activity patterns such as break duration, activity duration, project switching, task status, and recent development behavior.

The system analyzes developer activity and provides a **context-loss probability, risk level, behavioral signals, and recommendations** through an interactive Streamlit dashboard.

---

## 🚀 Features

* 🧠 Machine Learning-based context-loss prediction
* 📊 Context-loss probability score
* 🚨 Low / Medium / High risk classification
* 🔎 Developer behavior signal detection
* 🔄 Project-switch detection
* ⏱️ Activity and break duration analysis
* 📈 Feature importance visualization
* ⚖️ Logistic Regression vs Random Forest comparison
* 🎯 Confusion matrix analysis
* 💡 Context recovery recommendations
* 🖥️ Interactive Streamlit dashboard

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**

---

## 🤖 Machine Learning

Two classification models were evaluated:

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   65.50% |    59.43% | 39.87% |   47.73% |
| Random Forest       |   64.25% |    57.28% | 37.34% |   45.21% |

### Final Model

**Logistic Regression** was selected as the final model because it achieved the highest F1 score among the evaluated models.

---

## 📊 Important Features

The most influential behavioral features included:

1. Break duration
2. Activity duration
3. Recent project switching
4. Unfinished task status
5. In-progress task status
6. Context switching
7. Same-project activity

These signals help the system estimate the likelihood of context loss.

---

## 🔄 How It Works

```text
Developer Activity
        ↓
Feature Engineering
        ↓
One-Hot Encoding
        ↓
Machine Learning Model
        ↓
Context Loss Probability
        ↓
Risk Classification
        ↓
Behavior Signals
        ↓
Recommendation
```

---

## 🖥️ Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/mallikarajapure-collab/ContextGuard-AI.git
cd ContextGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 📁 Project Structure

```text
ContextGuard-AI/
│
├── app.py
├── main.py
├── requirements.txt
│
├── data/
│   ├── developer_activity.csv
│   ├── improved_developer_activity.csv
│   └── processed_developer_activity.csv
│
├── models/
│   ├── context_loss_model.pkl
│   ├── feature_names.pkl
│   └── threshold.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── outputs/
│   ├── feature_importance.csv
│   ├── final_feature_importance.csv
│   ├── final_model_comparison.csv
│   └── threshold_results.csv
│
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

---

## 🎯 Example

A developer provides:

* Project: ML Project
* Activity: Debugging
* Task: Incomplete
* Activity Duration: 75 minutes
* Break Duration: 75 minutes
* Recent Project Switch: Yes

The system can identify behavioral signals such as:

* ⚠️ Very long break
* ⚠️ Incomplete task
* ⚠️ Project switch
* ⚠️ Long activity session

and generate a context-loss risk prediction.

---

## 🔬 Model Evaluation

The final model was evaluated on a test set of **400 samples**.

**Training samples:** 1600
**Testing samples:** 400
**Total dataset:** 2000 samples

The classification threshold was tuned to **0.30** for the final prediction system.

---

💡 Future Improvements

* Real-time IDE activity tracking
* VS Code extension integration
* Developer productivity timeline
* Historical context-loss tracking
* Personalized recovery suggestions
* Real-time notifications
* Larger and more diverse datasets
* Advanced ML and deep learning models

---

 👩‍💻 Author

**Mallika Rajapure**

MCA Student | Aspiring AI & Software Developer
⭐ If you find this project interesting, consider giving the repository a star!
