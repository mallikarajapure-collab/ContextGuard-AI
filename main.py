import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="AI Lost-Context Detector", page_icon="🧠", layout="wide")

# ============================================================
# LOAD MODEL
# ============================================================
project_root = Path(__file__).resolve().parent
models_folder = project_root / "models"

try:
    model = joblib.load(models_folder / "context_loss_model.pkl")
    feature_names = joblib.load(models_folder / "feature_names.pkl")
    threshold = joblib.load(models_folder / "threshold.pkl")
except FileNotFoundError as e:
    st.error(
        f"Model files not found in `{models_folder}`. "
        f"Make sure `context_loss_model.pkl`, `feature_names.pkl`, "
        f"and `threshold.pkl` exist there.\n\nDetails: {e}"
    )
    st.stop()

# ============================================================
# CUSTOM CSS
# ============================================================
RISK_COLORS = {
    "HIGH": ("#3d1f1f", "#ff6b6b"),
    "MEDIUM": ("#3d341f", "#ffd166"),
    "LOW": ("#1f3d24", "#6bff8f"),
}

st.markdown(
    """
    <style>
    .main-title { font-size: 42px; font-weight: 700; margin-bottom: 5px; }
    .subtitle { font-size: 18px; margin-bottom: 30px; opacity: 0.85; }
    .risk-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .risk-title { font-size: 32px; font-weight: 700; }
    .risk-score { font-size: 48px; font-weight: 700; margin-top: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🧠 AI Lost-Context Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict the risk of losing development context '
    'from developer activity patterns.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Developer Activity")
st.sidebar.caption("Enter your current development activity.")

project = st.sidebar.selectbox(
    "Project", ["AI Chatbot", "College Project", "ML Project", "Portfolio Website"]
)
activity_type = st.sidebar.selectbox(
    "Activity", ["coding", "debugging", "documentation", "research", "testing"]
)
task_status = st.sidebar.selectbox("Task Status", ["completed", "in_progress", "incomplete"])
duration_minutes = st.sidebar.slider("Activity Duration (minutes)", 10, 180, 75)
break_minutes = st.sidebar.slider("Break Duration (minutes)", 0, 240, 75)
context_switch = st.sidebar.checkbox("Recent Project Switch")

analyze_button = st.sidebar.button("🔍 Analyze Context Risk", use_container_width=True)

# ============================================================
# SESSION STATE
# ============================================================
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# ============================================================
# ANALYSIS
# ============================================================
if analyze_button:
    # Engineered features
    long_break = int(break_minutes > 30)
    very_long_break = int(break_minutes > 60)
    unfinished_task = int(task_status == "incomplete")
    in_progress_task = int(task_status == "in_progress")
    short_activity = int(duration_minutes < 30)
    long_activity = int(duration_minutes > 60)
    same_project = int(not context_switch)
    recent_switch_count = int(context_switch)

    input_data = pd.DataFrame(
        {
            "duration_minutes": [duration_minutes],
            "break_minutes": [break_minutes],
            "context_switch": [int(context_switch)],
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

    # One-hot encoding to match training format
    input_data = pd.get_dummies(input_data, columns=["project", "activity_type", "task_status"])
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Prediction
    probability = float(model.predict_proba(input_data)[0][1])
    probability = min(max(probability, 0.0), 1.0)  # clamp for st.progress safety
    prediction = int(probability >= threshold)

    # Risk level
    if probability >= 0.70:
        risk_level, risk_icon = "HIGH", "🚨"
    elif probability >= 0.40:
        risk_level, risk_icon = "MEDIUM", "⚠️"
    else:
        risk_level, risk_icon = "LOW", "✅"

    # Save results to session state
    st.session_state.prediction_done = True
    st.session_state.probability = probability
    st.session_state.prediction = prediction
    st.session_state.risk_level = risk_level
    st.session_state.risk_icon = risk_icon
    st.session_state.signals = {
        "Very long break": break_minutes > 60,
        "Long break": break_minutes > 30,
        "Incomplete task": task_status == "incomplete",
        "Project switch": context_switch,
        "Short activity": duration_minutes < 30,
        "Long activity": duration_minutes > 60,
    }

# ============================================================
# RESULTS
# ============================================================
if st.session_state.prediction_done:
    probability = st.session_state.probability
    risk_level = st.session_state.risk_level
    risk_icon = st.session_state.risk_icon
    bg_color, text_color = RISK_COLORS[risk_level]

    st.markdown("---")
    st.subheader("📊 Context Loss Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Context Loss Probability", f"{probability * 100:.2f}%")
    with col2:
        st.metric("Risk Level", risk_level)
    with col3:
        st.metric("Decision Threshold", f"{threshold:.2f}")

    # Risk display box
    st.markdown(
        f"""
        <div class="risk-box" style="background-color:{bg_color}; color:{text_color};">
            <div class="risk-title">{risk_icon} {risk_level} RISK</div>
            <div class="risk-score">{probability * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Progress bar
    st.write("### 📈 Risk Probability")
    st.progress(probability)

    # Behavior signals
    st.write("### 🔎 Behavior Signals")
    active_signals = [name for name, active in st.session_state.signals.items() if active]

    if active_signals:
        for signal in active_signals:
            st.warning(f"⚠️ {signal} detected")
    else:
        st.success("No major risk signals detected.")

    # Recommendation
    st.write("### 💡 Recommendation")
    if risk_level == "HIGH":
        st.error(
            "Consider reviewing your last task, recent code changes, "
            "and unfinished work before continuing."
        )
    elif risk_level == "MEDIUM":
        st.warning("A quick review of your previous task may help you regain context.")
    else:
        st.success("Your current activity pattern appears stable. You can continue working.")

else:
    st.info(
        "👈 Enter your developer activity details from the sidebar "
        "and click **Analyze Context Risk**."
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("AI Lost-Context Detector • Machine Learning Based Developer Behavior Analysis")