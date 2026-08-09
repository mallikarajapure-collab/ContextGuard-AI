import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Lost-Context Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD MODEL
# ============================================================

project_root = Path(__file__).resolve().parent
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #0b1020 0%,
            #111827 50%,
            #0b1220 100%
        );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- HEADER ---------- */

    .hero {
        padding: 28px 32px;
        border-radius: 24px;
        margin-bottom: 25px;

        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.95),
            rgba(15, 23, 42, 0.95)
        );

        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 15px 45px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #94a3b8;
    }

    .status {
        display: inline-block;
        margin-top: 15px;
        padding: 7px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        background: rgba(34,197,94,0.12);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.25);
    }


    /* ---------- CARDS ---------- */

    .card {
        padding: 22px;
        border-radius: 20px;
        min-height: 135px;

        background: rgba(30, 41, 59, 0.72);

        border: 1px solid rgba(148,163,184,0.15);

        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .card-label {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 32px;
        font-weight: 800;
    }

    .card-description {
        font-size: 12px;
        color: #64748b;
        margin-top: 5px;
    }


    /* ---------- RISK ---------- */

    .risk-card {
        padding: 32px;
        border-radius: 24px;
        text-align: center;
        margin: 25px 0;

        background: linear-gradient(
            135deg,
            rgba(30,41,59,0.9),
            rgba(15,23,42,0.9)
        );

        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 15px 45px rgba(0,0,0,0.22);
    }

    .risk-label {
        color: #94a3b8;
        font-size: 14px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .risk-score {
        font-size: 64px;
        font-weight: 900;
        margin: 5px 0;
    }

    .risk-level {
        font-size: 22px;
        font-weight: 800;
    }


    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 24px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 15px;
    }


    /* ---------- SIGNAL ---------- */

    .signal-card {
        padding: 16px;
        border-radius: 14px;
        margin-bottom: 10px;

        background: rgba(127,29,29,0.18);
        border: 1px solid rgba(248,113,113,0.18);
    }

    .signal-name {
        font-weight: 700;
        font-size: 15px;
    }


    /* ---------- INFO ---------- */

    .info-card {
        padding: 20px;
        border-radius: 18px;

        background: rgba(30,41,59,0.65);
        border: 1px solid rgba(148,163,184,0.15);

        line-height: 1.6;
    }


    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148,163,184,0.15);
    }

    [data-testid="stSidebar"] h1 {
        font-weight: 800;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 12px;
        font-weight: 800;
        border: none;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🧠 AI Lost-Context Detector
        </div>

        <div class="hero-subtitle">
            Developer Behavior Intelligence System
            <br>
            Predicting the risk of losing development context
            from activity patterns.
        </div>

        <div class="status">
            ● ML SYSTEM ACTIVE
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Activity Input")

st.sidebar.caption(
    "Enter the developer's current activity."
)

st.sidebar.markdown("---")


project = st.sidebar.selectbox(
    "Project",
    [
        "AI Chatbot",
        "College Project",
        "ML Project",
        "Portfolio Website",
    ],
)


activity_type = st.sidebar.selectbox(
    "Activity Type",
    [
        "coding",
        "debugging",
        "documentation",
        "research",
        "testing",
    ],
)


task_status = st.sidebar.selectbox(
    "Task Status",
    [
        "completed",
        "in_progress",
        "incomplete",
    ],
)


duration_minutes = st.sidebar.slider(
    "Activity Duration",
    min_value=10,
    max_value=180,
    value=45,
    step=5,
)


break_minutes = st.sidebar.slider(
    "Break Duration",
    min_value=0,
    max_value=240,
    value=15,
    step=5,
)


context_switch = st.sidebar.checkbox(
    "Recent Project Switch"
)


st.sidebar.markdown("---")


analyze_button = st.sidebar.button(
    "🔍 ANALYZE CONTEXT RISK",
    use_container_width=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False


# ============================================================
# PREDICTION
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # ENGINEERED FEATURES
    # --------------------------------------------------------

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
        0 if context_switch else 1
    )

    recent_switch_count = (
        1 if context_switch else 0
    )


    # --------------------------------------------------------
    # CREATE INPUT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ONE HOT ENCODING
    # --------------------------------------------------------

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "project",
            "activity_type",
            "task_status",
        ],
    )


    # --------------------------------------------------------
    # MATCH TRAINING FEATURES
    # --------------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0,
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]


    prediction = (
        1
        if probability >= threshold
        else 0
    )


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if probability >= 0.70:

        risk_level = "HIGH"
        risk_icon = "🚨"

    elif probability >= 0.40:

        risk_level = "MEDIUM"
        risk_icon = "⚠️"

    else:

        risk_level = "LOW"
        risk_icon = "✅"


    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    signals = {
        "Very long break": break_minutes > 60,
        "Long break": break_minutes > 30,
        "Incomplete task": task_status == "incomplete",
        "Project switch": context_switch,
        "Short activity": duration_minutes < 30,
        "Long activity": duration_minutes > 60,
    }


    # --------------------------------------------------------
    # SAVE STATE
    # --------------------------------------------------------

    st.session_state.prediction_done = True

    st.session_state.probability = probability

    st.session_state.prediction = prediction

    st.session_state.risk_level = risk_level

    st.session_state.risk_icon = risk_icon

    st.session_state.signals = signals


# ============================================================
# RESULTS
# ============================================================

if st.session_state.prediction_done:

    probability = st.session_state.probability
    risk_level = st.session_state.risk_level
    risk_icon = st.session_state.risk_icon
    signals = st.session_state.signals


    # ========================================================
    # TOP METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-label">
                    CONTEXT LOSS PROBABILITY
                </div>

                <div class="card-value">
                    {probability * 100:.2f}%
                </div>

                <div class="card-description">
                    Model confidence
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with col2:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-label">
                    CURRENT RISK
                </div>

                <div class="card-value">
                    {risk_icon} {risk_level}
                </div>

                <div class="card-description">
                    Based on behavior patterns
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with col3:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-label">
                    DECISION THRESHOLD
                </div>

                <div class="card-value">
                    {threshold:.2f}
                </div>

                <div class="card-description">
                    Tuned classification threshold
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # RISK DISPLAY
    # ========================================================

    st.markdown(
        f"""
        <div class="risk-card">

            <div class="risk-label">
                CONTEXT RISK ANALYSIS
            </div>

            <div class="risk-score">
                {probability * 100:.2f}%
            </div>

            <div class="risk-level">
                {risk_icon} {risk_level} RISK
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.progress(
        float(probability)
    )


    # ========================================================
    # SIGNALS + ANALYSIS
    # ========================================================

    left, right = st.columns(2)


    # --------------------------------------------------------
    # BEHAVIOR SIGNALS
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="section-title">🔎 Behavior Signals</div>',
            unsafe_allow_html=True,
        )

        active_signals = [
            name
            for name, active in signals.items()
            if active
        ]


        if active_signals:

            for signal in active_signals:

                st.markdown(
                    f"""
                    <div class="signal-card">

                        <div class="signal-name">
                            ⚠️ {signal} detected
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.success(
                "✅ No major behavioral risk signals detected."
            )


    # --------------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">🧠 AI Analysis</div>',
            unsafe_allow_html=True,
        )


        reasons = []


        if break_minutes > 60:
            reasons.append(
                "Very long break may increase context recovery difficulty."
            )

        elif break_minutes > 30:
            reasons.append(
                "Long break detected between development activities."
            )


        if context_switch:
            reasons.append(
                "Recent project switching may increase context switching."
            )


        if task_status == "incomplete":
            reasons.append(
                "The unfinished task may require additional context recovery."
            )


        if duration_minutes > 60:
            reasons.append(
                "Long activity session detected."
            )


        if not reasons:

            reasons.append(
                "Current activity pattern appears relatively stable."
            )


        for reason in reasons:

            st.markdown(
                f"""
                <div class="info-card">
                    • {reason}
                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 Recommendation</div>',
        unsafe_allow_html=True,
    )


    if risk_level == "HIGH":

        st.error(
            "🚨 High context-loss risk. "
            "Review your previous task, recent code changes, "
            "unfinished work, and project state before continuing."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "⚠️ Moderate context-loss risk. "
            "A quick review of your previous task and recent "
            "changes may help you regain context."
        )

    else:

        st.success(
            "✅ Your current activity pattern appears stable. "
            "You can continue working."
        )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div class="risk-card">

            <div class="risk-score">
                🧠
            </div>

            <div class="risk-level">
                Ready to Analyze
            </div>

            <p style="color:#94a3b8;">
                Enter developer activity details from the
                sidebar and run the AI analysis.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 Model Intelligence</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Performance obtained during final model evaluation."
)


m1, m2, m3, m4 = st.columns(4)


with m1:
    st.metric(
        "Accuracy",
        "65.50%"
    )


with m2:
    st.metric(
        "Precision",
        "59.43%"
    )


with m3:
    st.metric(
        "Recall",
        "39.87%"
    )


with m4:
    st.metric(
        "F1 Score",
        "47.73%"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">⚖️ Model Comparison</div>',
    unsafe_allow_html=True,
)


comparison_data = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
        ],

        "Accuracy": [
            0.6550,
            0.6425,
        ],

        "Precision": [
            0.5943,
            0.5728,
        ],

        "Recall": [
            0.3987,
            0.3734,
        ],

        "F1 Score": [
            0.4773,
            0.4521,
        ],
    }
)


st.dataframe(
    comparison_data.style.format(
        {
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Important Behavior Features</div>',
    unsafe_allow_html=True,
)


feature_data = pd.DataFrame(
    {
        "Feature": [
            "break_minutes",
            "duration_minutes",
            "recent_switch_count",
            "unfinished_task",
            "in_progress_task",
            "context_switch",
            "same_project",
            "project_College Project",
            "project_ML Project",
            "activity_type_research",
        ],

        "Importance": [
            0.2220,
            0.2177,
            0.0858,
            0.0819,
            0.0342,
            0.0325,
            0.0315,
            0.0263,
            0.0263,
            0.0251,
        ],
    }
)


st.bar_chart(
    feature_data.set_index("Feature")
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Confusion Matrix</div>',
    unsafe_allow_html=True,
)


confusion_matrix = pd.DataFrame(
    [
        [199, 43],
        [95, 63],
    ],

    index=[
        "Actual: No Context Loss",
        "Actual: Context Loss",
    ],

    columns=[
        "Predicted: No Context Loss",
        "Predicted: Context Loss",
    ],
)


st.dataframe(
    confusion_matrix,
    use_container_width=True,
)


# ============================================================
# FINAL MODEL
# ============================================================

st.success(
    "🏆 Final Model: Logistic Regression — selected because "
    "it achieved the highest F1 Score among the evaluated models."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🧠 AI Lost-Context Detector
        • Machine Learning Based Developer Behavior Analysis
    </div>
    """,
    unsafe_allow_html=True,
)