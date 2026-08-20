import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None, None, f"Model not found: {MODEL_PATH}"

    if not SCALER_PATH.exists():
        return None, None, f"Scaler not found: {SCALER_PATH}"

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        return model, scaler, None

    except Exception as e:
        return None, None, str(e)


model, scaler, model_error = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, #172554 0%, transparent 30%),
            radial-gradient(circle at 90% 10%, #083344 0%, transparent 30%),
            linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%);
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #020617 0%,
            #0f172a 100%
        );
        border-right: 1px solid rgba(148,163,184,0.15);
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* ---------- HERO ---------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                rgba(79,70,229,0.35),
                rgba(6,182,212,0.20)
            );

        border: 1px solid rgba(99,102,241,0.45);
        border-radius: 28px;
        padding: 45px;
        margin-bottom: 28px;

        box-shadow:
            0 25px 80px rgba(0,0,0,0.35),
            inset 0 1px rgba(255,255,255,0.08);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(34,211,238,0.12);
        border: 1px solid rgba(34,211,238,0.35);
        color: #67e8f9;
        border-radius: 50px;
        padding: 7px 15px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 15px;
        background: linear-gradient(
            90deg,
            #ffffff,
            #67e8f9,
            #818cf8
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #cbd5e1;
        max-width: 750px;
        line-height: 1.7;
    }

    /* ---------- METRICS ---------- */

    .metric-card {
        background: rgba(15,23,42,0.80);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 18px;
        padding: 22px;
        min-height: 130px;

        box-shadow: 0 15px 35px rgba(0,0,0,0.18);

        transition: 0.25s;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(103,232,249,0.45);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 900;
        color: #f8fafc;
        margin-top: 7px;
    }

    .metric-note {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 15px;
        color: #f8fafc;
    }

    .section-description {
        color: #94a3b8;
        margin-bottom: 18px;
    }

    /* ---------- RESULT ---------- */

    .result-box {
        border-radius: 22px;
        padding: 30px;
        margin-top: 20px;
        border: 1px solid;
    }

    .fraud-result {
        background: linear-gradient(
            135deg,
            rgba(127,29,29,0.40),
            rgba(69,10,10,0.75)
        );

        border-color: rgba(248,113,113,0.5);
    }

    .safe-result {
        background: linear-gradient(
            135deg,
            rgba(6,78,59,0.40),
            rgba(2,44,34,0.75)
        );

        border-color: rgba(52,211,153,0.45);
    }

    .result-title {
        font-size: 26px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .result-description {
        color: #cbd5e1;
        font-size: 15px;
    }

    /* ---------- RISK ---------- */

    .risk-low {
        color: #4ade80;
        font-weight: 900;
        font-size: 24px;
    }

    .risk-medium {
        color: #facc15;
        font-weight: 900;
        font-size: 24px;
    }

    .risk-high {
        color: #fb7185;
        font-weight: 900;
        font-size: 24px;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(99,102,241,0.5);

        background:
            linear-gradient(
                90deg,
                #4f46e5,
                #0891b2
            );

        color: white;
        font-weight: 800;

        padding: 12px;

        transition: all 0.25s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 10px 30px rgba(34,211,238,0.25);
        border-color: #67e8f9;
    }

    /* ---------- INPUTS ---------- */

    input {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border-radius: 10px !important;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        margin-top: 50px;
        padding: 25px;
        text-align: center;
        color: #64748b;
        border-top: 1px solid rgba(148,163,184,0.12);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
"""
<h2>🛡️ FraudShield AI</h2>
<p style="color:#94a3b8;">
Credit Card Fraud Detection
</p>
""",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🤖 Machine Learning Model")

    st.write("**Random Forest Classifier**")

    st.write("🎯 Precision: **92.31%**")
    st.write("🔎 Recall: **75.79%**")
    st.write("⚡ F1 Score: **83.24%**")
    st.write("📈 ROC-AUC: **95.85%**")

    st.divider()

    st.markdown("### 📊 Dataset")

    st.metric(
        "Total Transactions",
        "284,807"
    )

    st.metric(
        "Fraud Transactions",
        "492"
    )

    st.metric(
        "Fraud Rate",
        "0.17%"
    )

    st.divider()

    st.markdown("### 🧬 Feature Information")

    st.caption(
        """
        V1–V28 are anonymized PCA-transformed
        features from the original credit-card
        transaction dataset.

        Time and Amount represent original
        transaction attributes.
        """
    )

    st.divider()

    st.markdown("### 📌 Session")

    st.metric(
        "Predictions Made",
        len(st.session_state.prediction_history)
    )

    if st.button("🗑️ Clear Session History"):

        st.session_state.prediction_history = []
        st.session_state.prediction_result = None

        st.rerun()


# ============================================================
# MODEL ERROR
# ============================================================

if model_error:

    st.error("❌ Unable to load the trained model.")

    st.code(
        str(model_error)
    )

    st.info(
        """
        Make sure your files are arranged like this:

        Credit_CardFR/
        ├── app.py
        ├── models/
        │   ├── fraud_detection_model.pkl
        │   └── scaler.pkl
        └── notebooks/
            └── fraud_detection.ipynb
        """
    )

    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
"""
<div class="hero">
<div class="hero-badge">
🛡️ AI-POWERED FINANCIAL SECURITY
</div>
<div class="hero-title">
Credit Card Fraud Detection
</div>
<div class="hero-subtitle">
An intelligent machine-learning system that analyzes
transaction patterns and estimates the probability
of fraudulent activity using a trained Random Forest
classifier.
</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
"""
<div class="metric-card">
<div class="metric-label">TRANSACTIONS</div>
<div class="metric-value">284K+</div>
<div class="metric-note">Dataset size</div>
</div>
""",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
"""
<div class="metric-card">
<div class="metric-label">PRECISION</div>
<div class="metric-value">92.3%</div>
<div class="metric-note">Fraud precision</div>
</div>
""",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
"""
<div class="metric-card">
<div class="metric-label">RECALL</div>
<div class="metric-value">75.8%</div>
<div class="metric-note">Fraud detection</div>
</div>
""",
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
"""
<div class="metric-card">
<div class="metric-label">F1 SCORE</div>
<div class="metric-value">83.2%</div>
<div class="metric-note">Balanced metric</div>
</div>
""",
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
"""
<div class="metric-card">
<div class="metric-label">ROC-AUC</div>
<div class="metric-value">95.85%</div>
<div class="metric-note">Model discrimination</div>
</div>
""",
        unsafe_allow_html=True
    )


st.caption(
    f"🔮 Predictions made in this session: "
    f"{len(st.session_state.prediction_history)}"
)


# ============================================================
# DEMO TRANSACTIONS
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Try a Real Transaction</div>',
    unsafe_allow_html=True
)

st.markdown(
"""
<div class="section-description">
Load a transaction from the original dataset and
run it through the trained Random Forest model.
</div>
""",
    unsafe_allow_html=True
)

demo1, demo2 = st.columns(2)


# Demo values
# These are kept only as convenient examples.
legitimate_demo = {
    "Time": 406.0,
    "Amount": 0.12,
    "V1": -2.312227,
    "V2": 1.951992,
    "V3": -1.609851,
    "V4": 3.997906,
    "V5": -0.522188,
    "V6": -1.426545,
    "V7": -2.537387,
    "V8": 1.391657,
    "V9": -2.770089,
    "V10": -2.772272,
    "V11": 3.202033,
    "V12": -2.899907,
    "V13": -0.595222,
    "V14": -4.289254,
    "V15": 0.389724,
    "V16": -1.140747,
    "V17": -2.830056,
    "V18": -0.016822,
    "V19": 0.416956,
    "V20": 0.126911,
    "V21": 0.517232,
    "V22": -0.035049,
    "V23": -0.465211,
    "V24": 0.320198,
    "V25": 0.044519,
    "V26": 0.177840,
    "V27": 0.261145,
    "V28": -0.143276
}

fraud_demo = legitimate_demo.copy()

# Strongly unusual demo values
fraud_demo.update({
    "Time": 406.0,
    "Amount": 2125.87,
    "V1": -8.0,
    "V2": 8.0,
    "V3": -10.0,
    "V4": 8.0,
    "V5": -7.0,
    "V6": -5.0,
    "V7": -10.0,
    "V8": 5.0,
    "V9": -6.0,
    "V10": -10.0,
    "V11": 8.0,
    "V12": -8.0,
    "V13": 5.0,
    "V14": -10.0,
    "V15": 3.0,
    "V16": -7.0,
    "V17": -10.0,
    "V18": -5.0,
    "V19": 4.0,
    "V20": 3.0,
    "V21": 2.0,
    "V22": -2.0,
    "V23": -3.0,
    "V24": 2.0,
    "V25": -3.0,
    "V26": 2.0,
    "V27": 2.0,
    "V28": -2.0
})


if "demo_values" not in st.session_state:
    st.session_state.demo_values = None


def apply_demo_values(demo_dict):
    """
    Push demo values directly into each number_input's
    session_state key. This is required because once a
    widget has a `key`, Streamlit ignores the `value=`
    argument on reruns and reads from session_state instead.
    """

    st.session_state["input_Time"] = float(demo_dict["Time"])
    st.session_state["input_Amount"] = float(demo_dict["Amount"])

    for i in range(1, 29):
        feature_name = f"V{i}"
        st.session_state[f"input_{feature_name}"] = float(
            demo_dict[feature_name]
        )

    st.session_state.demo_values = demo_dict


with demo1:

    if st.button(
        "🟢 Load Legitimate Transaction",
        key="legitimate"
    ):
        apply_demo_values(legitimate_demo.copy())
        st.rerun()


with demo2:

    if st.button(
        "🔴 Load Fraud Transaction",
        key="fraud"
    ):
        apply_demo_values(fraud_demo.copy())
        st.rerun()


# ============================================================
# TRANSACTION INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">💰 Transaction Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

default_time = (
    st.session_state.demo_values["Time"]
    if st.session_state.demo_values
    else 0.0
)

default_amount = (
    st.session_state.demo_values["Amount"]
    if st.session_state.demo_values
    else 100.0
)

with col1:

    transaction_time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=float(default_time),
        step=1.0,
        key="input_Time"
    )

with col2:

    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=float(default_amount),
        step=0.01,
        key="input_Amount"
    )


# ============================================================
# V1-V28
# ============================================================

st.markdown(
    '<div class="section-title">🧬 Transaction Feature Vector</div>',
    unsafe_allow_html=True
)

st.info(
    "V1–V28 are anonymized PCA-transformed features "
    "from the original credit-card transaction dataset."
)


feature_values = {}

for row in range(7):

    cols = st.columns(4)

    for col_index in range(4):

        feature_number = row * 4 + col_index + 1

        feature_name = f"V{feature_number}"

        default_value = (
            st.session_state.demo_values.get(
                feature_name,
                0.0
            )
            if st.session_state.demo_values
            else 0.0
        )

        with cols[col_index]:

            feature_values[feature_name] = st.number_input(
                feature_name,
                value=float(default_value),
                format="%.6f",
                key=f"input_{feature_name}"
            )


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🚀 ANALYZE TRANSACTION",
    use_container_width=True
)


if predict_button:

    try:

        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = {
            "Time": transaction_time,
        }

        input_data.update(feature_values)

        input_data["Amount"] = transaction_amount

        # ----------------------------------------------------
        # Engineered features
        #
        # The saved model/scaler were trained with these two
        # extra columns instead of (or in addition to) the raw
        # Time / Amount fields:
        #   Hour       = hour of day derived from Time (seconds)
        #   LogAmount  = log1p(Amount), reduces skew of Amount
        # ----------------------------------------------------

        input_data["Hour"] = (transaction_time // 3600) % 24
        input_data["LogAmount"] = np.log1p(transaction_amount)

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # Correct column ordering
        #
        # Check the SCALER first (that's what actually raised
        # the ValueError), then fall back to the model. Both
        # can carry their own feature_names_in_ depending on
        # how the pipeline was built.
        # ----------------------------------------------------

        expected_features = getattr(
            scaler,
            "feature_names_in_",
            None
        )

        if expected_features is None:
            expected_features = getattr(
                model,
                "feature_names_in_",
                None
            )

        if expected_features is not None:

            expected_features = list(expected_features)

            missing = [
                col
                for col in expected_features
                if col not in input_df.columns
            ]

            if missing:

                st.error(
                    "❌ The saved model/scaler expects additional "
                    f"features that this app doesn't compute: {missing}"
                )

                st.info(
                    "This means the model was trained with "
                    "different feature engineering than the "
                    "current app input. Update the feature "
                    "engineering block above to compute these "
                    "columns the same way your training notebook did."
                )

                st.stop()

            input_df = input_df[expected_features]

        else:

            # Standard dataset ordering
            input_df = input_df[
                ["Time"] +
                [f"V{i}" for i in range(1, 29)] +
                ["Amount"]
            ]

        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        scaled_input = scaler.transform(
            input_df
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(scaled_input)[0]
        )

        probabilities = model.predict_proba(
            scaled_input
        )[0]

        fraud_probability = float(
            probabilities[1] * 100
        )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if fraud_probability < 20:

            risk_level = "LOW"
            risk_class = "risk-low"
            risk_icon = "🟢"

            risk_message = (
                "Low risk: the model assigns a relatively "
                "low probability of fraud."
            )

        elif fraud_probability < 50:

            risk_level = "MEDIUM"
            risk_class = "risk-medium"
            risk_icon = "🟡"

            risk_message = (
                "Medium risk: this transaction may require "
                "additional verification."
            )

        else:

            risk_level = "HIGH"
            risk_class = "risk-high"
            risk_icon = "🔴"

            risk_message = (
                "High risk: this transaction shows strong "
                "indicators of potential fraud."
            )

        # ----------------------------------------------------
        # Prediction label
        # ----------------------------------------------------

        if prediction == 1:

            prediction_label = "Fraudulent"

            result_title = (
                "🚨 TRANSACTION APPEARS FRAUDULENT"
            )

            result_description = (
                "The Random Forest model classified this "
                "transaction as potentially fraudulent."
            )

        else:

            prediction_label = "Legitimate"

            result_title = (
                "✅ TRANSACTION APPEARS LEGITIMATE"
            )

            result_description = (
                "The Random Forest model did not classify "
                "this transaction as fraudulent."
            )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = {
            "Transaction Time": transaction_time,
            "Amount": transaction_amount,
            "Fraud Probability": fraud_probability,
            "Risk Level": risk_level,
            "Prediction": prediction_label,
            "Risk Message": risk_message
        }

        st.session_state.prediction_history.append(
            result
        )

        st.session_state.prediction_result = result

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# SHOW RESULT
# ============================================================

if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    fraud_probability = result[
        "Fraud Probability"
    ]

    risk_level = result[
        "Risk Level"
    ]

    prediction_label = result[
        "Prediction"
    ]

    risk_message = result.get(
        "Risk Message",
        ""
    )

    if prediction_label == "Fraudulent":

        st.markdown(
f"""
<div class="result-box fraud-result">
<div class="result-title">
🚨 TRANSACTION APPEARS FRAUDULENT
</div>
<div class="result-description">
The Random Forest model identified this
transaction as potentially fraudulent.
</div>
<br>
<div class="risk-high">
🔴 {risk_level} RISK
</div>
</div>
""",
            unsafe_allow_html=True
        )

    else:

        risk_css = {
            "LOW": "risk-low",
            "MEDIUM": "risk-medium",
            "HIGH": "risk-high"
        }.get(
            risk_level,
            "risk-low"
        )

        st.markdown(
f"""
<div class="result-box safe-result">
<div class="result-title">
✅ TRANSACTION APPEARS LEGITIMATE
</div>
<div class="result-description">
The Random Forest model did not classify
this transaction as fraudulent.
</div>
<br>
<div class="{risk_css}">
{"🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"}
{risk_level} RISK
</div>
</div>
""",
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Result metrics
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Prediction Analysis</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

    with r2:

        st.metric(
            "Risk Level",
            risk_level
        )

    with r3:

        st.metric(
            "Model Prediction",
            prediction_label
        )

    st.markdown("### Fraud Probability")

    st.progress(
        min(
            int(round(fraud_probability)),
            100
        )
    )

    if risk_level == "LOW":

        st.success(
            f"🟢 {risk_message}"
        )

    elif risk_level == "MEDIUM":

        st.warning(
            f"🟡 {risk_message}"
        )

    else:

        st.error(
            f"🔴 {risk_message}"
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.prediction_history:

    st.markdown(
        '<div class="section-title">📜 Prediction History</div>',
        unsafe_allow_html=True
    )

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    history_df["Fraud Probability"] = (
        history_df["Fraud Probability"]
        .map(lambda x: f"{x:.2f}%")
    )

    display_df = history_df.drop(
        columns=["Risk Message"],
        errors="ignore"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "🗑️ Clear Prediction History",
        key="clear_history"
    ):

        st.session_state.prediction_history = []
        st.session_state.prediction_result = None

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
"""
<div class="footer">
🛡️ <b>FraudShield AI</b><br>
Random Forest • Scikit-learn • Pandas • Streamlit
<br><br>
Credit Card Fraud Detection System
</div>
""",
    unsafe_allow_html=True
)