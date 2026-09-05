"""
SME Beneficiary Selection — ML Decision Support Demo
Pan-Atlantic University MSc Data Science Dissertation
Author: Ebere Onuoha | August 2026
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SME Selection · ML Demo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("lr_model.pkl"),
        "Decision Tree":       joblib.load("dt_model.pkl"),
        "Random Forest":       joblib.load("rf_model.pkl"),
        "SVM":                 joblib.load("svm_model.pkl"),
        "XGBoost":             joblib.load("xgb_model.pkl"),
    }

@st.cache_resource
def load_meta():
    thresholds  = joblib.load("thresholds.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return thresholds, feature_cols

models               = load_models()
thresholds, FEAT_COLS = load_meta()

# ── Band options ───────────────────────────────────────────────────────────────
BAND_LABELS = {
    0: "Not stated / Unknown",
    1: "₦0 – ₦5 million",
    2: "₦5 – ₦20 million",
    3: "₦20 – ₦50 million",
    4: "₦50 – ₦100 million",
    5: "₦100 million +",
}
STAFF_LABELS = {
    0: "Not stated",
    1: "1 – 5 staff",
    2: "6 – 20 staff",
    3: "21 – 50 staff",
    4: "51 – 100 staff",
    5: "100+ staff",
}
SECTOR_KEYS = [
    "sec_manufacturing", "sec_retail", "sec_logistics",
    "sec_agriculture",  "sec_technology", "sec_health",
    "sec_education",    "sec_financial",  "sec_construction",
    "sec_other",
]
SECTOR_LABELS = {
    "sec_manufacturing": "Manufacturing / Production",
    "sec_retail":        "Retail / Trading / Wholesale",
    "sec_logistics":     "Logistics / Transport / Haulage",
    "sec_agriculture":   "Agriculture / Food / Catering",
    "sec_technology":    "Technology / ICT / Digital",
    "sec_health":        "Health / Pharma / Wellness",
    "sec_education":     "Education / Training / Consulting",
    "sec_financial":     "Financial Services / Microfinance",
    "sec_construction":  "Construction / Real Estate",
    "sec_other":         "Other / Mixed",
}
MODEL_INFO = {
    "XGBoost":             {"auc": 0.871, "strength": "Best overall discrimination (AUC-ROC)"},
    "Logistic Regression": {"auc": 0.843, "strength": "Best minority-class recall — recommended for human-in-the-loop use"},
    "Decision Tree":       {"auc": 0.799, "strength": "Best F1 & MCC on minority class — most interpretable"},
    "Random Forest":       {"auc": 0.867, "strength": "Best precision (60%) — fewest false alarms"},
    "SVM":                 {"auc": 0.796, "strength": "Good overall accuracy"},
}

# ── Sidebar — model selection ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Coat_of_arms_of_Nigeria.svg/200px-Coat_of_arms_of_Nigeria.svg.png", width=60)
    st.title("Model selector")
    selected_model = st.selectbox(
        "Choose ML model",
        list(models.keys()),
        index=0,
        help="Select which trained model to use for prediction.",
    )
    info = MODEL_INFO[selected_model]
    st.caption(f"**Test AUC-ROC:** {info['auc']}")
    st.caption(f"**Best for:** {info['strength']}")
    st.divider()
    st.caption("Threshold used: Youden's J (optimised on training set)")
    thresh = thresholds[selected_model]
    st.caption(f"Decision threshold: **{thresh:.3f}**")
    st.divider()
    st.caption("MSc Data Science · Pan-Atlantic University · 2026")
    st.caption("Ebere Onuoha")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏦 SME Beneficiary Selection — ML Decision Support")
st.markdown(
    "Enter an SME applicant's details below. The model predicts whether they would "
    "have been selected for the commercial bank-sponsored development programme, "
    "based on patterns learned from 851 historical applications."
)
st.info(
    "⚠️ **Research tool only.** This system is a dissertation demonstration. "
    "All predictions are probabilistic and must be reviewed by a human decision-maker. "
    "Do not use for actual selection without a full fairness and compliance audit.",
    icon="ℹ️",
)

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("applicant_form"):
    st.subheader("Financial profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        last_yr_turnover = st.selectbox(
            "Last year turnover *",
            options=list(BAND_LABELS.keys()),
            format_func=lambda x: BAND_LABELS[x],
            index=2,
            help="Most important predictor per SHAP analysis.",
        )
        initial_capital = st.selectbox(
            "Initial capital",
            options=list(BAND_LABELS.keys()),
            format_func=lambda x: BAND_LABELS[x],
            index=1,
        )
    with col2:
        annual_profit = st.selectbox(
            "Annual profit *",
            options=list(BAND_LABELS.keys()),
            format_func=lambda x: BAND_LABELS[x],
            index=2,
            help="Top-3 SHAP predictor.",
        )
        amount_invested = st.selectbox(
            "Amount invested in business",
            options=list(BAND_LABELS.keys()),
            format_func=lambda x: BAND_LABELS[x],
            index=1,
        )
    with col3:
        monthly_turnover = st.selectbox(
            "Average monthly turnover",
            options=list(BAND_LABELS.keys()),
            format_func=lambda x: BAND_LABELS[x],
            index=2,
        )
        loan_amount = st.selectbox(
            "Loan amount (if applicable)",
            options=list(BAND_LABELS.keys()),
            format_func=lambda x: BAND_LABELS[x],
            index=0,
        )

    st.subheader("Business profile")
    col4, col5 = st.columns(2)
    with col4:
        accessed_loan = st.radio(
            "Has the business previously accessed a loan? *",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            horizontal=True,
            help="Fairness-critical field. Applicants without loan access face severe disparate impact (DIR = 0.154).",
        )
        staff_band = st.selectbox(
            "Number of staff",
            options=list(STAFF_LABELS.keys()),
            format_func=lambda x: STAFF_LABELS[x],
            index=1,
        )
    with col5:
        sector = st.selectbox(
            "Sector / Nature of business",
            options=SECTOR_KEYS,
            format_func=lambda x: SECTOR_LABELS[x],
            index=0,
        )

    st.subheader("Written application responses")
    st.caption("Type or paste actual responses. Prediction uses character count as a proxy for elaboration effort.")
    col6, col7 = st.columns(2)
    with col6:
        text_innov  = st.text_area("Innovation / unique idea",       height=80, placeholder="Describe your innovation...")
        text_compet = st.text_area("Top competitors",                height=80, placeholder="Name your main competitors...")
        text_chall  = st.text_area("Top challenges",                 height=80, placeholder="Describe key business challenges...")
    with col7:
        text_benefit = st.text_area("How will the programme benefit you?", height=80, placeholder="Explain expected benefits...")
        text_differ  = st.text_area("Product / service differentiator",    height=80, placeholder="What sets you apart?...")

    submitted = st.form_submit_button("🔍  Run prediction", use_container_width=True, type="primary")

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    # Build feature vector in the exact column order the model expects
    row = {c: 0.0 for c in FEAT_COLS}

    # Financial bands
    row["enc_initial_capital"]  = float(initial_capital)
    row["enc_amount_invested"]  = float(amount_invested)
    row["enc_monthly_turnover"] = float(monthly_turnover)
    row["enc_last_yr_turnover"] = float(last_yr_turnover)
    row["enc_annual_profit"]    = float(annual_profit)
    row["enc_loan_amount"]      = float(loan_amount)

    # Binary
    row["enc_loan"]  = float(accessed_loan)
    row["enc_staff"] = float(staff_band)

    # Text lengths
    row["len_innov"]   = float(len(text_innov))
    row["len_benefit"] = float(len(text_benefit))
    row["len_differ"]  = float(len(text_differ))
    row["len_compet"]  = float(len(text_compet))
    row["len_chall"]   = float(len(text_chall))

    # Sector one-hot
    for sk in SECTOR_KEYS:
        row[sk] = 1.0 if sk == sector else 0.0

    # Composites (must match notebook logic)
    fin_fields = ["enc_last_yr_turnover","enc_annual_profit","enc_amount_invested",
                  "enc_initial_capital","enc_monthly_turnover","enc_loan_amount"]
    row["composite_financial"] = sum(row[f] for f in fin_fields)
    row["composite_maturity"]  = row["enc_staff"]

    X_input = pd.DataFrame([row])[FEAT_COLS]

    model  = models[selected_model]
    prob   = float(model.predict_proba(X_input)[0, 1])
    pred   = int(prob >= thresh)

    st.divider()
    st.subheader("Prediction result")
    res_col, gauge_col = st.columns([2, 1])

    with res_col:
        if pred == 1:
            st.success(
                f"### ✅  LIKELY SELECTED\n\n"
                f"The model predicts this applicant **would have been selected** under historical patterns.",
                icon="✅",
            )
        else:
            st.error(
                f"### ❌  LIKELY NOT SELECTED\n\n"
                f"The model predicts this applicant **would not have been selected** under historical patterns.",
                icon="❌",
            )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Selection probability", f"{prob:.1%}")
        col_b.metric("Decision threshold", f"{thresh:.3f}")
        col_c.metric("Model used", selected_model)

        # Fairness warning
        if accessed_loan == 0:
            st.warning(
                "⚠️ **Fairness alert:** This applicant has **no prior loan access**. "
                "The fairness audit found that applicants without loan access were predicted "
                "as selected at **one-seventh the rate** of loan-holders "
                "(Disparate Impact Ratio = 0.154). This reflects bias embedded in the historical "
                "data, not a characteristic of this individual applicant.",
                icon="⚠️",
            )

        # Key feature summary
        with st.expander("Feature values sent to model"):
            display_df = pd.DataFrame({
                "Feature": FEAT_COLS,
                "Value": [row[c] for c in FEAT_COLS],
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with gauge_col:
        # Simple visual probability bar
        st.markdown("**Selection probability**")
        bar_pct = int(prob * 100)
        color   = "#16a34a" if pred == 1 else "#dc2626"
        st.markdown(
            f"""
            <div style="background:#e5e7eb;border-radius:8px;height:24px;width:100%;margin-bottom:6px">
              <div style="background:{color};border-radius:8px;height:24px;width:{bar_pct}%"></div>
            </div>
            <p style="text-align:center;font-size:2rem;font-weight:700;color:{color}">{bar_pct}%</p>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Threshold: {thresh:.3f} ({int(thresh*100)}%)")

    # All-models comparison
    st.divider()
    st.subheader("All models — for comparison")
    rows = []
    for mname, mobj in models.items():
        p = float(mobj.predict_proba(X_input)[0, 1])
        t = thresholds[mname]
        verdict = "✅ Selected" if p >= t else "❌ Not selected"
        rows.append({"Model": mname, "Probability": f"{p:.1%}",
                     "Threshold": f"{t:.3f}", "Verdict": verdict,
                     "Test AUC-ROC": MODEL_INFO[mname]["auc"]})
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Dissertation:** Application of Machine Learning Models for Data-Driven Beneficiary Selection "
    "in Sponsored SME Development Programmes in Nigeria · "
    "Pan-Atlantic University MSc Data Science · Ebere Onuoha · August 2026"
)
