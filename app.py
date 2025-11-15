import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ======================================================
#  CONFIG & CACHED MODEL LOADING
# ======================================================
st.set_page_config(
    page_title="Fares's Shawarma Shop | Churn Predictor",
    page_icon="🌯",
    layout="centered"
)

@st.cache_resource
def load_models():
    """Load model, scaler, and encoder once and cache them."""
    try:
        # get the absolute path of the current script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # construct safe paths for all model files
        model_path = os.path.join(base_dir, "xgb_best_model.pkl")
        scaler_path = os.path.join(base_dir, "scaler.pkl")
        ohe_path = os.path.join(base_dir, "ohe.pkl")
        
        # load all serialized objects
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        ohe = joblib.load(ohe_path)
        
        return model, scaler, ohe
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, None

model, scaler, ohe = load_models()
# ======================================================
#  HEADER & DESCRIPTION
# ======================================================
st.markdown("<h1 style='text-align:center;'>🌯 Fares's Shawarma Shop</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Customer Churn Prediction Dashboard</h3>", unsafe_allow_html=True)

st.markdown("""
Welcome to **Fares’s Shawarma Shop Customer Retention App**!  
Use this tool to predict whether a loyal shawarma lover might **churn** 🌯 or **stay** ❤️  
Built with 🧠 **XGBoost**, ⚙️ **Scikit-learn**, and 🌐 **Streamlit**.
""")

st.divider()

# ======================================================
#  INPUT SECTION
# ======================================================
st.subheader("📋 Customer Profile")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("🧓 Age", min_value=10, max_value=100, value=30)
    tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=130, value=12)
with col2:
    monthly_charges = st.number_input("💵 Monthly Spending ($)", min_value=0.0, max_value=200.0, value=70.0)
    gender = st.selectbox("🧍 Gender", ["Male", "Female"])

# Subtle design touch
st.markdown(
    "<hr style='border:1px solid #ccc; margin-top:20px; margin-bottom:20px;'>",
    unsafe_allow_html=True
)

# ======================================================
#  PREDICTION
# ======================================================
st.subheader("🔮 Prediction Engine")

if st.button("🍔 Predict Customer Churn", use_container_width=True):
    if not model or not scaler or not ohe:
        st.error("❌ Model files missing. Please check your deployment.")
    else:
        # --- Preprocess Input ---
        input_df = pd.DataFrame({
            "Age": [age],
            "MonthlyCharges": [monthly_charges],
            "Tenure": [tenure],
            "Gender": [gender]
        })

        # Encode gender
        gender_encoded = ohe.transform(input_df[["Gender"]])
        gender_encoded_df = pd.DataFrame(
            gender_encoded, 
            columns=ohe.get_feature_names_out(["Gender"]), 
            index=input_df.index
        )

        # Combine and scale
        X_final = pd.concat(
            [input_df[["Age", "MonthlyCharges", "Tenure"]], gender_encoded_df], axis=1
        )
        X_final[["Age", "MonthlyCharges", "Tenure"]] = scaler.transform(
            X_final[["Age", "MonthlyCharges", "Tenure"]]
        )

        # --- Predict ---
        prediction = model.predict(X_final)[0]
        proba = model.predict_proba(X_final)[0][1]

        st.markdown("### 🧾 Prediction Result")
        if prediction == 1:
            st.error(
                f"⚠️ **This customer is likely to churn!** (Probability: {proba:.2%})\n\n"
                "📉 Consider offering a *Shawarma Loyalty Discount* 🌯💸 and allow customer to put ketchup on food 🍅!"
            )
        else:
            st.success(
                f"✅ **This customer is likely to stay!** (Probability: {proba:.2%})\n\n"
                "🎉 Keep up the great service and tasty shawarmas! ABBAS DONT FORGET ZA BEBSI🥤!"
            )

        st.caption("Model: XGBoost (SMOTE-balanced, optimized for F1-score)")

# ======================================================
#  FOOTER
# ======================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center;'>
Made with ❤️ and extra garlic sauce by <b>Fares Jony</b>  
<br>
<a href='https://github.com/KnightBytePy' target='_blank'>🌐 GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
