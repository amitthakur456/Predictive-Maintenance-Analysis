import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Load trained model
# -----------------------
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙",
    layout="wide"
)

st.title("⚙ Predictive Maintenance System")
st.write("AI-based Machine Failure Prediction")

# -----------------------
# INPUT UI
# -----------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("🌡 Temperature")

    air_temp = st.number_input("Air temperature [K]", 250.0, 350.0, 300.0)

    process_temp = st.number_input("Process temperature [K]", 250.0, 400.0, 310.0)


    st.subheader("⚙ Machine")

    rot_speed = st.number_input("Rotational speed [rpm]", 500, 3000, 1500)

    torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)

    tool_wear = st.number_input("Tool wear [min]", 0, 300, 10)


with col2:

    st.subheader("🏷 Product Type")

    product_type = st.selectbox("Type", ["Low","Medium","High"])

    Type_L = 1 if product_type=="Low" else 0
    Type_M = 1 if product_type=="Medium" else 0


    st.subheader("⚠ Failure Flags")

    TWF = st.selectbox("Tool Wear Failure", [0,1])

    HDF = st.selectbox("Heat Dissipation Failure", [0,1])

    PWF = st.selectbox("Power Failure", [0,1])

    OSF = st.selectbox("Overstrain Failure", [0,1])

    RNF = st.selectbox("Random Failure", [0,1])


# -----------------------
# FEATURE ENGINEERING
# -----------------------

power = torque * rot_speed

temp_diff = process_temp - air_temp

wear_rate = tool_wear / rot_speed


st.markdown("### 🔧 Auto Features")

c1,c2,c3 = st.columns(3)

c1.metric("Power", round(power,2))

c2.metric("Temp Difference", round(temp_diff,2))

c3.metric("Wear Rate", round(wear_rate,5))


# -----------------------
# CREATE INPUT DATA
# -----------------------

input_dict = {

'Air temperature [K]': air_temp,

'Process temperature [K]': process_temp,

'Rotational speed [rpm]': rot_speed,

'Torque [Nm]': torque,

'Tool wear [min]': tool_wear,

'Type_L': Type_L,

'Type_M': Type_M,

'TWF': TWF,

'HDF': HDF,

'PWF': PWF,

'OSF': OSF,

'RNF': RNF,

'power': power,

'temp_diff': temp_diff,

'wear_rate': wear_rate

}

input_df = pd.DataFrame([input_dict])


# -----------------------
# REORDER FEATURES EXACTLY LIKE MODEL
# -----------------------

input_df = input_df[model.feature_names_in_]


# -----------------------
# PREDICT
# -----------------------

if st.button("🔍 Predict"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]


    st.markdown("---")

    if prediction == 1:

        st.error(f"⚠ High Failure Risk\nProbability: {probability:.2%}")

    else:

        st.success(f"✅ Machine Safe\nFailure Probability: {probability:.2%}")


    # download result

    result = input_df.copy()

    result["Failure Probability"] = probability

    result["Prediction"] = prediction


    csv = result.to_csv(index=False)


    st.download_button(

        "⬇ Download Prediction",

        csv,

        "prediction_result.csv",

        "text/csv"

    )


# -----------------------
# FOOTER
# -----------------------

st.markdown("---")

st.caption("Industry-level Predictive Maintenance ML App")