import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="Energy-based Abrasion Calculator", layout="centered")
st.title("Energy-based Abrasion Coefficient & Mass Prediction")

st.markdown(
"""
This app predicts **mass loss** based on **energy-based abrasion coefficients**:

\\[
m(E) = m_0 \cdot \exp(-a_E \cdot E)
\\]

You can select a mix and size fraction, enter initial mass and target revolution to compute the mass loss.
"""
)

# ----------------------------
# Dataset (your 8 mixes)
# ----------------------------
# Paste your existing data_1 ... data_8 here
# For brevity, I will just show data_1; in your app, include all 8 datasets

data_1 = {
    "Revolutions": [0, 3400, 6800, 10300, 13800, 17400, 21000, 28700],
    "90.5 - 128 mm": [35.5, 28.2, 25.1, 24.9, 24.7, 22.6, 20.2, 18.6],
    "64.0 - 90.5 mm": [46.2, 53.3, 53.5, 51.9, 50.6, 49.6, 51.7, 49.0],
    "45.3 - 64.0 mm": [51.8, 48.2, 47.6, 46.1, 45.6, 45.4, 43.7, 41.8],
}
# Repeat for data_2 ... data_8
mixes = {1: data_1}  # Add 2: data_2, ... 8: data_8

SIZE_ORDER = ["90.5 - 128 mm","64.0 - 90.5 mm","45.3 - 64.0 mm"]

# ----------------------------
# Helper functions
# ----------------------------
def energy_per_rev(P, rpm):
    return 60.0 * P / rpm

def fit_aE(E, m):
    E = np.asarray(E, float)
    m = np.asarray(m, float)
    y = np.log(m / m[0])
    E = E - E[0]
    return - (E @ y) / (E @ E)

def cumulative_energy(E_per_rev, rev):
    return E_per_rev * np.asarray(rev, float)

# ----------------------------
# Inputs
# ----------------------------
st.header("Operating conditions")
RPM = st.number_input("Mill speed (RPM)", value=32.0)
P_NET_W = st.number_input("Net power (W)", value=900.0)

st.header("Select dataset")
mix_id = st.selectbox("Select Mix", options=list(mixes.keys()))
df = pd.DataFrame(mixes[mix_id])
size = st.selectbox("Select size fraction", options=[c for c in df.columns if c != "Revolutions"])

# ----------------------------
# Target input
# ----------------------------
st.header("Predict mass at target revolution")
m0_input = st.number_input("Enter initial mass m0 (kg)", value=float(df[size].iloc[0]))
rev_target = st.number_input("Target revolution", min_value=0, value=int(df["Revolutions"].iloc[-1]))

# ----------------------------
# Compute a_E from data
# ----------------------------
rev = df["Revolutions"].values
m_measured = df[size].values
E_axis = cumulative_energy(energy_per_rev(P_NET_W, RPM), rev)
a_E = fit_aE(E_axis, m_measured)

# ----------------------------
# Predict mass
# ----------------------------
E_target = energy_per_rev(P_NET_W, RPM) * rev_target
m_pred_target = m0_input * np.exp(-a_E * E_target)
mass_loss = m0_input - m_pred_target

# ----------------------------
# Display results
# ----------------------------
st.subheader("Results")
st.write(f"Energy-based abrasion coefficient a_E = **{a_E:.4e} 1/J**")
st.write(f"Predicted mass at {rev_target} rev: **{m_pred_target:.2f} kg**")
st.write(f"Mass loss: **{mass_loss:.2f} kg**")

# ----------------------------
# Plot measured vs predicted
# ----------------------------
m_pred_full = m0_input * np.exp(-a_E * E_axis)

df_plot = pd.DataFrame({
    "Energy (J)": np.concatenate([E_axis, E_axis]),
    "Mass (kg)": np.concatenate([m_measured, m_pred_full]),
    "Type": ["Measured"]*len(E_axis) + ["Predicted"]*len(E_axis)
})

chart = alt.Chart(df_plot).mark_line(point=True).encode(
    x="Energy (J)",
    y="Mass (kg)",
    color="Type:N"
).properties(
    width=600,
    height=400,
    title=f"Measured vs Predicted Mass for Mix {mix_id}, Size {size}"
)

st.altair_chart(chart, use_container_width=True)



