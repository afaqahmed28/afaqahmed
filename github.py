import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy-based Abrasion Coefficient", layout="centered")

st.title("Energy-based Abrasion Coefficient Calculator")
st.markdown(
"""
This app computes the **energy-based abrasion coefficient**  
\\[
\\ln(m/m_0) = -a_E E
\\]
"""
)

# ----------------------------
# USER INPUTS
# ----------------------------
st.header("Operating conditions")

RPM = st.number_input("Mill speed (RPM)", value=32.0)
P_NET_W = st.number_input("Net power (W)", value=900.0)

st.header("Experimental data")

rev_input = st.text_area(
    "Cumulative revolutions (comma-separated)",
    value="0, 5000, 10000, 15000"
)

mass_input = st.text_area(
    "Measured mass values (comma-separated, same length)",
    value="50, 45, 41, 37"
)

# ----------------------------
# FUNCTIONS
# ----------------------------
def energy_per_rev(P, rpm):
    return 60.0 * P / rpm

def fit_aE(E, m):
    E = np.asarray(E, float)
    m = np.asarray(m, float)

    y = np.log(m / m[0])
    E = E - E[0]

    return - (E @ y) / (E @ E)

# ----------------------------
# COMPUTE
# ----------------------------
if st.button("Compute abrasion coefficient"):

    try:
        rev = np.array([float(x) for x in rev_input.split(",")])
        m = np.array([float(x) for x in mass_input.split(",")])

        if len(rev) != len(m):
            st.error("Revolutions and mass arrays must have the same length.")
            st.stop()

        # Energy axis
        E = energy_per_rev(P_NET_W, RPM) * rev

        # Fit coefficient
        a_E = fit_aE(E, m)

        st.success("Calculation successful")

        st.subheader("Results")
        st.write(f"**a_E = {a_E:.4e} 1/J**")
        st.write(f"**a_E = {a_E*1e3:.4e} 1/kJ**")
        st.write(f"**a_E = {a_E*1e6:.4e} 1/MJ**")

        # ----------------------------
        # PLOT
        # ----------------------------
        m_pred = m[0] * np.exp(-a_E * E)

        fig, ax = plt.subplots()
        ax.scatter(E, m, label="Measured")
        ax.plot(E, m_pred, label="Predicted", linestyle="--")
        ax.set_xlabel("Cumulative energy E (J)")
        ax.set_ylabel("Mass")
        ax.legend()
        ax.set_title("Measured vs Predicted Mass")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
