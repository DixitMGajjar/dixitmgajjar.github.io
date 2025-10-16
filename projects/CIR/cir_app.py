import streamlit as st
from CIR import dwnld_cln_data, simulate_cir_paths, calculate_statistics, plot_paths, analyze_model_fit, calibrate_cir
import numpy as np

st.title("CIR Interest Rate Simulator")

# User inputs
ticker = st.sidebar.text_input("Ticker", "^IRX")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

r0 = st.sidebar.number_input("Initial Rate r0", 0.0, 0.2, 0.05, 0.001)
a = st.sidebar.number_input("Mean Reversion Speed a", 0.0, 5.0, 0.1, 0.01)
b = st.sidebar.number_input("Long-term Mean b", 0.0, 0.2, 0.03, 0.001)
sigma = st.sidebar.number_input("Volatility σ", 0.0, 1.0, 0.01, 0.001)
T = st.sidebar.number_input("Time Horizon (years)", 0.1, 5.0, 1.0, 0.1)
n_steps = st.sidebar.number_input("Steps per Year", 1, 500, 252, 1)
n_paths = st.sidebar.number_input("Number of Paths", 10, 50000, 1000, 10)

if st.button("Run CIR Simulation"):
    # Optionally download data if you want calibration
    # data = dwnld_cln_data(ticker, start_date, end_date)
    # a, b, sigma = calibrate_cir(data)

    paths = simulate_cir_paths(r0, a, b, sigma, T, n_steps, n_paths)
    stats = calculate_statistics(paths)
    
    st.write("Simulation Statistics:", stats)
    
    # Plotting
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, T, n_steps + 1), paths[:, :min(50, n_paths)])
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Interest Rate")
    ax.set_title("Sample CIR Paths")
    st.pyplot(fig)
