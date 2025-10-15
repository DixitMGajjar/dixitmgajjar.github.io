import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------
# Title & Description
# -------------------------
st.title("Vasicek Interest Rate Simulator")

st.markdown("""
### About this project
This app implements the **Vasicek short-rate model** to simulate interest rate paths using Monte Carlo simulation.

The Vasicek model assumes that the short rate \( r_t \) evolves according to:

$$
dr_t = a(b - r_t) dt + \sigma dW_t
$$

Where:  
- **a (Mean Reversion Speed)**: How fast rates revert to the long-term mean. Higher means faster reversion.  
- **b (Long-term Mean)**: The average rate the process tends to revert to.  
- **σ (Volatility)**: Standard deviation of the rate changes (random fluctuations).  
- **r0 (Initial Rate)**: Starting interest rate.  
- **T (Time Horizon)**: Total time in years to simulate.  
- **n_steps (Steps per year)**: Number of steps per year in simulation.  
- **n_paths (Number of Paths)**: Number of Monte Carlo simulations to run.

You can also provide a **ticker symbol** to automatically use the last observed interest rate from Yahoo Finance.
""")

# -------------------------
# User Inputs
# -------------------------
ticker = st.text_input("Ticker Symbol", value="^IRX")
start_date = st.text_input("Start Date (YYYY-MM-DD)", value="2020-01-01")
end_date = st.text_input("End Date (YYYY-MM-DD)", value="2024-12-01")

r0 = st.number_input("Initial Rate (r0)", value=0.05, step=0.01, format="%.4f")
a = st.number_input("Mean Reversion Speed (a)", value=0.1, step=0.01, format="%.4f")
b = st.number_input("Long-term Mean (b)", value=0.03, step=0.01, format="%.4f")
sigma = st.number_input("Volatility (σ)", value=0.01, step=0.001, format="%.4f")
T = st.number_input("Time Horizon (years)", value=1.0, step=0.1, format="%.2f")
n_steps = st.number_input("Steps per year", value=252, step=1)
n_paths = st.number_input("Number of Paths", value=5000, step=100)

# -------------------------
# Functions
# -------------------------
def dwnld_cln_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    df = pd.DataFrame()
    df["actual_IRX"] = data["Close"] / 100  # Convert % to decimal
    df["return"] = df["actual_IRX"].pct_change()
    df["daily_change"] = df["actual_IRX"].diff()
    df.dropna(inplace=True)
    return df

def simulate_vasicek_paths(r0, a, b, sigma, T, n_steps, n_paths):
    dt = T / n_steps
    rates = np.zeros((n_steps + 1, n_paths))
    rates[0] = r0
    Z = np.random.normal(size=(n_steps, n_paths))
    for t in range(1, n_steps + 1):
        rates[t] = rates[t-1] + a*(b - rates[t-1])*dt + sigma*np.sqrt(dt)*Z[t-1]
    return rates

# -------------------------
# Run Simulation Button
# -------------------------
if st.button("Run Simulation"):
    # Download data if ticker provided
    if ticker:
        data = dwnld_cln_data(ticker, start_date, end_date)
        r0 = data["actual_IRX"].iloc[-1]  # overwrite r0 with last observed rate
        st.write(f"Using last observed rate r0 = {r0:.4f} from {ticker}")

    # Simulate paths
    paths = simulate_vasicek_paths(r0, a, b, sigma, T, n_steps, n_paths)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(50, n_paths)):
        ax.plot(np.linspace(0, T, n_steps + 1), paths[:, i], lw=0.5)
    ax.set_title("Sample Vasicek Paths")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Interest Rate")
    st.pyplot(fig)

    # Statistics
    final_rates = paths[-1]
    st.write("### Simulation Statistics")
    st.write(f"Mean final rate: {np.mean(final_rates):.4f}")
    st.write(f"Std of final rate: {np.std(final_rates):.4f}")
    st.write(f"5th percentile: {np.percentile(final_rates,5):.4f}")
    st.write(f"95th percentile: {np.percentile(final_rates,95):.4f}")
