# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import streamlit as st


# Reservoir Simulation using Darcy's Law (1D simplified)
def reservoir_simulation(length, num_points, k, mu, initial_saturation, time_steps, dt):
    # Create space grid
    dx = length / num_points  # Grid spacing (m)
    x = np.linspace(0, length, num_points)

    # Initial saturation profile
    saturation = np.full(num_points, initial_saturation)

    # Darcy's law for fluid flow: Rate of change in saturation
    def darcy_flow(saturation, k, mu, dx):
        flow = np.zeros_like(saturation)
        for i in range(1, len(saturation) - 1):
            flow[i] = (k / mu) * (saturation[i + 1] - saturation[i - 1]) / (2 * dx)
        return flow

    # Reservoir simulation loop
    saturation_profiles = []
    for t in range(time_steps):
        flow = darcy_flow(saturation, k, mu, dx)
        saturation += flow * dt
        saturation_profiles.append(saturation.copy())

    return np.array(saturation_profiles), x


# Objective Function for History Matching (simplified)
def history_matching(params, historical_data, reservoir_simulation_func):
    k, mu = params  # Permeability and viscosity
    simulated_data, _ = reservoir_simulation_func(length=1000, num_points=100, k=k, mu=mu,
                                                  initial_saturation=0.5, time_steps=100, dt=0.01)
    # Extract production data (for simplicity, use final saturation at last timestep as proxy for production)
    simulated_production = simulated_data[-1, :]  # Use final saturation as production proxy
    error = np.sum((historical_data - simulated_production) ** 2)
    return error


# Machine Learning Forecasting (Random Forest)
def train_forecasting_model(data):
    X = data[['time', 'pressure', 'temperature']]  # Features
    y = data['production_rate']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    st.write(f'Mean Squared Error: {mse}')
    return model


# Visualization of Simulation Results (Interactive with Plotly)
def plot_saturation(saturation_profiles, x):
    fig = go.Figure(data=[go.Surface(z=saturation_profiles)])
    fig.update_layout(title='Reservoir Saturation Over Time', scene=dict(
        xaxis_title='Reservoir Length (m)',
        yaxis_title='Time Step',
        zaxis_title='Saturation'))
    st.plotly_chart(fig)


def plot_production_forecast(model, X_test, y_test):
    predictions = model.predict(X_test)
    plt.scatter(X_test['time'], y_test, color='blue', label='True Production Rate')
    plt.scatter(X_test['time'], predictions, color='red', label='Predicted Production Rate')
    plt.xlabel('Time')
    plt.ylabel('Production Rate')
    plt.title('Production Rate Forecasting')
    plt.legend()
    st.pyplot()


# Streamlit interface
def main():
    st.title("Reservoir Simulation and Forecasting Tool")

    # User input for simulation parameters
    st.subheader("Enter Reservoir Simulation Parameters")

    length = st.number_input("Reservoir Length (m)", value=1000.0, min_value=1.0, step=1.0)
    num_points = st.number_input("Number of Grid Points", value=100, min_value=10, step=1)
    k = st.number_input("Permeability (mD)", value=100, min_value=1, step=1)
    mu = st.number_input("Viscosity (cp)", value=1.0, min_value=0.1, step=0.1)
    initial_saturation = st.slider("Initial Saturation", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    time_steps = st.number_input("Number of Time Steps", value=100, min_value=1, step=1)
    dt = st.number_input("Time Increment (dt)", value=0.01, min_value=0.0001, step=0.0001)

    # Upload historical production data
    uploaded_file = st.file_uploader("Upload Historical Production Data", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith("csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith("xlsx"):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith("json"):
            data = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format")

        st.write("Historical Production Data Preview:")
        st.write(data.head())

        # Run reservoir simulation
        saturation_profiles, x = reservoir_simulation(length, num_points, k, mu, initial_saturation, time_steps, dt)

        # Plot the results of reservoir simulation
        plot_saturation(saturation_profiles, x)

        # Simulate historical production data for history matching
        historical_data = saturation_profiles[-1,
                          :]  # For simplicity, use the last saturation profile as historical data

        # History Matching: Minimize error between model and historical data
        initial_guess = [100, 1]  # Initial guess for k and mu
        result = minimize(history_matching, initial_guess, args=(historical_data, reservoir_simulation))
        best_params = result.x
        st.write(f"Best Parameters after History Matching: k={best_params[0]}, mu={best_params[1]}")

        # Train a machine learning model to forecast future production
        model = train_forecasting_model(data)

        # Visualize the forecasting results
        X_test = data[['time', 'pressure', 'temperature']]
        y_test = data['production_rate']
        plot_production_forecast(model, X_test, y_test)


if __name__ == "__main__":
    main()
