import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import entropy, zscore
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)

# Parameters
num_agents = 100
num_steps = 100
space_size = 50
adaptation_rate = 0.05
leadership_influence = 0.2
shock_steps = [30, 60]
entropy_threshold = 2.0

# Initialize agents
agents = pd.DataFrame({
    'x': np.random.randint(0, space_size, num_agents),
    'y': np.random.randint(0, space_size, num_agents),
    'behavior': np.random.rand(num_agents),  # Proxy for SOP adherence/stress response
    'type': np.random.choice(['operator', 'supervisor'], num_agents, p=[0.9, 0.1])
})

# Containers
history = []
entropy_history = []
cluster_history = []
resilience_metric = []
oee_history = []

# Environment penalty (simulate workplace conditions)
def environment_penalty(x, y):
    # Example: a 'stress zone' in the workspace reduces behavior score
    if 10 < x < 20 and 10 < y < 20:
        return -0.1
    return 0.0

def get_neighbors(df, idx, radius=3):
    ax, ay = df.loc[idx, ['x', 'y']]
    neighbors = df[((df['x'] - ax)**2 + (df['y'] - ay)**2)**0.5 <= radius]
    return neighbors[neighbors.index != idx]

# Simulation loop
for step in range(num_steps):
    if step in shock_steps:
        # Some supervisors become operators during disruption
        leader_indices = agents[agents['type'] == 'supervisor'].sample(frac=0.5).index
        agents.loc[leader_indices, 'type'] = 'operator'

    for i in agents.index:
        neighbors = get_neighbors(agents, i)
        if not neighbors.empty:
            supervisors = neighbors[neighbors['type'] == 'supervisor']
            operators = neighbors[neighbors['type'] == 'operator']
            avg_behavior = 0
            if not supervisors.empty:
                avg_behavior += leadership_influence * supervisors['behavior'].mean()
            if not operators.empty:
                avg_behavior += (1 - leadership_influence) * operators['behavior'].mean()
            delta = avg_behavior - agents.loc[i, 'behavior']
            agents.loc[i, 'behavior'] += adaptation_rate * np.tanh(delta)

        x, y = agents.loc[i, ['x', 'y']]
        agents.loc[i, 'behavior'] += environment_penalty(x, y)
        agents.loc[i, 'behavior'] = np.clip(agents.loc[i, 'behavior'], 0, 1)

    # Agents move randomly on the floor
    agents['x'] += np.random.choice([-1, 0, 1], num_agents)
    agents['y'] += np.random.choice([-1, 0, 1], num_agents)
    agents['x'] = agents['x'].clip(0, space_size - 1)
    agents['y'] = agents['y'].clip(0, space_size - 1)
    agents['step'] = step
    history.append(agents.copy())

    # Calculate entropy (behavior variability)
    hist, _ = np.histogram(agents['behavior'], bins=10, range=(0, 1), density=True)
    ent = entropy(hist + 1e-9)
    entropy_history.append(ent)

    # Clustering to measure team dispersion
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(agents[['x', 'y', 'behavior']])
    inertia = kmeans.inertia_
    cluster_history.append(inertia)

    # Resilience: how much behavior deviates from initial average
    if step == 0:
        initial_avg = agents['behavior'].mean()
    deviation = abs(agents['behavior'].mean() - initial_avg)
    resilience = 1 - deviation
    resilience_metric.append(resilience)

    # Simulate OEE components with small noise
    availability = 0.95 + np.random.normal(0, 0.01)
    performance = 0.90 + np.random.normal(0, 0.01)
    quality = 0.98 + np.random.normal(0, 0.005)
    oee = availability * performance * quality
    oee_history.append({'step': step, 'availability': availability, 'performance': performance, 'quality': quality, 'oee': oee})

# Post-simulation processing
history_df = pd.concat(history).reset_index()
oee_df = pd.DataFrame(oee_history)

# Anomaly detection on entropy and cluster inertia
entropy_z = zscore(entropy_history)
cluster_z = zscore(cluster_history)
anomalies = [(i, e, c) for i, (e, c) in enumerate(zip(entropy_z, cluster_z))
             if abs(e) > entropy_threshold or abs(c) > 2]

# Forecasting future entropy and cluster inertia trends
X = np.arange(num_steps).reshape(-1, 1)
model_entropy = LinearRegression().fit(X, entropy_history)
model_cluster = LinearRegression().fit(X, cluster_history)
future_steps = np.arange(num_steps, num_steps + 10).reshape(-1, 1)
entropy_forecast = model_entropy.predict(future_steps)
cluster_forecast = model_cluster.predict(future_steps)

# Streamlit UI
st.title("Factory Operations Simulation Dashboard")

st.sidebar.header("Simulation Controls")
show_forecast = st.sidebar.checkbox("Show Trend Predictions", True)
export_data = st.sidebar.button("Export Simulation Data")

# Alerts
if any(abs(e) > entropy_threshold for e in entropy_z):
    st.error(f"⚠️ ALERT: Worker behavior variability above safe limits detected. "
             "Check SOP compliance and stress response.")

# Worker behavior variability plot
st.subheader("Worker Behavior Variability Over Time")
fig1, ax1 = plt.subplots()
ax1.plot(entropy_history, label="Actual Behavior Variability")
if show_forecast:
    ax1.plot(range(num_steps, num_steps + 10), entropy_forecast, '--', label="Predicted Variability")
for s in shock_steps:
    ax1.axvline(x=s, color='red', linestyle='--', label='Disruption Event')
ax1.set_xlabel("Time (Simulation Step)")
ax1.set_ylabel("Behavior Variability (Entropy Score)")
ax1.legend(loc='upper right')
ax1.grid(True)
st.pyplot(fig1)

# Team cohesion plot
st.subheader("Team Cohesion & Movement Over Time")
fig2, ax2 = plt.subplots()
ax2.plot(cluster_history, label="Team Spread (Cluster Inertia)")
if show_forecast:
    ax2.plot(range(num_steps, num_steps + 10), cluster_forecast, '--', label="Predicted Spread")
ax2.set_xlabel("Time (Simulation Step)")
ax2.set_ylabel("Team Cohesion Score (Lower = Better)")
ax2.legend(loc='upper right')
ax2.grid(True)
st.pyplot(fig2)

# Resilience score chart
st.subheader("Recovery After Disruption (Resilience Score)")
st.line_chart(resilience_metric)
st.caption("Scores close to 1 indicate full recovery of team behavior after disturbances.")

# OEE components plot
st.subheader("Overall Equipment Effectiveness (OEE) Metrics")
fig3, ax3 = plt.subplots(figsize=(10, 4))
oee_df.set_index('step')[['availability', 'performance', 'quality', 'oee']].plot(ax=ax3)
ax3.set_ylabel("Percentage (%)")
ax3.set_xlabel("Time (Simulation Step)")
ax3.set_title("Availability, Performance, Quality, and OEE Over Time")
ax3.legend(["Availability", "Performance", "Quality", "Overall OEE"], loc='lower left')
ax3.grid(True)
st.pyplot(fig3)

# Agent movement trajectories
st.subheader("Movement Paths of Supervisors and Operators")
sample_agents = history_df[history_df['index'].isin(history_df['index'].unique()[:10])]
fig4, ax4 = plt.subplots()
for agent_id in sample_agents['index'].unique():
    traj = sample_agents[sample_agents['index'] == agent_id]
    role = "Supervisor" if traj['type'].iloc[0] == 'supervisor' else "Operator"
    ax4.plot(traj['x'], traj['y'], label=f'{role} #{agent_id}')
ax4.set_title("Shop Floor Movement (X and Y Coordinates)")
ax4.set_xlabel("Position X")
ax4.set_ylabel("Position Y")
ax4.legend(loc='upper right', fontsize='small', ncol=2)
ax4.grid(True)
st.pyplot(fig4)

# Export CSV data
if export_data:
    csv = history_df.to_csv(index=False)
    st.download_button(label="Download Complete Simulation Data CSV",
                       data=csv,
                       file_name='factory_simulation_data.csv',
                       mime='text/csv')

st.success("Dashboard loaded — use insights to monitor team dynamics and operational efficiency.")

