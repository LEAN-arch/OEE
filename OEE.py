import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import entropy, zscore
from sklearn.linear_model import LinearRegression
import io

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
anomalies = []

# Environment penalty (simulate workplace conditions)
def environment_penalty(x, y):
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

    agents['x'] += np.random.choice([-1, 0, 1], num_agents)
    agents['y'] += np.random.choice([-1, 0, 1], num_agents)
    agents['x'] = agents['x'].clip(0, space_size - 1)
    agents['y'] = agents['y'].clip(0, space_size - 1)
    agents['step'] = step
    history.append(agents.copy())

    hist, _ = np.histogram(agents['behavior'], bins=10, range=(0, 1), density=True)
    ent = entropy(hist + 1e-9)
    entropy_history.append(ent)

    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(agents[['x', 'y', 'behavior']])
    inertia = kmeans.inertia_
    cluster_history.append(inertia)

    if step == 0:
        initial_avg = agents['behavior'].mean()
    deviation = abs(agents['behavior'].mean() - initial_avg)
    resilience = 1 - deviation
    resilience_metric.append(resilience)

    availability = 0.95 + np.random.normal(0, 0.01)
    performance = 0.90 + np.random.normal(0, 0.01)
    quality = 0.98 + np.random.normal(0, 0.005)
    oee = availability * performance * quality
    oee_history.append({'step': step, 'availability': availability, 'performance': performance, 'quality': quality, 'oee': oee})

# Post-simulation processing
history_df = pd.concat(history).reset_index()
oee_df = pd.DataFrame(oee_history)

# Anomaly detection
entropy_z = zscore(entropy_history)
cluster_z = zscore(cluster_history)
anomalies = [(i, e, c) for i, (e, c) in enumerate(zip(entropy_z, cluster_z)) if abs(e) > entropy_threshold or abs(c) > 2]

# Forecasting
X = np.arange(num_steps).reshape(-1, 1)
model_entropy = LinearRegression().fit(X, entropy_history)
model_cluster = LinearRegression().fit(X, cluster_history)
future_steps = np.arange(num_steps, num_steps + 10).reshape(-1, 1)
entropy_forecast = model_entropy.predict(future_steps)
cluster_forecast = model_cluster.predict(future_steps)

# Streamlit UI
st.title("Factory Simulation Dashboard")

st.sidebar.header("Settings")
show_forecast = st.sidebar.checkbox("Show Forecast", True)
export_data = st.sidebar.button("Download Simulation CSV")

# Alerts
if any(abs(e) > entropy_threshold for e in entropy_z):
    st.error(f"⚠️ Alert: Behavioral entropy exceeds threshold ({entropy_threshold}) at some time steps.")

# Plots
st.subheader("Behavioral Entropy Over Time")
fig1, ax1 = plt.subplots()
ax1.plot(entropy_history, label="Observed Entropy")
if show_forecast:
    ax1.plot(range(num_steps, num_steps + 10), entropy_forecast, '--', label="Forecast Entropy")
for s in shock_steps:
    ax1.axvline(x=s, color='orange', linestyle='--', label='Shock')
ax1.set_xlabel("Step")
ax1.set_ylabel("Entropy (Behavioral Variability)")
ax1.legend()
st.pyplot(fig1)

st.subheader("Cluster Inertia (Team Dispersion)")
fig2, ax2 = plt.subplots()
ax2.plot(cluster_history, label="Inertia")
if show_forecast:
    ax2.plot(range(num_steps, num_steps + 10), cluster_forecast, '--', label="Forecast Inertia")
ax2.set_xlabel("Step")
ax2.set_ylabel("Cluster Inertia")
ax2.legend()
st.pyplot(fig2)

st.subheader("Resilience Score (Behavioral Recovery)")
st.line_chart(resilience_metric)

st.subheader("OEE Components Over Time")
fig3, ax3 = plt.subplots(figsize=(10, 4))
oee_df.set_index('step')[['availability', 'performance', 'quality', 'oee']].plot(ax=ax3)
ax3.set_ylabel("Percentage")
ax3.set_title("Availability, Performance, Quality, and OEE")
st.pyplot(fig3)

# Trajectories
st.subheader("Sample Agent Trajectories")
sample_agents = history_df[history_df['index'].isin(history_df['index'].unique()[:10])]
fig4, ax4 = plt.subplots()
for agent_id in sample_agents['index'].unique():
    traj = sample_agents[sample_agents['index'] == agent_id]
    ax4.plot(traj['x'], traj['y'], label=f'Agent {agent_id}')
ax4.set_title("Agent Movement (Shift Leaders and Operators)")
ax4.set_xlabel("X Position")
ax4.set_ylabel("Y Position")
ax4.legend()
st.pyplot(fig4)

# Export
if export_data:
    csv = history_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='factory_simulation.csv', mime='text/csv')

st.success("Dashboard loaded with industrially relevant metrics and visualization.")
