import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.stats import entropy, zscore
from sklearn.linear_model import LinearRegression
import random

# Set global random seed for reproducibility
np.random.seed(42)
random.seed(42)

def initialize_operators(num_operators, factory_size):
    """Initialize operators with positions, SOP compliance, and roles."""
    return pd.DataFrame({
        'x': np.random.uniform(0, factory_size, num_operators),
        'y': np.random.uniform(0, factory_size, num_operators),
        'sop_compliance': np.random.uniform(0.7, 0.95, num_operators),
        'role': np.random.choice(['operator', 'supervisor'], num_operators, p=[0.85, 0.15]),
        'zone': np.random.choice(['assembly', 'maintenance', 'packaging'], num_operators)
    })

def environmental_impact(x, y, zone):
    """Apply environmental impact based on work zone."""
    if zone == 'assembly' and 20 < x < 40 and 20 < y < 40:
        return -0.05
    elif zone == 'maintenance' and 60 < x < 80 and 60 < y < 80:
        return -0.03
    return 0.0

def get_neighbors_kdtree(df, idx, radius=5):
    """Find nearby operators using KDTree (5-meter radius)."""
    tree = KDTree(df[['x', 'y']].values)
    ax, ay = df.loc[idx, ['x', 'y']]
    indices = tree.query_radius([[ax, ay]], r=radius)[0]
    indices = indices[indices != idx]
    return df.iloc[indices]

def run_simulation(num_operators, num_steps, factory_size, adaptation_rate, supervisor_influence, disruption_steps):
    """Simulate factory operations over one shift."""
    operators = initialize_operators(num_operators, factory_size)
    history = []
    compliance_entropy = []
    clustering_index = []
    resilience_scores = []
    oee_history = []
    productivity_loss = []
    initial_compliance_avg = operators['sop_compliance'].mean()

    for step in range(num_steps):
        if step in disruption_steps:
            event = "Machine Breakdown" if step == disruption_steps[0] else "Shift Change"
            leader_indices = operators[operators['role'] == 'supervisor'].sample(frac=0.3).index
            operators.loc[leader_indices, 'role'] = 'operator'
            operators['sop_compliance'] *= 0.9

        for i in operators.index:
            neighbors = get_neighbors_kdtree(operators, i)
            if not neighbors.empty:
                supervisors = neighbors[neighbors['role'] == 'supervisor']
                operators_subset = neighbors[neighbors['role'] == 'operator']
                avg_compliance = 0
                if not supervisors.empty:
                    avg_compliance += supervisor_influence * supervisors['sop_compliance'].mean()
                if not operators_subset.empty:
                    avg_compliance += (1 - supervisor_influence) * operators_subset['sop_compliance'].mean()
                delta = avg_compliance - operators.loc[i, 'sop_compliance']
                operators.loc[i, 'sop_compliance'] += adaptation_rate * np.tanh(delta)

            x, y, zone = operators.loc[i, ['x', 'y', 'zone']]
            operators.loc[i, 'sop_compliance'] += environmental_impact(x, y, zone)
            operators.loc[i, 'sop_compliance'] = np.clip(operators.loc[i, 'sop_compliance'], 0.5, 1.0)

        operators['x'] += np.random.uniform(-0.5, 0.5, num_operators)
        operators['y'] += np.random.uniform(-0.5, 0.5, num_operators)
        operators['x'] = operators['x'].clip(0, factory_size - 1)
        operators['y'] = operators['y'].clip(0, factory_size - 1)
        operators['step'] = step
        history.append(operators.copy())

        hist, _ = np.histogram(operators['sop_compliance'], bins=10, range=(0.5, 1.0), density=True)
        ent = entropy(hist + 1e-9)
        compliance_entropy.append(ent)

        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(operators[['x', 'y', 'sop_compliance']])
        clustering_index.append(kmeans.inertia_)

        deviation = abs(operators['sop_compliance'].mean() - initial_compliance_avg)
        resilience_scores.append(1 - deviation / initial_compliance_avg)

        avg_compliance = operators['sop_compliance'].mean()
        availability = min(0.90 + 0.1 * avg_compliance, 0.95) + np.random.normal(0, 0.005)
        performance = min(0.85 + 0.1 * avg_compliance, 0.90) + np.random.normal(0, 0.005)
        quality = min(0.97 + 0.03 * avg_compliance, 0.99) + np.random.normal(0, 0.002)
        oee = availability * performance * quality
        oee_history.append({
            'step': step,
            'availability': availability,
            'performance': performance,
            'quality': quality,
            'oee': oee
        })

        loss = max(0, initial_compliance_avg - operators['sop_compliance'].mean()) * 100
        productivity_loss.append(loss)

    # Forecasting
    X = np.arange(num_steps).reshape(-1, 1)
    model_compliance = LinearRegression().fit(X, compliance_entropy)
    model_clustering = LinearRegression().fit(X, clustering_index)
    future_steps = np.arange(num_steps, num_steps + 10).reshape(-1, 1)
    compliance_forecast = model_compliance.predict(future_steps)
    clustering_forecast = model_clustering.predict(future_steps)

    return (
        pd.concat(history).reset_index(),
        {'data': compliance_entropy, 'z_scores': zscore(compliance_entropy), 'forecast': compliance_forecast},
        {'data': clustering_index, 'z_scores': zscore(clustering_index), 'forecast': clustering_forecast},
        resilience_scores,
        oee_history,
        productivity_loss
    )

def plot_compliance_variability(compliance_entropy, disruption_steps, forecast=None):
    """Plot SOP compliance variability over shift."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(compliance_entropy, label="SOP Compliance Variability", color='#1f77b4')
    if forecast is not None:
        ax.plot(range(len(compliance_entropy), len(compliance_entropy) + 10), forecast, '--', label="Predicted Variability", color='#ff7f0e')
    for i, s in enumerate(disruption_steps):
        label = "Machine Breakdown" if i == 0 else "Shift Change"
        ax.axvline(x=s, color='red', linestyle='--', label=label)
    ax.set_xlabel("Time (2-min Intervals)")
    ax.set_ylabel("Compliance Variability (Entropy)")
    ax.set_title("Operator SOP Compliance Variability")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_team_clustering(clustering_index, forecast=None):
    """Plot team clustering index."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(clustering_index, label="Team Clustering Index", color='#2ca02c')
    if forecast is not None:
        ax.plot(range(len(clustering_index), len(clustering_index) + 10), forecast, '--', label="Predicted Clustering", color='#d62728')
    ax.set_xlabel("Time (2-min Intervals)")
    ax.set_ylabel("Clustering Index (Lower = Tighter Teams)")
    ax.set_title("Team Cohesion on Factory Floor")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_resilience(resilience_scores):
    """Plot production resilience score."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resilience_scores, label="Production Resilience", color='#9467bd')
    ax.set_xlabel("Time (2-min Intervals)")
    ax.set_ylabel("Resilience Score (1 = Full Recovery)")
    ax.set_title("Production Resilience After Disruptions")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_oee(oee_df):
    """Plot OEE and its components."""
    fig, ax = plt.subplots(figsize=(10, 4))
    oee_df.set_index('step')[['availability', 'performance', 'quality', 'oee']].plot(
        ax=ax,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Time (2-min Intervals)")
    ax.set_title("OEE: Availability, Performance, Quality")
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_worker_density(history_df, factory_size):
    """Plot heatmap of worker density on factory floor."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        data=history_df,
        x='x',
        y='y',
        fill=True,
        cmap='viridis',
        ax=ax
    )
    ax.set_title("Worker Density Heatmap on Factory Floor")
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.set_xlim(0, factory_size)
    ax.set_ylim(0, factory_size)
    return fig