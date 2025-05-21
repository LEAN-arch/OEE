import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, zscore, beta
from sklearn.linear_model import LinearRegression
import random

# Set global random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(num_operators, num_steps, factory_size, adaptation_rate, supervisor_influence, disruption_steps):
    """Generate synthetic factory operations data for one shift."""
    # Initialize operator data
    operators = pd.DataFrame({
        'operator_id': range(num_operators),
        'x': np.random.uniform(0, factory_size, num_operators),
        'y': np.random.uniform(0, factory_size, num_operators),
        'sop_compliance': np.random.uniform(0.7, 0.95, num_operators),
        'role': np.random.choice(['operator', 'supervisor'], num_operators, p=[0.85, 0.15]),
        'zone': np.random.choice(['assembly', 'maintenance', 'packaging'], num_operators)
    })

    # Initialize time-series data
    history = []
    compliance_entropy = []
    clustering_index = []
    resilience_scores = []
    oee_history = []
    productivity_loss = []
    initial_compliance_avg = operators['sop_compliance'].mean()

    # Generate data for each time step
    for step in range(num_steps):
        # Simulate disruptions
        if step in disruption_steps:
            event = "Machine Breakdown" if step == disruption_steps[0] else "Shift Change"
            leader_indices = operators[operators['role'] == 'supervisor'].sample(frac=0.3).index
            operators.loc[leader_indices, 'role'] = 'operator'
            operators['sop_compliance'] *= 0.9  # 10% compliance drop

        # Update SOP compliance with zone effects and supervisor influence
        for i in operators.index:
            zone = operators.loc[i, 'zone']
            x, y = operators.loc[i, ['x', 'y']]
            # Zone-specific compliance variability
            compliance_noise = np.random.normal(0, 0.02 if zone == 'assembly' else 0.01)
            # Supervisor influence
            supervisors = operators[operators['role'] == 'supervisor']
            if not supervisors.empty:
                avg_supervisor_compliance = supervisors['sop_compliance'].mean()
                operators.loc[i, 'sop_compliance'] += adaptation_rate * supervisor_influence * (avg_supervisor_compliance - operators.loc[i, 'sop_compliance'])
            # Apply zone effects
            if zone == 'assembly' and 20 < x < 40 and 20 < y < 40:
                operators.loc[i, 'sop_compliance'] -= 0.05
            elif zone == 'maintenance' and 60 < x < 80 and 60 < y < 80:
                operators.loc[i, 'sop_compliance'] -= 0.03
            operators.loc[i, 'sop_compliance'] += compliance_noise
            operators.loc[i, 'sop_compliance'] = np.clip(operators.loc[i, 'sop_compliance'], 0.5, 1.0)

        # Simulate movement
        operators['x'] += np.random.uniform(-0.5, 0.5, num_operators)
        operators['y'] += np.random.uniform(-0.5, 0.5, num_operators)
        operators['x'] = operators['x'].clip(0, factory_size - 1)
        operators['y'] = operators['y'].clip(0, factory_size - 1)
        operators['step'] = step
        history.append(operators.copy())

        # Compliance variability (entropy)
        hist, _ = np.histogram(operators['sop_compliance'], bins=10, range=(0.5, 1.0), density=True)
        ent = entropy(hist + 1e-9)
        compliance_entropy.append(ent)

        # Team clustering (synthetic inertia based on positions and compliance)
        distances = np.sqrt((operators['x'] - operators['x'].mean())**2 + (operators['y'] - operators['y'].mean())**2)
        compliance_var = operators['sop_compliance'].std()
        inertia = np.mean(distances) * (1 + compliance_var) + np.random.normal(0, 10)
        clustering_index.append(max(inertia, 0))

        # Resilience: recovery from initial compliance
        deviation = abs(operators['sop_compliance'].mean() - initial_compliance_avg)
        resilience_scores.append(1 - deviation / initial_compliance_avg)

        # OEE: tied to compliance
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

        # Productivity loss
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
        pd.concat(history).reset_index(drop=True),
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
