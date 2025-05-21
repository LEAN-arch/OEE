import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, zscore
from sklearn.linear_model import LinearRegression
import random

# Set global random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(num_operators, num_steps, factory_size, adaptation_rate, supervisor_influence, disruption_steps):
    """Generate synthetic factory operations data for one shift, aggregated by zone."""
    # Initialize team data by zone
    zones = ['assembly', 'maintenance', 'packaging']
    operators_per_zone = num_operators // len(zones)  # Equal distribution for fairness
    team_data = pd.DataFrame({
        'zone': np.repeat(zones, operators_per_zone),
        'x': np.random.uniform(0, factory_size, num_operators),
        'y': np.random.uniform(0, factory_size, num_operators),
        'sop_compliance': np.random.uniform(0.75, 0.95, num_operators),  # Higher baseline for fairness
        'supervisor_present': np.random.choice([True, False], num_operators, p=[0.15, 0.85])
    })

    # Initialize time-series data
    history = []
    compliance_entropy = []
    clustering_index = []
    resilience_scores = []
    oee_history = []
    productivity_loss = []
    wellbeing_scores = []
    initial_compliance_avg = team_data['sop_compliance'].mean()

    # Generate data for each time step
    for step in range(num_steps):
        # Simulate disruptions
        if step in disruption_steps:
            event = "Machine Breakdown" if step == disruption_steps[0] else "Shift Change"
            team_data['sop_compliance'] *= 0.9  # 10% compliance drop
            team_data['supervisor_present'] = team_data['supervisor_present'].sample(frac=1).values  # Shuffle supervision

        # Update SOP compliance with zone effects and supervisor influence
        for zone in zones:
            zone_mask = team_data['zone'] == zone
            zone_data = team_data[zone_mask]
            # Balanced compliance noise to avoid bias
            compliance_noise = np.random.normal(0, 0.015)
            # Supervisor influence
            if zone_data['supervisor_present'].any():
                avg_compliance = zone_data['sop_compliance'].mean()
                team_data.loc[zone_mask, 'sop_compliance'] += adaptation_rate * supervisor_influence * (avg_compliance - zone_data['sop_compliance'].mean())
            # Zone effects
            x_mean, y_mean = zone_data['x'].mean(), zone_data['y'].mean()
            if zone == 'assembly' and 20 < x_mean < 40 and 20 < y_mean < 40:
                team_data.loc[zone_mask, 'sop_compliance'] -= 0.03  # Reduced impact for fairness
            elif zone == 'maintenance' and 60 < x_mean < 80 and 60 < y_mean < 80:
                team_data.loc[zone_mask, 'sop_compliance'] -= 0.02
            team_data.loc[zone_mask, 'sop_compliance'] += compliance_noise
            team_data.loc[zone_mask, 'sop_compliance'] = np.clip(team_data.loc[zone_mask, 'sop_compliance'], 0.5, 1.0)

        # Simulate team movement
        team_data['x'] += np.random.uniform(-0.5, 0.5, num_operators)
        team_data['y'] += np.random.uniform(-0.5, 0.5, num_operators)
        team_data['x'] = team_data['x'].clip(0, factory_size - 1)
        team_data['y'] = team_data['y'].clip(0, factory_size - 1)
        team_data['step'] = step
        history.append(team_data.copy())

        # Compliance variability (entropy)
        hist, _ = np.histogram(team_data['sop_compliance'], bins=10, range=(0.5, 1.0), density=True)
        ent = entropy(hist + 1e-9)
        compliance_entropy.append(ent)

        # Team clustering (synthetic inertia based on zone positions)
        zone_distances = [np.mean(np.sqrt((team_data[team_data['zone'] == z]['x'] - team_data[team_data['zone'] == z]['x'].mean())**2 +
                                         (team_data[team_data['zone'] == z]['y'] - team_data[team_data['zone'] == z]['y'].mean())**2))
                          for z in zones]
        compliance_var = team_data['sop_compliance'].std()
        inertia = np.mean(zone_distances) * (1 + compliance_var) + np.random.normal(0, 8)  # Reduced noise for clarity
        clustering_index.append(max(inertia, 0))

        # Resilience: recovery from initial compliance
        deviation = abs(team_data['sop_compliance'].mean() - initial_compliance_avg)
        resilience_scores.append(1 - deviation / initial_compliance_avg)

        # OEE: tied to compliance
        avg_compliance = team_data['sop_compliance'].mean()
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
        loss = max(0, initial_compliance_avg - team_data['sop_compliance'].mean()) * 100
        productivity_loss.append(loss)

        # Well-being score: based on workload, disruptions, and shift duration
        workload_intensity = 1 - avg_compliance  # Lower compliance indicates higher workload
        disruption_impact = 0.1 if step in disruption_steps else 0
        shift_fatigue = min(step / num_steps, 1) * 0.2  # Fatigue increases over shift
        wellbeing = max(0, 1 - (workload_intensity + disruption_impact + shift_fatigue) + np.random.normal(0, 0.05))
        wellbeing_scores.append(wellbeing)

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
        productivity_loss,
        wellbeing_scores
    )

def plot_compliance_variability(compliance_entropy, disruption_steps, forecast=None):
    """Plot SOP compliance variability over shift."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(compliance_entropy, label="Team Compliance Variability", color='#1f77b4')
    if forecast is not None:
        ax.plot(range(len(compliance_entropy), len(compliance_entropy) + 10), forecast, '--', label="Predicted Variability", color='#ff7f0e')
    for i, s in enumerate(disruption_steps):
        label = "Machine Breakdown" if i == 0 else "Shift Change"
        ax.axvline(x=s, color='red', linestyle='--', label=label)
    ax.set_xlabel("Time (2-min Intervals)")
    ax.set_ylabel("Compliance Variability (Entropy)")
    ax.set_title("Team SOP Compliance Variability")
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
    ax.set_title("Team Cohesion Across Zones")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_resilience(resilience_scores):
    """Plot production resilience score."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resilience_scores, label="Team Resilience", color='#9467bd')
    ax.set_xlabel("Time (2-min Intervals)")
    ax.set_ylabel("Resilience Score (1 = Full Recovery)")
    ax.set_title("Team Resilience After Disruptions")
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
    """Plot heatmap of team density on factory floor."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        data=history_df,
        x='x',
        y='y',
        fill=True,
        cmap='viridis',
        ax=ax
    )
    ax.set_title("Team Density Heatmap on Factory Floor")
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.set_xlim(0, factory_size)
    ax.set_ylim(0, factory_size)
    return fig

def plot_wellbeing(wellbeing_scores):
    """Plot team well-being scores over shift."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wellbeing_scores, label="Team Well-Being Score", color='#17becf')
    ax.set_xlabel("Time (2-min Intervals)")
    ax.set_ylabel("Well-Being Score (1 = Optimal)")
    ax.set_title("Team Well-Being Trends")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig
