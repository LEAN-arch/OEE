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

def generate_synthetic_data(num_operators, num_steps, factory_size, adaptation_rate, supervisor_influence, disruption_steps, worker_priority):
    """Generate synthetic factory data with humane working conditions."""
    zones = ['assembly', 'maintenance', 'packaging']
    operators_per_zone = num_operators // len(zones)
    team_data = pd.DataFrame({
        'zone': np.repeat(zones, operators_per_zone),
        'sop_compliance': np.random.uniform(0.75, 0.95, num_operators),
        'supervisor_present': np.random.choice([True, False], num_operators, p=[0.2, 0.8]),
    })

    history = []
    compliance_entropy = []
    clustering_index = []
    resilience_scores = []
    oee_history = []
    productivity_loss = []
    wellbeing_scores = []
    safety_scores = []
    zone_wellbeing = {zone: [] for zone in zones}
    zone_rest_quality = {zone: [] for zone in zones}
    initial_compliance_avg = team_data['sop_compliance'].mean()

    # Well-being action impacts
    break_boost = 0.15
    supervisor_boost = 0.1
    workload_balance_boost = 0.15
    ergonomic_boost = 0.12
    worker_action_boost = {'More frequent breaks': 0.2, 'Task reduction': 0.18, 'Wellness resources': 0.15, 'Team recognition': 0.1}

    # Triggers
    threshold_triggers = []
    trend_triggers = []
    zone_triggers = {zone: [] for zone in zones}
    disruption_triggers = []
    from config import WELLBEING_THRESHOLD, WELLBEING_TREND_LENGTH, WELLBEING_DISRUPTION_WINDOW, BREAK_INTERVAL, WORKLOAD_CAP_STEPS

    # Feedback impact
    feedback_impact = {'wellbeing': 0, 'cohesion': 0}
    worker_actions_applied = 0

    # Synthetic positions for density (zone-level, not individual)
    zone_centers = {'assembly': (20, 20), 'maintenance': (60, 60), 'packaging': (80, 80)}
    for step in range(num_steps):
        # Proactive breaks
        if step % BREAK_INTERVAL == 0 and step > 0:
            team_data['sop_compliance'] += break_boost * 0.1
            for zone in zones:
                zone_rest_quality[zone].append(np.random.uniform(0.8, 1.0))
        else:
            for zone in zones:
                zone_rest_quality[zone].append(np.random.uniform(0.5, 0.7))

        # Simulate disruptions
        if step in disruption_steps:
            team_data['sop_compliance'] *= 0.85
            team_data['supervisor_present'] = team_data['supervisor_present'].sample(frac=1).values

        # Update SOP compliance and well-being
        for zone in zones:
            zone_mask = team_data['zone'] == zone
            zone_data = team_data[zone_mask]
            compliance_noise = np.random.normal(0, 0.01)
            if zone_data['supervisor_present'].any():
                avg_compliance = zone_data['sop_compliance'].mean()
                team_data.loc[zone_mask, 'sop_compliance'] += adaptation_rate * supervisor_influence * (avg_compliance - zone_data['sop_compliance'].mean())
            team_data.loc[zone_mask, 'sop_compliance'] += compliance_noise
            team_data.loc[zone_mask, 'sop_compliance'] = np.clip(team_data.loc[zone_mask, 'sop_compliance'], 0.6, 1.0)

            # Workload cap
            if step >= WORKLOAD_CAP_STEPS and all(team_data.loc[zone_mask, 'sop_compliance'].mean() < 0.7 for _ in range(WORKLOAD_CAP_STEPS)):
                team_data.loc[zone_mask, 'sop_compliance'] += workload_balance_boost * 0.1
                zone_triggers[zone].append(step)

            # Zone well-being
            workload_intensity = 0.5 * (1 - zone_data['sop_compliance'].mean())
            disruption_impact = 0.05 if step in disruption_steps else 0
            shift_fatigue = min(step / num_steps, 1) * 0.1
            rest_quality = zone_rest_quality[zone][-1]
            zone_wellbeing_score = max(0, 1 - (workload_intensity + disruption_impact + shift_fatigue) + rest_quality * 0.3 + np.random.normal(0, 0.03))
            zone_wellbeing[zone].append(zone_wellbeing_score)

        # Well-being actions
        avg_wellbeing = np.mean([zone_wellbeing[zone][-1] for zone in zones])
        if avg_wellbeing < WELLBEING_THRESHOLD:
            team_data['sop_compliance'] += (break_boost + ergonomic_boost) * 0.1
            threshold_triggers.append(step)
        if step >= WELLBEING_TREND_LENGTH and all(zone_wellbeing[zone][-i] < zone_wellbeing[zone][-i-1] for zone in zones for i in range(1, WELLBEING_TREND_LENGTH)):
            team_data['sop_compliance'] += workload_balance_boost * 0.1
            trend_triggers.append(step)
        for zone in zones:
            if zone_wellbeing[zone][-1] < min([zone_wellbeing[z][-1] for z in zones if z != zone]) - 0.08:
                team_data.loc[team_data['zone'] == zone, 'supervisor_present'] = True
                team_data.loc[team_data['zone'] == zone, 'sop_compliance'] += supervisor_boost * 0.1
                zone_triggers[zone].append(step)
        if any(abs(step - d) <= WELLBEING_DISRUPTION_WINDOW for d in disruption_steps) and avg_wellbeing < WELLBEING_THRESHOLD:
            team_data['sop_compliance'] += break_boost * 0.1
            disruption_triggers.append(step)

        # Worker-initiated actions
        if random.random() < 0.1:
            team_data['sop_compliance'] += worker_action_boost[worker_priority] * 0.1
            feedback_impact['wellbeing'] += worker_action_boost[worker_priority] * 0.05
            feedback_impact['cohesion'] += worker_action_boost[worker_priority] * 0.03
            worker_actions_applied += 1

        # Generate synthetic positions for density
        positions = []
        for zone in zones:
            center_x, center_y = zone_centers[zone]
            for _ in range(operators_per_zone):
                x = np.clip(np.random.normal(center_x, 10), 0, factory_size)
                y = np.clip(np.random.normal(center_y, 10), 0, factory_size)
                positions.append({'zone': zone, 'x': x, 'y': y, 'step': step})
        history.append(pd.DataFrame(positions))

        # Compliance variability
        hist, _ = np.histogram(team_data['sop_compliance'], bins=10, range=(0.5, 1.0), density=True)
        compliance_entropy.append(entropy(hist + 1e-9))

        # Team clustering
        cohesion_score = np.mean([1 / (1 + zone_wellbeing[zone][-1]) for zone in zones])
        clustering_index.append(cohesion_score + np.random.normal(0, 0.05))

        # Resilience
        deviation = abs(team_data['sop_compliance'].mean() - initial_compliance_avg)
        resilience_scores.append(1 - deviation / initial_compliance_avg)

        # OEE
        avg_compliance = team_data['sop_compliance'].mean()
        availability = min(0.92 + 0.08 * avg_compliance, 0.96) + np.random.normal(0, 0.004)
        performance = min(0.87 + 0.08 * avg_compliance, 0.92) + np.random.normal(0, 0.004)
        quality = min(0.98 + 0.02 * avg_compliance, 0.99) + np.random.normal(0, 0.001)
        oee = availability * performance * quality
        oee_history.append({
            'step': step,
            'availability': availability,
            'performance': performance,
            'quality': quality,
            'oee': oee
        })

        # Productivity loss
        loss = max(0, initial_compliance_avg - team_data['sop_compliance'].mean()) * 50
        productivity_loss.append(loss)

        # Well-being and psychological safety
        wellbeing_scores.append(avg_wellbeing)
        safety_score = np.mean([zone_wellbeing[zone][-1] * (1 + 0.3 if team_data[team_data['zone'] == zone]['supervisor_present'].any() else 0) * (1 + 0.2 * worker_actions_applied / (step + 1)) for zone in zones])
        safety_scores.append(safety_score)

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
        {'scores': wellbeing_scores, 'triggers': {
            'threshold': threshold_triggers,
            'trend': trend_triggers,
            'zone': zone_triggers,
            'disruption': disruption_triggers
        }},
        safety_scores,
        feedback_impact
    )

def plot_compliance_variability(compliance_entropy, disruption_steps, forecast=None):
    """Plot SOP compliance consistency."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(compliance_entropy, label="Compliance Consistency (Lower = More Uniform)", color='#1f77b4')
    if forecast is not None:
        ax.plot(range(len(compliance_entropy), len(compliance_entropy) + 10), forecast, '--', label="Predicted Consistency", color='#ff7f0e')
    for i, s in enumerate(disruption_steps):
        label = "Machine Breakdown" if i == 0 else "Shift Change"
        ax.axvline(x=s, color='red', linestyle='--', label=label)
    ax.set_xlabel("Shift Time (2-min Intervals)")
    ax.set_ylabel("Compliance Variation")
    ax.set_title("Team SOP Compliance Consistency")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_team_clustering(clustering_index, forecast=None):
    """Plot team collaboration strength."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(clustering_index, label="Collaboration Strength (Higher = Stronger Teams)", color='#2ca02c')
    if forecast is not None:
        ax.plot(range(len(clustering_index), len(clustering_index) + 10), forecast, '--', label="Predicted Collaboration", color='#d62728')
    ax.set_xlabel("Shift Time (2-min Intervals)")
    ax.set_ylabel("Collaboration Score")
    ax.set_title("Team Collaboration Strength")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_resilience(resilience_scores):
    """Plot team recovery after disruptions."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resilience_scores, label="Recovery Strength (1 = Full Recovery)", color='#9467bd')
    ax.set_xlabel("Shift Time (2-min Intervals)")
    ax.set_ylabel("Recovery Score")
    ax.set_title("Team Recovery After Disruptions")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_oee(oee_df):
    """Plot equipment efficiency (OEE)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    oee_df.set_index('step')[['availability', 'performance', 'quality', 'oee']].plot(
        ax=ax,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        label=['Availability (>90% Target)', 'Performance (>85% Target)', 'Quality (>97% Target)', 'Overall OEE']
    )
    ax.set_ylabel("Efficiency (%)")
    ax.set_xlabel("Shift Time (2-min Intervals)")
    ax.set_title("Equipment Efficiency (OEE)")
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_worker_density(history_df, factory_size):
    """Plot factory floor activity and congestion."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hexbin(
        history_df['x'], history_df['y'],
        gridsize=20, cmap='viridis', mincnt=1,
        extent=(0, factory_size, 0, factory_size)
    )
    cb = plt.colorbar(label='Activity Level (Low to High Congestion)')
    cb.set_ticks([cb.get_clim()[0], cb.get_clim()[1]])
    cb.set_ticklabels(['Low Activity', 'High Congestion'])
    
    # Add factory layout
    ax.plot([0, factory_size], [0, 0], 'k-', lw=1)
    ax.plot([0, factory_size], [factory_size, factory_size], 'k-', lw=1)
    ax.plot([0, 0], [0, factory_size], 'k-', lw=1)
    ax.plot([factory_size, factory_size], [0, factory_size], 'k-', lw=1)
    ax.text(20, 20, 'Assembly', color='white', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
    ax.text(60, 60, 'Maintenance', color='white', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
    ax.text(80, 80, 'Packaging', color='white', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
    
    ax.set_title("Factory Floor Activity and Congestion")
    ax.set_xlabel("Factory Width (m)")
    ax.set_ylabel("Factory Length (m)")
    ax.set_xlim(0, factory_size)
    ax.set_ylim(0, factory_size)
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_wellbeing(wellbeing_scores):
    """Plot team well-being trends."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wellbeing_scores, label="Well-Being Score (1 = Optimal Health)", color='#17becf')
    ax.set_xlabel("Shift Time (2-min Intervals)")
    ax.set_ylabel("Well-Being Score")
    ax.set_title("Team Well-Being Trends")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_psychological_safety(safety_scores):
    """Plot team psychological safety trends."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(safety_scores, label="Psychological Safety (1 = High Trust)", color='#ff9896')
    ax.set_xlabel("Shift Time (2-min Intervals)")
    ax.set_ylabel("Safety Score")
    ax.set_title("Team Psychological Safety Trends")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig
