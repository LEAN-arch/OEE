import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances
from config import CONFIG

def simulate_workplace_operations(
    num_team_members=51,
    num_steps=240,
    workplace_size=100,
    adaptation_rate=0.05,
    supervisor_influence=0.2,
    disruption_steps=None,
    team_initiative="More frequent breaks",
    skip_forecast=False,
    config=None
):
    """Simulate industrial workplace operations with team positions, compliance, and efficiency."""
    if config is None:
        config = CONFIG
    
    if disruption_steps is None:
        disruption_steps = config['DISRUPTION_STEPS']
    
    # Initialize team positions
    positions = np.random.uniform(0, workplace_size, (num_steps, num_team_members, 2))
    compliance = np.random.uniform(0.6, 1.0, (num_steps, num_team_members))
    wellbeing = np.ones((num_steps, num_team_members)) * 0.8
    safety = np.ones(num_steps) * config['SAFETY_THRESHOLD']
    efficiency = {'uptime': [], 'throughput': [], 'quality': [], 'oee': []}
    productivity_loss = np.zeros(num_steps)
    collaboration_index = {'data': [], 'z_scores': [], 'forecast': None}
    
    # Simulate disruptions and team initiatives
    for t in range(num_steps):
        if t in disruption_steps:
            compliance[t] *= np.random.uniform(0.7, 0.9)
            wellbeing[t] *= np.random.uniform(0.6, 0.8)
            productivity_loss[t] = np.random.uniform(5, 15)
        
        if team_initiative == "More frequent breaks" and t % config['BREAK_INTERVAL'] == 0:
            wellbeing[t] = np.minimum(wellbeing[t] + 0.1, 1.0)
        
        compliance[t] = np.clip(compliance[t] + adaptation_rate * supervisor_influence, 0, 1)
        
        uptime = np.random.uniform(0.85, 0.95)
        throughput = np.random.uniform(0.80, 0.90)
        quality = np.random.uniform(0.90, 0.98)
        oee = uptime * throughput * quality
        efficiency['uptime'].append(uptime)
        efficiency['throughput'].append(throughput)
        efficiency['quality'].append(quality)
        efficiency['oee'].append(oee)
        
        distances = euclidean_distances(positions[t])
        collab = np.mean(distances < workplace_size / 4)
        collaboration_index['data'].append(collab)
    
    # Convert to DataFrame
    team_positions_df = pd.DataFrame({
        'step': np.repeat(np.arange(num_steps), num_team_members),
        'team_member_id': np.tile(np.arange(num_team_members), num_steps),
        'x': positions[:, :, 0].flatten(),
        'y': positions[:, :, 1].flatten()
    })
    
    efficiency_metrics_df = pd.DataFrame(efficiency)
    
    # Compliance variability
    compliance_entropy = [entropy(compliance[t]) for t in range(num_steps)]
    compliance_z_scores = (compliance_entropy - np.mean(compliance_entropy)) / np.std(compliance_entropy)
    compliance_variability = {
        'data': compliance_entropy,
        'z_scores': compliance_z_scores,
        'forecast': np.polyval(np.polyfit(np.arange(num_steps), compliance_entropy, 1), np.arange(num_steps)) if not skip_forecast else None
    }
    
    # Collaboration z-scores
    collab_data = collaboration_index['data']
    collaboration_z_scores = (collab_data - np.mean(collab_data)) / np.std(collab_data)
    collaboration_index['z_scores'] = collaboration_z_scores
    if not skip_forecast:
        collaboration_index['forecast'] = np.polyval(np.polyfit(np.arange(num_steps), collab_data, 1), np.arange(num_steps))
    
    # Operational resilience
    operational_resilience = np.ones(num_steps)
    for t in disruption_steps:
        operational_resilience[t:t+config['WELLBEING_DISRUPTION_WINDOW']] *= np.linspace(0.6, 1.0, config['WELLBEING_DISRUPTION_WINDOW'])
    
    # Well-being triggers
    triggers = {
        'threshold': [t for t in range(num_steps) if np.mean(wellbeing[t]) < config['WELLBEING_THRESHOLD']],
        'trend': [
            t for t in range(config['WELLBEING_TREND_LENGTH'], num_steps)
            if all(wellbeing[t-i] < wellbeing[t-i-1] for i in range(config['WELLBEING_TREND_LENGTH']))
        ],
        'work_area': {ws: [] for ws in config['WORK_AREAS']},
        'disruption': [
            t for t in range(num_steps) if any(
                abs(t - d) <= config['WELLBEING_DISRUPTION_WINDOW'] for d in disruption_steps
            )
        ]
    }
    team_wellbeing = {'scores': np.mean(wellbeing, axis=1), 'triggers': triggers}
    
    # Team feedback impact
    feedback_impact = {
        'wellbeing': 0.05 if team_initiative == "More frequent breaks" else 0.03,
        'cohesion': 0.04 if team_initiative == "Team recognition" else 0.02
    }
    
    return (
        team_positions_df,
        compliance_variability,
        collaboration_index,
        operational_resilience,
        efficiency_metrics_df,
        productivity_loss,
        team_wellbeing,
        safety,
        feedback_impact
    )

def plot_task_compliance_trend(compliance_data, disruption_steps, forecast=None):
    fig, ax = plt.subplots()
    ax.plot(compliance_data, label='Compliance Variability')
    for t in disruption_steps:
        ax.axvline(t, color='red', linestyle='--', alpha=0.5)
    if forecast is not None:
        ax.plot(forecast, label='Forecast', linestyle='--')
    ax.set_xlabel('Shift Interval (2-min)')
    ax.set_ylabel('Compliance Entropy')
    ax.legend()
    return fig

def plot_worker_collaboration_trend(collab_data, forecast=None):
    fig, ax = plt.subplots()
    ax.plot(collab_data, label='Collaboration Index')
    if forecast is not None:
        ax.plot(forecast, label='Forecast', linestyle='--')
    ax.set_xlabel('Shift Interval (2-min)')
    ax.set_ylabel('Collaboration Strength')
    ax.legend()
    return fig

def plot_operational_resilience(resilience):
    fig, ax = plt.subplots()
    ax.plot(resilience, label='Resilience')
    ax.set_xlabel('Shift Interval (2-min)')
    ax.set_ylabel('Resilience Score')
    ax.legend()
    return fig

def plot_operational_efficiency(efficiency_df):
    fig, ax = plt.subplots()
    ax.plot(efficiency_df['uptime'], label='Uptime')
    ax.plot(efficiency_df['throughput'], label='Throughput')
    ax.plot(efficiency_df['quality'], label='Quality')
    ax.plot(efficiency_df['oee'], label='OEE', linewidth=2)
    ax.set_xlabel('Shift Interval (2-min)')
    ax.set_ylabel('Efficiency')
    ax.legend()
    return fig

def plot_worker_distribution(team_positions_df, workplace_size, config, use_plotly=True):
    if use_plotly:
        fig = px.scatter(
            team_positions_df,
            x='x', y='y',
            animation_frame='step',
            color='team_member_id',
            range_x=[0, workplace_size],
            range_y=[0, workplace_size],
            title=f"Team Distribution ({config['WORKPLACE_TYPE'].capitalize()} Workplace)"
        )
        return fig
    else:
        fig, ax = plt.subplots()
        hb = ax.hexbin(
            team_positions_df['x'],
            team_positions_df['y'],
            gridsize=config['HEXBIN_GRIDSIZE'],
            cmap='Blues',
            mincnt=1
        )
        ax.scatter(team_positions_df['x'], team_positions_df['y'], c='red', s=10, alpha=0.5)
        ax.set_xlim(0, workplace_size)
        ax.set_ylim(0, workplace_size)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        fig.colorbar(hb, ax=ax, label='Team Member Count')
        return fig

def plot_worker_wellbeing(wellbeing_scores):
    fig, ax = plt.subplots()
    ax.plot(wellbeing_scores, label='Well-Being')
    ax.set_xlabel('Shift Interval (2-min)')
    ax.set_ylabel('Well-Being Score')
    ax.legend()
    return fig

def plot_psychological_safety(safety_scores):
    fig, ax = plt.subplots()
    ax.plot(safety_scores, label='Psychological Safety')
    ax.set_xlabel('Shift Interval (2-min)')
    ax.set_ylabel('Safety Score')
    ax.legend()
    return fig

