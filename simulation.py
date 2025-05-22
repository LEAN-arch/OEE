import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances
from config import CONFIG

def simulate_workplace_operations(
    num_team_members=None,
    num_steps=None,
    workplace_size=None,
    adaptation_rate=None,
    supervisor_influence=None,
    disruption_intervals=None,
    team_initiative="More frequent breaks",
    skip_forecast=False,
    config=None
):
    """Simulate industrial workplace operations with team positions, compliance, and efficiency."""
    if config is None:
        config = CONFIG
    
    # Use config defaults if parameters are None
    num_team_members = num_team_members or config['TEAM_SIZE']
    num_steps = num_steps or config['SHIFT_DURATION_INTERVALS']
    workplace_size = workplace_size or config['FACILITY_SIZE']
    adaptation_rate = adaptation_rate or config['ADAPTATION_RATE']
    supervisor_influence = supervisor_influence or config['SUPERVISOR_INFLUENCE']
    disruption_intervals = disruption_intervals or config['DISRUPTION_INTERVALS']
    
    # Validate inputs
    if num_team_members < sum(zone['workers'] for zone in config['WORK_AREAS'].values()):
        raise ValueError("Number of team members must match sum of workers in WORK_AREAS")
    
    # Assign workers to zones
    zone_assignments = []
    worker_ids = np.arange(num_team_members)
    start_idx = 0
    for zone, zone_info in config['WORK_AREAS'].items():
        num_zone_workers = zone_info['workers']
        zone_assignments.extend([zone] * num_zone_workers)
        start_idx += num_zone_workers
    zone_assignments = np.array(zone_assignments)
    
    # Initialize team positions near zone centers
    positions = np.zeros((num_steps, num_team_members, 2))
    for i, zone in enumerate(zone_assignments):
        center = config['WORK_AREAS'][zone]['center']
        positions[:, i, :] = np.random.normal(center, workplace_size / 10, (num_steps, 2))
        positions[:, i, :] = np.clip(positions[:, i, :], 0, workplace_size)
    
    compliance = np.random.uniform(0.6, 1.0, (num_steps, num_team_members))
    wellbeing = np.ones((num_steps, num_team_members)) * 0.8
    safety = np.ones(num_steps) * config['SAFETY_COMPLIANCE_THRESHOLD']
    efficiency = {'uptime': [], 'throughput': [], 'quality': [], 'oee': []}
    productivity_loss = np.zeros(num_steps)
    collaboration_index = {'data': [], 'z_scores': [], 'forecast': None}
    
    # Simulate disruptions and team initiatives
    for t in range(num_steps):
        if t in disruption_intervals:
            compliance[t] *= np.random.uniform(0.7, 0.9)
            wellbeing[t] *= np.random.uniform(0.6, 0.8)
            safety[t] *= np.random.uniform(0.6, 0.8)
            productivity_loss[t] = np.random.uniform(5, 15)
        
        if team_initiative == "More frequent breaks" and t % config['BREAK_FREQUENCY_INTERVALS'] == 0:
            wellbeing[t] = np.minimum(wellbeing[t] + 0.1, 1.0)
            safety[t] = np.minimum(safety[t] + 0.05, 1.0)
        
        compliance[t] = np.clip(compliance[t] + adaptation_rate * supervisor_influence, 0, 1)
        safety[t] = np.clip(safety[t] + 0.01 * np.mean(compliance[t]), 0, 1)
        
        uptime = np.random.uniform(0.85, 0.95)
        throughput = np.random.uniform(0.80, 0.90)
        quality = np.random.uniform(0.90, 0.98)
        oee = uptime * throughput * quality
        efficiency['uptime'].append(uptime)
        efficiency['throughput'].append(throughput)
        efficiency['quality'].append(quality)
        efficiency['oee'].append(oee)
        
        # Zone-based collaboration
        collab = 0
        for zone in config['WORK_AREAS']:
            zone_workers = zone_assignments == zone
            if np.sum(zone_workers) > 1:
                distances = euclidean_distances(positions[t, zone_workers])
                collab += np.mean(distances < workplace_size / 10) * np.sum(zone_workers) / num_team_members
        collaboration_index['data'].append(collab)
    
    # Convert to DataFrame
    team_positions_df = pd.DataFrame({
        'step': np.repeat(np.arange(num_steps), num_team_members),
        'team_member_id': np.tile(worker_ids, num_steps),
        'x': positions[:, :, 0].flatten(),
        'y': positions[:, :, 1].flatten(),
        'zone': np.tile(zone_assignments, num_steps)
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
    for t in disruption_intervals:
        operational_resilience[t:t+config['DISRUPTION_RECOVERY_WINDOW']] *= np.linspace(0.6, 1.0, config['DISRUPTION_RECOVERY_WINDOW'])
    
    # Well-being triggers
    wellbeing_means = np.mean(wellbeing, axis=1)  # Compute mean well-being per step
    triggers = {
        'threshold': [t for t in range(num_steps) if wellbeing_means[t] < config['WELLBEING_THRESHOLD']],
        'trend': [
            t for t in range(config['WELLBEING_TREND_WINDOW'], num_steps)
            if all(wellbeing_means[t-i] < wellbeing_means[t-i-1] for i in range(1, config['WELLBEING_TREND_WINDOW'] + 1))
        ],
        'work_area': {
            zone: [t for t in range(num_steps) if np.mean(wellbeing[t, zone_assignments == zone]) < config['WELLBEING_THRESHOLD']]
            for zone in config['WORK_AREAS']
        },
        'disruption': [
            t for t in range(num_steps) if any(
                abs(t - d) <= config['DISRUPTION_RECOVERY_WINDOW'] for d in disruption_intervals
            )
        ]
    }
    team_wellbeing = {'scores': wellbeing_means, 'triggers': triggers}
    
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

def plot_task_compliance_trend(compliance_data, disruption_intervals, forecast=None):
    fig = px.line(
        x=list(range(len(compliance_data))),
        y=compliance_data,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Compliance Entropy'},
        title='Task Compliance Variability'
    )
    for t in disruption_intervals:
        fig.add_vline(x=t, line_dash="dash", line_color="red", opacity=0.5)
    if forecast is not None:
        fig.add_scatter(x=list(range(len(forecast))), y=forecast, name='Forecast', line=dict(dash='dash'))
    return fig

def plot_worker_collaboration_trend(collab_data, forecast=None):
    fig = px.line(
        x=list(range(len(collab_data))),
        y=collab_data,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Collaboration Strength'},
        title='Worker Collaboration Index'
    )
    if forecast is not None:
        fig.add_scatter(x=list(range(len(forecast))), y=forecast, name='Forecast', line=dict(dash='dash'))
    return fig

def plot_operational_resilience(resilience):
    fig = px.line(
        x=list(range(len(resilience))),
        y=resilience,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Resilience Score'},
        title='Operational Resilience'
    )
    return fig

def plot_operational_efficiency(efficiency_df):
    fig = px.line(
        efficiency_df,
        x=efficiency_df.index,
        y=['uptime', 'throughput', 'quality', 'oee'],
        labels={'value': 'Efficiency', 'index': 'Shift Interval (2-min)'},
        title='Operational Efficiency Metrics'
    )
    return fig

def plot_worker_distribution(team_positions_df, workplace_size, config, use_plotly=True):
    if use_plotly:
        fig = px.scatter(
            team_positions_df,
            x='x', y='y',
            animation_frame='step',
            color='zone',
            hover_data=['team_member_id'],
            range_x=[0, workplace_size],
            range_y=[0, workplace_size],
            title=f"Team Distribution ({config['FACILITY_TYPE'].capitalize()} Workplace)"
        )
        for zone, info in config['WORK_AREAS'].items():
            fig.add_scatter(x=[info['center'][0]], y=[info['center'][1]], mode='markers+text',
                            text=[info['label']], marker=dict(size=15, color='red'), name=zone)
        return fig
    else:
        fig, ax = plt.subplots()
        hb = ax.hexbin(
            team_positions_df['x'],
            team_positions_df['y'],
            gridsize=config['DENSITY_GRID_SIZE'],
            cmap='Blues',
            mincnt=1
        )
        for zone, info in config['WORK_AREAS'].items():
            ax.scatter(info['center'][0], info['center'][1], c='red', s=100, label=info['label'])
            ax.text(info['center'][0], info['center'][1], info['label'], fontsize=10)
        ax.set_xlim(0, workplace_size)
        ax.set_ylim(0, workplace_size)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        fig.colorbar(hb, ax=ax, label='Team Member Count')
        ax.legend()
        return fig

def plot_worker_wellbeing(wellbeing_scores):
    fig = px.line(
        x=list(range(len(wellbeing_scores))),
        y=wellbeing_scores,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Well-Being Score'},
        title='Team Well-Being'
    )
    fig.add_hline(y=CONFIG['WELLBEING_THRESHOLD'], line_dash="dash", line_color="red", annotation_text="Threshold")
    return fig

def plot_psychological_safety(safety_scores):
    fig = px.line(
        x=list(range(len(safety_scores))),
        y=safety_scores,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Safety Score'},
        title='Psychological Safety'
    )
    fig.add_hline(y=CONFIG['SAFETY_COMPLIANCE_THRESHOLD'], line_dash="dash", line_color="red", annotation_text="Threshold")
    return fig
