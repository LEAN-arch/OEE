"""
simulation.py
Core simulation logic for the Industrial Workplace Shift Monitoring Dashboard.
Simulates team positions, compliance, well-being, and efficiency metrics with SI units.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from config import DEFAULT_CONFIG, validate_config

logger = logging.getLogger(__name__)

def simulate_workplace_operations(
    num_team_members: int = None,
    num_steps: int = None,
    workplace_size: float = None,
    adaptation_rate: float = None,
    supervisor_influence: float = None,
    disruption_intervals: list = None,
    team_initiative: str = "More frequent breaks",
    skip_forecast: bool = False,
    config: dict = None
) -> tuple:
    """
    Simulate industrial workplace operations with actionable metrics.

    Args:
        num_team_members (int, optional): Number of team members.
        num_steps (int, optional): Number of 2-minute intervals.
        workplace_size (float, optional): Facility size (meters).
        adaptation_rate (float, optional): Rate of compliance improvement.
        supervisor_influence (float, optional): Supervisor impact on compliance.
        disruption_intervals (list, optional): Intervals (2-min) for disruptions.
        team_initiative (str): Strategy ("More frequent breaks" or "Team recognition").
        skip_forecast (bool): Skip forecasting for compliance and collaboration.
        config (dict, optional): Configuration dictionary.

    Returns:
        tuple: team_positions_df, task_compliance, collaboration_proximity,
               operational_recovery, efficiency_metrics_df, productivity_loss,
               worker_wellbeing, psychological_safety, feedback_impact, downtime_minutes,
               task_completion_rate.
    """
    config = config or DEFAULT_CONFIG
    validate_config(config)

    num_team_members = num_team_members or config['TEAM_SIZE']
    num_steps = num_steps or config['SHIFT_DURATION_INTERVALS']
    workplace_size = workplace_size or config['FACILITY_SIZE']
    adaptation_rate = adaptation_rate or config['ADAPTATION_RATE']
    supervisor_influence = supervisor_influence or config['SUPERVISOR_INFLUENCE']
    disruption_intervals = disruption_intervals or config['DISRUPTION_INTERVALS']

    # Assign workers to zones
    zone_assignments = []
    worker_ids = np.arange(num_team_members)
    for zone, zone_info in config['WORK_AREAS'].items():
        zone_assignments.extend([zone] * zone_info['workers'])
    zone_assignments = np.array(zone_assignments)

    # Initialize arrays
    positions = np.zeros((num_steps, num_team_members, 2))
    for i, zone in enumerate(zone_assignments):
        center = config['WORK_AREAS'][zone]['center']
        positions[:, i, :] = np.random.normal(center, workplace_size / 20, (num_steps, 2))
        positions[:, i, :] = np.clip(positions[:, i, :], 0, workplace_size)

    task_compliance = np.random.uniform(0.7, 1.0, (num_steps, num_team_members)) * 100  # 70-100%
    worker_wellbeing = np.ones((num_steps, num_team_members)) * 80  # 80% initial
    psychological_safety = np.ones(num_steps) * config['SAFETY_THRESHOLD'] * 100  # 70%
    efficiency = {'uptime': [], 'throughput': [], 'quality': [], 'oee': []}
    productivity_loss = np.zeros(num_steps)
    downtime_minutes = np.zeros(num_steps)
    task_completion_rate = np.random.uniform(0.85, 0.95, num_steps) * 100  # 85-95%
    collaboration_proximity = {'data': [], 'z_scores': [], 'forecast': None}

    # Fatigue model: Well-being decreases by 0.5% per 30 minutes (15 intervals)
    fatigue_rate = 0.005 / 15
    for t in range(num_steps):
        # Apply disruptions
        if t in disruption_intervals:
            task_compliance[t] *= np.random.uniform(0.8, 0.9)  # 10-20% drop
            worker_wellbeing[t] *= np.random.uniform(0.7, 0.85)  # 15-30% drop
            psychological_safety[t] *= np.random.uniform(0.75, 0.9)  # 10-25% drop
            productivity_loss[t] = np.random.uniform(8, 20)  # 8-20%
            downtime_minutes[t] = np.random.uniform(5, 15)  # 5-15 minutes
            task_completion_rate[t] *= np.random.uniform(0.7, 0.9)  # 10-30% drop

        # Apply team initiative
        if team_initiative == "More frequent breaks" and t % config['BREAK_FREQUENCY_INTERVALS'] == 0:
            worker_wellbeing[t] = np.minimum(worker_wellbeing[t] + 10, 100)  # +10%
            psychological_safety[t] = np.minimum(psychological_safety[t] + 5, 100)  # +5%
        elif team_initiative == "Team recognition":
            psychological_safety[t] = np.minimum(psychological_safety[t] + 3, 100)  # +3%

        # Fatigue effect
        worker_wellbeing[t] = np.maximum(worker_wellbeing[t] - fatigue_rate * t, 0)

        # Compliance and safety dynamics
        task_compliance[t] = np.clip(task_compliance[t] + adaptation_rate * supervisor_influence * 100, 0, 100)
        psychological_safety[t] = np.clip(psychological_safety[t] + 0.01 * np.mean(task_compliance[t]), 0, 100)

        # Efficiency metrics
        uptime = np.random.uniform(0.75, 0.95)  # 75-95%
        throughput = np.random.uniform(0.70, 0.90)  # 70-90%
        quality = np.random.uniform(0.85, 0.98)  # 85-98%
        oee = uptime * throughput * quality * 100  # 60-90%
        efficiency['uptime'].append(uptime * 100)
        efficiency['throughput'].append(throughput * 100)
        efficiency['quality'].append(quality * 100)
        efficiency['oee'].append(oee)

        # Collaboration proximity
        collab = 0
        for zone in config['WORK_AREAS']:
            zone_workers = zone_assignments == zone
            if np.sum(zone_workers) > 1:
                distances = euclidean_distances(positions[t, zone_workers])
                collab += np.mean(distances < 5) * np.sum(zone_workers) / num_team_members * 100  # % within 5m
        collaboration_proximity['data'].append(collab)

    # DataFrames
    team_positions_df = pd.DataFrame({
        'step': np.repeat(np.arange(num_steps), num_team_members),
        'team_member_id': np.tile(worker_ids, num_steps),
        'x': positions[:, :, 0].flatten(),
        'y': positions[:, :, 1].flatten(),
        'zone': np.tile(zone_assignments, num_steps)
    })
    efficiency_metrics_df = pd.DataFrame(efficiency)

    # Task compliance
    task_compliance_scores = np.mean(task_compliance, axis=1)
    z_scores = (task_compliance_scores - np.mean(task_compliance_scores)) / np.std(task_compliance_scores)
    task_compliance = {
        'data': task_compliance_scores,
        'z_scores': z_scores,
        'forecast': np.polyval(np.polyfit(np.arange(num_steps), task_compliance_scores, 1), np.arange(num_steps)) if not skip_forecast else None
    }

    # Collaboration z-scores
    collab_data = collaboration_proximity['data']
    collaboration_z_scores = (collab_data - np.mean(collab_data)) / np.std(collab_data)
    collaboration_proximity['z_scores'] = collaboration_z_scores
    if not skip_forecast:
        collaboration_proximity['forecast'] = np.polyval(np.polyfit(np.arange(num_steps), collab_data, 1), np.arange(num_steps))

    # Operational recovery
    operational_recovery = np.ones(num_steps) * 100
    for t in disruption_intervals:
        operational_recovery[t:t+config['DISRUPTION_RECOVERY_WINDOW']] *= np.linspace(0.7, 1.0, config['DISRUPTION_RECOVERY_WINDOW'])

    # Well-being triggers
    wellbeing_means = np.mean(worker_wellbeing, axis=1)
    triggers = {
        'threshold': [t for t in range(num_steps) if wellbeing_means[t] < config['WELLBEING_THRESHOLD'] * 100],
        'trend': [
            t for t in range(config['WELLBEING_TREND_WINDOW'], num_steps)
            if all(wellbeing_means[t-i] < wellbeing_means[t-i-1] for i in range(1, config['WELLBEING_TREND_WINDOW'] + 1))
        ],
        'work_area': {
            zone: [t for t in range(num_steps) if np.mean(worker_wellbeing[t, zone_assignments == zone]) < config['WELLBEING_THRESHOLD'] * 100]
            for zone in config['WORK_AREAS']
        },
        'disruption': [
            t for t in range(num_steps) if any(
                abs(t - d) <= config['DISRUPTION_RECOVERY_WINDOW'] for d in disruption_intervals
            )
        ]
    }
    worker_wellbeing = {'scores': wellbeing_means, 'triggers': triggers}

    # Feedback impact
    feedback_impact = {
        'wellbeing': 5 if team_initiative == "More frequent breaks" else 3,
        'cohesion': 4 if team_initiative == "Team recognition" else 2
    }

    return (
        team_positions_df,
        task_compliance,
        collaboration_proximity,
        operational_recovery,
        efficiency_metrics_df,
        productivity_loss,
        worker_wellbeing,
        psychological_safety,
        feedback_impact,
        downtime_minutes,
        task_completion_rate
    )
