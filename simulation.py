"""
simulation.py
Core simulation logic for the Industrial Workplace Shift Monitoring Dashboard.
Simulates team positions, compliance, well-being, and efficiency metrics.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import entropy
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
    Simulate industrial workplace operations.

    Args:
        num_team_members (int, optional): Number of team members.
        num_steps (int, optional): Number of time steps.
        workplace_size (float, optional): Size of the workplace (meters).
        adaptation_rate (float, optional): Rate of compliance adaptation.
        supervisor_influence (float, optional): Influence of supervisor on compliance.
        disruption_intervals (list, optional): Time steps for disruptions.
        team_initiative (str): Team initiative ("More frequent breaks" or "Team recognition").
        skip_forecast (bool): Skip forecasting for compliance and collaboration.
        config (dict, optional): Configuration dictionary.

    Returns:
        tuple: Contains team_positions_df, compliance_variability, collaboration_index,
               operational_resilience, efficiency_metrics_df, productivity_loss,
               team_wellbeing, safety, feedback_impact.

    Raises:
        ValueError: If inputs are invalid.
    """
    config = config or DEFAULT_CONFIG
    validate_config(config)

    # Use config defaults if parameters are None
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
        positions[:, i, :] = np.random.normal(center, workplace_size / 10, (num_steps, 2))
        positions[:, i, :] = np.clip(positions[:, i, :], 0, workplace_size)

    compliance = np.random.uniform(0.6, 1.0, (num_steps, num_team_members))
    wellbeing = np.ones((num_steps, num_team_members)) * 0.8
    safety = np.ones(num_steps) * config['SAFETY_COMPLIANCE_THRESHOLD']
    efficiency = {'uptime': [], 'throughput': [], 'quality': [], 'oee': []}
    productivity_loss = np.zeros(num_steps)
    collaboration_index = {'data': [], 'z_scores': [], 'forecast': None}

    # Simulate dynamics
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

        # Collaboration index
        collab = 0
        for zone in config['WORK_AREAS']:
            zone_workers = zone_assignments == zone
            if np.sum(zone_workers) > 1:
                distances = euclidean_distances(positions[t, zone_workers])
                collab += np.mean(distances < workplace_size / 10) * np.sum(zone_workers) / num_team_members
        collaboration_index['data'].append(collab)

    # DataFrames
    try:
        logger.info("Creating team_positions_df")
        team_positions_df = pd.DataFrame({
            'step': np.repeat(np.arange(num_steps), num_team_members),
            'team_member_id': np.tile(worker_ids, num_steps),
            'x': positions[:, :, 0].flatten(),
            'y': positions[:, :, 1].flatten(),
            'zone': np.tile(zone_assignments, num_steps)
        })
    except NameError as ne:
        logger.error(f"NameError creating team_positions_df: {str(ne)}")
        raise

    try:
        logger.info("Creating efficiency_metrics_df")
        efficiency_metrics_df = pd.DataFrame(efficiency)
    except NameError as ne:
        logger.error(f"NameError creating efficiency_metrics_df: {str(ne)}")
        raise

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
    wellbeing_means = np.mean(wellbeing, axis=1)
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

    # Feedback impact
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
