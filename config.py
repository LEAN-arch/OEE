"""
config.py
Configuration management for the Industrial Workplace Shift Monitoring Dashboard.
Provides default parameters with SI units and validation for simulation settings.
"""

def validate_config(config: dict) -> None:
    """
    Validate configuration parameters.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if config['FACILITY_SIZE'] <= 0:
        raise ValueError("Facility size (meters) must be positive")
    if not (0 <= config['WELLBEING_THRESHOLD'] <= 1):
        raise ValueError("Well-being index (0-1) must be between 0 and 1")
    if not (0 <= config['SAFETY_THRESHOLD'] <= 1):
        raise ValueError("Psychological safety score (0-1) must be between 0 and 1")
    if config['TEAM_SIZE'] <= 0:
        raise ValueError("Team size must be positive")
    if config['SHIFT_DURATION_INTERVALS'] <= 0:
        raise ValueError("Shift duration (intervals) must be positive")
    if config['SHIFT_DURATION_MINUTES'] != config['SHIFT_DURATION_INTERVALS'] * 2:
        raise ValueError("Shift duration (minutes) must equal intervals * 2")
    if any(t < 0 or t > config['SHIFT_DURATION_INTERVALS'] for t in config['DISRUPTION_INTERVALS']):
        raise ValueError("Disruption intervals out of range")
    if not (0 <= config['ADAPTATION_RATE'] <= 1):
        raise ValueError("Adaptation rate must be between 0 and 1")
    if not (0 <= config['SUPERVISOR_INFLUENCE'] <= 1):
        raise ValueError("Supervisor influence must be between 0 and 1")
    if sum(zone['workers'] for zone in config['WORK_AREAS'].values()) != config['TEAM_SIZE']:
        raise ValueError("Sum of workers in WORK_AREAS must equal TEAM_SIZE")
    if config['DOWNTIME_THRESHOLD'] < 0:
        raise ValueError("Downtime threshold (minutes) must be non-negative")
    if not (0 <= config['TASK_COMPLETION_THRESHOLD'] <= 1):
        raise ValueError("Task completion threshold (0-1) must be between 0 and 1")

DEFAULT_CONFIG = {
    'WORK_AREAS': {
        'Assembly Line': {'center': [20, 20], 'label': 'Assembly Line', 'workers': 20},
        'Packaging Zone': {'center': [60, 60], 'label': 'Packaging Zone', 'workers': 15},
        'Quality Control': {'center': [80, 80], 'label': 'Quality Control', 'workers': 15}
    },
    'WELLBEING_THRESHOLD': 0.7,  # Worker Well-Being Index (0-1) below which alerts are triggered
    'WELLBEING_TREND_WINDOW': 3,  # Intervals (6 minutes) for detecting well-being trends
    'DISRUPTION_RECOVERY_WINDOW': 10,  # Intervals (20 minutes) for recovery post-disruption
    'BREAK_FREQUENCY_INTERVALS': 30,  # Intervals (60 minutes) between breaks
    'WORKLOAD_CAP_INTERVALS': 10,  # Intervals (20 minutes) for workload limits
    'TEAM_SIZE': 50,  # Total number of workers
    'SHIFT_DURATION_INTERVALS': 480,  # Number of 2-minute intervals (960 minutes = 16 hours)
    'SHIFT_DURATION_MINUTES': 960,  # Total shift duration in minutes
    'FACILITY_SIZE': 100,  # Facility size in meters (100m x 100m)
    'ADAPTATION_RATE': 0.05,  # Rate of compliance improvement per interval
    'SUPERVISOR_INFLUENCE': 0.2,  # Supervisor impact on compliance (0-1)
    'DISRUPTION_INTERVALS': [60, 180],  # Intervals (120, 360 minutes) for disruptions
    'ANOMALY_THRESHOLD': 2.0,  # Z-score threshold for anomaly detection
    'SAFETY_THRESHOLD': 0.7,  # Psychological Safety Score (0-1) below which alerts are triggered
    'DENSITY_GRID_SIZE': 20,  # Grid size for density heatmap
    'FACILITY_TYPE': 'manufacturing',  # Facility type for context
    'DOWNTIME_THRESHOLD': 10,  # Minutes of downtime triggering alerts
    'TASK_COMPLETION_THRESHOLD': 0.9,  # Task completion rate (0-1) below which alerts are triggered
    'COMPANY_LOGO_PATH': 'logo.png'  # Path to company logo for sidebar
}

# Validate default configuration
validate_config(DEFAULT_CONFIG)
