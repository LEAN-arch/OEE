"""
config.py
Configuration management for the Industrial Workplace Shift Monitoring Dashboard.
Provides default parameters and validation for simulation settings.
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
        raise ValueError("Facility size must be positive")
    if not (0 <= config['WELLBEING_THRESHOLD'] <= 1):
        raise ValueError("Well-being threshold must be between 0 and 1")
    if not (0 <= config['SAFETY_COMPLIANCE_THRESHOLD'] <= 1):
        raise ValueError("Safety threshold must be between 0 and 1")
    if config['TEAM_SIZE'] <= 0:
        raise ValueError("Team size must be positive")
    if config['SHIFT_DURATION_INTERVALS'] <= 0:
        raise ValueError("Shift duration must be positive")
    if any(t < 0 or t > config['SHIFT_DURATION_INTERVALS'] for t in config['DISRUPTION_INTERVALS']):
        raise ValueError("Disruption intervals out of range")
    if not (0 <= config['ADAPTATION_RATE'] <= 1):
        raise ValueError("Adaptation rate must be between 0 and 1")
    if not (0 <= config['SUPERVISOR_INFLUENCE'] <= 1):
        raise ValueError("Supervisor influence must be between 0 and 1")
    if sum(zone['workers'] for zone in config['WORK_AREAS'].values()) != config['TEAM_SIZE']:
        raise ValueError("Sum of workers in WORK_AREAS must equal TEAM_SIZE")

DEFAULT_CONFIG = {
    'WORK_AREAS': {
        'Assembly Line': {'center': [20, 20], 'label': 'Assembly Line', 'workers': 20},
        'Packaging Zone': {'center': [60, 60], 'label': 'Packaging Zone', 'workers': 15},
        'Quality Control': {'center': [80, 80], 'label': 'Quality Control', 'workers': 15}
    },
    'WELLBEING_THRESHOLD': 0.7,
    'WELLBEING_TREND_WINDOW': 3,
    'DISRUPTION_RECOVERY_WINDOW': 10,
    'BREAK_FREQUENCY_INTERVALS': 30,
    'WORKLOAD_CAP_INTERVALS': 10,
    'TEAM_SIZE': 50,
    'SHIFT_DURATION_INTERVALS': 480,
    'FACILITY_SIZE': 100,
    'ADAPTATION_RATE': 0.05,
    'SUPERVISOR_INFLUENCE': 0.2,
    'DISRUPTION_INTERVALS': [60, 180],
    'ANOMALY_THRESHOLD': 2.0,
    'SAFETY_COMPLIANCE_THRESHOLD': 0.7,
    'DENSITY_GRID_SIZE': 20,
    'FACILITY_TYPE': 'manufacturing'
}

# Validate default configuration
validate_config(DEFAULT_CONFIG)
