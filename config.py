"""
config.py
Configuration management for the Workplace Shift Monitoring Dashboard.
Provides default parameters with SI units and validation for simulation settings.
"""

import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log'
)

def validate_config(config: dict) -> None:
    """
    Validate configuration parameters.

    Args:
        config (dict): Configuration dictionary.

    Raises:
        ValueError: If any parameter is invalid.
    """
    try:
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
        if any(t < 0 or t >= config['SHIFT_DURATION_INTERVALS'] for t in config['DISRUPTION_INTERVALS']):
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
        if config['WELLBEING_TREND_WINDOW'] <= 0:
            raise ValueError("Well-being trend window must be positive")
        if config['DISRUPTION_RECOVERY_WINDOW'] <= 0:
            raise ValueError("Disruption recovery window must be positive")
        if config['BREAK_FREQUENCY_INTERVALS'] <= 0:
            raise ValueError("Break frequency intervals must be positive")
        if config['WORKLOAD_CAP_INTERVALS'] <= 0:
            raise ValueError("Workload cap intervals must be positive")
        if config['DENSITY_GRID_SIZE'] <= 0:
            raise ValueError("Density grid size must be positive")
        if config['ANOMALY_THRESHOLD'] <= 0:
            raise ValueError("Anomaly threshold (z-score) must be positive")
        # Validate entry/exit points
        entry_exit_labels = [point['label'] for point in config['ENTRY_EXIT_POINTS']]
        if len(entry_exit_labels) != len(set(entry_exit_labels)):
            raise ValueError("Entry/Exit point labels must be unique")
        for point in config['ENTRY_EXIT_POINTS']:
            x, y = point['coords']
            if not (0 <= x <= config['FACILITY_SIZE'] and 0 <= y <= config['FACILITY_SIZE']):
                raise ValueError(f"Entry/Exit point {point['label']} coords ({x}, {y}) out of facility bounds")
        # Validate production lines
        production_line_labels = [line['label'] for line in config['PRODUCTION_LINES']]
        if len(production_line_labels) != len(set(production_line_labels)):
            raise ValueError("Production line labels must be unique")
        for line in config['PRODUCTION_LINES']:
            start_x, start_y = line['start']
            end_x, end_y = line['end']
            if not (0 <= start_x <= config['FACILITY_SIZE'] and 0 <= start_y <= config['FACILITY_SIZE']):
                raise ValueError(f"Production line {line['label']} start ({start_x}, {start_y}) out of bounds")
            if not (0 <= end_x <= config['FACILITY_SIZE'] and 0 <= end_y <= config['FACILITY_SIZE']):
                raise ValueError(f"Production line {line['label']} end ({end_x}, {end_y}) out of bounds")
        logger.info("Configuration validated successfully", extra={'user_action': 'Validate Config'})
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}", extra={'user_action': 'Validate Config'})
        raise

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
    'SHIFT_DURATION_MINUTES': 960,
    'FACILITY_SIZE': 100,
    'ADAPTATION_RATE': 0.05,
    'SUPERVISOR_INFLUENCE': 0.2,
    'DISRUPTION_INTERVALS': [60, 180],
    'ANOMALY_THRESHOLD': 2.0,
    'SAFETY_THRESHOLD': 0.7,
    'DENSITY_GRID_SIZE': 20,
    'FACILITY_TYPE': 'Workplace',
    'DOWNTIME_THRESHOLD': 10,
    'TASK_COMPLETION_THRESHOLD': 0.9,
    'COMPANY_LOGO_PATH': 'logo.png',  # Optional, handled in main.py
    'ENTRY_EXIT_POINTS': [
        {'label': 'Entry 1', 'coords': [0, 50], 'type': 'Entry'},
        {'label': 'Exit 1', 'coords': [100, 50], 'type': 'Exit'},
        {'label': 'Entry/Exit 2', 'coords': [50, 0], 'type': 'Entry/Exit'}
    ],
    'PRODUCTION_LINES': [
        {'label': 'Line 1', 'start': [10, 10], 'end': [40, 40]},
        {'label': 'Line 2', 'start': [50, 50], 'end': [70, 70]}
    ]
}

# Validate default configuration
validate_config(DEFAULT_CONFIG)
