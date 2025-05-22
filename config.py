# config.py
import numpy as np
import logging 

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'TEAM_SIZE': 30, 
    'SHIFT_DURATION_MINUTES': 240,  # 4 hours for quicker testing
    # SHIFT_DURATION_INTERVALS will be derived
    'DISRUPTION_INTERVALS': [30, 90],  # Example disruption steps (time_in_minutes / 2)
    
    # TARGETS for Actionable Insights (examples, adjust as needed)
    'TARGET_COMPLIANCE': 80.0,
    'TARGET_COLLABORATION': 65.0,
    'TARGET_WELLBEING': 75.0,
    'DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENTAGE': 0.05, # As a percentage of total shift
    'WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET': 0.85, # e.g. if wellbeing is < 85% of target, it's critical

    'FACILITY_SIZE': (100, 80),
    'WORK_AREAS': {
        'Assembly Line A': {'coords': [(10, 10), (70, 20)], 'workers': 10, 'tasks_per_interval': 5, 'task_complexity': 0.6},
        'Quality Control': {'coords': [(75, 10), (90, 20)], 'workers': 3, 'tasks_per_interval': 2, 'task_complexity': 0.8},
        'Packing Station': {'coords': [(10, 50), (40, 70)], 'workers': 7, 'tasks_per_interval': 4, 'task_complexity': 0.5},
        'Warehouse': {'coords': [(45, 50), (90, 70)], 'workers': 7, 'tasks_per_interval': 3, 'task_complexity': 0.4},
        'Break Room': {'coords': [(1, 1), (10, 8)], 'workers': 0, 'tasks_per_interval': 0, 'task_complexity': 0},
        'Office': {'coords': [(1, 25), (10, 45)], 'workers': 3, 'tasks_per_interval': 1, 'task_complexity': 0.3},
    },
    'ENTRY_EXIT_POINTS': [{'name': 'Main Entrance', 'coords': (0, 40), 'type': 'entry_exit'}, {'name': 'Loading Dock', 'coords': (100, 60), 'type': 'exit_only'}],
    'WORKER_SPEED_MEAN': 1.2, 'WORKER_SPEED_STD': 0.2,
    'COLLABORATION_RADIUS': 5, 'COMMUNICATION_SUCCESS_RATE': 0.85,
    'BASE_TASK_COMPLETION_PROB': 0.95,
    'DISRUPTION_IMPACT_FACTOR': 0.3, 'RECOVERY_RATE_PER_INTERVAL': 0.1,
    'WELLBEING_THRESHOLD': 0.6, 
    'INITIAL_WELLBEING_MEAN': 0.8, 'INITIAL_WELLBEING_STD': 0.1,
    'FATIGUE_INCREASE_RATE': 0.005, 'STRESS_FROM_DISRUPTION': 0.15,
    'PSYCH_SAFETY_INITIAL_MEAN': 0.75, 'PSYCH_SAFETY_INITIAL_STD': 0.1,
    'FEEDBACK_POSITIVE_IMPACT': 0.05,
    'THEORETICAL_MAX_THROUGHPUT_PER_WORKER': 10, 'BASE_QUALITY_RATE': 0.98,
    'EQUIPMENT_UPTIME_MEAN': 0.95,
    'DOWNTIME_PROB_PER_DISRUPTION': 0.4,
    'DOWNTIME_DURATION_MEAN_MINUTES': 5, 'DOWNTIME_DURATION_STD_MINUTES': 2,
    'DOWNTIME_THRESHOLD': 10, # Per interval downtime threshold for plotting
    'INITIATIVE_BREAKS_FATIGUE_REDUCTION': 0.1,
    'INITIATIVE_RECOGNITION_WELLBEING_BOOST': 0.08,
    'INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST': 0.06,
}
DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'] = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] // 2
# Calculate DOWNTIME_THRESHOLD_TOTAL_SHIFT based on percentage and duration
DEFAULT_CONFIG['DOWNTIME_THRESHOLD_TOTAL_SHIFT'] = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] * DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENTAGE', 0.05)


def validate_config(config):
    # ... (validation logic remains the same)
    if not isinstance(config, dict): raise ValueError("Configuration must be a dictionary.")
    required_keys = ['TEAM_SIZE', 'SHIFT_DURATION_MINUTES', 'SHIFT_DURATION_INTERVALS', 'FACILITY_SIZE', 'WORK_AREAS']
    for key in required_keys:
        if key not in config: raise ValueError(f"Missing required configuration key: {key}")
    if not (isinstance(config['FACILITY_SIZE'], tuple) and len(config['FACILITY_SIZE']) == 2 and all(isinstance(x, (int, float)) for x in config['FACILITY_SIZE'])): raise ValueError("FACILITY_SIZE must be a tuple of two numbers (width, height).")
    if not isinstance(config['WORK_AREAS'], dict) or not config['WORK_AREAS']: raise ValueError("WORK_AREAS must be a non-empty dictionary.")
    for area_name, area_details in config['WORK_AREAS'].items():
        if not isinstance(area_details, dict) or 'coords' not in area_details or 'workers' not in area_details: raise ValueError(f"Work area '{area_name}' is missing required keys 'coords' or 'workers' or is not a dictionary.")
    total_workers_in_zones = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values() if isinstance(zone, dict))
    if total_workers_in_zones != config['TEAM_SIZE']: logger.warning(f"Config Validation: Sum of workers in WORK_AREAS ({total_workers_in_zones}) does not precisely match TEAM_SIZE ({config['TEAM_SIZE']}). Simulation logic will attempt to reconcile this.")
    logger.info("Configuration structure validated.", extra={'user_action': 'System Check'})
    return True
