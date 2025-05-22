# config.py
import numpy as np
import logging 

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'TEAM_SIZE': 30, 
    'SHIFT_DURATION_MINUTES': 480,  # 8 hours 
    # SHIFT_DURATION_INTERVALS will be derived (SHIFT_DURATION_MINUTES // 2)
    
    'DISRUPTION_TIMES_MINUTES': [120, 300],  # Times (in minutes from start) when major disruptions START
                                            # Simulation will convert these to step numbers.
    
    # --- TARGETS for Actionable Insights & Plotting (examples, adjust as needed) ---
    'TARGET_COMPLIANCE': 85.0,                  # Target for average task compliance (%)
    'TARGET_COLLABORATION': 70.0,               # Target for average collaboration index (%)
    'TARGET_WELLBEING': 75.0,                   # Target for average worker well-being (%)
    'DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENTAGE': 0.05, # Overall acceptable downtime as % of shift
    'WELLBEING_ALERT_THRESHOLD': 0.60,          # Wellbeing score (0-1 scale) below which is a direct alert
    'DOWNTIME_PLOT_ALERT_THRESHOLD': 10,        # Per-interval downtime (minutes) for plot highlighting

    # --- Facility & Work Area Setup ---
    'FACILITY_SIZE': (100, 80), # meters
    'WORK_AREAS': {
        'Assembly Line A': {'coords': [(10, 10), (70, 20)], 'workers': 10, 'tasks_per_interval': 5, 'task_complexity': 0.6, 'base_productivity': 0.9},
        'Quality Control': {'coords': [(75, 10), (90, 20)], 'workers': 3, 'tasks_per_interval': 2, 'task_complexity': 0.8, 'base_productivity': 0.95},
        'Packing Station': {'coords': [(10, 50), (40, 70)], 'workers': 7, 'tasks_per_interval': 4, 'task_complexity': 0.5, 'base_productivity': 0.85},
        'Warehouse': {'coords': [(45, 50), (90, 70)], 'workers': 7, 'tasks_per_interval': 3, 'task_complexity': 0.4, 'base_productivity': 0.8},
        'Break Room': {'coords': [(1, 1), (10, 8)], 'workers': 0, 'tasks_per_interval': 0, 'task_complexity': 0, 'base_productivity': 1.0},
        'Office': {'coords': [(1, 25), (10, 45)], 'workers': 3, 'tasks_per_interval': 1, 'task_complexity': 0.3, 'base_productivity': 0.9},
    },
    'ENTRY_EXIT_POINTS': [{'name': 'Main Entrance', 'coords': (0, 40), 'type': 'entry_exit'}, {'name': 'Loading Dock', 'coords': (100, 60), 'type': 'exit_only'}],
    
    # --- Simulation Dynamics Parameters ---
    'WORKER_SPEED_MEAN': 1.2, 'WORKER_SPEED_STD': 0.2, # m/s
    'COLLABORATION_RADIUS': 5, # meters
    'COMMUNICATION_SUCCESS_RATE': 0.85, # probability (0-1)
    
    # Task & Performance
    'BASE_TASK_COMPLETION_PROB': 0.95, 
    'FATIGUE_IMPACT_ON_COMPLIANCE': 0.15, # Max % reduction in compliance due to full fatigue
    'COMPLEXITY_IMPACT_ON_COMPLIANCE': 0.2, # Max % reduction due to highest complexity
    'MIN_COMPLIANCE_DURING_DISRUPTION': 30.0, 

    # Disruptions
    'DISRUPTION_DURATION_MEAN_INTERVALS': 5, 
    'DISRUPTION_DURATION_STD_INTERVALS': 2,
    'DISRUPTION_COMPLIANCE_REDUCTION_FACTOR': 0.5, 
    'DISRUPTION_WELLBEING_DROP': 0.2, # Absolute drop in well-being score (0-1 scale) from baseline
    'RECOVERY_HALFLIFE_INTERVALS': 10, # Intervals to recover half of loss after disruption ends

    # Well-being & Safety
    'WELLBEING_BASELINE': 0.80, # Target equilibrium for well-being (0-1 scale)
    'WELLBEING_FATIGUE_RATE_PER_INTERVAL': 0.002, # Gradual decline per interval
    'WELLBEING_RECOVERY_AT_BREAK_ABS': 0.15, # Absolute boost during simulated breaks
    'PSYCH_SAFETY_BASELINE': 0.75, # (0-1 scale)
    'PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL': 0.0005, # Slow decline if not maintained
    'PSYCH_SAFETY_BOOST_FROM_RECOGNITION_ABS': 0.1,
    'FEEDBACK_POSITIVE_IMPACT_ON_PSYCH_SAFETY': 0.05, 
    
    # OEE Components
    'THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL': 100, # Facility total theoretical max output
    'BASE_QUALITY_DEFECT_RATE': 0.02, # 2% defects baseline
    'EQUIPMENT_FAILURE_PROB_PER_INTERVAL': 0.005, 
    'EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS': 3, 

    # Downtime (distinct from DISRUPTION_TIMES_MINUTES which are event triggers)
    'DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB': 0.7, # Chance equipment failure leads to reported downtime
    'DOWNTIME_FROM_DISRUPTION_EVENT_PROB': 0.5, # Chance a major disruption event leads to reported downtime
    'DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE': 8, 
    'DOWNTIME_STD_MINUTES_PER_OCCURRENCE': 4,
    
    # Team Initiatives (Effects used in simulation.py)
    'INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR': 0.3, # Reduces fatigue accumulation rate by 30%
    'INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST_ABS': 0.05, # Additional well-being recovery during breaks
    'INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS': 0.05, 
    'INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS': 0.08,
}

# Dynamically set derived values
DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'] = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] // 2
DEFAULT_CONFIG['DOWNTIME_THRESHOLD_TOTAL_SHIFT'] = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] * DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENTAGE', 0.05)


def validate_config(config):
    if not isinstance(config, dict): raise ValueError("Configuration must be a dictionary.")
    required_keys = ['TEAM_SIZE', 'SHIFT_DURATION_MINUTES', 'SHIFT_DURATION_INTERVALS', 'FACILITY_SIZE', 'WORK_AREAS', 'DISRUPTION_TIMES_MINUTES']
    for key in required_keys:
        if key not in config: raise ValueError(f"Missing required configuration key: {key}")
    if not (isinstance(config['FACILITY_SIZE'], tuple) and len(config['FACILITY_SIZE']) == 2 and all(isinstance(x, (int, float)) for x in config['FACILITY_SIZE'])): raise ValueError("FACILITY_SIZE must be a tuple of two numbers (width, height).")
    if not isinstance(config['WORK_AREAS'], dict) or not config['WORK_AREAS']: raise ValueError("WORK_AREAS must be a non-empty dictionary.")
    for area_name, area_details in config['WORK_AREAS'].items():
        if not isinstance(area_details, dict) or 'coords' not in area_details or 'workers' not in area_details: raise ValueError(f"Work area '{area_name}' is missing required keys 'coords' or 'workers' or is not a dictionary.")
    total_workers_in_zones = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values() if isinstance(zone, dict))
    if total_workers_in_zones != config['TEAM_SIZE'] and config['TEAM_SIZE'] > 0 and total_workers_in_zones == 0: 
        logger.info(f"Config Validation: Initial workers in zones is 0. Simulation logic will distribute TEAM_SIZE={config['TEAM_SIZE']}.")
    elif total_workers_in_zones != config['TEAM_SIZE']:
         logger.warning(f"Config Validation: Sum of workers in WORK_AREAS ({total_workers_in_zones}) does not precisely match TEAM_SIZE ({config['TEAM_SIZE']}). Simulation logic will reconcile this.")
    if not isinstance(config['DISRUPTION_TIMES_MINUTES'], list):
        logger.warning(f"Config Validation: DISRUPTION_TIMES_MINUTES is not a list (type: {type(config['DISRUPTION_TIMES_MINUTES'])}). Ensure UI passes a list.")
    logger.info("Configuration structure partially validated.", extra={'user_action': 'System Check'})
    return True
