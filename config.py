# config.py
import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # --- Simulation Core Parameters ---
    "TEAM_SIZE": 30, # Default changed for testing, adjust as needed
    "SHIFT_DURATION_MINUTES": 240,  # 4 hours default for quicker tests
    "SHIFT_DURATION_INTERVALS": 120, # Calculated as DURATION_MINUTES / 2

    # --- Facility & Work Area Configuration ---
    "FACILITY_SIZE": (100, 80),
    "ENTRY_EXIT_POINTS": [
        {"name": "Main Entry/Exit", "coords": (5, 40)},
        {"name": "Goods In", "coords": (95, 10)},
        {"name": "Dispatch", "coords": (95, 70)},
    ],
    "WORK_AREAS": {
        "Assembly Line A": {
            "coords": [(10, 5), (70, 25)], "workers": 8, "tasks_per_interval": 5,
            "max_concurrent_tasks": 20, "task_complexity": 0.6, "base_productivity": 0.85,
            "equipment_dependency": ["Conveyor A", "Robot Arm A1"], "proximity_bonus_factor": 0.05
        },
        "Assembly Line B": {
            "coords": [(10, 30), (70, 50)], "workers": 8, "tasks_per_interval": 5,
            "max_concurrent_tasks": 20, "task_complexity": 0.65, "base_productivity": 0.8,
            "equipment_dependency": ["Conveyor B", "Robot Arm B1"], "proximity_bonus_factor": 0.05
        },
        "Quality Control": {
            "coords": [(75, 20), (95, 40)], "workers": 4, "tasks_per_interval": 2,
            "max_concurrent_tasks": 5, "task_complexity": 0.75, "base_productivity": 0.9,
            "equipment_dependency": ["QC Station Alpha"], "proximity_bonus_factor": 0.02
        },
        "Warehouse": {
            "coords": [(10, 55), (90, 75)], "workers": 7, "tasks_per_interval": 8,
            "max_concurrent_tasks": 15, "task_complexity": 0.4, "base_productivity": 0.9,
            "equipment_dependency": ["Forklift Main", "AGV System"], "proximity_bonus_factor": 0.03
        },
        "Break Room": {
            "coords": [(0, 70), (10, 80)], "workers": 0, "tasks_per_interval": 0,
            "max_concurrent_tasks": 0, "task_complexity": 0, "base_productivity": 0,
            "equipment_dependency": [], "is_休憩area": True
        }
        # Ensure sum of workers here matches default TEAM_SIZE or is handled by redistribution logic
    },

    # --- Event Scheduling ---
    "DEFAULT_SCHEDULED_EVENTS": [
        {"Event Type": "Major Disruption", "Start Time (min)": 60, "Duration (min)": 10, "Intensity": 0.8, "Affected Zones": ["Assembly Line A"]},
        {"Event Type": "Scheduled Break", "Start Time (min)": 120, "Duration (min)": 15, "Scope": "All"},
        {"Event Type": "Minor Disruption", "Start Time (min)": 180, "Duration (min)": 5, "Intensity": 0.3, "Affected Zones": ["Warehouse"]},
    ],
    "EVENT_TYPE_CONFIG": { # Parameters for how simulation.py should interpret each event type
        "Major Disruption": {
            "compliance_reduction_factor": 0.7, # Max portion of compliance lost
            "wellbeing_drop_factor": 0.3,       # Max portion of wellbeing lost (applied to a base drop)
            "downtime_prob_modifier": 0.6,      # Additional probability of downtime
            "downtime_mean_factor": 1.5,        # Multiplier for mean downtime duration if it occurs
            "fatigue_rate_modifier": 1.5,       # Fatigue accumulates 1.5x faster
        },
        "Minor Disruption": {
            "compliance_reduction_factor": 0.3, "wellbeing_drop_factor": 0.1,
            "downtime_prob_modifier": 0.2, "downtime_mean_factor": 0.8,
            "fatigue_rate_modifier": 1.2,
        },
        "Scheduled Break": {
            "fatigue_recovery_factor": 0.6, # e.g., worker_fatigue *= (1 - 0.6)
            "wellbeing_boost_abs": 10.0,    # Absolute points added to wellbeing score
            "productivity_multiplier": 0.0, # No productive work during break
        },
        "Short Pause": {
            "fatigue_recovery_factor": 0.2, "wellbeing_boost_abs": 3.0,
            "productivity_multiplier": 0.1, # Minimal productivity
        },
        "Team Meeting": {
            "cohesion_boost_abs": 2.0,      # Absolute points added to cohesion
            "productivity_multiplier": 0.05,# Low productivity during meeting
            "psych_safety_boost_abs": 0.02 * 100, # Small boost
        },
        "Maintenance": { # This would need more complex equipment modeling in simulation
            "downtime_prob_modifier": 0.8, # High chance of downtime in affected zone/equipment
            "downtime_mean_factor": 2.0,   # Longer downtime if it occurs
            "specific_zone_uptime_multiplier": 0.0 # For affected zones/equipment
        },
        "Custom Event": { # Example for user-defined events; simulation would need generic handling or mapping
            "wellbeing_drop_factor": 0.05,
            "fatigue_rate_modifier": 1.1,
        }
    },

    # --- Worker Behavior & Psychosocial Factors ---
    "INITIAL_WELLBEING_MEAN": 0.8, "WELLBEING_FATIGUE_RATE_PER_INTERVAL": 0.0025,
    "WELLBEING_ALERT_THRESHOLD": 60.0, "WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET": 0.85,
    "FATIGUE_IMPACT_ON_COMPLIANCE": 0.3, "COMPLEXITY_IMPACT_ON_COMPLIANCE": 0.35,
    "STRESS_FROM_LOW_CONTROL_FACTOR": 0.025, "ISOLATION_IMPACT_ON_WELLBEING": 0.15,
    "PSYCH_SAFETY_BASELINE": 0.75, "PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL": 0.0005,
    "UNCERTAINTY_DURING_DISRUPTION_IMPACT_PSYCH_SAFETY": 0.15,
    "TEAM_COHESION_IMPACT_ON_PSYCH_SAFETY": 0.2,
    "TEAM_COHESION_BASELINE": 0.7,
    "PERCEIVED_WORKLOAD_THRESHOLD_HIGH": 7.5, "PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH": 8.5,
    "TARGET_PERCEIVED_WORKLOAD": 6.0,

    # --- Operational Factors ---
    "BASE_TASK_COMPLETION_PROB": 0.97, "MIN_COMPLIANCE_DURING_DISRUPTION": 15.0,
    "EQUIPMENT_FAILURE_PROB_PER_INTERVAL": 0.003,
    "DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB": 0.75,
    "EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS": 4, # Number of 2-min intervals
    "THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL": 120.0,
    "BASE_QUALITY_DEFECT_RATE": 0.015,

    # --- Recovery (General, for after events end) ---
    "RECOVERY_HALFLIFE_INTERVALS": 8,

    # --- Downtime Specifics (Can be overridden by event-specific downtime) ---
    "DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE": 7.0, # General mean for random small downtimes
    "DOWNTIME_STD_MINUTES_PER_OCCURRENCE": 3.0,
    "DOWNTIME_CAUSES_LIST": ["Equipment Failure", "Material Shortage", "Process Bottleneck", "Human Error", "Utility Outage", "External Supply Chain", "Software Glitch", "Meeting Overrun", "Minor Stoppage"],
    "DOWNTIME_PLOT_ALERT_THRESHOLD": 10,

    # --- Leadership & Communication ---
    "LEADERSHIP_SUPPORT_FACTOR": 0.65, "COMMUNICATION_EFFECTIVENESS_FACTOR": 0.75,

    # --- Initiatives ---
    "INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR": 0.4, # General reduction if this initiative is active (complements scheduled breaks)
    "INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST_ABS": 0.03 * 100,
    "INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS": 0.06 * 100,
    "INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS": 0.09 * 100,
    "INITIATIVE_RECOGNITION_COHESION_BOOST_ABS": 7.0,
    "INITIATIVE_AUTONOMY_WELLBEING_BOOST_ABS": 0.05 * 100,
    "INITIATIVE_AUTONOMY_PSYCHSAFETY_BOOST_ABS": 0.08 * 100,
    "INITIATIVE_AUTONOMY_COMPLIANCE_BOOST_FACTOR": 0.06,

    # --- Dashboard Targets ---
    "TARGET_COMPLIANCE": 85.0, "TARGET_COLLABORATION": 65.0, "TARGET_WELLBEING": 75.0,
    "TARGET_PSYCH_SAFETY": 80.0, "TARGET_TEAM_COHESION": 75.0,
    "DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT": 0.05,
}

def validate_config(config_to_validate):
    logger.info("Validating configuration...")
    required_keys = [
        "TEAM_SIZE", "SHIFT_DURATION_MINUTES", "FACILITY_SIZE", "WORK_AREAS",
        "SCHEDULED_EVENTS", "EVENT_TYPE_CONFIG"
    ]
    for key in required_keys:
        if key not in config_to_validate:
            logger.error(f"Configuration validation failed: Missing critical key '{key}'.")
            raise ValueError(f"Configuration validation failed: Missing critical key '{key}'.")

    if not isinstance(config_to_validate["TEAM_SIZE"], int) or config_to_validate["TEAM_SIZE"] < 0:
        logger.error(f"Invalid TEAM_SIZE: {config_to_validate['TEAM_SIZE']}. Must be a non-negative integer.")
        raise ValueError("Invalid TEAM_SIZE.")
    
    if not isinstance(config_to_validate["SHIFT_DURATION_MINUTES"], int) or config_to_validate["SHIFT_DURATION_MINUTES"] <= 0:
        logger.error(f"Invalid SHIFT_DURATION_MINUTES: {config_to_validate['SHIFT_DURATION_MINUTES']}. Must be a positive integer.")
        raise ValueError("Invalid SHIFT_DURATION_MINUTES.")

    if not isinstance(config_to_validate["FACILITY_SIZE"], tuple) or len(config_to_validate["FACILITY_SIZE"]) != 2 or \
       not all(isinstance(dim, (int, float)) and dim > 0 for dim in config_to_validate["FACILITY_SIZE"]):
        logger.error(f"Invalid FACILITY_SIZE: {config_to_validate['FACILITY_SIZE']}. Must be a tuple of two positive numbers (width, height).")
        raise ValueError("Invalid FACILITY_SIZE.")

    if not isinstance(config_to_validate["WORK_AREAS"], dict):
        logger.error("WORK_AREAS must be a dictionary.")
        raise ValueError("Invalid WORK_AREAS structure.")
    
    for area_name, details in config_to_validate["WORK_AREAS"].items():
        if not isinstance(details, dict):
            logger.error(f"Configuration for WORK_AREA '{area_name}' must be a dictionary.")
            raise ValueError(f"Invalid configuration for WORK_AREA '{area_name}'.")
        if "coords" in details:
            if not (isinstance(details["coords"], list) and len(details["coords"]) == 2 and
                    all(isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(c, (int, float)) for c in pt) for pt in details["coords"])):
                logger.warning(f"Potentially invalid 'coords' for WORK_AREA '{area_name}'. Expected list of two (x,y) tuples.")
    
    if not isinstance(config_to_validate.get("SCHEDULED_EVENTS"), list):
        logger.warning("'SCHEDULED_EVENTS' is missing or not a list. Defaulting to empty for simulation.")
        config_to_validate["SCHEDULED_EVENTS"] = []
    else:
        for event_idx, event in enumerate(config_to_validate["SCHEDULED_EVENTS"]):
            if not isinstance(event, dict):
                logger.error(f"Event at index {event_idx} in SCHEDULED_EVENTS is not a dictionary.")
                raise ValueError(f"Invalid event structure in SCHEDULED_EVENTS.")
            if not all(k in event for k in ["Event Type", "Start Time (min)", "Duration (min)"]):
                logger.error(f"Event at index {event_idx} ({event.get('Event Type', 'Unknown')}) is missing required keys (Event Type, Start Time (min), Duration (min)).")
                raise ValueError("Event in SCHEDULED_EVENTS missing required keys.")


    if not isinstance(config_to_validate.get("EVENT_TYPE_CONFIG"), dict):
        logger.warning("'EVENT_TYPE_CONFIG' is missing or not a dictionary. Event impacts may not be applied correctly.")
        config_to_validate["EVENT_TYPE_CONFIG"] = {}

    logger.info("Configuration validation passed (or warnings issued).")
    return True

if __name__ == "__main__":
    try:
        validate_config(DEFAULT_CONFIG)
        print("Default configuration is valid.")
    except ValueError as e:
        print(f"Configuration error: {e}")
