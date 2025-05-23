# config.py
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # --- Simulation Core Parameters ---
    "MINUTES_PER_INTERVAL": 2,
    "TEAM_SIZE": 30,
    "SHIFT_DURATION_MINUTES": 240,  # Default 4 hours
    # SHIFT_DURATION_INTERVALS is calculated in run_simulation_logic

    # --- Facility & Work Area Configuration ---
    "FACILITY_SIZE": (100, 80), # width, height in meters
    "ENTRY_EXIT_POINTS": [
        {"name": "Main Entry/Exit", "coords": (5, 40)},
        {"name": "Goods In", "coords": (95, 10)},
        {"name": "Dispatch", "coords": (95, 70)},
    ],
    "WORK_AREAS": {
        "Assembly Line A": {
            "coords": [(10, 5), (70, 25)], "workers": 8, "tasks_per_interval": 5,
            "max_concurrent_tasks": 20, "task_complexity": 0.6, "base_productivity": 0.85, # 0-1
            "equipment_dependency": ["Conveyor A", "Robot Arm A1"], "proximity_bonus_factor": 0.05, # 0-1
            "is_rest_area": False
        },
        "Assembly Line B": {
            "coords": [(10, 30), (70, 50)], "workers": 8, "tasks_per_interval": 5,
            "max_concurrent_tasks": 20, "task_complexity": 0.65, "base_productivity": 0.8,
            "equipment_dependency": ["Conveyor B", "Robot Arm B1"], "proximity_bonus_factor": 0.05,
            "is_rest_area": False
        },
        "Quality Control": {
            "coords": [(75, 20), (95, 40)], "workers": 4, "tasks_per_interval": 2,
            "max_concurrent_tasks": 5, "task_complexity": 0.75, "base_productivity": 0.9,
            "equipment_dependency": ["QC Station Alpha"], "proximity_bonus_factor": 0.02,
            "is_rest_area": False
        },
        "Warehouse": {
            "coords": [(10, 55), (90, 75)], "workers": 7, "tasks_per_interval": 8,
            "max_concurrent_tasks": 15, "task_complexity": 0.4, "base_productivity": 0.9,
            "equipment_dependency": ["Forklift Main", "AGV System"], "proximity_bonus_factor": 0.03,
            "is_rest_area": False
        },
        "Break Room": {
            "coords": [(0, 70), (10, 80)], "workers": 3, # Sum of workers = 30
            "tasks_per_interval": 0, "max_concurrent_tasks": 0, "task_complexity": 0,
            "base_productivity": 0, "equipment_dependency": [],
            "is_rest_area": True
        }
    },

    # --- Event Scheduling ---
    "DEFAULT_SCHEDULED_EVENTS": [
        {"Event Type": "Major Disruption", "Start Time (min)": 60, "Duration (min)": 10, "Intensity": 0.8, "Affected Zones": ["Assembly Line A"]},
        {"Event Type": "Scheduled Break", "Start Time (min)": 120, "Duration (min)": 15, "Scope": "All"},
        {"Event Type": "Minor Disruption", "Start Time (min)": 180, "Duration (min)": 5, "Intensity": 0.3, "Affected Zones": ["Warehouse"]},
    ],
    "EVENT_TYPE_CONFIG": { # Factors are 0-1, Abs values are points on 0-100 scale unless specified
        "Major Disruption": {"compliance_reduction_factor": 0.7, "wellbeing_drop_factor": 0.3, "downtime_prob_modifier": 0.6, "downtime_mean_factor": 1.5, "fatigue_rate_modifier": 1.5},
        "Minor Disruption": {"compliance_reduction_factor": 0.3, "wellbeing_drop_factor": 0.1, "downtime_prob_modifier": 0.2, "downtime_mean_factor": 0.8, "fatigue_rate_modifier": 1.2},
        "Scheduled Break": {"fatigue_recovery_factor": 0.6, "wellbeing_boost_abs": 10.0, "productivity_multiplier": 0.0},
        "Short Pause": {"fatigue_recovery_factor": 0.2, "wellbeing_boost_abs": 3.0, "productivity_multiplier": 0.1},
        "Team Meeting": {"cohesion_boost_abs": 2.0, "productivity_multiplier": 0.05, "psych_safety_boost_abs": 2.0},
        "Maintenance": {"downtime_prob_modifier": 0.8, "downtime_mean_factor": 2.0, "specific_zone_uptime_multiplier": 0.0},
        "Custom Event": {"wellbeing_drop_factor": 0.05, "fatigue_rate_modifier": 1.1} # Factors modify existing rates/values
    },

    # --- Worker Behavior & Psychosocial Factors ---
    "INITIAL_WELLBEING_MEAN": 80.0,
    "BASE_FATIGUE_RATE_PER_INTERVAL": 0.0025, # Additive to 0-1 fatigue scale
    "WELLBEING_ALERT_THRESHOLD": 60.0,
    "WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET": 0.85, # Factor
    "FATIGUE_IMPACT_ON_COMPLIANCE": 0.3, # Factor: compliance_reduction = fatigue_level * this_factor
    "COMPLEXITY_IMPACT_ON_COMPLIANCE": 0.35, # Factor: compliance_reduction = complexity_level * this_factor
    "STRESS_FROM_LOW_CONTROL_POINTS_DROP": 2.5, # Absolute points drop from wellbeing (0-100)
    "ISOLATION_IMPACT_ON_WELLBEING_POINTS_MAX_DROP": 15.0, # Max points drop from wellbeing (0-100)
    "PSYCH_SAFETY_BASELINE": 75.0,
    "BASE_PSYCH_SAFETY_EROSION_PER_INTERVAL": 0.05, # Absolute points drop from psych safety (0-100)
    "UNCERTAINTY_DISRUPTION_PSYCH_SAFETY_POINTS_DROP": 15.0,
    "TEAM_COHESION_IMPACT_ON_PSYCH_SAFETY_FACTOR": 0.2, # Factor: ps_boost = (cohesion - baseline_cohesion) * this_factor
    "TEAM_COHESION_BASELINE": 70.0,
    "PERCEIVED_WORKLOAD_THRESHOLD_HIGH": 7.5, # Scale 0-10
    "PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH": 8.5, # Scale 0-10
    "TARGET_PERCEIVED_WORKLOAD": 6.0, # Scale 0-10

    # --- Operational Factors ---
    "BASE_TASK_COMPLETION_PROB": 0.97, # Probability 0-1
    "MIN_COMPLIANCE_DURING_DISRUPTION": 15.0, # Floor for compliance score 0-100
    "EQUIPMENT_FAILURE_PROB_PER_INTERVAL": 0.003, # Probability 0-1
    "DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB": 0.75, # Probability 0-1
    "EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS": 4, # Number of intervals
    "THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL": 120.0,
    "BASE_QUALITY_DEFECT_RATE": 0.015, # Probability 0-1 (rate of defects, so 1-this is quality yield)

    # --- Recovery ---
    "RECOVERY_HALFLIFE_INTERVALS": 8,

    # --- Downtime Specifics ---
    "DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE": 7.0,
    "DOWNTIME_STD_MINUTES_PER_OCCURRENCE": 3.0,
    "DOWNTIME_CAUSES_LIST": ["Equipment Failure", "Material Shortage", "Process Bottleneck", "Human Error", "Utility Outage", "External Supply Chain", "Software Glitch", "Meeting Overrun", "Minor Stoppage"],
    "DOWNTIME_PLOT_ALERT_THRESHOLD": 10, # In minutes (for y-axis line on downtime trend plot)

    # --- Leadership & Communication (Factors 0-1, where 0.5 is neutral baseline impact) ---
    "LEADERSHIP_SUPPORT_FACTOR": 0.65,
    "COMMUNICATION_EFFECTIVENESS_FACTOR": 0.75,

    # --- Initiatives ---
    "INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR": 0.4, # Reduces fatigue accumulation rate by this factor
    "INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST_ABS": 3.0, # Absolute points boost to wellbeing
    "INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS": 6.0,
    "INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS": 9.0,
    "INITIATIVE_RECOGNITION_COHESION_BOOST_ABS": 7.0,
    "INITIATIVE_AUTONOMY_WELLBEING_BOOST_ABS": 5.0,
    "INITIATIVE_AUTONOMY_PSYCHSAFETY_BOOST_ABS": 8.0,
    "INITIATIVE_AUTONOMY_COMPLIANCE_BOOST_FACTOR": 0.06, # Multiplicative boost (e.g., compliance *= (1 + 0.06))

    # --- Dashboard Targets ---
    "TARGET_COMPLIANCE": 85.0,
    "TARGET_COLLABORATION": 65.0, # For the 'collaboration_metric'
    "TARGET_WELLBEING": 75.0,
    "TARGET_PSYCH_SAFETY": 80.0,
    "TARGET_TEAM_COHESION": 75.0,
    "DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT": 0.05, # Factor 0-1 (5%)
}

def validate_config(config_to_validate):
    logger.info("Validating configuration...")
    required_keys = [
        "TEAM_SIZE", "SHIFT_DURATION_MINUTES", "MINUTES_PER_INTERVAL",
        "FACILITY_SIZE", "WORK_AREAS", "DEFAULT_SCHEDULED_EVENTS", "EVENT_TYPE_CONFIG"
    ]
    for key in required_keys:
        if key not in config_to_validate:
            logger.error(f"Config validation failed: Missing critical key '{key}'.")
            raise ValueError(f"Configuration validation failed: Missing critical key '{key}'.")

    if not isinstance(config_to_validate["TEAM_SIZE"], int) or config_to_validate["TEAM_SIZE"] < 0:
        raise ValueError(f"Invalid TEAM_SIZE: {config_to_validate['TEAM_SIZE']}. Must be non-negative int.")
    if not isinstance(config_to_validate["SHIFT_DURATION_MINUTES"], int) or config_to_validate["SHIFT_DURATION_MINUTES"] <= 0:
        raise ValueError(f"Invalid SHIFT_DURATION_MINUTES: {config_to_validate['SHIFT_DURATION_MINUTES']}. Must be positive int.")
    if not isinstance(config_to_validate["MINUTES_PER_INTERVAL"], (int, float)) or config_to_validate["MINUTES_PER_INTERVAL"] <= 0:
        raise ValueError(f"Invalid MINUTES_PER_INTERVAL: {config_to_validate['MINUTES_PER_INTERVAL']}. Must be positive.")
    if not (isinstance(config_to_validate["FACILITY_SIZE"], tuple) and len(config_to_validate["FACILITY_SIZE"]) == 2 and
            all(isinstance(dim, (int, float)) and dim > 0 for dim in config_to_validate["FACILITY_SIZE"])):
        raise ValueError(f"Invalid FACILITY_SIZE: {config_to_validate['FACILITY_SIZE']}. Tuple of 2 positive numbers.")

    if not isinstance(config_to_validate["WORK_AREAS"], dict):
        raise ValueError("WORK_AREAS must be a dictionary.")
    
    total_workers_in_areas_val = 0
    for area_name_val, details_val in config_to_validate["WORK_AREAS"].items():
        if not isinstance(details_val, dict):
            raise ValueError(f"WORK_AREA '{area_name_val}' config must be a dict.")
        if "coords" in details_val and not (isinstance(details_val["coords"], list) and len(details_val["coords"]) == 2 and
                all(isinstance(pt_val, tuple) and len(pt_val) == 2 and all(isinstance(c_val, (int, float)) for c_val in pt_val) for pt_val in details_val["coords"])):
            logger.warning(f"Potentially invalid 'coords' for WORK_AREA '{area_name_val}'. Expected list of two (x,y) tuples.")
        total_workers_in_areas_val += details_val.get("workers", 0)
    
    # This is an informative warning; main.py's run_simulation_logic handles redistribution
    if total_workers_in_areas_val != config_to_validate["TEAM_SIZE"]:
        logger.warning(f"Sum of workers in WORK_AREAS ({total_workers_in_areas_val}) != TEAM_SIZE ({config_to_validate['TEAM_SIZE']}). main.py should redistribute if this is the final config for simulation.")

    # Validate scheduled events (whether from DEFAULT_SCHEDULED_EVENTS or dynamically set SCHEDULED_EVENTS)
    events_to_validate = config_to_validate.get("SCHEDULED_EVENTS", config_to_validate.get("DEFAULT_SCHEDULED_EVENTS", []))
    if not isinstance(events_to_validate, list):
        logger.warning("'SCHEDULED_EVENTS' (or DEFAULT_SCHEDULED_EVENTS) is not a list. Defaulting to empty.")
        # Ensure SCHEDULED_EVENTS key exists for simulation if it's going to be used directly
        if "SCHEDULED_EVENTS" not in config_to_validate:
            config_to_validate["SCHEDULED_EVENTS"] = []
    else:
        for idx, event_val in enumerate(events_to_validate):
            if not isinstance(event_val, dict): raise ValueError(f"Event {idx} in SCHEDULED_EVENTS not a dict.")
            req_event_keys_val = ["Event Type", "Start Time (min)", "Duration (min)"]
            if not all(k_val in event_val for k_val in req_event_keys_val):
                raise ValueError(f"Event {idx} ({event_val.get('Event Type', 'Unknown')}) missing required keys: {req_event_keys_val}.")
            if not (isinstance(event_val["Start Time (min)"], (int,float)) and event_val["Start Time (min)"] >= 0):
                raise ValueError(f"Event '{event_val['Event Type']}' invalid Start Time: {event_val['Start Time (min)']}")
            if not (isinstance(event_val["Duration (min)"], (int,float)) and event_val["Duration (min)"] > 0):
                raise ValueError(f"Event '{event_val['Event Type']}' invalid Duration: {event_val['Duration (min)']}")

    if not isinstance(config_to_validate.get("EVENT_TYPE_CONFIG"), dict):
        logger.warning("'EVENT_TYPE_CONFIG' missing/not dict. Event impacts may not apply.")
        config_to_validate["EVENT_TYPE_CONFIG"] = {}

    logger.info("Configuration validation completed (may include warnings).")
    return True
