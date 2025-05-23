# config.py
import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # --- Simulation Core Parameters ---
    "TEAM_SIZE": 50,
    "SHIFT_DURATION_MINUTES": 480,  # 8 hours
    "SHIFT_DURATION_INTERVALS": 240, # Calculated as DURATION_MINUTES / 2 (since each interval is 2 min)
    
    # --- Facility & Work Area Configuration ---
    "FACILITY_SIZE": (100, 80),  # Width, Height in meters
    "ENTRY_EXIT_POINTS": [ # For visualization
        {"name": "Main Entry/Exit", "coords": (5, 40)},
        {"name": "Goods In", "coords": (95, 10)},
        {"name": "Dispatch", "coords": (95, 70)},
    ],
    "WORK_AREAS": { # Detailed configuration for different zones
        "Assembly Line A": {
            "coords": [(10, 5), (70, 25)], # (x0,y0), (x1,y1)
            "workers": 15, # Initial assignment, can be overridden by simulation setup
            "tasks_per_interval": 5, # Avg tasks arriving or needing processing
            "max_concurrent_tasks": 20, # Max tasks this zone can handle at once (capacity buffer)
            "task_complexity": 0.6, # Scale 0-1, higher is more complex
            "base_productivity": 0.85, # Worker productivity factor in this zone (0-1)
            "equipment_dependency": ["Conveyor A", "Robot Arm A1"], # Link to specific equipment
            "proximity_bonus_factor": 0.05 # Small bonus to collaboration if workers are close
        },
        "Assembly Line B": {
            "coords": [(10, 30), (70, 50)],
            "workers": 15,
            "tasks_per_interval": 5,
            "max_concurrent_tasks": 20,
            "task_complexity": 0.65,
            "base_productivity": 0.8,
            "equipment_dependency": ["Conveyor B", "Robot Arm B1"],
            "proximity_bonus_factor": 0.05
        },
        "Quality Control": {
            "coords": [(75, 20), (95, 40)],
            "workers": 5,
            "tasks_per_interval": 2, # Fewer, but critical tasks
            "max_concurrent_tasks": 5,
            "task_complexity": 0.75,
            "base_productivity": 0.9,
            "equipment_dependency": ["QC Station Alpha"],
            "proximity_bonus_factor": 0.02
        },
        "Warehouse": {
            "coords": [(10, 55), (90, 75)],
            "workers": 10,
            "tasks_per_interval": 8, # Material movement tasks
            "max_concurrent_tasks": 15,
            "task_complexity": 0.4,
            "base_productivity": 0.9,
            "equipment_dependency": ["Forklift Main", "AGV System"],
            "proximity_bonus_factor": 0.03
        },
        "Break Room": { # Non-productive area
            "coords": [(0, 70), (10, 80)],
            "workers": 0, # Workers move here during breaks
            "tasks_per_interval": 0,
            "max_concurrent_tasks": 0,
            "task_complexity": 0,
            "base_productivity": 0,
            "equipment_dependency": [],
            "is_休憩area": True # Custom flag to identify break area
        }
    },

    # --- Event Scheduling (NEW) ---
    "DEFAULT_SCHEDULED_EVENTS": [ # Example default schedule
        {"Event Type": "Major Disruption", "Start Time (min)": 60, "Duration (min)": 10, "Intensity": 0.8, "Affected Zones": ["Assembly Line A", "Assembly Line B"]},
        {"Event Type": "Scheduled Break", "Start Time (min)": 120, "Duration (min)": 15, "Scope": "All"},
        {"Event Type": "Minor Disruption", "Start Time (min)": 180, "Duration (min)": 5, "Intensity": 0.3, "Affected Zones": ["Warehouse"]},
        {"Event Type": "Short Pause", "Start Time (min)": 240, "Duration (min)": 5, "Scope": "All"},
        {"Event Type": "Team Meeting", "Start Time (min)": 300, "Duration (min)": 20, "Scope": "All", "Affected Zones": ["Break Room"]}, # Meeting in break room
        {"Event Type": "Maintenance", "Start Time (min)": 360, "Duration (min)": 30, "Affected Zones": ["Assembly Line A"], "Equipment": ["Conveyor A"]}
    ],
    # Parameters for different event types (to be used by simulation.py)
    "EVENT_TYPE_CONFIG": {
        "Major Disruption": {"disruption_compliance_reduction_factor": 0.7, "disruption_wellbeing_drop": 0.3, "downtime_prob": 0.6, "downtime_mean_factor": 1.2},
        "Minor Disruption": {"disruption_compliance_reduction_factor": 0.3, "disruption_wellbeing_drop": 0.1, "downtime_prob": 0.2, "downtime_mean_factor": 0.5},
        "Scheduled Break": {"fatigue_recovery_factor": 0.7, "wellbeing_boost": 10.0}, # Factor to reduce fatigue, absolute boost
        "Short Pause": {"fatigue_recovery_factor": 0.3, "wellbeing_boost": 3.0},
        "Team Meeting": {"cohesion_boost_factor": 0.1, "productivity_during_meeting": 0.1}, # Productivity might drop
        "Maintenance": {"uptime_impact_factor": 0.0, "downtime_prob": 0.9} # e.g. 해당 장비 0% 가동률
    },

    # --- Worker Behavior & Psychosocial Factors ---
    "INITIAL_WELLBEING_MEAN": 0.8, # Scale 0-1
    "WELLBEING_FATIGUE_RATE_PER_INTERVAL": 0.002, # Base rate of fatigue increase per 2-min interval
    "WELLBEING_ALERT_THRESHOLD": 60.0, # % score below which an alert is triggered
    "WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET": 0.85, # For insights: e.g. 85% of 70 target
    "FATIGUE_IMPACT_ON_COMPLIANCE": 0.25, # How much max fatigue (1.0) reduces compliance (e.g., 0.25 means 25% reduction)
    "COMPLEXITY_IMPACT_ON_COMPLIANCE": 0.30, # How much max complexity (1.0) reduces compliance
    "STRESS_FROM_LOW_CONTROL_FACTOR": 0.02, # Impact on wellbeing if autonomy is low
    "ISOLATION_IMPACT_ON_WELLBEING": 0.1, # Factor for wellbeing drop due to low collaboration

    "PSYCH_SAFETY_BASELINE": 0.75, # Scale 0-1
    "PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL": 0.0005, # Natural erosion if not supported
    "UNCERTAINTY_DURING_DISRUPTION_IMPACT_PSYCH_SAFETY": 0.1, # Factor
    "TEAM_COHESION_IMPACT_ON_PSYCH_SAFETY": 0.15, # Factor

    "TEAM_COHESION_BASELINE": 0.7, # Scale 0-1
    "PERCEIVED_WORKLOAD_THRESHOLD_HIGH": 7.5, # Scale 0-10
    "PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH": 8.5, # Scale 0-10
    "TARGET_PERCEIVED_WORKLOAD": 6.5, # Desired average

    # --- Operational Factors ---
    "BASE_TASK_COMPLETION_PROB": 0.95, # If all other factors are neutral
    "MIN_COMPLIANCE_DURING_DISRUPTION": 20.0, # Minimum % compliance can drop to during severe disruption
    "EQUIPMENT_FAILURE_PROB_PER_INTERVAL": 0.005,
    "DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB": 0.7, # If equipment fails, prob it causes downtime
    "EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS": 3, # How many 2-min intervals it's down
    "THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL": 100.0,
    "BASE_QUALITY_DEFECT_RATE": 0.02, # 2% defect rate if compliance is 100%

    # --- Disruption & Recovery (some might be superseded by new Event Scheduling) ---
    "DISRUPTION_COMPLIANCE_REDUCTION_FACTOR": 0.6, # Max reduction factor
    "DISRUPTION_WELLBEING_DROP": 0.2, # Factor for direct wellbeing drop
    "RECOVERY_HALFLIFE_INTERVALS": 10, # Intervals for metrics to recover half-way to potential
    "DOWNTIME_FROM_DISRUPTION_EVENT_PROB": 0.5, # General disruptions causing downtime
    "DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE": 8.0,
    "DOWNTIME_STD_MINUTES_PER_OCCURRENCE": 4.0,
    "DOWNTIME_CAUSES_LIST": ["Equipment Failure", "Material Shortage", "Process Bottleneck", "Human Error", "Utility Outage", "External Supply Chain", "Software Glitch", "Meeting Overrun"],
    "DOWNTIME_PLOT_ALERT_THRESHOLD": 10, # Minutes, for highlighting bars on downtime trend plot

    # --- Leadership & Communication ---
    "LEADERSHIP_SUPPORT_FACTOR": 0.6, # Scale 0-1, impacts wellbeing & psych safety
    "COMMUNICATION_EFFECTIVENESS_FACTOR": 0.7, # Scale 0-1, impacts compliance & psych safety

    # --- Initiatives (Factors applied if initiative is active) ---
    "INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR": 0.4, # e.g., 40% reduction in fatigue accumulation rate
    "INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST_ABS": 0.05, # Absolute % boost to wellbeing score during a break cycle
    "INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS": 0.05,
    "INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS": 0.08,
    "INITIATIVE_RECOGNITION_COHESION_BOOST_ABS": 5.0, # Absolute points boost
    "INITIATIVE_AUTONOMY_WELLBEING_BOOST_ABS": 0.04,
    "INITIATIVE_AUTONOMY_PSYCHSAFETY_BOOST_ABS": 0.07,
    "INITIATIVE_AUTONOMY_COMPLIANCE_BOOST_FACTOR": 0.05, # e.g. 5% boost to compliance

    # --- Dashboard Targets (for display and insights) ---
    "TARGET_COMPLIANCE": 85.0,
    "TARGET_COLLABORATION": 65.0,
    "TARGET_WELLBEING": 75.0,
    "TARGET_PSYCH_SAFETY": 80.0,
    "TARGET_TEAM_COHESION": 75.0,
    "DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT": 0.05, # 5% of total shift duration as acceptable downtime
    # DOWNTIME_THRESHOLD_TOTAL_SHIFT will be calculated from this: SHIFT_DURATION_MINUTES * DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT
}

def validate_config(config_to_validate):
    """
    Validates the provided configuration dictionary.
    Logs warnings or errors for missing or invalid critical parameters.
    """
    logger.info("Validating configuration...")
    required_keys = [
        "TEAM_SIZE", "SHIFT_DURATION_MINUTES", "FACILITY_SIZE", "WORK_AREAS",
        "SCHEDULED_EVENTS", # Check for the new event structure
        "EVENT_TYPE_CONFIG" 
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
        # Add more specific checks for work area details if needed (e.g., coords format)
        if "coords" in details:
            if not (isinstance(details["coords"], list) and len(details["coords"]) == 2 and
                    all(isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(c, (int, float)) for c in pt) for pt in details["coords"])):
                logger.error(f"Invalid 'coords' for WORK_AREA '{area_name}'. Must be a list of two (x,y) tuples.")
                # raise ValueError(f"Invalid 'coords' for WORK_AREA '{area_name}'.") # Can be warning if fallback exists
    
    if not isinstance(config_to_validate.get("SCHEDULED_EVENTS"), list):
        logger.warning("'SCHEDULED_EVENTS' is missing or not a list. Defaulting to empty if simulation handles it.")
        # Depending on how critical this is, you might raise an error or allow an empty list.
        # For now, we assume simulation.py can handle an empty list.
        if "SCHEDULED_EVENTS" not in config_to_validate:
            config_to_validate["SCHEDULED_EVENTS"] = []


    if not isinstance(config_to_validate.get("EVENT_TYPE_CONFIG"), dict):
        logger.warning("'EVENT_TYPE_CONFIG' is missing or not a dictionary. Event impacts may not be applied correctly.")
        if "EVENT_TYPE_CONFIG" not in config_to_validate:
            config_to_validate["EVENT_TYPE_CONFIG"] = {}


    # Example: Check if sum of workers in WORK_AREAS matches TEAM_SIZE (can be optional)
    # total_configured_workers = sum(details.get('workers', 0) for details in config_to_validate["WORK_AREAS"].values())
    # if config_to_validate["TEAM_SIZE"] > 0 and total_configured_workers != config_to_validate["TEAM_SIZE"]:
    #     logger.warning(f"Sum of workers in WORK_AREAS ({total_configured_workers}) does not match TEAM_SIZE ({config_to_validate['TEAM_SIZE']}). Simulation will redistribute.")

    logger.info("Configuration validation passed (or warnings issued).")
    return True # Or return the validated (potentially modified) config

# Example usage:
if __name__ == "__main__":
    try:
        validate_config(DEFAULT_CONFIG)
        print("Default configuration is valid.")
    except ValueError as e:
        print(f"Configuration error: {e}")
