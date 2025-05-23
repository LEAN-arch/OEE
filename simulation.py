# simulation.py
import numpy as np
import pandas as pd
import random
import math
import logging

logger = logging.getLogger(__name__)
EPSILON = 1e-6

def _get_config_param(config, key, default):
    return config.get(key, default)

# MODIFIED Function Signature: Added 'scheduled_events', removed 'disruption_event_steps'
def simulate_workplace_operations(num_team_members: int, num_steps: int, 
                                  scheduled_events: list, # NEW PARAMETER
                                  team_initiative: str, config: dict):
    np.random.seed(42)
    random.seed(42)

    facility_width, facility_height = _get_config_param(config, 'FACILITY_SIZE', (100, 80))
    work_areas_config = _get_config_param(config, 'WORK_AREAS', {})
    event_type_config = _get_config_param(config, 'EVENT_TYPE_CONFIG', {}) # Get event type specifics
    
    # --- Initialize Metric Arrays --- (same as before)
    _task_compliance_scores = np.zeros(num_steps)
    _collaboration_scores = np.zeros(num_steps)
    _operational_recovery_scores = np.zeros(num_steps)
    _wellbeing_scores = np.zeros(num_steps)
    _psych_safety_scores = np.zeros(num_steps)
    _team_cohesion_scores = np.zeros(num_steps)
    _perceived_workload_scores = np.zeros(num_steps)
    _productivity_loss_percent = np.zeros(num_steps)
    _downtime_events_per_interval = [{'duration': 0.0, 'cause': 'None'} for _ in range(num_steps)]
    _task_completion_rate_percent = np.zeros(num_steps)
    _uptime_percent = np.ones(num_steps) * 100.0
    _quality_rate_percent = np.ones(num_steps) * 100.0
    _throughput_percent_of_max = np.zeros(num_steps)

    # --- Worker and Zone Setup --- (same as before)
    team_positions_data = []
    worker_current_x = np.random.uniform(0, facility_width, num_team_members) if num_team_members > 0 else np.array([])
    worker_current_y = np.random.uniform(0, facility_height, num_team_members) if num_team_members > 0 else np.array([])
    
    if not work_areas_config and num_team_members > 0:
        logger.warning("Simulation: No WORK_AREAS defined in config. Creating and assigning all workers to 'FallbackZone'.")
        work_areas_config["FallbackZone"] = {
            'coords': [(0,0), (facility_width, facility_height)], 'workers': num_team_members, 
            'tasks_per_interval': 1, 'task_complexity': 0.5, 'base_productivity': 0.7, 
            'max_concurrent_tasks': max(1, num_team_members)
        }
    
    worker_zone_names_from_config = list(work_areas_config.keys())
    worker_assigned_zone = []

    if num_team_members > 0:
        if not worker_zone_names_from_config:
            logger.error("Critical: No zones available for worker assignment even after fallback check.")
            worker_assigned_zone = ["ErrorZone"] * num_team_members
        else:
            configured_workers_sum = sum(zone_details.get('workers', 0) for zone_details in work_areas_config.values())
            if configured_workers_sum == num_team_members:
                for zone_name, zone_details in work_areas_config.items():
                    worker_assigned_zone.extend([zone_name] * zone_details.get('workers', 0))
            else: 
                logger.info(f"Sum of configured workers ({configured_workers_sum}) does not match team size ({num_team_members}). Distributing workers.")
                for i in range(num_team_members):
                    worker_assigned_zone.append(worker_zone_names_from_config[i % len(worker_zone_names_from_config)])
            if len(worker_assigned_zone) > num_team_members: worker_assigned_zone = worker_assigned_zone[:num_team_members]
            while len(worker_assigned_zone) < num_team_members: worker_assigned_zone.append(random.choice(worker_zone_names_from_config))
            random.shuffle(worker_assigned_zone)

    worker_fatigue = np.random.uniform(0.0, 0.1, num_team_members) if num_team_members > 0 else np.array([])
    zone_task_backlog = {zn: 0 for zn in work_areas_config.keys()}

    # --- Simulation State Variables ---
    # active_disruption_intensity will now be determined by active events
    # disruption_linger_factor can still be used for gradual recovery after disruptions end

    recovery_halflife_intervals = _get_config_param(config, 'RECOVERY_HALFLIFE_INTERVALS', 10)
    disruption_linger_factor = 1.0 - (1.0 / (recovery_halflife_intervals + EPSILON))
    
    # Store active event effects, multiple events could be active
    current_event_effects = {
        "compliance_reduction_factor": 0.0,
        "wellbeing_drop_factor": 0.0, # Factor, not absolute drop
        "downtime_prob_modifier": 0.0, # Additive to base prob
        "downtime_mean_modifier": 1.0, # Multiplicative
        "fatigue_rate_modifier": 1.0,  # Multiplicative
        "fatigue_recovery_factor": 0.0, # How much fatigue reduces e.g. during break
        "wellbeing_boost_abs": 0.0,     # Absolute wellbeing boost
        "cohesion_boost_abs": 0.0,
        "psych_safety_boost_abs": 0.0,
        "specific_zone_effects": {} # e.g. {"Assembly Line A": {"uptime_multiplier": 0.1}}
    }

    # Initialize step 0 values (same as before)
    if num_steps > 0:
        _wellbeing_scores[0] = _get_config_param(config, 'INITIAL_WELLBEING_MEAN', 0.8) * 100.0
        _psych_safety_scores[0] = _get_config_param(config, 'PSYCH_SAFETY_BASELINE', 0.75) * 100.0
        _team_cohesion_scores[0] = _get_config_param(config, 'TEAM_COHESION_BASELINE', 0.7) * 100.0
        _operational_recovery_scores[0] = 100.0
        _perceived_workload_scores[0] = np.clip(
             sum(zone_task_backlog.values()) / 
             (sum(zd.get('max_concurrent_tasks', zd.get('tasks_per_interval', 0) * 1.5) for zd in work_areas_config.values()) + EPSILON) * 10.0,
             0, 10
        )

    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa: [] for wa in work_areas_config}, 'disruption': []}
    downtime_causes_list = _get_config_param(config, 'DOWNTIME_CAUSES_LIST', ["Equipment Failure", "Material Shortage", "Process Bottleneck", "Human Error", "Utility Outage", "External Supply Chain"])

    # --- Main Simulation Loop ---
    for step in range(num_steps):
        current_minute_of_shift = step * 2 # Each step is 2 minutes

        # Reset event effects for the current step (they are re-calculated based on active events)
        current_event_effects = {
            "compliance_reduction_factor": 0.0, "wellbeing_drop_factor": 0.0,
            "downtime_prob_modifier": 0.0, "downtime_mean_modifier": 1.0,
            "fatigue_rate_modifier": 1.0, "fatigue_recovery_factor": 0.0,
            "wellbeing_boost_abs": 0.0, "cohesion_boost_abs": 0.0,
            "psych_safety_boost_abs": 0.0, "specific_zone_effects": {}
        }
        is_any_disruption_active = False # Flag if any disruption-type event is active

        # --- Process Scheduled Events ---
        active_events_this_step = []
        for event_details in scheduled_events:
            if not isinstance(event_details, dict): continue # Skip malformed events

            event_start_min = event_details.get("Start Time (min)", -1)
            event_duration_min = event_details.get("Duration (min)", 0)
            event_type = event_details.get("Event Type", "Unknown")

            if event_start_min <= current_minute_of_shift < event_start_min + event_duration_min:
                active_events_this_step.append(event_details) # This event is active
                event_params = event_type_config.get(event_type, {})
                
                logger.debug(f"Step {step} ({current_minute_of_shift} min): Event '{event_type}' ACTIVE. Params: {event_params}")

                if "Disruption" in event_type: # Handle general disruption effects
                    is_any_disruption_active = True
                    # Aggregate effects if multiple disruptions overlap (e.g., take max impact)
                    current_event_effects["compliance_reduction_factor"] = max(
                        current_event_effects["compliance_reduction_factor"], 
                        event_params.get("disruption_compliance_reduction_factor", 0.0) * event_details.get("Intensity", 1.0)
                    )
                    current_event_effects["wellbeing_drop_factor"] = max(
                        current_event_effects["wellbeing_drop_factor"],
                        event_params.get("disruption_wellbeing_drop", 0.0) * event_details.get("Intensity", 1.0)
                    )
                    # For downtime, these might need more complex aggregation if multiple disruptions increase prob
                    current_event_effects["downtime_prob_modifier"] += event_params.get("downtime_prob", 0.0) 
                    current_event_effects["downtime_mean_modifier"] *= event_params.get("downtime_mean_factor", 1.0)
                    current_event_effects["fatigue_rate_modifier"] = max(current_event_effects["fatigue_rate_modifier"], 1.3) # Example: disruptions increase fatigue rate

                    if current_minute_of_shift == event_start_min : # If it's the start of this disruption
                         wellbeing_triggers_dict['disruption'].append(step)


                elif event_type == "Scheduled Break" or event_type == "Short Pause":
                    current_event_effects["fatigue_recovery_factor"] = max(
                        current_event_effects["fatigue_recovery_factor"],
                        event_params.get("fatigue_recovery_factor", 0.0)
                    )
                    current_event_effects["wellbeing_boost_abs"] += event_params.get("wellbeing_boost", 0.0)
                
                elif event_type == "Team Meeting":
                    current_event_effects["cohesion_boost_abs"] += event_params.get("cohesion_boost_abs", 0.0)
                    # Could also make workers in "Break Room" (if specified as meeting location) non-productive
                    affected_zones_meeting = event_details.get("Affected Zones", [])
                    if not affected_zones_meeting or "All" in affected_zones_meeting: # Apply to all if not specified or "All"
                        for zone_name in work_areas_config.keys():
                            current_event_effects["specific_zone_effects"].setdefault(zone_name, {})["productivity_multiplier"] = event_params.get("productivity_during_meeting", 0.1)
                    else:
                        for zone_name in affected_zones_meeting:
                             current_event_effects["specific_zone_effects"].setdefault(zone_name, {})["productivity_multiplier"] = event_params.get("productivity_during_meeting", 0.1)


                elif event_type == "Maintenance":
                    affected_zones_maint = event_details.get("Affected Zones", [])
                    # This implies direct impact on uptime for these zones/equipment
                    # This needs to be integrated with how uptime is calculated per zone if you have per-zone uptime
                    # For now, let's assume it causes a general downtime probability increase
                    current_event_effects["downtime_prob_modifier"] += event_params.get("downtime_prob", 0.0)
                    # And could set specific equipment uptime to 0 (needs equipment modeling)

        # --- Task Backlog and Workload --- (Largely same, but productivity might be affected by events)
        avg_fatigue_current_step = np.mean(worker_fatigue) if num_team_members > 0 else 0.0
        for zone_name, zone_details in work_areas_config.items():
            tasks_arriving = zone_details.get('tasks_per_interval', 0) * (1.0 + (0.2*random.random()-0.1) if is_any_disruption_active else 0.0) # Disruption can slightly alter task arrival
            workers_in_this_zone_count = worker_assigned_zone.count(zone_name) if num_team_members > 0 else 0
            
            prev_compliance_score = _task_compliance_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'BASE_TASK_COMPLETION_PROB', 0.95) * 100.0
            compliance_factor_for_processing = prev_compliance_score / 100.0
            
            zone_productivity_multiplier = current_event_effects["specific_zone_effects"].get(zone_name, {}).get("productivity_multiplier", 1.0)

            zone_processing_capacity = workers_in_this_zone_count * \
                                     zone_details.get('base_productivity', 0.8) * \
                                     (1.0 - avg_fatigue_current_step * 0.3) * \
                                     compliance_factor_for_processing * \
                                     zone_productivity_multiplier # Apply event-specific productivity modifier
            
            zone_task_backlog[zone_name] = max(0, zone_task_backlog.get(zone_name,0) + tasks_arriving - zone_processing_capacity)

        total_backlog_current_step = sum(zone_task_backlog.values())
        max_concurrent_total_facility = sum(zd.get('max_concurrent_tasks', zd.get('tasks_per_interval', 0) * 1.5) for zd in work_areas_config.values())
        
        workload_pressure_from_backlog_metric = total_backlog_current_step / (max_concurrent_total_facility + EPSILON)
        _perceived_workload_scores[step] = np.clip(workload_pressure_from_backlog_metric * 10.0 + (step / (num_steps + EPSILON)) * 1.5, 0, 10)

        # --- Fatigue ---
        fatigue_rate_this_step = _get_config_param(config, 'WELLBEING_FATIGUE_RATE_PER_INTERVAL', 0.002)
        # Initiative effects on fatigue rate
        if team_initiative == "More frequent breaks": # This initiative's effect is now primarily handled by SCHEDULED_EVENTS
            fatigue_rate_this_step *= (1.0 - _get_config_param(config, 'INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.3)) # Still keep a general reduction if configured
        
        fatigue_rate_this_step *= (1.0 + _perceived_workload_scores[step] / 15.0) 
        fatigue_rate_this_step *= current_event_effects["fatigue_rate_modifier"] # Apply event-based modifier

        if num_team_members > 0:
            worker_fatigue += fatigue_rate_this_step
            # Apply fatigue recovery from active breaks/pauses
            if current_event_effects["fatigue_recovery_factor"] > 0:
                worker_fatigue *= (1.0 - current_event_effects["fatigue_recovery_factor"])
            worker_fatigue = np.clip(worker_fatigue, 0.0, 1.0)
        avg_fatigue_current_step = np.mean(worker_fatigue) if num_team_members > 0 else 0.0

        # --- Wellbeing ---
        wb_now = _wellbeing_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'INITIAL_WELLBEING_MEAN', 0.8) * 100.0
        wb_now -= (avg_fatigue_current_step * 20.0 + (_perceived_workload_scores[step] - 5.0) * 1.0) 
        wb_now -= current_event_effects["wellbeing_drop_factor"] * 25.0 # Apply aggregated disruption drop factor
        wb_now += current_event_effects["wellbeing_boost_abs"] # Apply boost from breaks/pauses
        
        leadership_support_factor = _get_config_param(config, 'LEADERSHIP_SUPPORT_FACTOR', 0.5)
        wb_now += (leadership_support_factor - 0.5) * 10.0 

        if team_initiative != "Increased Autonomy" and not any(event.get("Event Type") == "Increased Autonomy Mode" for event in active_events_this_step): # Example check for a hypothetical Autonomy event
            wb_now -= _get_config_param(config, 'STRESS_FROM_LOW_CONTROL_FACTOR', 0.02) * 15.0
        
        prev_collab_score = _collaboration_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'TARGET_COLLABORATION', 60.0)
        if prev_collab_score < 50.0: 
            wb_now -= _get_config_param(config, 'ISOLATION_IMPACT_ON_WELLBEING', 0.1) * (50.0 - prev_collab_score) * 0.4
        
        # Initiative-specific static wellbeing boosts (could be event-driven too)
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,3) if num_steps > 2 else 1) == 0:
            wb_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS', 0.05) * 100.0
        if team_initiative == "Increased Autonomy": # General boost for having the initiative
            wb_now += _get_config_param(config, 'INITIATIVE_AUTONOMY_WELLBEING_BOOST_ABS', 0.04) * 100.0
        _wellbeing_scores[step] = np.clip(wb_now, 5.0, 100.0)

        # --- Psychological Safety ---
        ps_now = _psych_safety_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'PSYCH_SAFETY_BASELINE', 0.75) * 100.0
        ps_now -= _get_config_param(config, 'PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL', 0.0005) * 100.0
        if is_any_disruption_active: # Simplified: any disruption affects psych safety
             ps_now -= _get_config_param(config, 'UNCERTAINTY_DURING_DISRUPTION_IMPACT_PSYCH_SAFETY', 0.1) * 15.0
        ps_now += (leadership_support_factor - 0.5) * 15.0
        ps_now += (_get_config_param(config, 'COMMUNICATION_EFFECTIVENESS_FACTOR', 0.5) - 0.5) * 10.0
        ps_now += current_event_effects["psych_safety_boost_abs"] # If events boost this
        
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,3) if num_steps > 2 else 1) == 1: 
            ps_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS', 0.08) * 100.0
        if team_initiative == "Increased Autonomy": 
            ps_now += _get_config_param(config, 'INITIATIVE_AUTONOMY_PSYCHSAFETY_BOOST_ABS', 0.07) * 100.0
        
        prev_team_cohesion_score = _team_cohesion_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'TEAM_COHESION_BASELINE', 0.7) * 100.0
        ps_now += (prev_team_cohesion_score - _get_config_param(config, 'TEAM_COHESION_BASELINE', 0.7) * 100.0) * _get_config_param(config, 'TEAM_COHESION_IMPACT_ON_PSYCH_SAFETY', 0.15)
        _psych_safety_scores[step] = np.clip(ps_now, 10.0, 100.0)

        # --- Team Cohesion ---
        cohesion_now = prev_team_cohesion_score
        cohesion_now -= (0.15 + (0.7 if is_any_disruption_active else 0.0) + (_perceived_workload_scores[step] / 50.0)) # Disruption impact
        cohesion_now += current_event_effects["cohesion_boost_abs"]
        if _psych_safety_scores[step] > 70.0: cohesion_now += 0.8
        if prev_collab_score > 60.0: cohesion_now += 0.6
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,3) if num_steps > 2 else 1) == 0:
            cohesion_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_COHESION_BOOST_ABS', 5.0)
        _team_cohesion_scores[step] = np.clip(cohesion_now, 10.0, 100.0) 

        # --- Wellbeing Triggers --- (same logic, just ensure data is from current step)
        if _wellbeing_scores[step] < _get_config_param(config, 'WELLBEING_ALERT_THRESHOLD', 60.0):
            wellbeing_triggers_dict['threshold'].append(step)
        if step > 0 and _wellbeing_scores[step] < _wellbeing_scores[step - 1] - 12.0: 
            wellbeing_triggers_dict['trend'].append(step)
        
        # Work area specific triggers if a disruption started and wellbeing dropped significantly for workers in certain zones
        # This part needs to map general disruption effect to specific zones if not already in event_details
        if is_any_disruption_active and _wellbeing_scores[step] < (_wellbeing_scores[max(0, step - 1)] if step > 0 else _wellbeing_scores[0]) * 0.90:
            if num_team_members > 0:
                affected_worker_indices = [i for i in range(num_team_members) if worker_fatigue[i] > 0.65 or random.random() < current_event_effects["compliance_reduction_factor"] * 0.2] # Example link to disruption intensity proxy
                affected_zones_this_step = [worker_assigned_zone[i] for i in affected_worker_indices if i < len(worker_assigned_zone)]
                for zone_name_affected in set(affected_zones_this_step):
                    if zone_name_affected in wellbeing_triggers_dict['work_area']:
                        wellbeing_triggers_dict['work_area'][zone_name_affected].append(step)
            
        # --- Operational Metrics (Uptime, Compliance, Quality, Throughput) ---
        prev_uptime = _uptime_percent[max(0, step - 1)] if step > 0 else 100.0
        current_uptime = prev_uptime
        equipment_failed_this_step_flag = False
        if random.random() < _get_config_param(config, 'EQUIPMENT_FAILURE_PROB_PER_INTERVAL', 0.005):
            current_uptime -= random.uniform(15, 35)
            equipment_failed_this_step_flag = True
        
        if is_any_disruption_active: # Generic disruption impact on uptime
            current_uptime -= 30.0 * current_event_effects.get("compliance_reduction_factor", 0.0) # Example scaling by intensity proxy
        _uptime_percent[step] = np.clip(current_uptime, 15.0, 100.0)
        
        base_compliance_val = _get_config_param(config, 'BASE_TASK_COMPLETION_PROB', 0.95) * 100.0
        compliance_now = base_compliance_val * (1.0 - avg_fatigue_current_step * _get_config_param(config, 'FATIGUE_IMPACT_ON_COMPLIANCE', 0.25))
        avg_worker_task_complexity = 0.5
        if num_team_members > 0 and worker_assigned_zone:
            complexities = [work_areas_config.get(worker_assigned_zone[w], {}).get('task_complexity', 0.5) for w in range(num_team_members) if w < len(worker_assigned_zone)]
            if complexities: avg_worker_task_complexity = np.mean(complexities)
        compliance_now -= avg_worker_task_complexity * _get_config_param(config, 'COMPLEXITY_IMPACT_ON_COMPLIANCE', 0.30) * 100.0
        compliance_now *= (_psych_safety_scores[step] / 100.0 * 0.1 + 0.9) 
        compliance_now *= (_get_config_param(config, 'COMMUNICATION_EFFECTIVENESS_FACTOR', 0.7) * 0.3 + 0.7)
        
        min_compliance_during_disruption = _get_config_param(config, 'MIN_COMPLIANCE_DURING_DISRUPTION', 20.0)
        compliance_now = max(min_compliance_during_disruption, compliance_now * (1.0 - current_event_effects["compliance_reduction_factor"]))
        _task_compliance_scores[step] = np.clip(compliance_now, 0.0, 100.0)

        quality_now = (100.0 - _get_config_param(config, 'BASE_QUALITY_DEFECT_RATE', 0.02) * 100.0) * \
                      (_task_compliance_scores[step] / 100.0)**1.1 
        if is_any_disruption_active: quality_now -= 12.0 * current_event_effects.get("compliance_reduction_factor", 0.0) # Scaled impact
        _quality_rate_percent[step] = np.clip(quality_now, 30.0, 100.0)

        max_potential_throughput_facility = _get_config_param(config, 'THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL', 100.0)
        effective_capacity_factor = (_uptime_percent[step] / 100.0) * \
                                    (_task_compliance_scores[step] / 100.0) * \
                                    (1.0 - avg_fatigue_current_step * 0.6) * \
                                    (1.0 - _perceived_workload_scores[step] / 30.0)
        throughput_disruption_impact = 0.8 * current_event_effects.get("compliance_reduction_factor", 0.0) # Higher impact for disruptions
        actual_units_produced = max_potential_throughput_facility * effective_capacity_factor * (1.0 - throughput_disruption_impact)
        _throughput_percent_of_max[step] = np.clip((actual_units_produced / (max_potential_throughput_facility + EPSILON)) * 100.0, 0.0, 100.0)
        _task_completion_rate_percent[step] = _throughput_percent_of_max[step]

        # --- Downtime ---
        current_downtime_duration = 0.0; current_downtime_cause = "None"
        # Event-driven downtime (from disruptions or maintenance)
        if current_event_effects["downtime_prob_modifier"] > 0 and random.random() < current_event_effects["downtime_prob_modifier"]:
            downtime_mean_event = _get_config_param(config, 'DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE', 8.0) * current_event_effects["downtime_mean_modifier"]
            downtime_std_event = _get_config_param(config, 'DOWNTIME_STD_MINUTES_PER_OCCURRENCE', 4.0) * math.sqrt(current_event_effects["downtime_mean_modifier"]) # Scale std dev
            current_downtime_duration = max(0.0, np.random.normal(downtime_mean_event, downtime_std_event))
            # Determine cause based on active event types
            active_disrupt_types = [evt.get("Event Type") for evt in active_events_this_step if "Disruption" in evt.get("Event Type","") or "Maintenance" in evt.get("Event Type","")]
            if active_disrupt_types: current_downtime_cause = active_disrupt_types[0] # Simplistic: first one wins
            else: current_downtime_cause = random.choice([c for c in downtime_causes_list if c not in ["Equipment Failure", "Human Error"]])
        
        if equipment_failed_this_step_flag and random.random() < _get_config_param(config, 'DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB', 0.7): 
             duration_equip_fail = _get_config_param(config, 'EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS', 3) * 2.0 
             current_downtime_duration = max(current_downtime_duration, duration_equip_fail) 
             current_downtime_cause = "Equipment Failure" if current_downtime_cause == "None" else f"{current_downtime_cause}, Equip.Fail"
        if (_task_compliance_scores[step] < 50 or avg_fatigue_current_step > 0.9) and random.random() < 0.03: 
            duration_human_error = max(0.0, np.random.normal(_get_config_param(config, 'DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',8.0)*0.4, _get_config_param(config, 'DOWNTIME_STD_MINUTES_PER_OCCURRENCE',4.0)*0.2))
            current_downtime_duration = max(current_downtime_duration, duration_human_error)
            current_downtime_cause = "Human Error" if current_downtime_cause == "None" else f"{current_downtime_cause}, HumanError"
        
        interval_actual_length_minutes = (shift_duration_minutes_config / (num_steps if num_steps > 0 else 1.0)) 
        _downtime_events_per_interval[step] = {'duration': np.clip(current_downtime_duration, 0, interval_actual_length_minutes), 'cause': current_downtime_cause if current_downtime_duration > EPSILON else "None"}

        # --- Operational Recovery & Productivity Loss --- (same logic)
        current_oee_calc = (_uptime_percent[step]/100) * (_throughput_percent_of_max[step]/100) * (_quality_rate_percent[step]/100)
        if not is_any_disruption_active: # Check if truly no disruptions active based on new system
            prev_recovery = _operational_recovery_scores[max(0,step-1)]; target_potential_recovery = current_oee_calc * 100
            _operational_recovery_scores[step] = np.clip(prev_recovery + (target_potential_recovery - prev_recovery) * (1 - math.exp(-1/(recovery_halflife_intervals + EPSILON))), 0, 100)
        else: _operational_recovery_scores[step] = np.clip(current_oee_calc * 100, 0, 100)
        _productivity_loss_percent[step] = np.clip(100 - _operational_recovery_scores[step] + np.random.normal(0,0.5), 0, 100) 

        # --- Collaboration Score --- (Adjust disruption impact)
        base_collab = 60 + (_team_cohesion_scores[step]-70)*0.5 - (_perceived_workload_scores[step]-5)*2.5 
        if is_any_disruption_active: base_collab -= 25.0 * current_event_effects.get("compliance_reduction_factor", 0.0) # Scale by intensity proxy
        _collaboration_scores[step] = np.clip(base_collab + np.random.normal(0,2) ,5, 95)

        # --- Worker Movement and Status --- (Consider if events affect status, e.g., 'break' status if break event is active)
        if num_team_members > 0 : 
            for i in range(num_team_members):
                current_assigned_zone_for_worker_i = worker_assigned_zone[i]
                zone_details = work_areas_config.get(current_assigned_zone_for_worker_i, {}); zone_coords = zone_details.get('coords')
                if zone_coords and len(zone_coords) == 2: (zx0, zy0), (zx1, zy1) = zone_coords; target_x, target_y = (zx0+zx1)/2, (zy0+zy1)/2
                else: target_x, target_y = facility_width/2, facility_height/2 
                move_x = (target_x - worker_current_x[i]) * 0.2 + np.random.normal(0, 0.8); move_y = (target_y - worker_current_y[i]) * 0.2 + np.random.normal(0, 0.8)
                worker_current_x[i] = np.clip(worker_current_x[i] + move_x, 0, facility_width); worker_current_y[i] = np.clip(worker_current_y[i] + move_y, 0, facility_height)
                
                status_now = 'working'
                # Check for active break/pause events affecting this worker/zone
                is_on_scheduled_break = False
                for active_event in active_events_this_step:
                    event_type = active_event.get("Event Type")
                    if event_type in ["Scheduled Break", "Short Pause"]:
                        scope = active_event.get("Scope", "All")
                        affected_zones_event = active_event.get("Affected Zones", [])
                        if scope == "All" or current_assigned_zone_for_worker_i in affected_zones_event:
                            is_on_scheduled_break = True
                            break
                
                if is_on_scheduled_break:
                    status_now = 'break'
                elif worker_fatigue[i] > 0.85: status_now = 'exhausted' 
                elif worker_fatigue[i] > 0.65: status_now = 'fatigued'
                elif is_any_disruption_active and random.random() < 0.3: status_now = 'disrupted' # Some chance of being disrupted status
                
                team_positions_data.append({'step': step, 'worker_id': i, 'x': worker_current_x[i], 'y': worker_current_y[i], 'z': np.random.uniform(0, 0.1), 'zone': current_assigned_zone_for_worker_i, 'status': status_now})
    
    # --- Prepare Output Data --- (same as before)
    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame(columns=['step', 'worker_id', 'x', 'y', 'z', 'zone', 'status'])
    task_compliance_forecast = [max(0, s - random.uniform(1,5)) for s in _task_compliance_scores]
    collab_proximity_forecast = [min(100, s + random.uniform(1,5)) for s in _collaboration_scores]
    task_compliance = {'data': list(_task_compliance_scores), 'z_scores': list(np.random.normal(0, 0.5, num_steps)), 'forecast': task_compliance_forecast}
    collaboration_proximity = {'data': list(_collaboration_scores), 'forecast': collab_proximity_forecast}
    operational_recovery = list(_operational_recovery_scores)
    efficiency_df_data = {'uptime': list(_uptime_percent), 'throughput': list(_throughput_percent_of_max), 'quality': list(_quality_rate_percent)}
    efficiency_df = pd.DataFrame(efficiency_df_data); efficiency_df['oee'] = np.clip((efficiency_df['uptime']/100.0 * efficiency_df['throughput']/100.0 * efficiency_df['quality']/100.0) * 100.0, 0.0, 100.0)
    productivity_loss = list(_productivity_loss_percent)
    worker_wellbeing = {'scores': list(_wellbeing_scores), 'triggers': wellbeing_triggers_dict, 'team_cohesion_scores': list(_team_cohesion_scores), 'perceived_workload_scores': list(_perceived_workload_scores)} 
    psychological_safety = list(_psych_safety_scores)
    feedback_impact = list(np.random.choice([-0.1, -0.05, 0, 0.05, 0.1], num_steps, p=[0.1,0.2,0.4,0.2,0.1])) 
    downtime_events_final = _downtime_events_per_interval
    task_completion_rate = list(_task_completion_rate_percent)

    logger.info(f"Simulation completed for {num_steps} steps with {num_team_members} team members. Initiative: {team_initiative}.")

    return (
        team_positions_df, task_compliance, collaboration_proximity, 
        operational_recovery, efficiency_df, productivity_loss, 
        worker_wellbeing, psychological_safety, feedback_impact, 
        downtime_events_final, task_completion_rate
    )
