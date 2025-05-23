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

def simulate_workplace_operations(num_team_members: int, num_steps: int, 
                                  scheduled_events: list, 
                                  team_initiative: str, config: dict):
    np.random.seed(42)
    random.seed(42)

    facility_width, facility_height = _get_config_param(config, 'FACILITY_SIZE', (100, 80))
    work_areas_config = _get_config_param(config, 'WORK_AREAS', {})
    event_type_params_config = _get_config_param(config, 'EVENT_TYPE_CONFIG', {}) 
    shift_duration_minutes_sim = float(_get_config_param(config, 'SHIFT_DURATION_MINUTES', 480))
    
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

    team_positions_data = []
    worker_current_x = np.random.uniform(0, facility_width, num_team_members) if num_team_members > 0 else np.array([])
    worker_current_y = np.random.uniform(0, facility_height, num_team_members) if num_team_members > 0 else np.array([])
    
    if not work_areas_config and num_team_members > 0:
        work_areas_config["FallbackZone"] = {
            'coords': [(0,0), (facility_width, facility_height)], 'workers': num_team_members, 
            'tasks_per_interval': 1, 'task_complexity': 0.5, 'base_productivity': 0.7, 
            'max_concurrent_tasks': max(1, num_team_members)
        }
    worker_zone_names_from_config = list(work_areas_config.keys())
    worker_assigned_zone = ["FallbackZone"] * num_team_members 
    if num_team_members > 0 and worker_zone_names_from_config:
        temp_assigned = []
        for zn, zd in work_areas_config.items():
            temp_assigned.extend([zn] * zd.get('workers',0))
        
        if len(temp_assigned) == num_team_members:
            worker_assigned_zone = temp_assigned
        else: 
            worker_assigned_zone = [worker_zone_names_from_config[i % len(worker_zone_names_from_config)] for i in range(num_team_members)]
        random.shuffle(worker_assigned_zone)

    worker_fatigue = np.random.uniform(0.0, 0.1, num_team_members) if num_team_members > 0 else np.array([])
    zone_task_backlog = {zn: 0.0 for zn in work_areas_config.keys()}

    recovery_halflife_intervals = _get_config_param(config, 'RECOVERY_HALFLIFE_INTERVALS', 10)
        
    if num_steps > 0:
        _wellbeing_scores[0] = _get_config_param(config, 'INITIAL_WELLBEING_MEAN', 0.8) * 100.0
        _psych_safety_scores[0] = _get_config_param(config, 'PSYCH_SAFETY_BASELINE', 0.75) * 100.0
        _team_cohesion_scores[0] = _get_config_param(config, 'TEAM_COHESION_BASELINE', 0.7) * 100.0
        _operational_recovery_scores[0] = 100.0
        _perceived_workload_scores[0] = 0.0

    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa: [] for wa in work_areas_config}, 'disruption': []}
    downtime_causes_list = _get_config_param(config, 'DOWNTIME_CAUSES_LIST', ["Equipment Failure", "Material Shortage", "Process Bottleneck", "Human Error", "Utility Outage", "External Supply Chain"])

    for step in range(num_steps):
        current_minute_of_shift = step * 2 

        active_event_effects = { 
            "compliance_reduction_factor": 0.0, "wellbeing_drop_factor": 0.0,
            "downtime_prob_modifier": 0.0, "downtime_mean_factor": 1.0, # Default to 1.0 for multiplicative factor
            "fatigue_rate_modifier": 1.0, "fatigue_recovery_factor": 0.0,
            "wellbeing_boost_abs": 0.0, "cohesion_boost_abs": 0.0,
            "psych_safety_boost_abs": 0.0,
            "zone_productivity_multiplier": {zn: 1.0 for zn in work_areas_config.keys()},
            "zone_uptime_multiplier": {zn: 1.0 for zn in work_areas_config.keys()}
        }
        is_any_disruption_active_this_step = False
        active_event_types_this_step = set()
        active_events_details_this_step = [] 

        for event_def in scheduled_events: 
            if not isinstance(event_def, dict): continue 

            event_start_min = event_def.get("Start Time (min)", -1)
            event_duration_min = event_def.get("Duration (min)", 0)
            event_type = event_def.get("Event Type", "Unknown")

            if event_start_min <= current_minute_of_shift < event_start_min + event_duration_min:
                active_event_types_this_step.add(event_type)
                active_events_details_this_step.append(event_def) 
                event_params = event_type_params_config.get(event_type, {}) 
                
                logger.debug(f"Step {step} ({current_minute_of_shift} min): Event '{event_type}' ACTIVE. Config Params: {event_params}")

                if "Disruption" in event_type: 
                    is_any_disruption_active_this_step = True
                    intensity = event_def.get("Intensity", 1.0)
                    active_event_effects["compliance_reduction_factor"] = max(
                        active_event_effects["compliance_reduction_factor"], 
                        event_params.get("compliance_reduction_factor", 0.0) * intensity
                    )
                    active_event_effects["wellbeing_drop_factor"] = max(
                        active_event_effects["wellbeing_drop_factor"],
                        event_params.get("wellbeing_drop_factor", 0.0) * intensity
                    )
                    # Ensure robust access with defaults
                    active_event_effects["downtime_prob_modifier"] += event_params.get("downtime_prob_modifier", 0.0) 
                    active_event_effects["downtime_mean_factor"] *= event_params.get("downtime_mean_factor", 1.0) # Default to 1.0
                    active_event_effects["fatigue_rate_modifier"] = max(
                        active_event_effects["fatigue_rate_modifier"], 
                        event_params.get("fatigue_rate_modifier", 1.0) 
                    )
                    if current_minute_of_shift == event_start_min and step < num_steps: 
                         wellbeing_triggers_dict['disruption'].append(step)
                
                elif event_type in ["Scheduled Break", "Short Pause"]:
                    active_event_effects["fatigue_recovery_factor"] = max(
                        active_event_effects["fatigue_recovery_factor"],
                        event_params.get("fatigue_recovery_factor", 0.0)
                    )
                    active_event_effects["wellbeing_boost_abs"] += event_params.get("wellbeing_boost_abs", 0.0)
                    productivity_multiplier_event = event_params.get("productivity_multiplier", 1.0) # Default to 1 if not specified
                    if event_def.get("Scope", "All") == "All":
                        for zn in active_event_effects["zone_productivity_multiplier"].keys():
                            active_event_effects["zone_productivity_multiplier"][zn] = min(active_event_effects["zone_productivity_multiplier"][zn], productivity_multiplier_event)
                
                elif event_type == "Team Meeting":
                    active_event_effects["cohesion_boost_abs"] += event_params.get("cohesion_boost_abs", 0.0)
                    active_event_effects["psych_safety_boost_abs"] += event_params.get("psych_safety_boost_abs",0.0)
                    affected_zones = event_def.get("Affected Zones", [])
                    prod_multiplier = event_params.get("productivity_multiplier", 0.1) # Default to low productivity
                    if "All" in affected_zones or not affected_zones:
                        for zn in active_event_effects["zone_productivity_multiplier"].keys(): active_event_effects["zone_productivity_multiplier"][zn] = min(active_event_effects["zone_productivity_multiplier"][zn], prod_multiplier)
                    else:
                        for zn in affected_zones: 
                            if zn in active_event_effects["zone_productivity_multiplier"]: active_event_effects["zone_productivity_multiplier"][zn] = min(active_event_effects["zone_productivity_multiplier"][zn], prod_multiplier)
                
                elif event_type == "Maintenance":
                    active_event_effects["downtime_prob_modifier"] += event_params.get("downtime_prob_modifier", 0.0)
                    # Robustly get downtime_mean_factor for maintenance as well
                    active_event_effects["downtime_mean_factor"] *= event_params.get("downtime_mean_factor", 1.0) 
                    affected_zones_maint = event_def.get("Affected Zones", [])
                    uptime_mult = event_params.get("specific_zone_uptime_multiplier", 0.1) 
                    if "All" in affected_zones_maint or not affected_zones_maint:
                         for zn in active_event_effects["zone_uptime_multiplier"].keys(): active_event_effects["zone_uptime_multiplier"][zn] = min(active_event_effects["zone_uptime_multiplier"][zn], uptime_mult)
                    else:
                        for zn in affected_zones_maint:
                            if zn in active_event_effects["zone_uptime_multiplier"]: active_event_effects["zone_uptime_multiplier"][zn] = min(active_event_effects["zone_uptime_multiplier"][zn], uptime_mult)
                
                elif event_type == "Custom Event": 
                    active_event_effects["wellbeing_drop_factor"] = max(
                        active_event_effects["wellbeing_drop_factor"],
                        event_params.get("wellbeing_drop_factor", 0.0) 
                    )
                    active_event_effects["fatigue_rate_modifier"] = max(
                        active_event_effects["fatigue_rate_modifier"],
                        event_params.get("fatigue_rate_modifier", 1.0) 
                    )
                    # Add robust access for other custom event params if they affect shared effects
                    active_event_effects["downtime_prob_modifier"] += event_params.get("downtime_prob_modifier", 0.0)
                    active_event_effects["downtime_mean_factor"] *= event_params.get("downtime_mean_factor", 1.0)


        # --- Task Backlog and Workload --- (rest of the loop continues)
        avg_fatigue_current_step = np.mean(worker_fatigue) if num_team_members > 0 else 0.0
        for zone_name, zone_details in work_areas_config.items():
            tasks_arriving = zone_details.get('tasks_per_interval', 0) * (1.0 + (0.1*random.random()-0.05) if is_any_disruption_active_this_step else 0.0)
            workers_in_this_zone_count = worker_assigned_zone.count(zone_name) if num_team_members > 0 else 0
            prev_compliance_score = _task_compliance_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'BASE_TASK_COMPLETION_PROB', 0.95) * 100.0
            compliance_factor_for_processing = prev_compliance_score / 100.0
            
            zone_prod_mult = active_event_effects["zone_productivity_multiplier"].get(zone_name, 1.0)

            zone_processing_capacity = workers_in_this_zone_count * \
                                     zone_details.get('base_productivity', 0.8) * \
                                     (1.0 - avg_fatigue_current_step * 0.3) * \
                                     compliance_factor_for_processing * \
                                     zone_prod_mult
            zone_task_backlog[zone_name] = max(0, zone_task_backlog.get(zone_name,0) + tasks_arriving - zone_processing_capacity)

        total_backlog_current_step = sum(zone_task_backlog.values())
        max_concurrent_total_facility = sum(zd.get('max_concurrent_tasks', zd.get('tasks_per_interval', 0) * 1.5) for zd in work_areas_config.values())
        if max_concurrent_total_facility == 0: max_concurrent_total_facility = EPSILON 
        workload_pressure_from_backlog_metric = total_backlog_current_step / (max_concurrent_total_facility + EPSILON)
        _perceived_workload_scores[step] = np.clip(workload_pressure_from_backlog_metric * 10.0 + (step / (num_steps + EPSILON)) * 1.5, 0, 10)

        # --- Fatigue ---
        fatigue_rate_this_step = _get_config_param(config, 'WELLBEING_FATIGUE_RATE_PER_INTERVAL', 0.0025)
        if team_initiative == "More frequent breaks": 
            fatigue_rate_this_step *= (1.0 - _get_config_param(config, 'INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.3))
        fatigue_rate_this_step *= (1.0 + _perceived_workload_scores[step] / 15.0) 
        fatigue_rate_this_step *= active_event_effects["fatigue_rate_modifier"] 
        if num_team_members > 0:
            worker_fatigue += fatigue_rate_this_step
            if active_event_effects["fatigue_recovery_factor"] > 0:
                worker_fatigue *= (1.0 - active_event_effects["fatigue_recovery_factor"])
            worker_fatigue = np.clip(worker_fatigue, 0.0, 1.0)
        avg_fatigue_current_step = np.mean(worker_fatigue) if num_team_members > 0 else 0.0

        # --- Wellbeing ---
        wb_now = _wellbeing_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'INITIAL_WELLBEING_MEAN', 0.8) * 100.0
        wb_now -= (avg_fatigue_current_step * 20.0 + (_perceived_workload_scores[step] - 5.0) * 1.0) 
        wb_now -= active_event_effects["wellbeing_drop_factor"] * 25.0  
        wb_now += active_event_effects["wellbeing_boost_abs"]
        leadership_support_factor = _get_config_param(config, 'LEADERSHIP_SUPPORT_FACTOR', 0.65)
        wb_now += (leadership_support_factor - 0.5) * 10.0 
        if team_initiative != "Increased Autonomy" and "Increased Autonomy Mode" not in active_event_types_this_step:
            wb_now -= _get_config_param(config, 'STRESS_FROM_LOW_CONTROL_FACTOR', 0.025) * 15.0
        prev_collab_score = _collaboration_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'TARGET_COLLABORATION', 60.0)
        if prev_collab_score < 50.0: 
            wb_now -= _get_config_param(config, 'ISOLATION_IMPACT_ON_WELLBEING', 0.15) * (50.0 - prev_collab_score) * 0.4
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,3) if num_steps > 2 else 1) == 0:
            wb_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS', 0.06) * 100.0
        if team_initiative == "Increased Autonomy": 
            wb_now += _get_config_param(config, 'INITIATIVE_AUTONOMY_WELLBEING_BOOST_ABS', 0.05) * 100.0
        _wellbeing_scores[step] = np.clip(wb_now, 5.0, 100.0)

        # --- Psych Safety, Cohesion ---
        ps_now = _psych_safety_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'PSYCH_SAFETY_BASELINE', 0.75) * 100.0
        ps_now -= _get_config_param(config, 'PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL', 0.0005) * 100.0
        if is_any_disruption_active_this_step: 
             ps_now -= _get_config_param(config, 'UNCERTAINTY_DURING_DISRUPTION_IMPACT_PSYCH_SAFETY', 0.15) * 15.0
        ps_now += (leadership_support_factor - 0.5) * 15.0
        ps_now += (_get_config_param(config, 'COMMUNICATION_EFFECTIVENESS_FACTOR', 0.75) - 0.5) * 10.0
        ps_now += active_event_effects["psych_safety_boost_abs"] 
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,3) if num_steps > 2 else 1) == 1: 
            ps_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS', 0.09) * 100.0 
        if team_initiative == "Increased Autonomy": 
            ps_now += _get_config_param(config, 'INITIATIVE_AUTONOMY_PSYCHSAFETY_BOOST_ABS', 0.08) * 100.0
        prev_team_cohesion_score = _team_cohesion_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'TEAM_COHESION_BASELINE', 0.7) * 100.0
        ps_now += (prev_team_cohesion_score - _get_config_param(config, 'TEAM_COHESION_BASELINE', 0.7) * 100.0) * _get_config_param(config, 'TEAM_COHESION_IMPACT_ON_PSYCH_SAFETY', 0.2)
        _psych_safety_scores[step] = np.clip(ps_now, 10.0, 100.0)

        cohesion_now = prev_team_cohesion_score
        cohesion_now -= (0.15 + (0.7 if is_any_disruption_active_this_step else 0.0) + (_perceived_workload_scores[step] / 50.0)) 
        cohesion_now += active_event_effects["cohesion_boost_abs"]
        if _psych_safety_scores[step] > 70.0: cohesion_now += 0.8
        if prev_collab_score > 60.0: cohesion_now += 0.6
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,3) if num_steps > 2 else 1) == 0:
            cohesion_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_COHESION_BOOST_ABS', 7.0) 
        _team_cohesion_scores[step] = np.clip(cohesion_now, 10.0, 100.0) 

        # --- Wellbeing Triggers ---
        if _wellbeing_scores[step] < _get_config_param(config, 'WELLBEING_ALERT_THRESHOLD', 60.0):
            wellbeing_triggers_dict['threshold'].append(step)
        if step > 0 and _wellbeing_scores[step] < _wellbeing_scores[step - 1] - 12.0: 
            wellbeing_triggers_dict['trend'].append(step)
        if is_any_disruption_active_this_step and _wellbeing_scores[step] < (_wellbeing_scores[max(0, step - 1)] if step > 0 else _wellbeing_scores[0]) * 0.90:
            if num_team_members > 0:
                affected_worker_indices = [i for i in range(num_team_members) if worker_fatigue[i] > 0.65 or random.random() < active_event_effects.get("compliance_reduction_factor", 0.0) * 0.2] 
                affected_zones_this_step = [worker_assigned_zone[i] for i in affected_worker_indices if i < len(worker_assigned_zone)]
                for zone_name_affected in set(affected_zones_this_step):
                    if zone_name_affected in wellbeing_triggers_dict['work_area']:
                        wellbeing_triggers_dict['work_area'][zone_name_affected].append(step)
            
        # --- Operational Metrics ---
        prev_uptime = _uptime_percent[max(0, step - 1)] if step > 0 else 100.0
        current_uptime = prev_uptime
        equipment_failed_this_step_flag = False
        if random.random() < _get_config_param(config, 'EQUIPMENT_FAILURE_PROB_PER_INTERVAL', 0.003): 
            current_uptime -= random.uniform(10, 30) 
            equipment_failed_this_step_flag = True
        
        general_uptime_multiplier = min(active_event_effects["zone_uptime_multiplier"].values()) if active_event_effects["zone_uptime_multiplier"] else 1.0
        current_uptime *= general_uptime_multiplier 

        if is_any_disruption_active_this_step and general_uptime_multiplier == 1.0: 
             current_uptime -= 20.0 * active_event_effects.get("compliance_reduction_factor", 0.0) 
        _uptime_percent[step] = np.clip(current_uptime, 10.0, 100.0) 
        
        base_compliance_val = _get_config_param(config, 'BASE_TASK_COMPLETION_PROB', 0.97) * 100.0
        compliance_now = base_compliance_val * (1.0 - avg_fatigue_current_step * _get_config_param(config, 'FATIGUE_IMPACT_ON_COMPLIANCE', 0.3))
        avg_worker_task_complexity = 0.5
        if num_team_members > 0 and worker_assigned_zone:
            complexities = [work_areas_config.get(worker_assigned_zone[w], {}).get('task_complexity', 0.5) for w in range(num_team_members) if w < len(worker_assigned_zone)]
            if complexities: avg_worker_task_complexity = np.mean(complexities)
        compliance_now -= avg_worker_task_complexity * _get_config_param(config, 'COMPLEXITY_IMPACT_ON_COMPLIANCE', 0.35) * 100.0
        compliance_now *= (_psych_safety_scores[step] / 100.0 * 0.15 + 0.85) 
        compliance_now *= (_get_config_param(config, 'COMMUNICATION_EFFECTIVENESS_FACTOR', 0.75) * 0.35 + 0.65) 
        min_compliance_during_disruption = _get_config_param(config, 'MIN_COMPLIANCE_DURING_DISRUPTION', 15.0)
        compliance_now = max(min_compliance_during_disruption, compliance_now * (1.0 - active_event_effects["compliance_reduction_factor"]))
        _task_compliance_scores[step] = np.clip(compliance_now, 0.0, 100.0)

        quality_now = (100.0 - _get_config_param(config, 'BASE_QUALITY_DEFECT_RATE', 0.015) * 100.0) * \
                      (_task_compliance_scores[step] / 100.0)**1.2 
        if is_any_disruption_active_this_step: quality_now -= 10.0 * active_event_effects.get("compliance_reduction_factor", 0.0) 
        _quality_rate_percent[step] = np.clip(quality_now, 25.0, 100.0) 

        max_potential_throughput_facility = _get_config_param(config, 'THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL', 120.0)
        effective_capacity_factor = (_uptime_percent[step] / 100.0) * \
                                    (_task_compliance_scores[step] / 100.0) * \
                                    (1.0 - avg_fatigue_current_step * 0.65) * \
                                    (1.0 - _perceived_workload_scores[step] / 25.0) 
        throughput_disruption_impact = 0.85 * active_event_effects.get("compliance_reduction_factor", 0.0) 
        actual_units_produced = max_potential_throughput_facility * effective_capacity_factor * (1.0 - throughput_disruption_impact)
        _throughput_percent_of_max[step] = np.clip((actual_units_produced / (max_potential_throughput_facility + EPSILON)) * 100.0, 0.0, 100.0)
        _task_completion_rate_percent[step] = _throughput_percent_of_max[step]

        # --- Downtime ---
        current_downtime_duration = 0.0; current_downtime_cause = "None"
        downtime_prob_mod = active_event_effects.get("downtime_prob_modifier", 0.0) # Get with default
        downtime_mean_fact = active_event_effects.get("downtime_mean_factor", 1.0)   # Get with default

        if downtime_prob_mod > 0 and random.random() < downtime_prob_mod:
            downtime_mean_event = _get_config_param(config, 'DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE', 7.0) * downtime_mean_fact
            downtime_std_event = _get_config_param(config, 'DOWNTIME_STD_MINUTES_PER_OCCURRENCE', 3.0) * math.sqrt(downtime_mean_fact)
            current_downtime_duration = max(0.0, np.random.normal(downtime_mean_event, downtime_std_event))
            
            active_downtime_event_types = [evt.get("Event Type") for evt in active_events_details_this_step
                                           if event_type_params_config.get(evt.get("Event Type",{}),{}).get("downtime_prob_modifier",0)>0 or \
                                              event_type_params_config.get(evt.get("Event Type",{}),{}).get("downtime_prob",0)>0 ]
            if active_downtime_event_types: current_downtime_cause = active_downtime_event_types[0] 
            else: current_downtime_cause = random.choice([c for c in downtime_causes_list if c not in ["Equipment Failure", "Human Error"]])
        
        if equipment_failed_this_step_flag and random.random() < _get_config_param(config, 'DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB', 0.75): 
             duration_equip_fail = _get_config_param(config, 'EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS', 4) * 2.0 
             current_downtime_duration = max(current_downtime_duration, duration_equip_fail) 
             current_downtime_cause = "Equipment Failure" if current_downtime_cause == "None" or current_downtime_duration == duration_equip_fail else f"{current_downtime_cause}, Equip.Fail"
        if (_task_compliance_scores[step] < 45 or avg_fatigue_current_step > 0.92) and random.random() < 0.035: 
            duration_human_error = max(0.0, np.random.normal(_get_config_param(config, 'DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',7.0)*0.35, _get_config_param(config, 'DOWNTIME_STD_MINUTES_PER_OCCURRENCE',3.0)*0.2))
            current_downtime_duration = max(current_downtime_duration, duration_human_error)
            current_downtime_cause = "Human Error" if current_downtime_cause == "None" or current_downtime_duration == duration_human_error else f"{current_downtime_cause}, HumanError"
        
        interval_actual_length_minutes = (shift_duration_minutes_sim / (num_steps if num_steps > 0 else 1.0)) 
        _downtime_events_per_interval[step] = {'duration': np.clip(current_downtime_duration, 0, interval_actual_length_minutes), 'cause': current_downtime_cause if current_downtime_duration > EPSILON else "None"}

        # --- Operational Recovery & Productivity Loss ---
        current_oee_calc = (_uptime_percent[step]/100) * (_throughput_percent_of_max[step]/100) * (_quality_rate_percent[step]/100)
        if not is_any_disruption_active_this_step: 
            prev_recovery = _operational_recovery_scores[max(0,step-1)]; target_potential_recovery = current_oee_calc * 100
            recovery_rate_factor = 1.0 - math.exp(-1.0 / (recovery_halflife_intervals + EPSILON))
            _operational_recovery_scores[step] = np.clip(prev_recovery + (target_potential_recovery - prev_recovery) * recovery_rate_factor, 0, 100)
        else: _operational_recovery_scores[step] = np.clip(current_oee_calc * 100, 0, 100)
        _productivity_loss_percent[step] = np.clip(100 - _operational_recovery_scores[step] + np.random.normal(0,0.5), 0, 100) 

        # --- Collaboration Score ---
        base_collab = 60 + (_team_cohesion_scores[step]-70)*0.5 - (_perceived_workload_scores[step]-5)*2.5 
        if is_any_disruption_active_this_step: base_collab -= 25.0 * active_event_effects.get("compliance_reduction_factor", 0.0) 
        _collaboration_scores[step] = np.clip(base_collab + np.random.normal(0,2) ,5, 95)

        # --- Worker Movement and Status ---
        if num_team_members > 0 : 
            for i in range(num_team_members):
                current_assigned_zone_for_worker_i = worker_assigned_zone[i]
                zone_details = work_areas_config.get(current_assigned_zone_for_worker_i, {}); zone_coords = zone_details.get('coords')
                target_x, target_y = facility_width/2, facility_height/2 
                if zone_coords and len(zone_coords) == 2: 
                    (zx0, zy0), (zx1, zy1) = zone_coords
                    target_x, target_y = random.uniform(min(zx0,zx1), max(zx0,zx1)), random.uniform(min(zy0,zy1), max(zy0,zy1)) 
                
                is_on_scheduled_break_or_meeting = False
                for active_event_detail in active_events_details_this_step: 
                    event_type_active = active_event_detail.get("Event Type")
                    if event_type_active in ["Scheduled Break", "Short Pause", "Team Meeting"]:
                        scope = active_event_detail.get("Scope", "All")
                        affected_zones_event = active_event_detail.get("Affected Zones", [])
                        if scope == "All" or current_assigned_zone_for_worker_i in affected_zones_event:
                            is_on_scheduled_break_or_meeting = True
                            if "Break Room" in work_areas_config and event_type_active != "Team Meeting": 
                                br_coords = work_areas_config["Break Room"].get('coords')
                                if br_coords and len(br_coords) == 2:
                                    target_x = random.uniform(min(br_coords[0][0],br_coords[1][0]), max(br_coords[0][0],br_coords[1][0]))
                                    target_y = random.uniform(min(br_coords[0][1],br_coords[1][1]), max(br_coords[0][1],br_coords[1][1]))
                            elif event_type_active == "Team Meeting" and "Break Room" in affected_zones_event and "Break Room" in work_areas_config: 
                                br_coords = work_areas_config["Break Room"].get('coords')
                                if br_coords and len(br_coords) == 2:
                                    target_x = random.uniform(min(br_coords[0][0],br_coords[1][0]), max(br_coords[0][0],br_coords[1][0]))
                                    target_y = random.uniform(min(br_coords[0][1],br_coords[1][1]), max(br_coords[0][1],br_coords[1][1]))
                            break 
                
                move_x = (target_x - worker_current_x[i]) * 0.3 + np.random.normal(0, 1.0); 
                move_y = (target_y - worker_current_y[i]) * 0.3 + np.random.normal(0, 1.0)
                worker_current_x[i] = np.clip(worker_current_x[i] + move_x, 0, facility_width); worker_current_y[i] = np.clip(worker_current_y[i] + move_y, 0, facility_height)
                
                status_now = 'working'
                if is_on_scheduled_break_or_meeting: status_now = 'break' 
                elif worker_fatigue[i] > 0.85: status_now = 'exhausted' 
                elif worker_fatigue[i] > 0.65: status_now = 'fatigued'
                elif is_any_disruption_active_this_step and random.random() < 0.3: status_now = 'disrupted' 
                
                team_positions_data.append({'step': step, 'worker_id': i, 'x': worker_current_x[i], 'y': worker_current_y[i], 'z': np.random.uniform(0, 0.1), 'zone': current_assigned_zone_for_worker_i, 'status': status_now})
    
    # --- Prepare Output Data ---
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
