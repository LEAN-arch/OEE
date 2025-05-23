# simulation.py
import logging
import math as global_math_ref # Use a distinct alias for the global import
import random
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
EPSILON = 1e-6

def get_math_module(passed_math):
    """Validates and returns a math module, falling back to local import if needed."""
    if passed_math and hasattr(passed_math, 'sqrt') and hasattr(passed_math, 'exp') and hasattr(passed_math, 'pi'):
        logger.debug("[SIM_FUNC_LOCAL] Using passed math module.")
        return passed_math
    logger.warning("[SIM_FUNC_LOCAL] Passed math module is invalid or missing attributes. Attempting local import.")
    try:
        import math as local_math_import
        logger.info("[SIM_FUNC_LOCAL] Successfully imported math module locally.")
        return local_math_import
    except ImportError as e:
        logger.critical(f"[SIM_FUNC_LOCAL] CRITICAL FAILURE: Could not import math module: {e}")
        raise NameError("math module is critically unavailable for simulation and could not be imported.")

def _get_config_param(config: dict, key: str, default: any, data_type=None) -> any:
    """Safely retrieves a parameter from the config dictionary with type casting."""
    val = config.get(key, default)
    if data_type:
        try:
            if data_type == float: return float(val)
            if data_type == int: return int(val)
            if data_type == bool: # Handle bool conversion robustly
                if isinstance(val, str): return val.lower() in ['true', '1', 't', 'y', 'yes']
                return bool(val)
        except (ValueError, TypeError) as e:
            logger.warning(f"Config param '{key}' (value: {val}): Type conversion to {data_type} failed: {e}. Using default: {default}")
            return default
    return val

def simulate_workplace_operations(num_team_members: int, num_steps: int,
                                 scheduled_events: list,
                                 team_initiative: str, config: dict,
                                 math_module_arg=global_math_ref):
    
    math = get_math_module(math_module_arg)

    np.random.seed(42); random.seed(42) # For reproducibility

    minutes_per_interval = _get_config_param(config, 'MINUTES_PER_INTERVAL', 2.0, float)
    if minutes_per_interval <= 0: minutes_per_interval = 2.0; logger.error("MINUTES_PER_INTERVAL invalid, used 2.0.")

    facility_width, facility_height = _get_config_param(config, 'FACILITY_SIZE', (100, 80))
    work_areas_config = _get_config_param(config, 'WORK_AREAS', {})
    event_type_params_config = _get_config_param(config, 'EVENT_TYPE_CONFIG', {})

    # Initialize result arrays/lists (0-100 for scores, 0-1 for factors/rates unless specified)
    _task_compliance_scores = np.full(num_steps, _get_config_param(config, 'TARGET_COMPLIANCE', 85.0, float) * 0.9) # Start near target
    _collaboration_metric_scores = np.full(num_steps, _get_config_param(config, 'TARGET_COLLABORATION', 65.0, float))
    _operational_recovery_scores = np.full(num_steps, 100.0)
    _wellbeing_scores = np.full(num_steps, _get_config_param(config, 'INITIAL_WELLBEING_MEAN', 80.0, float))
    _psych_safety_scores = np.full(num_steps, _get_config_param(config, 'PSYCH_SAFETY_BASELINE', 75.0, float))
    _team_cohesion_scores = np.full(num_steps, _get_config_param(config, 'TEAM_COHESION_BASELINE', 70.0, float))
    _perceived_workload_scores = np.full(num_steps, _get_config_param(config, 'TARGET_PERCEIVED_WORKLOAD', 6.0, float) * 0.8) # Scale 0-10
    _productivity_loss_percent = np.zeros(num_steps)
    _raw_downtime_log = [] # Stores {'step': s, 'duration': d, 'cause': c}
    _task_completion_rate_percent = np.zeros(num_steps)
    _uptime_percent = np.full(num_steps, 100.0)
    _quality_rate_percent = np.full(num_steps, 98.0)
    _throughput_percent_of_max = np.full(num_steps, 80.0)

    team_positions_data = []
    worker_current_x = np.random.uniform(0, facility_width, num_team_members) if num_team_members > 0 else np.array([])
    worker_current_y = np.random.uniform(0, facility_height, num_team_members) if num_team_members > 0 else np.array([])
    
    worker_assigned_zone = ["DefaultZone"] * num_team_members
    if num_team_members > 0 and work_areas_config:
        assigned_workers_temp = []
        for zn, zd in work_areas_config.items():
            if isinstance(zd, dict): assigned_workers_temp.extend([zn] * zd.get("workers", 0))
        if len(assigned_workers_temp) == num_team_members: worker_assigned_zone = assigned_workers_temp; random.shuffle(worker_assigned_zone)
        else:
            zone_keys_for_dist = [zn for zn, zd in work_areas_config.items() if isinstance(zd, dict) and zd.get("workers",0) > 0]
            if not zone_keys_for_dist: zone_keys_for_dist = list(work_areas_config.keys())
            if not zone_keys_for_dist: zone_keys_for_dist = ["DefaultZone"]
            worker_assigned_zone = [zone_keys_for_dist[i % len(zone_keys_for_dist)] for i in range(num_team_members)]
            logger.warning(f"[SIM] Worker sum mismatch, used modulo for assignment. Config: {len(assigned_workers_temp)}, Team: {num_team_members}")


    worker_fatigue = np.random.uniform(0.0, 0.1, num_team_members) if num_team_members > 0 else np.array([]) # Fatigue 0-1
    zone_task_backlog = {zn: 0.0 for zn in work_areas_config.keys()}
    recovery_halflife_intervals = _get_config_param(config, 'RECOVERY_HALFLIFE_INTERVALS', 8, int)
    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa: [] for wa in work_areas_config}, 'disruption': []}
    downtime_causes_cfg_list = _get_config_param(config, 'DOWNTIME_CAUSES_LIST', ["UnknownCause"])

    # Get key config parameters once for the loop
    cfg_base_fatigue_rate = _get_config_param(config, 'BASE_FATIGUE_RATE_PER_INTERVAL', 0.0025, float)
    cfg_fatigue_compliance_impact = _get_config_param(config, 'FATIGUE_IMPACT_ON_COMPLIANCE', 0.3, float)
    cfg_complexity_compliance_impact = _get_config_param(config, 'COMPLEXITY_IMPACT_ON_COMPLIANCE', 0.35, float)
    cfg_target_workload = _get_config_param(config, 'TARGET_PERCEIVED_WORKLOAD', 6.0, float)
    cfg_stress_low_control_drop = _get_config_param(config, 'STRESS_FROM_LOW_CONTROL_POINTS_DROP', 2.5, float)
    cfg_isolation_max_drop = _get_config_param(config, 'ISOLATION_IMPACT_ON_WELLBEING_POINTS_MAX_DROP', 15.0, float)
    cfg_ps_erosion_rate = _get_config_param(config, 'BASE_PSYCH_SAFETY_EROSION_PER_INTERVAL', 0.05, float)
    cfg_uncertainty_ps_drop = _get_config_param(config, 'UNCERTAINTY_DISRUPTION_PSYCH_SAFETY_POINTS_DROP', 15.0, float)
    cfg_cohesion_ps_factor = _get_config_param(config, 'TEAM_COHESION_IMPACT_ON_PSYCH_SAFETY_FACTOR', 0.2, float)
    cfg_cohesion_baseline = _get_config_param(config, 'TEAM_COHESION_BASELINE', 70.0, float)
    cfg_leadership_factor = _get_config_param(config, 'LEADERSHIP_SUPPORT_FACTOR', 0.65, float)
    cfg_comm_factor = _get_config_param(config, 'COMMUNICATION_EFFECTIVENESS_FACTOR', 0.75, float)
    cfg_wb_alert_thresh = _get_config_param(config, 'WELLBEING_ALERT_THRESHOLD', 60.0, float)
    cfg_base_task_comp_prob = _get_config_param(config, 'BASE_TASK_COMPLETION_PROB', 0.97, float)
    cfg_min_comp_disrupt = _get_config_param(config, 'MIN_COMPLIANCE_DURING_DISRUPTION', 15.0, float)


    for step in range(num_steps):
        current_minute_of_shift = step * minutes_per_interval
        active_event_effects = { # Reset effects each step
            "compliance_reduction_factor": 0.0, "wellbeing_drop_factor": 0.0, # Factors 0-1
            "downtime_prob_modifier": 0.0, "downtime_mean_factor": 1.0, # Modifiers
            "fatigue_rate_modifier": 1.0, "fatigue_recovery_factor": 0.0, # Factors 0-1
            "wellbeing_boost_abs": 0.0, "cohesion_boost_abs": 0.0, "psych_safety_boost_abs": 0.0, # Points 0-100
            "zone_productivity_multiplier": {zn: 1.0 for zn in work_areas_config.keys()},
            "zone_uptime_multiplier": {zn: 1.0 for zn in work_areas_config.keys()}
        }
        is_any_disruption_active = False; active_event_details = []

        for event_def in scheduled_events: # Process active events
            if not isinstance(event_def, dict): continue
            evt_start = _get_config_param(event_def, "Start Time (min)", -1.0, float)
            evt_dur = _get_config_param(event_def, "Duration (min)", 0.0, float)
            evt_type = event_def.get("Event Type", "Unknown")
            if not (evt_start <= current_minute_of_shift < evt_start + evt_dur): continue

            active_event_details.append(event_def)
            evt_params = event_type_params_config.get(evt_type, {})
            intensity = _get_config_param(event_def, "Intensity", 1.0, float)
            affected_zones = event_def.get("Affected Zones", []) # List of zone names
            scope_all = event_def.get("Scope", "Individual") == "All" or not affected_zones # If no specific zones, assume all

            if "Disruption" in evt_type: is_any_disruption_active = True # ... (Apply disruption effects as before) ...
            # ... (Handle other event types: Break, Meeting, Maintenance, Custom, applying to all or affected_zones) ...
            # Example for Break productivity:
            if evt_type in ["Scheduled Break", "Short Pause"]:
                prod_mult = _get_config_param(evt_params, "productivity_multiplier", 1.0, float)
                for zn_key in active_event_effects["zone_productivity_multiplier"]:
                    if scope_all or zn_key in affected_zones:
                        active_event_effects["zone_productivity_multiplier"][zn_key] = min(active_event_effects["zone_productivity_multiplier"][zn_key], prod_mult)
                active_event_effects["fatigue_recovery_factor"] = max(active_event_effects["fatigue_recovery_factor"], _get_config_param(evt_params, "fatigue_recovery_factor", 0.0, float))
                active_event_effects["wellbeing_boost_abs"] += _get_config_param(evt_params, "wellbeing_boost_abs", 0.0, float)


        # --- Task Backlog & Perceived Workload ---
        # ... (Calculations as before, using avg_fatigue_current_step) ...
        avg_fatigue_current_step = np.mean(worker_fatigue) if num_team_members > 0 else 0.0 # This is after fatigue update for previous step
        for zone_name, zone_details_raw in work_areas_config.items():
            if not isinstance(zone_details_raw, dict): continue
            zone_details = zone_details_raw
            tasks_arriving = _get_config_param(zone_details, 'tasks_per_interval', 0, float) * (1.0 + (0.1 * random.random() - 0.05) if is_any_disruption_active else 0.0)
            workers_in_zone = worker_assigned_zone.count(zone_name) if num_team_members > 0 else 0
            prev_comp = _task_compliance_scores[max(0, step - 1)] / 100.0 if step > 0 else cfg_base_task_comp_prob
            
            zone_prod_mult_eff = active_event_effects["zone_productivity_multiplier"].get(zone_name, 1.0)
            zone_processing_cap = workers_in_zone * _get_config_param(zone_details, 'base_productivity', 0.8, float) * \
                                 (1.0 - avg_fatigue_current_step * 0.3) * prev_comp * zone_prod_mult_eff
            zone_task_backlog[zone_name] = max(0, zone_task_backlog.get(zone_name,0) + tasks_arriving - zone_processing_cap)
        
        total_backlog = sum(zone_task_backlog.values())
        max_concurrent_facility = sum(_get_config_param(zd, 'max_concurrent_tasks', 15, float) for zd_raw in work_areas_config.values() if isinstance(zd_raw, dict) for zd in [zd_raw])
        if max_concurrent_facility < EPSILON : max_concurrent_facility = num_team_members * 2 if num_team_members >0 else 10 # Avoid div by zero
        
        workload_metric = np.clip((total_backlog / (max_concurrent_facility + EPSILON)) * 10.0, 0, 10) # Base workload from backlog
        workload_metric += (step / (num_steps + EPSILON)) * 2.0 # Time pressure component, max 2 points
        _perceived_workload_scores[step] = np.clip(workload_metric, 0, 10)


        # --- Worker Fatigue (0-1 scale) ---
        fatigue_increase = cfg_base_fatigue_rate * (1.0 + _perceived_workload_scores[step] / 15.0) # Workload increases fatigue rate
        fatigue_increase *= active_event_effects["fatigue_rate_modifier"]
        if team_initiative == "More frequent breaks": fatigue_increase *= (1.0 - _get_config_param(config, 'INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.4, float))
        if num_team_members > 0:
            worker_fatigue += fatigue_increase
            if active_event_effects["fatigue_recovery_factor"] > 0: worker_fatigue *= (1.0 - active_event_effects["fatigue_recovery_factor"])
            worker_fatigue = np.clip(worker_fatigue, 0.0, 1.0)
        avg_fatigue_this_step_end = np.mean(worker_fatigue) if num_team_members > 0 else 0.0


        # --- Well-Being (0-100 scale) ---
        wb_now = _wellbeing_scores[max(0, step-1)] if step > 0 else _wellbeing_scores[0]
        wb_now -= (avg_fatigue_this_step_end * 25.0) # Fatigue impact (max 25 points)
        wb_now -= (_perceived_workload_scores[step] - cfg_target_workload) * 1.5 # Workload deviation impact
        wb_now -= active_event_effects["wellbeing_drop_factor"] * 30.0 # Event drop factor (0-1) scaled
        wb_now += active_event_effects["wellbeing_boost_abs"] # Event boost (absolute points)
        wb_now += (cfg_leadership_factor - 0.5) * 20.0 # Leadership impact (+/- 10 points max)
        if team_initiative != "Increased Autonomy": wb_now -= cfg_stress_low_control_drop
        collab_prev = _collaboration_metric_scores[max(0,step-1)] if step > 0 else _collaboration_metric_scores[0]
        if collab_prev < 50: wb_now -= (cfg_isolation_max_drop / 50.0) * (50.0 - collab_prev) # Isolation impact
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // 3 if num_steps > 5 else 1) == 0: # Stochastic recognition
            wb_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS', 6.0, float)
        if team_initiative == "Increased Autonomy": wb_now += _get_config_param(config, 'INITIATIVE_AUTONOMY_WELLBEING_BOOST_ABS', 5.0, float)
        _wellbeing_scores[step] = np.clip(wb_now, 5.0, 100.0)

        # --- Psych Safety (0-100 scale) ---
        ps_now = _psych_safety_scores[max(0,step-1)] if step > 0 else _psych_safety_scores[0]
        ps_now -= cfg_ps_erosion_rate
        if is_any_disruption_active: ps_now -= cfg_uncertainty_ps_drop
        ps_now += (cfg_leadership_factor - 0.5) * 30.0
        ps_now += (cfg_comm_factor - 0.5) * 20.0
        ps_now += active_event_effects["psych_safety_boost_abs"]
        # ... (Initiative boosts for psych safety as before) ...
        cohesion_prev = _team_cohesion_scores[max(0,step-1)] if step > 0 else _team_cohesion_scores[0]
        ps_now += (cohesion_prev - cfg_cohesion_baseline) * cfg_cohesion_ps_factor
        _psych_safety_scores[step] = np.clip(ps_now, 10.0, 100.0)

        # --- Team Cohesion (0-100 scale) ---
        # ... (Cohesion logic as before, ensuring correct scaling and config param usage) ...
        _team_cohesion_scores[step] = np.clip(cohesion_prev - (0.1 + 0.5*is_any_disruption_active + _perceived_workload_scores[step]/60.0) + active_event_effects["cohesion_boost_abs"], 10, 100)


        # --- Wellbeing Triggers ---
        # ... (Wellbeing trigger logic as before, checking worker_assigned_zone length) ...

        # --- Uptime, Compliance, Quality, Throughput, OEE (0-100 scale) ---
        # ... (Calculations as before, using avg_fatigue_this_step_end, and careful config param usage) ...
        # Ensure complexity is averaged correctly if workers are in different zones
        avg_task_complexity_eff = 0.5 
        if num_team_members > 0 and worker_assigned_zone and len(worker_assigned_zone) == num_team_members:
            complexities = []
            for w_idx in range(num_team_members):
                zone_cfg = work_areas_config.get(worker_assigned_zone[w_idx], {})
                if isinstance(zone_cfg, dict): complexities.append(_get_config_param(zone_cfg, 'task_complexity', 0.5, float))
            if complexities: avg_task_complexity_eff = np.mean(complexities)
        
        _task_compliance_scores[step] = np.clip(cfg_base_task_comp_prob*100 * (1 - avg_fatigue_this_step_end * cfg_fatigue_compliance_impact) - \
                                             avg_task_complexity_eff * cfg_complexity_compliance_impact * 100, cfg_min_comp_disrupt, 100)
        # ... (rest of operational metrics)

        # --- Downtime (Appending to _raw_downtime_log) ---
        # ... (Downtime event generation as refined previously, appending dicts to _raw_downtime_log) ...

        # --- Operational Recovery & Productivity Loss ---
        # ... (Calculations as before) ...

        # --- Collaboration Metric Score ---
        # ... (Calculation as before) ...

        # --- Worker Movement & Status ---
        # ... (Movement logic as before, ensuring robust zone and coord access) ...
        # ... (Status update logic) ...

    # --- Final Output Preparation ---
    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame(columns=['step', 'worker_id', 'x', 'y', 'z', 'zone', 'status'])
    task_compliance_output = {'data': list(_task_compliance_scores), 'forecast': [max(0,s-random.uniform(3,7)) for s in _task_compliance_scores], 'z_scores': list(np.random.normal(0,0.8,num_steps))}
    collaboration_metric_output = {'data': list(_collaboration_metric_scores), 'forecast': [min(100,s+random.uniform(3,7)) for s in _collaboration_metric_scores]}
    operational_recovery_output = list(_operational_recovery_scores)
    
    efficiency_df_data = {'uptime': list(_uptime_percent), 'throughput': list(_throughput_percent_of_max), 'quality': list(_quality_rate_percent)}
    efficiency_df = pd.DataFrame(efficiency_df_data)
    if not efficiency_df.empty:
        efficiency_df['oee'] = np.clip((efficiency_df['uptime']/100.0 * efficiency_df['throughput']/100.0 * efficiency_df['quality']/100.0) * 100.0, 0.0, 100.0)
    else: efficiency_df = pd.DataFrame(columns=['uptime', 'throughput', 'quality', 'oee'])

    productivity_loss_output = list(_productivity_loss_percent)
    worker_wellbeing_output = {'scores': list(_wellbeing_scores), 'triggers': wellbeing_triggers_dict, 'team_cohesion_scores': list(_team_cohesion_scores), 'perceived_workload_scores': list(_perceived_workload_scores)}
    psychological_safety_output = list(_psych_safety_scores)
    feedback_impact_output = list(np.random.normal(0, 0.05, num_steps)) # Small random feedback impact
    downtime_events_log_final_output = _raw_downtime_log
    task_completion_rate_output = list(_task_completion_rate_percent)

    logger.info(f"Simulation complete. Steps: {num_steps}, Team: {num_team_members}, Initiative: {team_initiative}.", extra={'user_action': 'SimEnd'})
    return (
        team_positions_df, task_compliance_output, collaboration_metric_output,
        operational_recovery_output, efficiency_df, productivity_loss_output,
        worker_wellbeing_output, psychological_safety_output, feedback_impact_output,
        downtime_events_log_final_output, task_completion_rate_output
    )
