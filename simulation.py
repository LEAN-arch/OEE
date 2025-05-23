# simulation.py
import logging
import math as global_math_ref # Use a distinct alias for the global import
import random
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
EPSILON = 1e-6 # For float comparisons

def get_math_module(passed_math):
    """Validates and returns a math module, falling back to local import if needed."""
    if passed_math and hasattr(passed_math, 'sqrt') and hasattr(passed_math, 'exp') and hasattr(passed_math, 'pi'):
        # logger.debug("[SIM_FUNC_LOCAL] Using passed math module.")
        return passed_math
    logger.warning("[SIM_FUNC_LOCAL] Passed math module is invalid or missing attributes. Attempting local import.")
    try:
        import math as local_math_import # Renamed to avoid conflict with function argument
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
            # Log only if the value being converted is not already the default
            if val != default:
                logger.warning(f"Config param '{key}' (value: {val}): Type conversion to {data_type} failed: {e}. Using default: {default}")
            return default
    return val

def simulate_workplace_operations(num_team_members: int, num_steps: int,
                                 scheduled_events: list,
                                 team_initiative: str, config: dict,
                                 math_module_arg=global_math_ref):
    
    math = get_math_module(math_module_arg) # Use the validated math module

    np.random.seed(42); random.seed(42)

    minutes_per_interval = _get_config_param(config, 'MINUTES_PER_INTERVAL', 2.0, float)
    if minutes_per_interval <= 0: minutes_per_interval = 2.0; logger.error("MINUTES_PER_INTERVAL invalid, used 2.0.")

    facility_width, facility_height = _get_config_param(config, 'FACILITY_SIZE', (100, 80))
    work_areas_config = _get_config_param(config, 'WORK_AREAS', {})
    event_type_params_config = _get_config_param(config, 'EVENT_TYPE_CONFIG', {})

    # Initialize result arrays
    _task_compliance_scores = np.full(num_steps, _get_config_param(config, 'TARGET_COMPLIANCE', 85.0, float) * 0.95) # Start high
    _collaboration_metric_scores = np.full(num_steps, _get_config_param(config, 'TARGET_COLLABORATION', 65.0, float))
    _operational_recovery_scores = np.full(num_steps, 100.0)
    _wellbeing_scores = np.full(num_steps, _get_config_param(config, 'INITIAL_WELLBEING_MEAN', 80.0, float))
    _psych_safety_scores = np.full(num_steps, _get_config_param(config, 'PSYCH_SAFETY_BASELINE', 75.0, float))
    _team_cohesion_scores = np.full(num_steps, _get_config_param(config, 'TEAM_COHESION_BASELINE', 70.0, float))
    _perceived_workload_scores = np.full(num_steps, _get_config_param(config, 'TARGET_PERCEIVED_WORKLOAD', 6.0, float) * 0.8)
    _productivity_loss_percent = np.zeros(num_steps)
    _raw_downtime_log = []
    _task_completion_rate_percent = np.zeros(num_steps)
    _uptime_percent = np.full(num_steps, 100.0)
    _quality_rate_percent = np.full(num_steps, 98.5) # Start with very high quality
    _throughput_percent_of_max = np.full(num_steps, 85.0) # Start with reasonable throughput

    team_positions_data = []
    worker_current_x = np.random.uniform(0, facility_width, num_team_members) if num_team_members > 0 else np.array([])
    worker_current_y = np.random.uniform(0, facility_height, num_team_members) if num_team_members > 0 else np.array([])
    
    worker_assigned_zone = ["DefaultZone"] * num_team_members
    if num_team_members > 0 and work_areas_config:
        assigned_workers_temp = []
        for zn, zd in work_areas_config.items():
            if isinstance(zd, dict): assigned_workers_temp.extend([zn] * _get_config_param(zd, "workers", 0, int)) # Use _get_config_param
        
        if len(assigned_workers_temp) == num_team_members: 
            worker_assigned_zone = assigned_workers_temp
            random.shuffle(worker_assigned_zone)
        else:
            logger.warning(f"[SIM] Worker sum in config ({len(assigned_workers_temp)}) != team size ({num_team_members}). Main.py's redistribution should handle this; if not, using modulo fallback.")
            zone_keys_for_dist = [zn for zn, zd in work_areas_config.items() if isinstance(zd, dict) and _get_config_param(zd, "workers",0,int) > 0 and not _get_config_param(zd,"is_rest_area",False,bool)]
            if not zone_keys_for_dist: zone_keys_for_dist = [zn for zn in work_areas_config.keys() if isinstance(work_areas_config[zn], dict) and not _get_config_param(work_areas_config[zn],"is_rest_area",False,bool)] # All non-rest areas
            if not zone_keys_for_dist: zone_keys_for_dist = list(work_areas_config.keys()) # All areas if no non-rest
            if not zone_keys_for_dist: zone_keys_for_dist = ["DefaultZone"] # Absolute fallback
            worker_assigned_zone = [zone_keys_for_dist[i % len(zone_keys_for_dist)] for i in range(num_team_members)]

    worker_fatigue = np.random.uniform(0.0, 0.1, num_team_members) if num_team_members > 0 else np.array([]) # Fatigue 0-1
    zone_task_backlog = {zn: 0.0 for zn in work_areas_config.keys()}
    recovery_halflife_intervals = _get_config_param(config, 'RECOVERY_HALFLIFE_INTERVALS', 8, int)
    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa: [] for wa in work_areas_config}, 'disruption': []}
    downtime_causes_cfg_list = _get_config_param(config, 'DOWNTIME_CAUSES_LIST', ["UnknownCause"])

    # Get key config parameters once for the loop efficiency
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
    cfg_equip_fail_prob = _get_config_param(config, 'EQUIPMENT_FAILURE_PROB_PER_INTERVAL', 0.003, float)
    cfg_dt_from_equip_fail_prob = _get_config_param(config, 'DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB', 0.75, float)
    cfg_equip_dt_intervals = _get_config_param(config, 'EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS', 4, int)
    cfg_downtime_mean_min = _get_config_param(config, 'DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE', 7.0, float)
    cfg_downtime_std_min = _get_config_param(config, 'DOWNTIME_STD_MINUTES_PER_OCCURRENCE', 3.0, float)

    for step in range(num_steps):
        current_minute_of_shift = step * minutes_per_interval
        active_event_effects = {
            "compliance_reduction_factor": 0.0, "wellbeing_drop_factor": 0.0,
            "downtime_prob_modifier": 0.0, "downtime_mean_factor": 1.0,
            "fatigue_rate_modifier": 1.0, "fatigue_recovery_factor": 0.0,
            "wellbeing_boost_abs": 0.0, "cohesion_boost_abs": 0.0, "psych_safety_boost_abs": 0.0,
            "zone_productivity_multiplier": {zn: 1.0 for zn in work_areas_config.keys()},
            "zone_uptime_multiplier": {zn: 1.0 for zn in work_areas_config.keys()}
        }
        is_any_disruption_active = False; active_event_details = []

        for event_def in scheduled_events:
            if not isinstance(event_def, dict): continue
            evt_start = _get_config_param(event_def, "Start Time (min)", -1.0, float)
            evt_dur = _get_config_param(event_def, "Duration (min)", 0.0, float)
            evt_type = event_def.get("Event Type", "Unknown")
            if not (evt_start <= current_minute_of_shift < evt_start + evt_dur): continue

            active_event_details.append(event_def)
            evt_params = event_type_params_config.get(evt_type, {})
            intensity = _get_config_param(event_def, "Intensity", 1.0, float)
            affected_zones = event_def.get("Affected Zones", [])
            scope_all = event_def.get("Scope", "Individual") == "All" or not affected_zones

            if "Disruption" in evt_type:
                is_any_disruption_active = True
                active_event_effects["compliance_reduction_factor"] = max(active_event_effects["compliance_reduction_factor"], _get_config_param(evt_params, "compliance_reduction_factor", 0.0, float) * intensity)
                active_event_effects["wellbeing_drop_factor"] = max(active_event_effects["wellbeing_drop_factor"], _get_config_param(evt_params, "wellbeing_drop_factor", 0.0, float) * intensity)
                active_event_effects["downtime_prob_modifier"] += _get_config_param(evt_params, "downtime_prob_modifier", 0.0, float) * intensity
                active_event_effects["downtime_mean_factor"] *= (1 + (_get_config_param(evt_params, "downtime_mean_factor", 1.0, float) -1) * intensity)
                active_event_effects["fatigue_rate_modifier"] = max(active_event_effects["fatigue_rate_modifier"], 1 + (_get_config_param(evt_params, "fatigue_rate_modifier", 1.0, float)-1)*intensity)
                if current_minute_of_shift >= evt_start and current_minute_of_shift < evt_start + minutes_per_interval:
                     if step not in wellbeing_triggers_dict['disruption']: wellbeing_triggers_dict['disruption'].append(step)
            elif evt_type in ["Scheduled Break", "Short Pause"]:
                active_event_effects["fatigue_recovery_factor"] = max(active_event_effects["fatigue_recovery_factor"], _get_config_param(evt_params, "fatigue_recovery_factor", 0.0, float))
                active_event_effects["wellbeing_boost_abs"] += _get_config_param(evt_params, "wellbeing_boost_abs", 0.0, float)
                prod_mult = _get_config_param(evt_params, "productivity_multiplier", 1.0, float)
                for zn_key in active_event_effects["zone_productivity_multiplier"]:
                    if scope_all or zn_key in affected_zones: active_event_effects["zone_productivity_multiplier"][zn_key] = min(active_event_effects["zone_productivity_multiplier"][zn_key], prod_mult)
            elif evt_type == "Team Meeting":
                 active_event_effects["cohesion_boost_abs"] += _get_config_param(evt_params, "cohesion_boost_abs", 0.0, float)
                 active_event_effects["psych_safety_boost_abs"] += _get_config_param(evt_params, "psych_safety_boost_abs", 0.0, float)
                 prod_mult = _get_config_param(evt_params, "productivity_multiplier", 0.1, float) # Low productivity during meetings
                 for zn_key in active_event_effects["zone_productivity_multiplier"]:
                    if scope_all or zn_key in affected_zones: active_event_effects["zone_productivity_multiplier"][zn_key] = min(active_event_effects["zone_productivity_multiplier"][zn_key], prod_mult)
            elif evt_type == "Maintenance":
                active_event_effects["downtime_prob_modifier"] += _get_config_param(evt_params, "downtime_prob_modifier", 0.0, float) * intensity
                active_event_effects["downtime_mean_factor"] *= (1 + (_get_config_param(evt_params, "downtime_mean_factor", 1.0, float) -1) * intensity)
                uptime_mult = _get_config_param(evt_params, "specific_zone_uptime_multiplier", 0.1, float)
                for zn_key in active_event_effects["zone_uptime_multiplier"]:
                    if scope_all or zn_key in affected_zones: active_event_effects["zone_uptime_multiplier"][zn_key] = min(active_event_effects["zone_uptime_multiplier"][zn_key], uptime_mult)
            elif evt_type == "Custom Event":
                active_event_effects["wellbeing_drop_factor"] = max(active_event_effects["wellbeing_drop_factor"], _get_config_param(evt_params, "wellbeing_drop_factor", 0.0, float) * intensity)
                active_event_effects["fatigue_rate_modifier"] = max(active_event_effects["fatigue_rate_modifier"], 1 + (_get_config_param(evt_params, "fatigue_rate_modifier", 1.0, float)-1)*intensity)
                active_event_effects["downtime_prob_modifier"] += _get_config_param(evt_params, "downtime_prob_modifier", 0.0, float) * intensity
                active_event_effects["downtime_mean_factor"] *= (1 + (_get_config_param(evt_params, "downtime_mean_factor", 1.0, float) -1) * intensity)

        # --- Task Backlog & Perceived Workload ---
        avg_fatigue_prev_step_end = np.mean(worker_fatigue) if num_team_members > 0 else 0.0 # Use fatigue from end of *previous* step for capacity calc
        for zone_name, zone_details_raw in work_areas_config.items():
            if not isinstance(zone_details_raw, dict): continue
            zone_details = zone_details_raw
            tasks_arriving = _get_config_param(zone_details, 'tasks_per_interval', 0, float) * (1.0 + (0.05 * random.uniform(-1,1)) if is_any_disruption_active else 0.0) # Slight variation during disruption
            workers_in_zone = worker_assigned_zone.count(zone_name) if num_team_members > 0 else 0
            prev_comp_factor = _task_compliance_scores[max(0, step - 1)] / 100.0 if step > 0 else cfg_base_task_comp_prob
            
            zone_prod_mult_eff = active_event_effects["zone_productivity_multiplier"].get(zone_name, 1.0)
            zone_processing_cap = workers_in_zone * _get_config_param(zone_details, 'base_productivity', 0.8, float) * \
                                 (1.0 - avg_fatigue_prev_step_end * cfg_fatigue_compliance_impact) * prev_comp_factor * zone_prod_mult_eff # Use fatigue impact here
            zone_task_backlog[zone_name] = max(0, zone_task_backlog.get(zone_name,0) + tasks_arriving - zone_processing_cap)
        
        total_backlog = sum(zone_task_backlog.values())
        max_concurrent_facility = sum(_get_config_param(zd, 'max_concurrent_tasks', 15, float) for zd_raw in work_areas_config.values() if isinstance(zd_raw, dict) for zd in [zd_raw])
        if max_concurrent_facility < EPSILON : max_concurrent_facility = num_team_members * 2 if num_team_members >0 else 10
        
        workload_metric = np.clip((total_backlog / (max_concurrent_facility + EPSILON)) * 10.0, 0, 10)
        workload_metric += (step / (num_steps + EPSILON)) * 2.0 # Time pressure
        _perceived_workload_scores[step] = np.clip(workload_metric, 0, 10)

        # --- Worker Fatigue (0-1 scale) ---
        fatigue_increase = cfg_base_fatigue_rate * (1.0 + _perceived_workload_scores[step] / 20.0) # Workload effect more gradual
        fatigue_increase *= active_event_effects["fatigue_rate_modifier"]
        if team_initiative == "More frequent breaks": fatigue_increase *= (1.0 - _get_config_param(config, 'INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.4, float))
        if num_team_members > 0:
            worker_fatigue += fatigue_increase
            if active_event_effects["fatigue_recovery_factor"] > 0: worker_fatigue *= (1.0 - active_event_effects["fatigue_recovery_factor"])
            worker_fatigue = np.clip(worker_fatigue, 0.0, 1.0)
        avg_fatigue_this_step_end = np.mean(worker_fatigue) if num_team_members > 0 else 0.0

        # --- Well-Being (0-100 scale) ---
        wb_now = _wellbeing_scores[max(0, step-1)] if step > 0 else _wellbeing_scores[0]
        wb_now -= (avg_fatigue_this_step_end * 25.0) 
        wb_now -= (_perceived_workload_scores[step] - cfg_target_workload) * 1.5
        wb_now -= active_event_effects["wellbeing_drop_factor"] * 30.0 
        wb_now += active_event_effects["wellbeing_boost_abs"]
        wb_now += (cfg_leadership_factor - 0.5) * 20.0
        if team_initiative != "Increased Autonomy": wb_now -= cfg_stress_low_control_drop
        collab_prev = _collaboration_metric_scores[max(0,step-1)] if step > 0 else _collaboration_metric_scores[0]
        if collab_prev < 50: wb_now -= (cfg_isolation_max_drop / 50.0) * (50.0 - collab_prev)
        if team_initiative == "Team recognition" and step > 0 and step % (max(1,num_steps // 3)) == 0 :
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
        if team_initiative == "Team recognition" and step > 0 and step % (max(1,num_steps // 3)) == 1 : 
            ps_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS', 9.0, float)
        if team_initiative == "Increased Autonomy": 
            ps_now += _get_config_param(config, 'INITIATIVE_AUTONOMY_PSYCHSAFETY_BOOST_ABS', 8.0, float)
        cohesion_prev_ps = _team_cohesion_scores[max(0,step-1)] if step > 0 else _team_cohesion_scores[0]
        ps_now += (cohesion_prev_ps - cfg_cohesion_baseline) * cfg_cohesion_ps_factor
        _psych_safety_scores[step] = np.clip(ps_now, 10.0, 100.0)

        # --- Team Cohesion (0-100 scale) ---
        cohesion_now = _team_cohesion_scores[max(0,step-1)] if step > 0 else _team_cohesion_scores[0]
        cohesion_now -= (0.1 + 0.4*is_any_disruption_active + _perceived_workload_scores[step]/70.0) # Base decay + disruption/workload
        cohesion_now += active_event_effects["cohesion_boost_abs"]
        if _psych_safety_scores[step] > 70.0: cohesion_now += 0.5 # Synergies
        if collab_prev > 60.0: cohesion_now += 0.4
        if team_initiative == "Team recognition" and step > 0 and step % (max(1,num_steps // 3)) == 0 :
            cohesion_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_COHESION_BOOST_ABS', 7.0, float)
        _team_cohesion_scores[step] = np.clip(cohesion_now, 10.0, 100.0)

        # --- Wellbeing Triggers ---
        if _wellbeing_scores[step] < cfg_wb_alert_thresh: wellbeing_triggers_dict['threshold'].append(step)
        if step > 0 and _wellbeing_scores[step] < _wellbeing_scores[step-1] - 12.0: wellbeing_triggers_dict['trend'].append(step)
        if is_any_disruption_active:
            for active_evt in active_event_details: # Check all active events
                if "Disruption" in active_evt.get("Event Type",""):
                    evt_aff_zones = active_evt.get("Affected Zones", [])
                    intensity_evt = active_evt.get("Intensity", 1.0)
                    if num_team_members > 0 and worker_assigned_zone and len(worker_assigned_zone) == num_team_members:
                        for i_w in range(num_team_members):
                            w_zone = worker_assigned_zone[i_w]
                            if not evt_aff_zones or "All" in evt_aff_zones or w_zone in evt_aff_zones:
                                if worker_fatigue[i_w] > 0.7 or random.random() < intensity_evt * 0.15: # Higher chance if fatigued
                                    if w_zone in wellbeing_triggers_dict['work_area'] and step not in wellbeing_triggers_dict['work_area'][w_zone]:
                                        wellbeing_triggers_dict['work_area'][w_zone].append(step)
        
        # --- Operational Metrics (Uptime, Compliance, Quality, Throughput, OEE) ---
        # Uptime
        uptime_now = _uptime_percent[max(0, step-1)] if step > 0 else 100.0
        equipment_failed_this_step = False
        if random.random() < cfg_equip_fail_prob : uptime_now -= random.uniform(5, 20); equipment_failed_this_step = True # Smaller random drops
        uptime_mult_from_event = min(active_event_effects["zone_uptime_multiplier"].values()) if active_event_effects["zone_uptime_multiplier"] else 1.0
        uptime_now *= uptime_mult_from_event
        if is_any_disruption_active and uptime_mult_from_event == 1.0: uptime_now -= 15.0 * active_event_effects["compliance_reduction_factor"] # General disruption impact
        _uptime_percent[step] = np.clip(uptime_now, 5.0, 100.0) # Min uptime 5%

        # Compliance
        compliance_now = cfg_base_task_comp_prob * 100.0
        compliance_now *= (1.0 - avg_fatigue_this_step_end * cfg_fatigue_compliance_impact) # Fatigue effect
        avg_task_complexity = 0.5
        if num_team_members > 0 and worker_assigned_zone and len(worker_assigned_zone)==num_team_members and work_areas_config:
            complexities_list = [_get_config_param(work_areas_config.get(worker_assigned_zone[w], {}), 'task_complexity', 0.5, float) for w in range(num_team_members)]
            if complexities_list: avg_task_complexity = np.mean(complexities_list)
        compliance_now -= avg_task_complexity * cfg_complexity_compliance_impact * 100.0 # Complexity reduces points
        compliance_now *= ( (_psych_safety_scores[step]/100.0 * 0.2) + 0.8 ) # Psych safety factor (max 20% boost/penalty from baseline)
        compliance_now *= ( (cfg_comm_factor * 0.3) + 0.7 ) # Comm factor
        if team_initiative == "Increased Autonomy": compliance_now *= (1 + _get_config_param(config, 'INITIATIVE_AUTONOMY_COMPLIANCE_BOOST_FACTOR', 0.06, float))
        compliance_now = max(cfg_min_comp_disrupt, compliance_now * (1.0 - active_event_effects["compliance_reduction_factor"])) # Event impact
        _task_compliance_scores[step] = np.clip(compliance_now, 0.0, 100.0)

        # Quality
        quality_now = (100.0 - _get_config_param(config, 'BASE_QUALITY_DEFECT_RATE', 0.015, float) * 100.0) * \
                      math.pow((_task_compliance_scores[step] / 100.0), 1.1) # Quality strongly tied to compliance
        if is_any_disruption_active: quality_now -= 8.0 * active_event_effects["compliance_reduction_factor"]
        _quality_rate_percent[step] = np.clip(quality_now, 20.0, 100.0) # Min quality 20%

        # Throughput & Task Completion Rate
        max_tput_cfg = _get_config_param(config, 'THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL', 120.0, float)
        capacity_factor = (_uptime_percent[step]/100.0) * (_task_compliance_scores[step]/100.0) * \
                          (1.0 - avg_fatigue_this_step_end * 0.6) * \
                          np.clip((1.0 - (_perceived_workload_scores[step] / 30.0)), 0.05, 1.0) # Workload impact, min 5% capacity
        tput_disrupt_impact = 0.75 * active_event_effects["compliance_reduction_factor"] # Disruption impacts throughput
        actual_units = max_tput_cfg * capacity_factor * (1.0 - tput_disrupt_impact)
        _throughput_percent_of_max[step] = np.clip((actual_units / (max_tput_cfg + EPSILON)) * 100.0, 0.0, 100.0)
        _task_completion_rate_percent[step] = _throughput_percent_of_max[step] # Assumed same for now

        # --- Downtime Logging ---
        current_step_downtime_duration = 0.0 # Track total for this step if needed, but primary is raw log

        # Event-induced general downtime
        if active_event_effects["downtime_prob_modifier"] > 0 and random.random() < active_event_effects["downtime_prob_modifier"]:
            mean_dur = cfg_downtime_mean_min * active_event_effects["downtime_mean_factor"]
            std_dur = cfg_downtime_std_min * math.sqrt(abs(active_event_effects["downtime_mean_factor"])) # sqrt for variance scaling
            duration = max(0.0, np.random.normal(mean_dur, std_dur))
            if duration > EPSILON:
                causal_evt_type = "Generic Event Impact"
                if active_event_details:
                    for evt_d in active_event_details:
                        evt_p = event_type_params_config.get(evt_d.get("Event Type",""),{})
                        if _get_config_param(evt_p, "downtime_prob_modifier",0.0,float) > 0 or _get_config_param(evt_p, "downtime_mean_factor",1.0,float) != 1.0 :
                            causal_evt_type = evt_d.get("Event Type","Unknown Event") ; break
                _raw_downtime_log.append({'step': step, 'duration': min(duration, minutes_per_interval), 'cause': causal_evt_type})
        
        # Equipment Failure downtime
        if equipment_failed_this_step and random.random() < cfg_dt_from_equip_fail_prob:
            duration = cfg_equip_dt_intervals * minutes_per_interval
            if duration > EPSILON: _raw_downtime_log.append({'step': step, 'duration': min(duration, minutes_per_interval), 'cause': "Equipment Failure"})
        
        # Human Error downtime
        if (_task_compliance_scores[step] < 40 or avg_fatigue_this_step_end > 0.9) and random.random() < 0.04: # Stricter conditions
            duration = max(0.0, np.random.normal(cfg_downtime_mean_min*0.3, cfg_downtime_std_min*0.15))
            if duration > EPSILON: _raw_downtime_log.append({'step': step, 'duration': min(duration, minutes_per_interval), 'cause': "Human Error"})
        
        # Add a small chance of random "Minor Stoppage" if no other major downtime logged for the step
        current_downtime_causes_in_step = [d['cause'] for d in _raw_downtime_log if d['step'] == step]
        if not any(c != "Minor Stoppage" for c in current_downtime_causes_in_step) and random.random() < 0.015: # Low chance for minor stoppage
             duration = random.uniform(minutes_per_interval * 0.1, minutes_per_interval * 0.3) # 10-30% of interval
             if duration > EPSILON: _raw_downtime_log.append({'step': step, 'duration': duration, 'cause': "Minor Stoppage"})


        # --- Operational Recovery & Productivity Loss ---
        current_oee_val = (_uptime_percent[step]/100.0) * (_throughput_percent_of_max[step]/100.0) * (_quality_rate_percent[step]/100.0)
        if not is_any_disruption_active: 
            prev_rec = _operational_recovery_scores[max(0,step-1)] if step > 0 else 100.0
            target_rec = current_oee_val * 100.0
            rec_rate = 1.0 - math.exp(-math.log(2) / (recovery_halflife_intervals + EPSILON)) # Exponential recovery towards target
            _operational_recovery_scores[step] = np.clip(prev_rec + (target_rec - prev_rec) * rec_rate, 0, 100)
        else: 
            _operational_recovery_scores[step] = np.clip(current_oee_val * 100.0, 0, 100) # During disruption, recovery is current OEE
        _productivity_loss_percent[step] = np.clip(100.0 - _operational_recovery_scores[step], 0, 100)

        # --- Collaboration Metric Score ---
        base_collab_val = 60.0 + (_team_cohesion_scores[step]-cfg_cohesion_baseline)*0.4 - (_perceived_workload_scores[step]-cfg_target_workload)*2.0
        if is_any_disruption_active: base_collab_val -= 20.0 * active_event_effects["compliance_reduction_factor"]
        _collaboration_metric_scores[step] = np.clip(base_collab_val + np.random.normal(0,1.5) ,5, 95)

        # --- Worker Movement & Status ---
        if num_team_members > 0 and worker_assigned_zone and len(worker_assigned_zone) == num_team_members:
            for i_w in range(num_team_members):
                assigned_zone_name = worker_assigned_zone[i_w]
                zone_cfg = work_areas_config.get(assigned_zone_name, {})
                target_x_w, target_y_w = facility_width / 2.0, facility_height / 2.0 # Default center

                if isinstance(zone_cfg, dict) and zone_cfg.get('coords'):
                    zc_w = zone_cfg['coords']
                    if isinstance(zc_w, list) and len(zc_w) == 2 and isinstance(zc_w[0], tuple) and isinstance(zc_w[1], tuple):
                         target_x_w = random.uniform(min(zc_w[0][0], zc_w[1][0]), max(zc_w[0][0], zc_w[1][0]))
                         target_y_w = random.uniform(min(zc_w[0][1], zc_w[1][1]), max(zc_w[0][1], zc_w[1][1]))
                
                on_break_meeting = False
                for evt_d_mov in active_event_details: # Check active events for this worker
                    evt_type_mov = evt_d_mov.get("Event Type")
                    if evt_type_mov in ["Scheduled Break", "Short Pause", "Team Meeting"]:
                        scope_mov = evt_d_mov.get("Scope", "Individual")
                        aff_zones_mov = evt_d_mov.get("Affected Zones", [])
                        if scope_mov == "All" or assigned_zone_name in aff_zones_mov or not aff_zones_mov :
                            on_break_meeting = True
                            # Try to move to a designated Break Room if one exists and event is not a meeting in specific work zone
                            break_room_name_mov = next((zn_br for zn_br, zd_br in work_areas_config.items() if isinstance(zd_br, dict) and zd_br.get("is_rest_area")), None)
                            if break_room_name_mov and (evt_type_mov != "Team Meeting" or break_room_name_mov in aff_zones_mov or not aff_zones_mov or "All" in aff_zones_mov):
                                br_cfg_mov = work_areas_config.get(break_room_name_mov,{})
                                if isinstance(br_cfg_mov, dict) and br_cfg_mov.get('coords'):
                                    br_c = br_cfg_mov['coords']
                                    if isinstance(br_c, list) and len(br_c) == 2 and isinstance(br_c[0], tuple) and isinstance(br_c[1], tuple):
                                        target_x_w = random.uniform(min(br_c[0][0], br_c[1][0]), max(br_c[0][0], br_c[1][0]))
                                        target_y_w = random.uniform(min(br_c[0][1], br_c[1][1]), max(br_c[0][1], br_c[1][1]))
                            break 
                
                move_factor = 0.4 if on_break_meeting else 0.25 # Move faster to break/meeting
                dx = (target_x_w - worker_current_x[i_w]) * move_factor + np.random.normal(0, 0.8) # Smaller random jitter
                dy = (target_y_w - worker_current_y[i_w]) * move_factor + np.random.normal(0, 0.8)
                worker_current_x[i_w] = np.clip(worker_current_x[i_w] + dx, 0, facility_width)
                worker_current_y[i_w] = np.clip(worker_current_y[i_w] + dy, 0, facility_height)
                
                current_status = 'working'
                if on_break_meeting: current_status = 'break' 
                elif worker_fatigue[i_w] > 0.9: current_status = 'exhausted' # Higher threshold for exhausted
                elif worker_fatigue[i_w] > 0.7: current_status = 'fatigued'  # Higher threshold for fatigued
                elif is_any_disruption_active and random.random() < 0.25: current_status = 'disrupted' 
                elif zone_task_backlog.get(assigned_zone_name,0) < EPSILON and not _get_config_param(work_areas_config.get(assigned_zone_name,{}), "is_rest_area", False, bool):
                    if random.random() < 0.1: current_status = 'idle' # Small chance of idle if no backlog
                
                team_positions_data.append({
                    'step': step, 'worker_id': i_w, 'x': worker_current_x[i_w], 'y': worker_current_y[i_w], 
                    'z': random.uniform(0, 0.05), 'zone': assigned_zone_name, 'status': current_status
                })

    # --- Final Output Preparation ---
    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame(columns=['step', 'worker_id', 'x', 'y', 'z', 'zone', 'status'])
    task_compliance_output = {'data': list(_task_compliance_scores), 'forecast': [max(0,s-random.uniform(2,6)) for s in _task_compliance_scores], 'z_scores': list(np.random.normal(0,0.7,num_steps))}
    collaboration_metric_output = {'data': list(_collaboration_metric_scores), 'forecast': [min(100,s+random.uniform(2,6)) for s in _collaboration_metric_scores]}
    operational_recovery_output = list(_operational_recovery_scores)
    
    efficiency_df_data = {'uptime': list(_uptime_percent), 'throughput': list(_throughput_percent_of_max), 'quality': list(_quality_rate_percent)}
    efficiency_df = pd.DataFrame(efficiency_df_data)
    if not efficiency_df.empty:
        efficiency_df['oee'] = np.clip((efficiency_df['uptime']/100.0 * efficiency_df['throughput']/100.0 * efficiency_df['quality']/100.0) * 100.0, 0.0, 100.0)
    else: efficiency_df = pd.DataFrame(columns=['uptime', 'throughput', 'quality', 'oee'])

    productivity_loss_output = list(_productivity_loss_percent)
    worker_wellbeing_output = {'scores': list(_wellbeing_scores), 'triggers': wellbeing_triggers_dict, 'team_cohesion_scores': list(_team_cohesion_scores), 'perceived_workload_scores': list(_perceived_workload_scores)}
    psychological_safety_output = list(_psych_safety_scores)
    feedback_impact_output = list(np.random.normal(0, 0.05, num_steps))
    downtime_events_log_final_output = _raw_downtime_log
    task_completion_rate_output = list(_task_completion_rate_percent)

    logger.info(f"Simulation completed. Steps: {num_steps}, Team Members: {num_team_members}, Initiative: {team_initiative}.", extra={'user_action': 'SimEnd'})
    return (
        team_positions_df, task_compliance_output, collaboration_metric_output,
        operational_recovery_output, efficiency_df, productivity_loss_output,
        worker_wellbeing_output, psychological_safety_output, feedback_impact_output,
        downtime_events_log_final_output, task_completion_rate_output
    )
    
