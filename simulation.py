# simulation.py
import numpy as np
import pandas as pd
import random
import math
import logging

logger = logging.getLogger(__name__)
EPSILON = 1e-6 # Small constant to prevent division by zero

# --- Constants from Config (or defaults if not found) ---
# These would ideally be more systematically pulled from config or have clear defaults
# For now, assuming they are within the config dictionary or have reasonable implicit defaults
# in the original code.

def _get_config_param(config, key, default):
    """Helper to safely get config parameters."""
    return config.get(key, default)

def simulate_workplace_operations(num_team_members: int, num_steps: int, disruption_event_steps: list, team_initiative: str, config: dict):
    np.random.seed(42)
    random.seed(42)

    # --- Configuration Parameters ---
    facility_width, facility_height = _get_config_param(config, 'FACILITY_SIZE', (100, 80))
    work_areas_config = _get_config_param(config, 'WORK_AREAS', {})
    
    initial_wellbeing_mean = _get_config_param(config, 'INITIAL_WELLBEING_MEAN', 0.8)
    psych_safety_baseline = _get_config_param(config, 'PSYCH_SAFETY_BASELINE', 0.75)
    team_cohesion_baseline = _get_config_param(config, 'TEAM_COHESION_BASELINE', 0.7)
    recovery_halflife_intervals = _get_config_param(config, 'RECOVERY_HALFLIFE_INTERVALS', 10)
    disruption_duration_mean_intervals = _get_config_param(config, 'DISRUPTION_DURATION_MEAN_INTERVALS', 5)
    disruption_duration_std_intervals = _get_config_param(config, 'DISRUPTION_DURATION_STD_INTERVALS', 2)
    wellbeing_alert_threshold = _get_config_param(config, 'WELLBEING_ALERT_THRESHOLD', 60.0)
    shift_duration_minutes_config = _get_config_param(config, 'SHIFT_DURATION_MINUTES', 480)


    # --- Initialize Metric Arrays ---
    _task_compliance_scores = np.zeros(num_steps)
    _collaboration_scores = np.zeros(num_steps)
    _operational_recovery_scores = np.zeros(num_steps)
    _wellbeing_scores = np.zeros(num_steps)
    _psych_safety_scores = np.zeros(num_steps)
    _team_cohesion_scores = np.zeros(num_steps)
    _perceived_workload_scores = np.zeros(num_steps)
    _productivity_loss_percent = np.zeros(num_steps)
    _downtime_events_per_interval = [{'duration': 0.0, 'cause': 'None'} for _ in range(num_steps)] # duration as float
    _task_completion_rate_percent = np.zeros(num_steps)
    _uptime_percent = np.ones(num_steps) * 100.0
    _quality_rate_percent = np.ones(num_steps) * 100.0
    _throughput_percent_of_max = np.zeros(num_steps)

    # --- Worker and Zone Setup ---
    team_positions_data = []
    worker_current_x = np.random.uniform(0, facility_width, num_team_members) if num_team_members > 0 else np.array([])
    worker_current_y = np.random.uniform(0, facility_height, num_team_members) if num_team_members > 0 else np.array([])
    
    # Create a FallbackZone if no work areas are defined or if needed
    if not work_areas_config and num_team_members > 0:
        logger.warning("Simulation: No WORK_AREAS defined in config. Creating and assigning all workers to 'FallbackZone'.")
        work_areas_config["FallbackZone"] = {
            'coords': [(0,0), (facility_width, facility_height)], 
            'workers': num_team_members, 
            'tasks_per_interval': 1, 
            'task_complexity': 0.5, 
            'base_productivity': 0.7, 
            'max_concurrent_tasks': max(1, num_team_members)
        }
    
    worker_zone_names_from_config = list(work_areas_config.keys())
    worker_assigned_zone = []

    if num_team_members > 0:
        if not worker_zone_names_from_config: # Should not happen if FallbackZone is created
            logger.error("Critical: No zones available for worker assignment even after fallback check.")
            worker_assigned_zone = ["ErrorZone"] * num_team_members # Should not happen
        else:
            configured_workers_sum = sum(zone_details.get('workers', 0) for zone_details in work_areas_config.values())
            
            if configured_workers_sum == num_team_members:
                for zone_name, zone_details in work_areas_config.items():
                    worker_assigned_zone.extend([zone_name] * zone_details.get('workers', 0))
            else: # Distribute as evenly as possible if sum doesn't match or is 0
                logger.info(f"Sum of configured workers ({configured_workers_sum}) does not match team size ({num_team_members}). Distributing workers.")
                for i in range(num_team_members):
                    worker_assigned_zone.append(worker_zone_names_from_config[i % len(worker_zone_names_from_config)])
            
            # Ensure correct length and shuffle
            if len(worker_assigned_zone) > num_team_members:
                worker_assigned_zone = worker_assigned_zone[:num_team_members]
            while len(worker_assigned_zone) < num_team_members:
                 worker_assigned_zone.append(random.choice(worker_zone_names_from_config)) # Fill any gaps
            random.shuffle(worker_assigned_zone)

    worker_fatigue = np.random.uniform(0.0, 0.1, num_team_members) if num_team_members > 0 else np.array([])
    zone_task_backlog = {zn: 0 for zn in work_areas_config.keys()}

    # --- Simulation State Variables ---
    active_disruption_intensity = 0.0
    disruption_linger_factor = 1.0 - (1.0 / (recovery_halflife_intervals + EPSILON))
    current_disruption_end_step = -1

    # Initialize step 0 values
    if num_steps > 0:
        _wellbeing_scores[0] = initial_wellbeing_mean * 100.0
        _psych_safety_scores[0] = psych_safety_baseline * 100.0
        _team_cohesion_scores[0] = team_cohesion_baseline * 100.0
        _operational_recovery_scores[0] = 100.0
        # Initial perceived workload (can be low if no backlog initially)
        _perceived_workload_scores[0] = np.clip(
             sum(zone_task_backlog.values()) / 
             (sum(zd.get('max_concurrent_tasks', zd.get('tasks_per_interval', 0) * 1.5) for zd in work_areas_config.values()) + EPSILON) * 10.0,
             0, 10
        )


    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa: [] for wa in work_areas_config}, 'disruption': []}
    downtime_causes_list = _get_config_param(config, 'DOWNTIME_CAUSES_LIST', ["Equipment Failure", "Material Shortage", "Process Bottleneck", "Human Error", "Utility Outage", "External Supply Chain"])


    # --- Main Simulation Loop ---
    for step in range(num_steps):
        is_new_disruption_starting_this_step = False
        if step in disruption_event_steps:
            active_disruption_intensity = 1.0
            is_new_disruption_starting_this_step = True
            if step < num_steps : wellbeing_triggers_dict['disruption'].append(step) # Check bounds for list append
            
            disruption_duration_raw = np.random.normal(disruption_duration_mean_intervals, disruption_duration_std_intervals)
            current_disruption_duration = max(1, int(round(disruption_duration_raw)))
            current_disruption_end_step = step + current_disruption_duration
        elif active_disruption_intensity > 0:
            if step >= current_disruption_end_step: # If formal disruption period ended
                active_disruption_intensity *= disruption_linger_factor
            if active_disruption_intensity < 0.05: # Disruption fully faded
                active_disruption_intensity = 0.0
                current_disruption_end_step = -1
        
        # --- Task Backlog and Workload ---
        avg_fatigue_current_step = np.mean(worker_fatigue) if num_team_members > 0 else 0.0
        for zone_name, zone_details in work_areas_config.items():
            tasks_arriving = zone_details.get('tasks_per_interval', 0) * (1.0 + active_disruption_intensity * random.uniform(-0.2, 0.2))
            workers_in_this_zone_count = worker_assigned_zone.count(zone_name) if num_team_members > 0 else 0
            
            prev_compliance_score = _task_compliance_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'BASE_TASK_COMPLETION_PROB', 0.95) * 100.0
            compliance_factor_for_processing = prev_compliance_score / 100.0
            
            zone_processing_capacity = workers_in_this_zone_count * \
                                     zone_details.get('base_productivity', 0.8) * \
                                     (1.0 - avg_fatigue_current_step * 0.3) * \
                                     compliance_factor_for_processing
            
            zone_task_backlog[zone_name] = max(0, zone_task_backlog.get(zone_name,0) + tasks_arriving - zone_processing_capacity)

        total_backlog_current_step = sum(zone_task_backlog.values())
        max_concurrent_total_facility = sum(zd.get('max_concurrent_tasks', zd.get('tasks_per_interval', 0) * 1.5) for zd in work_areas_config.values())
        
        workload_pressure_from_backlog_metric = total_backlog_current_step / (max_concurrent_total_facility + EPSILON)
        # Perceived workload also influenced by time progression (anticipation/fatigue)
        _perceived_workload_scores[step] = np.clip(workload_pressure_from_backlog_metric * 10.0 + (step / (num_steps + EPSILON)) * 1.5, 0, 10)

        # --- Fatigue ---
        fatigue_rate_this_step = _get_config_param(config, 'WELLBEING_FATIGUE_RATE_PER_INTERVAL', 0.002)
        if team_initiative == "More frequent breaks":
            fatigue_rate_this_step *= (1.0 - _get_config_param(config, 'INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.3))
        
        fatigue_rate_this_step *= (1.0 + _perceived_workload_scores[step] / 15.0) # Higher workload increases fatigue rate
        if num_team_members > 0:
            worker_fatigue += fatigue_rate_this_step * (1.0 + active_disruption_intensity * 0.3) # Disruptions exacerbate fatigue
            worker_fatigue = np.clip(worker_fatigue, 0.0, 1.0)
        avg_fatigue_current_step = np.mean(worker_fatigue) if num_team_members > 0 else 0.0 # Recalculate after update

        # --- Wellbeing ---
        wb_now = _wellbeing_scores[max(0, step - 1)] if step > 0 else initial_wellbeing_mean * 100.0
        wb_now -= (avg_fatigue_current_step * 20.0 + (_perceived_workload_scores[step] - 5.0) * 1.0) # Impact of fatigue and workload
        wb_now -= active_disruption_intensity * _get_config_param(config, 'DISRUPTION_WELLBEING_DROP', 0.2) * 25.0 # Direct disruption impact
        
        leadership_support_factor = _get_config_param(config, 'LEADERSHIP_SUPPORT_FACTOR', 0.5)
        wb_now += (leadership_support_factor - 0.5) * 10.0 # Leadership support effect

        if team_initiative != "Increased Autonomy": # Stress from low control if autonomy is not active
            wb_now -= _get_config_param(config, 'STRESS_FROM_LOW_CONTROL_FACTOR', 0.02) * 15.0
        
        prev_collab_score = _collaboration_scores[max(0, step - 1)] if step > 0 else _get_config_param(config, 'TARGET_COLLABORATION', 60.0)
        if prev_collab_score < 50.0: # Isolation impact
            wb_now -= _get_config_param(config, 'ISOLATION_IMPACT_ON_WELLBEING', 0.1) * (50.0 - prev_collab_score) * 0.4
        
        # Initiative-specific wellbeing boosts
        break_interval = num_steps // random.randint(3,5) if num_steps > 5 else 1
        recognition_interval = num_steps // random.randint(2,3) if num_steps > 2 else 1

        if team_initiative == "More frequent breaks" and step > 0 and step % break_interval == 0:
            wb_now += _get_config_param(config, 'INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST_ABS', 0.05) * 100.0
            if num_team_members > 0: worker_fatigue *= 0.6 # Reduce fatigue during these breaks
        if team_initiative == "Team recognition" and step > 0 and step % recognition_interval == 0:
            wb_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS', 0.05) * 100.0
        if team_initiative == "Increased Autonomy":
            wb_now += _get_config_param(config, 'INITIATIVE_AUTONOMY_WELLBEING_BOOST_ABS', 0.04) * 100.0
        _wellbeing_scores[step] = np.clip(wb_now, 5.0, 100.0)


        # --- Psychological Safety ---
        ps_now = _psych_safety_scores[max(0, step - 1)] if step > 0 else psych_safety_baseline * 100.0
        ps_now -= _get_config_param(config, 'PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL', 0.0005) * 100.0
        ps_now -= active_disruption_intensity * _get_config_param(config, 'UNCERTAINTY_DURING_DISRUPTION_IMPACT_PSYCH_SAFETY', 0.1) * 15.0
        ps_now += (leadership_support_factor - 0.5) * 15.0
        ps_now += (_get_config_param(config, 'COMMUNICATION_EFFECTIVENESS_FACTOR', 0.5) - 0.5) * 10.0
        
        if team_initiative == "Team recognition" and step > 0 and step % recognition_interval == 1: # Offset for variety
            ps_now += _get_config_param(config, 'INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS', 0.08) * 100.0
        if team_initiative == "Increased Autonomy":
            ps_now += _get_config_param(config, 'INITIATIVE_AUTONOMY_PSYCHSAFETY_BOOST_ABS', 0.07) * 100.0
        
        prev_team_cohesion_score = _team_cohesion_scores[max(0, step - 1)] if step > 0 else team_cohesion_baseline * 100.0
        ps_now += (prev_team_cohesion_score - team_cohesion_baseline * 100.0) * _get_config_param(config, 'TEAM_COHESION_IMPACT_ON_PSYCH_SAFETY', 0.15)
        _psych_safety_scores[step] = np.clip(ps_now, 10.0, 100.0)

        # --- Team Cohesion ---
        cohesion_now = prev_team_cohesion_score
        cohesion_now -= (0.15 + active_disruption_intensity * 0.7 + (_perceived_workload_scores[step] / 50.0))
        if _psych_safety_scores[step] > 70.0: cohesion_now += 0.8
        if prev_collab_score > 60.0: cohesion_now += 0.6
        if team_initiative == "Team recognition" and step > 0 and step % recognition_interval == 0:
            cohesion_now += 5.0
        _team_cohesion_scores[step] = np.clip(cohesion_now, 10.0, 100.0)

        # --- Wellbeing Triggers ---
        if _wellbeing_scores[step] < wellbeing_alert_threshold:
            wellbeing_triggers_dict['threshold'].append(step)
        if step > 0 and _wellbeing_scores[step] < _wellbeing_scores[step - 1] - 12.0: # Significant drop
            wellbeing_triggers_dict['trend'].append(step)
        
        if is_new_disruption_starting_this_step and _wellbeing_scores[step] < (_wellbeing_scores[max(0, step - 1)] if step > 0 else _wellbeing_scores[0]) * 0.90:
            if num_team_members > 0:
                # Identify workers with high fatigue or randomly affected by disruption
                affected_worker_indices = [i for i in range(num_team_members) if worker_fatigue[i] > 0.65 or random.random() < active_disruption_intensity * 0.15]
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
        current_uptime -= active_disruption_intensity * 30.0 # Direct disruption impact on uptime
        _uptime_percent[step] = np.clip(current_uptime, 15.0, 100.0)
        
        # Task Compliance
        base_compliance_val = _get_config_param(config, 'BASE_TASK_COMPLETION_PROB', 0.95) * 100.0
        compliance_now = base_compliance_val * (1.0 - avg_fatigue_current_step * _get_config_param(config, 'FATIGUE_IMPACT_ON_COMPLIANCE', 0.15))
        
        avg_worker_task_complexity = 0.5
        if num_team_members > 0 and worker_assigned_zone:
            complexities = [work_areas_config.get(worker_assigned_zone[w], {}).get('task_complexity', 0.5) for w in range(num_team_members) if w < len(worker_assigned_zone)]
            if complexities: avg_worker_task_complexity = np.mean(complexities)
            
        compliance_now -= avg_worker_task_complexity * _get_config_param(config, 'COMPLEXITY_IMPACT_ON_COMPLIANCE', 0.2) * 100.0
        compliance_now *= (_psych_safety_scores[step] / 100.0 * 0.1 + 0.9) # Psych safety influence
        compliance_now *= (_get_config_param(config, 'COMMUNICATION_EFFECTIVENESS_FACTOR', 0.7) * 0.3 + 0.7) # Communication
        
        min_compliance_during_disruption = _get_config_param(config, 'MIN_COMPLIANCE_DURING_DISRUPTION', 20.0)
        disruption_compliance_reduction = _get_config_param(config, 'DISRUPTION_COMPLIANCE_REDUCTION_FACTOR', 0.6)
        compliance_now = max(min_compliance_during_disruption, compliance_now * (1.0 - active_disruption_intensity * disruption_compliance_reduction))
        _task_compliance_scores[step] = np.clip(compliance_now, 0.0, 100.0)

        # Quality Rate
        quality_now = (100.0 - _get_config_param(config, 'BASE_QUALITY_DEFECT_RATE', 0.02) * 100.0) * \
                      (_task_compliance_scores[step] / 100.0)**1.1 # Quality tied to compliance
        quality_now -= active_disruption_intensity * 12.0 # Disruptions impact quality
        _quality_rate_percent[step] = np.clip(quality_now, 30.0, 100.0)

        # Throughput and Task Completion Rate
        max_potential_throughput_facility = _get_config_param(config, 'THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL', 100.0)
        effective_capacity_factor = (_uptime_percent[step] / 100.0) * \
                                    (_task_compliance_scores[step] / 100.0) * \
                                    (1.0 - avg_fatigue_current_step * 0.6) * \
                                    (1.0 - _perceived_workload_scores[step] / 30.0)
        actual_units_produced = max_potential_throughput_facility * effective_capacity_factor * (1.0 - active_disruption_intensity * 0.8)
        _throughput_percent_of_max[step] = np.clip((actual_units_produced / (max_potential_throughput_facility + EPSILON)) * 100.0, 0.0, 100.0)
        _task_completion_rate_percent[step] = _throughput_percent_of_max[step] # Assuming task completion rate is tied to throughput

        # --- Downtime ---
        current_downtime_duration = 0.0
        current_downtime_cause = "None"
        
        if is_new_disruption_starting_this_step and random.random() < _get_config_param(config, 'DOWNTIME_FROM_DISRUPTION_EVENT_PROB', 0.5):
            downtime_mean_disrupt = _get_config_param(config, 'DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE', 8.0)
            downtime_std_disrupt = _get_config_param(config, 'DOWNTIME_STD_MINUTES_PER_OCCURRENCE', 4.0)
            current_downtime_duration = max(0.0, np.random.normal(downtime_mean_disrupt, downtime_std_disrupt))
            current_downtime_cause = random.choice([c for c in downtime_causes_list if c not in ["Equipment Failure", "Human Error"]]) # Choose non-specific cause
        
        if equipment_failed_this_step_flag and random.random() < _get_config_param(config, 'DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB', 0.7):
            duration_equip_fail = _get_config_param(config, 'EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS', 3) * 2.0 # Assuming 2 min per interval for this calc
            current_downtime_duration = max(current_downtime_duration, duration_equip_fail)
            current_downtime_cause = "Equipment Failure" if current_downtime_cause == "None" else f"{current_downtime_cause}, Equip.Fail"
            
        if (_task_compliance_scores[step] < 50.0 or avg_fatigue_current_step > 0.9) and random.random() < 0.03: # Human error related downtime
            downtime_mean_human = _get_config_param(config, 'DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE', 8.0) * 0.4
            downtime_std_human = _get_config_param(config, 'DOWNTIME_STD_MINUTES_PER_OCCURRENCE', 4.0) * 0.2
            duration_human_error = max(0.0, np.random.normal(downtime_mean_human, downtime_std_human))
            current_downtime_duration = max(current_downtime_duration, duration_human_error)
            current_downtime_cause = "Human Error" if current_downtime_cause == "None" else f"{current_downtime_cause}, HumanError"
        
        interval_actual_length_minutes = (shift_duration_minutes_config / (num_steps if num_steps > 0 else 1.0))
        _downtime_events_per_interval[step] = {
            'duration': np.clip(current_downtime_duration, 0.0, interval_actual_length_minutes),
            'cause': current_downtime_cause if current_downtime_duration > EPSILON else "None"
        }

        # --- Operational Recovery & Productivity Loss ---
        current_oee_calc = (_uptime_percent[step] / 100.0) * \
                           (_throughput_percent_of_max[step] / 100.0) * \
                           (_quality_rate_percent[step] / 100.0)
        
        if active_disruption_intensity == 0.0 and step > (current_disruption_end_step if current_disruption_end_step >= 0 else -1):
            prev_recovery = _operational_recovery_scores[max(0, step - 1)]
            target_potential_recovery = current_oee_calc * 100.0
            # Exponential recovery towards target potential
            recovery_rate_factor = 1.0 - math.exp(-1.0 / (recovery_halflife_intervals + EPSILON))
            _operational_recovery_scores[step] = np.clip(prev_recovery + (target_potential_recovery - prev_recovery) * recovery_rate_factor, 0.0, 100.0)
        else: # During disruption or if intensity is still high, recovery is limited by current OEE
            _operational_recovery_scores[step] = np.clip(current_oee_calc * 100.0, 0.0, 100.0)
        
        _productivity_loss_percent[step] = np.clip(100.0 - _operational_recovery_scores[step] + np.random.normal(0, 0.5), 0.0, 100.0)

        # --- Collaboration Score ---
        base_collab = 60.0 + (_team_cohesion_scores[step] - 70.0) * 0.5 - (_perceived_workload_scores[step] - 5.0) * 2.5
        base_collab -= active_disruption_intensity * 25.0
        _collaboration_scores[step] = np.clip(base_collab + np.random.normal(0, 2), 5.0, 95.0)

        # --- Worker Movement and Status ---
        if num_team_members > 0:
            for i in range(num_team_members):
                current_assigned_zone_for_worker_i = worker_assigned_zone[i] # Should be valid now
                zone_details = work_areas_config.get(current_assigned_zone_for_worker_i, {})
                zone_coords = zone_details.get('coords')
                
                if zone_coords and len(zone_coords) == 2:
                    (zx0, zy0), (zx1, zy1) = zone_coords
                    target_x, target_y = (zx0 + zx1) / 2.0, (zy0 + zy1) / 2.0
                else: # Fallback if zone coords are invalid or missing
                    target_x, target_y = facility_width / 2.0, facility_height / 2.0
                
                # Movement logic
                move_factor = 0.2
                random_walk_std = 0.8
                move_x = (target_x - worker_current_x[i]) * move_factor + np.random.normal(0, random_walk_std)
                move_y = (target_y - worker_current_y[i]) * move_factor + np.random.normal(0, random_walk_std)
                worker_current_x[i] = np.clip(worker_current_x[i] + move_x, 0, facility_width)
                worker_current_y[i] = np.clip(worker_current_y[i] + move_y, 0, facility_height)
                
                # Status determination
                status_now = 'working'
                if worker_fatigue[i] > 0.85: status_now = 'exhausted'
                elif worker_fatigue[i] > 0.65: status_now = 'fatigued'
                
                if active_disruption_intensity > 0.55: status_now = 'disrupted'
                
                if team_initiative == "More frequent breaks" and step > 0 and num_steps > 5:
                    break_check_interval = num_steps // (random.randint(3,5) + 1) # Ensure interval > 0
                    if break_check_interval > 0 and step % break_check_interval == (i % break_check_interval):
                        status_now = 'break'
                        
                team_positions_data.append({
                    'step': step, 'worker_id': i, 'x': worker_current_x[i], 'y': worker_current_y[i],
                    'z': np.random.uniform(0, 0.1), # Minimal z variation for 3D view
                    'zone': current_assigned_zone_for_worker_i, 'status': status_now
                })
    
    # --- Prepare Output Data ---
    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame(columns=['step', 'worker_id', 'x', 'y', 'z', 'zone', 'status'])
    
    # Forecasts are simple illustrative shifts for now
    task_compliance_forecast = [max(0, s - random.uniform(1,5)) for s in _task_compliance_scores]
    collab_proximity_forecast = [min(100, s + random.uniform(1,5)) for s in _collaboration_scores]

    task_compliance = {'data': list(_task_compliance_scores), 'z_scores': list(np.random.normal(0, 0.5, num_steps)), 'forecast': task_compliance_forecast}
    collaboration_proximity = {'data': list(_collaboration_scores), 'forecast': collab_proximity_forecast}
    operational_recovery = list(_operational_recovery_scores)
    
    efficiency_df_data = {
        'uptime': list(_uptime_percent), 
        'throughput': list(_throughput_percent_of_max), 
        'quality': list(_quality_rate_percent)
    }
    efficiency_df = pd.DataFrame(efficiency_df_data)
    efficiency_df['oee'] = np.clip((efficiency_df['uptime']/100.0 * efficiency_df['throughput']/100.0 * efficiency_df['quality']/100.0) * 100.0, 0.0, 100.0)
    
    productivity_loss = list(_productivity_loss_percent)
    worker_wellbeing = {
        'scores': list(_wellbeing_scores), 
        'triggers': wellbeing_triggers_dict, 
        'team_cohesion_scores': list(_team_cohesion_scores), 
        'perceived_workload_scores': list(_perceived_workload_scores)
    }
    psychological_safety = list(_psych_safety_scores)
    
    # Illustrative feedback impact (not deeply modeled)
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
