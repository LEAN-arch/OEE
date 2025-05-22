# simulation.py
import numpy as np
import pandas as pd
import random
import math
import logging # Added for logging

logger = logging.getLogger(__name__)


def simulate_workplace_operations(num_team_members, num_steps, disruption_event_steps, team_initiative, config):
    np.random.seed(42); random.seed(42) 

    facility_width, facility_height = config.get('FACILITY_SIZE', (100, 80))
    work_areas_config = config.get('WORK_AREAS', {})
    
    _task_compliance_scores = np.zeros(num_steps); _collaboration_scores = np.zeros(num_steps)
    _operational_recovery_scores = np.zeros(num_steps); _wellbeing_scores = np.zeros(num_steps)
    _psych_safety_scores = np.zeros(num_steps); _team_cohesion_scores = np.zeros(num_steps) 
    _perceived_workload_scores = np.zeros(num_steps) 
    _productivity_loss_percent = np.zeros(num_steps); _downtime_minutes_per_interval = np.zeros(num_steps)
    _task_completion_rate_percent = np.zeros(num_steps); _uptime_percent = np.ones(num_steps) * 100 
    _quality_rate_percent = np.ones(num_steps) * 100; _throughput_percent_of_max = np.zeros(num_steps) 

    team_positions_data = []; 
    worker_current_x = np.random.uniform(0, facility_width, num_team_members)
    worker_current_y = np.random.uniform(0, facility_height, num_team_members)
    
    # --- Robust initialization of worker_assigned_zone and worker_zone_names ---
    worker_zone_names = list(work_areas_config.keys()) # Get all defined zone names
    if not worker_zone_names: # If no zones are defined in config
        logger.warning("No work areas defined in config. Assigning workers to 'DefaultZone'.")
        worker_zone_names = ["DefaultZone"] # Create a fallback zone name
        # If using this DefaultZone, ensure it has some placeholder details if accessed later
        if "DefaultZone" not in work_areas_config : # Add it to config if not present
            work_areas_config["DefaultZone"] = {'coords': [(0,0),(facility_width, facility_height)], 'workers':0, 'tasks_per_interval':1, 'task_complexity':0.5, 'base_productivity':0.7, 'max_concurrent_tasks': 5}


    worker_assigned_zone = []
    configured_workers_sum = sum(zone.get('workers', 0) for zone in work_areas_config.values())

    if configured_workers_sum == num_team_members and num_team_members > 0 : # If config worker sum matches team size
        for zone_name, zone_details in work_areas_config.items():
            worker_assigned_zone.extend([zone_name] * zone_details.get('workers',0))
        # Ensure length matches in case of rounding or slight mismatch issues
        if len(worker_assigned_zone) > num_team_members:
            worker_assigned_zone = worker_assigned_zone[:num_team_members]
        while len(worker_assigned_zone) < num_team_members and worker_zone_names: # If short, fill with random available zones
             worker_assigned_zone.append(random.choice(worker_zone_names))

    else: # Fallback to even or random distribution if config doesn't match or is 0
        for i in range(num_team_members):
             worker_assigned_zone.append(worker_zone_names[i % len(worker_zone_names)]) # Cycle through available zones
    
    if num_team_members > 0 and len(worker_assigned_zone) == num_team_members : # Only shuffle if populated correctly
        random.shuffle(worker_assigned_zone)
    elif num_team_members > 0 : # Log if there's a mismatch in final assignment
        logger.error(f"Worker assigned zone length mismatch: {len(worker_assigned_zone)} vs team_size {num_team_members}. Fallback might be imperfect.")
        # Ensure it has some assignment to prevent errors, even if suboptimal
        if not worker_assigned_zone and worker_zone_names:
             worker_assigned_zone = [worker_zone_names[0]] * num_team_members
        elif not worker_assigned_zone and not worker_zone_names : # Total fallback
             worker_assigned_zone = ["DefaultZone"]*num_team_members


    worker_fatigue = np.random.uniform(0.0, 0.1, num_team_members) 
    # Initialize zone_task_backlog based on *actual* zone names present
    zone_task_backlog = {zn: 0 for zn in worker_zone_names if zn in work_areas_config}


    active_disruption_intensity = 0.0; disruption_linger_factor = 0.85; current_disruption_end_step = -1
    _wellbeing_scores[0] = config.get('INITIAL_WELLBEING_MEAN', 0.8) * 100
    _psych_safety_scores[0] = config.get('PSYCH_SAFETY_BASELINE', 0.75) * 100
    _team_cohesion_scores[0] = config.get('TEAM_COHESION_BASELINE', 0.7) * 100
    _operational_recovery_scores[0] = 100 
    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa:[] for wa in worker_zone_names if wa in work_areas_config}, 'disruption': []}


    for step in range(num_steps):
        is_new_disruption_starting_this_step = False
        if step in disruption_event_steps:
            active_disruption_intensity = 1.0; is_new_disruption_starting_this_step = True
            wellbeing_triggers_dict['disruption'].append(step)
            current_disruption_duration = max(1, int(round(np.random.normal(config.get('DISRUPTION_DURATION_MEAN_INTERVALS',5), config.get('DISRUPTION_DURATION_STD_INTERVALS',2)))))
            current_disruption_end_step = step + current_disruption_duration
        elif active_disruption_intensity > 0:
            active_disruption_intensity = active_disruption_intensity * disruption_linger_factor if step >= current_disruption_end_step else active_disruption_intensity
            if active_disruption_intensity < 0.05: active_disruption_intensity = 0.0; current_disruption_end_step = -1
        
        avg_fatigue = np.mean(worker_fatigue) if num_team_members > 0 else 0
        
        # Iterate over zones present in work_areas_config to avoid errors if zone_task_backlog doesn't have a key
        for zone_name, zone_details in work_areas_config.items():
            if zone_name not in zone_task_backlog: zone_task_backlog[zone_name] = 0 # Ensure key exists

            tasks_arriving = zone_details.get('tasks_per_interval', 0) * (1 + active_disruption_intensity * random.uniform(-0.2, 0.2))
            workers_in_this_zone_count = worker_assigned_zone.count(zone_name) if num_team_members > 0 else 0
            
            zone_processing_capacity = workers_in_this_zone_count * zone_details.get('base_productivity', 0.8) * (1-avg_fatigue*0.3) * (_task_compliance_scores[max(0,step-1)]/100 if step>0 else 0.9)
            zone_task_backlog[zone_name] = max(0, zone_task_backlog.get(zone_name,0) + tasks_arriving - zone_processing_capacity)

        total_backlog = sum(zone_task_backlog.values()); active_workers_for_calc = num_team_members if num_team_members > 0 else 1
        workload_ratio = (total_backlog / active_workers_for_calc) * (num_steps / (step+1 + 1e-6)) # Normalize factor needs care
        _perceived_workload_scores[step] = np.clip(workload_ratio * 1.5, 0, 10) # Adjusted scaling


        fatigue_rate_this_step = config.get('WELLBEING_FATIGUE_RATE_PER_INTERVAL', 0.002)
        if team_initiative == "More frequent breaks": fatigue_rate_this_step *= (1 - config.get('INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.3))
        fatigue_rate_this_step *= (1 + _perceived_workload_scores[step]/20) 
        if num_team_members > 0 : worker_fatigue += fatigue_rate_this_step * (1 + active_disruption_intensity * 0.3) 
        worker_fatigue = np.clip(worker_fatigue, 0, 1.0); avg_fatigue = np.mean(worker_fatigue) if num_team_members > 0 else 0

        wb_now = _wellbeing_scores[max(0, step-1)] if step > 0 else config.get('INITIAL_WELLBEING_MEAN', 0.8) * 100
        wb_now -= (avg_fatigue * 15 + (_perceived_workload_scores[step] - 5) * 0.5) 
        wb_now -= active_disruption_intensity * config.get('DISRUPTION_WELLBEING_DROP', 0.2) * 20 
        if team_initiative != "Increased Autonomy": wb_now -= config.get('STRESS_FROM_LOW_CONTROL_FACTOR',0.02) * 10 
        collab_score_prev = _collaboration_scores[max(0,step-1)] if step > 0 else 60
        if collab_score_prev < 40 : wb_now -= config.get('ISOLATION_IMPACT_ON_WELLBEING',0.1) * (40 - collab_score_prev) * 0.5
        if team_initiative == "More frequent breaks" and step > 0 and step % (num_steps // random.randint(4,6) if num_steps > 5 else 1) == 0 : wb_now += config.get('INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST_ABS', 0.05) * 100; 
        if num_team_members > 0 : worker_fatigue *= 0.65 # Assuming break applies to all for avg_fatigue next step
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,4) if num_steps > 3 else 1) == 0: wb_now += config.get('INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS', 0.05) * 100
        if team_initiative == "Increased Autonomy": wb_now += config.get('INITIATIVE_AUTONOMY_WELLBEING_BOOST_ABS', 0.04) * 100
        _wellbeing_scores[step] = np.clip(wb_now, 5, 100)

        ps_now = _psych_safety_scores[max(0, step-1)] if step > 0 else config.get('PSYCH_SAFETY_BASELINE', 0.75) * 100
        ps_now -= config.get('PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL', 0.0005) * 100 
        ps_now -= active_disruption_intensity * config.get('UNCERTAINTY_DURING_DISRUPTION_IMPACT_PSYCH_SAFETY', 0.1) * 10 
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,4) if num_steps > 3 else 1) == 1: ps_now += config.get('INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS', 0.08) * 100
        if team_initiative == "Increased Autonomy": ps_now += config.get('INITIATIVE_AUTONOMY_PSYCHSAFETY_BOOST_ABS',0.07)*100
        current_cohesion = _team_cohesion_scores[max(0,step-1)] if step > 0 else config.get('TEAM_COHESION_BASELINE',0.7)*100
        ps_now += (current_cohesion - config.get('TEAM_COHESION_BASELINE',0.7)*100) * config.get('TEAM_COHESION_IMPACT_ON_PSYCH_SAFETY', 0.15)
        _psych_safety_scores[step] = np.clip(ps_now, 15, 100)

        cohesion_now = _team_cohesion_scores[max(0,step-1)] if step > 0 else config.get('TEAM_COHESION_BASELINE', 0.7) * 100
        cohesion_now -= 0.1 + active_disruption_intensity * 0.5 
        if _psych_safety_scores[step] > 75 : cohesion_now += 0.5 
        if collab_score_prev > 70 : cohesion_now += 0.5 
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,4) if num_steps > 3 else 1) == 0: cohesion_now += 3
        _team_cohesion_scores[step] = np.clip(cohesion_now, 20, 100)

        if _wellbeing_scores[step] < config.get('WELLBEING_ALERT_THRESHOLD', 0.6)*100: wellbeing_triggers_dict['threshold'].append(step)
        if step > 0 and _wellbeing_scores[step] < _wellbeing_scores[step-1] - 10: wellbeing_triggers_dict['trend'].append(step) 
        if is_new_disruption_starting_this_step and _wellbeing_scores[step] < (_wellbeing_scores[max(0,step-1)] if step > 0 else _wellbeing_scores[0]) * 0.95 :
            if num_team_members > 0:
                affected_zones_this_step = [worker_assigned_zone[i] for i in range(num_team_members) if worker_fatigue[i] > 0.6 or random.random() < active_disruption_intensity*0.1] 
                for zone_name_affected in set(affected_zones_this_step):
                    if zone_name_affected in wellbeing_triggers_dict['work_area']: # Ensure zone exists as key
                        wellbeing_triggers_dict['work_area'][zone_name_affected].append(step)
            
        current_uptime = _uptime_percent[max(0,step-1)] if step > 0 else 100
        if random.random() < config.get('EQUIPMENT_FAILURE_PROB_PER_INTERVAL', 0.005): current_uptime -= random.uniform(10,25) 
        current_uptime -= active_disruption_intensity * 25 
        _uptime_percent[step] = np.clip(current_uptime, 30, 100)
        
        base_compliance_val = config.get('BASE_TASK_COMPLETION_PROB',0.95) * 100
        compliance_now = base_compliance_val * (1 - avg_fatigue * config.get('FATIGUE_IMPACT_ON_COMPLIANCE', 0.15))
        # Average complexity from actual worker assignments (if available and num_team_members > 0)
        avg_worker_task_complexity = 0.5 # Default complexity
        if num_team_members > 0 and worker_assigned_zone:
             avg_worker_task_complexity = np.mean([work_areas_config.get(worker_assigned_zone[w], {}).get('task_complexity', 0.5) for w in range(num_team_members)])
        
        compliance_now -= avg_worker_task_complexity * config.get('COMPLEXITY_IMPACT_ON_COMPLIANCE',0.2) * 100 
        compliance_now *= (_psych_safety_scores[step]/100 * 0.2 + 0.8) 
        compliance_now = max(config.get('MIN_COMPLIANCE_DURING_DISRUPTION',30), compliance_now * (1 - active_disruption_intensity * config.get('DISRUPTION_COMPLIANCE_REDUCTION_FACTOR',0.5)))
        _task_compliance_scores[step] = np.clip(compliance_now, 0, 100)

        _quality_rate_percent[step] = np.clip( (100 - config.get('BASE_QUALITY_DEFECT_RATE',0.02)*100) * (_task_compliance_scores[step]/100)**1.2 , 40, 100) 
        _quality_rate_percent[step] -= active_disruption_intensity * 10

        max_potential_throughput_facility = config.get('THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL',100) 
        effective_capacity_factor = (_uptime_percent[step]/100) * (_task_compliance_scores[step]/100) * (1 - avg_fatigue * 0.5) * (1 - _perceived_workload_scores[step]/25) 
        actual_units_produced = max_potential_throughput_facility * effective_capacity_factor * (1 - active_disruption_intensity * 0.75) 
        _throughput_percent_of_max[step] = np.clip((actual_units_produced / (max_potential_throughput_facility  + 1e-6) ) * 100, 0, 100) # Avoid div by zero
        _task_completion_rate_percent[step] = _throughput_percent_of_max[step] 

        current_downtime_this_interval = 0
        if is_new_disruption_starting_this_step and random.random() < config.get('DOWNTIME_FROM_DISRUPTION_EVENT_PROB', 0.5):
             current_downtime_this_interval += max(0, np.random.normal(config.get('DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',8), config.get('DOWNTIME_STD_MINUTES_PER_OCCURRENCE',4)))
        if _uptime_percent[step] < 60 and random.random() < config.get('DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB', 0.7): 
             current_downtime_this_interval += max(0, np.random.normal(config.get('DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',8)*0.6, config.get('DOWNTIME_STD_MINUTES_PER_OCCURRENCE',4)*0.4))
        _downtime_minutes_per_interval[step] = np.clip(current_downtime_this_interval, 0, (config.get('SHIFT_DURATION_MINUTES', 480) / num_steps if num_steps > 0 else 2) * 2) # Max downtime is interval length (2 mins per step * interval duration in steps)

        current_oee_calc = (_uptime_percent[step]/100) * (_throughput_percent_of_max[step]/100) * (_quality_rate_percent[step]/100)
        if active_disruption_intensity == 0 and step > (current_disruption_end_step if current_disruption_end_step > 0 else -1): 
            prev_recovery = _operational_recovery_scores[max(0,step-1)]; target_potential_recovery = current_oee_calc * 100
            _operational_recovery_scores[step] = np.clip(prev_recovery + (target_potential_recovery - prev_recovery) * (1 - math.exp(-1/config.get('RECOVERY_HALFLIFE_INTERVALS', 10))), 0, 100)
        else: _operational_recovery_scores[step] = np.clip(current_oee_calc * 100, 0, 100)
        _productivity_loss_percent[step] = np.clip(100 - _operational_recovery_scores[step] + np.random.normal(0,1), 0, 100)

        base_collab = 60 + (_team_cohesion_scores[step]-70)*0.3 - (_perceived_workload_scores[step]-5)*2 # Cohesion & workload influence base collab
        base_collab -= active_disruption_intensity * 15
        _collaboration_scores[step] = np.clip(base_collab + np.random.normal(0,3) ,10, 95)

        if num_team_members > 0 : # only run if there are team members
            for i in range(num_team_members):
                zone_details = work_areas_config.get(worker_assigned_zone[i], {}); zone_coords = zone_details.get('coords')
                if zone_coords and len(zone_coords) == 2: (zx0, zy0), (zx1, zy1) = zone_coords; target_x, target_y = (zx0+zx1)/2, (zy0+zy1)/2
                else: target_x, target_y = facility_width/2, facility_height/2 
                move_x = (target_x - worker_current_x[i]) * 0.2 + np.random.normal(0, 0.8); move_y = (target_y - worker_current_y[i]) * 0.2 + np.random.normal(0, 0.8)
                worker_current_x[i] = np.clip(worker_current_x[i] + move_x, 0, facility_width); worker_current_y[i] = np.clip(worker_current_y[i] + move_y, 0, facility_height)
                status_now = 'working';
                if worker_fatigue[i] > 0.75: status_now = 'fatigued'
                elif worker_fatigue[i] > 0.9: status_now = 'exhausted' 
                if active_disruption_intensity > 0.6 : status_now = 'disrupted'
                if team_initiative == "More frequent breaks" and step > 0 and num_steps > 5: # Ensure num_steps is not too small
                     if step % (num_steps // (random.randint(3,5)+1)) == (i % (num_steps // (random.randint(3,5)+1))): status_now = 'break'
                team_positions_data.append({'step': step, 'worker_id': i, 'x': worker_current_x[i], 'y': worker_current_y[i], 'z': np.random.uniform(0, 0.1), 'zone': worker_assigned_zone[i], 'status': status_now})
    
    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame()
    task_compliance = {'data': list(_task_compliance_scores), 'z_scores': list(np.random.normal(0,0.5,num_steps)), 'forecast': [max(0,s-random.uniform(1,5)) for s in _task_compliance_scores]}
    collaboration_proximity = {'data': list(_collaboration_scores), 'forecast': [min(100,s+random.uniform(1,5)) for s in _collaboration_scores]}
    operational_recovery = list(_operational_recovery_scores)
    efficiency_df_data = {'uptime': list(_uptime_percent), 'throughput': list(_throughput_percent_of_max), 'quality': list(_quality_rate_percent)}
    efficiency_df = pd.DataFrame(efficiency_df_data); efficiency_df['oee'] = np.clip((efficiency_df['uptime']/100 * efficiency_df['throughput']/100 * efficiency_df['quality']/100) * 100, 0, 100)
    productivity_loss = list(_productivity_loss_percent)
    worker_wellbeing = {'scores': list(_wellbeing_scores), 'triggers': wellbeing_triggers_dict, 'team_cohesion_scores': list(_team_cohesion_scores), 'perceived_workload_scores': list(_perceived_workload_scores)} 
    psychological_safety = list(_psych_safety_scores)
    feedback_impact = list(np.random.choice([-0.1, -0.05, 0, 0.05, 0.1], num_steps, p=[0.1,0.2,0.4,0.2,0.1])) 
    downtime_minutes = list(_downtime_minutes_per_interval)
    task_completion_rate = list(_task_completion_rate_percent)
    return (team_positions_df, task_compliance, collaboration_proximity, operational_recovery, efficiency_df, productivity_loss, worker_wellbeing, psychological_safety, feedback_impact, downtime_minutes, task_completion_rate)
