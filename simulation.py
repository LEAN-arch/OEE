# simulation.py
import numpy as np
import pandas as pd
import random
import math

def simulate_workplace_operations(num_team_members, num_steps, disruption_event_steps, team_initiative, config):
    np.random.seed(42); random.seed(42) # For reproducibility

    facility_width, facility_height = config.get('FACILITY_SIZE', (100, 80))
    work_areas_config = config.get('WORK_AREAS', {})
    
    # --- Initialize State Variables for Time Series Metrics ---
    _task_compliance_scores = np.zeros(num_steps)
    _collaboration_scores = np.zeros(num_steps)
    _operational_recovery_scores = np.zeros(num_steps)
    _wellbeing_scores = np.zeros(num_steps)
    _psych_safety_scores = np.zeros(num_steps)
    _productivity_loss_percent = np.zeros(num_steps)
    _downtime_minutes_per_interval = np.zeros(num_steps)
    _task_completion_rate_percent = np.zeros(num_steps)
    _uptime_percent = np.ones(num_steps) * 100 
    _quality_rate_percent = np.ones(num_steps) * 100
    _throughput_percent_of_max = np.zeros(num_steps) 

    # --- Initialize Worker-Specific States (these persist and update across steps) ---
    team_positions_data = [] # To build the DataFrame
    worker_current_x = np.random.uniform(0, facility_width, num_team_members)
    worker_current_y = np.random.uniform(0, facility_height, num_team_members)
    # Assign initial zones based on worker distribution in config, then fallback
    # This is a simplified assignment; a more complex one might consider worker capacities per zone
    configured_workers_per_zone = {
        zn: zd.get('workers', 0) for zn, zd in work_areas_config.items()
    }
    worker_assigned_zone = []
    temp_worker_idx = 0
    if sum(configured_workers_per_zone.values()) == num_team_members : # If config perfectly matches
        for zone_name, num_w in configured_workers_per_zone.items():
            worker_assigned_zone.extend([zone_name] * num_w)
    else: # Fallback to random or even distribution if config doesn't match
        zone_keys = list(work_areas_config.keys())
        if not zone_keys: zone_keys = ["DefaultZone"]
        for i in range(num_team_members):
             worker_assigned_zone.append(zone_keys[i % len(zone_keys)])
    
    random.shuffle(worker_assigned_zone) # Shuffle if initial assignment was sequential

    worker_fatigue = np.random.uniform(0.0, 0.1, num_team_members) # Start with slight initial fatigue (0-1 scale)

    # Disruption state
    active_disruption_intensity = 0.0 
    disruption_linger_factor = 0.85 
    current_disruption_end_step = -1 # Track when the current active disruption effect should end

    # --- Initialize first step values ---
    _wellbeing_scores[0] = config.get('INITIAL_WELLBEING_MEAN', 0.8) * 100
    _psych_safety_scores[0] = config.get('PSYCH_SAFETY_INITIAL_MEAN', 0.75) * 100
    _operational_recovery_scores[0] = 100 
    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa:[] for wa in work_areas_config}, 'disruption': []}

    # --- Main Simulation Loop ---
    for step in range(num_steps):
        # --- Disruption Management ---
        is_new_disruption_starting_this_step = False
        if step in disruption_event_steps:
            active_disruption_intensity = 1.0 
            is_new_disruption_starting_this_step = True
            wellbeing_triggers_dict['disruption'].append(step)
            # Determine duration for this specific disruption event
            current_disruption_duration = max(1, int(round(np.random.normal(
                config.get('DISRUPTION_DURATION_MEAN_INTERVALS', 5), 
                config.get('DISRUPTION_DURATION_STD_INTERVALS', 2)
            ))))
            current_disruption_end_step = step + current_disruption_duration
        
        # Fade effect if a disruption period is over but intensity still high
        if step >= current_disruption_end_step and active_disruption_intensity > 0:
            active_disruption_intensity *= disruption_linger_factor
            if active_disruption_intensity < 0.05: active_disruption_intensity = 0.0; current_disruption_end_step = -1 # Reset
        
        # --- Worker Level Updates (simplified for aggregate metrics) ---
        fatigue_rate_this_step = config.get('WELLBEING_FATIGUE_RATE_PER_INTERVAL', 0.002)
        if team_initiative == "More frequent breaks": fatigue_rate_this_step *= (1 - config.get('INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.3))
        worker_fatigue += fatigue_rate_this_step * (1 + active_disruption_intensity * 0.5) # Disruptions can accelerate fatigue
        worker_fatigue = np.clip(worker_fatigue, 0, 1.0)
        avg_fatigue = np.mean(worker_fatigue)

        wb_now = _wellbeing_scores[max(0, step-1)] if step > 0 else config.get('INITIAL_WELLBEING_MEAN', 0.8) * 100
        wb_now -= (avg_fatigue * 10) # More direct impact of avg_fatigue on wellbeing score reduction
        wb_now -= active_disruption_intensity * config.get('DISRUPTION_WELLBEING_DROP', 0.2) * 15 # Stronger impact if active
        if team_initiative == "More frequent breaks" and step > 0 and step % (num_steps // random.randint(4,6)) == 0 : 
             wb_now += config.get('INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST_ABS', 0.05) * 100
             worker_fatigue *= 0.6 # Breaks reduce fatigue significantly
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,4)) == 0:
            wb_now += config.get('INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS', 0.05) * 100
        _wellbeing_scores[step] = np.clip(wb_now, 10, 100)

        ps_now = _psych_safety_scores[max(0, step-1)] if step > 0 else config.get('PSYCH_SAFETY_INITIAL_MEAN', 0.75) * 100
        ps_now -= config.get('PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL', 0.0005) * 100 
        if active_disruption_intensity > 0.6: ps_now -= 3 
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,4)) == 1:
             ps_now += config.get('INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS', 0.08) * 100
        _psych_safety_scores[step] = np.clip(ps_now, 20, 100)

        if _wellbeing_scores[step] < config.get('WELLBEING_ALERT_THRESHOLD', 0.6)*100: wellbeing_triggers_dict['threshold'].append(step)
        if step > 0 and _wellbeing_scores[step] < _wellbeing_scores[step-1] - 10: wellbeing_triggers_dict['trend'].append(step) 
        if is_new_disruption_starting_this_step and _wellbeing_scores[step] < (_wellbeing_scores[max(0,step-1)] if step > 0 else _wellbeing_scores[0]) * 0.95 :
            current_worker_zones = [worker_assigned_zone[i] for i in range(num_team_members)] # Get current zones
            most_affected_zone = max(set(current_worker_zones), key=current_worker_zones.count) if current_worker_zones else "UnknownZone"
            if most_affected_zone not in wellbeing_triggers_dict['work_area']: wellbeing_triggers_dict['work_area'][most_affected_zone] = []
            wellbeing_triggers_dict['work_area'][most_affected_zone].append(step)
            
        # --- Operational Metrics ---
        current_uptime = _uptime_percent[max(0,step-1)] if step > 0 else 100
        if random.random() < config.get('EQUIPMENT_FAILURE_PROB_PER_INTERVAL', 0.005): current_uptime -= random.uniform(5,15) # Equipment hiccup
        current_uptime -= active_disruption_intensity * 15 # Disruptions reduce available uptime
        _uptime_percent[step] = np.clip(current_uptime, 40, 100) # Min uptime cannot be 0 if still producing
        
        base_compliance_val = config.get('BASE_TASK_COMPLETION_PROB',0.95) * 100
        compliance_now = base_compliance_val * (1 - avg_fatigue * config.get('FATIGUE_IMPACT_ON_COMPLIANCE', 0.15))
        compliance_now *= (1 - np.mean([work_areas_config.get(z,{}).get('task_complexity',0.5) for z in worker_assigned_zone]) * config.get('COMPLEXITY_IMPACT_ON_COMPLIANCE',0.2)) # Avg complexity impact
        compliance_now = max(config.get('MIN_COMPLIANCE_DURING_DISRUPTION',30), compliance_now * (1 - active_disruption_intensity * config.get('DISRUPTION_COMPLIANCE_REDUCTION_FACTOR',0.5)))
        _task_compliance_scores[step] = np.clip(compliance_now, 0, 100)

        _quality_rate_percent[step] = np.clip( (100 - config.get('BASE_QUALITY_DEFECT_RATE',0.02)*100) * (_task_compliance_scores[step]/100)**1.5 , 50, 100) # Quality highly dependent on compliance (exponentiated effect)
        _quality_rate_percent[step] -= active_disruption_intensity * 5 # Further quality drop during disruptions

        max_potential_throughput_facility = config.get('THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL',100)
        effective_capacity_factor = (_uptime_percent[step]/100) * (_task_compliance_scores[step]/100) * (1 - avg_fatigue * 0.4) # Fatigue also reduces speed/throughput
        actual_units_produced = max_potential_throughput_facility * effective_capacity_factor * (1 - active_disruption_intensity * 0.6) # Stronger hit
        _throughput_percent_of_max[step] = np.clip((actual_units_produced / max_potential_throughput_facility) * 100 if max_potential_throughput_facility > 0 else 0, 0, 100)
        _task_completion_rate_percent[step] = _throughput_percent_of_max[step] 

        # Downtime
        current_downtime_this_interval = 0
        if is_new_disruption_starting_this_step and random.random() < config.get('DOWNTIME_FROM_DISRUPTION_EVENT_PROB', 0.5):
             current_downtime_this_interval += max(0, np.random.normal(config.get('DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',8), config.get('DOWNTIME_STD_MINUTES_PER_OCCURRENCE',4)))
        if _uptime_percent[step] < 70 and random.random() < config.get('DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB', 0.7): # Low uptime due to failure can cause downtime
             current_downtime_this_interval += max(0, np.random.normal(config.get('DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',8)*0.75, config.get('DOWNTIME_STD_MINUTES_PER_OCCURRENCE',4)*0.5))
        _downtime_minutes_per_interval[step] = np.clip(current_downtime_this_interval, 0, 120) # Max downtime per interval capped (e.g., 2-min intervals shouldn't have > 2 min)

        # Operational Recovery (ability to maintain high OEE or recover quickly)
        current_oee_calc = (_uptime_percent[step]/100) * (_throughput_percent_of_max[step]/100) * (_quality_rate_percent[step]/100)
        if active_disruption_intensity < 0.1 and not is_new_disruption_starting_this_step : # If recovering
            prev_recovery = _operational_recovery_scores[max(0,step-1)]
            target_potential_recovery = current_oee_calc * 100 # Ideal recovery point based on current capability
            _operational_recovery_scores[step] = np.clip(prev_recovery + (target_potential_recovery - prev_recovery) * 0.2, 0, 100) # Recovers 20% of the gap each step
        else: # During disruption, or stable state
            _operational_recovery_scores[step] = np.clip(current_oee_calc * 100, 0, 100)
        
        _productivity_loss_percent[step] = np.clip(100 - _operational_recovery_scores[step] + np.random.normal(0,2), 0, 100)


        # --- Update Team Positions for DataFrame ---
        # This loop is just for generating the positions DataFrame, actual simulation logic uses aggregated/averaged worker states above
        for i in range(num_team_members):
            # Movement logic: biased towards assigned zone, random jitter
            zone_details = work_areas_config.get(worker_assigned_zone[i], {})
            zone_coords = zone_details.get('coords')
            if zone_coords and len(zone_coords) == 2:
                (zx0, zy0), (zx1, zy1) = zone_coords
                target_x, target_y = (zx0+zx1)/2, (zy0+zy1)/2
            else:
                target_x, target_y = facility_width/2, facility_height/2 # Default target

            move_x = (target_x - worker_current_x[i]) * 0.1 + np.random.normal(0, 0.5)
            move_y = (target_y - worker_current_y[i]) * 0.1 + np.random.normal(0, 0.5)
            
            worker_current_x[i] = np.clip(worker_current_x[i] + move_x, 0, facility_width)
            worker_current_y[i] = np.clip(worker_current_y[i] + move_y, 0, facility_height)
            
            # Determine status based on current simulation state for this worker (simplified)
            status_now = 'working'
            if worker_fatigue[i] > 0.7: status_now = 'fatigued'
            if active_disruption_intensity > 0.5 : status_now = 'disrupted'
            if step > 0 and step % (num_steps // random.randint(4,6)) == 0 and team_initiative == "More frequent breaks": status_now = 'break'


            team_positions_data.append({
                'step': step, 'worker_id': i, 
                'x': worker_current_x[i], 'y': worker_current_y[i], 'z': np.random.uniform(0, 0.5), # Small z variation
                'zone': worker_assigned_zone[i], # Assigned zone for simplicity of coloring
                'status': status_now
            })
    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame()

    # --- Assemble final data structures for Return ---
    task_compliance = {'data': list(_task_compliance_scores), 'z_scores': list(np.random.normal(0,0.5,num_steps)), 'forecast': [max(0,s-random.uniform(1,5)) for s in _task_compliance_scores]}
    collaboration_proximity = {'data': list(_collaboration_scores), 'forecast': [min(100,s+random.uniform(1,5)) for s in _collaboration_scores]}
    operational_recovery = list(_operational_recovery_scores)
    efficiency_df_data = {'uptime': list(_uptime_percent), 'throughput': list(_throughput_percent_of_max), 'quality': list(_quality_rate_percent)}
    efficiency_df = pd.DataFrame(efficiency_df_data)
    efficiency_df['oee'] = np.clip((efficiency_df['uptime']/100 * efficiency_df['throughput']/100 * efficiency_df['quality']/100) * 100, 0, 100)
    productivity_loss = list(_productivity_loss_percent)
    worker_wellbeing = {'scores': list(_wellbeing_scores), 'triggers': wellbeing_triggers_dict}
    psychological_safety = list(_psych_safety_scores)
    feedback_impact = list(np.random.choice([-0.1, -0.05, 0, 0.05, 0.1], num_steps, p=[0.1,0.2,0.4,0.2,0.1])) # Scale down feedback
    downtime_minutes = list(_downtime_minutes_per_interval)
    task_completion_rate = list(_task_completion_rate_percent)

    return (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
            efficiency_df, productivity_loss, worker_wellbeing, psychological_safety,
            feedback_impact, downtime_minutes, task_completion_rate)
