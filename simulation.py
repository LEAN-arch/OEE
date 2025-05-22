# simulation.py
import numpy as np
import pandas as pd
import random
import math

def simulate_workplace_operations(num_team_members, num_steps, disruption_event_steps, team_initiative, config):
    np.random.seed(42); random.seed(42)

    facility_width, facility_height = config.get('FACILITY_SIZE', (100, 80))
    work_areas_config = config.get('WORK_AREAS', {})
    
    # --- Initialize State Variables ---
    _task_compliance_scores = np.zeros(num_steps)
    _collaboration_scores = np.zeros(num_steps)
    _operational_recovery_scores = np.zeros(num_steps)
    _wellbeing_scores = np.zeros(num_steps)
    _psych_safety_scores = np.zeros(num_steps)
    _productivity_loss_percent = np.zeros(num_steps)
    _downtime_minutes_per_interval = np.zeros(num_steps)
    _task_completion_rate_actual = np.zeros(num_steps) # Actual tasks completed
    _task_completion_rate_percent = np.zeros(num_steps) # As a percentage of potential

    _uptime_percent = np.ones(num_steps) * 100 # Start at 100%
    _quality_rate_percent = np.ones(num_steps) * 100
    _throughput_percent_of_max = np.zeros(num_steps) # OEE throughput component

    # Worker-specific states (more advanced, but can average for aggregate metrics)
    worker_fatigue = np.zeros(num_team_members) # Scale 0-1
    worker_zone = [random.choice(list(work_areas_config.keys())) if work_areas_config else "DefaultZone" for _ in range(num_team_members)]
    # worker_task_complexity = np.array([work_areas_config.get(z, {}).get('task_complexity', 0.5) for z in worker_zone])
    
    # Disruption state
    active_disruption_intensity = 0.0 # 0 to 1
    disruption_linger_factor = 0.85 # How much effect lingers per step after duration

    # --- Initialize first step ---
    _wellbeing_scores[0] = config.get('INITIAL_WELLBEING_MEAN', 0.8) * 100
    _psych_safety_scores[0] = config.get('PSYCH_SAFETY_INITIAL_MEAN', 0.75) * 100
    _operational_recovery_scores[0] = 100 # Starts perfect
    # Others will be calculated in loop

    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa:[] for wa in work_areas_config}, 'disruption': []}

    # --- Simulation Loop ---
    for step in range(num_steps):
        # --- Disruption Management ---
        current_disruption_this_step = False
        if step in disruption_event_steps:
            active_disruption_intensity = 1.0 # Full impact starts
            current_disruption_this_step = True
            wellbeing_triggers_dict['disruption'].append(step)
        elif active_disruption_intensity > 0:
            active_disruption_intensity *= disruption_linger_factor # Effect lingers and fades
            if active_disruption_intensity < 0.05: active_disruption_intensity = 0 # Consider it ended
        
        # --- Worker Level Updates (simplified for aggregate metrics) ---
        # Fatigue
        fatigue_rate = config.get('WELLBEING_FATIGUE_RATE_PER_INTERVAL', 0.002)
        if team_initiative == "More frequent breaks": fatigue_rate *= (1 - config.get('INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.3))
        worker_fatigue += fatigue_rate
        worker_fatigue = np.clip(worker_fatigue, 0, 1.0) # Cap fatigue at 1
        avg_fatigue = np.mean(worker_fatigue)

        # Well-being
        wb_now = _wellbeing_scores[max(0, step-1)] # Previous step's wellbeing
        wb_now -= fatigue_rate * 15 # Fatigue directly reduces wellbeing points
        wb_now -= active_disruption_intensity * config.get('DISRUPTION_WELLBEING_DROP', 0.2) * 10 # Disruptions hit hard
        if team_initiative == "More frequent breaks" and step > 0 and step % (num_steps // random.randint(4,6)) == 0 : # Simulating breaks
            wb_now += config.get('INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST_ABS', 0.05) * 100
            worker_fatigue *= 0.7 # Breaks reduce fatigue
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,4)) == 0:
            wb_now += config.get('INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS', 0.05) * 100
        _wellbeing_scores[step] = np.clip(wb_now, 10, 100) # Min wellbeing is 10%

        # Psychological Safety
        ps_now = _psych_safety_scores[max(0, step-1)]
        ps_now -= config.get('PSYCH_SAFETY_EROSION_RATE_PER_INTERVAL', 0.0005) * 100
        if active_disruption_intensity > 0.6: ps_now -= 2.5 # High intensity disruption impacts safety perception
        if team_initiative == "Team recognition" and step > 0 and step % (num_steps // random.randint(2,4)) == 1:
            ps_now += config.get('INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS', 0.08) * 100
        _psych_safety_scores[step] = np.clip(ps_now, 20, 100) # Min psych safety

        # Trigger checks for wellbeing
        if _wellbeing_scores[step] < config.get('WELLBEING_ALERT_THRESHOLD', 0.6)*100: wellbeing_triggers_dict['threshold'].append(step)
        if step > 0 and _wellbeing_scores[step] < _wellbeing_scores[step-1] - 10: wellbeing_triggers_dict['trend'].append(step) # More sensitive trend drop
        if current_disruption_this_step and _wellbeing_scores[step] < _wellbeing_scores[max(0,step-1)] * 0.9 : # If wellbeing dropped due to this step's disruption
            # Find current dominant zone (simplified)
            dominant_zone = random.choice(list(work_areas_config.keys())) if work_areas_config else "UnknownZone"
            if dominant_zone not in wellbeing_triggers_dict['work_area']: wellbeing_triggers_dict['work_area'][dominant_zone] = []
            wellbeing_triggers_dict['work_area'][dominant_zone].append(step)
            
        # --- Operational Metrics ---
        # Uptime: starts high, can drop due to equipment failure or severe disruption
        current_uptime = _uptime_percent[max(0,step-1)]
        if random.random() < config.get('EQUIPMENT_FAILURE_PROB_PER_INTERVAL', 0.005): current_uptime -= random.uniform(5,15)
        current_uptime -= active_disruption_intensity * 10 # Disruptions can impact uptime
        _uptime_percent[step] = np.clip(current_uptime, 50, 100) # Min uptime 50%
        if _uptime_percent[step] < 80 and random.random() < config.get('DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB', 0.7):
            _downtime_minutes_per_interval[step] += max(0, np.random.normal(config.get('DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',8)*0.5, config.get('DOWNTIME_STD_MINUTES_PER_OCCURRENCE',4)*0.5))


        # Task Compliance: Affected by fatigue, psych safety (more confident workers), and disruptions
        compliance = config.get('BASE_TASK_COMPLETION_PROB',0.95) * 100
        compliance -= avg_fatigue * config.get('FATIGUE_IMPACT_ON_COMPLIANCE', 0.15) * 100
        compliance += (_psych_safety_scores[step] - config.get('PSYCH_SAFETY_BASELINE',0.75)*100) * 0.1 # Positive psych safety slightly helps
        compliance -= active_disruption_intensity * config.get('DISRUPTION_COMPLIANCE_REDUCTION_FACTOR',0.5) * 50 # Disruptions have strong effect
        _task_compliance_scores[step] = np.clip(compliance, config.get('MIN_COMPLIANCE_DURING_DISRUPTION',30), 100)

        # Quality Rate: Strongly tied to compliance
        _quality_rate_percent[step] = np.clip(100 - (100 - _task_compliance_scores[step])*1.2 - config.get('BASE_QUALITY_DEFECT_RATE',0.02)*100 , 60, 100) # Errors in compliance amplify quality issues

        # Throughput (% of max): Affected by uptime, compliance, and direct disruption effects
        throughput_potential = config.get('THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL',100) # Max units for facility
        actual_production = throughput_potential * (_uptime_percent[step]/100) * (_task_compliance_scores[step]/100) * (1-avg_fatigue*0.3)
        actual_production *= (1 - active_disruption_intensity * 0.4) # Direct hit on throughput from disruption
        _throughput_percent_of_max[step] = np.clip((actual_production / throughput_potential) * 100 if throughput_potential > 0 else 0, 0, 100)
        _task_completion_rate_actual[step] = actual_production # Or use task-based metric if tasks are explicitly modelled

        # Task Completion Rate (%)
        # Simplified: proportion of throughput relative to max, adjusted for disruptions
        _task_completion_rate_percent[step] = _throughput_percent_of_max[step] 

        # Downtime from Major Disruptions
        if current_disruption_this_step and random.random() < config.get('DOWNTIME_FROM_DISRUPTION_EVENT_PROB', 0.5):
             _downtime_minutes_per_interval[step] += max(0, np.random.normal(config.get('DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',8), config.get('DOWNTIME_STD_MINUTES_PER_OCCURRENCE',4)))
        _downtime_minutes_per_interval[step] = np.clip(_downtime_minutes_per_interval[step], 0, 2 * 60) # Cap downtime per interval (e.g., 2 hours)


        # Operational Recovery
        # Score is high if OEE components are high, lower otherwise
        oee_now = (_uptime_percent[step]/100) * (_throughput_percent_of_max[step]/100) * (_quality_rate_percent[step]/100)
        if active_disruption_intensity == 0 and step > (disruption_event_steps[-1] if disruption_event_steps else -1): # If recovering after ALL disruptions passed
            # Gradual recovery of overall operational score (OEE represents this)
            prev_recovery = _operational_recovery_scores[max(0, step-1)]
            target_recovery = oee_now * 100
            _operational_recovery_scores[step] = np.clip(prev_recovery + (target_recovery - prev_recovery) * (1 - math.exp(-1/config.get('RECOVERY_HALFLIFE_INTERVALS', 10))), 0, 100)
        else: # During disruption or before any, recovery is tied to current OEE
            _operational_recovery_scores[step] = np.clip(oee_now * 100, 0, 100)

        _productivity_loss_percent[step] = 100 - _operational_recovery_scores[step]

        # --- Update Team Positions (Simplified) ---
        for i in range(num_team_members):
            # Rudimentary movement - can be expanded
            worker_current_x[i] = np.clip(worker_current_x[i] + np.random.normal(0,1), 0, facility_width)
            worker_current_y[i] = np.clip(worker_current_y[i] + np.random.normal(0,1), 0, facility_height)
            team_positions_data.append({'step': step, 'worker_id': i, 'x': worker_current_x[i], 'y': worker_current_y[i], 'z': 0, 'zone': worker_assigned_zone[i], 'status': 'working'})


    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame(columns=['step', 'worker_id', 'x', 'y', 'z', 'zone', 'status'])

    # --- Assemble Final Data Structures for Return ---
    task_compliance = {'data': list(_task_compliance_scores), 'z_scores': list(np.random.normal(0,0.5,num_steps)), 'forecast': [max(0,s-random.uniform(1,5)) for s in _task_compliance_scores]}
    collaboration_proximity = {'data': list(_collaboration_scores), 'forecast': [min(100,s+random.uniform(1,5)) for s in _collaboration_scores]}
    operational_recovery = list(_operational_recovery_scores)
    
    efficiency_df_data = {'uptime': list(_uptime_percent), 'throughput': list(_throughput_percent_of_max), 'quality': list(_quality_rate_percent)}
    efficiency_df = pd.DataFrame(efficiency_df_data)
    efficiency_df['oee'] = np.clip((efficiency_df['uptime']/100 * efficiency_df['throughput']/100 * efficiency_df['quality']/100) * 100, 0, 100)

    productivity_loss = list(_productivity_loss_percent)
    worker_wellbeing = {'scores': list(_wellbeing_scores), 'triggers': wellbeing_triggers_dict}
    psychological_safety = list(_psych_safety_scores)
    feedback_impact = list(np.random.choice([-0.5, -0.2, 0, 0.2, 0.5], num_steps, p=[0.1,0.2,0.4,0.2,0.1])) # More structured dummy feedback
    downtime_minutes = list(_downtime_minutes_per_interval)
    task_completion_rate = list(_task_completion_rate_percent)

    return (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
            efficiency_df, productivity_loss, worker_wellbeing, psychological_safety,
            feedback_impact, downtime_minutes, task_completion_rate)
