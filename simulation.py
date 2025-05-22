# simulation.py
import numpy as np
import pandas as pd
import random
import math

def simulate_workplace_operations(num_team_members, num_steps, disruption_event_steps, team_initiative, config):
    np.random.seed(42); random.seed(42)

    # --- Extract relevant config parameters ---
    facility_width, facility_height = config.get('FACILITY_SIZE', (100, 80))
    work_areas_config = config.get('WORK_AREAS', {})
    
    # --- Initialize state variables ---
    # These will store the time series data for each metric
    # Score type metrics (0-100 or 0-1 scale converted later)
    _task_compliance_scores = np.ones(num_steps) * config.get('BASE_TASK_COMPLETION_PROB', 0.95) * 100
    _collaboration_scores = np.ones(num_steps) * 60 # Start at a baseline
    _operational_recovery_scores = np.ones(num_steps) * 100 # Starts high, drops with disruptions
    _wellbeing_scores = np.ones(num_steps) * config.get('INITIAL_WELLBEING_MEAN', 0.8) * 100
    _psych_safety_scores = np.ones(num_steps) * config.get('PSYCH_SAFETY_INITIAL_MEAN', 0.75) * 100
    
    # Rate/Loss type metrics
    _productivity_loss_percent = np.zeros(num_steps)
    _downtime_minutes_per_interval = np.zeros(num_steps)
    _task_completion_rate = np.ones(num_steps) * 90 # Percentage of 'possible' tasks completed

    # OEE components (0-100 scale)
    _uptime_percent = np.ones(num_steps) * config.get('EQUIPMENT_UPTIME_MEAN', 0.95) * 100
    _quality_rate_percent = np.ones(num_steps) * config.get('BASE_QUALITY_RATE', 0.98) * 100
    _throughput_actual = np.zeros(num_steps) # Actual units produced

    # Intermediate simulation states
    current_fatigue_level = np.zeros(num_team_members) # 0 to 1
    active_disruption_effect = 0 # Factor from 0 to 1 indicating current disruption impact
    disruption_end_step = -1

    # Well-being triggers
    wellbeing_triggers_dict = {'threshold': [], 'trend': [], 'work_area': {wa:[] for wa in work_areas_config}, 'disruption': []}

    # --- Team Positions DataFrame (Simplified) ---
    team_positions_data = []
    # Initial positions could be more strategic (e.g., in assigned work areas)
    worker_current_x = np.random.uniform(0, facility_width, num_team_members)
    worker_current_y = np.random.uniform(0, facility_height, num_team_members)
    worker_assigned_zone = [random.choice(list(work_areas_config.keys())) if work_areas_config else "DefaultZone" for _ in range(num_team_members)]


    # --- Main Simulation Loop ---
    for step in range(num_steps):
        # --- Update Disruption State ---
        if step in disruption_event_steps: # A major configured disruption event starts
            active_disruption_effect = 1.0 # Full impact
            disruption_end_step = step + np.random.normal(
                config.get('DISRUPTION_DURATION_MEAN_INTERVALS', 5), 
                config.get('DISRUPTION_DURATION_STD_INTERVALS', 2)
            )
            disruption_end_step = max(step + 1, int(round(disruption_end_step))) # Must last at least one step
            wellbeing_triggers_dict['disruption'].append(step)
        
        if step >= disruption_end_step: # Disruption ends, recovery starts
            active_disruption_effect = max(0, active_disruption_effect * (1 - config.get('RECOVERY_RATE_PER_INTERVAL', 0.1))) # Exponential decay
        elif active_disruption_effect > 0 and step < disruption_end_step : # Still in active disruption period
            pass # Effect remains high or decays slightly if modeled that way
        elif active_disruption_effect > 0 and step >= disruption_end_step: # Post-disruption recovery
             active_disruption_effect = max(0, active_disruption_effect * (1 - config.get('RECOVERY_RATE_PER_INTERVAL', 0.1)))


        # --- Worker State Updates & Metric Calculations for current step ---
        # Well-being & Fatigue
        fatigue_increase = config.get('WELLBEING_FATIGUE_RATE_PER_INTERVAL', 0.002)
        if team_initiative == "More frequent breaks": fatigue_increase *= (1 - config.get('INITIATIVE_BREAKS_FATIGUE_REDUCTION_FACTOR', 0.3))
        current_fatigue_level += fatigue_increase
        current_fatigue_level = np.clip(current_fatigue_level, 0, 0.8) # Cap fatigue effect

        # Simple well-being model: baseline - fatigue + disruption_impact + initiative_boost
        wellbeing_now = config.get('WELLBEING_BASELINE', 0.8) * 100
        wellbeing_now -= np.mean(current_fatigue_level) * 100 # Avg fatigue impacts overall well-being display
        wellbeing_now -= active_disruption_effect * config.get('DISRUPTION_WELLBEING_DROP', 0.2) * 100
        
        if team_initiative == "More frequent breaks" and step % (num_steps // 5) == 0: # Simulating periodic effect of breaks
             current_fatigue_level *= 0.5 # Breaks reduce fatigue
             wellbeing_now += config.get('INITIATIVE_BREAKS_WELLBEING_RECOVERY_BOOST', 0.05) * 100
        if team_initiative == "Team recognition" and step % (num_steps // 3) == 0: # Simulating periodic recognition
            wellbeing_now += config.get('INITIATIVE_RECOGNITION_WELLBEING_BOOST_ABS', 0.05) * 100
        _wellbeing_scores[step] = np.clip(wellbeing_now, 0, 100)

        if _wellbeing_scores[step] < config.get('WELLBEING_ALERT_THRESHOLD', 0.6) * 100: wellbeing_triggers_dict['threshold'].append(step)
        if step > 0 and _wellbeing_scores[step] < _wellbeing_scores[step-1] - 7: wellbeing_triggers_dict['trend'].append(step) # Significant drop

        # Psychological Safety
        psych_safety_now = _psych_safety_scores[step-1] if step > 0 else config.get('PSYCH_SAFETY_INITIAL_MEAN', 0.75) * 100
        psych_safety_now -= config.get('PSYCH_SAFETY_EROSION_RATE', 0.001) * 100 # Slow natural erosion
        if active_disruption_effect > 0.5: psych_safety_now -= 5 # Disruptions can erode safety
        if team_initiative == "Team recognition" and step % (num_steps // 3) == 1: # Day after recognition
             psych_safety_now += config.get('INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST_ABS', 0.08) * 100
        _psych_safety_scores[step] = np.clip(psych_safety_now, 0, 100)


        # Task Compliance & Completion Rate
        base_compliance = config.get('BASE_TASK_COMPLETION_PROB', 0.95) * 100
        compliance_after_fatigue = base_compliance * (1 - np.mean(current_fatigue_level) * config.get('FATIGUE_IMPACT_ON_COMPLIANCE', 0.15))
        # Average complexity impact (could be zone specific)
        avg_complexity = np.mean([z.get('task_complexity', 0.5) for z in work_areas_config.values()])
        compliance_after_complexity = compliance_after_fatigue * (1 - avg_complexity * config.get('COMPLEXITY_IMPACT_ON_COMPLEXITY', 0.2)) # Mistake in original config, should be COMPLEXITY_IMPACT_ON_COMPLIANCE
        
        current_compliance = compliance_after_complexity
        if active_disruption_effect > 0:
            current_compliance = max(config.get('MIN_COMPLIANCE_DURING_DISRUPTION', 30.0), current_compliance * (1 - active_disruption_effect * config.get('DISRUPTION_COMPLIANCE_REDUCTION_FACTOR', 0.5)))
        _task_compliance_scores[step] = np.clip(current_compliance, 0, 100)
        _task_completion_rate[step] = np.clip(current_compliance * random.uniform(0.9, 1.1), 0, 100) # Related but with some variance


        # Collaboration (simplified: drops during disruption, higher if wellbeing is high)
        collab_score = 65 + (_wellbeing_scores[step] - 70) * 0.5 # Base + bonus from well-being
        collab_score -= active_disruption_effect * 20 # Disruptions reduce collaboration
        _collaboration_scores[step] = np.clip(collab_score, 10, 95)

        # OEE Components
        # Uptime: small chance of random dips, larger during disruption
        uptime_mod = 1.0
        if random.random() < config.get('EQUIPMENT_FAILURE_PROB_PER_INTERVAL', 0.005): uptime_mod -= 0.1
        if active_disruption_effect > 0.2: uptime_mod -= active_disruption_effect * 0.15
        _uptime_percent[step] = np.clip(_uptime_percent[step-1 if step > 0 else 0] * uptime_mod + np.random.normal(0,0.5), 70, 100)
        
        # Quality: affected by compliance and disruptions
        _quality_rate_percent[step] = np.clip(config.get('BASE_QUALITY_RATE', 0.98)*100 * (_task_compliance_scores[step]/100) * (1 - active_disruption_effect * 0.1), 70, 100)
        
        # Throughput: base potential * compliance * uptime factor * (1-fatigue)
        max_throughput = config.get('THEORETICAL_MAX_THROUGHPUT_UNITS_PER_INTERVAL', 100)
        # Base productivity sum for active workers (simplified)
        total_base_prod = sum(zone.get('base_productivity',0.8) * zone.get('workers',0) for zone in work_areas_config.values()) 
        throughput_potential_factor = total_base_prod / (num_team_members * 0.9) if num_team_members > 0 else 0 # Relative to avg base prod of 0.9

        _throughput_actual[step] = max_throughput * (_task_compliance_scores[step]/100) * (_uptime_percent[step]/100) * (1-np.mean(current_fatigue_level)*0.5) * throughput_potential_factor
        _throughput_actual[step] = np.clip(_throughput_actual[step], 0, max_throughput)
        # For OEE, throughput is often a percentage of theoretical max. Let's keep it simple here.
        # The 'efficiency_df' expects percentages, so normalize _throughput_actual if needed, or redefine 'throughput' metric for OEE.
        # Here, let's assume throughput_percentage is directly derived from compliance and uptime for simplicity in OEE.
        throughput_oee_component = _task_compliance_scores[step] * (_uptime_percent[step]/100) # simplified

        # Downtime
        if active_disruption_effect > 0.1 and random.random() < config.get('DOWNTIME_FROM_DISRUPTION_EVENT_PROB', 0.5):
            _downtime_minutes_per_interval[step] += max(0, np.random.normal(config.get('DOWNTIME_MEAN_MINUTES_PER_OCCURRENCE',8), config.get('DOWNTIME_STD_MINUTES_PER_OCCURRENCE',4)))
        if uptime_mod < 0.95 and random.random() < config.get('DOWNTIME_FROM_EQUIPMENT_FAILURE_PROB',0.7): # If uptime dipped from equip fail
             _downtime_minutes_per_interval[step] += config.get('EQUIPMENT_DOWNTIME_IF_FAIL_INTERVALS',3) * 2 # intervals to minutes

        _downtime_minutes_per_interval[step] = np.clip(_downtime_minutes_per_interval[step], 0, config['SHIFT_DURATION_MINUTES']) # Can't be more than shift

        # Operational Recovery & Productivity Loss
        # Recovery is high if compliance is high and downtime is low.
        recovery_score_now = ((_task_compliance_scores[step]/100) * (1- (_downtime_minutes_per_interval[step]/(2*60))) ) * 100 # Max downtime in an interval is e.g. 2 hrs. This needs refinement.
        # Simplified recovery post disruption - how quickly does it bounce back from lowest point
        if active_disruption_effect < 0.1 and step > disruption_end_step and disruption_end_step > 0:
             recovery_factor = 1 - math.exp(-(step - disruption_end_step) / config.get('RECOVERY_HALFLIFE_INTERVALS',10))
             lowest_compliance_during_disruption = np.min(_task_compliance_scores[max(0, int(disruption_end_step - config.get('DISRUPTION_DURATION_MEAN_INTERVALS',5))) : int(disruption_end_step)+1]) if disruption_end_step>0 else 30
             _operational_recovery_scores[step] = lowest_compliance_during_disruption + (base_compliance - lowest_compliance_during_disruption) * recovery_factor
        elif active_disruption_effect > 0:
            _operational_recovery_scores[step] = _task_compliance_scores[step] # During disruption, recovery matches compliance
        else:
            _operational_recovery_scores[step] = np.clip(recovery_score_now + np.random.normal(0,2), 0,100) # Normal fluctuation
        
        _productivity_loss_percent[step] = 100 - _operational_recovery_scores[step]


        # --- Update team positions (simplified random movement within assigned zone or facility) ---
        for i in range(num_team_members):
            worker_current_x[i] += np.random.normal(0, config.get('WORKER_SPEED_MEAN', 1.2) * 0.5) # smaller steps per interval
            worker_current_y[i] += np.random.normal(0, config.get('WORKER_SPEED_MEAN', 1.2) * 0.5)
            worker_current_x[i] = np.clip(worker_current_x[i], 0, facility_width)
            worker_current_y[i] = np.clip(worker_current_y[i], 0, facility_height)
            
            # Append to DataFrame data list for this step (if a snapshot per step is desired)
            # For efficiency, this part could be sampled if not every step/worker is needed.
            # Here, we keep it to show structure, but it can be large.
            zone_details = work_areas_config.get(worker_assigned_zone[i], {})
            actual_zone_for_pos = worker_assigned_zone[i] # Simplified, real check would use coordinates
            
            team_positions_data.append({
                'step': step, 'worker_id': i, 'x': worker_current_x[i], 'y': worker_current_y[i],
                'z': np.random.uniform(0, 3), 'zone': actual_zone_for_pos, 
                'status': 'working' if active_disruption_effect < 0.3 else 'affected_by_disruption'
            })


    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame()

    # --- Assemble final data structures ---
    task_compliance = {'data': list(_task_compliance_scores), 'z_scores': list(np.random.normal(0,1,num_steps)), 'forecast': [max(0,s-np.random.uniform(2,8)) for s in _task_compliance_scores]} # Dummy forecast
    collaboration_proximity = {'data': list(_collaboration_scores), 'forecast': [min(100,s+np.random.uniform(2,8)) for s in _collaboration_scores]}
    operational_recovery = list(_operational_recovery_scores)
    
    efficiency_df_data = {'uptime': list(_uptime_percent), 'throughput': list(np.clip(throughput_oee_component,0,100)), 'quality': list(_quality_rate_percent)}
    efficiency_df = pd.DataFrame(efficiency_df_data)
    efficiency_df['oee'] = np.clip((efficiency_df['uptime']/100 * efficiency_df['throughput']/100 * efficiency_df['quality']/100) * 100, 0, 100)

    productivity_loss = list(_productivity_loss_percent)
    worker_wellbeing = {'scores': list(_wellbeing_scores), 'triggers': wellbeing_triggers_dict}
    psychological_safety = list(_psych_safety_scores)
    feedback_impact = list(np.random.normal(0, 0.1, num_steps)) # Dummy
    downtime_minutes = list(_downtime_minutes_per_interval)
    task_completion_rate = list(_task_completion_rate)

    return (team_positions_df, task_compliance, collaboration_proximity, operational_recovery,
            efficiency_df, productivity_loss, worker_wellbeing, psychological_safety,
            feedback_impact, downtime_minutes, task_completion_rate)
