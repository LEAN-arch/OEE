# simulation.py
import numpy as np
import pandas as pd
import random

def simulate_workplace_operations(num_team_members, num_steps, disruption_intervals, team_initiative, config):
    np.random.seed(42); random.seed(42) 
    facility_width, facility_height = config.get('FACILITY_SIZE', (100, 80))
    work_areas_config = config.get('WORK_AREAS', {})
    team_positions_data = []
    for step in range(num_steps):
        for worker_id in range(num_team_members):
            zone_names = list(work_areas_config.keys())
            current_zone = random.choice(zone_names) if zone_names else "UnassignedZone"
            zone_details = work_areas_config.get(current_zone, {})
            zone_coords_list = zone_details.get('coords', [])
            if zone_coords_list and len(zone_coords_list) == 2:
                (x0, y0), (x1, y1) = zone_coords_list
                pos_x = np.random.uniform(min(x0,x1), max(x0,x1))
                pos_y = np.random.uniform(min(y0,y1), max(y0,y1))
            else: 
                pos_x = np.random.uniform(0, facility_width)
                pos_y = np.random.uniform(0, facility_height)
            team_positions_data.append({
                'step': step, 'worker_id': worker_id, 'x': pos_x, 'y': pos_y,
                'z': np.random.uniform(0, 3) if config.get('use_3d_plotting', False) else 0, # Renamed for main.py consistency
                'zone': current_zone, 'status': random.choice(['working', 'idle', 'break'])
            })
    team_positions_df = pd.DataFrame(team_positions_data) if team_positions_data else pd.DataFrame(columns=['step', 'worker_id', 'x', 'y', 'z', 'zone', 'status'])
    def generate_series(length, base, trend, noise, disrupt_effect=0.2, min_v=0, max_v=100, lower_is_better=False):
        series = np.zeros(length); val = base
        for i in range(length):
            val += trend + np.random.normal(0, noise)
            if i in disruption_intervals: val += disrupt_effect * base if lower_is_better else -disrupt_effect * base
            series[i] = np.clip(val, min_v, max_v)
        return list(series)
    tc_data = generate_series(num_steps, 85, -0.05, 2, disrupt_effect=0.2)
    task_compliance = {'data': tc_data, 'z_scores': list(np.random.normal(0, 1, num_steps)), 'forecast': [max(0, x - 5 + np.random.normal(0,3)) for x in tc_data]}
    cp_data = generate_series(num_steps, 60, 0.02, 3, disrupt_effect=0.15)
    collaboration_proximity = {'data': cp_data, 'forecast': [max(0, x + 5 + np.random.normal(0,3)) for x in cp_data]}
    operational_recovery = generate_series(num_steps, 70, 0.01, 5, disrupt_effect=0.3)
    eff_data = {'uptime': generate_series(num_steps, 95, -0.01, 1, disrupt_effect=0.05), 'throughput': generate_series(num_steps, 80, 0, 2, disrupt_effect=0.1), 'quality': generate_series(num_steps, 98, -0.005, 0.5, disrupt_effect=0.02)}
    efficiency_df = pd.DataFrame(eff_data); efficiency_df['oee'] = (efficiency_df['uptime']/100 * efficiency_df['throughput']/100 * efficiency_df['quality']/100) * 100; efficiency_df = efficiency_df.clip(0,100)
    productivity_loss = [max(0, 100 - x + np.random.normal(0,5)) for x in operational_recovery]
    wb_scores = generate_series(num_steps, 75, -0.03, 2.5, disrupt_effect=0.15)
    if team_initiative == "More frequent breaks": wb_scores = [min(100, s + config.get('INITIATIVE_BREAKS_FATIGUE_REDUCTION',0.1)*100*random.uniform(0.7,1)) for s in wb_scores]
    elif team_initiative == "Team recognition": wb_scores = [min(100, s + config.get('INITIATIVE_RECOGNITION_WELLBEING_BOOST',0.08)*100*random.uniform(0.7,1)) for s in wb_scores]
    worker_wellbeing = {'scores': wb_scores, 'triggers': {'threshold': [i for i,s in enumerate(wb_scores) if s < config.get('WELLBEING_THRESHOLD',0.6)*100 and random.random()<0.3], 'trend': [i for i in range(1,num_steps) if wb_scores[i] < wb_scores[i-1]-7 and random.random()<0.3], 'work_area': {zn: [i for i in disruption_intervals if random.random()<0.15] for zn in work_areas_config if 'Assembly' in zn or 'Quality' in zn}, 'disruption': [i for i in disruption_intervals if random.random()<0.4]}}
    ps_scores = generate_series(num_steps, 65, 0.015, 2, disrupt_effect=0.1)
    if team_initiative == "Team recognition": ps_scores = [min(100, s + config.get('INITIATIVE_RECOGNITION_PSYCHSAFETY_BOOST',0.06)*100*random.uniform(0.7,1)) for s in ps_scores]
    psychological_safety = ps_scores
    feedback_impact = list(np.random.normal(0, 0.5, num_steps))
    downtime_mins_arr = np.zeros(num_steps)
    for i in range(num_steps):
        if i in disruption_intervals and random.random() < config.get('DOWNTIME_PROB_PER_DISRUPTION', 0.4): downtime_mins_arr[i] = max(0, np.random.normal(config.get('DOWNTIME_DURATION_MEAN_MINUTES',5), config.get('DOWNTIME_DURATION_STD_MINUTES',2)))
    downtime_minutes = list(downtime_mins_arr)
    task_completion_rate = generate_series(num_steps, 90, -0.04, 3, disrupt_effect=0.25)
    return (team_positions_df, task_compliance, collaboration_proximity, operational_recovery, efficiency_df, productivity_loss, worker_wellbeing, psychological_safety, feedback_impact, downtime_minutes, task_completion_rate)
