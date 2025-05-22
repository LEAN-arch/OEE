"""
simulation.py
Simulate workplace operations for the Workplace Shift Monitoring Dashboard.
"""

import pandas as pd
import numpy as np
from scipy import stats

def simulate_workplace_operations(num_team_members, num_steps, disruption_intervals, team_initiative, config):
    """
    Simulate workplace operations with realistic worker movement and workload status.
    
    Args:
        num_team_members (int): Number of workers.
        num_steps (int): Number of 2-minute intervals.
        disruption_intervals (list): Steps where disruptions occur.
        team_initiative (str): Initiative to improve well-being/safety.
        config (dict): Configuration settings.
    
    Returns:
        tuple: Simulation results including worker positions with workload status.
    """
    # Initialize data
    np.random.seed(42)
    steps = range(num_steps)
    
    # Simulate worker positions with Brownian motion
    positions = []
    worker_states = {}  # Track each worker's position and workload over time
    for zone, area in config['WORK_AREAS'].items():
        num_workers_in_zone = area['workers']
        for worker in range(num_workers_in_zone):
            worker_id = f"{zone}_{worker}"
            # Initialize position at zone center
            x, y = area['center'][0], area['center'][1]
            workload = np.random.uniform(0.5, 1.0)  # Initial workload (0-1)
            worker_states[worker_id] = {'x': x, 'y': y, 'workload': workload, 'zone': zone}
    
    for step in steps:
        for worker_id, state in worker_states.items():
            # Simulate movement using Brownian motion
            dx = np.random.normal(0, 2)  # Small step in x
            dy = np.random.normal(0, 2)  # Small step in y
            state['x'] += dx
            state['y'] += dy
            # Keep workers within facility bounds
            state['x'] = np.clip(state['x'], 0, config['FACILITY_SIZE'])
            state['y'] = np.clip(state['y'], 0, config['FACILITY_SIZE'])
            # Update workload: increases over time, decreases with breaks
            state['workload'] += 0.01  # Gradual increase
            if step % config['BREAK_FREQUENCY_INTERVALS'] == 0 and step > 0:
                state['workload'] -= 0.2  # Break reduces workload
            if step in disruption_intervals:
                state['workload'] += 0.3  # Disruption increases workload
            state['workload'] = np.clip(state['workload'], 0, 1)
            # Determine workload status
            workload_status = "Normal" if state['workload'] < 0.7 else "High" if state['workload'] < 0.9 else "Critical"
            positions.append({
                'step': step,
                'worker': worker_id,
                'x': state['x'],
                'y': state['y'],
                'zone': state['zone'],
                'workload': state['workload'],
                'workload_status': workload_status
            })
    team_positions_df = pd.DataFrame(positions)
    
    # Simulate task compliance with adaptation and supervisor influence
    task_compliance = np.random.normal(90, 5, num_steps)
    z_scores = stats.zscore(task_compliance)
    for i in range(1, num_steps):
        task_compliance[i] += config['ADAPTATION_RATE'] * (100 - task_compliance[i-1]) + config['SUPERVISOR_INFLUENCE'] * 10
        if i in disruption_intervals:
            task_compliance[i] -= 20
    task_compliance = np.clip(task_compliance, 0, 100)
    forecast = [task_compliance[i] + np.random.normal(0, 2) for i in range(num_steps)]
    task_compliance_data = {'data': task_compliance.tolist(), 'z_scores': z_scores.tolist(), 'forecast': forecast}
    
    # Simulate collaboration proximity
    collaboration_proximity = np.random.normal(70, 10, num_steps)
    for i in range(num_steps):
        if i in disruption_intervals:
            collaboration_proximity[i] -= 15
        for d in disruption_intervals:
            if 0 <= i - d < config['DISRUPTION_RECOVERY_WINDOW']:
                collaboration_proximity[i] += 5
    collaboration_proximity = np.clip(collaboration_proximity, 0, 100)
    collab_forecast = [collaboration_proximity[i] + np.random.normal(0, 2) for i in range(num_steps)]
    collaboration_proximity_data = {'data': collaboration_proximity.tolist(), 'forecast': collab_forecast}
    
    # Simulate operational recovery
    operational_recovery = np.random.normal(80, 5, num_steps)
    for i in range(num_steps):
        if i in disruption_intervals:
            operational_recovery[i] -= 10
        for d in disruption_intervals:
            if 0 <= i - d < config['DISRUPTION_RECOVERY_WINDOW']:
                operational_recovery[i] += 5
    operational_recovery = np.clip(operational_recovery, 0, 100).tolist()
    
    # Simulate efficiency metrics
    efficiency_metrics = {
        'uptime': np.random.normal(85, 5, num_steps),
        'throughput': np.random.normal(80, 5, num_steps),
        'quality': np.random.normal(90, 5, num_steps)
    }
    for i in range(num_steps):
        if i in disruption_intervals:
            efficiency_metrics['uptime'][i] -= 10
            efficiency_metrics['throughput'][i] -= 10
            efficiency_metrics['quality'][i] -= 10
    efficiency_metrics['uptime'] = np.clip(efficiency_metrics['uptime'], 0, 100)
    efficiency_metrics['throughput'] = np.clip(efficiency_metrics['throughput'], 0, 100)
    efficiency_metrics['quality'] = np.clip(efficiency_metrics['quality'], 0, 100)
    efficiency_metrics['oee'] = (efficiency_metrics['uptime'] * efficiency_metrics['throughput'] * efficiency_metrics['quality']) / 10000
    efficiency_metrics_df = pd.DataFrame(efficiency_metrics)
    
    # Simulate productivity loss
    productivity_loss = np.random.normal(5, 2, num_steps)
    for i in range(num_steps):
        if i in disruption_intervals:
            productivity_loss[i] += 10
        for d in disruption_intervals:
            if 0 <= i - d < config['DISRUPTION_RECOVERY_WINDOW']:
                productivity_loss[i] -= 2
    productivity_loss = np.clip(productivity_loss, 0, 100).tolist()
    
    # Simulate worker well-being with breaks and workload caps
    wellbeing_scores = np.random.normal(80, 5, num_steps)
    for i in range(num_steps):
        if i % config['BREAK_FREQUENCY_INTERVALS'] == 0 and i > 0:
            wellbeing_scores[i] += 5
        if i % config['WORKLOAD_CAP_INTERVALS'] == 0 and i > 0:
            wellbeing_scores[i] -= 2
        if team_initiative == "More frequent breaks":
            wellbeing_scores[i] += 2
        if i in disruption_intervals:
            wellbeing_scores[i] -= 10
    wellbeing_scores = np.clip(wellbeing_scores, 0, 100)
    trends = []
    for i in range(config['WELLBEING_TREND_WINDOW'], num_steps):
        window = wellbeing_scores[i - config['WELLBEING_TREND_WINDOW']:i]
        if all(window[j] > window[j+1] for j in range(len(window)-1)) and i not in disruption_intervals:
            trends.append(i)
    triggers = {
        'threshold': [i for i, score in enumerate(wellbeing_scores) if score < config['WELLBEING_THRESHOLD'] * 100],
        'trend': trends,
        'work_area': {zone: [] for zone in config['WORK_AREAS'].keys()},
        'disruption': disruption_intervals
    }
    for zone in config['WORK_AREAS'].keys():
        zone_positions = team_positions_df[team_positions_df['zone'] == zone]
        for step in steps:
            step_positions = zone_positions[zone_positions['step'] == step]
            if len(step_positions) > 0:
                if len(step_positions) > config['WORK_AREAS'][zone]['workers'] * 1.5:
                    triggers['work_area'][zone].append(step)
    worker_wellbeing = {'scores': wellbeing_scores.tolist(), 'triggers': triggers}
    
    # Simulate psychological safety
    psychological_safety = np.random.normal(75, 5, num_steps)
    for i in range(num_steps):
        if team_initiative == "Team recognition":
            psychological_safety[i] += 5
        if i in disruption_intervals:
            psychological_safety[i] -= 10
    psychological_safety = np.clip(psychological_safety, 0, 100).tolist()
    
    # Simulate feedback impact
    feedback_impact = 5.0 if team_initiative == "More frequent breaks" else 3.0
    
    # Simulate downtime
    downtime_minutes = np.random.normal(2, 1, num_steps)
    for i in range(num_steps):
        if i in disruption_intervals:
            downtime_minutes[i] += 5
    downtime_minutes = np.clip(downtime_minutes, 0, None).tolist()
    
    # Simulate task completion rate
    task_completion_rate = np.random.normal(95, 3, num_steps)
    for i in range(num_steps):
        if i in disruption_intervals:
            task_completion_rate[i] -= 15
        if task_completion_rate[i] < config['TASK_COMPLETION_THRESHOLD'] * 100:
            task_completion_rate[i] += config['ADAPTATION_RATE'] * 10
    task_completion_rate = np.clip(task_completion_rate, 0, 100).tolist()
    
    return (team_positions_df, task_compliance_data, collaboration_proximity_data, operational_recovery,
            efficiency_metrics_df, productivity_loss, worker_wellbeing, psychological_safety,
            feedback_impact, downtime_minutes, task_completion_rate)
