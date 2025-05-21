import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, zscore
from sklearn.linear_model import LinearRegression
import random
import logging
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set global random seed
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(num_team_members, num_steps, workplace_size, adaptation_rate, supervisor_influence, disruption_steps, worker_priority, skip_forecast=False):
    """Generate synthetic workplace data with humane conditions."""
    from config import WORK_AREAS
    areas = list(WORK_AREAS.keys())
    if not areas:
        raise ValueError("No work areas defined in config.py")
    members_per_area = num_team_members // len(areas)
    team_data = pd.DataFrame({
        'area': np.repeat(areas, members_per_area),
        'task_adherence': np.random.uniform(0.75, 0.95, num_team_members),
        'supervisor_present': np.random.choice([True, False], num_team_members, p=[0.2, 0.8]),
    })

    history = []
    adherence_entropy = []
    clustering_index = []
    resilience_scores = []
    efficiency_history = []
    productivity_loss = []
    wellbeing_scores = []
    safety_scores = []
    area_wellbeing = {area: [] for area in areas}
    area_rest_quality = {area: [] for area in areas}
    initial_adherence_avg = team_data['task_adherence'].mean()

    # Well-being action impacts
    break_boost = 0.15
    supervisor_boost = 0.1
    workload_balance_boost = 0.15
    ergonomic_boost = 0.12
    worker_action_boost = {'More frequent breaks': 0.2, 'Task reduction': 0.18, 'Wellness resources': 0.15, 'Team recognition': 0.1}

    # Triggers
    threshold_triggers = []
    trend_triggers = []
    area_triggers = {area: [] for area in areas}
    disruption_triggers = []
    from config import WELLBEING_THRESHOLD, WELLBEING_TREND_LENGTH, WELLBEING_DISRUPTION_WINDOW, BREAK_INTERVAL, WORKLOAD_CAP_STEPS

    # Feedback impact
    feedback_impact = {'wellbeing': 0, 'cohesion': 0}
    worker_actions_applied = 0

    # Synthetic positions
    for step in range(num_steps):
        logging.debug(f"Step {step}: Starting data generation")
        
        # Proactive breaks
        if step % BREAK_INTERVAL == 0 and step > 0:
            team_data['task_adherence'] += break_boost * 0.1
            for area in areas:
                rest_quality = np.random.uniform(0.8, 1.0)
                area_rest_quality[area].append(rest_quality)
                logging.debug(f"Step {step}, Area {area}: Rest quality {rest_quality}")
        else:
            for area in areas:
                rest_quality = np.random.uniform(0.5, 0.7)
                area_rest_quality[area].append(rest_quality)
                logging.debug(f"Step {step}, Area {area}: Rest quality {rest_quality}")

        # Simulate disruptions
        if step in disruption_steps:
            team_data['task_adherence'] *= 0.85
            team_data['supervisor_present'] = team_data['supervisor_present'].sample(frac=1).values
            logging.debug(f"Step {step}: Applied disruption")

        # Update task adherence and well-being
        for area in areas:
            area_mask = team_data['area'] == area
            area_data = team_data[area_mask]
            adherence_noise = np.random.normal(0, 0.01)
            if area_data['supervisor_present'].any():
                avg_adherence = area_data['task_adherence'].mean()
                team_data.loc[area_mask, 'task_adherence'] += adaptation_rate * supervisor_influence * (avg_adherence - area_data['task_adherence'].mean())
            team_data.loc[area_mask, 'task_adherence'] += adherence_noise
            team_data.loc[area_mask, 'task_adherence'] = np.clip(team_data.loc[area_mask, 'task_adherence'], 0.6, 1.0)
            logging.debug(f"Step {step}, Area {area}: Task adherence mean {team_data.loc[area_mask, 'task_adherence'].mean()}")

            # Workload cap
            if step >= WORKLOAD_CAP_STEPS and all(team_data.loc[area_mask, 'task_adherence'].mean() < 0.7 for _ in range(WORKLOAD_CAP_STEPS)):
                team_data.loc[area_mask, 'task_adherence'] += workload_balance_boost * 0.1
                area_triggers[area].append(step)
                logging.debug(f"Step {step}, Area {area}: Applied workload cap")

            # Area well-being
            workload_intensity = 0.5 * (1 - area_data['task_adherence'].mean())
            disruption_impact = 0.05 if step in disruption_steps else 0
            shift_fatigue = min(step / num_steps, 1) * 0.1
            rest_quality = area_rest_quality[area][-1]
            area_wellbeing_score = max(0, 1 - (workload_intensity + disruption_impact + shift_fatigue) + rest_quality * 0.3 + np.random.normal(0, 0.03))
            if not np.isfinite(area_wellbeing_score):
                logging.warning(f"Step {step}, Area {area}: Invalid well-being score {area_wellbeing_score}, setting to 0.5")
                area_wellbeing_score = 0.5
            area_wellbeing[area].append(area_wellbeing_score)
            logging.debug(f"Step {step}, Area {area}: Well-being score {area_wellbeing_score}")

        # Well-being actions
        avg_wellbeing = np.mean([area_wellbeing[area][-1] for area in areas])
        if not np.isfinite(avg_wellbeing):
            logging.warning(f"Step {step}: Invalid avg well-being {avg_wellbeing}, setting to 0.5")
            avg_wellbeing = 0.5
        if avg_wellbeing < WELLBEING_THRESHOLD:
            team_data['task_adherence'] += (break_boost + ergonomic_boost) * 0.1
            threshold_triggers.append(step)
        if step >= WELLBEING_TREND_LENGTH and all(area_wellbeing[area][-i] < area_wellbeing[area][-i-1] for area in areas for i in range(1, WELLBEING_TREND_LENGTH)):
            team_data['task_adherence'] += workload_balance_boost * 0.1
            trend_triggers.append(step)
        for area in areas:
            other_wellbeing = [area_wellbeing[z][-1] for z in areas if z != area]
            if other_wellbeing and area_wellbeing[area][-1] < min(other_wellbeing) - 0.08:
                team_data.loc[team_data['area'] == area, 'supervisor_present'] = True
                team_data.loc[team_data['area'] == area, 'task_adherence'] += supervisor_boost * 0.1
                area_triggers[area].append(step)
        if any(abs(step - d) <= WELLBEING_DISRUPTION_WINDOW for d in disruption_steps) and avg_wellbeing < WELLBEING_THRESHOLD:
            team_data['task_adherence'] += break_boost * 0.1
            disruption_triggers.append(step)

        # Worker-initiated actions
        if random.random() < 0.1:
            team_data['task_adherence'] += worker_action_boost[worker_priority] * 0.1
            feedback_impact['wellbeing'] += worker_action_boost[worker_priority] * 0.05
            feedback_impact['cohesion'] += worker_action_boost[worker_priority] * 0.03
            worker_actions_applied += 1
            logging.debug(f"Step {step}: Applied worker action {worker_priority}")

        # Generate synthetic positions
        positions = []
        for area in areas:
            center_x, center_y = WORK_AREAS[area]["center"]
            for _ in range(members_per_area):
                x = np.clip(np.random.normal(center_x, 10), 0, workplace_size)
                y = np.clip(np.random.normal(center_y, 10), 0, workplace_size)
                positions.append({'area': area, 'x': x, 'y': y, 'step': step})
        history.append(pd.DataFrame(positions))

        # Adherence variability
        hist, _ = np.histogram(team_data['task_adherence'], bins=10, range=(0.5, 1.0), density=True)
        if not np.all(np.isfinite(hist)) or np.sum(hist) == 0:
            logging.warning(f"Step {step}: Invalid histogram for entropy, setting to 0")
            adherence_entropy.append(0)
        else:
            ent = entropy(hist + 1e-9)
            if not np.isfinite(ent):
                logging.warning(f"Step {step}: Invalid entropy {ent}, setting to 0")
                ent = 0
            adherence_entropy.append(ent)
        logging.debug(f"Step {step}: Adherence entropy {adherence_entropy[-1]}")

        # Team clustering
        cohesion_score = np.mean([1 / (1 + max(0, area_wellbeing[area][-1])) for area in areas])
        if not np.isfinite(cohesion_score):
            logging.warning(f"Step {step}: Invalid cohesion score {cohesion_score}, setting to 0.5")
            cohesion_score = 0.5
        clustering_index.append(cohesion_score + np.random.normal(0, 0.05))
        logging.debug(f"Step {step}: Clustering index {clustering_index[-1]}")

        # Resilience
        deviation = abs(team_data['task_adherence'].mean() - initial_adherence_avg)
        resilience_score = 1 - deviation / initial_adherence_avg
        if not np.isfinite(resilience_score):
            logging.warning(f"Step {step}: Invalid resilience score {resilience_score}, setting to 0.5")
            resilience_score = 0.5
        resilience_scores.append(resilience_score)
        logging.debug(f"Step {step}: Resilience score {resilience_scores[-1]}")

        # Operational Efficiency
        avg_adherence = team_data['task_adherence'].mean()
        uptime = min(0.92 + 0.08 * avg_adherence, 0.96) + np.random.normal(0, 0.004)
        throughput = min(0.87 + 0.08 * avg_adherence, 0.92) + np.random.normal(0, 0.004)
        quality = min(0.98 + 0.02 * avg_adherence, 0.99) + np.random.normal(0, 0.001)
        efficiency = uptime * throughput * quality
        if not np.isfinite(efficiency):
            logging.warning(f"Step {step}: Invalid efficiency {efficiency}, setting to 0.8")
            efficiency = 0.8
        efficiency_history.append({
            'step': step,
            'uptime': uptime,
            'throughput': throughput,
            'quality': quality,
            'efficiency': efficiency
        })
        logging.debug(f"Step {step}: Efficiency {efficiency}")

        # Productivity loss
        loss = max(0, initial_adherence_avg - team_data['task_adherence'].mean()) * 50
        if not np.isfinite(loss):
            logging.warning(f"Step {step}: Invalid productivity loss {loss}, setting to 0")
            loss = 0
        productivity_loss.append(loss)
        logging.debug(f"Step {step}: Productivity loss {loss}")

        # Well-being and psychological safety
        wellbeing_score = avg_wellbeing
        if not np.isfinite(wellbeing_score):
            logging.warning(f"Step {step}: Invalid well-being score {wellbeing_score}, setting to 0.5")
            wellbeing_score = 0.5
        wellbeing_scores.append(wellbeing_score)
        safety_score = np.mean([area_wellbeing[area][-1] * (1 + 0.3 if team_data[team_data['area'] == area]['supervisor_present'].any() else 0) * (1 + 0.2 * worker_actions_applied / (step + 1)) for area in areas])
        if not np.isfinite(safety_score):
            logging.warning(f"Step {step}: Invalid safety score {safety_score}, setting to 0.5")
            safety_score = 0.5
        safety_scores.append(safety_score)
        logging.debug(f"Step {step}: Well-being {wellbeing_score}, Safety {safety_score}")

    # Forecasting
    adherence_forecast = None
    clustering_forecast = None
    if not skip_forecast:
        try:
            X = np.arange(num_steps).reshape(-1, 1)
            adherence_entropy = np.nan_to_num(adherence_entropy, nan=0.0)
            clustering_index = np.nan_to_num(clustering_index, nan=0.5)
            if np.any(~np.isfinite(adherence_entropy)) or np.any(~np.isfinite(clustering_index)):
                logging.error("Non-finite values in adherence_entropy or clustering_index")
                raise ValueError("Invalid data for forecasting")
            model_adherence = LinearRegression().fit(X, adherence_entropy)
            model_clustering = LinearRegression().fit(X, clustering_index)
            future_steps = np.arange(num_steps, num_steps + 10).reshape(-1, 1)
            adherence_forecast = model_adherence.predict(future_steps)
            clustering_forecast = model_clustering.predict(future_steps)
            logging.debug("Forecasting completed successfully")
        except Exception as e:
            logging.error(f"Forecasting failed: {str(e)}")
            if not skip_forecast:
                raise

    return (
        pd.concat(history).reset_index(drop=True),
        {'data': adherence_entropy, 'z_scores': zscore(np.nan_to_num(adherence_entropy, nan=0.0)), 'forecast': adherence_forecast},
        {'data': clustering_index, 'z_scores': zscore(np.nan_to_num(clustering_index, nan=0.5)), 'forecast': clustering_forecast},
        resilience_scores,
        efficiency_history,
        productivity_loss,
        {'scores': wellbeing_scores, 'triggers': {
            'threshold': threshold_triggers,
            'trend': trend_triggers,
            'zone': area_triggers,
            'disruption': disruption_triggers
        }},
        safety_scores,
        feedback_impact
    )

def plot_compliance_variability(adherence_entropy, disruption_steps, forecast=None):
    """Plot task adherence trends."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(adherence_entropy, label="Adherence (Lower = Uniform)", color='#1f77b4')
    if forecast is not None:
        ax.plot(range(len(adherence_entropy), len(adherence_entropy) + 10), forecast, '--', label="Predicted", color='#ff7f0e')
    for i, s in enumerate(disruption_steps):
        label = "Disruption" if i == 0 else "Shift Change"
        ax.axvline(x=s, color='red', linestyle='--', label=label)
    ax.set_xlabel("Shift Time (2-min)")
    ax.set_ylabel("Adherence Variation")
    ax.set_title("Task Adherence Trends")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_team_clustering(clustering_index, forecast=None):
    """Plot team collaboration trends."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(clustering_index, label="Collaboration (Higher = Stronger)", color='#2ca02c')
    if forecast is not None:
        ax.plot(range(len(clustering_index), len(clustering_index) + 10), forecast, '--', label="Predicted", color='#d62728')
    ax.set_xlabel("Shift Time (2-min)")
    ax.set_ylabel("Collaboration Score")
    ax.set_title("Team Collaboration Trends")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_resilience(resilience_scores):
    """Plot team resilience trends."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(resilience_scores, label="Resilience (1 = Full Recovery)", color='#9467bd')
    ax.set_xlabel("Shift Time (2-min)")
    ax.set_ylabel("Resilience Score")
    ax.set_title("Team Resilience Trends")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_oee(efficiency_df):
    """Plot operational efficiency trends."""
    fig, ax = plt.subplots(figsize=(8, 4))
    efficiency_df.set_index('step')[['uptime', 'throughput', 'quality', 'efficiency']].plot(
        ax=ax,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        label=['Uptime (>90%)', 'Throughput (>85%)', 'Quality (>97%)', 'Efficiency']
    )
    ax.set_ylabel("Efficiency (%)")
    ax.set_xlabel("Shift Time (2-min)")
    ax.set_title("Operational Efficiency Trends")
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_worker_density(history_df, workplace_size, use_plotly=True):
    """Plot workplace activity and density."""
    from config import WORK_AREAS
    if use_plotly:
        try:
            if history_df.empty or not {'x', 'y', 'area'}.issubset(history_df.columns):
                raise ValueError("Invalid history_df: Missing x, y, or area columns")
            fig = ff.create_hexbin_mapbox(
                data_frame=history_df,
                lat='y', lon='x',
                nx_hexagon=20,
                opacity=0.7,
                min_count=1,
                color_continuous_scale='cividis',
                labels={'color': 'Activity Level'},
                show_original_data=False
            )
            fig.update_layout(
                xaxis_title="Workplace Width (m)",
                yaxis_title="Workplace Length (m)",
                title="Workplace Activity & Density",
                xaxis_range=[0, workplace_size],
                yaxis_range=[0, workplace_size],
                showlegend=False
            )
            # Add area labels
            fig.add_trace(go.Scatter(
                x=[WORK_AREAS[area]["center"][0] for area in WORK_AREAS],
                y=[WORK_AREAS[area]["center"][1] for area in WORK_AREAS],
                text=[WORK_AREAS[area]["label"] for area in WORK_AREAS],
                mode='text',
                textfont=dict(color='white', size=12),
                showlegend=False
            ))
            # Add workplace border
            fig.add_trace(go.Scatter(
                x=[0, workplace_size, workplace_size, 0, 0],
                y=[0, 0, workplace_size, workplace_size, 0],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))
            return fig
        except Exception as e:
            logging.warning(f"Plotly hexbin failed: {str(e)}. Falling back to Matplotlib.")
    
    # Matplotlib fallback
    fig, ax = plt.subplots(figsize=(8, 6))
    if history_df.empty or not {'x', 'y'}.issubset(history_df.columns):
        logging.warning("Empty or invalid history_df, returning empty plot")
        ax.text(0.5, 0.5, "No activity data", ha='center', va='center')
        ax.set_xlim(0, workplace_size)
        ax.set_ylim(0, workplace_size)
        return fig
    hb = plt.hexbin(
        history_df['x'], history_df['y'],
        gridsize=20, cmap='cividis', mincnt=1,
        extent=(0, workplace_size, 0, workplace_size)
    )
    cb = plt.colorbar(hb, label='Activity Level')
    counts = hb.get_array()
    if len(counts) > 0:
        min_count = 1
        max_count = np.max(counts)
        cb.set_ticks([min_count, max_count])
        cb.set_ticklabels(['Low Activity', 'High Density'])
    else:
        logging.warning("No data in hexbin plot, setting default colorbar ticks")
        cb.set_ticks([1, 10])
        cb.set_ticklabels(['Low Activity', 'High Density'])
    
    ax.plot([0, workplace_size], [0, 0], 'k-', lw=1)
    ax.plot([0, workplace_size], [workplace_size, workplace_size], 'k-', lw=1)
    ax.plot([0, 0], [0, workplace_size], 'k-', lw=1)
    ax.plot([workplace_size, workplace_size], [0, workplace_size], 'k-', lw=1)
    for area in WORK_AREAS:
        x, y = WORK_AREAS[area]["center"]
        ax.text(x, y, WORK_AREAS[area]["label"], color='white', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
    
    ax.set_title("Workplace Activity & Density")
    ax.set_xlabel("Workplace Width (m)")
    ax.set_ylabel("Workplace Length (m)")
    ax.set_xlim(0, workplace_size)
    ax.set_ylim(0, workplace_size)
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_wellbeing(wellbeing_scores):
    """Plot team well-being trends."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wellbeing_scores, label="Well-Being (1 = Optimal)", color='#17becf')
    ax.set_xlabel("Shift Time (2-min)")
    ax.set_ylabel("Well-Being Score")
    ax.set_title("Team Well-Being Trends")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_psychological_safety(safety_scores):
    """Plot psychological safety trends."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(safety_scores, label="Safety (1 = High Trust)", color='#ff9896')
    ax.set_xlabel("Shift Time (2-min)")
    ax.set_ylabel("Safety Score")
    ax.set_title("Psychological Safety Trends")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig
