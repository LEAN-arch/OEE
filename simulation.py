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
import plotly.express as px

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
        logging.error("No work areas defined in config.py")
        raise ValueError("No work areas defined in config.py")
    
    # Ensure at least one team member per area
    members_per_area = max(1, num_team_members // len(areas))
    total_team_members = members_per_area * len(areas)
    logging.debug(f"Adjusted team members: {total_team_members} (min 1 per area)")

    team_data = pd.DataFrame({
        'area': np.repeat(areas, members_per_area)[:total_team_members],
        'task_adherence': np.random.uniform(0.75, 0.95, total_team_members),
        'supervisor_present': np.random.choice([True, False], total_team_members, p=[0.2, 0.8]),
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

        # Generate synthetic positions with validation
        positions = []
        for area in areas:
            center_x, center_y = WORK_AREAS[area]["center"]
            for _ in range(members_per_area):
                x = np.clip(np.random.normal(center_x, 10), 0, workplace_size)
                y = np.clip(np.random.normal(center_y, 10), 0, workplace_size)
                if not (np.isfinite(x) and np.isfinite(y)):
                    logging.warning(f"Step {step}, Area {area}: Invalid position (x={x}, y={y}), setting to center")
                    x, y = center_x, center_y
                positions.append({'area': area, 'x': x, 'y': y, 'step': step})
        if not positions:
            logging.error(f"Step {step}: No positions generated")
            raise ValueError("No positions generated for history_df")
        step_df = pd.DataFrame(positions)
        history.append(step_df)
        logging.debug(f"Step {step}: Generated {len(step_df)} positions")

    history_df = pd.concat(history).reset_index(drop=True)
    if history_df.empty or not {'x', 'y', 'area', 'step'}.issubset(history_df.columns):
        logging.error("Invalid history_df: Empty or missing required columns")
        raise ValueError("Invalid history_df: Empty or missing required columns")
    logging.debug(f"Generated history_df with {len(history_df)} rows")

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
        history_df,
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
    """Plot workplace activity and density with enhanced interactivity."""
    from config import WORK_AREAS
    area_colors = px.colors.qualitative.Plotly[:len(WORK_AREAS)]  # Assign unique colors to areas
    area_color_map = {area: color for area, color in zip(WORK_AREAS.keys(), area_colors)}

    if use_plotly:
        try:
            if history_df.empty or not {'x', 'y', 'area', 'step'}.issubset(history_df.columns):
                logging.error("Invalid history_df: Missing x, y, area, or step columns")
                raise ValueError("Invalid history_df: Missing x, y, area, or step columns")

            # Create hexbin plot
            fig = go.Figure()

            # Add hexbin trace
            hexbin = ff.create_hexbin_mapbox(
                data_frame=history_df,
                lat='y', lon='x',
                nx_hexagon=20,
                opacity=0.7,
                min_count=1,
                color_continuous_scale='Viridis',
                labels={'color': 'Team Members'},
                show_original_data=False
            ).data[0]
            fig.add_trace(hexbin)

            # Add scatter points for team members, colored by area
            for area in WORK_AREAS:
                area_df = history_df[history_df['area'] == area]
                if not area_df.empty:
                    fig.add_trace(go.Scatter(
                        x=area_df['x'],
                        y=area_df['y'],
                        mode='markers',
                        name=WORK_AREAS[area]["label"],
                        marker=dict(size=8, color=area_color_map[area], opacity=0.5),
                        text=[f"Area: {area}<br>Step: {row['step']}<br>Team Members: 1" for _, row in area_df.iterrows()],
                        hoverinfo='text'
                    ))

            # Add area center annotations
            for area in WORK_AREAS:
                x, y = WORK_AREAS[area]["center"]
                fig.add_annotation(
                    x=x, y=y,
                    text=WORK_AREAS[area]["label"],
                    showarrow=False,
                    font=dict(size=12, color='white', family='Arial Black'),
                    bgcolor='black',
                    opacity=0.8
                )

            # Add workplace boundary
            fig.add_trace(go.Scatter(
                x=[0, workplace_size, workplace_size, 0, 0],
                y=[0, 0, workplace_size, workplace_size, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                name='Workplace Boundary',
                showlegend=False
            ))

            # Update layout
            fig.update_layout(
                title="Workplace Activity & Density",
                xaxis_title="Workplace Width (m)",
                yaxis_title="Workplace Length (m)",
                xaxis_range=[-10, workplace_size + 10],
                yaxis_range=[-10, workplace_size + 10],
                showlegend=True,
                coloraxis_colorbar_title="Team Members",
                margin=dict(l=40, r=40, t=60, b=40),
                uirevision='constant'  # Preserve zoom state
            )

            # Add interactive controls
            # Area filter dropdown
            area_buttons = [
                dict(
                    label="All Areas",
                    method="update",
                    args=[{"visible": [True] * len(fig.data)},
                          {"title": "Workplace Activity & Density"}]
                )
            ]
            for i, area in enumerate(WORK_AREAS, start=1):
                visible = [True if j == 0 or j == i or j == len(WORK_AREAS) + 1 else False for j in range(len(fig.data))]
                area_buttons.append(
                    dict(
                        label=WORK_AREAS[area]["label"],
                        method="update",
                        args=[{"visible": visible},
                              {"title": f"Activity & Density: {WORK_AREAS[area]['label']}"}]
                    )
                )

            # Time slider for animation
            steps = []
            for step in sorted(history_df['step'].unique()):
                step_visible = [True if j == 0 or j == len(WORK_AREAS) + 1 else history_df[history_df['step'] == step]['area'].isin([area for area, _ in zip(WORK_AREAS, range(j-1))]).any() for j in range(len(fig.data))]
                steps.append(
                    dict(
                        method="update",
                        args=[{"visible": step_visible},
                              {"title": f"Activity & Density at Step {step}"}],
                        label=str(step)
                    )
                )

            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=area_buttons,
                        direction="down",
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    ),
                    dict(
                        type="buttons",
                        buttons=[dict(label="Reset Zoom", method="relayout", args=["xaxis.range", [-10, workplace_size + 10], "yaxis.range", [-10, workplace_size + 10]])],
                        x=0.3,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    )
                ],
                sliders=[dict(
                    active=0,
                    steps=steps,
                    x=0.1,
                    xanchor="left",
                    y=0,
                    yanchor="top",
                    len=0.9
                )]
            )

            return fig
        except Exception as e:
            logging.warning(f"Plotly hexbin failed: {str(e)}. Falling back to Matplotlib.")

    # Matplotlib fallback
    fig, ax = plt.subplots(figsize=(8, 6))
    if history_df.empty or not {'x', 'y', 'area'}.issubset(history_df.columns):
        logging.warning("Empty or invalid history_df, returning minimal plot")
        ax.text(0.5, 0.5, "No activity data", ha='center', va='center')
        ax.set_xlim(0, workplace_size)
        ax.set_ylim(0, workplace_size)
    else:
        # Hexbin plot
        hb = plt.hexbin(
            history_df['x'], history_df['y'],
            gridsize=20, cmap='viridis', mincnt=1,
            extent=(0, workplace_size, 0, workplace_size)
        )
        cb = plt.colorbar(hb, label='Team Members')
        counts = hb.get_array()
        if len(counts) > 0:
            min_count = 1
            max_count = np.max(counts)
            cb.set_ticks([min_count, max_count])
            cb.set_ticklabels(['1 Member', f'{int(max_count)} Members'])
        else:
            cb.set_ticks([1, 10])
            cb.set_ticklabels(['1 Member', '10 Members'])

        # Scatter points for team members
        for area in WORK_AREAS:
            area_df = history_df[history_df['area'] == area]
            if not area_df.empty:
                ax.scatter(
                    area_df['x'], area_df['y'],
                    c=area_color_map[area], s=50, alpha=0.5, label=WORK_AREAS[area]["label"]
                )

    # Workplace boundary
    ax.plot([0, workplace_size], [0, 0], 'k--', lw=1)
    ax.plot([0, workplace_size], [workplace_size, workplace_size], 'k--', lw=1)
    ax.plot([0, 0], [0, workplace_size], 'k--', lw=1)
    ax.plot([workplace_size, workplace_size], [0, workplace_size], 'k--', lw=1)

    # Area labels
    for area in WORK_AREAS:
        x, y = WORK_AREAS[area]["center"]
        ax.text(x, y, WORK_AREAS[area]["label"], color='white', ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    ax.set_title("Workplace Activity & Density")
    ax.set_xlabel("Workplace Width (m)")
    ax.set_ylabel("Workplace Length (m)")
    ax.set_xlim(0, workplace_size)
    ax.set_ylim(0, workplace_size)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
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
