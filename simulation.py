import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, zscore
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
import random
import logging
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from uuid import uuid4

# Configure logging for operational diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set global random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Default configuration for industrial workplace
DEFAULT_CONFIG = {
    'WORKSTATIONS': {
        'production_line': {'center': [50, 50], 'label': 'Production Line'},
        'assembly_zone': {'center': [150, 50], 'label': 'Assembly Zone'},
        'quality_control': {'center': [50, 150], 'label': 'Quality Control'},
        'logistics_hub': {'center': [150, 150], 'label': 'Logistics Hub'}
    },
    'COMPLIANCE_THRESHOLD': 0.7,  # Minimum acceptable compliance score
    'COMPLIANCE_TREND_WINDOW': 5,  # Steps to evaluate declining compliance trends
    'DISRUPTION_IMPACT_WINDOW': 3,  # Steps to monitor post-disruption recovery
    'BREAK_SCHEDULE_INTERVAL': 10,  # Steps between scheduled breaks
    'WORKLOAD_LIMIT_STEPS': 5  # Steps to assess sustained low compliance
}

def simulate_workplace_operations(
    num_workers=20,
    num_shifts=50,
    facility_size=200,
    compliance_adjustment_rate=0.1,
    supervisor_impact_factor=0.2,
    disruption_shifts=[10, 30],
    worker_initiative='More frequent breaks',
    skip_forecast=False,
    config=None
):
    """
    Simulate industrial workplace operations to monitor worker compliance, spatial distribution, and well-being.

    Parameters:
    - num_workers (int): Number of workers (adjusted to ensure min 1 per workstation).
    - num_shifts (int): Number of 2-minute shift intervals to simulate.
    - facility_size (float): Facility dimensions in meters (square).
    - compliance_adjustment_rate (float): Rate at which workers adjust task compliance.
    - supervisor_impact_factor (float): Supervisor's influence on compliance improvement.
    - disruption_shifts (list): Shift intervals where operational disruptions occur (e.g., equipment failure).
    - worker_initiative (str): Worker-initiated action to improve well-being (e.g., 'More frequent breaks').
    - skip_forecast (bool): Skip compliance and collaboration forecasting if True.
    - config (dict): Configuration dictionary; uses DEFAULT_CONFIG if None.

    Returns:
    - worker_positions_df (pd.DataFrame): Worker positions (x, y, workstation, shift).
    - compliance_variability (dict): Task compliance variability data, z-scores, and forecast.
    - collaboration_index (dict): Worker collaboration data, z-scores, and forecast.
    - operational_resilience (list): Resilience scores per shift.
    - efficiency_metrics (pd.DataFrame): Efficiency metrics (uptime, throughput, quality, oee).
    - productivity_loss (list): Productivity loss percentage per shift.
    - worker_wellbeing (dict): Well-being scores and intervention triggers.
    - safety_scores (list): Psychological safety scores per shift.
    - worker_feedback_impact (dict): Impact of worker initiatives on well-being and team cohesion.
    """
    # Use provided config or default
    config = config if config is not None else DEFAULT_CONFIG
    WORKSTATIONS = config['WORKSTATIONS']
    COMPLIANCE_THRESHOLD = config['COMPLIANCE_THRESHOLD']
    COMPLIANCE_TREND_WINDOW = config['COMPLIANCE_TREND_WINDOW']
    DISRUPTION_IMPACT_WINDOW = config['DISRUPTION_IMPACT_WINDOW']
    BREAK_SCHEDULE_INTERVAL = config['BREAK_SCHEDULE_INTERVAL']
    WORKLOAD_LIMIT_STEPS = config['WORKLOAD_LIMIT_STEPS']

    workstations = list(WORKSTATIONS.keys())
    if not workstations:
        logging.error("No workstations defined in configuration")
        raise ValueError("No workstations defined in configuration")

    # Ensure at least one worker per workstation
    workers_per_workstation = max(1, num_workers // len(workstations))
    total_workers = workers_per_workstation * len(workstations)
    logging.info(f"Adjusted total workers: {total_workers} (min 1 per workstation)")

    # Initialize worker data
    worker_data = pd.DataFrame({
        'workstation': np.repeat(workstations, workers_per_workstation)[:total_workers],
        'task_compliance': np.random.uniform(0.75, 0.95, total_workers),  # Initial compliance score
        'supervisor_present': np.random.choice([True, False], total_workers, p=[0.2, 0.8]),
    })

    # Initialize metrics
    worker_positions = []
    compliance_variability = []
    collaboration_index = []
    operational_resilience = []
    efficiency_metrics = []
    productivity_loss = []
    worker_wellbeing_scores = []
    safety_scores = []
    workstation_wellbeing = {ws: [] for ws in workstations}
    workstation_rest_quality = {ws: [] for ws in workstations}
    initial_compliance_avg = worker_data['task_compliance'].mean()

    # Intervention impacts
    break_improvement = 0.15  # Compliance boost from scheduled breaks
    supervisor_improvement = 0.1  # Compliance boost from supervisor oversight
    workload_balance_improvement = 0.15  # Compliance boost from workload balancing
    ergonomic_improvement = 0.12  # Compliance boost from ergonomic adjustments
    worker_initiative_impact = {
        'More frequent breaks': 0.2,
        'Task reduction': 0.18,
        'Wellness programs': 0.15,
        'Team recognition': 0.1
    }

    # Intervention triggers
    threshold_interventions = []
    trend_interventions = []
    workstation_interventions = {ws: [] for ws in workstations}
    disruption_interventions = []

    # Worker feedback impact
    worker_feedback_impact = {'wellbeing': 0, 'cohesion': 0}
    worker_initiatives_applied = 0

    # Simulation loop
    for shift in range(num_shifts):
        logging.info(f"Shift {shift}: Simulating operations")

        # Scheduled breaks
        if shift % BREAK_SCHEDULE_INTERVAL == 0 and shift > 0:
            worker_data['task_compliance'] += break_improvement * 0.1
            rest_quality = np.random.uniform(0.8, 1.0, len(workstations))
        else:
            rest_quality = np.random.uniform(0.5, 0.7, len(workstations))
        for ws, rq in zip(workstations, rest_quality):
            workstation_rest_quality[ws].append(rq)
            logging.debug(f"Shift {shift}, Workstation {ws}: Rest quality {rq:.3f}")

        # Simulate disruptions (e.g., equipment failure, supply chain issue)
        if shift in disruption_shifts:
            worker_data['task_compliance'] *= 0.85  # 15% compliance drop
            worker_data['supervisor_present'] = worker_data['supervisor_present'].sample(frac=1).values
            logging.info(f"Shift {shift}: Operational disruption applied")

        # Update task compliance
        for ws in workstations:
            ws_mask = worker_data['workstation'] == ws
            ws_data = worker_data[ws_mask]
            compliance_noise = np.random.normal(0, 0.01, ws_data.shape[0])
            if ws_data['supervisor_present'].any():
                avg_compliance = ws_data['task_compliance'].mean()
                worker_data.loc[ws_mask, 'task_compliance'] += (
                    compliance_adjustment_rate * supervisor_impact_factor * (avg_compliance - ws_data['task_compliance'])
                )
            worker_data.loc[ws_mask, 'task_compliance'] += compliance_noise
            worker_data.loc[ws_mask, 'task_compliance'] = np.clip(worker_data.loc[ws_mask, 'task_compliance'], 0.6, 1.0)

            # Workload management
            if shift >= WORKLOAD_LIMIT_STEPS and all(
                worker_data.loc[ws_mask, 'task_compliance'].mean() < 0.7 for _ in range(WORKLOAD_LIMIT_STEPS)
            ):
                worker_data.loc[ws_mask, 'task_compliance'] += workload_balance_improvement * 0.1
                workstation_interventions[ws].append(shift)
                logging.info(f"Shift {shift}, Workstation {ws}: Workload management intervention applied")

            # Workstation well-being
            workload_intensity = 0.5 * (1 - ws_data['task_compliance'].mean())
            disruption_impact = 0.05 if shift in disruption_shifts else 0
            shift_fatigue = min(shift / num_shifts, 1) * 0.1
            rest_quality = workstation_rest_quality[ws][-1]
            ws_wellbeing_score = max(0, 1 - (workload_intensity + disruption_impact + shift_fatigue) + rest_quality * 0.3)
            if not np.isfinite(ws_wellbeing_score):
                logging.warning(f"Shift {shift}, Workstation {ws}: Invalid well-being score, using previous or 0.5")
                ws_wellbeing_score = workstation_wellbeing[ws][-1] if workstation_wellbeing[ws] else 0.5
            workstation_wellbeing[ws].append(ws_wellbeing_score)
            logging.debug(f"Shift {shift}, Workstation {ws}: Well-being score {ws_wellbeing_score:.3f}")

        # Calculate operational metrics
        # Compliance variability (entropy of compliance distribution)
        compliance_dist = worker_data['task_compliance'].value_counts(normalize=True)
        shift_variability = entropy(compliance_dist) if not compliance_dist.empty else 0.0
        compliance_variability.append(shift_variability)

        # Worker positions
        positions = []
        for ws in workstations:
            center_x, center_y = WORKSTATIONS[ws]["center"]
            for _ in range(workers_per_workstation):
                x = np.clip(np.random.normal(center_x, 10), 0, facility_size)
                y = np.clip(np.random.normal(center_y, 10), 0, facility_size)
                if not (np.isfinite(x) and np.isfinite(y)):
                    logging.warning(f"Shift {shift}, Workstation {ws}: Invalid position (x={x}, y={y}), using center")
                    x, y = center_x, center_y
                positions.append({'workstation': ws, 'x': x, 'y': y, 'shift': shift})
        shift_positions = pd.DataFrame(positions)
        worker_positions.append(shift_positions)

        # Collaboration index (average intra-workstation distance)
        collaboration_score = 0
        for ws in workstations:
            ws_positions = shift_positions[shift_positions['workstation'] == ws][['x', 'y']].values
            if len(ws_positions) > 1:
                distances = pdist(ws_positions, metric='euclidean')
                collaboration_score += np.mean(distances) if distances.size > 0 else 0
        collaboration_index.append(collaboration_score / len(workstations) if collaboration_score > 0 else 0.5)

        # Well-being and psychological safety
        avg_wellbeing = np.mean([workstation_wellbeing[ws][-1] for ws in workstations])
        worker_wellbeing_scores.append(avg_wellbeing if np.isfinite(avg_wellbeing) else 0.5)
        safety_scores.append(avg_wellbeing * 0.8 + np.random.normal(0, 0.02))

        # Operational resilience (recovery from initial compliance)
        resilience = 1 - abs(worker_data['task_compliance'].mean() - initial_compliance_avg) / initial_compliance_avg
        operational_resilience.append(max(0, min(1, resilience)))

        # Efficiency metrics
        uptime = worker_data['task_compliance'].mean() * 100  # % of time workers are compliant
        throughput = (1 - (shift / num_shifts)) * 100 * worker_data['task_compliance'].mean()  # Output rate
        quality = worker_data['task_compliance'].mean() * 100 * (1 - 0.1 * len([s for s in disruption_shifts if s <= shift]))
        oee = (uptime * throughput * quality) / 10000  # Overall Equipment Effectiveness
        efficiency_metrics.append({
            'shift': shift,
            'uptime': uptime,
            'throughput': throughput,
            'quality': quality,
            'oee': oee
        })
        productivity_loss.append((1 - worker_data['task_compliance'].mean()) * 100)

        # Well-being interventions
        if avg_wellbeing < COMPLIANCE_THRESHOLD:
            worker_data['task_compliance'] += (break_improvement + ergonomic_improvement) * 0.1
            threshold_interventions.append(shift)
            logging.info(f"Shift {shift}: Low well-being intervention applied")
        if shift >= COMPLIANCE_TREND_WINDOW and all(
            workstation_wellbeing[ws][-i] < workstation_wellbeing[ws][-i-1]
            for ws in workstations for i in range(1, COMPLIANCE_TREND_WINDOW)
        ):
            worker_data['task_compliance'] += workload_balance_improvement * 0.1
            trend_interventions.append(shift)
            logging.info(f"Shift {shift}: Declining well-being trend intervention applied")
        for ws in workstations:
            other_wellbeing = [workstation_wellbeing[z][-1] for z in workstations if z != ws]
            if other_wellbeing and workstation_wellbeing[ws][-1] < min(other_wellbeing) - 0.08:
                worker_data.loc[worker_data['workstation'] == ws, 'supervisor_present'] = True
                worker_data.loc[worker_data['workstation'] == ws, 'task_compliance'] += supervisor_improvement * 0.1
                workstation_interventions[ws].append(shift)
                logging.info(f"Shift {shift}, Workstation {ws}: Supervisor intervention applied")
        if any(abs(shift - d) <= DISRUPTION_IMPACT_WINDOW for d in disruption_shifts) and avg_wellbeing < COMPLIANCE_THRESHOLD:
            worker_data['task_compliance'] += break_improvement * 0.1
            disruption_interventions.append(shift)
            logging.info(f"Shift {shift}: Post-disruption recovery intervention applied")

        # Worker-initiated actions
        if random.random() < 0.1:
            worker_data['task_compliance'] += worker_initiative_impact[worker_initiative] * 0.1
            worker_feedback_impact['wellbeing'] += worker_initiative_impact[worker_initiative] * 0.05
            worker_feedback_impact['cohesion'] += worker_initiative_impact[worker_initiative] * 0.03
            worker_initiatives_applied += 1
            logging.info(f"Shift {shift}: Worker-initiated action '{worker_initiative}' applied")

    worker_positions_df = pd.concat(worker_positions).reset_index(drop=True)
    if worker_positions_df.empty or not {'x', 'y', 'workstation', 'shift'}.issubset(worker_positions_df.columns):
        logging.error("Invalid worker_positions_df: Empty or missing required columns")
        raise ValueError("Invalid worker_positions_df: Empty or missing required columns")
    logging.info(f"Generated worker_positions_df with {len(worker_positions_df)} rows")

    efficiency_metrics_df = pd.DataFrame(efficiency_metrics)

    # Forecast compliance and collaboration trends
    compliance_forecast = None
    collaboration_forecast = None
    if not skip_forecast:
        try:
            X = np.arange(num_shifts).reshape(-1, 1)
            compliance_variability = np.nan_to_num(compliance_variability, nan=0.0)
            collaboration_index = np.nan_to_num(collaboration_index, nan=0.5)
            if np.any(~np.isfinite(compliance_variability)) or np.any(~np.isfinite(collaboration_index)):
                logging.error("Non-finite values in compliance_variability or collaboration_index")
                raise ValueError("Invalid data for forecasting")
            model_compliance = LinearRegression().fit(X, compliance_variability)
            model_collaboration = LinearRegression().fit(X, collaboration_index)
            future_shifts = np.arange(num_shifts, num_shifts + 10).reshape(-1, 1)
            compliance_forecast = model_compliance.predict(future_shifts)
            collaboration_forecast = model_collaboration.predict(future_shifts)
            logging.info("Forecasting completed successfully")
        except Exception as e:
            logging.error(f"Forecasting failed: {str(e)}")
            if not skip_forecast:
                raise

    return (
        worker_positions_df,
        {'data': compliance_variability, 'z_scores': zscore(np.nan_to_num(compliance_variability, nan=0.0)), 'forecast': compliance_forecast},
        {'data': collaboration_index, 'z_scores': zscore(np.nan_to_num(collaboration_index, nan=0.5)), 'forecast': collaboration_forecast},
        operational_resilience,
        efficiency_metrics_df,
        productivity_loss,
        {'scores': worker_wellbeing_scores, 'triggers': {
            'threshold': threshold_interventions,
            'trend': trend_interventions,
            'workstation': workstation_interventions,
            'disruption': disruption_interventions
        }},
        safety_scores,
        worker_feedback_impact
    )

def plot_task_compliance_trend(compliance_variability, disruption_shifts, forecast=None):
    """Plot task compliance variability trend to identify operational inconsistencies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(compliance_variability, label="Task Compliance Variability (Lower = Consistent)", color='#1f77b4')
    if forecast is not None:
        ax.plot(range(len(compliance_variability), len(compliance_variability) + 10), forecast, '--',
                label="Forecasted Variability", color='#ff7f0e')
    for i, s in enumerate(disruption_shifts):
        label = "Equipment Failure" if i == 0 else "Shift Change Disruption"
        ax.axvline(x=s, color='red', linestyle='--', label=label)
    ax.set_xlabel("Shift Interval (2-min)")
    ax.set_ylabel("Compliance Variability (Entropy)")
    ax.set_title("Task Compliance Consistency Over Time")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_worker_collaboration_trend(collaboration_index, forecast=None):
    """Plot worker collaboration trend to assess teamwork strength."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(collaboration_index, label="Collaboration Index (Higher = Stronger Teamwork)", color='#2ca02c')
    if forecast is not None:
        ax.plot(range(len(collaboration_index), len(collaboration_index) + 10), forecast, '--',
                label="Forecasted Collaboration", color='#d62728')
    ax.set_xlabel("Shift Interval (2-min)")
    ax.set_ylabel("Collaboration Index (m)")
    ax.set_title("Worker Collaboration Strength Over Time")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_operational_resilience(operational_resilience):
    """Plot operational resilience to evaluate recovery from disruptions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(operational_resilience, label="Resilience Score (1 = Full Recovery)", color='#9467bd')
    ax.set_xlabel("Shift Interval (2-min)")
    ax.set_ylabel("Resilience Score")
    ax.set_title("Operational Resilience Over Time")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_operational_efficiency(efficiency_metrics_df):
    """Plot operational efficiency metrics (OEE) to monitor performance."""
    if efficiency_metrics_df.empty:
        logging.warning("Empty efficiency_metrics_df, returning empty plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No efficiency data available", ha='center', va='center')
        return fig
    fig, ax = plt.subplots(figsize=(10, 5))
    efficiency_metrics_df.set_index('shift')[['uptime', 'throughput', 'quality', 'oee']].plot(
        ax=ax,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        label=['Uptime (% Operational)', 'Throughput (% Output)', 'Quality (% Defect-Free)', 'OEE (%)']
    )
    ax.set_ylabel("Performance (%)")
    ax.set_xlabel("Shift Interval (2-min)")
    ax.set_title("Operational Efficiency (OEE) Metrics")
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_worker_distribution(worker_positions_df, facility_size, config=None, use_plotly=True):
    """Plot worker distribution and density to optimize facility layout and movement."""
    config = config if config is not None else DEFAULT_CONFIG
    WORKSTATIONS = config['WORKSTATIONS']
    area_colors = px.colors.qualitative.Plotly[:len(WORKSTATIONS)]
    workstation_color_map = {ws: color for ws, color in zip(WORKSTATIONS.keys(), area_colors)}

    if use_plotly:
        try:
            if worker_positions_df.empty or not {'x', 'y', 'workstation', 'shift'}.issubset(worker_positions_df.columns):
                logging.error("Invalid worker_positions_df: Missing x, y, workstation, or shift columns")
                raise ValueError("Invalid worker_positions_df: Missing x, y, workstation, or shift columns")

            fig = go.Figure()
            hexbin = ff.create_hexbin_mapbox(
                data_frame=worker_positions_df,
                lat='y', lon='x',
                nx_hexagon=20,
                opacity=0.7,
                min_count=1,
                color_continuous_scale='Viridis',
                labels={'color': 'Worker Density'},
                show_original_data=False
            ).data[0]
            fig.add_trace(hexbin)

            for ws in WORKSTATIONS:
                ws_df = worker_positions_df[worker_positions_df['workstation'] == ws]
                if not ws_df.empty:
                    fig.add_trace(go.Scatter(
                        x=ws_df['x'],
                        y=ws_df['y'],
                        mode='markers',
                        name=WORKSTATIONS[ws]["label"],
                        marker=dict(size=8, color=workstation_color_map[ws], opacity=0.5),
                        text=[f"Workstation: {ws}<br>Shift: {row['shift']}<br>Workers: 1" for _, row in ws_df.iterrows()],
                        hoverinfo='text'
                    ))

            for ws in WORKSTATIONS:
                x, y = WORKSTATIONS[ws]["center"]
                fig.add_annotation(
                    x=x, y=y,
                    text=WORKSTATIONS[ws]["label"],
                    showarrow=False,
                    font=dict(size=12, color='white', family='Arial Black'),
                    bgcolor='black',
                    opacity=0.8
                )

            fig.add_trace(go.Scatter(
                x=[0, facility_size, facility_size, 0, 0],
                y=[0, 0, facility_size, facility_size, 0],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                name='Facility Boundary',
                showlegend=False
            ))

            fig.update_layout(
                title="Worker Distribution and Density in Facility",
                xaxis_title="Facility Width (m)",
                yaxis_title="Facility Length (m)",
                xaxis_range=[-10, facility_size + 10],
                yaxis_range=[-10, facility_size + 10],
                showlegend=True,
                coloraxis_colorbar_title="Worker Density",
                margin=dict(l=40, r=40, t=60, b=40),
                uirevision='constant'
            )

            workstation_buttons = [
                dict(
                    label="All Workstations",
                    method="update",
                    args=[{"visible": [True] * len(fig.data)},
                          {"title": "Worker Distribution and Density in Facility"}]
                )
            ]
            for i, ws in enumerate(WORKSTATIONS, start=1):
                visible = [True if j == 0 or j == i or j == len(WORKSTATIONS) + 1 else False for j in range(len(fig.data))]
                workstation_buttons.append(
                    dict(
                        label=WORKSTATIONS[ws]["label"],
                        method="update",
                        args=[{"visible": visible},
                              {"title": f"Worker Distribution: {WORKSTATIONS[ws]['label']}"}]
                    )
                )

            steps = []
            for shift in sorted(worker_positions_df['shift'].unique()):
                shift_visible = [
                    True if j == 0 or j == len(WORKSTATIONS) + 1 else
                    worker_positions_df[worker_positions_df['shift'] == shift]['workstation'].isin(
                        [ws for ws, _ in zip(WORKSTATIONS, range(j-1))]
                    ).any() for j in range(len(fig.data))
                ]
                steps.append(
                    dict(
                        method="update",
                        args=[{"visible": shift_visible},
                              {"title": f"Worker Distribution at Shift {shift}"}],
                        label=str(shift)
                    )
                )

            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=workstation_buttons,
                        direction="down",
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.15,
                        yanchor="top"
                    ),
                    dict(
                        type="buttons",
                        buttons=[dict(label="Reset Zoom", method="relayout",
                                      args=["xaxis.range", [-10, facility_size + 10],
                                            "yaxis.range", [-10, facility_size + 10]])],
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
            logging.warning(f"Plotly visualization failed: {str(e)}. Falling back to Matplotlib.")

    # Matplotlib fallback
    fig, ax = plt.subplots(figsize=(10, 6))
    if worker_positions_df.empty or not {'x', 'y', 'workstation'}.issubset(worker_positions_df.columns):
        logging.warning("Empty or invalid worker_positions_df, returning minimal plot")
        ax.text(0.5, 0.5, "No worker distribution data", ha='center', va='center')
        ax.set_xlim(0, facility_size)
        ax.set_ylim(0, facility_size)
    else:
        hb = plt.hexbin(
            worker_positions_df['x'], worker_positions_df['y'],
            gridsize=20, cmap='viridis', mincnt=1,
            extent=(0, facility_size, 0, facility_size)
        )
        cb = plt.colorbar(hb, label='Worker Density')
        counts = hb.get_array()
        if len(counts) > 0:
            min_count = 1
            max_count = np.max(counts)
            cb.set_ticks([min_count, max_count])
            cb.set_ticklabels(['1 Worker', f'{int(max_count)} Workers'])
        else:
            cb.set_ticks([1, 10])
            cb.set_ticklabels(['1 Worker', '10 Workers'])

        for ws in WORKSTATIONS:
            ws_df = worker_positions_df[worker_positions_df['workstation'] == ws]
            if not ws_df.empty:
                ax.scatter(
                    ws_df['x'], ws_df['y'],
                    c=workstation_color_map[ws], s=50, alpha=0.5, label=WORKSTATIONS[ws]["label"]
                )

    ax.plot([0, facility_size], [0, 0], 'k--', lw=1)
    ax.plot([0, facility_size], [facility_size, facility_size], 'k--', lw=1)
    ax.plot([0, 0], [0, facility_size], 'k--', lw=1)
    ax.plot([facility_size, facility_size], [0, facility_size], 'k--', lw=1)

    for ws in WORKSTATIONS:
        x, y = WORKSTATIONS[ws]["center"]
        ax.text(x, y, WORKSTATIONS[ws]["label"], color='white', ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    ax.set_title("Worker Distribution and Density in Facility")
    ax.set_xlabel("Facility Width (m)")
    ax.set_ylabel("Facility Length (m)")
    ax.set_xlim(0, facility_size)
    ax.set_ylim(0, facility_size)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    return fig

def plot_worker_wellbeing(wellbeing_scores):
    """Plot worker well-being trend to assess workforce health and morale."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wellbeing_scores, label="Well-Being Score (1 = Optimal Health)", color='#17becf')
    ax.set_xlabel("Shift Interval (2-min)")
    ax.set_ylabel("Well-Being Score")
    ax.set_title("Worker Well-Being Over Time")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_psychological_safety(safety_scores):
    """Plot psychological safety trend to evaluate trust and communication."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(safety_scores, label="Psychological Safety (1 = High Trust)", color='#ff9896')
    ax.set_xlabel("Shift Interval (2-min)")
    ax.set_ylabel("Safety Score")
    ax.set_title("Psychological Safety in Workplace")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# Example usage
if __name__ == "__main__":
    # Run simulation
    results = simulate_workplace_operations(
        num_workers=20,
        num_shifts=50,
        facility_size=200,
        compliance_adjustment_rate=0.1,
        supervisor_impact_factor=0.2,
        disruption_shifts=[10, 30],
        worker_initiative='More frequent breaks'
    )
    (
        worker_positions_df,
        compliance_variability,
        collaboration_index,
        operational_resilience,
        efficiency_metrics_df,
        productivity_loss,
        worker_wellbeing,
        safety_scores,
        worker_feedback_impact
    ) = results

    # Generate plots
    plt.figure(plot_task_compliance_trend(compliance_variability['data'], [10, 30], compliance_variability['forecast']))
    plt.figure(plot_worker_collaboration_trend(collaboration_index['data'], collaboration_index['forecast']))
    plt.figure(plot_operational_resilience(operational_resilience))
    plt.figure(plot_operational_efficiency(efficiency_metrics_df))
    plt.figure(plot_worker_distribution(worker_positions_df, 200))
    plt.figure(plot_worker_wellbeing(worker_wellbeing['scores']))
    plt.figure(plot_psychological_safety(safety_scores))
    plt.show()
```
