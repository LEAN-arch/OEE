"""
visualizations.py
Visualization functions for the Industrial Workplace Shift Monitoring Dashboard.
Generates Plotly charts with consistent styling and accessibility features.
"""

import logging
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Consistent Plotly styling
PLOTLY_TEMPLATE = "plotly_white"
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c',
    'danger': '#d62728',
    'background': '#ffffff',
    'text': '#333333'
}

def plot_task_compliance_trend(compliance_data: list, disruption_intervals: list, forecast: list = None) -> go.Figure:
    """
    Plot task compliance variability with optional forecast.

    Args:
        compliance_data (list): Compliance entropy values.
        disruption_intervals (list): Time steps for disruptions.
        forecast (list, optional): Forecasted compliance values.

    Returns:
        go.Figure: Plotly figure.
    """
    logger.info("Plotting task compliance trend")
    fig = px.line(
        x=list(range(len(compliance_data))),
        y=compliance_data,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Compliance Entropy'},
        title='Task Compliance Variability',
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_SCHEME['primary']]
    )
    for t in disruption_intervals:
        fig.add_vline(x=t, line_dash="dash", line_color=COLOR_SCHEME['danger'], opacity=0.5)
    if forecast is not None:
        fig.add_scatter(x=list(range(len(forecast))), y=forecast, name='Forecast',
                        line=dict(dash='dash', color=COLOR_SCHEME['secondary']))
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text']),
        title_x=0.5,
        hovermode='x unified'
    )
    fig.update_traces(line=dict(width=2))
    return fig

def plot_worker_collaboration_trend(collab_data: list, forecast: list = None) -> go.Figure:
    """
    Plot worker collaboration index with optional forecast.

    Args:
        collab_data (list): Collaboration index values.
        forecast (list, optional): Forecasted collaboration values.

    Returns:
        go.Figure: Plotly figure.
    """
    logger.info("Plotting worker collaboration trend")
    fig = px.line(
        x=list(range(len(collab_data))),
        y=collab_data,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Collaboration Strength'},
        title='Worker Collaboration Index',
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_SCHEME['primary']]
    )
    if forecast is not None:
        fig.add_scatter(x=list(range(len(forecast))), y=forecast, name='Forecast',
                        line=dict(dash='dash', color=COLOR_SCHEME['secondary']))
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text']),
        title_x=0.5,
        hovermode='x unified'
    )
    return fig

def plot_operational_resilience(resilience: list) -> go.Figure:
    """
    Plot operational resilience over time.

    Args:
        resilience (list): Resilience scores.

    Returns:
        go.Figure: Plotly figure.
    """
    logger.info("Plotting operational resilience")
    fig = px.line(
        x=list(range(len(resilience))),
        y=resilience,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Resilience Score'},
        title='Operational Resilience',
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_SCHEME['primary']]
    )
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text']),
        title_x=0.5,
        hovermode='x unified'
    )
    return fig

def plot_operational_efficiency(efficiency_df: pd.DataFrame) -> go.Figure:
    """
    Plot operational efficiency metrics (uptime, throughput, quality, OEE).

    Args:
        efficiency_df (pd.DataFrame): DataFrame with efficiency metrics.

    Returns:
        go.Figure: Plotly figure.
    """
    logger.info("Plotting operational efficiency with DataFrame")
    fig = px.line(
        efficiency_df,
        x=efficiency_df.index,
        y=['uptime', 'throughput', 'quality', 'oee'],
        labels={'value': 'Efficiency', 'index': 'Shift Interval (2-min)'},
        title='Operational Efficiency Metrics',
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['accent'], COLOR_SCHEME['danger']]
    )
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text']),
        title_x=0.5,
        hovermode='x unified',
        legend_title_text='Metric'
    )
    return fig

def plot_oee_gauge(oee: float) -> go.Figure:
    """
    Plot a gauge chart for Overall Equipment Effectiveness (OEE).

    Args:
        oee (float): OEE value (0 to 1).

    Returns:
        go.Figure: Plotly gauge chart.
    """
    logger.info("Plotting OEE gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=oee * 100,
        title={'text': "Overall Equipment Effectiveness (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': COLOR_SCHEME['primary']},
            'steps': [
                {'range': [0, 60], 'color': COLOR_SCHEME['danger']},
                {'range': [60, 80], 'color': COLOR_SCHEME['secondary']},
                {'range': [80, 100], 'color': COLOR_SCHEME['accent']}
            ],
            'threshold': {
                'line': {'color': COLOR_SCHEME['text'], 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text']),
        template=PLOTLY_TEMPLATE
    )
    return fig

def plot_worker_distribution(team_positions_df: pd.DataFrame, workplace_size: float, config: dict) -> go.Figure:
    """
    Plot animated team distribution by zone.

    Args:
        team_positions_df (pd.DataFrame): DataFrame with team positions.
        workplace_size (float): Size of the workplace.
        config (dict): Configuration dictionary.

    Returns:
        go.Figure: Plotly animated scatter plot.
    """
    logger.info("Plotting worker distribution with DataFrame")
    fig = px.scatter(
        team_positions_df,
        x='x', y='y',
        animation_frame='step',
        color='zone',
        hover_data=['team_member_id'],
        range_x=[0, workplace_size],
        range_y=[0, workplace_size],
        title=f"Team Distribution ({config['FACILITY_TYPE'].capitalize()} Workplace)",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    for zone, info in config['WORK_AREAS'].items():
        fig.add_scatter(x=[info['center'][0]], y=[info['center'][1]], mode='markers+text',
                        text=[info['label']], marker=dict(size=15, color=COLOR_SCHEME['danger']), name=zone)
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text']),
        title_x=0.5,
        hovermode='closest',
        showlegend=True
    )
    return fig

def plot_worker_wellbeing(wellbeing_scores: list) -> go.Figure:
    """
    Plot team well-being scores over time.

    Args:
        wellbeing_scores (list): Well-being scores.

    Returns:
        go.Figure: Plotly figure.
    """
    logger.info("Plotting worker well-being")
    fig = px.line(
        x=list(range(len(wellbeing_scores))),
        y=wellbeing_scores,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Well-Being Score'},
        title='Team Well-Being',
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_SCHEME['primary']]
    )
    fig.add_hline(y=DEFAULT_CONFIG['WELLBEING_THRESHOLD'], line_dash="dash", line_color=COLOR_SCHEME['danger'],
                  annotation_text="Threshold")
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text']),
        title_x=0.5,
        hovermode='x unified'
    )
    return fig

def plot_psychological_safety(safety_scores: list) -> go.Figure:
    """
    Plot psychological safety scores over time.

    Args:
        safety_scores (list): Safety scores.

    Returns:
        go.Figure: Plotly figure.
    """
    logger.info("Plotting psychological safety")
    fig = px.line(
        x=list(range(len(safety_scores))),
        y=safety_scores,
        labels={'x': 'Shift Interval (2-min)', 'y': 'Safety Score'},
        title='Psychological Safety',
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[COLOR_SCHEME['primary']]
    )
    fig.add_hline(y=DEFAULT_CONFIG['SAFETY_COMPLIANCE_THRESHOLD'], line_dash="dash", line_color=COLOR_SCHEME['danger'],
                  annotation_text="Threshold")
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text']),
        title_x=0.5,
        hovermode='x unified'
    )
    return fig
