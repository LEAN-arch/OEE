"""
visualizations.py
Visualization functions for the Industrial Workplace Shift Monitoring Dashboard.
Generates Plotly charts with enhanced styling and accessibility features.
"""

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Plotly template for dark theme
PLOTLY_TEMPLATE = "plotly_dark"

# Enhanced color scheme (colorblind-friendly, high contrast)
COLOR_SCHEME = {
    'background': '#1E2A44',  # Dark navy
    'text': '#E6E9F0',        # Light gray
    'primary': '#4C78A8',     # Muted blue
    'secondary': '#F58518',   # Vibrant orange
    'accent': '#54A24B',      # Green
    'danger': '#E45756',      # Red
    'purple': '#9B59B6',      # Purple (for multi-series)
    'teal': '#1ABC9C',        # Teal (for multi-series)
    'yellow': '#F1C40F'       # Yellow (for multi-series)
}

def plot_task_compliance_trend(compliance_data: list, disruption_intervals: list, forecast: list = None) -> go.Figure:
    """
    Plot task compliance variability with enhanced styling and forecast.

    Args:
        compliance_data (list): Compliance entropy values.
        disruption_intervals (list): Time steps for disruptions.
        forecast (list, optional): Forecasted compliance values.

    Returns:
        go.Figure: Plotly figure with modern styling.
    """
    logger.info("Plotting task compliance trend")
    fig = go.Figure()
    
    # Main compliance line
    fig.add_trace(go.Scatter(
        x=list(range(len(compliance_data))),
        y=compliance_data,
        mode='lines',
        name='Compliance Entropy',
        line=dict(color=COLOR_SCHEME['primary'], width=3),
        hovertemplate='Interval: %{x}<br>Entropy: %{y:.2f}<extra></extra>'
    ))
    
    # Forecast line
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=list(range(len(forecast))),
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color=COLOR_SCHEME['secondary'], width=2, dash='dash'),
            hovertemplate='Interval: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
        ))
    
    # Disruption markers
    for t in disruption_intervals:
        fig.add_vline(x=t, line_dash="dash", line_color=COLOR_SCHEME['danger'], opacity=0.7,
                      annotation_text="Disruption", annotation_position="top left")
    
    fig.update_layout(
        title=dict(text='Task Compliance Variability', x=0.5, font_size=20),
        xaxis_title='Shift Interval (2-min)',
        yaxis_title='Compliance Entropy',
        font=dict(color=COLOR_SCHEME['text'], size=14),
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        showlegend=True
    )
    return fig

def plot_worker_collaboration_trend(collab_data: list, forecast: list = None) -> go.Figure:
    """
    Plot worker collaboration index with enhanced styling and forecast.

    Args:
        collab_data (list): Collaboration index values.
        forecast (list, optional): Forecasted collaboration values.

    Returns:
        go.Figure: Plotly figure with modern styling.
    """
    logger.info("Plotting worker collaboration trend")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(collab_data))),
        y=collab_data,
        mode='lines+markers',
        name='Collaboration Strength',
        line=dict(color=COLOR_SCHEME['accent'], width=3),
        marker=dict(size=8, line=dict(width=1, color=COLOR_SCHEME['text'])),
        hovertemplate='Interval: %{x}<br>Strength: %{y:.2f}<extra></extra>'
    ))
    
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=list(range(len(forecast))),
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color=COLOR_SCHEME['yellow'], width=2, dash='dot'),
            hovertemplate='Interval: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='Worker Collaboration Index', x=0.5, font_size=20),
        xaxis_title='Shift Interval (2-min)',
        yaxis_title='Collaboration Strength',
        font=dict(color=COLOR_SCHEME['text'], size=14),
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background']
    )
    return fig

def plot_operational_resilience(resilience: list) -> go.Figure:
    """
    Plot operational resilience over time with gradient fill.

    Args:
        resilience (list): Resilience scores.

    Returns:
        go.Figure: Plotly figure with modern styling.
    """
    logger.info("Plotting operational resilience")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(resilience))),
        y=resilience,
        mode='lines',
        name='Resilience Score',
        line=dict(color=COLOR_SCHEME['teal'], width=3),
        fill='tozeroy',
        fillcolor='rgba(26, 188, 156, 0.2)',
        hovertemplate='Interval: %{x}<br>Score: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Operational Resilience', x=0.5, font_size=20),
        xaxis_title='Shift Interval (2-min)',
        yaxis_title='Resilience Score',
        font=dict(color=COLOR_SCHEME['text'], size=14),
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background']
    )
    return fig

def plot_operational_efficiency(efficiency_df: pd.DataFrame) -> go.Figure:
    """
    Plot operational efficiency metrics with distinct colors and hover effects.

    Args:
        efficiency_df (pd.DataFrame): DataFrame with efficiency metrics.

    Returns:
        go.Figure: Plotly figure with modern styling.
    """
    logger.info("Plotting operational efficiency with DataFrame")
    fig = go.Figure()
    
    metrics = ['uptime', 'throughput', 'quality', 'oee']
    colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['accent'], COLOR_SCHEME['purple']]
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Scatter(
            x=efficiency_df.index,
            y=efficiency_df[metric],
            mode='lines+markers',
            name=metric.capitalize(),
            line=dict(color=color, width=3),
            marker=dict(size=6, line=dict(width=1, color=COLOR_SCHEME['text'])),
            hovertemplate=f'{metric.capitalize()}: %{{y:.2f}}<br>Interval: %{{x}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='Operational Efficiency Metrics', x=0.5, font_size=20),
        xaxis_title='Shift Interval (2-min)',
        yaxis_title='Efficiency',
        font=dict(color=COLOR_SCHEME['text'], size=14),
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        legend_title_text='Metric'
    )
    return fig

def plot_oee_gauge(oee: float) -> go.Figure:
    """
    Plot a gauge chart for Overall Equipment Effectiveness with gradient bar.

    Args:
        oee (float): OEE value (0 to 1).

    Returns:
        go.Figure: Plotly gauge chart with modern styling.
    """
    logger.info("Plotting OEE gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=oee * 100,
        title={'text': "Overall Equipment Effectiveness (%)", 'font': {'size': 20}},
        number={'font': {'size': 30, 'color': COLOR_SCHEME['text']}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLOR_SCHEME['text'], 'tickfont': {'color': COLOR_SCHEME['text']}},
            'bar': {'color': COLOR_SCHEME['accent'], 'line': {'color': COLOR_SCHEME['text'], 'width': 2}},
            'steps': [
                {'range': [0, 60], 'color': COLOR_SCHEME['danger']},
                {'range': [60, 80], 'color': COLOR_SCHEME['secondary']},
                {'range': [80, 100], 'color': COLOR_SCHEME['accent']}
            ],
            'threshold': {
                'line': {'color': COLOR_SCHEME['yellow'], 'width': 4},
                'thickness': 0.75,
                'value': 80
            },
            'bgcolor': COLOR_SCHEME['background']
        }
    ))
    
    fig.update_layout(
        font=dict(color=COLOR_SCHEME['text'], size=14),
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background']
    )
    return fig

def plot_worker_distribution(team_positions_df: pd.DataFrame, workplace_size: float, config: dict) -> go.Figure:
    """
    Plot animated team distribution by zone with enhanced markers and transitions.

    Args:
        team_positions_df (pd.DataFrame): DataFrame with team positions.
        workplace_size (float): Size of the workplace.
        config (dict): Configuration dictionary.

    Returns:
        go.Figure: Plotly animated scatter plot with modern styling.
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
        color_discrete_sequence=[COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['teal']]
    )
    
    # Enhance markers and animations
    fig.update_traces(
        marker=dict(size=12, opacity=0.8, line=dict(width=1, color=COLOR_SCHEME['text'])),
        selector=dict(mode='markers')
    )
    
    # Add zone centers
    for zone, info in config['WORK_AREAS'].items():
        fig.add_scatter(
            x=[info['center'][0]], y=[info['center'][1]],
            mode='markers+text',
            text=[info['label']],
            marker=dict(size=20, color=COLOR_SCHEME['danger'], symbol='star'),
            name=zone,
            textfont=dict(color=COLOR_SCHEME['text'], size=14)
        )
    
    fig.update_layout(
        title=dict(x=0.5, font_size=20),
        font=dict(color=COLOR_SCHEME['text'], size=14),
        hovermode='closest',
        showlegend=True,
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background'],
        transition={'duration': 500, 'easing': 'cubic-in-out'}
    )
    return fig

def plot_worker_wellbeing(wellbeing_scores: list) -> go.Figure:
    """
    Plot team well-being scores with threshold line and annotations.

    Args:
        wellbeing_scores (list): Well-being scores.

    Returns:
        go.Figure: Plotly figure with modern styling.
    """
    logger.info("Plotting worker well-being")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(wellbeing_scores))),
        y=wellbeing_scores,
        mode='lines',
        name='Well-Being Score',
        line=dict(color=COLOR_SCHEME['purple'], width=3),
        hovertemplate='Interval: %{x}<br>Score: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(
        y=DEFAULT_CONFIG['WELLBEING_THRESHOLD'],
        line_dash="dash",
        line_color=COLOR_SCHEME['danger'],
        annotation_text="Threshold",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_SCHEME['text'])
    )
    
    fig.update_layout(
        title=dict(text='Team Well-Being', x=0.5, font_size=20),
        xaxis_title='Shift Interval (2-min)',
        yaxis_title='Well-Being Score',
        font=dict(color=COLOR_SCHEME['text'], size=14),
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background']
    )
    return fig

def plot_psychological_safety(safety_scores: list) -> go.Figure:
    """
    Plot psychological safety scores with threshold line and annotations.

    Args:
        safety_scores (list): Safety scores.

    Returns:
        go.Figure: Plotly figure with modern styling.
    """
    logger.info("Plotting psychological safety")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(safety_scores))),
        y=safety_scores,
        mode='lines',
        name='Safety Score',
        line=dict(color=COLOR_SCHEME['teal'], width=3),
        hovertemplate='Interval: %{x}<br>Score: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(
        y=DEFAULT_CONFIG['SAFETY_COMPLIANCE_THRESHOLD'],
        line_dash="dash",
        line_color=COLOR_SCHEME['danger'],
        annotation_text="Threshold",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_SCHEME['text'])
    )
    
    fig.update_layout(
        title=dict(text='Psychological Safety', x=0.5, font_size=20),
        xaxis_title='Shift Interval (2-min)',
        yaxis_title='Safety Score',
        font=dict(color=COLOR_SCHEME['text'], size=14),
        template=PLOTLY_TEMPLATE,
        hovermode='x unified',
        plot_bgcolor=COLOR_SCHEME['background'],
        paper_bgcolor=COLOR_SCHEME['background']
    )
    return fig
