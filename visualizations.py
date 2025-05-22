# -*- coding: utf-8 -*-
"""
visualizations.py
Advanced visualization functions for the Industrial Workplace Shift Monitoring Dashboard.
Generates sophisticated, interactive Plotly charts with statistical insights.
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dashboard.log'
)

# Plotly template for dark theme
PLOTLY_TEMPLATE = "plotly_dark"

# Refined color scheme (colorblind-friendly, high contrast)
COLOR_SCHEME = {
    'background': '#1E2A44',  # Dark navy
    'text': '#F5F6F5',        # Soft white
    'primary': '#636EFA',     # Bright blue
    'secondary': '#EF553B',   # Coral
    'accent': '#00CC96',      # Teal
    'danger': '#F15C80',      # Pink-red
    'purple': '#AB63FA',      # Purple
    'orange': '#FFA15A',      # Orange
    'cyan': '#19D3F3'         # Cyan
}

def plot_task_compliance_trend(compliance_data: list, disruption_intervals: list, forecast: list = None, z_scores: list = None) -> go.Figure:
    """
    Plot task compliance with rolling average, anomalies, and confidence interval.
    """
    logger.info("Defining plot_task_compliance_trend")
    try:
        fig = go.Figure()
        
        # Rolling average
        rolling_avg = pd.Series(compliance_data).rolling(window=10, min_periods=1).mean()
        
        # Main compliance line
        fig.add_trace(go.Scatter(
            x=list(range(len(compliance_data))),
            y=compliance_data,
            mode='lines',
            name='Compliance Entropy',
            line=dict(color=COLOR_SCHEME['primary'], width=2),
            hovertemplate='Interval: %{x}<br>Entropy: %{y:.2f}'
        ))
        
        # Rolling average
        fig.add_trace(go.Scatter(
            x=list(range(len(compliance_data))),
            y=rolling_avg,
            mode='lines',
            name='Rolling Avg',
            line=dict(color=COLOR_SCHEME['orange'], width=3, dash='dot'),
            hovertemplate='Interval: %{x}<br>Rolling Avg: %{y:.2f}'
        ))
        
        # Anomalies
        if z_scores is not None:
            anomalies = [i for i, z in enumerate(z_scores) if abs(z) > DEFAULT_CONFIG['ANOMALY_THRESHOLD']]
            fig.add_trace(go.Scatter(
                x=anomalies,
                y=[compliance_data[i] for i in anomalies],
                mode='markers',
                name='Anomalies',
                marker=dict(color=COLOR_SCHEME['danger'], size=10, symbol='x'),
                hovertemplate='Interval: %{x}<br>Entropy: %{y:.2f}<br>Z-Score: %{text:.2f}',
                text=[z_scores[i] for i in anomalies]
            ))
        
        # Forecast with confidence interval
        if forecast is not None:
            forecast_series = pd.Series(forecast)
            ci = 1.96 * forecast_series.std() / np.sqrt(len(forecast))
            fig.add_trace(go.Scatter(
                x=list(range(len(forecast))),
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color=COLOR_SCHEME['secondary'], width=2, dash='dash'),
                hovertemplate='Interval: %{x}<br>Forecast: %{y:.2f}'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(forecast))),
                y=forecast_series + ci,
                mode='lines',
                line=dict(color=COLOR_SCHEME['secondary'], width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(forecast))),
                y=forecast_series - ci,
                mode='lines',
                line=dict(color=COLOR_SCHEME['secondary'], width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(239, 85, 59, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Disruption markers
        for t in disruption_intervals:
            fig.add_vline(x=t, line_dash="dash", line_color=COLOR_SCHEME['danger'], opacity=0.7,
                          annotation_text="Disruption", annotation_position="top left")
        
        fig.update_layout(
            title=dict(text='Task Compliance Variability (Advanced)', x=0.5, font_size=22),
            xaxis_title='Shift Interval (2-min)',
            yaxis_title='Compliance Entropy',
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_task_compliance_trend: {str(e)}")
        raise

def plot_worker_collaboration_trend(collab_data: list, disruption_intervals: list, forecast: list = None) -> go.Figure:
    """
    Plot collaboration index with bar overlay for disruptions.
    """
    logger.info("Defining plot_worker_collaboration_trend")
    try:
        fig = go.Figure()
        
        # Collaboration line
        fig.add_trace(go.Scatter(
            x=list(range(len(collab_data))),
            y=collab_data,
            mode='lines+markers',
            name='Collaboration Strength',
            line=dict(color=COLOR_SCHEME['accent'], width=3),
            marker=dict(size=8, line=dict(width=1, color=COLOR_SCHEME['text'])),
            hovertemplate='Interval: %{x}<br>Strength: %{y:.2f}'
        ))
        
        # Disruption bars
        disruption_values = [1 if i in disruption_intervals else 0 for i in range(len(collab_data))]
        fig.add_trace(go.Bar(
            x=list(range(len(collab_data))),
            y=disruption_values,
            name='Disruptions',
            marker_color=COLOR_SCHEME['danger'],
            opacity=0.3,
            yaxis='y2',
            hovertemplate='Interval: %{x}<br>Disruption: %{y}'
        ))
        
        # Forecast
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=list(range(len(forecast))),
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color=COLOR_SCHEME['cyan'], width=2, dash='dot'),
                hovertemplate='Interval: %{x}<br>Forecast: %{y:.2f}'
            ))
        
        fig.update_layout(
            title=dict(text='Worker Collaboration Index (Advanced)', x=0.5, font_size=22),
            xaxis_title='Shift Interval (2-min)',
            yaxis_title='Collaboration Strength',
            yaxis2=dict(title='Disruption Events', overlaying='y', side='right', showgrid=False, range=[0, 1.5]),
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            showlegend=True
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_worker_collaboration_trend: {str(e)}")
        raise

def plot_operational_resilience(resilience: list, productivity_loss: list) -> go.Figure:
    """
    Plot resilience with productivity loss on secondary axis.
    """
    logger.info("Defining plot_operational_resilience")
    try:
        fig = go.Figure()
        
        # Resilience filled area
        fig.add_trace(go.Scatter(
            x=list(range(len(resilience))),
            y=resilience,
            mode='lines',
            name='Resilience Score',
            line=dict(color=COLOR_SCHEME['cyan'], width=3),
            fill='tozeroy',
            fillcolor='rgba(25, 211, 243, 0.2)',
            hovertemplate='Interval: %{x}<br>Resilience: %{y:.2f}'
        ))
        
        # Productivity loss
        fig.add_trace(go.Scatter(
            x=list(range(len(productivity_loss))),
            y=productivity_loss,
            mode='lines',
            name='Productivity Loss (%)',
            line=dict(color=COLOR_SCHEME['danger'], width=2),
            yaxis='y2',
            hovertemplate='Interval: %{x}<br>Loss: %{y:.2f}%'
        ))
        
        fig.update_layout(
            title=dict(text='Operational Resilience vs Productivity Loss', x=0.5, font_size=22),
            xaxis_title='Shift Interval (2-min)',
            yaxis_title='Resilience Score',
            yaxis2=dict(title='Productivity Loss (%)', overlaying='y', side='right', showgrid=False),
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background']
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_operational_resilience: {str(e)}")
        raise

def plot_operational_efficiency(efficiency_df: pd.DataFrame, selected_metrics: list = None) -> go.Figure:
    """
    Plot efficiency metrics with composite line/bar.
    """
    logger.info("Defining plot_operational_efficiency")
    try:
        metrics = selected_metrics or ['uptime', 'throughput', 'quality', 'oee']
        colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['accent'], COLOR_SCHEME['purple']]
        
        fig = go.Figure()
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Scatter(
                x=efficiency_df.index,
                y=efficiency_df[metric],
                mode='lines+markers',
                name=f'{metric.capitalize()} (Trend)',
                line=dict(color=color, width=3),
                marker=dict(size=6),
                hovertemplate=f'{metric.capitalize()}: %{{y:.2f}}<br>Interval: %{{x}}'
            ))
            rolling_avg = efficiency_df[metric].rolling(window=10, min_periods=1).mean()
            fig.add_trace(go.Bar(
                x=efficiency_df.index,
                y=rolling_avg,
                name=f'{metric.capitalize()} (Avg)',
                marker_color=color,
                opacity=0.3,
                hovertemplate=f'{metric.capitalize()} Avg: %{{y:.2f}}<br>Interval: %{{x}}'
            ))
        
        stats = efficiency_df[metrics].mean().to_dict()
        annotation_text = "<br>".join([f"{m.capitalize()}: {v:.2f}" for m, v in stats.items()])
        fig.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"Mean Metrics:<br>{annotation_text}",
            showarrow=False, font=dict(color=COLOR_SCHEME['text'], size=12),
            bgcolor='rgba(0,0,0,0.5)', bordercolor=COLOR_SCHEME['text']
        )
        
        fig.update_layout(
            title=dict(text='Operational Efficiency Metrics (Composite)', x=0.5, font_size=22),
            xaxis_title='Shift Interval (2-min)',
            yaxis_title='Efficiency',
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            legend_title_text='Metric',
            barmode='overlay'
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_operational_efficiency: {str(e)}")
        raise

def plot_oee_gauge(oee: float, benchmark: float = 0.85) -> go.Figure:
    """
    Plot OEE gauge with dynamic needle.
    """
    logger.info("Defining plot_oee_gauge")
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=oee * 100,
            delta={'reference': benchmark * 100, 'increasing': {'color': COLOR_SCHEME['accent']}, 'decreasing': {'color': COLOR_SCHEME['danger']}},
            title={'text': "Overall Equipment Effectiveness (%)", 'font': {'size': 22}},
            number={'font': {'size': 30, 'color': COLOR_SCHEME['text']}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': COLOR_SCHEME['text'], 'tickfont': {'color': COLOR_SCHEME['text']}},
                'bar': {'color': COLOR_SCHEME['accent'], 'line': {'color': COLOR_SCHEME['text'], 'width': 2}},
                'steps': [
                    {'range': [0, 60], 'color': COLOR_SCHEME['danger']},
                    {'range': [60, 80], 'color': COLOR_SCHEME['orange']},
                    {'range': [80, 100], 'color': COLOR_SCHEME['accent']}
                ],
                'threshold': {
                    'line': {'color': COLOR_SCHEME['cyan'], 'width': 4},
                    'thickness': 0.75,
                    'value': benchmark * 100
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
    except Exception as e:
        logger.error(f"Error in plot_oee_gauge: {str(e)}")
        raise

def plot_worker_distribution(team_positions_df: pd.DataFrame, workplace_size: float, config: dict, use_3d: bool = False) -> go.Figure:
    """
    Plot 3D or 2D animated scatter for team distribution.
    """
    logger.info("Defining plot_worker_distribution")
    try:
        if use_3d:
            fig = go.Figure()
            for zone in team_positions_df['zone'].unique():
                df_zone = team_positions_df[team_positions_df['zone'] == zone]
                fig.add_trace(go.Scatter3d(
                    x=df_zone['x'], y=df_zone['y'], z=df_zone['step'],
                    mode='markers',
                    name=zone,
                    marker=dict(
                        size=5,
                        color=COLOR_SCHEME['primary'] if zone == 'Assembly Line' else COLOR_SCHEME['secondary'] if zone == 'Packaging Zone' else COLOR_SCHEME['cyan'],
                        opacity=0.7
                    ),
                    hovertemplate='ID: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Step: %{z}',
                    text=df_zone['team_member_id']
                ))
            
            for zone, info in config['WORK_AREAS'].items():
                fig.add_trace(go.Scatter3d(
                    x=[info['center'][0]], y=[info['center'][1]], z=[0],
                    mode='markers+text',
                    text=[info['label']],
                    marker=dict(size=10, color=COLOR_SCHEME['danger']),
                    textfont=dict(color=COLOR_SCHEME['text'])
                ))
            
            fig.update_layout(
                title=dict(text=f"Team Distribution (3D - {config['FACILITY_TYPE'].capitalize()})", x=0.5, font_size=22),
                scene=dict(
                    xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Step',
                    xaxis=dict(range=[0, workplace_size]),
                    yaxis=dict(range=[0, workplace_size]),
                    zaxis=dict(range=[0, max(team_positions_df['step'])]),
                    bgcolor=COLOR_SCHEME['background']
                ),
                font=dict(color=COLOR_SCHEME['text'], size=14),
                template=PLOTLY_TEMPLATE,
                showlegend=True
            )
        else:
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
                color_discrete_sequence=[COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['cyan']]
            )
            fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color=COLOR_SCHEME['text'])))
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
                title=dict(x=0.5, font_size=22),
                font=dict(color=COLOR_SCHEME['text'], size=14),
                hovermode='closest',
                showlegend=True,
                plot_bgcolor=COLOR_SCHEME['background'],
                paper_bgcolor=COLOR_SCHEME['background'],
                transition={'duration': 500, 'easing': 'cubic-in-out'}
            )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_worker_distribution: {str(e)}")
        raise

def plot_worker_density_heatmap(team_positions_df: pd.DataFrame, workplace_size: float, config: dict) -> go.Figure:
    """
    Plot heatmap of worker density by zone.
    """
    logger.info("Defining plot_worker_density_heatmap")
    try:
        grid_size = config['DENSITY_GRID_SIZE']
        x_bins = np.linspace(0, workplace_size, grid_size)
        y_bins = np.linspace(0, workplace_size, grid_size)
        
        heatmap_data = np.zeros((grid_size-1, grid_size-1))
        for _, row in team_positions_df.iterrows():
            x_idx = np.searchsorted(x_bins, row['x'], side='right') - 1
            y_idx = np.searchsorted(y_bins, row['y'], side='right') - 1
            if 0 <= x_idx < grid_size-1 and 0 <= y_idx < grid_size-1:
                heatmap_data[y_idx, x_idx] += 1
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=x_bins[:-1], y=y_bins[:-1],
            colorscale='Viridis',
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Density: %{z}'
        ))
        
        for zone, info in config['WORK_AREAS'].items():
            fig.add_scatter(
                x=[info['center'][0]], y=[info['center'][1]],
                mode='markers+text',
                text=[info['label']],
                marker=dict(size=15, color=COLOR_SCHEME['danger']),
                textfont=dict(color=COLOR_SCHEME['text'])
            )
        
        fig.update_layout(
            title=dict(text=f"Worker Density Heatmap ({config['FACILITY_TYPE'].capitalize()})", x=0.5, font_size=22),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background']
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_worker_density_heatmap: {str(e)}")
        raise

def plot_worker_wellbeing(wellbeing_scores: list, triggers: dict) -> go.Figure:
    """
    Plot well-being with trend line and trigger annotations.
    """
    logger.info("Defining plot_worker_wellbeing")
    try:
        fig = go.Figure()
        
        # Well-being line
        fig.add_trace(go.Scatter(
            x=list(range(len(wellbeing_scores))),
            y=wellbeing_scores,
            mode='lines',
            name='Well-Being Score',
            line=dict(color=COLOR_SCHEME['purple'], width=3),
            hovertemplate='Interval: %{x}<br>Score: %{y:.2f}'
        ))
        
        # Trend line
        x = np.arange(len(wellbeing_scores))
        trend = np.polyval(np.polyfit(x, wellbeing_scores, 1), x)
        fig.add_trace(go.Scatter(
            x=x, y=trend,
            mode='lines',
            name='Trend',
            line=dict(color=COLOR_SCHEME['orange'], width=2, dash='dash'),
            hovertemplate='Interval: %{x}<br>Trend: %{y:.2f}'
        ))
        
        # Threshold triggers
        for t in triggers['threshold']:
            fig.add_trace(go.Scatter(
                x=[t], y=[wellbeing_scores[t]],
                mode='markers',
                name='Threshold Alert',
                marker=dict(color=COLOR_SCHEME['danger'], size=10, symbol='x'),
                hovertemplate='Interval: %{x}<br>Score: %{y:.2f}',
                showlegend=False
            ))
        
        fig.add_hline(
            y=DEFAULT_CONFIG['WELLBEING_THRESHOLD'],
            line_dash="dash",
            line_color=COLOR_SCHEME['danger'],
            annotation_text="Threshold",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title=dict(text='Team Well-Being (Advanced)', x=0.5, font_size=22),
            xaxis_title='Shift Interval (2-min)',
            yaxis_title='Well-Being Score',
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background']
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_worker_wellbeing: {str(e)}")
        raise

def plot_psychological_safety(safety_scores: list) -> go.Figure:
    """
    Plot psychological safety with trend line and anomaly annotations.
    """
    logger.info("Defining plot_psychological_safety")
    try:
        fig = go.Figure()
        
        # Safety line
        fig.add_trace(go.Scatter(
            x=list(range(len(safety_scores))),
            y=safety_scores,
            mode='lines',
            name='Safety Score',
            line=dict(color=COLOR_SCHEME['cyan'], width=3),
            hovertemplate='Interval: %{x}<br>Score: %{y:.2f}'
        ))
        
        # Trend line
        x = np.arange(len(safety_scores))
        trend = np.polyval(np.polyfit(x, safety_scores, 1), x)
        fig.add_trace(go.Scatter(
            x=x, y=trend,
            mode='lines',
            name='Trend',
            line=dict(color=COLOR_SCHEME['orange'], width=2, dash='dash'),
            hovertemplate='Interval: %{x}<br>Trend: %{y:.2f}'
        ))
        
        # Anomaly detection
        z_scores = (safety_scores - np.mean(safety_scores)) / np.std(safety_scores)
        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > DEFAULT_CONFIG['ANOMALY_THRESHOLD']]
        fig.add_trace(go.Scatter(
            x=anomalies,
            y=[safety_scores[i] for i in anomalies],
            mode='markers',
            name='Anomalies',
            marker=dict(color=COLOR_SCHEME['danger'], size=10, symbol='x'),
            hovertemplate='Interval: %{x}<br>Score: %{y:.2f}<br>Z-Score: %{text:.2f}',
            text=[z_scores[i] for i in anomalies]
        ))
        
        fig.add_hline(
            y=DEFAULT_CONFIG['SAFETY_COMPLIANCE_THRESHOLD'],
            line_dash="dash",
            line_color=COLOR_SCHEME['danger'],
            annotation_text="Threshold",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title=dict(text='Psychological Safety (Advanced)', x=0.5, font_size=22),
            xaxis_title='Shift Interval (2-min)',
            yaxis_title='Safety Score',
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background']
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_psychological_safety: {str(e)}")
        raise
