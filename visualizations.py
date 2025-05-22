"""
visualizations.py
Advanced visualization functions for the Industrial Workplace Shift Monitoring Dashboard.
Generates professional, interactive Plotly charts with actionable insights.
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dashboard.log'
)

PLOTLY_TEMPLATE = "plotly_dark"

COLOR_SCHEME = {
    'background': '#1A252F',  # Dark navy
    'text': '#E6ECEF',        # Soft white
    'primary': '#3B82F6',     # Vibrant blue
    'secondary': '#EC4899',   # Bold magenta
    'accent': '#10B981',      # Emerald green
    'warning': '#F59E0B',     # Amber
    'danger': '#EF4444',      # Strong red
    'neutral': '#6B7280'      # Gray
}

def plot_task_compliance_score(compliance_data: list, disruption_intervals: list, forecast: list = None, z_scores: list = None) -> go.Figure:
    """
    Plot task compliance score with rolling average, anomalies, and forecast.
    """
    logger.info("Defining plot_task_compliance_score")
    try:
        minutes = [i * 2 for i in range(len(compliance_data))]
        fig = go.Figure()
        
        rolling_avg = pd.Series(compliance_data).rolling(window=10, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=minutes,
            y=compliance_data,
            mode='lines',
            name='Task Compliance Score',
            line=dict(color=COLOR_SCHEME['primary'], width=2),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%'
        ))
        
        fig.add_trace(go.Scatter(
            x=minutes,
            y=rolling_avg,
            mode='lines',
            name='Rolling Average',
            line=dict(color=COLOR_SCHEME['warning'], width=3, dash='dot'),
            hovertemplate='Time: %{x} min<br>Average: %{y:.1f}%'
        ))
        
        if z_scores is not None:
            anomalies = [i for i, z in enumerate(z_scores) if abs(z) > DEFAULT_CONFIG['ANOMALY_THRESHOLD']]
            fig.add_trace(go.Scatter(
                x=[minutes[i] for i in anomalies],
                y=[compliance_data[i] for i in anomalies],
                mode='markers',
                name='Anomalies',
                marker=dict(color=COLOR_SCHEME['danger'], size=10, symbol='x'),
                hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<br>Z-Score: %{text:.2f}',
                text=[z_scores[i] for i in anomalies]
            ))
            for i in anomalies:
                fig.add_annotation(
                    x=minutes[i], y=compliance_data[i],
                    text="Low compliance; review tasks",
                    showarrow=True, arrowhead=1, ax=20, ay=-30,
                    font=dict(color=COLOR_SCHEME['danger'])
                )
        
        if forecast is not None:
            forecast_series = pd.Series(forecast)
            ci = 1.96 * forecast_series.std() / np.sqrt(len(forecast))
            fig.add_trace(go.Scatter(
                x=minutes,
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color=COLOR_SCHEME['secondary'], width=2, dash='dash'),
                hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%'
            ))
            fig.add_trace(go.Scatter(
                x=minutes,
                y=forecast_series + ci,
                mode='lines',
                line=dict(color=COLOR_SCHEME['secondary'], width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=minutes,
                y=forecast_series - ci,
                mode='lines',
                line=dict(color=COLOR_SCHEME['secondary'], width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(236, 72, 153, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        for t in disruption_intervals:
            fig.add_vline(x=t*2, line_dash="dash", line_color=COLOR_SCHEME['danger'], opacity=0.7,
                          annotation_text="Disruption", annotation_position="top left")
        
        fig.update_layout(
            title=dict(text='Task Compliance Score Over Time', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Compliance Score (%)',
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
        logger.error(f"Error in plot_task_compliance_score: {str(e)}")
        raise

def plot_collaboration_proximity_index(collab_data: list, disruption_intervals: list, forecast: list = None) -> go.Figure:
    """
    Plot collaboration proximity index with disruption bars.
    """
    logger.info("Defining plot_collaboration_proximity_index")
    try:
        minutes = [i * 2 for i in range(len(collab_data))]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=minutes,
            y=collab_data,
            mode='lines+markers',
            name='Collaboration Proximity',
            line=dict(color=COLOR_SCHEME['accent'], width=3),
            marker=dict(size=8, line=dict(width=1, color=COLOR_SCHEME['text'])),
            hovertemplate='Time: %{x} min<br>Proximity: %{y:.1f}%'
        ))
        
        disruption_values = [1 if i in disruption_intervals else 0 for i in range(len(collab_data))]
        fig.add_trace(go.Bar(
            x=minutes,
            y=disruption_values,
            name='Disruptions',
            marker_color=COLOR_SCHEME['danger'],
            opacity=0.3,
            yaxis='y2',
            hovertemplate='Time: %{x} min<br>Disruption: %{y}'
        ))
        
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=minutes,
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color=COLOR_SCHEME['neutral'], width=2, dash='dot'),
                hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%'
            ))
        
        fig.update_layout(
            title=dict(text='Collaboration Proximity Index', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Proximity Index (%)',
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
        logger.error(f"Error in plot_collaboration_proximity_index: {str(e)}")
        raise

def plot_operational_recovery(recovery: list, productivity_loss: list) -> go.Figure:
    """
    Plot operational recovery with productivity loss.
    """
    logger.info("Defining plot_operational_recovery")
    try:
        minutes = [i * 2 for i in range(len(recovery))]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=minutes,
            y=recovery,
            mode='lines',
            name='Recovery Score',
            line=dict(color=COLOR_SCHEME['accent'], width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.2)',
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%'
        ))
        
        fig.add_trace(go.Scatter(
            x=minutes,
            y=productivity_loss,
            mode='lines',
            name='Productivity Loss',
            line=dict(color=COLOR_SCHEME['danger'], width=2),
            yaxis='y2',
            hovertemplate='Time: %{x} min<br>Loss: %{y:.1f}%'
        ))
        
        fig.update_layout(
            title=dict(text='Operational Recovery vs Productivity Loss', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Recovery Score (%)',
            yaxis2=dict(title='Productivity Loss (%)', overlaying='y', side='right', showgrid=False),
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background']
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_operational_recovery: {str(e)}")
        raise

def plot_operational_efficiency(efficiency_df: pd.DataFrame, selected_metrics: list = None) -> go.Figure:
    """
    Plot efficiency metrics with composite line/bar.
    """
    logger.info("Defining plot_operational_efficiency")
    try:
        minutes = [i * 2 for i in range(len(efficiency_df))]
        metrics = selected_metrics or ['uptime', 'throughput', 'quality', 'oee']
        colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['accent'], COLOR_SCHEME['neutral']]
        
        fig = go.Figure()
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Scatter(
                x=minutes,
                y=efficiency_df[metric],
                mode='lines+markers',
                name=f'{metric.capitalize()}',
                line=dict(color=color, width=3),
                marker=dict(size=6),
                hovertemplate=f'{metric.capitalize()}: %{{y:.1f}}%<br>Time: %{{x}} min'
            ))
            rolling_avg = efficiency_df[metric].rolling(window=10, min_periods=1).mean()
            fig.add_trace(go.Bar(
                x=minutes,
                y=rolling_avg,
                name=f'{metric.capitalize()} (Avg)',
                marker_color=color,
                opacity=0.3,
                hovertemplate=f'{metric.capitalize()} Avg: %{{y:.1f}}%<br>Time: %{{x}} min'
            ))
        
        stats = efficiency_df[metrics].mean().to_dict()
        annotation_text = "<br>".join([f"{m.capitalize()}: {v:.1f}%" for m, v in stats.items()])
        fig.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"Mean Metrics:<br>{annotation_text}",
            showarrow=False, font=dict(color=COLOR_SCHEME['text'], size=12),
            bgcolor=COLOR_SCHEME['background'], bordercolor=COLOR_SCHEME['text']
        )
        
        fig.update_layout(
            title=dict(text='Operational Efficiency Metrics', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Efficiency (%)',
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
                    x=df_zone['x'], y=df_zone['y'], z=df_zone['step'] * 2,
                    mode='markers',
                    name=zone,
                    marker=dict(
                        size=5,
                        color=COLOR_SCHEME['primary'] if zone == 'Assembly Line' else COLOR_SCHEME['secondary'] if zone == 'Packaging Zone' else COLOR_SCHEME['accent'],
                        opacity=0.7
                    ),
                    hovertemplate='ID: %{text}<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Time: %{z} min',
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
                    xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Time (min)',
                    xaxis=dict(range=[0, workplace_size]),
                    yaxis=dict(range=[0, workplace_size]),
                    zaxis=dict(range=[0, max(team_positions_df['step']) * 2]),
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
                color_discrete_sequence=[COLOR_SCHEME['primary'], COLOR_SCHEME['secondary'], COLOR_SCHEME['accent']]
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
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
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
            hovertemplate='X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Density: %{z}'
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
    Plot worker well-being index with triggers and recommendations.
    """
    logger.info("Defining plot_worker_wellbeing")
    try:
        minutes = [i * 2 for i in range(len(wellbeing_scores))]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=minutes,
            y=wellbeing_scores,
            mode='lines',
            name='Well-Being Index',
            line=dict(color=COLOR_SCHEME['primary'], width=3),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%'
        ))
        
        x = np.arange(len(wellbeing_scores))
        trend = np.polyval(np.polyfit(x, wellbeing_scores, 1), x)
        fig.add_trace(go.Scatter(
            x=minutes, y=trend,
            mode='lines',
            name='Trend',
            line=dict(color=COLOR_SCHEME['warning'], width=2, dash='dash'),
            hovertemplate='Time: %{x} min<br>Trend: %{y:.1f}%'
        ))
        
        for t in triggers['threshold']:
            fig.add_trace(go.Scatter(
                x=[minutes[t]], y=[wellbeing_scores[t]],
                mode='markers',
                name='Low Well-Being',
                marker=dict(color=COLOR_SCHEME['danger'], size=10, symbol='x'),
                hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%',
                showlegend=False
            ))
            fig.add_annotation(
                x=minutes[t], y=wellbeing_scores[t],
                text="Recommend breaks",
                showarrow=True, arrowhead=1, ax=20, ay=-30,
                font=dict(color=COLOR_SCHEME['danger'])
            )
        
        fig.add_hline(
            y=DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100,
            line_dash="dash",
            line_color=COLOR_SCHEME['danger'],
            annotation_text="Threshold",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title=dict(text='Worker Well-Being Index', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Well-Being Index (%)',
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
    Plot psychological safety score with anomalies and recommendations.
    """
    logger.info("Defining plot_psychological_safety")
    try:
        minutes = [i * 2 for i in range(len(safety_scores))]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=minutes,
            y=safety_scores,
            mode='lines',
            name='Psychological Safety',
            line=dict(color=COLOR_SCHEME['accent'], width=3),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%'
        ))
        
        x = np.arange(len(safety_scores))
        trend = np.polyval(np.polyfit(x, safety_scores, 1), x)
        fig.add_trace(go.Scatter(
            x=minutes, y=trend,
            mode='lines',
            name='Trend',
            line=dict(color=COLOR_SCHEME['warning'], width=2, dash='dash'),
            hovertemplate='Time: %{x} min<br>Trend: %{y:.1f}%'
        ))
        
        z_scores = (safety_scores - np.mean(safety_scores)) / np.std(safety_scores)
        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > DEFAULT_CONFIG['ANOMALY_THRESHOLD']]
        fig.add_trace(go.Scatter(
            x=[minutes[i] for i in anomalies],
            y=[safety_scores[i] for i in anomalies],
            mode='markers',
            name='Anomalies',
            marker=dict(color=COLOR_SCHEME['danger'], size=10, symbol='x'),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<br>Z-Score: %{text:.2f}',
            text=[z_scores[i] for i in anomalies]
        ))
        for i in anomalies:
            fig.add_annotation(
                x=minutes[i], y=safety_scores[i],
                text="Enhance safety training",
                showarrow=True, arrowhead=1, ax=20, ay=-30,
                font=dict(color=COLOR_SCHEME['danger'])
            )
        
        fig.add_hline(
            y=DEFAULT_CONFIG['SAFETY_THRESHOLD'] * 100,
            line_dash="dash",
            line_color=COLOR_SCHEME['danger'],
            annotation_text="Threshold",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title=dict(text='Psychological Safety Score', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Safety Score (%)',
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

def plot_downtime_trend(downtime_minutes: list, threshold: float) -> go.Figure:
    """
    Plot downtime trend with threshold alerts.
    """
    logger.info("Defining plot_downtime_trend")
    try:
        minutes = [i * 2 for i in range(len(downtime_minutes))]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=minutes,
            y=downtime_minutes,
            mode='lines',
            name='Downtime',
            line=dict(color=COLOR_SCHEME['danger'], width=3),
            hovertemplate='Time: %{x} min<br>Downtime: %{y:.1f} min'
        ))
        
        alerts = [i for i, d in enumerate(downtime_minutes) if d > threshold]
        fig.add_trace(go.Scatter(
            x=[minutes[i] for i in alerts],
            y=[downtime_minutes[i] for i in alerts],
            mode='markers',
            name='High Downtime',
            marker=dict(color=COLOR_SCHEME['danger'], size=10, symbol='x'),
            hovertemplate='Time: %{x} min<br>Downtime: %{y:.1f} min'
        ))
        for i in alerts:
            fig.add_annotation(
                x=minutes[i], y=downtime_minutes[i],
                text="Investigate downtime",
                showarrow=True, arrowhead=1, ax=20, ay=-30,
                font=dict(color=COLOR_SCHEME['danger'])
            )
        
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=COLOR_SCHEME['danger'],
            annotation_text="Threshold",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title=dict(text='Downtime Trend', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Downtime (minutes)',
            font=dict(color=COLOR_SCHEME['text'], size=14),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background']
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_downtime_trend: {str(e)}")
        raise
