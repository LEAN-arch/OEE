"""
visualizations.py
Advanced visualization functions for the Industrial Workplace Shift Monitoring Dashboard.
Generates professional, interactive Plotly charts with actionable insights and non-overlapping labels.
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

def plot_psychological_safety(safety_scores):
    """
    Create a line plot for psychological safety scores with annotations for low scores.
    
    Args:
        safety_scores (list or array): Psychological safety scores over time (0–100%).
    
    Returns:
        go.Figure: Plotly figure object.
    """
    try:
        # Time indices (assuming 2-minute intervals)
        minutes = [i * 2 for i in range(len(safety_scores))]
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=safety_scores,
            mode='lines+markers',
            name='Psychological Safety',
            line=dict(color='#EC4899', width=3),
            hovertemplate='Time: %{x} min<br>Safety: %{y:.1f}%'
        ))
        
        # Add threshold line
        threshold = DEFAULT_CONFIG['SAFETY_THRESHOLD'] * 100
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Threshold ({threshold}%)",
            annotation_position="bottom right"
        )
        
        # Add annotations for low safety scores
        annotations = []
        y_offset = 5  # Initial vertical offset
        max_annotations = 5  # Limit to avoid clutter
        
        for i, (time, score) in enumerate(zip(minutes, safety_scores)):
            if score < threshold:
                annotation = dict(
                    x=time,
                    y=score + y_offset,
                    xref="x",
                    yref="y",
                    text=f"Low safety: {score:.1f}%",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30,
                    font=dict(color="#EF4444", size=10),
                    align="left"
                )
                annotations.append(annotation)
                y_offset += 5  # Increment offset for the next annotation
                if y_offset > 20:  # Reset offset to avoid pushing off-screen
                    y_offset = 5
        
        # Update layout with limited annotations
        fig.update_layout(
            annotations=annotations[:max_annotations],
            title=dict(text='Psychological Safety Score', x=0.5, font_size=22),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            plot_bgcolor='#1A252F',
            paper_bgcolor='#1A252F',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in plot_psychological_safety: {str(e)}")
        raise

def plot_task_compliance_score(scores, disruptions, forecast=None, z_scores=None):
    """Plot Task Compliance Score with anomaly detection."""
    try:
        minutes = [i * 2 for i in range(len(scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes, y=scores, mode='lines+markers', name='Compliance Score',
            line=dict(color='#3B82F6', width=3),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%'
        ))
        
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=minutes, y=forecast, mode='lines', name='Forecast',
                line=dict(color='#F59E0B', dash='dash'),
                hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%'
            ))
        
        if z_scores is not None:
            anomalies = [i for i, z in enumerate(z_scores) if abs(z) > 2.0]
            if anomalies:
                fig.add_trace(go.Scatter(
                    x=[minutes[i] for i in anomalies], y=[scores[i] for i in anomalies],
                    mode='markers', name='Anomalies',
                    marker=dict(color='#EF4444', size=10, symbol='x'),
                    hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<br>Anomaly'
                ))
        
        for t in disruptions:
            fig.add_vline(x=t * 2, line=dict(color='#EF4444', dash='dash'))
        
        # Prevent label overlap
        fig.update_layout(
            title=dict(text='Task Compliance Score', x=0.5, font_size=22),
            xaxis_title='Time (minutes)', yaxis_title='Score (%)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='#1A252F', paper_bgcolor='#1A252F',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_task_compliance_score: {str(e)}")
        raise

def plot_collaboration_proximity_index(scores, disruptions, forecast=None):
    """Plot Collaboration Proximity Index."""
    try:
        minutes = [i * 2 for i in range(len(scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes, y=scores, mode='lines+markers', name='Proximity Index',
            line=dict(color='#10B981', width=3),
            hovertemplate='Time: %{x} min<br>Index: %{y:.1f}%'
        ))
        
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=minutes, y=forecast, mode='lines', name='Forecast',
                line=dict(color='#F59E0B', dash='dash'),
                hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%'
            ))
        
        for t in disruptions:
            fig.add_vline(x=t * 2, line=dict(color='#EF4444', dash='dash'))
        
        fig.update_layout(
            title=dict(text='Collaboration Proximity Index', x=0.5, font_size=22),
            xaxis_title='Time (minutes)', yaxis_title='Index (%)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='#1A252F', paper_bgcolor='#1A252F',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_collaboration_proximity_index: {str(e)}")
        raise

def plot_operational_recovery(recovery_scores, productivity_loss):
    """Plot Operational Recovery vs Productivity Loss."""
    try:
        minutes = [i * 2 for i in range(len(recovery_scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes, y=recovery_scores, mode='lines', name='Recovery Score',
            line=dict(color='#3B82F6', width=3),
            hovertemplate='Time: %{x} min<br>Recovery: %{y:.1f}%'
        ))
        fig.add_trace(go.Scatter(
            x=minutes, y=productivity_loss, mode='lines', name='Productivity Loss',
            line=dict(color='#EF4444', width=3),
            hovertemplate='Time: %{x} min<br>Loss: %{y:.1f}%'
        ))
        
        fig.update_layout(
            title=dict(text='Operational Resilience', x=0.5, font_size=22),
            xaxis_title='Time (minutes)', yaxis_title='Score (%)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='#1A252F', paper_bgcolor='#1A252F',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_operational_recovery: {str(e)}")
        raise

def plot_operational_efficiency(df, metrics):
    """Plot Operational Efficiency Metrics."""
    try:
        minutes = [i * 2 for i in range(len(df))]
        fig = go.Figure()
        colors = {'uptime': '#3B82F6', 'throughput': '#10B981', 'quality': '#EC4899', 'oee': '#F59E0B'}
        
        for metric in metrics:
            rolling_avg = df[metric].rolling(window=5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=minutes, y=df[metric], mode='lines', name=metric.capitalize(),
                line=dict(color=colors[metric], width=3),
                hovertemplate=f'Time: %{{x}} min<br>{metric.capitalize()}: %{{y:.1f}}%'
            ))
            fig.add_trace(go.Scatter(
                x=minutes, y=rolling_avg, mode='lines', name=f'{metric.capitalize()} (Rolling Avg)',
                line=dict(color=colors[metric], width=2, dash='dash'),
                hovertemplate=f'Time: %{{x}} min<br>{metric.capitalize()} (Avg): %{{y:.1f}}%'
            ))
        
        fig.update_layout(
            title=dict(text='Operational Efficiency Metrics', x=0.5, font_size=22),
            xaxis_title='Time (minutes)', yaxis_title='Score (%)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='#1A252F', paper_bgcolor='#1A252F',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_operational_efficiency: {str(e)}")
        raise

def plot_worker_distribution(df, facility_size, config, use_3d=False):
    """Plot Worker Distribution (2D/3D)."""
    try:
        if use_3d:
            fig = px.scatter_3d(
                df, x='x', y='y', z='step', color='zone',
                color_discrete_map=config['ZONE_COLORS'],
                labels={'x': 'X (m)', 'y': 'Y (m)', 'step': 'Time Step'},
                hover_data={'worker_id': True, 'step': True}
            )
            fig.update_traces(marker=dict(size=5), hovertemplate='Worker: %{customdata[0]}<br>X: %{x} m<br>Y: %{y} m<br>Step: %{customdata[1]}')
        else:
            fig = px.scatter(
                df, x='x', y='y', color='zone',
                color_discrete_map=config['ZONE_COLORS'],
                labels={'x': 'X (m)', 'y': 'Y (m)'},
                hover_data={'worker_id': True, 'step': True}
            )
            fig.update_traces(marker=dict(size=8), hovertemplate='Worker: %{customdata[0]}<br>X: %{x} m<br>Y: %{y} m<br>Step: %{customdata[1]}')
        
        fig.update_layout(
            title=dict(text='Worker Distribution', x=0.5, font_size=22),
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            plot_bgcolor='#1A252F', paper_bgcolor='#1A252F',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Time Step') if use_3d else {}
        )
        fig.update_xaxes(range=[0, facility_size], tickangle=45)
        fig.update_yaxes(range=[0, facility_size])
        return fig
    except Exception as e:
        logger.error(f"Error in plot_worker_distribution: {str(e)}")
        raise

def plot_worker_density_heatmap(df, facility_size, config):
    """Plot Worker Density Heatmap."""
    try:
        heatmap, xedges, yedges = np.histogram2d(
            df['x'], df['y'], bins=20, range=[[0, facility_size], [0, facility_size]]
        )
        fig = go.Figure(data=go.Heatmap(
            z=heatmap.T, x=xedges, y=yedges,
            colorscale='Viridis', showscale=True,
            hovertemplate='X: %{x} m<br>Y: %{y} m<br>Count: %{z}'
        ))
        
        fig.update_layout(
            title=dict(text='Worker Density Heatmap', x=0.5, font_size=22),
            xaxis_title='X (m)', yaxis_title='Y (m)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            plot_bgcolor='#1A252F', paper_bgcolor='#1A252F',
            xaxis=dict(tickangle=45, nticks=10)
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_worker_density_heatmap: {str(e)}")
        raise

def plot_worker_wellbeing(scores, triggers):
    """Plot Worker Well-Being Index with alerts."""
    try:
        minutes = [i * 2 for i in range(len(scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes, y=scores, mode='lines+markers', name='Well-Being Index',
            line=dict(color='#10B981', width=3),
            hovertemplate='Time: %{x} min<br>Index: %{y:.1f}%'
        ))
        
        annotations = []
        y_offset = 10
        for t in triggers['threshold']:
            annotations.append(dict(
                x=minutes[t], y=scores[t] + y_offset, text='Low Score Alert',
                showarrow=True, arrowhead=1, ax=20, ay=-30, font=dict(color='#EF4444')
            ))
            y_offset += 5
        for t in triggers['trend']:
            annotations.append(dict(
                x=minutes[t], y=scores[t] + y_offset, text='Declining Trend',
                showarrow=True, arrowhead=1, ax=20, ay=-30, font=dict(color='#EF4444')
            ))
            y_offset += 5
        for zone, times in triggers['work_area'].items():
            for t in times:
                annotations.append(dict(
                    x=minutes[t], y=scores[t] + y_offset, text=f'{zone} Alert',
                    showarrow=True, arrowhead=1, ax=20, ay=-30, font=dict(color='#EF4444')
                ))
                y_offset += 5
        for t in triggers['disruption']:
            annotations.append(dict(
                x=minutes[t], y=scores[t] + y_offset, text='Disruption Alert',
                showarrow=True, arrowhead=1, ax=20, ay=-30, font=dict(color='#EF4444')
            ))
            y_offset += 5
        
        fig.update_layout(
            title=dict(text='Worker Well-Being Index', x=0.5, font_size=22),
            xaxis_title='Time (minutes)', yaxis_title='Index (%)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='#1A252F', paper_bgcolor='#1A252F',
            annotations=annotations[:5],
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_worker_wellbeing: {str(e)}")
        raise

def plot_downtime_trend(downtime_minutes, threshold):
    """Plot Downtime Trend."""
    try:
        minutes = [i * 2 for i in range(len(downtime_minutes))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes, y=downtime_minutes, mode='lines+markers', name='Downtime',
            line=dict(color='#EF4444', width=3),
            hovertemplate='Time: %{x} min<br>Downtime: %{y:.1f} min'
        ))
        
        annotations = []
        y_offset = 5
        for i, dt in enumerate(downtime_minutes):
            if dt > threshold:
                annotations.append(dict(
                    x=minutes[i], y=dt + y_offset, text='High Downtime',
                    showarrow=True, arrowhead=1, ax=20, ay=-30, font=dict(color='#EF4444')
                ))
                y_offset += 3
        
        fig.update_layout(
            title=dict(text='Downtime Trend', x=0.5, font_size=22),
            xaxis_title='Time (minutes)', yaxis_title='Downtime (minutes)',
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            hovermode='x unified',
            plot_bgcolor='#1A252F', paper_bgcolor='#1A252F',
            annotations=annotations[:5],
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
        )
        return fig
    except Exception as e:
        logger.error(f"Error in plot_downtime_trend: {str(e)}")
        raise
