"""
visualizations.py
Functions to create Plotly visualizations for the Industrial Workplace Shift Monitoring Dashboard.
"""

import plotly.graph_objects as go
import numpy as np
from config import DEFAULT_CONFIG

def plot_task_compliance_score(compliance_scores, disruptions, forecast, z_scores):
    """
    Plot task compliance scores with disruptions, forecast, and anomaly detection.
    """
    minutes = [i * 2 for i in range(len(compliance_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=compliance_scores,
        mode='lines',
        name='Task Compliance',
        line=dict(color='#3B82F6', width=3),
        hovertemplate='Time: %{x} min<br>Compliance: %{y:.1f}%'
    ))
    
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#FBBF24', width=2, dash='dash'),
            hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%'
        ))
    
    # Add disruption markers
    for disruption in disruptions:
        if 0 <= disruption < len(minutes):
            fig.add_vline(
                x=minutes[disruption],
                line_dash="dot",
                line_color="#EF4444",
                annotation_text="Disruption",
                annotation_position="top"
            )
    
    # Add anomaly annotations
    annotations = []
    y_offset = 5
    for i, (score, z) in enumerate(zip(compliance_scores, z_scores)):
        if abs(z) > DEFAULT_CONFIG['ANOMALY_THRESHOLD']:
            annotations.append(dict(
                x=minutes[i],
                y=score + y_offset,
                text=f"Anomaly: {score:.1f}%",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#EF4444', size=10)
            ))
            y_offset += 5
            if y_offset > 20:
                y_offset = 5
    
    fig.update_layout(
        title=dict(text='Task Compliance Score', x=0.5, font_size=22),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        font=dict(color='#E6ECEF', size=14),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations[:5],
        xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
    )
    return fig

def plot_collaboration_proximity_index(proximity_scores, disruptions, forecast):
    """
    Plot collaboration proximity index with disruptions and forecast.
    """
    minutes = [i * 2 for i in range(len(proximity_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=proximity_scores,
        mode='lines',
        name='Proximity Index',
        line=dict(color='#10B981', width=3),
        hovertemplate='Time: %{x} min<br>Proximity: %{y:.1f}%'
    ))
    
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#FBBF24', width=2, dash='dash'),
            hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%'
        ))
    
    for disruption in disruptions:
        if 0 <= disruption < len(minutes):
            fig.add_vline(
                x=minutes[disruption],
                line_dash="dot",
                line_color="#EF4444",
                annotation_text="Disruption",
                annotation_position="top"
            )
    
    fig.update_layout(
        title=dict(text='Collaboration Proximity Index', x=0.5, font_size=22),
        xaxis_title='Time (minutes)',
        yaxis_title='Index (%)',
        font=dict(color='#E6ECEF', size=14),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
    )
    return fig

def plot_operational_recovery(recovery_scores, productivity_loss):
    """
    Plot operational recovery scores vs. productivity loss.
    """
    minutes = [i * 2 for i in range(len(recovery_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=recovery_scores,
        mode='lines',
        name='Operational Recovery',
        line=dict(color='#3B82F6', width=3),
        hovertemplate='Time: %{x} min<br>Recovery: %{y:.1f}%'
    ))
    fig.add_trace(go.Scatter(
        x=minutes,
        y=productivity_loss,
        mode='lines',
        name='Productivity Loss',
        line=dict(color='#EF4444', width=3),
        hovertemplate='Time: %{x} min<br>Loss: %{y:.1f}%'
    ))
    
    fig.update_layout(
        title=dict(text='Operational Recovery vs. Productivity Loss', x=0.5, font_size=22),
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

def plot_operational_efficiency(efficiency_df, selected_metrics):
    """
    Plot operational efficiency metrics (uptime, throughput, quality, OEE).
    """
    minutes = [i * 2 for i in range(len(efficiency_df))]
    fig = go.Figure()
    colors = {'uptime': '#3B82F6', 'throughput': '#10B981', 'quality': '#EC4899', 'oee': '#FBBF24'}
    
    for metric in selected_metrics:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=efficiency_df[metric],
            mode='lines',
            name=metric.capitalize(),
            line=dict(color=colors[metric], width=3),
            hovertemplate=f'Time: %{{x}} min<br>{metric.capitalize()}: %{{y:.1f}}%'
        ))
    
    fig.update_layout(
        title=dict(text='Operational Efficiency Metrics', x=0.5, font_size=22),
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

def plot_worker_distribution(positions_df, facility_size, config, use_3d=False):
    """
    Plot worker distribution as a 2D or 3D scatter plot.
    """
    if use_3d:
        fig = go.Figure()
        for zone, area in config['WORK_AREAS'].items():
            zone_df = positions_df[positions_df['zone'] == zone]
            fig.add_trace(go.Scatter3d(
                x=zone_df['x'],
                y=zone_df['y'],
                z=zone_df['step'],
                mode='markers',
                name=area['label'],
                marker=dict(size=5)
            ))
        fig.update_layout(
            title=dict(text='3D Worker Distribution', x=0.5, font_size=22),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Step',
                xaxis=dict(range=[0, facility_size]),
                yaxis=dict(range=[0, facility_size])
            ),
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            plot_bgcolor='#1A252F',
            paper_bgcolor='#1A252F'
        )
    else:
        fig = go.Figure()
        for zone, area in config['WORK_AREAS'].items():
            zone_df = positions_df[positions_df['zone'] == zone]
            fig.add_trace(go.Scatter(
                x=zone_df['x'],
                y=zone_df['y'],
                mode='markers',
                name=area['label'],
                marker=dict(size=8)
            ))
        fig.update_layout(
            title=dict(text='2D Worker Distribution', x=0.5, font_size=22),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            xaxis=dict(range=[0, facility_size]),
            yaxis=dict(range=[0, facility_size]),
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            plot_bgcolor='#1A252F',
            paper_bgcolor='#1A252F'
        )
    return fig

def plot_worker_density_heatmap(positions_df, facility_size, config):
    """
    Plot worker density as a heatmap.
    """
    x_bins = np.linspace(0, facility_size, config['DENSITY_GRID_SIZE'])
    y_bins = np.linspace(0, facility_size, config['DENSITY_GRID_SIZE'])
    heatmap, xedges, yedges = np.histogram2d(positions_df['x'], positions_df['y'], bins=[x_bins, y_bins])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap.T,
        x=xedges,
        y=yedges,
        colorscale='Viridis',
        showscale=True
    ))
    fig.update_layout(
        title=dict(text='Worker Density Heatmap', x=0.5, font_size=22),
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        font=dict(color='#E6ECEF', size=14),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F'
    )
    return fig

def plot_worker_wellbeing(wellbeing_scores, triggers):
    """
    Plot worker well-being scores with trigger annotations.
    """
    minutes = [i * 2 for i in range(len(wellbeing_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=wellbeing_scores,
        mode='lines',
        name='Well-Being',
        line=dict(color='#10B981', width=3),
        hovertemplate='Time: %{x} min<br>Well-Being: %{y:.1f}%'
    ))
    
    # Add threshold line
    threshold = DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Threshold ({threshold}%)",
        annotation_position="bottom right"
    )
    
    # Add trigger annotations
    annotations = []
    y_offset = 5
    for trigger_type, times in triggers.items():
        if trigger_type == 'work_area':
            for zone, zone_times in times.items():
                for t in zone_times:
                    if 0 <= t < len(minutes):
                        score = wellbeing_scores[t]
                        annotations.append(dict(
                            x=minutes[t],
                            y=score + y_offset,
                            text=f"{DEFAULT_CONFIG['WORK_AREAS'][zone]['label']} Alert",
                            showarrow=True,
                            arrowhead=1,
                            ax=20,
                            ay=-30,
                            font=dict(color='#EF4444', size=10)
                        ))
                        y_offset += 5
                        if y_offset > 20:
                            y_offset = 5
        else:
            for t in times:
                if 0 <= t < len(minutes):
                    score = wellbeing_scores[t]
                    annotations.append(dict(
                        x=minutes[t],
                        y=score + y_offset,
                        text=f"{trigger_type.capitalize()} Alert",
                        showarrow=True,
                        arrowhead=1,
                        ax=20,
                        ay=-30,
                        font=dict(color='#EF4444', size=10)
                    ))
                    y_offset += 5
                    if y_offset > 20:
                        y_offset = 5
    
    fig.update_layout(
        title=dict(text='Worker Well-Being Index', x=0.5, font_size=22),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        font=dict(color='#E6ECEF', size=14),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations[:5],
        xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
    )
    return fig

def plot_psychological_safety(safety_scores):
    """
    Plot psychological safety scores with annotations for low scores.
    """
    minutes = [i * 2 for i in range(len(safety_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=safety_scores,
        mode='lines',
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
    y_offset = 5
    max_annotations = 5
    
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
            y_offset += 5
            if y_offset > 20:
                y_offset = 5
    
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

def plot_downtime_trend(downtime_minutes, threshold):
    """
    Plot downtime trend with threshold alerts.
    """
    minutes = [i * 2 for i in range(len(downtime_minutes))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=downtime_minutes,
        mode='lines',
        name='Downtime',
        line=dict(color='#EF4444', width=3),
        hovertemplate='Time: %{x} min<br>Downtime: %{y:.1f} min'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#FBBF24",
        annotation_text=f"Threshold ({threshold} min)",
        annotation_position="bottom right"
    )
    
    # Add annotations for high downtime
    annotations = []
    y_offset = 5
    for i, (time, downtime) in enumerate(zip(minutes, downtime_minutes)):
        if downtime > threshold:
            annotations.append(dict(
                x=time,
                y=downtime + y_offset,
                text=f"High downtime: {downtime:.1f} min",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#FBBF24', size=10)
            ))
            y_offset += 5
            if y_offset > 20:
                y_offset = 5
    
    fig.update_layout(
        title=dict(text='Downtime Trend', x=0.5, font_size=22),
        xaxis_title='Time (minutes)',
        yaxis_title='Downtime (minutes)',
        font=dict(color='#E6ECEF', size=14),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations[:5],
        xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
    )
    return fig
