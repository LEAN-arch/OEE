"""
visualizations.py
Functions to create Plotly visualizations for the Workplace Shift Monitoring Dashboard.
"""

import plotly.graph_objects as go
import numpy as np
from config import DEFAULT_CONFIG

def plot_gauge_chart(value, title, threshold, max_value=100, recommendation=None):
    """
    Create a gauge chart for key metrics with color-coded ranges and recommendations.
    
    Args:
        value (float): Metric value (0-100).
        title (str): Chart title.
        threshold (float): Threshold for warning (0-100).
        max_value (float): Maximum value for the gauge.
        recommendation (str): Actionable recommendation if below threshold.
    
    Returns:
        go.Figure: Gauge chart.
    """
    # Define color ranges
    if value < threshold:
        color = "#EF4444"  # Red for below threshold
    elif value < threshold + 10:
        color = "#FBBF24"  # Yellow for near threshold
    else:
        color = "#10B981"  # Green for good
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#E6ECEF'}},
        number={'suffix': "%", 'font': {'size': 30, 'color': '#E6ECEF'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': "#E6ECEF"},
            'bar': {'color': color},
            'bgcolor': "#2D3748",
            'borderwidth': 2,
            'bordercolor': "#E6ECEF",
            'steps': [
                {'range': [0, threshold], 'color': "#EF4444"},
                {'range': [threshold, threshold + 10], 'color': "#FBBF24"},
                {'range': [threshold + 10, max_value], 'color': "#10B981"}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        font=dict(color='#E6ECEF', size=14),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig, recommendation

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
    
    # Add anomaly annotations with recommendations
    annotations = []
    y_offset = 5
    for i, (score, z) in enumerate(zip(compliance_scores, z_scores)):
        if abs(z) > DEFAULT_CONFIG['ANOMALY_THRESHOLD']:
            annotations.append(dict(
                x=minutes[i],
                y=score + y_offset,
                text=f"Anomaly: {score:.1f}%<br>Review task protocols",
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
    
    # Add recommendation if proximity is low
    annotations = []
    if np.mean(proximity_scores) < 60:
        annotations.append(dict(
            x=minutes[0],
            y=max(proximity_scores) + 5,
            text="Low collaboration<br>Encourage team activities",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(color='#EF4444', size=10)
        ))
    
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
        annotations=annotations,
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
    
    # Add recommendation if loss is high
    annotations = []
    max_loss_idx = np.argmax(productivity_loss)
    if productivity_loss[max_loss_idx] > 10:
        annotations.append(dict(
            x=minutes[max_loss_idx],
            y=productivity_loss[max_loss_idx] + 5,
            text=f"High loss: {productivity_loss[max_loss_idx]:.1f}%<br>Investigate cause",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(color='#EF4444', size=10)
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
        annotations=annotations,
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
    
    # Add recommendation if OEE is low
    annotations = []
    if 'oee' in selected_metrics and np.mean(efficiency_df['oee']) < 75:
        annotations.append(dict(
            x=minutes[0],
            y=max(efficiency_df['oee']) + 5,
            text="Low OEE<br>Optimize processes",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(color='#EF4444', size=10)
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
        annotations=annotations,
        xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
    )
    return fig

def plot_worker_distribution(positions_df, facility_size, config, use_3d=False, selected_step=None):
    """
    Plot worker distribution with a facility layout, color-coded zones, and workload status.
    
    Args:
        positions_df (pd.DataFrame): Worker positions with workload status.
        facility_size (int): Size of the facility in meters.
        config (dict): Configuration settings.
        use_3d (bool): Whether to use a 3D plot.
        selected_step (int): Specific step to display (for 3D time slider).
    
    Returns:
        go.Figure: Worker distribution plot.
    """
    if use_3d:
        fig = go.Figure()
        if selected_step is not None:
            positions_df = positions_df[positions_df['step'] == selected_step]
        
        for zone, area in config['WORK_AREAS'].items():
            zone_df = positions_df[positions_df['zone'] == zone]
            for status in ['Normal', 'High', 'Critical']:
                status_df = zone_df[zone_df['workload_status'] == status]
                color = '#10B981' if status == 'Normal' else '#FBBF24' if status == 'High' else '#EF4444'
                fig.add_trace(go.Scatter3d(
                    x=status_df['x'],
                    y=status_df['y'],
                    z=status_df['step'],
                    mode='markers',
                    name=f"{area['label']} ({status})",
                    marker=dict(size=5, color=color),
                    text=[f"Worker: {w}<br>Zone: {z}<br>Workload: {wl:.2f} ({s})"
                          for w, z, wl, s in zip(status_df['worker'], status_df['zone'], status_df['workload'], status_df['workload_status'])],
                    hoverinfo='text'
                ))
        fig.update_layout(
            title=dict(text='3D Worker Distribution Over Time', x=0.5, font_size=22),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Step',
                xaxis=dict(range=[0, facility_size], backgroundcolor="#2D3748"),
                yaxis=dict(range=[0, facility_size], backgroundcolor="#2D3748"),
                zaxis=dict(backgroundcolor="#2D3748")
            ),
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            plot_bgcolor='#1A252F',
            paper_bgcolor='#1A252F',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1)
        )
    else:
        fig = go.Figure()
        
        # Add facility layout with zones
        shapes = []
        colors = {'Assembly Line': 'rgba(59, 130, 246, 0.3)', 'Packaging Zone': 'rgba(16, 185, 129, 0.3)', 'Quality Control': 'rgba(236, 72, 153, 0.3)'}
        for zone, area in config['WORK_AREAS'].items():
            center_x, center_y = area['center']
            shapes.append(dict(
                type="rect",
                x0=center_x - 15,
                y0=center_y - 15,
                x1=center_x + 15,
                y1=center_y + 15,
                fillcolor=colors[zone],
                line=dict(color=colors[zone].replace('0.3', '1.0')),
                opacity=0.3,
                layer='below'
            ))
        
        # Plot workers by workload status
        for zone, area in config['WORK_AREAS'].items():
            zone_df = positions_df[positions_df['zone'] == zone]
            if selected_step is not None:
                zone_df = zone_df[zone_df['step'] == selected_step]
            for status in ['Normal', 'High', 'Critical']:
                status_df = zone_df[zone_df['workload_status'] == status]
                color = '#10B981' if status == 'Normal' else '#FBBF24' if status == 'High' else '#EF4444'
                fig.add_trace(go.Scatter(
                    x=status_df['x'],
                    y=status_df['y'],
                    mode='markers',
                    name=f"{area['label']} ({status})",
                    marker=dict(size=8, color=color),
                    text=[f"Worker: {w}<br>Zone: {z}<br>Workload: {wl:.2f} ({s})"
                          for w, z, wl, s in zip(status_df['worker'], status_df['zone'], status_df['workload'], status_df['workload_status'])],
                    hoverinfo='text'
                ))
        
        # Detect overcrowding
        annotations = []
        for zone, area in config['WORK_AREAS'].items():
            zone_df = positions_df[positions_df['zone'] == zone]
            if selected_step is not None:
                zone_df = zone_df[zone_df['step'] == selected_step]
            if len(zone_df) > area['workers'] * 1.5:
                annotations.append(dict(
                    x=area['center'][0],
                    y=area['center'][1],
                    text=f"Overcrowded: {len(zone_df)} workers<br>Reassign tasks",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30,
                    font=dict(color='#EF4444', size=10)
                ))
        
        fig.update_layout(
            title=dict(text='Worker Distribution with Facility Layout', x=0.5, font_size=22),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            xaxis=dict(range=[0, facility_size], gridcolor="#444", zerolinecolor="#444"),
            yaxis=dict(range=[0, facility_size], gridcolor="#444", zerolinecolor="#444"),
            font=dict(color='#E6ECEF', size=14),
            template='plotly_dark',
            plot_bgcolor='#1A252F',
            paper_bgcolor='#1A252F',
            shapes=shapes,
            annotations=annotations,
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1)
        )
    return fig

def plot_worker_density_heatmap(positions_df, facility_size, config):
    """
    Plot worker density as a heatmap with facility layout.
    """
    x_bins = np.linspace(0, facility_size, config['DENSITY_GRID_SIZE'])
    y_bins = np.linspace(0, facility_size, config['DENSITY_GRID_SIZE'])
    heatmap, xedges, yedges = np.histogram2d(positions_df['x'], positions_df['y'], bins=[x_bins, y_bins])
    
    # Add facility layout
    shapes = []
    colors = {'Assembly Line': 'rgba(59, 130, 246, 0.3)', 'Packaging Zone': 'rgba(16, 185, 129, 0.3)', 'Quality Control': 'rgba(236, 72, 153, 0.3)'}
    for zone, area in config['WORK_AREAS'].items():
        center_x, center_y = area['center']
        shapes.append(dict(
            type="rect",
            x0=center_x - 15,
            y0=center_y - 15,
            x1=center_x + 15,
            y1=center_y + 15,
            fillcolor=colors[zone],
            line=dict(color=colors[zone].replace('0.3', '1.0')),
            opacity=0.3,
            layer='below'
        ))
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap.T,
        x=xedges,
        y=yedges,
        colorscale='Viridis',
        showscale=True,
        hovertemplate='X: %{x} m<br>Y: %{y} m<br>Workers: %{z}'
    ))
    
    # Add overcrowding annotations
    annotations = []
    for zone, area in config['WORK_AREAS'].items():
        zone_df = positions_df[positions_df['zone'] == zone]
        if len(zone_df) > area['workers'] * 1.5:
            annotations.append(dict(
                x=area['center'][0],
                y=area['center'][1],
                text=f"High density<br>Reassign workers",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#EF4444', size=10)
            ))
    
    fig.update_layout(
        title=dict(text='Worker Density Heatmap', x=0.5, font_size=22),
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        font=dict(color='#E6ECEF', size=14),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        shapes=shapes,
        annotations=annotations
    )
    return fig

def plot_worker_wellbeing(wellbeing_scores, triggers):
    """
    Plot worker well-being scores with trigger annotations and recommendations.
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
    
    # Add trigger annotations with recommendations
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
                            text=f"{DEFAULT_CONFIG['WORK_AREAS'][zone]['label']} Alert<br>Check workload",
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
                    text = f"{trigger_type.capitalize()} Alert"
                    if trigger_type == 'threshold':
                        text += "<br>Schedule break"
                    elif trigger_type == 'trend':
                        text += "<br>Monitor closely"
                    annotations.append(dict(
                        x=minutes[t],
                        y=score + y_offset,
                        text=text,
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
    
    # Add annotations for low safety scores with recommendations
    annotations = []
    y_offset = 5
    max_annotations = 5
    for i, (time, score) in enumerate(zip(minutes, safety_scores)):
        if score < threshold:
            annotations.append(dict(
                x=time,
                y=score + y_offset,
                text=f"Low safety: {score:.1f}%<br>Enhance training",
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
        title=dict(text='Psychological Safety Score', x=0.5, font_size=22),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        font=dict(color='#E6ECEF', size=14),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations[:max_annotations],
        xaxis=dict(tickangle=45, nticks=len(minutes)//10 + 1)
    )
    return fig

def plot_downtime_trend(downtime_minutes, threshold):
    """
    Plot downtime trend with threshold alerts and recommendations.
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
    
    # Add annotations for high downtime with recommendations
    annotations = []
    y_offset = 5
    for i, (time, downtime) in enumerate(zip(minutes, downtime_minutes)):
        if downtime > threshold:
            annotations.append(dict(
                x=time,
                y=downtime + y_offset,
                text=f"High downtime: {downtime:.1f} min<br>Investigate cause",
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
