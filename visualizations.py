"""
visualizations.py
Functions to create enhanced Plotly visualizations for the Workplace Shift Monitoring Dashboard.
Optimized for user experience, graphics, and actionable metrics.
"""

import plotly.graph_objects as go
import numpy as np
from plotly.colors import sequential
from config import DEFAULT_CONFIG

def plot_gauge_chart(value, title, threshold, max_value=100, recommendation=None):
    """
    Create an enhanced gauge chart with color gradients and interactive tooltips.
    
    Args:
        value (float): Metric value (0-100).
        title (str): Chart title.
        threshold (float): Threshold for warning (0-100).
        max_value (float): Maximum value for the gauge.
        recommendation (str): Actionable recommendation if below threshold.
    
    Returns:
        go.Figure: Enhanced gauge chart.
    """
    # Validate inputs
    max_value = max(max_value, 1)  # Prevent division by zero
    value = max(0, min(value, max_value))  # Clamp value between 0 and max_value

    # Define color gradient
    colors = sequential.Viridis
    color_idx = int((value / max_value) * (len(colors) - 1))
    color_idx = max(0, min(color_idx, len(colors) - 1))  # Ensure index is within bounds
    bar_color = colors[color_idx]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': threshold, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#E6ECEF'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#E6ECEF'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 2, 'tickcolor': "#E6ECEF"},
            'bar': {'color': bar_color},
            'bgcolor': "#2D3748",
            'borderwidth': 2,
            'bordercolor': "#E6ECEF",
            'steps': [
                {'range': [0, threshold], 'color': "#EF4444"},
                {'range': [threshold, max_value], 'color': "#10B981"}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 4},
                'thickness': 0.8,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        margin=dict(l=30, r=30, t=60, b=30),
        height=300,
        annotations=[
            dict(
                text=recommendation or "N/A",
                x=0.5,
                y=-0.2,
                showarrow=False,
                font=dict(size=14, color='#FBBF24' if value < threshold else '#10B981')
            ) if recommendation else None
        ]
    )
    return fig, recommendation

def plot_task_compliance_score(compliance_scores, disruptions, forecast, z_scores):
    """
    Plot task compliance scores with interactive disruptions and anomaly detection.
    """
    minutes = [i * 2 for i in range(len(compliance_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=compliance_scores,
        mode='lines+markers',
        name='Task Compliance',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8),
        hovertemplate='Time: %{x} min<br>Compliance: %{y:.1f}%<extra></extra>'
    ))
    
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#FBBF24', width=2, dash='dash'),
            hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%<extra></extra>'
        ))
    
    for disruption in disruptions:
        if 0 <= disruption < len(minutes):
            fig.add_vline(
                x=minutes[disruption],
                line_dash="dot",
                line_color="#EF4444",
                annotation_text="Disruption",
                annotation_position="top",
                annotation_font_size=12
            )
    
    annotations = []
    y_offset = 5
    for i, (score, z) in enumerate(zip(compliance_scores, z_scores)):
        if abs(z) > DEFAULT_CONFIG['ANOMALY_THRESHOLD']:
            annotations.append(dict(
                x=minutes[i],
                y=score + y_offset,
                text=f"Anomaly: {score:.1f}%<br>Review protocols",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#EF4444', size=12)
            ))
            y_offset += 5 if y_offset < 20 else -15
    
    fig.update_layout(
        title=dict(text='Task Compliance Score', x=0.5, font_size=24),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#444"),
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations[:5],
        transition_duration=500
    )
    return fig

def plot_collaboration_proximity_index(proximity_scores, disruptions, forecast):
    """
    Plot collaboration proximity index with interactive elements.
    """
    minutes = [i * 2 for i in range(len(proximity_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=proximity_scores,
        mode='lines+markers',
        name='Proximity Index',
        line=dict(color='#10B981', width=3),
        marker=dict(size=8),
        hovertemplate='Time: %{x} min<br>Proximity: %{y:.1f}%<extra></extra>'
    ))
    
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#FBBF24', width=2, dash='dash'),
            hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%<extra></extra>'
        ))
    
    for disruption in disruptions:
        if 0 <= disruption < len(minutes):
            fig.add_vline(
                x=minutes[disruption],
                line_dash="dot",
                line_color="#EF4444",
                annotation_text="Disruption",
                annotation_position="top",
                annotation_font_size=12
            )
    
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
            font=dict(color='#EF4444', size=12)
        ))
    
    fig.update_layout(
        title=dict(text='Collaboration Proximity Index', x=0.5, font_size=24),
        xaxis_title='Time (minutes)',
        yaxis_title='Index (%)',
        yaxis=dict(range=[0, 100], gridcolor="#444"),
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations,
        transition_duration=500
    )
    return fig

def plot_operational_recovery(recovery_scores, productivity_loss):
    """
    Plot operational recovery vs. productivity loss with enhanced visuals.
    """
    minutes = [i * 2 for i in range(len(recovery_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=recovery_scores,
        mode='lines+markers',
        name='Operational Recovery',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=8),
        hovertemplate='Time: %{x} min<br>Recovery: %{y:.1f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=minutes,
        y=productivity_loss,
        mode='lines+markers',
        name='Productivity Loss',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=8),
        hovertemplate='Time: %{x} min<br>Loss: %{y:.1f}%<extra></extra>'
    ))
    
    annotations = []
    max_loss_idx = np.argmax(productivity_loss)
    if productivity_loss[max_loss_idx] > 10:
        annotations.append(dict(
            x=minutes[max_loss_idx],
            y=productivity_loss[max_loss_idx] + 5,
            text=f"High loss: {productivity_loss[max_loss_idx]:.1f}%<br>Investigate",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(color='#EF4444', size=12)
        ))
    
    fig.update_layout(
        title=dict(text='Operational Recovery vs. Productivity Loss', x=0.5, font_size=24),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#444"),
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations,
        transition_duration=500
    )
    return fig

def plot_operational_efficiency(efficiency_df, selected_metrics):
    """
    Plot operational efficiency metrics with interactive selection.
    """
    minutes = [i * 2 for i in range(len(efficiency_df))]
    fig = go.Figure()
    colors = {'uptime': '#3B82F6', 'throughput': '#10B981', 'quality': '#EC4899', 'oee': '#FBBF24'}
    
    for metric in selected_metrics:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=efficiency_df[metric],
            mode='lines+markers',
            name=metric.capitalize(),
            line=dict(color=colors[metric], width=3),
            marker=dict(size=8),
            hovertemplate=f'Time: %{{x}} min<br>{metric.capitalize()}: %{{y:.1f}}%<extra></extra>'
        ))
    
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
            font=dict(color='#EF4444', size=12)
        ))
    
    fig.update_layout(
        title=dict(text='Operational Efficiency Metrics', x=0.5, font_size=24),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#444"),
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations,
        transition_duration=500
    )
    return fig

def plot_worker_distribution(positions_df, facility_size, config, use_3d=False, selected_step=None, show_entry_exit=True, show_production_lines=True):
    """
    Plot worker distribution with enhanced 3D/2D visuals and interactivity.
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
                    marker=dict(size=6, color=color, opacity=0.8),
                    text=[f"Worker: {w}<br>Zone: {z}<br>Workload: {wl:.2f} ({s})"
                          for w, z, wl, s in zip(status_df['worker'], status_df['zone'], status_df['workload'], status_df['workload_status'])],
                    hoverinfo='text'
                ))
        
        if show_production_lines:
            for line in config['PRODUCTION_LINES']:
                fig.add_trace(go.Scatter3d(
                    x=[line['start'][0], line['end'][0]],
                    y=[line['start'][1], line['end'][1]],
                    z=[selected_step, selected_step],
                    mode='lines',
                    name=line['label'],
                    line=dict(color='#FFFFFF', width=4, dash='dash'),
                    hovertemplate=f"{line['label']}<br>Start: ({line['start'][0]}, {line['start'][1]})<br>End: ({line['end'][0]}, {line['end'][1]})"
                ))
                mid_x = (line['start'][0] + line['end'][0]) / 2
                mid_y = (line['start'][1] + line['end'][1]) / 2
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    z=selected_step,
                    text=line['label'],
                    showarrow=False,
                    font=dict(color='#FFFFFF', size=12),
                    yshift=15
                )
        
        if show_entry_exit:
            for point in config['ENTRY_EXIT_POINTS']:
                marker_color = '#00FF00' if point['type'] == 'Entry' else '#FF0000'
                fig.add_trace(go.Scatter3d(
                    x=[point['coords'][0]],
                    y=[point['coords'][1]],
                    z=[selected_step],
                    mode='markers+text',
                    name=point['label'],
                    marker=dict(size=10, color=marker_color, symbol='diamond'),
                    text=[point['label']],
                    textposition='top center',
                    hovertemplate=f"{point['label']} ({point['type']})<br>Coords: ({point['coords'][0]}, {point['coords'][1]})"
                ))
        
        fig.update_layout(
            title=dict(text='3D Worker Distribution Over Time', x=0.5, font_size=24),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Step',
                xaxis=dict(range=[0, facility_size], backgroundcolor="#2D3748", gridcolor="#444"),
                yaxis=dict(range=[0, facility_size], backgroundcolor="#2D3748", gridcolor="#444"),
                zaxis=dict(backgroundcolor="#2D3748", gridcolor="#444")
            ),
            font=dict(color='#E6ECEF', size=16),
            template='plotly_dark',
            plot_bgcolor='#1A252F',
            paper_bgcolor='#1A252F',
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            transition_duration=500
        )
    else:
        fig = go.Figure()
        
        shapes = []
        colors = {'Assembly Line': 'rgba(59, 130, 246, 0.4)', 'Packaging Zone': 'rgba(16, 185, 129, 0.4)', 'Quality Control': 'rgba(236, 72, 153, 0.4)'}
        for zone, area in config['WORK_AREAS'].items():
            center_x, center_y = area['center']
            shapes.append(dict(
                type="rect",
                x0=center_x - 15,
                y0=center_y - 15,
                x1=center_x + 15,
                y1=center_y + 15,
                fillcolor=colors[zone],
                line=dict(color=colors[zone].replace('0.4', '1.0'), width=2),
                opacity=0.4,
                layer='below'
            ))
        
        if show_production_lines:
            for line in config['PRODUCTION_LINES']:
                shapes.append(dict(
                    type="line",
                    x0=line['start'][0],
                    y0=line['start'][1],
                    x1=line['end'][0],
                    y1=line['end'][1],
                    line=dict(color='#FFFFFF', width=4, dash='dash'),
                    layer='below'
                ))
        
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
                    marker=dict(size=10, color=color, opacity=0.8),
                    text=[f"Worker: {w}<br>Zone: {z}<br>Workload: {wl:.2f} ({s})"
                          for w, z, wl, s in zip(status_df['worker'], status_df['zone'], status_df['workload'], status_df['workload_status'])],
                    hoverinfo='text'
                ))
        
        if show_entry_exit:
            for point in config['ENTRY_EXIT_POINTS']:
                marker_color = '#00FF00' if point['type'] == 'Entry' else '#FF0000'
                fig.add_trace(go.Scatter(
                    x=[point['coords'][0]],
                    y=[point['coords'][1]],
                    mode='markers+text',
                    name=point['label'],
                    marker=dict(size=12, color=marker_color, symbol='diamond'),
                    text=[point['label']],
                    textposition='top center',
                    hovertemplate=f"{point['label']} ({point['type']})<br>Coords: ({point['coords'][0]}, {point['coords'][1]})"
                ))
        
        annotations = []
        for zone, area in config['WORK_AREAS'].items():
            zone_df = positions_df[positions_df['zone'] == zone]
            if selected_step is not None:
                zone_df = zone_df[zone_df['step'] == selected_step]
            if len(zone_df) > area['workers'] * 1.5:
                annotations.append(dict(
                    x=area['center'][0],
                    y=area['center'][1],
                    text=f"Overcrowded: {len(zone_df)} workers<br>Reassign",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30,
                    font=dict(color='#EF4444', size=12)
                ))
        
        if show_entry_exit:
            for point in config['ENTRY_EXIT_POINTS']:
                x, y = point['coords']
                nearby_workers = positions_df[
                    (positions_df['step'] == selected_step) &
                    (np.sqrt((positions_df['x'] - x)**2 + (positions_df['y'] - y)**2) < 10)
                ]
                if len(nearby_workers) > 5:
                    annotations.append(dict(
                        x=x,
                        y=y,
                        text=f"High traffic: {len(nearby_workers)}<br>Monitor",
                        showarrow=True,
                        arrowhead=1,
                        ax=20,
                        ay=30,
                        font=dict(color='#FBBF24', size=12)
                    ))
        
        if show_production_lines:
            for line in config['PRODUCTION_LINES']:
                mid_x = (line['start'][0] + line['end'][0]) / 2
                mid_y = (line['start'][1] + line['end'][1]) / 2
                fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    text=line['label'],
                    showarrow=False,
                    font=dict(color='#FFFFFF', size=12),
                    yshift=15
                )
        
        fig.update_layout(
            title=dict(text='Worker Distribution with Facility Layout', x=0.5, font_size=24),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            xaxis=dict(range=[0, facility_size], gridcolor="#444", zerolinecolor="#444"),
            yaxis=dict(range=[0, facility_size], gridcolor="#444", zerolinecolor="#444"),
            font=dict(color='#E6ECEF', size=16),
            template='plotly_dark',
            plot_bgcolor='#1A252F',
            paper_bgcolor='#1A252F',
            shapes=shapes,
            annotations=annotations,
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
            height=600,
            transition_duration=500
        )
    return fig

def plot_worker_density_heatmap(positions_df, facility_size, config, show_entry_exit=True, show_production_lines=True):
    """
    Plot worker density heatmap with enhanced graphics and annotations.
    """
    x_bins = np.linspace(0, facility_size, config['DENSITY_GRID_SIZE'])
    y_bins = np.linspace(0, facility_size, config['DENSITY_GRID_SIZE'])
    heatmap, xedges, yedges = np.histogram2d(positions_df['x'], positions_df['y'], bins=[x_bins, y_bins])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap.T,
        x=xedges,
        y=yedges,
        colorscale=sequential.Plasma,
        showscale=True,
        colorbar=dict(title="Workers", tickfont=dict(size=14)),
        hovertemplate='X: %{x} m<br>Y: %{y} m<br>Workers: %{z}<extra></extra>'
    ))
    
    shapes = []
    colors = {'Assembly Line': 'rgba(59, 130, 246, 0.4)', 'Packaging Zone': 'rgba(16, 185, 129, 0.4)', 'Quality Control': 'rgba(236, 72, 153, 0.4)'}
    for zone, area in config['WORK_AREAS'].items():
        center_x, center_y = area['center']
        shapes.append(dict(
            type="rect",
            x0=center_x - 15,
            y0=center_y - 15,
            x1=center_x + 15,
            y1=center_y + 15,
            fillcolor=colors[zone],
            line=dict(color=colors[zone].replace('0.4', '1.0'), width=2),
            opacity=0.4,
            layer='below'
        ))
    
    if show_production_lines:
        for line in config['PRODUCTION_LINES']:
            shapes.append(dict(
                type="line",
                x0=line['start'][0],
                y0=line['start'][1],
                x1=line['end'][0],
                y1=line['end'][1],
                line=dict(color='#FFFFFF', width=4, dash='dash'),
                layer='below'
            ))
    
    if show_entry_exit:
        for point in config['ENTRY_EXIT_POINTS']:
            marker_color = '#00FF00' if point['type'] == 'Entry' else '#FF0000'
            fig.add_trace(go.Scatter(
                x=[point['coords'][0]],
                y=[point['coords'][1]],
                mode='markers+text',
                name=point['label'],
                marker=dict(size=12, color=marker_color, symbol='diamond'),
                text=[point['label']],
                textposition='top center',
                hovertemplate=f"{point['label']} ({point['type']})<br>Coords: ({point['coords'][0]}, {point['coords'][1]})"
            ))
    
    if show_production_lines:
        for line in config['PRODUCTION_LINES']:
            mid_x = (line['start'][0] + line['end'][0]) / 2
            mid_y = (line['start'][1] + line['end'][1]) / 2
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                text=line['label'],
                showarrow=False,
                font=dict(color='#FFFFFF', size=12),
                yshift=15
            )
    
    annotations = []
    for zone, area in config['WORK_AREAS'].items():
        zone_df = positions_df[positions_df['zone'] == zone]
        if len(zone_df) > area['workers'] * 1.5:
            annotations.append(dict(
                x=area['center'][0],
                y=area['center'][1],
                text=f"High density: {len(zone_df)}<br>Reassign",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#EF4444', size=12)
            ))
    
    if show_entry_exit:
        for point in config['ENTRY_EXIT_POINTS']:
            x, y = point['coords']
            nearby_workers = positions_df[
                (np.sqrt((positions_df['x'] - x)**2 + (positions_df['y'] - y)**2) < 10)
            ]
            if len(nearby_workers) > 5:
                annotations.append(dict(
                    x=x,
                    y=y,
                    text=f"High traffic: {len(nearby_workers)}<br>Monitor",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=30,
                    font=dict(color='#FBBF24', size=12)
                ))
    
    fig.update_layout(
        title=dict(text='Worker Density Heatmap', x=0.5, font_size=24),
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        xaxis=dict(range=[0, facility_size], gridcolor="#444", zerolinecolor="#444"),
        yaxis=dict(range=[0, facility_size], gridcolor="#444", zerolinecolor="#444"),
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        shapes=shapes,
        annotations=annotations,
        height=600,
        transition_duration=500
    )
    return fig

def plot_worker_wellbeing(wellbeing_scores, triggers):
    """
    Plot worker well-being with animated transitions and alerts.
    """
    minutes = [i * 2 for i in range(len(wellbeing_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=wellbeing_scores,
        mode='lines+markers',
        name='Well-Being',
        line=dict(color='#10B981', width=3),
        marker=dict(size=8),
        hovertemplate='Time: %{x} min<br>Well-Being: %{y:.1f}%<extra></extra>'
    ))
    
    threshold = DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Threshold ({threshold}%)",
        annotation_position="bottom right",
        annotation_font_size=12
    )
    
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
                            font=dict(color='#EF4444', size=12)
                        ))
                        y_offset += 5 if y_offset < 20 else -15
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
                        font=dict(color='#EF4444', size=12)
                    ))
                    y_offset += 5 if y_offset < 20 else -15
    
    fig.update_layout(
        title=dict(text='Worker Well-Being Index', x=0.5, font_size=24),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#444"),
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations[:5],
        transition_duration=500
    )
    return fig

def plot_psychological_safety(safety_scores):
    """
    Plot psychological safety with enhanced visualization.
    """
    minutes = [i * 2 for i in range(len(safety_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=safety_scores,
        mode='lines+markers',
        name='Psychological Safety',
        line=dict(color='#EC4899', width=3),
        marker=dict(size=8),
        hovertemplate='Time: %{x} min<br>Safety: %{y:.1f}%<extra></extra>'
    ))
    
    threshold = DEFAULT_CONFIG['SAFETY_THRESHOLD'] * 100
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Threshold ({threshold}%)",
        annotation_position="bottom right",
        annotation_font_size=12
    )
    
    annotations = []
    y_offset = 5
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
                font=dict(color='#EF4444', size=12)
            ))
            y_offset += 5 if y_offset < 20 else -15
    
    fig.update_layout(
        title=dict(text='Psychological Safety Score', x=0.5, font_size=24),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#444"),
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations[:5],
        transition_duration=500
    )
    return fig

def plot_downtime_trend(downtime_minutes, threshold):
    """
    Plot downtime trend with enhanced alerts.
    """
    minutes = [i * 2 for i in range(len(downtime_minutes))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=downtime_minutes,
        mode='lines+markers',
        name='Downtime',
        line=dict(color='#EF4444', width=3),
        marker=dict(size=8),
        hovertemplate='Time: %{x} min<br>Downtime: %{y:.1f} min<extra></extra>'
    ))
    
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#FBBF24",
        annotation_text=f"Threshold ({threshold} min)",
        annotation_position="bottom right",
        annotation_font_size=12
    )
    
    annotations = []
    y_offset = 2
    for i, (time, downtime) in enumerate(zip(minutes, downtime_minutes)):
        if downtime > threshold:
            annotations.append(dict(
                x=time,
                y=downtime + y_offset,
                text=f"High downtime: {downtime:.1f} min<br>Investigate",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#FBBF24', size=12)
            ))
            y_offset += 2 if y_offset < 10 else -5
    
    fig.update_layout(
        title=dict(text='Downtime Trend', x=0.5, font_size=24),
        xaxis_title='Time (minutes)',
        yaxis_title='Downtime (minutes)',
        yaxis=dict(gridcolor="#444"),
        font=dict(color='#E6ECEF', size=16),
        template='plotly_dark',
        plot_bgcolor='#1A252F',
        paper_bgcolor='#1A252F',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='right', x=1),
        annotations=annotations[:5],
        transition_duration=500
    )
    return fig
