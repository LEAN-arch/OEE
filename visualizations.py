# visualizations.py
# Enhanced Plotly visualizations for the Workplace Shift Monitoring Dashboard.

import plotly.graph_objects as go
import numpy as np
from plotly.colors import sequential

def plot_key_metrics_summary(compliance_score, proximity_score, wellbeing_score, downtime_minutes):
    """
    Create a 2x2 grid of gauge charts with consistent styling.
    """
    compliance_score = max(0, min(compliance_score, 100))
    proximity_score = max(0, min(proximity_score, 100))
    wellbeing_score = max(0, min(wellbeing_score, 100))
    downtime_minutes = max(0, downtime_minutes)

    compliance_threshold = 75
    proximity_threshold = 60
    wellbeing_threshold = 70
    downtime_threshold = 30

    figs = [
        plot_gauge_chart(compliance_score, "Task Compliance", compliance_threshold, 100, "Review protocols if <75%"),
        plot_gauge_chart(proximity_score, "Collaboration Proximity", proximity_threshold, 100, "Encourage activities if <60%"),
        plot_gauge_chart(wellbeing_score, "Worker Well-Being", wellbeing_threshold, 100, "Schedule break if <70%"),
        plot_gauge_chart(downtime_minutes, "Downtime", downtime_threshold, 60, "Investigate if >30 min")
    ]
    return figs

def plot_gauge_chart(value, title, threshold, max_value=100, recommendation=None):
    """
    Create an enhanced gauge chart with interactivity.
    """
    value = max(0, min(value, max_value))
    colors = sequential.Plasma
    color_idx = int((value / max_value) * (len(colors) - 1))
    bar_color = colors[color_idx]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': threshold, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#F5F7FA', 'family': 'Inter'}},
        number={'suffix': "%" if max_value == 100 else " min", 'font': {'size': 36, 'color': '#F5F7FA'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': "#D1D5DB", 'tickfont': {'color': '#D1D5DB'}},
            'bar': {'color': bar_color},
            'bgcolor': "#2D3B55",
            'borderwidth': 1,
            'bordercolor': "#4B5EAA",
            'steps': [
                {'range': [0, threshold], 'color': "#EF4444"},
                {'range': [threshold, max_value], 'color': "#10B981"}
            ],
            'threshold': {
                'line': {'color': "#F5F7FA", 'width': 3},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        font=dict(color='#F5F7FA', size=14, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        margin=dict(l=20, r=20, t=50, b=50),
        height=280,
        annotations=[
            dict(
                text=recommendation,
                x=0.5,
                y=-0.2,
                showarrow=False,
                font=dict(size=12, color='#FBBF24' if value < threshold else '#10B981')
            ) if recommendation else None
        ],
        transition={'duration': 500}
    )
    return fig

def plot_task_compliance_score(compliance_scores, disruptions, forecast, z_scores):
    """
    Plot task compliance with enhanced interactivity.
    """
    minutes = [i * 2 for i in range(len(compliance_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=compliance_scores,
        mode='lines+markers',
        name='Task Compliance',
        line=dict(color='#4F46E5', width=2.5),
        marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Compliance: %{y:.1f}%<br>Z-Score: %{customdata:.2f}<extra></extra>',
        customdata=z_scores
    ))
    if forecast:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#FBBF24', width=1.5, dash='dash'),
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
                annotation_font=dict(size=10, color='#EF4444')
            )
    annotations = []
    for i, (score, z) in enumerate(zip(compliance_scores, z_scores)):
        if abs(z) > 2.0:
            annotations.append(dict(
                x=minutes[i],
                y=score,
                text=f"Anomaly: {score:.1f}%",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#EF4444', size=10)
            ))
    fig.update_layout(
        title=dict(text='Task Compliance Score', x=0.5, font=dict(size=20, family='Inter')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#4B5EAA"),
        font=dict(color='#F5F7FA', size=12, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
        annotations=annotations[:5],
        transition={'duration': 500},
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_collaboration_proximity_index(proximity_scores, disruptions, forecast):
    """
    Plot collaboration proximity with consistent styling.
    """
    minutes = [i * 2 for i in range(len(proximity_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=proximity_scores,
        mode='lines+markers',
        name='Proximity Index',
        line=dict(color='#10B981', width=2.5),
        marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Proximity: %{y:.1f}%<extra></extra>'
    ))
    if forecast:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#FBBF24', width=1.5, dash='dash'),
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
                annotation_font=dict(size=10, color='#EF4444')
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
            font=dict(color='#EF4444', size=10)
        ))
    fig.update_layout(
        title=dict(text='Collaboration Proximity Index', x=0.5, font=dict(size=20, family='Inter')),
        xaxis_title='Time (minutes)',
        yaxis_title='Index (%)',
        yaxis=dict(range=[0, 100], gridcolor="#4B5EAA"),
        font=dict(color='#F5F7FA', size=12, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
        annotations=annotations,
        transition={'duration': 500},
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_operational_recovery(recovery_scores, productivity_loss):
    """
    Plot operational recovery vs. productivity loss.
    """
    minutes = [i * 2 for i in range(len(recovery_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=recovery_scores,
        mode='lines+markers',
        name='Operational Recovery',
        line=dict(color='#4F46E5', width=2.5),
        marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Recovery: %{y:.1f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=minutes,
        y=productivity_loss,
        mode='lines+markers',
        name='Productivity Loss',
        line=dict(color='#EF4444', width=2.5),
        marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Loss: %{y:.1f}%<extra></extra>'
    ))
    annotations = []
    max_loss_idx = np.argmax(productivity_loss)
    if productivity_loss[max_loss_idx] > 10:
        annotations.append(dict(
            x=minutes[max_loss_idx],
            y=productivity_loss[max_loss_idx] + 5,
            text=f"High loss: {productivity_loss[max_loss_idx]:.1f}%",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-30,
            font=dict(color='#EF4444', size=10)
        ))
    fig.update_layout(
        title=dict(text='Operational Recovery vs. Productivity Loss', x=0.5, font=dict(size=20, family='Inter')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#4B5EAA"),
        font=dict(color='#F5F7FA', size=12, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
        annotations=annotations,
        transition={'duration': 500},
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_operational_efficiency(efficiency_df, selected_metrics):
    """
    Plot operational efficiency metrics.
    """
    minutes = [i * 2 for i in range(len(efficiency_df))]
    fig = go.Figure()
    colors = {'uptime': '#4F46E5', 'throughput': '#10B981', 'quality': '#EC4899', 'oee': '#FBBF24'}
    for metric in selected_metrics:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=efficiency_df[metric],
            mode='lines+markers',
            name=metric.capitalize(),
            line=dict(color=colors[metric], width=2.5),
            marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
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
            font=dict(color='#EF4444', size=10)
        ))
    fig.update_layout(
        title=dict(text='Operational Efficiency Metrics', x=0.5, font=dict(size=20, family='Inter')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#4B5EAA"),
        font=dict(color='#F5F7FA', size=12, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
        annotations=annotations,
        transition={'duration': 500},
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_worker_distribution(df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_production_lines=True):
    """
    Plot worker distribution in 2D or 3D.
    """
    filtered_df = df[df['step'] == selected_step]
    if use_3d:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=filtered_df['x'],
            y=filtered_df['y'],
            z=[selected_step * 2] * len(filtered_df),
            mode='markers',
            marker=dict(
                size=5,
                color=filtered_df['workload'],
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title='Workload', tickfont=dict(color='#F5F7FA'))
            ),
            text=filtered_df['worker'],
            hovertemplate='Worker: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Workload: %{marker.color:.2f}<extra></extra>'
        ))
        if show_entry_exit:
            for point in config['ENTRY_EXIT_POINTS']:
                fig.add_trace(go.Scatter3d(
                    x=[point[0]], y=[point[1]], z=[0],
                    mode='markers+text',
                    marker=dict(size=8, color='#EF4444'),
                    text=['Entry/Exit'],
                    textposition='top center',
                    hoverinfo='none'
                ))
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (m)', range=[0, facility_size], backgroundcolor='#1E2A44', gridcolor='#4B5EAA'),
                yaxis=dict(title='Y (m)', range=[0, facility_size], backgroundcolor='#1E2A44', gridcolor='#4B5EAA'),
                zaxis=dict(title='Time (min)', backgroundcolor='#1E2A44', gridcolor='#4B5EAA')
            ),
            title=dict(text='Worker Distribution (3D)', x=0.5, font=dict(size=20, family='Inter')),
            font=dict(color='#F5F7FA', size=12, family='Inter'),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=False
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['x'],
            y=filtered_df['y'],
            mode='markers',
            marker=dict(
                size=10,
                color=filtered_df['workload'],
                colorscale='Plasma',
                opacity=0.8,
                line=dict(width=1, color='#F5F7FA'),
                colorbar=dict(title='Workload', tickfont=dict(color='#F5F7FA'))
            ),
            text=filtered_df['worker'],
            hovertemplate='Worker: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Workload: %{marker.color:.2f}<extra></extra>'
        ))
        if show_entry_exit:
            for point in config['ENTRY_EXIT_POINTS']:
                fig.add_trace(go.Scatter(
                    x=[point[0]], y=[point[1]],
                    mode='markers+text',
                    marker=dict(size=12, color='#EF4444'),
                    text=['Entry/Exit'],
                    textposition='top center',
                    hoverinfo='none'
                ))
        if show_production_lines:
            for zone, area in config['WORK_AREAS'].items():
                fig.add_shape(
                    type="rect",
                    x0=area['center'][0] - 5, x1=area['center'][0] + 5,
                    y0=area['center'][1] - 5, y1=area['center'][1] + 5,
                    line=dict(color='#10B981', width=2, dash='dash'),
                    fillcolor='rgba(16, 185, 129, 0.1)'
                )
                fig.add_annotation(
                    x=area['center'][0], y=area['center'][1],
                    text=zone,
                    showarrow=False,
                    font=dict(color='#10B981', size=10)
                )
        fig.update_layout(
            title=dict(text=f'Worker Distribution at {selected_step * 2} min', x=0.5, font=dict(size=20, family='Inter')),
            xaxis=dict(title='X (m)', range=[0, facility_size], gridcolor='#4B5EAA'),
            yaxis=dict(title='Y (m)', range=[0, facility_size], gridcolor='#4B5EAA'),
            font=dict(color='#F5F7FA', size=12, family='Inter'),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40)
        )
    return fig

def plot_worker_density_heatmap(df, facility_size, config, show_entry_exit=True, show_production_lines=True):
    """
    Plot worker density heatmap.
    """
    x_bins = np.linspace(0, facility_size, 50)
    y_bins = np.linspace(0, facility_size, 50)
    heatmap, xedges, yedges = np.histogram2d(df['x'], df['y'], bins=[x_bins, y_bins])
    fig = go.Figure(go.Heatmap(
        x=x_bins,
        y=y_bins,
        z=heatmap.T,
        colorscale='Plasma',
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Count: %{z}<extra></extra>',
        colorbar=dict(title='Worker Count', tickfont=dict(color='#F5F7FA'))
    ))
    if show_entry_exit:
        for point in config['ENTRY_EXIT_POINTS']:
            fig.add_trace(go.Scatter(
                x=[point[0]], y=[point[1]],
                mode='markers+text',
                marker=dict(size=12, color='#EF4444'),
                text=['Entry/Exit'],
                textposition='top center',
                hoverinfo='none'
            ))
    if show_production_lines:
        for zone, area in config['WORK_AREAS'].items():
            fig.add_shape(
                type="rect",
                x0=area['center'][0] - 5, x1=area['center'][0] + 5,
                y0=area['center'][1] - 5, y1=area['center'][1] + 5,
                line=dict(color='#10B981', width=2, dash='dash'),
                fillcolor='rgba(16, 185, 129, 0.1)'
            )
            fig.add_annotation(
                x=area['center'][0], y=area['center'][1],
                text=zone,
                showarrow=False,
                font=dict(color='#10B981', size=10)
            )
    fig.update_layout(
        title=dict(text='Worker Density Heatmap', x=0.5, font=dict(size=20, family='Inter')),
        xaxis=dict(title='X (m)', range=[0, facility_size], gridcolor='#4B5EAA'),
        yaxis=dict(title='Y (m)', range=[0, facility_size], gridcolor='#4B5EAA'),
        font=dict(color='#F5F7FA', size=12, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_worker_wellbeing(scores, triggers):
    """
    Plot worker well-being with alerts.
    """
    minutes = [i * 2 for i in range(len(scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=scores,
        mode='lines+markers',
        name='Well-Being',
        line=dict(color='#4F46E5', width=2.5),
        marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<extra></extra>'
    ))
    for t in triggers['threshold']:
        if 0 <= t < len(minutes):
            fig.add_vline(
                x=minutes[t],
                line_dash="dot",
                line_color="#EF4444",
                annotation_text="Low Score",
                annotation_position="top",
                annotation_font=dict(size=10, color='#EF4444')
            )
    for t in triggers['trend']:
        if 0 <= t < len(minutes):
            fig.add_vline(
                x=minutes[t],
                line_dash="dot",
                line_color="#FBBF24",
                annotation_text="Declining",
                annotation_position="top",
                annotation_font=dict(size=10, color='#FBBF24')
            )
    fig.update_layout(
        title=dict(text='Worker Well-Being Index', x=0.5, font=dict(size=20, family='Inter')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#4B5EAA"),
        font=dict(color='#F5F7FA', size=12, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
        transition={'duration': 500},
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_psychological_safety(scores):
    """
    Plot psychological safety score.
    """
    minutes = [i * 2 for i in range(len(scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=scores,
        mode='lines+markers',
        name='Psychological Safety',
        line=dict(color='#EC4899', width=2.5),
        marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='Psychological Safety Score', x=0.5, font=dict(size=20, family='Inter')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100], gridcolor="#4B5EAA"),
        font=dict(color='#F5F7FA', size=12, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
        transition={'duration': 500},
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig

def plot_downtime_trend(downtime_minutes, threshold):
    """
    Plot downtime trend with alerts.
    """
    minutes = [i * 2 for i in range(len(downtime_minutes))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=downtime_minutes,
        mode='lines+markers',
        name='Downtime',
        line=dict(color='#EF4444', width=2.5),
        marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Downtime: %{y:.1f} min<extra></extra>'
    ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#FBBF24",
        annotation_text=f"Threshold: {threshold} min",
        annotation_position="top right",
        annotation_font=dict(size=10, color='#FBBF24')
    )
    fig.update_layout(
        title=dict(text='Downtime Trend', x=0.5, font=dict(size=20, family='Inter')),
        xaxis_title='Time (minutes)',
        yaxis_title='Downtime (min)',
        yaxis=dict(gridcolor="#4B5EAA"),
        font=dict(color='#F5F7FA', size=12, family='Inter'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
        transition={'duration': 500},
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig
