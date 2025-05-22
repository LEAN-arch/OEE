# visualizations.py
# Plotly visualizations for Workplace Shift Monitoring Dashboard with robust error handling and enhanced Worker Insights plots.

import logging
import plotly.graph_objects as go
import numpy as np
from plotly.colors import sequential

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log'
)

def plot_key_metrics_summary(compliance_score, proximity_score, wellbeing_score, downtime_minutes):
    compliance_score = max(0, min(compliance_score, 100))
    proximity_score = max(0, min(proximity_score, 100))
    wellbeing_score = max(0, min(wellbeing_score, 100))
    downtime_minutes = max(0, downtime_minutes)

    figs = [
        go.Figure(go.Indicator(
            mode="gauge+number",
            value=compliance_score,
            title={'text': "Task Compliance", 'font': {'size': 18, 'color': '#F5F7FA'}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#4F46E5'}}
        )),
        go.Figure(go.Indicator(
            mode="gauge+number",
            value=proximity_score,
            title={'text': "Collaboration Proximity", 'font': {'size': 18, 'color': '#F5F7FA'}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#10B981'}}
        )),
        go.Figure(go.Indicator(
            mode="gauge+number",
            value=wellbeing_score,
            title={'text': "Worker Well-Being", 'font': {'size': 18, 'color': '#F5F7FA'}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#EC4899'}}
        )),
        go.Figure(go.Indicator(
            mode="gauge+number",
            value=downtime_minutes,
            title={'text': "Downtime", 'font': {'size': 18, 'color': '#F5F7FA'}},
            gauge={'axis': {'range': [0, 60]}, 'bar': {'color': '#EF4444'}}
        ))
    ]
    for fig in figs:
        fig.update_layout(
            font=dict(color='#F5F7FA', size=12, family='Inter'),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            margin=dict(l=20, r=20, t=50, b=50),
            height=280
        )
    return figs

def plot_task_compliance_score(compliance_scores, disruptions, forecast, z_scores):
    try:
        if not compliance_scores or not z_scores or len(compliance_scores) != len(z_scores):
            logger.error(
                f"Invalid inputs: compliance_scores={len(compliance_scores)}, z_scores={len(z_scores)}",
                extra={'user_action': 'Plot Task Compliance'}
            )
            raise ValueError("Invalid or mismatched compliance/z_scores data")
        
        compliance_scores = np.array(compliance_scores, dtype=float)
        z_scores = np.array(z_scores, dtype=float)
        if forecast is not None:
            forecast = np.array(forecast, dtype=float)
            if len(forecast) != len(compliance_scores):
                logger.error(
                    f"Forecast length mismatch: forecast={len(forecast)}, compliance={len(compliance_scores)}",
                    extra={'user_action': 'Plot Task Compliance'}
                )
                raise ValueError("Forecast length must match compliance_scores")
        
        compliance_scores = np.nan_to_num(compliance_scores, nan=0.0)
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        if forecast is not None:
            forecast = np.nan_to_num(forecast, nan=0.0)

        minutes = [i * 2 for i in range(len(compliance_scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=compliance_scores,
            mode='lines+markers',
            name='Task Compliance',
            line=dict(color='#4F46E5', width=2.5),
            marker=dict(size=6),
            hovertemplate='Time: %{x} min<br>Compliance: %{y:.1f}%<br>Z-Score: %{customdata:.2f}<extra></extra>',
            customdata=z_scores
        ))
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=minutes,
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color='#FBBF24', width=1.5, dash='dash')
            ))
        for disruption in disruptions:
            if 0 <= disruption < len(minutes):
                fig.add_vline(x=minutes[disruption], line_dash="dot", line_color="#EF4444")
        annotations = []
        for i, (score, z) in enumerate(zip(compliance_scores, z_scores)):
            if abs(z) > 2.0 and i < len(minutes):
                annotations.append(dict(
                    x=minutes[i],
                    y=float(score),
                    text=f"Anomaly: {score:.1f}%",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30,
                    font=dict(color='#EF4444', size=10)
                ))
        valid_annotations = [ann for ann in annotations[:5] if isinstance(ann, dict) and 'x' in ann and 'y' in ann]
        fig.update_layout(
            title=dict(text='Task Compliance Score', x=0.5),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            yaxis=dict(range=[0, 100]),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            annotations=valid_annotations
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot task compliance: {str(e)}", extra={'user_action': 'Plot Task Compliance'})
        raise

def plot_collaboration_proximity_index(proximity_scores, disruptions, forecast):
    if not proximity_scores:
        logger.error("Empty proximity_scores", extra={'user_action': 'Plot Collaboration Proximity'})
        raise ValueError("Proximity scores cannot be empty")
    
    minutes = [i * 2 for i in range(len(proximity_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=proximity_scores,
        mode='lines+markers',
        name='Proximity Index',
        line=dict(color='#10B981', width=2.5),
        marker=dict(size=6)
    ))
    if forecast is not None and len(forecast) == len(proximity_scores):
        fig.add_trace(go.Scatter(
            x=minutes,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#FBBF24', width=1.5, dash='dash')
        ))
    for disruption in disruptions:
        if 0 <= disruption < len(minutes):
            fig.add_vline(x=minutes[disruption], line_dash="dot", line_color="#EF4444")
    fig.update_layout(
        title=dict(text='Collaboration Proximity Index', x=0.5),
        xaxis_title='Time (minutes)',
        yaxis_title='Index (%)',
        yaxis=dict(range=[0, 100]),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44'
    )
    return fig

def plot_operational_recovery(recovery_scores, productivity_loss):
    if not recovery_scores or not productivity_loss:
        logger.error("Empty recovery_scores or productivity_loss", extra={'user_action': 'Plot Operational Recovery'})
        raise ValueError("Recovery scores and productivity loss cannot be empty")
    
    minutes = [i * 2 for i in range(len(recovery_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=recovery_scores,
        mode='lines+markers',
        name='Operational Recovery',
        line=dict(color='#4F46E5', width=2.5),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=minutes,
        y=productivity_loss,
        mode='lines+markers',
        name='Productivity Loss',
        line=dict(color='#EF4444', width=2.5),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title=dict(text='Operational Recovery vs. Productivity Loss', x=0.5),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100]),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44'
    )
    return fig

def plot_operational_efficiency(efficiency_df, selected_metrics):
    if efficiency_df.empty or not selected_metrics:
        logger.error("Empty efficiency_df or selected_metrics", extra={'user_action': 'Plot Operational Efficiency'})
        raise ValueError("Efficiency dataframe or metrics cannot be empty")
    
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
            marker=dict(size=6)
        ))
    fig.update_layout(
        title=dict(text='Operational Efficiency Metrics', x=0.5),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100]),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44'
    )
    return fig

def plot_worker_distribution(df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_production_lines=True, high_contrast=False):
    try:
        if df.empty:
            logger.warning("Empty dataframe for worker distribution", extra={'user_action': 'Plot Worker Distribution'})
            return go.Figure(layout=dict(title="No Worker Data Available"))

        filtered_df = df[df['step'] == selected_step]
        if filtered_df.empty:
            logger.warning(f"No data for step {selected_step}", extra={'user_action': 'Plot Worker Distribution'})
            return go.Figure(layout=dict(title=f"No Data at Step {selected_step}"))

        role_colors = {
            'Operator': '#4F46E5' if not high_contrast else '#00F',
            'Supervisor': '#10B981' if not high_contrast else '#0F0',
            'Technician': '#EC4899' if not high_contrast else '#F0F',
            'Unknown': '#6B7280'
        }
        fig = go.Figure()

        if use_3d:
            for role in filtered_df['role'].unique():
                role_df = filtered_df[filtered_df['role'] == role]
                fig.add_trace(go.Scatter3d(
                    x=role_df['x'],
                    y=role_df['y'],
                    z=[selected_step * 2] * len(role_df),
                    mode='markers',
                    name=role,
                    marker=dict(size=8, color=role_colors.get(role, role_colors['Unknown'])),
                    hovertemplate='Worker: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>',
                    text=role_df['worker']
                ))
        else:
            for role in filtered_df['role'].unique():
                role_df = filtered_df[filtered_df['role'] == role]
                fig.add_trace(go.Scatter(
                    x=role_df['x'],
                    y=role_df['y'],
                    mode='markers',
                    name=role,
                    marker=dict(size=12, color=role_colors.get(role, role_colors['Unknown'])),
                    hovertemplate='Worker: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>',
                    text=role_df['worker']
                ))

        if show_entry_exit:
            for point in config.get('ENTRY_EXIT_POINTS', []):
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    if use_3d:
                        fig.add_trace(go.Scatter3d(
                            x=[point[0]], y=[point[1]], z=[0],
                            mode='markers',
                            marker=dict(size=10, color='#EF4444' if not high_contrast else '#F00')
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=[point[0]], y=[point[1]],
                            mode='markers',
                            marker=dict(size=14, color='#EF4444' if not high_contrast else '#F00')
                        ))

        if show_production_lines:
            for zone, area in config.get('WORK_AREAS', {}).items():
                center = area.get('center', [])
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    if use_3d:
                        fig.add_trace(go.Scatter3d(
                            x=[center[0] - 5, center[0] + 5, center[0] + 5, center[0] - 5, center[0] - 5],
                            y=[center[1] - 5, center[1] - 5, center[1] + 5, center[1] + 5, center[1] - 5],
                            z=[0, 0, 0, 0, 0],
                            mode='lines',
                            line=dict(color='#10B981' if not high_contrast else '#0F0', width=2, dash='dash'),
                            name=zone
                        ))
                    else:
                        fig.add_shape(
                            type="rect",
                            x0=center[0] - 5, x1=center[0] + 5,
                            y0=center[1] - 5, y1=center[1] + 5,
                            line=dict(color='#10B981' if not high_contrast else '#0F0', width=2, dash='dash')
                        )

        fig.update_layout(
            title=dict(text=f'Worker Distribution at {selected_step * 2} min', x=0.5),
            xaxis=dict(title='X (meters)', range=[0, facility_size]),
            yaxis=dict(title='Y (meters)', range=[0, facility_size]),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            height=400,
            scene=dict(
                xaxis=dict(title='X (meters)', range=[0, facility_size]),
                yaxis=dict(title='Y (meters)', range=[0, facility_size]),
                zaxis=dict(title='Time (min)')
            ) if use_3d else None
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot worker distribution: {str(e)}", extra={'user_action': 'Plot Worker Distribution'})
        return go.Figure(layout=dict(title="Error Rendering Worker Distribution"))

def plot_worker_density_heatmap(df, facility_size, config, show_entry_exit=True, show_production_lines=True, intensity=1.0, high_contrast=False):
    try:
        if df.empty:
            logger.warning("Empty dataframe for heatmap", extra={'user_action': 'Plot Worker Density Heatmap'})
            return go.Figure(layout=dict(title="No Worker Data Available"))

        x_bins = np.linspace(0, facility_size, 50)
        y_bins = np.linspace(0, facility_size, 50)
        heatmap, xedges, yedges = np.histogram2d(df['x'], df['y'], bins=[x_bins, y_bins])
        heatmap = heatmap * intensity

        colorscale = 'Plasma' if not high_contrast else [
            [0, '#000'], [0.2, '#00F'], [0.4, '#0FF'], [0.6, '#0F0'], [0.8, '#FF0'], [1, '#F00']
        ]
        fig = go.Figure(go.Heatmap(
            x=x_bins,
            y=y_bins,
            z=heatmap.T,
            colorscale=colorscale,
            hovertemplate='X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Count: %{z:.1f}<extra></extra>',
            colorbar=dict(title='Worker Count')
        ))

        if show_entry_exit:
            for point in config.get('ENTRY_EXIT_POINTS', []):
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    fig.add_trace(go.Scatter(
                        x=[point[0]], y=[point[1]],
                        mode='markers',
                        marker=dict(size=14, color='#EF4444' if not high_contrast else '#F00')
                    ))

        if show_production_lines:
            for zone, area in config.get('WORK_AREAS', {}).items():
                center = area.get('center', [])
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    fig.add_shape(
                        type="rect",
                        x0=center[0] - 5, x1=center[0] + 5,
                        y0=center[1] - 5, y1=center[1] + 5,
                        line=dict(color='#10B981' if not high_contrast else '#0F0', width=2, dash='dash')
                    )

        fig.update_layout(
            title=dict(text='Worker Density Heatmap', x=0.5),
            xaxis=dict(title='X (meters)', range=[0, facility_size]),
            yaxis=dict(title='Y (meters)', range=[0, facility_size]),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            height=400
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot worker density heatmap: {str(e)}", extra={'user_action': 'Plot Worker Density Heatmap'})
        return go.Figure(layout=dict(title="Error Rendering Density Heatmap"))

def plot_worker_wellbeing(scores, triggers, high_contrast=False):
    try:
        if not scores:
            logger.warning("Empty well-being scores", extra={'user_action': 'Plot Worker Well-Being'})
            return go.Figure(layout=dict(title="No Well-Being Data Available"))

        minutes = [i * 2 for i in range(len(scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=scores,
            mode='lines+markers',
            name='Well-Being',
            line=dict(color='#4F46E5' if not high_contrast else '#00F', width=2.5),
            marker=dict(size=8),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<extra></extra>'
        ))
        fig.add_hline(
            y=DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100,
            line_dash="dash",
            line_color='#FBBF24' if not high_contrast else '#FF0'
        )
        fig.update_layout(
            title=dict(text='Worker Well-Being Index', x=0.5),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            yaxis=dict(range=[0, 100]),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            height=400
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot worker well-being: {str(e)}", extra={'user_action': 'Plot Worker Well-Being'})
        return go.Figure(layout=dict(title="Error Rendering Well-Being Chart"))

def plot_psychological_safety(scores, high_contrast=False):
    try:
        if not scores:
            logger.warning("Empty psychological safety scores", extra={'user_action': 'Plot Psychological Safety'})
            return go.Figure(layout=dict(title="No Psychological Safety Data Available"))

        minutes = [i * 2 for i in range(len(scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=scores,
            mode='lines+markers',
            name='Psychological Safety',
            line=dict(color='#EC4899' if not high_contrast else '#F0F', width=2.5),
            marker=dict(size=8),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<extra></extra>'
        ))
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color='#FBBF24' if not high_contrast else '#FF0'
        )
        fig.update_layout(
            title=dict(text='Psychological Safety Score', x=0.5),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            yaxis=dict(range=[0, 100]),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            height=400
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot psychological safety: {str(e)}", extra={'user_action': 'Plot Psychological Safety'})
        return go.Figure(layout=dict(title="Error Rendering Psychological Safety Chart"))

def plot_downtime_trend(downtime_minutes, threshold):
    if not downtime_minutes:
        logger.warning("Empty downtime_minutes", extra={'user_action': 'Plot Downtime Trend'})
        return go.Figure(layout=dict(title="No Downtime Data Available"))

    minutes = [i * 2 for i in range(len(downtime_minutes))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=downtime_minutes,
        mode='lines+markers',
        name='Downtime',
        line=dict(color='#EF4444', width=2.5),
        marker=dict(size=6)
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#FBBF24")
    fig.update_layout(
        title=dict(text='Downtime Trend', x=0.5),
        xaxis_title='Time (minutes)',
        yaxis_title='Downtime (min)',
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        height=400
    )
    return fig
