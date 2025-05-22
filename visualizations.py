# visualizations.py
# Enhanced Plotly visualizations for the Workplace Shift Monitoring Dashboard with error handling and improved Worker Insights plots.

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
    try:
        if not (len(compliance_scores) == len(z_scores) and (forecast is None or len(forecast) == len(compliance_scores))):
            logger.error(
                f"Input length mismatch: compliance_scores={len(compliance_scores)}, "
                f"z_scores={len(z_scores)}, forecast={'None' if forecast is None else len(forecast)}",
                extra={'user_action': 'Plot Task Compliance'}
            )
            raise ValueError("Input arrays must have the same length")
        
        compliance_scores = np.array(compliance_scores, dtype=float)
        z_scores = np.array(z_scores, dtype=float)
        if forecast is not None:
            forecast = np.array(forecast, dtype=float)
        
        if np.any(np.isnan(compliance_scores)) or np.any(np.isnan(z_scores)) or (forecast is not None and np.any(np.isnan(forecast))):
            logger.warning(
                "NaN values detected in inputs; replacing with zeros",
                extra={'user_action': 'Plot Task Compliance'}
            )
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
            marker=dict(size=6, line=dict(width=1, color='#F5F7FA')),
            hovertemplate='Time: %{x} min<br>Compliance: %{y:.1f}%<br>Z-Score: %{customdata:.2f}<extra></extra>',
            customdata=z_scores
        ))
        if forecast is not None:
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
                annotation = dict(
                    x=minutes[i],
                    y=float(score),
                    text=f"Anomaly: {score:.1f}%",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30,
                    font=dict(color='#EF4444', size=10)
                )
                annotations.append(annotation)
        
        valid_annotations = []
        for ann in annotations[:5]:
            if isinstance(ann, dict) and all(k in ann for k in ['x', 'y', 'text', 'showarrow']):
                if isinstance(ann['x'], (int, float)) and isinstance(ann['y'], (int, float)):
                    valid_annotations.append(ann)
                else:
                    logger.warning(
                        f"Invalid annotation coordinates: x={ann.get('x')}, y={ann.get('y')}",
                        extra={'user_action': 'Plot Task Compliance'}
                    )
            else:
                logger.warning(
                    f"Invalid annotation format: {ann}",
                    extra={'user_action': 'Plot Task Compliance'}
                )

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
            annotations=valid_annotations,
            transition={'duration': 500},
            showlegend=True,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot task compliance: {str(e)}",
            extra={'user_action': 'Plot Task Compliance'}
        )
        raise

def plot_collaboration_proximity_index(proximity_scores, disruptions, forecast):
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
    if forecast is not None:
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

def plot_worker_distribution(df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_production_lines=True, high_contrast=False):
    """
    Enhanced worker distribution plot with role-based coloring, annotations, and accessibility.
    """
    try:
        filtered_df = df[df['step'] == selected_step]
        if filtered_df.empty:
            logger.warning("Empty dataframe for selected step", extra={'user_action': 'Plot Worker Distribution'})
            return go.Figure()

        # Role-based colors
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
                    mode='markers+text',
                    name=role,
                    text=role_df['worker'],
                    textposition='top center',
                    marker=dict(
                        size=8,
                        color=role_colors.get(role, role_colors['Unknown']),
                        opacity=0.9,
                        line=dict(width=1, color='#F5F7FA')
                    ),
                    hovertemplate='Worker: %{text}<br>Role: ' + role + '<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Workload: %{customdata:.2f}<extra></extra>',
                    customdata=role_df['workload']
                ))
        else:
            for role in filtered_df['role'].unique():
                role_df = filtered_df[filtered_df['role'] == role]
                fig.add_trace(go.Scatter(
                    x=role_df['x'],
                    y=role_df['y'],
                    mode='markers+text',
                    name=role,
                    text=role_df['worker'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=role_colors.get(role, role_colors['Unknown']),
                        opacity=0.9,
                        line=dict(width=1, color='#F5F7FA')
                    ),
                    hovertemplate='Worker: %{text}<br>Role: ' + role + '<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Workload: %{customdata:.2f}<extra></extra>',
                    customdata=role_df['workload']
                ))

        if show_entry_exit:
            for point in config.get('ENTRY_EXIT_POINTS', []):
                try:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        logger.warning(f"Invalid entry/exit point: {point}", extra={'user_action': 'Plot Worker Distribution'})
                        continue
                    if use_3d:
                        fig.add_trace(go.Scatter3d(
                            x=[point[0]], y=[point[1]], z=[0],
                            mode='markers+text',
                            marker=dict(size=10, color='#EF4444' if not high_contrast else '#F00'),
                            text=['Entry/Exit'],
                            textposition='top center',
                            hoverinfo='none'
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=[point[0]], y=[point[1]],
                            mode='markers+text',
                            marker=dict(size=14, color='#EF4444' if not high_contrast else '#F00'),
                            text=['Entry/Exit'],
                            textposition='top center',
                            hoverinfo='none'
                        ))
                except (IndexError, TypeError) as e:
                    logger.error(f"Failed to plot entry/exit point {point}: {str(e)}", extra={'user_action': 'Plot Worker Distribution'})

        if show_production_lines:
            for zone, area in config.get('WORK_AREAS', {}).items():
                try:
                    center = area['center']
                    if not isinstance(center, (list, tuple)) or len(center) < 2:
                        logger.warning(f"Invalid center for zone {zone}: {center}", extra={'user_action': 'Plot Worker Distribution'})
                        continue
                    if use_3d:
                        fig.add_trace(go.Scatter3d(
                            x=[center[0] - 5, center[0] + 5, center[0] + 5, center[0] - 5, center[0] - 5],
                            y=[center[1] - 5, center[1] - 5, center[1] + 5, center[1] + 5, center[1] - 5],
                            z=[0, 0, 0, 0, 0],
                            mode='lines',
                            line=dict(color='#10B981' if not high_contrast else '#0F0', width=2, dash='dash'),
                            name=zone
                        ))
                        fig.add_trace(go.Scatter3d(
                            x=[center[0]], y=[center[1]], z=[0],
                            mode='text',
                            text=[zone],
                            textposition='middle center',
                            textfont=dict(color='#10B981' if not high_contrast else '#0F0', size=12)
                        ))
                    else:
                        fig.add_shape(
                            type="rect",
                            x0=center[0] - 5, x1=center[0] + 5,
                            y0=center[1] - 5, y1=center[1] + 5,
                            line=dict(color='#10B981' if not high_contrast else '#0F0', width=2, dash='dash'),
                            fillcolor='rgba(16, 185, 129, 0.1)' if not high_contrast else 'rgba(0, 255, 0, 0.1)'
                        )
                        fig.add_annotation(
                            x=center[0], y=center[1],
                            text=zone,
                            showarrow=False,
                            font=dict(color='#10B981' if not high_contrast else '#0F0', size=12)
                        )
                except (KeyError, TypeError) as e:
                    logger.error(f"Failed to plot production line for zone {zone}: {str(e)}", extra={'user_action': 'Plot Worker Distribution'})

        # Annotations for high workload
        annotations = []
        high_workload_df = filtered_df[filtered_df['workload'] > 0.8]
        for _, row in high_workload_df.head(3).iterrows():
            annotations.append(dict(
                x=row['x'],
                y=row['y'],
                z=selected_step * 2 if use_3d else None,
                text=f"High Workload: {row['workload']:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#FBBF24' if not high_contrast else '#FF0', size=10)
            ))

        fig.update_layout(
            title=dict(
                text=f'Worker Distribution at {selected_step * 2} min',
                x=0.5,
                font=dict(size=20, family='Inter'),
                automargin=True
            ),
            xaxis=dict(
                title='X (meters)',
                range=[0, facility_size],
                gridcolor='#4B5EAA',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='#F5F7FA'
            ),
            yaxis=dict(
                title='Y (meters)',
                range=[0, facility_size],
                gridcolor='#4B5EAA',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='#F5F7FA'
            ),
            font=dict(color='#F5F7FA', size=12, family='Inter'),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            hovermode='closest',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12)
            ),
            annotations=annotations if not use_3d else None,
            margin=dict(l=50, r=50, t=100, b=50),
            height=500,
            scene=dict(
                xaxis=dict(
                    title='X (meters)',
                    range=[0, facility_size],
                    backgroundcolor='#1E2A44',
                    gridcolor='#4B5EAA',
                    showline=True,
                    linewidth=1,
                    linecolor='#F5F7FA'
                ),
                yaxis=dict(
                    title='Y (meters)',
                    range=[0, facility_size],
                    backgroundcolor='#1E2A44',
                    gridcolor='#4B5EAA',
                    showline=True,
                    linewidth=1,
                    linecolor='#F5F7FA'
                ),
                zaxis=dict(
                    title='Time (min)',
                    backgroundcolor='#1E2A44',
                    gridcolor='#4B5EAA',
                    showline=True,
                    linewidth=1,
                    linecolor='#F5F7FA'
                ),
                annotations=annotations
            ) if use_3d else None,
            dragmode='pan',
            showlegend=True
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot worker distribution: {str(e)}", extra={'user_action': 'Plot Worker Distribution'})
        raise

def plot_worker_density_heatmap(df, facility_size, config, show_entry_exit=True, show_production_lines=True, intensity=1.0, high_contrast=False):
    """
    Enhanced heatmap with intensity control, annotations, and accessibility.
    """
    try:
        if df.empty:
            logger.warning("Empty dataframe for heatmap", extra={'user_action': 'Plot Worker Density Heatmap'})
            return go.Figure()

        x_bins = np.linspace(0, facility_size, 50)
        y_bins = np.linspace(0, facility_size, 50)
        heatmap, xedges, yedges = np.histogram2d(df['x'], df['y'], bins=[x_bins, y_bins])
        heatmap = heatmap * intensity  # Apply intensity scaling
        
        colorscale = 'Plasma' if not high_contrast else [
            [0, '#000'], [0.2, '#00F'], [0.4, '#0FF'], [0.6, '#0F0'], [0.8, '#FF0'], [1, '#F00']
        ]
        fig = go.Figure(go.Heatmap(
            x=x_bins,
            y=y_bins,
            z=heatmap.T,
            colorscale=colorscale,
            hovertemplate='X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Worker Count: %{z:.1f}<extra></extra>',
            colorbar=dict(
                title='Worker Count',
                tickfont=dict(color='#F5F7FA', size=12),
                titleside='right',
                thickness=20
            ),
            zmin=0,
            zmax=np.max(heatmap) * 1.2
        ))

        if show_entry_exit:
            for point in config.get('ENTRY_EXIT_POINTS', []):
                try:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        logger.warning(f"Invalid entry/exit point: {point}", extra={'user_action': 'Plot Worker Density Heatmap'})
                        continue
                    fig.add_trace(go.Scatter(
                        x=[point[0]], y=[point[1]],
                        mode='markers+text',
                        marker=dict(size=14, color='#EF4444' if not high_contrast else '#F00'),
                        text=['Entry/Exit'],
                        textposition='top center',
                        hoverinfo='none'
                    ))
                except (IndexError, TypeError) as e:
                    logger.error(f"Failed to plot entry/exit point {point}: {str(e)}", extra={'user_action': 'Plot Worker Density Heatmap'})

        if show_production_lines:
            for zone, area in config.get('WORK_AREAS', {}).items():
                try:
                    center = area['center']
                    if not isinstance(center, (list, tuple)) or len(center) < 2:
                        logger.warning(f"Invalid center for zone {zone}: {center}", extra={'user_action': 'Plot Worker Density Heatmap'})
                        continue
                    fig.add_shape(
                        type="rect",
                        x0=center[0] - 5, x1=center[0] + 5,
                        y0=center[1] - 5, y1=center[1] + 5,
                        line=dict(color='#10B981' if not high_contrast else '#0F0', width=2, dash='dash'),
                        fillcolor='rgba(16, 185, 129, 0.1)' if not high_contrast else 'rgba(0, 255, 0, 0.1)'
                    )
                    fig.add_annotation(
                        x=center[0], y=center[1],
                        text=zone,
                        showarrow=False,
                        font=dict(color='#10B981' if not high_contrast else '#0F0', size=12)
                    )
                except (KeyError, TypeError) as e:
                    logger.error(f"Failed to plot production line for zone {zone}: {str(e)}", extra={'user_action': 'Plot Worker Density Heatmap'})

        # Annotations for high-density areas
        annotations = []
        max_density = np.max(heatmap)
        if max_density > 5:
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            max_x = x_bins[max_idx[0]]
            max_y = y_bins[max_idx[1]]
            annotations.append(dict(
                x=max_x,
                y=max_y,
                text=f"High Density: {max_density:.1f}",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                font=dict(color='#FBBF24' if not high_contrast else '#FF0', size=12)
            ))

        fig.update_layout(
            title=dict(
                text='Worker Density Heatmap',
                x=0.5,
                font=dict(size=20, family='Inter'),
                automargin=True
            ),
            xaxis=dict(
                title='X (meters)',
                range=[0, facility_size],
                gridcolor='#4B5EAA',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='#F5F7FA'
            ),
            yaxis=dict(
                title='Y (meters)',
                range=[0, facility_size],
                gridcolor='#4B5EAA',
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='#F5F7FA'
            ),
            font=dict(color='#F5F7FA', size=12, family='Inter'),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            hovermode='closest',
            annotations=annotations,
            margin=dict(l=50, r=50, t=100, b=50),
            height=500,
            showlegend=False
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot worker density heatmap: {str(e)}", extra={'user_action': 'Plot Worker Density Heatmap'})
        raise

def plot_worker_wellbeing(scores, triggers, high_contrast=False):
    """
    Enhanced well-being plot with threshold line, annotations, and accessibility.
    """
    try:
        minutes = [i * 2 for i in range(len(scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=scores,
            mode='lines+markers',
            name='Well-Being',
            line=dict(color='#4F46E5' if not high_contrast else '#00F', width=3),
            marker=dict(size=8, line=dict(width=1, color='#F5F7FA')),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<br>Trend: %{customdata}<extra></extra>',
            customdata=['Declining' if i in triggers['trend'] else 'Stable' for i in range(len(scores))]
        ))
        
        # Add threshold line
        fig.add_hline(
            y=DEFAULT_CONFIG['WELLBEING_THRESHOLD'] * 100,
            line_dash="dash",
            line_color='#FBBF24' if not high_contrast else '#FF0',
            annotation_text="Threshold",
            annotation_position="top right",
            annotation_font=dict(size=12, color='#FBBF24' if not high_contrast else '#FF0')
        )

        # Annotations for triggers
        annotations = []
        for t in triggers['threshold']:
            if 0 <= t < len(minutes):
                annotations.append(dict(
                    x=minutes[t],
                    y=scores[t],
                    text="Low Score",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30,
                    font=dict(color='#EF4444' if not high_contrast else '#F00', size=12)
                ))
        for t in triggers['trend']:
            if 0 <= t < len(minutes):
                annotations.append(dict(
                    x=minutes[t],
                    y=scores[t],
                    text="Declining Trend",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30,
                    font=dict(color='#FBBF24' if not high_contrast else '#FF0', size=12)
                ))

        fig.update_layout(
            title=dict(
                text='Worker Well-Being Index',
                x=0.5,
                font=dict(size=20, family='Inter'),
                automargin=True
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            yaxis=dict(range=[0, 100], gridcolor="#4B5EAA", zeroline=False),
            font=dict(color='#F5F7FA', size=12, family='Inter'),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
            annotations=annotations,
            transition={'duration': 500},
            showlegend=True,
            margin=dict(l=50, r=50, t=100, b=50),
            height=450
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot worker well-being: {str(e)}", extra={'user_action': 'Plot Worker Well-Being'})
        raise

def plot_psychological_safety(scores, high_contrast=False):
    """
    Enhanced psychological safety plot with threshold line and annotations.
    """
    try:
        minutes = [i * 2 for i in range(len(scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=scores,
            mode='lines+markers',
            name='Psychological Safety',
            line=dict(color='#EC4899' if not high_contrast else '#F0F', width=3),
            marker=dict(size=8, line=dict(width=1, color='#F5F7FA')),
            hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<extra></extra>'
        ))
        
        # Add threshold line
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color='#FBBF24' if not high_contrast else '#FF0',
            annotation_text="Threshold",
            annotation_position="top right",
            annotation_font=dict(size=12, color='#FBBF24' if not high_contrast else '#FF0')
        )

        # Annotations for low scores
        annotations = []
        for i, score in enumerate(scores):
            if score < 70:
                annotations.append(dict(
                    x=minutes[i],
                    y=score,
                    text="Low Safety",
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30,
                    font=dict(color='#EF4444' if not high_contrast else '#F00', size=12)
                ))

        fig.update_layout(
            title=dict(
                text='Psychological Safety Score',
                x=0.5,
                font=dict(size=20, family='Inter'),
                automargin=True
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            yaxis=dict(range=[0, 100], gridcolor="#4B5EAA", zeroline=False),
            font=dict(color='#F5F7FA', size=12, family='Inter'),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5),
            annotations=annotations,
            transition={'duration': 500},
            showlegend=True,
            margin=dict(l=50, r=50, t=100, b=50),
            height=450
        )
        return fig
    except Exception as e:
        logger.error(f"Failed to plot psychological safety: {str(e)}", extra={'user_action': 'Plot Psychological Safety'})
        raise

def plot_downtime_trend(downtime_minutes, threshold):
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
