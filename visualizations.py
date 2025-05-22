# visualizations.py
# Enhanced Plotly visualizations for the Workplace Shift Monitoring Dashboard with error handling, improved aesthetics, animations, and accessibility.

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
    """
    Create a 2x2 grid of gauge charts with consistent styling, animations, and accessibility.

    Args:
        compliance_score (float): Average task compliance score (%).
        proximity_score (float): Average collaboration proximity score (%).
        wellbeing_score (float): Average worker well-being score (%).
        downtime_minutes (float): Total downtime in minutes.

    Returns:
        list: List of Plotly gauge chart figures.
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
        plot_gauge_chart(wellbeing_score, "Worker Well-Being", wellbeing_threshold, 100, "Schedule breaks if <70%"),
        plot_gauge_chart(downtime_minutes, "Downtime", downtime_threshold, 60, "Investigate if >30 min")
    ]
    return figs

def plot_gauge_chart(value, title, threshold, max_value=100, recommendation=None):
    """
    Create an enhanced gauge chart with interactivity, animations, and accessibility.

    Args:
        value (float): Current value to display.
        title (str): Chart title.
        threshold (float): Threshold for acceptable performance.
        max_value (float): Maximum value for the gauge.
        recommendation (str, optional): Actionable recommendation text.

    Returns:
        go.Figure: Plotly gauge chart figure.
    """
    value = max(0, min(value, max_value))
    colors = sequential.Plasma_r
    color_idx = int((value / max_value) * (len(colors) - 1))
    bar_color = colors[color_idx]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': threshold, 'increasing': {'color': "#34D399"}, 'decreasing': {'color': "#F87171"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#F5F7FA', 'family': 'Roboto'}},
        number={'suffix': "%" if max_value == 100 else " min", 'font': {'size': 40, 'color': '#F5F7FA'}},
        gauge={
            'axis': {
                'range': [0, max_value],
                'tickwidth': 2,
                'tickcolor': "#D1D5DB",
                'tickfont': {'color': '#D1D5DB', 'size': 14},
                'showticklabels': True
            },
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(45, 59, 85, 0.8)",
            'borderwidth': 2,
            'bordercolor': "#4B5EAA",
            'steps': [
                {'range': [0, threshold], 'color': "rgba(248, 113, 113, 0.5)"},
                {'range': [threshold, max_value], 'color': "rgba(52, 211, 153, 0.5)"}
            ],
            'threshold': {
                'line': {'color': "#F5F7FA", 'width': 4},
                'thickness': 0.8,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        margin=dict(l=30, r=30, t=60, b=60),
        height=300,
        annotations=[
            dict(
                text=recommendation,
                x=0.5,
                y=-0.25,
                showarrow=False,
                font=dict(size=12, color='#FBBF24' if value < threshold else '#34D399')
            ) if recommendation else None
        ],
        transition={'duration': 1000, 'easing': 'cubic-in-out'},
        showlegend=False
    )
    return fig

def plot_task_compliance_score(compliance_scores, disruptions, forecast, z_scores):
    """
    Plot task compliance with enhanced interactivity, animations, and input validation.

    Args:
        compliance_scores (list): Task compliance scores (%).
        disruptions (list): Time steps of disruptions.
        forecast (list, optional): Forecasted compliance scores.
        z_scores (list): Z-scores for anomaly detection.

    Returns:
        go.Figure: Plotly line chart figure.
    """
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
            line=dict(color='#4F46E5', width=3),
            marker=dict(size=8, line=dict(width=1.5, color='#F5F7FA')),
            hovertemplate='Time: %{x} min<br>Compliance: %{y:.1f}%<br>Z-Score: %{customdata:.2f}<extra></extra>',
            customdata=z_scores,
            showlegend=True
        ))
        if forecast is not None:
            fig.add_trace(go.Scatter(
                x=minutes,
                y=forecast,
                mode='lines',
                name='Forecast',
                line=dict(color='#FBBF24', width=2, dash='dash'),
                hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%<extra></extra>',
                showlegend=True
            ))
        for disruption in disruptions:
            if 0 <= disruption < len(minutes):
                fig.add_vline(
                    x=minutes[disruption],
                    line_dash="dot",
                    line_color="#F87171",
                    annotation_text="Disruption",
                    annotation_position="top",
                    annotation_font=dict(size=12, color='#F87171')
                )
        annotations = []
        for i, (score, z) in enumerate(zip(compliance_scores, z_scores)):
            if abs(z) > 2.0:
                annotation = dict(
                    x=minutes[i],
                    y=float(score),
                    text=f"Anomaly: {score:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    ax=30,
                    ay=-40,
                    font=dict(color='#F87171', size=12)
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
            title=dict(text='Task Compliance Score', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            xaxis=dict(
                gridcolor="#4B5EAA",
                zeroline=False,
                title_font=dict(size=16, family='Roboto'),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#4B5EAA",
                zeroline=False,
                title_font=dict(size=16, family='Roboto'),
                tickfont=dict(size=12)
            ),
            font=dict(color='#F5F7FA', size=14, family='Roboto'),
            template='plotly_dark',
            plot_bgcolor='#1E2A44',
            paper_bgcolor='#1E2A44',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Roboto')
            ),
            annotations=valid_annotations,
            transition={'duration': 1000, 'easing': 'cubic-in-out'},
            margin=dict(l=50, r=50, t=100, b=50),
            showlegend=True
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot task compliance: {str(e)}",
            extra={'user_action': 'Plot Task Compliance'}
        )
        raise

def plot_collaboration_proximity_index(proximity_scores, disruptions, forecast):
    """
    Plot collaboration proximity with consistent styling, animations, and accessibility.

    Args:
        proximity_scores (list): Collaboration proximity scores (%).
        disruptions (list): Time steps of disruptions.
        forecast (list, optional): Forecasted proximity scores.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    minutes = [i * 2 for i in range(len(proximity_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=proximity_scores,
        mode='lines+markers',
        name='Proximity Index',
        line=dict(color='#34D399', width=3),
        marker=dict(size=8, line=dict(width=1.5, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Proximity: %{y:.1f}%<extra></extra>',
        showlegend=True
    ))
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#FBBF24', width=2, dash='dash'),
            hovertemplate='Time: %{x} min<br>Forecast: %{y:.1f}%<extra></extra>',
            showlegend=True
        ))
    for disruption in disruptions:
        if 0 <= disruption < len(minutes):
            fig.add_vline(
                x=minutes[disruption],
                line_dash="dot",
                line_color="#F87171",
                annotation_text="Disruption",
                annotation_position="top",
                annotation_font=dict(size=12, color='#F87171')
            )
    annotations = []
    if np.mean(proximity_scores) < 60:
        annotations.append(dict(
            x=minutes[0],
            y=max(proximity_scores) + 5,
            text="Low collaboration<br>Encourage team activities",
            showarrow=True,
            arrowhead=2,
            ax=30,
            ay=-40,
            font=dict(color='#F87171', size=12)
        ))
    fig.update_layout(
        title=dict(text='Collaboration Proximity Index', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
        xaxis_title='Time (minutes)',
        yaxis_title='Index (%)',
        xaxis=dict(
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12, family='Roboto')
        ),
        annotations=annotations,
        transition={'duration': 1000, 'easing': 'cubic-in-out'},
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True
    )
    return fig

def plot_operational_recovery(recovery_scores, productivity_loss):
    """
    Plot operational recovery vs. productivity loss with animations and accessibility.

    Args:
        recovery_scores (list): Operational recovery scores (%).
        productivity_loss (list): Productivity loss percentages (%).

    Returns:
        go.Figure: Plotly line chart figure.
    """
    minutes = [i * 2 for i in range(len(recovery_scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=recovery_scores,
        mode='lines+markers',
        name='Operational Recovery',
        line=dict(color='#4F46E5', width=3),
        marker=dict(size=8, line=dict(width=1.5, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Recovery: %{y:.1f}%<extra></extra>',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=minutes,
        y=productivity_loss,
        mode='lines+markers',
        name='Productivity Loss',
        line=dict(color='#F87171', width=3),
        marker=dict(size=8, line=dict(width=1.5, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Loss: %{y:.1f}%<extra></extra>',
        showlegend=True
    ))
    annotations = []
    max_loss_idx = np.argmax(productivity_loss)
    if productivity_loss[max_loss_idx] > 10:
        annotations.append(dict(
            x=minutes[max_loss_idx],
            y=productivity_loss[max_loss_idx] + 5,
            text=f"High loss: {productivity_loss[max_loss_idx]:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=30,
            ay=-40,
            font=dict(color='#F87171', size=12)
        ))
    fig.update_layout(
        title=dict(text='Operational Recovery vs. Productivity Loss', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        xaxis=dict(
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12, family='Roboto')
        ),
        annotations=annotations,
        transition={'duration': 1000, 'easing': 'cubic-in-out'},
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True
    )
    return fig

def plot_operational_efficiency(efficiency_df, selected_metrics):
    """
    Plot operational efficiency metrics with animations and accessibility.

    Args:
        efficiency_df (pd.DataFrame): DataFrame with efficiency metrics.
        selected_metrics (list): List of metrics to plot.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    minutes = [i * 2 for i in range(len(efficiency_df))]
    fig = go.Figure()
    colors = {'uptime': '#4F46E5', 'throughput': '#34D399', 'quality': '#EC4899', 'oee': '#FBBF24'}
    for metric in selected_metrics:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=efficiency_df[metric],
            mode='lines+markers',
            name=metric.capitalize(),
            line=dict(color=colors[metric], width=3),
            marker=dict(size=8, line=dict(width=1.5, color='#F5F7FA')),
            hovertemplate=f'Time: %{{x}} min<br>{metric.capitalize()}: %{{y:.1f}}%<extra></extra>',
            showlegend=True
        ))
    annotations = []
    if 'oee' in selected_metrics and np.mean(efficiency_df['oee']) < 75:
        annotations.append(dict(
            x=minutes[0],
            y=max(efficiency_df['oee']) + 5,
            text="Low OEE<br>Optimize processes",
            showarrow=True,
            arrowhead=2,
            ax=30,
            ay=-40,
            font=dict(color='#F87171', size=12)
        ))
    fig.update_layout(
        title=dict(text='Operational Efficiency Metrics', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        xaxis=dict(
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12, family='Roboto')
        ),
        annotations=annotations,
        transition={'duration': 1000, 'easing': 'cubic-in-out'},
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True
    )
    return fig

def plot_worker_distribution(df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_production_lines=True):
    """
    Plot worker distribution in 2D or 3D with error handling, animations, and accessibility.

    Args:
        df (pd.DataFrame): Worker position data.
        facility_size (float): Size of the facility (meters).
        config (dict): Configuration dictionary.
        use_3d (bool): Whether to use 3D scatter plot.
        selected_step (int): Time step to display.
        show_entry_exit (bool): Whether to show entry/exit points.
        show_production_lines (bool): Whether to show production lines.

    Returns:
        go.Figure: Plotly scatter or 3D scatter figure.
    """
    filtered_df = df[df['step'] == selected_step]
    fig = go.Figure()

    if use_3d:
        fig.add_trace(go.Scatter3d(
            x=filtered_df['x'],
            y=filtered_df['y'],
            z=[selected_step * 2] * len(filtered_df),
            mode='markers',
            marker=dict(
                size=6,
                color=filtered_df['workload'],
                colorscale='Plasma',
                opacity=0.85,
                colorbar=dict(
                    title='Workload',
                    tickfont=dict(color='#F5F7FA', size=12),
                    titlefont=dict(size=14, family='Roboto')
                )
            ),
            text=filtered_df['worker'],
            hovertemplate='Worker: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Workload: %{marker.color:.2f}<extra></extra>',
            showlegend=False
        ))
        if show_entry_exit:
            for point in config.get('ENTRY_EXIT_POINTS', []):
                try:
                    x, y = point['coords']
                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[0],
                        mode='markers+text',
                        marker=dict(size=10, color='#F87171'),
                        text=[point['label']],
                        textposition='top center',
                        hoverinfo='none',
                        showlegend=False
                    ))
                except (KeyError, TypeError) as e:
                    logger.error(f"Failed to plot entry/exit point {point}: {str(e)}", extra={'user_action': 'Plot Worker Distribution'})
    else:
        fig.add_trace(go.Scatter(
            x=filtered_df['x'],
            y=filtered_df['y'],
            mode='markers',
            marker=dict(
                size=12,
                color=filtered_df['workload'],
                colorscale='Plasma',
                opacity=0.85,
                line=dict(width=1.5, color='#F5F7FA'),
                colorbar=dict(
                    title='Workload',
                    tickfont=dict(color='#F5F7FA', size=12),
                    titlefont=dict(size=14, family='Roboto')
                )
            ),
            text=filtered_df['worker'],
            hovertemplate='Worker: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Workload: %{marker.color:.2f}<extra></extra>',
            showlegend=False
        ))
        if show_entry_exit:
            for point in config.get('ENTRY_EXIT_POINTS', []):
                try:
                    x, y = point['coords']
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers+text',
                        marker=dict(size=14, color='#F87171'),
                        text=[point['label']],
                        textposition='top center',
                        hoverinfo='none',
                        showlegend=False
                    ))
                except (KeyError, TypeError) as e:
                    logger.error(f"Failed to plot entry/exit point {point}: {str(e)}", extra={'user_action': 'Plot Worker Distribution'})
        if show_production_lines:
            for zone, area in config.get('WORK_AREAS', {}).items():
                try:
                    center = area['center']
                    fig.add_shape(
                        type="rect",
                        x0=center[0] - 5, x1=center[0] + 5,
                        y0=center[1] - 5, y1=center[1] + 5,
                        line=dict(color='#34D399', width=2.5, dash='dash'),
                        fillcolor='rgba(52, 211, 153, 0.15)'
                    )
                    fig.add_annotation(
                        x=center[0], y=center[1],
                        text=zone,
                        showarrow=False,
                        font=dict(color='#34D399', size=12)
                    )
                except (KeyError, TypeError) as e:
                    logger.error(f"Failed to plot production line for zone {zone}: {str(e)}", extra={'user_action': 'Plot Worker Distribution'})
    
    fig.update_layout(
        title=dict(text=f'Worker Distribution at {selected_step * 2} min', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
        xaxis=dict(
            title='X (m)',
            range=[0, facility_size],
            gridcolor='#4B5EAA',
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Y (m)',
            range=[0, facility_size],
            gridcolor='#4B5EAA',
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=50),
        scene=dict(
            xaxis=dict(
                title='X (m)',
                range=[0, facility_size],
                backgroundcolor='#1E2A44',
                gridcolor='#4B5EAA',
                title_font=dict(size=16, family='Roboto'),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title='Y (m)',
                range=[0, facility_size],
                backgroundcolor='#1E2A44',
                gridcolor='#4B5EAA',
                title_font=dict(size=16, family='Roboto'),
                tickfont=dict(size=12)
            ),
            zaxis=dict(
                title='Time (min)',
                backgroundcolor='#1E2A44',
                gridcolor='#4B5EAA',
                title_font=dict(size=16, family='Roboto'),
                tickfont=dict(size=12)
            )
        ) if use_3d else None,
        transition={'duration': 1000, 'easing': 'cubic-in-out'}
    )
    return fig

def plot_worker_density_heatmap(df, facility_size, config, show_entry_exit=True, show_production_lines=True):
    """
    Plot worker density heatmap with error handling, animations, and accessibility.

    Args:
        df (pd.DataFrame): Worker position data.
        facility_size (float): Size of the facility (meters).
        config (dict): Configuration dictionary.
        show_entry_exit (bool): Whether to show entry/exit points.
        show_production_lines (bool): Whether to show production lines.

    Returns:
        go.Figure: Plotly heatmap figure.
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
        colorbar=dict(
            title='Worker Count',
            tickfont=dict(color='#F5F7FA', size=12),
            titlefont=dict(size=14, family='Roboto')
        )
    ))
    if show_entry_exit:
        for point in config.get('ENTRY_EXIT_POINTS', []):
            try:
                x, y = point['coords']
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=14, color='#F87171'),
                    text=[point['label']],
                    textposition='top center',
                    hoverinfo='none',
                    showlegend=False
                ))
            except (KeyError, TypeError) as e:
                logger.error(f"Failed to plot entry/exit point {point}: {str(e)}", extra={'user_action': 'Plot Worker Density Heatmap'})
    if show_production_lines:
        for zone, area in config.get('WORK_AREAS', {}).items():
            try:
                center = area['center']
                fig.add_shape(
                    type="rect",
                    x0=center[0] - 5, x1=center[0] + 5,
                    y0=center[1] - 5, y1=center[1] + 5,
                    line=dict(color='#34D399', width=2.5, dash='dash'),
                    fillcolor='rgba(52, 211, 153, 0.15)'
                )
                fig.add_annotation(
                    x=center[0], y=center[1],
                    text=zone,
                    showarrow=False,
                    font=dict(color='#34D399', size=12)
                )
            except (KeyError, TypeError) as e:
                logger.error(f"Failed to plot production line for zone {zone}: {str(e)}", extra={'user__()
    fig.update_layout(
        title=dict(text='Worker Density Heatmap', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
        xaxis=dict(
            title='X (m)',
            range=[0, facility_size],
            gridcolor='#4B5EAA',
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Y (m)',
            range=[0, facility_size],
            gridcolor='#4B5EAA',
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        margin=dict(l=50, r=50, t=100, b=50),
        transition={'duration': 1000, 'easing': 'cubic-in-out'},
        showlegend=False
    )
    return fig

def plot_worker_wellbeing(scores, triggers):
    """
    Plot worker well-being with alerts, animations, and accessibility.

    Args:
        scores (list): Well-being scores (%).
        triggers (dict): Trigger events for alerts.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    minutes = [i * 2 for i in range(len(scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=scores,
        mode='lines+markers',
        name='Well-Being',
        line=dict(color='#4F46E5', width=3),
        marker=dict(size=8, line=dict(width=1.5, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<extra></extra>',
        showlegend=True
    ))
    for t in triggers['threshold']:
        if 0 <= t < len(minutes):
            fig.add_vline(
                x=minutes[t],
                line_dash="dot",
                line_color="#F87171",
                annotation_text="Low Score",
                annotation_position="top",
                annotation_font=dict(size=12, color='#F87171')
            )
    for t in triggers['trend']:
        if 0 <= t < len(minutes):
            fig.add_vline(
                x=minutes[t],
                line_dash="dot",
                line_color="#FBBF24",
                annotation_text="Declining",
                annotation_position="top",
                annotation_font=dict(size=12, color='#FBBF24')
            )
    fig.update_layout(
        title=dict(text='Worker Well-Being Index', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        xaxis=dict(
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12, family='Roboto')
        ),
        transition={'duration': 1000, 'easing': 'cubic-in-out'},
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True
    )
    return fig

def plot_psychological_safety(scores):
    """
    Plot psychological safety score with animations and accessibility.

    Args:
        scores (list): Psychological safety scores (%).

    Returns:
        go.Figure: Plotly line chart figure.
    """
    minutes = [i * 2 for i in range(len(scores))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=scores,
        mode='lines+markers',
        name='Psychological Safety',
        line=dict(color='#EC4899', width=3),
        marker=dict(size=8, line=dict(width=1.5, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Score: %{y:.1f}%<extra></extra>',
        showlegend=True
    ))
    annotations = []
    if np.mean(scores) < 70:
        annotations.append(dict(
            x=minutes[0],
            y=max(scores) + 5,
            text="Low safety<br>Promote open dialogue",
            showarrow=True,
            arrowhead=2,
            ax=30,
            ay=-40,
            font=dict(color='#F87171', size=12)
        ))
    fig.update_layout(
        title=dict(text='Psychological Safety Score', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
        xaxis_title='Time (minutes)',
        yaxis_title='Score (%)',
        xaxis=dict(
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, 100],
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12, family='Roboto')
        ),
        annotations=annotations,
        transition={'duration': 1000, 'easing': 'cubic-in-out'},
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True
    )
    return fig

def plot_downtime_trend(downtime_minutes, threshold):
    """
    Plot downtime trend with alerts, animations, and accessibility.

    Args:
        downtime_minutes (list): Downtime values in minutes.
        threshold (float): Downtime threshold for alerts.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    minutes = [i * 2 for i in range(len(downtime_minutes))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=minutes,
        y=downtime_minutes,
        mode='lines+markers',
        name='Downtime',
        line=dict(color='#F87171', width=3),
        marker=dict(size=8, line=dict(width=1.5, color='#F5F7FA')),
        hovertemplate='Time: %{x} min<br>Downtime: %{y:.1f} min<extra></extra>',
        showlegend=True
    ))
    annotations = []
    for i, downtime in enumerate(downtime_minutes):
        if downtime > threshold:
            annotations.append(dict(
                x=minutes[i],
                y=downtime + 0.5,
                text=f"High: {downtime:.1f} min",
                showarrow=True,
                arrowhead=2,
                ax=30,
                ay=-40,
                font=dict(color='#F87171', size=12)
            ))
    fig.update_layout(
        title=dict(text='Downtime Trend', x=0.5, xanchor='center', font=dict(size=22, family='Roboto')),
        xaxis_title='Time (minutes)',
        yaxis_title='Downtime (min)',
        xaxis=dict(
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            range=[0, max(max(downtime_minutes, default=0) + 2, threshold + 2)],
            gridcolor="#4B5EAA",
            zeroline=False,
            title_font=dict(size=16, family='Roboto'),
            tickfont=dict(size=12)
        ),
        font=dict(color='#F5F7FA', size=14, family='Roboto'),
        template='plotly_dark',
        plot_bgcolor='#1E2A44',
        paper_bgcolor='#1E2A44',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12, family='Roboto')
        ),
        annotations=annotations[:5],  # Limit annotations for clarity
        transition={'duration': 1000, 'easing': 'cubic-in-out'},
        margin=dict(l=50, r=50, t=100, b=50),
        showlegend=True
    )
    return fig
