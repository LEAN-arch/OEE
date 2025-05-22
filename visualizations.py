import logging
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log'
)

def plot_key_metrics_summary(compliance_score, proximity_score, wellbeing_score, downtime_minutes):
    """
    Create a 2x2 grid of gauge charts with vibrant colors, gradients, and enhanced interactivity.

    Args:
        compliance_score (float): Average task compliance score (%).
        proximity_score (float): Average collaboration proximity score (%).
        wellbeing_score (float): Average worker well-being score (%).
        downtime_minutes (float): Total downtime in minutes.

    Returns:
        list: List of Plotly gauge chart figures.
    """
    # Validate inputs
    for score, name in [
        (compliance_score, "compliance_score"),
        (proximity_score, "proximity_score"),
        (wellbeing_score, "wellbeing_score"),
        (downtime_minutes, "downtime_minutes")
    ]:
        if not isinstance(score, (int, float)) or score is None or np.isnan(score):
            logger.error(
                f"Invalid input: {name}={score}, type={type(score)}",
                extra={'user_action': 'Plot Key Metrics'}
            )
            raise ValueError(f"Invalid {name}: must be a number, got {score}")

    # Log inputs
    logger.info(
        f"Inputs: compliance={compliance_score}, proximity={proximity_score}, "
        f"wellbeing={wellbeing_score}, downtime={downtime_minutes}",
        extra={'user_action': 'Plot Key Metrics'}
    )

    # Clamp values
    compliance_score = max(0, min(compliance_score, 100))
    proximity_score = max(0, min(proximity_score, 100))
    wellbeing_score = max(0, min(wellbeing_score, 100))
    downtime_minutes = max(0, downtime_minutes)

    # Define metrics
    metrics = [
        {
            'value': compliance_score,
            'title': "Task Compliance",
            'threshold': 75,
            'max_value': 100,
            'recommendation': "Review protocols if <75%",
            'colors': {'gradient': ['#10B981', '#34D399'], 'threshold': '#F472B6'}
        },
        {
            'value': proximity_score,
            'title': "Collaboration Proximity",
            'threshold': 60,
            'max_value': 100,
            'recommendation': "Encourage team activities if <60%",
            'colors': {'gradient': ['#3B82F6', '#60A5FA'], 'threshold': '#F472B6'}
        },
        {
            'value': wellbeing_score,
            'title': "Worker Well-Being",
            'threshold': 70,
            'max_value': 100,
            'recommendation': "Schedule breaks if <70%",
            'colors': {'gradient': ['#8B5CF6', '#A78BFA'], 'threshold': '#F472B6'}
        },
        {
            'value': downtime_minutes,
            'title': "Downtime",
            'threshold': 30,
            'max_value': 60,
            'recommendation': "Investigate causes if >30 min",
            'colors': {'gradient': ['#EF4444', '#F87171'], 'threshold': '#FBBF24'}
        }
    ]

    try:
        figs = [
            plot_gauge_chart(
                metric['value'],
                metric['title'],
                metric['threshold'],
                metric['max_value'],
                metric['recommendation'],
                metric['colors']
            ) for metric in metrics
        ]
        return figs
    except Exception as e:
        logger.error(
            f"Failed to plot key metrics: {str(e)}",
            extra={'user_action': 'Plot Key Metrics'}
        )
        raise

def plot_gauge_chart(value, title, threshold, max_value=100, recommendation=None, colors=None):
    """
    Create a gauge chart with gradients, glassmorphic design, and dynamic tooltips.

    Args:
        value (float): Current value of the metric.
        title (str): Title of the gauge chart.
        threshold (float): Threshold for acceptable performance.
        max_value (float): Maximum value of the gauge.
        recommendation (str): Recommendation for low/high values.
        colors (dict): Gradient and threshold colors.

    Returns:
        go.Figure: Plotly gauge chart figure.
    """
    # Validate inputs
    if not isinstance(value, (int, float)) or value is None or np.isnan(value):
        logger.error(
            f"Invalid value: {value} for title={title}",
            extra={'user_action': 'Plot Gauge Chart'}
        )
        raise ValueError(f"Invalid value: must be a number, got {value}")
    if not isinstance(threshold, (int, float)) or threshold is None or np.isnan(threshold):
        logger.error(
            f"Invalid threshold: {threshold} for title={title}",
            extra={'user_action': 'Plot Gauge Chart'}
        )
        raise ValueError(f"Invalid threshold: must be a number, got {threshold}")
    if not isinstance(max_value, (int, float)) or max_value <= 0:
        logger.error(
            f"Invalid max_value: {max_value} for title={title}",
            extra={'user_action': 'Plot Gauge Chart'}
        )
        raise ValueError(f"Invalid max_value: must be positive, got {max_value}")
    if recommendation is not None and not isinstance(recommendation, str):
        logger.error(
            f"Invalid recommendation: {recommendation} for title={title}",
            extra={'user_action': 'Plot Gauge Chart'}
        )
        raise ValueError(f"Invalid recommendation: must be a string, got {recommendation}")

    # Default colors
    colors = colors or {
        'gradient': ['#10B981', '#34D399'],
        'threshold': '#F472B6'
    }

    # Clamp value
    value = max(0, min(value, max_value))

    # Calculate status
    status = "Optimal" if value >= threshold else "Needs Attention"
    delta_color = "#34D399" if value >= threshold else "#F87171"

    try:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            delta={
                'reference': threshold,
                'increasing': {'color': delta_color},
                'decreasing': {'color': delta_color},
                'font': {'size': 20, 'family': 'Inter, sans-serif'}
            },
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': title,
                'font': {'size': 24, 'color': '#E5E7EB', 'family': 'Inter, sans-serif', 'weight': '600'}
            },
            number={
                'suffix': "%" if max_value == 100 else " min",
                'font': {'size': 48, 'color': '#E5E7EB', 'family': 'Inter, sans-serif', 'weight': '600'},
                'valueformat': '.1f'
            },
            gauge={
                'axis': {
                    'range': [0, max_value],
                    'tickwidth': 1,
                    'tickcolor': "#6B7280",
                    'tickfont': {'color': '#6B7280', 'size': 14, 'family': 'Inter, sans-serif'},
                    'showticklabels': True
                },
                'bar': {
                    'color': {
                        'gradient': {
                            'type': 'linear',
                            'color': colors['gradient']
                        }
                    },
                    'thickness': 0.3,
                    'line': {'color': '#111827', 'width': 1}
                },
                'bgcolor': "rgba(31, 41, 55, 0.8)",  # Glassmorphic
                'borderwidth': 0,
                'shape': 'angular',
                'steps': [
                    {'range': [0, threshold], 'color': "rgba(248, 113, 113, 0.2)"},
                    {'range': [threshold, max_value], 'color': "rgba(52, 211, 153, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': colors['threshold'], 'width': 4},
                    'thickness': 0.8,
                    'value': threshold
                }
            }
        ))

        # Add neumorphic shadow
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.05)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            borderpad=10,
            opacity=0.7,
            width=280,
            height=280
        )

        # Dynamic tooltip
        hover_template = (
            f"<b>{title}</b><br>"
            f"Value: %{{value:.1f}}{'%' if max_value == 100 else ' min'}<br>"
            f"Status: {status}<br>"
            f"Recommendation: {recommendation}<extra></extra>"
        )
        fig.update_traces(hovertemplate=hover_template)

        # Update layout
        fig.update_layout(
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='rgba(17, 24, 39, 0.9)',
            paper_bgcolor='rgba(17, 24, 39, 0.9)',
            height=320,
            margin=dict(l=30, r=30, t=60, b=30),
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            annotations=[
                dict(
                    x=0.5,
                    y=-0.1,
                    xref="paper",
                    yref="paper",
                    text=f"Status: {status}",
                    showarrow=False,
                    font=dict(size=12, color='#9CA3AF', family='Inter, sans-serif')
                )
            ]
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot gauge chart for {title}: {str(e)}",
            extra={'user_action': 'Plot Gauge Chart'}
        )
        raise

def plot_task_compliance_score(compliance_scores, disruptions, forecast, z_scores):
    """
    Plot task compliance with modern colors and annotations.

    Args:
        compliance_scores (list): List of compliance scores (%).
        disruptions (list): List of disruption time indices.
        forecast (list): List of forecast values, or None.
        z_scores (list): List of z-scores for anomaly detection.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    try:
        # Validate inputs
        compliance_scores = np.array(compliance_scores, dtype=float)
        z_scores = np.array(z_scores, dtype=float)
        if forecast is not None:
            forecast = np.array(forecast, dtype=float)
        if not (len(compliance_scores) == len(z_scores) and (forecast is None or len(forecast) == len(compliance_scores))):
            logger.error(
                f"Input length mismatch: compliance_scores={len(compliance_scores)}, "
                f"z_scores={len(z_scores)}, forecast={'None' if forecast is None else len(forecast)}",
                extra={'user_action': 'Plot Task Compliance'}
            )
            raise ValueError("Input arrays must have the same length")
        if not disruptions or not isinstance(disruptions, (list, np.ndarray)):
            disruptions = []

        # Handle NaNs
        compliance_scores = np.nan_to_num(compliance_scores, nan=0.0)
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        forecast = np.nan_to_num(forecast, nan=0.0) if forecast is not None else None

        minutes = [i * 2 for i in range(len(compliance_scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=compliance_scores,
            mode='lines+markers',
            name='Task Compliance',
            line=dict(color='#4F46E5', width=3),
            marker=dict(size=8, line=dict(width=1, color='#E5E7EB')),
            hovertemplate=(
                'Time: %{x} min<br>'
                'Compliance: %{y:.1f}%<br>'
                'Z-Score: %{customdata:.2f}<br>'
                'Action: Review protocols if <75%'
                '<extra></extra>'
            ),
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
            if isinstance(disruption, (int, float)) and 0 <= disruption < len(minutes):
                fig.add_vline(
                    x=minutes[int(disruption)],
                    line_dash="dot",
                    line_color="#F87171",
                    annotation_text="Disruption",
                    annotation_position="top",
                    annotation_font=dict(size=12, color='#F87171', family='Inter, sans-serif')
                )
            else:
                logger.warning(
                    f"Invalid disruption index: {disruption}, max index: {len(minutes)-1}",
                    extra={'user_action': 'Plot Task Compliance'}
                )
        annotations = []
        for i, (score, z) in enumerate(zip(compliance_scores, z_scores)):
            if abs(z) > 2.0:
                annotations.append(dict(
                    x=minutes[i],
                    y=float(score),
                    text=f"Anomaly: {score:.1f}%",
                    showarrow=True,
                    arrowhead=1,
                    ax=30,
                    ay=-40,
                    font=dict(color='#F87171', size=12, family='Inter, sans-serif')
                ))

        fig.update_layout(
            title=dict(
                text='Task Compliance Score',
                x=0.5,
                font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            xaxis=dict(
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Inter, sans-serif')
            ),
            annotations=annotations,
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            margin=dict(l=50, r=50, t=100, b=50)
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
    Plot collaboration proximity with modern colors and annotations.

    Args:
        proximity_scores (list): List of proximity scores (%).
        disruptions (list): List of disruption time indices.
        forecast (list): List of forecast values, or None.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    try:
        # Validate inputs
        proximity_scores = np.array(proximity_scores, dtype=float)
        if forecast is not None:
            forecast = np.array(forecast, dtype=float)
        if not (forecast is None or len(forecast) == len(proximity_scores)):
            logger.error(
                f"Input length mismatch: proximity_scores={len(proximity_scores)}, "
                f"forecast={'None' if forecast is None else len(forecast)}",
                extra={'user_action': 'Plot Collaboration Proximity'}
            )
            raise ValueError("Input arrays must have the same length")
        if not disruptions or not isinstance(disruptions, (list, np.ndarray)):
            disruptions = []

        # Handle NaNs
        proximity_scores = np.nan_to_num(proximity_scores, nan=0.0)
        forecast = np.nan_to_num(forecast, nan=0.0) if forecast is not None else None

        minutes = [i * 2 for i in range(len(proximity_scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=proximity_scores,
            mode='lines+markers',
            name='Proximity Index',
            line=dict(color='#34D399', width=3),
            marker=dict(size=8, line=dict(width=1, color='#E5E7EB')),
            hovertemplate=(
                'Time: %{x} min<br>'
                'Proximity: %{y:.1f}%<br>'
                'Action: Encourage activities if <60%'
                '<extra></extra>'
            ),
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
            if isinstance(disruption, (int, float)) and 0 <= disruption < len(minutes):
                fig.add_vline(
                    x=minutes[int(disruption)],
                    line_dash="dot",
                    line_color="#F87171",
                    annotation_text="Disruption",
                    annotation_position="top",
                    annotation_font=dict(size=12, color='#F87171', family='Inter, sans-serif')
                )
            else:
                logger.warning(
                    f"Invalid disruption index: {disruption}, max index: {len(minutes)-1}",
                    extra={'user_action': 'Plot Collaboration Proximity'}
                )
        annotations = []
        if np.mean(proximity_scores) < 60:
            annotations.append(dict(
                x=minutes[0],
                y=max(proximity_scores) + 5,
                text="Low collaboration<br>Encourage team activities",
                showarrow=True,
                arrowhead=1,
                ax=30,
                ay=-40,
                font=dict(color='#F87171', size=12, family='Inter, sans-serif')
            ))
        fig.update_layout(
            title=dict(
                text='Collaboration Proximity Index',
                x=0.5,
                font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Index (%)',
            xaxis=dict(
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Inter, sans-serif')
            ),
            annotations=annotations,
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            margin=dict(l=50, r=50, t=100, b=50)
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot collaboration proximity: {str(e)}",
            extra={'user_action': 'Plot Collaboration Proximity'}
        )
        raise

def plot_operational_recovery(recovery_scores, productivity_loss):
    """
    Plot operational recovery vs. productivity loss with modern colors.

    Args:
        recovery_scores (list): List of recovery scores (%).
        productivity_loss (list): List of productivity loss percentages.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    try:
        # Validate inputs
        recovery_scores = np.array(recovery_scores, dtype=float)
        productivity_loss = np.array(productivity_loss, dtype=float)
        if len(recovery_scores) != len(productivity_loss):
            logger.error(
                f"Input length mismatch: recovery_scores={len(recovery_scores)}, "
                f"productivity_loss={len(productivity_loss)}",
                extra={'user_action': 'Plot Operational Recovery'}
            )
            raise ValueError("Input arrays must have the same length")

        # Handle NaNs
        recovery_scores = np.nan_to_num(recovery_scores, nan=0.0)
        productivity_loss = np.nan_to_num(productivity_loss, nan=0.0)

        minutes = [i * 2 for i in range(len(recovery_scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=recovery_scores,
            mode='lines+markers',
            name='Operational Recovery',
            line=dict(color='#4F46E5', width=3),
            marker=dict(size=8, line=dict(width=1, color='#E5E7EB')),
            hovertemplate='Time: %{x} min<br>Recovery: %{y:.1f}%<extra></extra>',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=minutes,
            y=productivity_loss,
            mode='lines+markers',
            name='Productivity Loss',
            line=dict(color='#F87171', width=3),
            marker=dict(size=8, line=dict(width=1, color='#E ratings: []E7EB')),
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
                arrowhead=1,
                ax=30,
                ay=-40,
                font=dict(color='#F87171', size=12, family='Inter, sans-serif')
            ))
        fig.update_layout(
            title=dict(
                text='Operational Recovery vs. Productivity Loss',
                x=0.5,
                font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            xaxis=dict(
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Inter, sans-serif')
            ),
            annotations=annotations,
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            margin=dict(l=50, r=50, t=100, b=50)
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot operational recovery: {str(e)}",
            extra={'user_action': 'Plot Operational Recovery'}
        )
        raise

def plot_operational_efficiency(efficiency_df, selected_metrics):
    """
    Plot operational efficiency metrics with modern colors.

    Args:
        efficiency_df (pd.DataFrame): DataFrame with efficiency metrics (uptime, throughput, quality, oee).
        selected_metrics (list): List of metrics to display.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    try:
        # Validate inputs
        if not isinstance(efficiency_df, pd.DataFrame):
            logger.error(
                f"Invalid efficiency_df: expected DataFrame, got {type(efficiency_df)}",
                extra={'user_action': 'Plot Operational Efficiency'}
            )
            raise ValueError("efficiency_df must be a pandas DataFrame")
        if not selected_metrics or not all(metric in efficiency_df.columns for metric in selected_metrics):
            logger.error(
                f"Invalid selected_metrics: {selected_metrics}, available: {list(efficiency_df.columns)}",
                extra={'user_action': 'Plot Operational Efficiency'}
            )
            raise ValueError("Selected metrics must be non-empty and in DataFrame columns")

        # Handle NaNs
        efficiency_df = efficiency_df.fillna(0.0)
        if efficiency_df.empty:
            logger.warning(
                "Empty efficiency DataFrame",
                extra={'user_action': 'Plot Operational Efficiency'}
            )
            return go.Figure()

        minutes = [i * 2 for i in range(len(efficiency_df))]
        fig = go.Figure()
        colors = {
            'uptime': '#4F46E5',
            'throughput': '#34D399',
            'quality': '#EC4899',
            'oee': '#FBBF24'
        }
        for metric in selected_metrics:
            fig.add_trace(go.Scatter(
                x=minutes,
                y=efficiency_df[metric],
                mode='lines+markers',
                name=metric.capitalize(),
                line=dict(color=colors[metric], width=3),
                marker=dict(size=8, line=dict(width=1, color='#E5E7EB')),
                hovertemplate=f'Time: %{{x}} min<br>{metric.capitalize()}: %{{y:.1f}}%<extra></extra>',
                showlegend=True
            ))
        annotations = []
        if 'oee' in selected_metrics and efficiency_df['oee'].mean() < 75:
            annotations.append(dict(
                x=minutes[0],
                y=max(efficiency_df['oee']) + 5,
                text="Low OEE<br>Optimize processes",
                showarrow=True,
                arrowhead=1,
                ax=30,
                ay=-40,
                font=dict(color='#F87171', size=12, family='Inter, sans-serif')
            ))
        fig.update_layout(
            title=dict(
                text='Operational Efficiency Metrics',
                x=0.5,
                font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            xaxis=dict(
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Inter, sans-serif')
            ),
            annotations=annotations,
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            margin=dict(l=50, r=50, t=100, b=50)
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot operational efficiency: {str(e)}",
            extra={'user_action': 'Plot Operational Efficiency'}
        )
        raise

def plot_worker_distribution(team_positions_df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_production_lines=True):
    """
    Plot worker positions with modern colors and optional 3D visualization.

    Args:
        team_positions_df (pd.DataFrame): DataFrame with worker positions.
        facility_size (float): Size of the facility (meters).
        config (dict): Configuration dictionary.
        use_3d (bool): Whether to use 3D scatter plot.
        selected_step (int): Time step to display.
        show_entry_exit (bool): Whether to show entry/exit points.
        show_production_lines (bool): Whether to show production lines.

    Returns:
        go.Figure: Plotly scatter or 3D scatter figure.
    """
    try:
        # Validate inputs
        if not isinstance(team_positions_df, pd.DataFrame):
            logger.error(
                f"Invalid team_positions_df: expected DataFrame, got {type(team_positions_df)}",
                extra={'user_action': 'Plot Worker Distribution'}
            )
            raise ValueError("team_positions_df must be a pandas DataFrame")
        if not isinstance(config, dict):
            logger.error(
                f"Invalid config: expected dict, got {type(config)}",
                extra={'user_action': 'Plot Worker Distribution'}
            )
            raise ValueError("config must be a dictionary")
        if not isinstance(facility_size, (int, float)) or facility_size <= 0:
            logger.error(
                f"Invalid facility_size: {facility_size}",
                extra={'user_action': 'Plot Worker Distribution'}
            )
            raise ValueError("facility_size must be a positive number")

        # Handle NaNs
        team_positions_df = team_positions_df.fillna({'x': 0.0, 'y': 0.0, 'workload': 0.0})
        if team_positions_df.empty:
            logger.warning(
                "Empty team positions DataFrame",
                extra={'user_action': 'Plot Worker Distribution'}
            )
            return go.Figure()

        filtered_df = team_positions_df[team_positions_df['step'] == selected_step]
        if filtered_df.empty:
            logger.warning(
                f"No data for step {selected_step}",
                extra={'user_action': 'Plot Worker Distribution'}
            )
            return go.Figure()

        colors = {
            'Assembly Line': '#4F46E5',
            'Packaging Zone': '#34D399',
            'Quality Control': '#EC4899'
        }
        if use_3d:
            fig = go.Figure()
            for zone in filtered_df['zone'].unique():
                zone_df = filtered_df[filtered_df['zone'] == zone]
                fig.add_trace(go.Scatter3d(
                    x=zone_df['x'],
                    y=zone_df['y'],
                    z=zone_df['workload'] * 100,
                    mode='markers',
                    name=zone,
                    marker=dict(
                        size=6,
                        color=colors.get(zone, '#FBBF24'),
                        opacity=0.7,
                        line=dict(width=1, color='#E5E7EB')
                    ),
                    hovertemplate=(
                        'Worker: %{text}<br>'
                        'X: %{x:.1f} m<br>'
                        'Y: %{y:.1f} m<br>'
                        'Workload: %{z:.1f}%<extra></extra>'
                    ),
                    text=zone_df['worker']
                ))
            if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
                for point in config['ENTRY_EXIT_POINTS']:
                    fig.add_trace(go.Scatter3d(
                        x=[point['coords'][0]],
                        y=[point['coords'][1]],
                        z=[0],
                        mode='markers+text',
                        name=point['label'],
                        marker=dict(size=8, color='#F87171', symbol='diamond'),
                        text=[point['label']],
                        textposition='top center'
                    ))
            if show_production_lines and 'PRODUCTION_LINES' in config:
                for line in config['PRODUCTION_LINES']:
                    fig.add_trace(go.Scatter3d(
                        x=[line['start'][0], line['end'][0]],
                        y=[line['start'][1], line['end'][1]],
                        z=[0, 0],
                        mode='lines',
                        name=line['label'],
                        line=dict(color='#FBBF24', width=3)
                    ))
            fig.update_layout(
                title=dict(
                    text=f'Worker Distribution at {selected_step * 2} min',
                    x=0.5,
                    font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
                ),
                scene=dict(
                    xaxis_title='X (meters)',
                    yaxis_title='Y (meters)',
                    zaxis_title='Workload (%)',
                    xaxis=dict(range=[0, facility_size], gridcolor="#374151"),
                    yaxis=dict(range=[0, facility_size], gridcolor="#374151"),
                    zaxis=dict(range=[0, 100], gridcolor="#374151"),
                    bgcolor='#111827'
                ),
                font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
                template='plotly_dark',
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='top',
                    y=1.15,
                    xanchor='center',
                    x=0.5,
                    font=dict(size=12, family='Inter, sans-serif')
                ),
                transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
                margin=dict(l=50, r=50, t=100, b=50)
            )
        else:
            fig = go.Figure()
            for zone in filtered_df['zone'].unique():
                zone_df = filtered_df[filtered_df['zone'] == zone]
                fig.add_trace(go.Scatter(
                    x=zone_df['x'],
                    y=zone_df['y'],
                    mode='markers',
                    name=zone,
                    marker=dict(
                        size=10,
                        color=colors.get(zone, '#FBBF24'),
                        opacity=0.7,
                        line=dict(width=1, color='#E5E7EB')
                    ),
                    hovertemplate=(
                        'Worker: %{text}<br>'
                        'X: %{x:.1f} m<br>'
                        'Y: %{y:.1f} m<br>'
                        'Workload: %{customdata:.1f}%<extra></extra>'
                    ),
                    text=zone_df['worker'],
                    customdata=zone_df['workload'] * 100,
                    showlegend=True
                ))
            if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
                for point in config['ENTRY_EXIT_POINTS']:
                    fig.add_trace(go.Scatter(
                        x=[point['coords'][0]],
                        y=[point['coords'][1]],
                        mode='markers+text',
                        name=point['label'],
                        marker=dict(size=8, color='#F87171', symbol='diamond'),
                        text=[point['label']],
                        textposition='top center',
                        showlegend=True
                    ))
            if show_production_lines and 'PRODUCTION_LINES' in config:
                for line in config['PRODUCTION_LINES']:
                    fig.add_trace(go.Scatter(
                        x=[line['start'][0], line['end'][0]],
                        y=[line['start'][1], line['end'][1]],
                        mode='lines',
                        name=line['label'],
                        line=dict(color='#FBBF24', width=3),
                        showlegend=True
                    ))
            fig.update_layout(
                title=dict(
                    text=f'Worker Distribution at {selected_step * 2} min',
                    x=0.5,
                    font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
                ),
                xaxis_title='X (meters)',
                yaxis_title='Y (meters)',
                xaxis=dict(range=[0, facility_size], gridcolor="#374151"),
                yaxis=dict(range=[0, facility_size], gridcolor="#374151"),
                font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
                template='plotly_dark',
                plot_bgcolor='#111827',
                paper_bgcolor='#111827',
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='top',
                    y=1.15,
                    xanchor='center',
                    x=0.5,
                    font=dict(size=12, family='Inter, sans-serif')
                ),
                transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
                margin=dict(l=50, r=50, t=100, b=50)
            )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot worker distribution: {str(e)}",
            extra={'user_action': 'Plot Worker Distribution'}
        )
        raise

def plot_worker_density_heatmap(team_positions_df, facility_size, config, show_entry_exit=True, show_production_lines=True):
    """
    Plot worker density heatmap with modern colors.

    Args:
        team_positions_df (pd.DataFrame): DataFrame with worker positions.
        facility_size (float): Size of the facility (meters).
        config (dict): Configuration dictionary.
        show_entry_exit (bool): Whether to show entry/exit points.
        show_production_lines (bool): Whether to show production lines.

    Returns:
        go.Figure: Plotly heatmap figure.
    """
    try:
        # Validate inputs
        if not isinstance(team_positions_df, pd.DataFrame):
            logger.error(
                f"Invalid team_positions_df: expected DataFrame, got {type(team_positions_df)}",
                extra={'user_action': 'Plot Worker Density Heatmap'}
            )
            raise ValueError("team_positions_df must be a pandas DataFrame")
        if not isinstance(config, dict):
            logger.error(
                f"Invalid config: expected dict, got {type(config)}",
                extra={'user_action': 'Plot Worker Density Heatmap'}
            )
            raise ValueError("config must be a dictionary")
        if not isinstance(facility_size, (int, float)) or facility_size <= 0:
            logger.error(
                f"Invalid facility_size: {facility_size}",
                extra={'user_action': 'Plot Worker Density Heatmap'}
            )
            raise ValueError("facility_size must be a positive number")

        # Handle NaNs
        team_positions_df = team_positions_df.fillna({'x': 0.0, 'y': 0.0})
        if team_positions_df.empty:
            logger.warning(
                "Empty team positions DataFrame",
                extra={'user_action': 'Plot Worker Density Heatmap'}
            )
            return go.Figure()

        grid_size = config.get('DENSITY_GRID_SIZE', 50)
        x_bins = np.linspace(0, facility_size, grid_size + 1)
        y_bins = np.linspace(0, facility_size, grid_size + 1)
        heatmap, x_edges, y_edges = np.histogram2d(
            team_positions_df['x'], team_positions_df['y'], bins=[x_bins, y_bins]
        )
        fig = go.Figure(data=go.Heatmap(
            z=heatmap.T,
            x=x_edges[:-1],
            y=y_edges[:-1],
            colorscale='Plasma',
            hovertemplate='X: %{x:.1f} m<br>Y: %{y:.1f} m<br>Density: %{z}<extra></extra>',
            showscale=True
        ))
        if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
            for point in config['ENTRY_EXIT_POINTS']:
                fig.add_trace(go.Scatter(
                    x=[point['coords'][0]],
                    y=[point['coords'][1]],
                    mode='markers+text',
                    name=point['label'],
                    marker=dict(size=8, color='#F87171', symbol='diamond'),
                    text=[point['label']],
                    textposition='top center',
                    showlegend=True
                ))
        if show_production_lines and 'PRODUCTION_LINES' in config:
            for line in config['PRODUCTION_LINES']:
                fig.add_trace(go.Scatter(
                    x=[line['start'][0], line['end'][0]],
                    y=[line['start'][1], line['end'][1]],
                    mode='lines',
                    name=line['label'],
                    line=dict(color='#FBBF24', width=3),
                    showlegend=True
                ))
        fig.update_layout(
            title=dict(
                text='Worker Density Heatmap',
                x=0.5,
                font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
            ),
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            xaxis=dict(range=[0, facility_size], gridcolor="#374151"),
            yaxis=dict(range=[0, facility_size], gridcolor="#374151"),
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Inter, sans-serif')
            ),
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            margin=dict(l=50, r=50, t=100, b=50)
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot worker density heatmap: {str(e)}",
            extra={'user_action': 'Plot Worker Density Heatmap'}
        )
        raise

def plot_worker_wellbeing(wellbeing_scores, triggers):
    """
    Plot worker well-being with modern colors and trigger annotations.

    Args:
        wellbeing_scores (list): List of well-being scores (%).
        triggers (dict): Dictionary of trigger events.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    try:
        # Validate inputs
        wellbeing_scores = np.array(wellbeing_scores, dtype=float)
        if not isinstance(triggers, dict):
            logger.error(
                f"Invalid triggers: expected dict, got {type(triggers)}",
                extra={'user_action': 'Plot Worker Wellbeing'}
            )
            raise ValueError("triggers must be a dictionary")

        # Handle NaNs
        wellbeing_scores = np.nan_to_num(wellbeing_scores, nan=0.0)

        minutes = [i * 2 for i in range(len(wellbeing_scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=wellbeing_scores,
            mode='lines+markers',
            name='Well-Being Index',
            line=dict(color='#34D399', width=3),
            marker=dict(size=8, line=dict(width=1, color='#E5E7EB')),
            hovertemplate='Time: %{x} min<br>Well-Being: %{y:.1f}%<extra></extra>',
            showlegend=True
        ))
        annotations = []
        for trigger in triggers.get('threshold', []):
            if isinstance(trigger, (int, float)) and 0 <= trigger < len(minutes):
                annotations.append(dict(
                    x=minutes[int(trigger)],
                    y=wellbeing_scores[int(trigger)],
                    text="Low Well-Being",
                    showarrow=True,
                    arrowhead=1,
                    ax=30,
                    ay=-40,
                    font=dict(color='#F87171', size=12, family='Inter, sans-serif')
                ))
        for trigger in triggers.get('trend', []):
            if isinstance(trigger, (int, float)) and 0 <= trigger < len(minutes):
                annotations.append(dict(
                    x=minutes[int(trigger)],
                    y=wellbeing_scores[int(trigger)],
                    text="Declining Trend",
                    showarrow=True,
                    arrowhead=1,
                    ax=30,
                    ay=-40,
                    font=dict(color='#F87171', size=12, family='Inter, sans-serif')
                ))
        for zone, zone_triggers in triggers.get('work_area', {}).items():
            for trigger in zone_triggers:
                if isinstance(trigger, (int, float)) and 0 <= trigger < len(minutes):
                    annotations.append(dict(
                        x=minutes[int(trigger)],
                        y=wellbeing_scores[int(trigger)],
                        text=f"{zone} Alert",
                        showarrow=True,
                        arrowhead=1,
                        ax=30,
                        ay=-40,
                        font=dict(color='#F87171', size=12, family='Inter, sans-serif')
                    ))
        for disruption in triggers.get('disruption', []):
            if isinstance(disruption, (int, float)) and 0 <= disruption < len(minutes):
                fig.add_vline(
                    x=minutes[int(disruption)],
                    line_dash="dot",
                    line_color="#F87171",
                    annotation_text="Disruption",
                    annotation_position="top",
                    annotation_font=dict(size=12, color='#F87171', family='Inter, sans-serif')
                )
        fig.update_layout(
            title=dict(
                text='Worker Well-Being Index',
                x=0.5,
                font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Index (%)',
            xaxis=dict(
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Inter, sans-serif')
            ),
            annotations=annotations,
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            margin=dict(l=50, r=50, t=100, b=50)
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot worker wellbeing: {str(e)}",
            extra={'user_action': 'Plot Worker Wellbeing'}
        )
        raise

def plot_psychological_safety(safety_scores):
    """
    Plot psychological safety with modern colors and annotations.

    Args:
        safety_scores (list): List of psychological safety scores (%).

    Returns:
        go.Figure: Plotly line chart figure.
    """
    try:
        # Validate inputs
        safety_scores = np.array(safety_scores, dtype=float)

        # Handle NaNs
        safety_scores = np.nan_to_num(safety_scores, nan=0.0)

        minutes = [i * 2 for i in range(len(safety_scores))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=safety_scores,
            mode='lines+markers',
            name='Psychological Safety',
            line=dict(color='#EC4899', width=3),
            marker=dict(size=8, line=dict(width=1, color='#E5E7EB')),
            hovertemplate='Time: %{x} min<br>Safety: %{y:.1f}%<extra></extra>',
            showlegend=True
        ))
        annotations = []
        if np.mean(safety_scores) < 70:
            annotations.append(dict(
                x=minutes[0],
                y=max(safety_scores) + 5,
                text="Low safety<br>Promote open communication",
                showarrow=True,
                arrowhead=1,
                ax=30,
                ay=-40,
                font=dict(color='#F87171', size=12, family='Inter, sans-serif')
            ))
        fig.update_layout(
            title=dict(
                text='Psychological Safety Score',
                x=0.5,
                font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Score (%)',
            xaxis=dict(
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Inter, sans-serif')
            ),
            annotations=annotations,
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            margin=dict(l=50, r=50, t=100, b=50)
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot psychological safety: {str(e)}",
            extra={'user_action': 'Plot Psychological Safety'}
        )
        raise

def plot_downtime_trend(downtime_minutes, threshold):
    """
    Plot downtime trend with modern colors and threshold alerts.

    Args:
        downtime_minutes (list): List of downtime values (minutes).
        threshold (float): Threshold for high downtime alerts.

    Returns:
        go.Figure: Plotly line chart figure.
    """
    try:
        # Validate inputs
        downtime_minutes = np.array(downtime_minutes, dtype=float)
        if not isinstance(threshold, (int, float)):
            logger.error(
                f"Invalid threshold: {threshold}, expected number",
                extra={'user_action': 'Plot Downtime Trend'}
            )
            raise ValueError("threshold must be a number")

        # Handle NaNs
        downtime_minutes = np.nan_to_num(downtime_minutes, nan=0.0)

        minutes = [i * 2 for i in range(len(downtime_minutes))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=minutes,
            y=downtime_minutes,
            mode='lines+markers',
            name='Downtime',
            line=dict(color='#F87171', width=3),
            marker=dict(size=8, line=dict(width=1, color='#E5E7EB')),
            hovertemplate='Time: %{x} min<br>Downtime: %{y:.1f} min<extra></extra>',
            showlegend=True
        ))
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="#FBBF24",
            annotation_text="Threshold",
            annotation_position="right",
            annotation_font=dict(size=12, color='#FBBF24', family='Inter, sans-serif')
        )
        annotations = []
        for i, downtime in enumerate(downtime_minutes):
            if downtime > threshold:
                annotations.append(dict(
                    x=minutes[i],
                    y=downtime,
                    text=f"High: {downtime:.1f} min",
                    showarrow=True,
                    arrowhead=1,
                    ax=30,
                    ay=-40,
                    font=dict(color='#F87171', size=12, family='Inter, sans-serif')
                ))
        fig.update_layout(
            title=dict(
                text='Downtime Trend',
                x=0.5,
                font=dict(size=24, family='Inter, sans-serif', color='#E5E7EB', weight='600')
            ),
            xaxis_title='Time (minutes)',
            yaxis_title='Downtime (minutes)',
            xaxis=dict(
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            yaxis=dict(
                gridcolor="#374151",
                zeroline=False,
                title_font=dict(size=16, family='Inter, sans-serif'),
                tickfont=dict(size=12, family='Inter, sans-serif')
            ),
            font=dict(color='#E5E7EB', size=14, family='Inter, sans-serif'),
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='center',
                x=0.5,
                font=dict(size=12, family='Inter, sans-serif')
            ),
            annotations=annotations,
            transition={'duration': 600, 'easing': 'cubic-bezier(0.4, 0, 0.2, 1)'},
            margin=dict(l=50, r=50, t=100, b=50)
        )
        return fig
    except Exception as e:
        logger.error(
            f"Failed to plot downtime trend: {str(e)}",
            extra={'user_action': 'Plot Downtime Trend'}
        )
        raise
