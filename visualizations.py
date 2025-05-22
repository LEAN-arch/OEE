"""
visualizations.py
Functions to create Plotly visualizations for the Industrial Workplace Shift Monitoring Dashboard.
"""

import plotly.graph_objects as go
from config import DEFAULT_CONFIG

def plot_psychological_safety(safety_scores):
    """
    Create a line plot for psychological safety scores with annotations for low scores.
    
    Args:
        safety_scores (list or array): Psychological safety scores over time (0–100%).
    
    Returns:
        go.Figure: Plotly figure object.
    """
    # Time indices (assuming 2-minute intervals)
    minutes = [i * 2 for i in range(len(safety_scores))]
    
    # Create figure
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
            y_offset += 5  # Increment offset for the next annotation at the same x
            
            # Reset offset if it exceeds a limit to avoid pushing off-screen
            if y_offset > 20:
                y_offset = 5
    
    # Limit annotations to max_annotations
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

# Other visualization functions remain unchanged (e.g., plot_task_compliance_score, plot_collaboration_proximity_index, etc.)
# Include them here if needed, but they are not affected by this fix.
