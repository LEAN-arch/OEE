# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Consistent color scheme definitions
PLOTLY_TEMPLATE = "plotly_dark"
# Standard Theme Colors
PRIMARY_COLOR_STD = "#6366F1"  # Indigo-500
SECONDARY_COLOR_STD = "#22D3EE" # Cyan-400
ALERT_COLOR_STD = "#FACC15"    # Yellow-400
CRITICAL_COLOR_STD = "#F87171" # Red-400
HIGHLIGHT_COLOR_STD = "#A78BFA" # Violet-400
NEUTRAL_COLOR_STD = "#9CA3AF"  # Gray-400

# High Contrast Theme Colors
PRIMARY_COLOR_HC = "#FFFFFF"   # White
SECONDARY_COLOR_HC = "#FFFF00" # Yellow
ALERT_COLOR_HC = "#FFBF00"    # Amber
CRITICAL_COLOR_HC = "#FF0000" # Red
HIGHLIGHT_COLOR_HC = "#00FFFF" # Cyan
NEUTRAL_COLOR_HC = "#BEBEBE"  # Gray

def _get_colors(high_contrast=False):
    if high_contrast:
        return PRIMARY_COLOR_HC, SECONDARY_COLOR_HC, ALERT_COLOR_HC, CRITICAL_COLOR_HC, HIGHLIGHT_COLOR_HC, NEUTRAL_COLOR_HC
    return PRIMARY_COLOR_STD, SECONDARY_COLOR_STD, ALERT_COLOR_STD, CRITICAL_COLOR_STD, HIGHLIGHT_COLOR_STD, NEUTRAL_COLOR_STD

def _apply_common_layout_settings(fig, title_text, high_contrast=False, yaxis_title=None, xaxis_title="Time Step (Interval)"):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title_text, x=0.5, font=dict(size=16, color=p_c if high_contrast else "#EAEAEA")),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="#1F2937" if not high_contrast else "#111111",
        font=dict(color=p_c if high_contrast else "#D1D5DB", size=10), # Base font color and size
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.1)", bordercolor=n_c, borderwidth=0.5),
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(title=xaxis_title, gridcolor=n_c if high_contrast else "#374151", zerolinecolor=n_c if high_contrast else "#4A5568", showline=True, linewidth=1, linecolor=n_c if high_contrast else "#4A5568"),
        yaxis=dict(title=yaxis_title, gridcolor=n_c if high_contrast else "#374151", zerolinecolor=n_c if high_contrast else "#4A5568", showline=True, linewidth=1, linecolor=n_c if high_contrast else "#4A5568"),
        hovermode="x unified" # Better hover experience for time series
    )

def plot_key_metrics_summary(compliance, proximity, wellbeing, downtime, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    metrics_config = [
        ("Task Compliance", compliance, 75, p_c, "%"),
        ("Collaboration", proximity, 60, s_c, "%"),
        ("Well-Being", wellbeing, 70, h_c, "%"),
        ("Total Downtime", downtime, 30, a_c, " min") 
    ]
    figs = []

    for title, value, target, color, suffix in metrics_config:
        fig = go.Figure()
        is_lower_better = "Downtime" in title
        
        fig.add_trace(go.Indicator(
            mode="gauge+number" + ("+delta" if not is_lower_better else ""),
            value=float(value),
            delta={'reference': float(target), 
                   'increasing': {'color': color if not is_lower_better else cr_c}, 
                   'decreasing': {'color': cr_c if not is_lower_better else color}} if not is_lower_better else None,
            title={'text': title, 'font': {'size': 12, 'color': n_c if high_contrast else "#D1D5DB"}}, # Smaller title
            number={'suffix': suffix, 'font': {'size': 18, 'color': p_c if high_contrast else "#FFFFFF"}}, # Larger number
            gauge={
                'axis': {'range': [0, 100 if suffix == "%" else max(50, value * 1.5)], 'tickwidth': 1, 'tickcolor': n_c},
                'bar': {'color': color, 'thickness': 0.65},
                'bgcolor': "#2a3447" if not high_contrast else "#222222", # Slightly different gauge bg
                'borderwidth': 0.5, 'bordercolor': n_c,
                'steps': [
                    {'range': [0, target * (0.6 if not is_lower_better else 1.5) ], 'color': cr_c if not is_lower_better else color},
                    {'range': [target * (0.6 if not is_lower_better else 1.5), target * (0.9 if not is_lower_better else 2.5)], 'color': a_c if not is_lower_better else cr_c},
                ],
                'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': target} if not is_lower_better else None
            }
        ))
        fig.update_layout(height=180, margin=dict(l=15,r=15,t=35,b=15), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=p_c if high_contrast else "#D1D5DB"))
        figs.append(fig)
    return figs

def plot_task_compliance_score(data, disruption_points, forecast_data=None, z_scores=None, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    fig = go.Figure()
    x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Compliance', line=dict(color=p_c, width=2), marker=dict(size=5, symbol="circle-open")))
    if forecast_data and len(forecast_data)==len(data): fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast', line=dict(color=h_c, dash='dash', width=1.5)))
    for dp in disruption_points:
        if 0 <= dp < len(data): fig.add_vline(x=dp, line=dict(color=a_c, width=1, dash="dash"), annotation_text="D", annotation_position="top left", annotation_font_size=10)
    _apply_common_layout_settings(fig, "Task Compliance Score", high_contrast, yaxis_title="Score (%)")
    fig.update_yaxes(range=[max(0, min(data)-10 if data else 0), min(105, max(data)+10 if data else 100)])
    return fig

def plot_collaboration_proximity_index(data, disruption_points, forecast_data=None, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    fig = go.Figure()
    x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Collab. Index', line=dict(color=s_c, width=2), marker=dict(size=5, symbol="diamond-open")))
    if forecast_data and len(forecast_data)==len(data): fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast', line=dict(color=h_c, dash='dash', width=1.5)))
    for dp in disruption_points:
        if 0 <= dp < len(data): fig.add_vline(x=dp, line=dict(color=a_c, width=1, dash="dash"), annotation_text="D", annotation_position="bottom right", annotation_font_size=10)
    _apply_common_layout_settings(fig, "Collaboration Proximity Index", high_contrast, yaxis_title="Index (%)")
    fig.update_yaxes(range=[max(0, min(data)-10 if data else 0), min(105, max(data)+10 if data else 100)])
    return fig

def plot_operational_recovery(recovery_data, productivity_loss_data, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    fig = go.Figure()
    x_vals = list(range(len(recovery_data)))
    fig.add_trace(go.Scatter(x=x_vals, y=recovery_data, mode='lines', name='Op. Recovery', line=dict(color=p_c, width=2)))
    if productivity_loss_data and len(productivity_loss_data)==len(recovery_data): fig.add_trace(go.Scatter(x=x_vals, y=productivity_loss_data, mode='lines', name='Prod. Loss', line=dict(color=cr_c, dash='dot', width=1.5)))
    _apply_common_layout_settings(fig, "Operational Recovery & Loss", high_contrast, yaxis_title="Percentage (%)")
    fig.update_yaxes(range=[0, 105])
    return fig

def plot_operational_efficiency(efficiency_df, selected_metrics, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    fig = go.Figure()
    colors = {'uptime': p_c, 'throughput': s_c, 'quality': h_c, 'oee': cr_c}
    for metric in selected_metrics:
        if metric in efficiency_df.columns:
            fig.add_trace(go.Scatter(x=efficiency_df.index, y=efficiency_df[metric], mode='lines', name=metric.capitalize(), line=dict(color=colors.get(metric, n_c), width=2)))
    _apply_common_layout_settings(fig, "Operational Efficiency Metrics", high_contrast, yaxis_title="Efficiency Score (%)")
    fig.update_yaxes(range=[0, 105])
    return fig

def plot_worker_distribution(team_positions_df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    df_step = team_positions_df[team_positions_df['step'] == selected_step]
    if df_step.empty: return go.Figure().update_layout(title_text=f"No worker data for Step {selected_step}", template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1F2937")

    facility_width, facility_height = facility_size
    zone_colors = px.colors.qualitative.Plotly if not high_contrast else px.colors.qualitative.Bold # Bold for HC

    if use_3d:
        fig = px.scatter_3d(df_step, x='x', y='y', z='z', color='zone', hover_name='worker_id', 
                            range_x=[0, facility_width], range_y=[0, facility_height], range_z=[0,max(5, df_step['z'].max() if not df_step['z'].empty else 5)],
                            color_discrete_sequence=zone_colors, symbol='status')
        fig.update_scenes(aspectmode='data', xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False, 
                          xaxis_backgroundcolor="rgba(0,0,0,0)", yaxis_backgroundcolor="rgba(0,0,0,0)", zaxis_backgroundcolor="rgba(0,0,0,0)")
    else:
        fig = px.scatter(df_step, x='x', y='y', color='zone', hover_name='worker_id',
                         range_x=[-5, facility_width + 5], range_y=[-5, facility_height + 5],
                         color_discrete_sequence=zone_colors, symbol='status')
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False)

    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height, line=dict(color=n_c, width=1), fillcolor="rgba(0,0,0,0)", layer="below")]
    annotations = []
    if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
        for point in config['ENTRY_EXIT_POINTS']:
            shapes.append(go.layout.Shape(type="circle", x0=point['coords'][0]-1.5, y0=point['coords'][1]-1.5, x1=point['coords'][0]+1.5, y1=point['coords'][1]+1.5, fillcolor=a_c, line_color="white", opacity=0.8))
            annotations.append(dict(x=point['coords'][0], y=point['coords'][1]+3, text=point['name'][:1].upper(), showarrow=False, font=dict(size=9, color=p_c if high_contrast else "#FFFFFF")))
    if show_prod_lines and 'WORK_AREAS' in config:
         for area_name, area_details in config['WORK_AREAS'].items():
            if 'coords' in area_details and ('Assembly' in area_name or 'Packing' in area_name):
                (x0,y0), (x1,y1) = area_details['coords']
                shapes.append(go.layout.Shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=s_c, dash="dot", width=1), fillcolor="rgba(0,0,0,0)", layer="below"))
                annotations.append(dict(x=(x0+x1)/2, y=(y0+y1)/2, text=area_name, showarrow=False, font=dict(size=8, color=n_c), opacity=0.7))

    _apply_common_layout_settings(fig, f"Worker Distribution @ {selected_step*2} min", high_contrast, yaxis_title="Y Coordinate (m)", xaxis_title="X Coordinate (m)")
    fig.update_layout(shapes=shapes, annotations=annotations, legend_title_text='Zone')
    return fig

def plot_worker_density_heatmap(team_positions_df, facility_size, config, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    if team_positions_df.empty: return go.Figure().update_layout(title_text="No worker data for Heatmap", template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1F2937")
    facility_width, facility_height = facility_size
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)

    fig = go.Figure(go.Histogram2dContour(
        x=team_positions_df['x'], y=team_positions_df['y'],
        colorscale='Blues' if not high_contrast else 'Greys', reversescale=high_contrast,
        showscale=True, line=dict(width=0.5, color=n_c),
        contours=dict(coloring='heatmap', showlabels=False), # showlabels can be noisy
        xbins=dict(start=0, end=facility_width, size=facility_width/25), 
        ybins=dict(start=0, end=facility_height, size=facility_height/25),
        colorbar=dict(title='Density', len=0.7, y=0.5)
    ))
    # Add scatter for raw points if desired (can be too much)
    # fig.add_trace(go.Scatter(x=team_positions_df['x'], y=team_positions_df['y'], mode='markers', marker=dict(size=1, color=p_c, opacity=0.3), showlegend=False))

    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height, line=dict(color=n_c, width=1))]
    # Can add annotations for E/E or areas as in distribution plot if needed
    
    _apply_common_layout_settings(fig, "Aggregated Worker Density", high_contrast, yaxis_title="Y Coordinate (m)", xaxis_title="X Coordinate (m)")
    fig.update_layout(xaxis_range=[0, facility_width], yaxis_range=[0, facility_height], shapes=shapes)
    fig.update_xaxes(scaleanchor="y", scaleratio=1); fig.update_yaxes(constrain="domain")
    return fig

def plot_worker_wellbeing(scores, triggers, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    fig = go.Figure()
    x_vals = list(range(len(scores)))
    fig.add_trace(go.Scatter(x=x_vals, y=scores, mode='lines', name='Well-Being Index', line=dict(color=p_c, width=2)))
    
    trigger_colors = {'threshold': cr_c, 'trend': a_c, 'disruption': h_c}
    trigger_symbols = {'threshold': 'x-thin-open', 'trend': 'triangle-down-open', 'disruption': 'star-diamond-open'}
    
    for trigger_type, points in triggers.items():
        if isinstance(points, list) and points: # Basic triggers
            valid_points = [p for p in points if 0 <= p < len(scores)]
            if valid_points:
                fig.add_trace(go.Scatter(x=valid_points, y=[scores[p] for p in valid_points], mode='markers', 
                                         name=f'{trigger_type.capitalize()} Alert', 
                                         marker=dict(color=trigger_colors.get(trigger_type, n_c), size=9, symbol=trigger_symbols.get(trigger_type, 'circle'))))
        elif trigger_type == 'work_area' and isinstance(points, dict): # Work area triggers
            for i, (area, area_points) in enumerate(points.items()):
                if area_points:
                    valid_area_points = [p for p in area_points if 0 <= p < len(scores)]
                    if valid_area_points:
                         fig.add_trace(go.Scatter(x=valid_area_points, y=[scores[p] for p in valid_area_points], mode='markers', 
                                                 name=f'{area} Alert', marker=dict(color=px.colors.qualitative.Plotly[i%len(px.colors.qualitative.Plotly)], size=8, symbol='diamond-open')))

    _apply_common_layout_settings(fig, "Worker Well-Being Index", high_contrast, yaxis_title="Index (%)")
    fig.update_yaxes(range=[0, 105])
    return fig

def plot_psychological_safety(data, high_contrast=False):
    p_c, _, _, _, h_c, _ = _get_colors(high_contrast) # Using highlight color here
    fig = go.Figure()
    x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name='Psych. Safety', line=dict(color=h_c, width=2)))
    _apply_common_layout_settings(fig, "Psychological Safety Score", high_contrast, yaxis_title="Score (%)")
    fig.update_yaxes(range=[0, 105])
    return fig

def plot_downtime_trend(downtime_minutes, threshold, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c = _get_colors(high_contrast)
    fig = go.Figure()
    x_vals = list(range(len(downtime_minutes)))
    bar_colors = [cr_c if d > threshold else s_c for d in downtime_minutes] # Green if below/at threshold, Red if above
    
    fig.add_trace(go.Bar(x=x_vals, y=downtime_minutes, name='Downtime', marker_color=bar_colors, width=0.8))
    fig.add_hline(y=threshold, line=dict(color=a_c, width=1.5, dash="dash"), 
                  annotation_text=f"Threshold: {threshold} min", annotation_position="top right", 
                  annotation_font=dict(size=10, color=a_c))
    _apply_common_layout_settings(fig, "Downtime per Interval", high_contrast, yaxis_title="Downtime (minutes)")
    max_y = max(max(downtime_minutes) * 1.1 if downtime_minutes else threshold * 1.5, threshold * 1.5) # Ensure threshold is visible
    fig.update_yaxes(range=[0, max_y])
    return fig
