# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots 
import numpy as np
import pandas as pd

# NEW: ACCESSIBLE COLOR PALETTE for Light Theme & Colorblindness consideration
# References: ColorBrewer, Paul Tol's notes, IBM Design Language
# Using distinct hues, and ensuring enough brightness difference.

PLOTLY_TEMPLATE_LIGHT = "plotly_white" # Base light template

# Standard Theme Colors (Accessible for Light Background)
# Using a more vibrant and distinguishable set.
PRIMARY_COLOR_STD = "#0072B2"  # Blue (Tol's muted) / (Previously Indigo)
SECONDARY_COLOR_STD = "#009E73" # Bluish Green (Tol) / (Previously Cyan)
ALERT_COLOR_STD = "#E69F00"    # Orange/Amber (Tol)
CRITICAL_COLOR_STD = "#D55E00" # Vermillion/Reddish Orange (Tol)
HIGHLIGHT_COLOR_STD = "#56B4E9" # Sky Blue (Tol)
NEUTRAL_COLOR_STD = "#737373"  # Darker Gray for text/elements on light bg
POSITIVE_COLOR_STD = "#009E73" # Using Bluish Green for positive

# High Contrast Theme Colors (Light background, very distinct colors)
# For HC, we focus on maximum distinguishability.
PRIMARY_COLOR_HC = "#000000"   # Black text/lines
SECONDARY_COLOR_HC = "#E69F00" # Orange
ALERT_COLOR_HC = "#D55E00"    # Vermillion
CRITICAL_COLOR_HC = "#CC0000" # Darker, distinct Red
HIGHLIGHT_COLOR_HC = "#0072B2" # Blue
NEUTRAL_COLOR_HC = "#555555"  # Dark Gray
POSITIVE_COLOR_HC = "#00A000" # Dark Green

def _get_colors(high_contrast=False):
    if high_contrast: 
        return PRIMARY_COLOR_HC, SECONDARY_COLOR_HC, ALERT_COLOR_HC, CRITICAL_COLOR_HC, HIGHLIGHT_COLOR_HC, NEUTRAL_COLOR_HC, POSITIVE_COLOR_HC
    return PRIMARY_COLOR_STD, SECONDARY_COLOR_STD, ALERT_COLOR_STD, CRITICAL_COLOR_STD, HIGHLIGHT_COLOR_STD, NEUTRAL_COLOR_STD, POSITIVE_COLOR_STD

def _apply_common_layout_settings(fig, title_text, high_contrast=False, yaxis_title=None, xaxis_title="Time Step (Interval)", yaxis_range=None, show_legend=True):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    
    # Base colors for light theme
    paper_bg = "#F0F2F6" if not high_contrast else "#FFFFFF" # Off-white or pure white
    plot_bg = "#FFFFFF" if not high_contrast else "#F0F0F0"   # White or light gray
    font_color = "#262730" if not high_contrast else "#000000" # Dark gray or black text
    grid_color = "#D lijnEDEE" if not high_contrast else "#CCCCCC" # Lighter grid lines

    fig.update_layout(
        template=PLOTLY_TEMPLATE_LIGHT, # Use Plotly's light theme as a base
        title=dict(text=title_text, x=0.5, font=dict(size=18, color=font_color)), # Increased title size
        paper_bgcolor=paper_bg, 
        plot_bgcolor=plot_bg,
        font=dict(color=font_color, size=12), # Increased base font size
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.7)" if not high_contrast else "rgba(240,240,240,0.7)", 
            bordercolor=n_c, borderwidth=1, font_size=11,
            traceorder="normal"
        ) if show_legend else None,
        margin=dict(l=70, r=40, t=80, b=70), # Adjusted margins for titles
        xaxis=dict(
            title=xaxis_title, 
            gridcolor=grid_color, 
            zerolinecolor=n_c, 
            showline=True, linewidth=1.5, linecolor=n_c,
            rangemode='tozero' if xaxis_title and "Time" in xaxis_title else 'normal',
            titlefont=dict(size=13), tickfont=dict(size=11)
        ),
        yaxis=dict(
            title=yaxis_title, 
            gridcolor=grid_color, 
            zerolinecolor=n_c, 
            showline=True, linewidth=1.5, linecolor=n_c,
            range=yaxis_range,
            titlefont=dict(size=13), tickfont=dict(size=11)
        ),
        hovermode="x unified", 
        dragmode='pan' 
    )

def plot_key_metrics_summary(compliance, proximity, wellbeing, downtime, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); figs = []
    font_color = "#262730" if not high_contrast else "#000000"
    gauge_bgcolor = "#E9ECEF" if not high_contrast else "#DBDBDB" # Lighter gauge background
    
    metrics_config = [
        ("Task Compliance", compliance, 75, p_c, pos_c, cr_c, "%", False), 
        ("Collaboration Idx", proximity, 60, s_c, pos_c, a_c, "%", False), 
        ("Well-Being Idx", wellbeing, 70, h_c, pos_c, cr_c, "%", False), 
        ("Total Downtime", downtime, 30, a_c, cr_c, pos_c, " min", True)
    ]
    for title, value, target, bar_color, pos_delta_c, neg_delta_c, suffix, lower_is_better in metrics_config:
        fig = go.Figure(); 
        delta_increasing_color = pos_delta_c if not lower_is_better else neg_delta_c
        delta_decreasing_color = neg_delta_c if not lower_is_better else pos_delta_c
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta", value=float(value),
            delta={'reference': float(target), 'increasing': {'color': delta_increasing_color}, 'decreasing': {'color': delta_decreasing_color}, 'font': {'size': 14}},
            title={'text': title, 'font': {'size': 14, 'color': font_color}}, 
            number={'suffix': suffix, 'font': {'size': 22, 'color': font_color}},
            gauge={
                'axis': {'range': [0, 100 if suffix == "%" else max(target * 1.5, value * 1.2, 10)], 'tickwidth': 1, 'tickcolor': n_c}, 
                'bar': {'color': bar_color, 'thickness': 0.65}, 
                'bgcolor': gauge_bgcolor, 
                'borderwidth': 1, 'bordercolor': n_c,
                'steps': [
                    {'range': [0, target * (0.8 if not lower_is_better else 1.2) ], 'color': cr_c},
                    {'range': [target * (0.8 if not lower_is_better else 1.2), target * (1.0 if not lower_is_better else target*100)], 'color': a_c}, # if lower_is_better, good range step might go way above
                    # Added a "good" step for metrics where higher is better
                    {'range': [target, 100 if suffix == "%" else max(target*1.5, value*1.2, 10)], 'color': pos_c if not lower_is_better else cr_c } # For high end
                ],
                'threshold': {'line': {'color': font_color if high_contrast else NEUTRAL_COLOR_STD, 'width': 3}, 'thickness': 0.8, 'value': target}
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color))
        figs.append(fig)
    return figs

# In subsequent plot functions, use `_apply_common_layout_settings` and ensure specific elements (lines, markers)
# also respect `high_contrast` by using the `_get_colors` output.

def plot_task_compliance_score(data, disruption_points, forecast_data=None, z_scores=None, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Compliance', line=dict(color=p_c, width=2.5), marker=dict(size=6, symbol="circle"), hovertemplate='Compliance: %{y:.1f}%<extra></extra>'))
    if forecast_data and len(forecast_data)==len(data): fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast', line=dict(color=h_c, dash='dashdot', width=2), hovertemplate='Forecast: %{y:.1f}%<extra></extra>'))
    for dp_step in disruption_points: 
        if 0 <= dp_step < len(data): fig.add_vline(x=dp_step, line=dict(color=a_c, width=1.5, dash="longdash"), annotation_text="D", annotation_position="top left", annotation=dict(font_size=10, bgcolor="rgba(255,255,255,0.7)" if not high_contrast else "rgba(0,0,0,0.7)", borderpad=2, textangle=-45, font_color = n_c if not high_contrast else p_c)) 
    _apply_common_layout_settings(fig, "Task Compliance Score Trend", high_contrast, yaxis_title="Score (%)", yaxis_range=[max(0, (min(data) if data else 0)-10), min(105, (max(data) if data else 100)+10)]); return fig

def plot_collaboration_proximity_index(data, disruption_points, forecast_data=None, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Collab. Index', line=dict(color=s_c, width=2.5), marker=dict(size=6, symbol="diamond"), hovertemplate='Collab. Index: %{y:.1f}%<extra></extra>'))
    if forecast_data and len(forecast_data)==len(data): fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast', line=dict(color=h_c, dash='dashdot', width=2), hovertemplate='Forecast: %{y:.1f}%<extra></extra>'))
    for dp_step in disruption_points:
        if 0 <= dp_step < len(data): fig.add_vline(x=dp_step, line=dict(color=a_c, width=1.5, dash="longdash"), annotation_text="D", annotation_position="bottom right", annotation=dict(font_size=10, bgcolor="rgba(255,255,255,0.7)" if not high_contrast else "rgba(0,0,0,0.7)", borderpad=2, textangle=-45, font_color = n_c if not high_contrast else p_c))
    _apply_common_layout_settings(fig, "Collaboration Proximity Index Trend", high_contrast, yaxis_title="Index (%)", yaxis_range=[max(0, (min(data) if data else 0)-10), min(105, (max(data) if data else 100)+10)]); return fig

def plot_operational_recovery(recovery_data, productivity_loss_data, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(recovery_data)))
    fig.add_trace(go.Scatter(x=x_vals, y=recovery_data, mode='lines', name='Op. Recovery', line=dict(color=pos_c, width=2), hovertemplate='Recovery: %{y:.1f}%<extra></extra>'))
    if productivity_loss_data and len(productivity_loss_data)==len(recovery_data): fig.add_trace(go.Scatter(x=x_vals, y=productivity_loss_data, mode='lines', name='Prod. Loss', line=dict(color=cr_c, dash='dot', width=2), hovertemplate='Prod. Loss: %{y:.1f}%<extra></extra>'))
    _apply_common_layout_settings(fig, "Operational Resilience: Recovery vs. Loss", high_contrast, yaxis_title="Percentage (%)", yaxis_range=[0, 105]); return fig

def plot_operational_efficiency(efficiency_df, selected_metrics, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure()
    colors = {'uptime': p_c, 'throughput': s_c, 'quality': h_c, 'oee': pos_c}; line_styles = {'uptime': 'solid', 'throughput': 'solid', 'quality': 'solid', 'oee': 'solid'}; line_widths = {'uptime': 1.5, 'throughput': 1.5, 'quality': 1.5, 'oee': 2.5}
    for metric in selected_metrics:
        if metric in efficiency_df.columns: fig.add_trace(go.Scatter(x=efficiency_df.index, y=efficiency_df[metric], mode='lines', name=metric.upper(), line=dict(color=colors.get(metric, n_c), width=line_widths.get(metric, 1.5), dash=line_styles.get(metric, 'solid')), hovertemplate=f'{metric.upper()}: %{{y:.1f}}%<extra></extra>'))
    _apply_common_layout_settings(fig, "Overall Equipment Effectiveness (OEE) & Components", high_contrast, yaxis_title="Efficiency Score (%)", yaxis_range=[0, 105]); return fig

def plot_worker_distribution(team_positions_df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); df_step = team_positions_df[team_positions_df['step'] == selected_step].copy() 
    if df_step.empty: return go.Figure().update_layout(title_text=f"No worker data for Step {selected_step}", template=PLOTLY_TEMPLATE_LIGHT, paper_bgcolor="#F0F2F6", plot_bgcolor="#FFFFFF")
    facility_width, facility_height = facility_size; zone_names = df_step['zone'].unique(); 
    # More distinct colors for zones
    color_map_base = px.colors.qualitative.Vivid if not high_contrast else px.colors.qualitative.Safe
    zone_color_map = {zone: color_map_base[i % len(color_map_base)] for i, zone in enumerate(zone_names)}
    df_step['color'] = df_step['zone'].map(zone_color_map)

    status_symbols = {"working": "circle", "idle": "square", "break":"diamond", "fatigued": "cross", "exhausted": "x", "disrupted":"hourglass"}
    df_step['symbol'] = df_step['status'].map(status_symbols).fillna("circle")


    if use_3d:
        fig = px.scatter_3d(df_step, x='x', y='y', z='z', color='zone', color_discrete_map=zone_color_map, hover_name='worker_id', hover_data={'zone': True, 'status': True, 'x':':.1f', 'y':':.1f', 'z':':.1f'}, range_x=[0, facility_width], range_y=[0, facility_height], range_z=[0,max(5, df_step['z'].max() if 'z' in df_step.columns and not df_step['z'].empty else 5)], symbol='symbol', opacity=0.9, size_max=12)
        fig.update_scenes(aspectmode='data', xaxis_showgrid=True, yaxis_showgrid=True, zaxis_showgrid=True, xaxis_backgroundcolor="rgba(0,0,0,0)", yaxis_backgroundcolor="rgba(0,0,0,0)", zaxis_backgroundcolor="rgba(0,0,0,0)")
    else:
        fig = px.scatter(df_step, x='x', y='y', color='zone', color_discrete_map=zone_color_map, hover_name='worker_id', hover_data={'zone': True, 'status': True, 'x':':.1f', 'y':':.1f'}, range_x=[-5, facility_width + 5], range_y=[-5, facility_height + 5], symbol='symbol', opacity=0.9, size_max=12) 
        fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True, xaxis_zeroline=True, yaxis_zeroline=True) # Show grid for 2D as well
    
    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height, line=dict(color=n_c, width=2), fillcolor="rgba(0,0,0,0)", layer="below")]; annotations = []
    if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
        for point in config['ENTRY_EXIT_POINTS']: 
            shapes.append(go.layout.Shape(type="circle", x0=point['coords'][0]-2, y0=point['coords'][1]-2, x1=point['coords'][0]+2, y1=point['coords'][1]+2, fillcolor=a_c, line_color=p_c if high_contrast else "black", line_width=1, opacity=0.9, layer="above"))
            annotations.append(dict(x=point['coords'][0], y=point['coords'][1]+5, text=point['name'][:2].upper(), showarrow=False, font=dict(size=10, color=p_c if high_contrast else font_color), borderpad=2, bgcolor="rgba(255,255,255,0.5)"))
    if show_prod_lines and 'WORK_AREAS' in config:
         for area_name, area_details in config['WORK_AREAS'].items():
            if 'coords' in area_details and ('Assembly' in area_name or 'Packing' in area_name or 'Quality' in area_name or 'Warehouse' in area_name): 
                (x0,y0), (x1,y1) = area_details['coords']; shapes.append(go.layout.Shape(type="rect", x0=min(x0,x1), y0=min(y0,y1), x1=max(x0,x1), y1=max(y0,y1), line=dict(color=s_c, dash="solid", width=1.5), fillcolor="rgba(34,211,238,0.1)", layer="below")); annotations.append(dict(x=(x0+x1)/2, y=min(y0,y1)-4, text=area_name, showarrow=False, font=dict(size=9, color=n_c), opacity=0.9, yanchor="top")) 
    _apply_common_layout_settings(fig, f"Worker Spatial Distribution (Time: {selected_step*2} min)", high_contrast, yaxis_title="Y (m)", xaxis_title="X (m)", show_legend=True); fig.update_layout(shapes=shapes, annotations=annotations, legend_title_text='Zone / Status'); return fig

def plot_worker_density_heatmap(team_positions_df, facility_size, config, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    if team_positions_df.empty: 
        fig = go.Figure(); _apply_common_layout_settings(fig, "Aggregated Worker Density (No Data)", high_contrast)
        fig.add_annotation(text="No worker data available for heatmap in the selected range.", showarrow=False, font=dict(size=14)); return fig
    facility_width, facility_height = facility_size; p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    
    # Using Histogram2d for clearer density representation without map context
    fig = go.Figure(go.Histogram2d(
        x=team_positions_df['x'], 
        y=team_positions_df['y'],
        colorscale='Viridis' if not high_contrast else 'Greys', # Viridis is perceptually uniform
        reversescale=high_contrast, # If Greys, light is high density, reverse for dark bg
        showscale=True, 
        xbins=dict(start=0, end=facility_width, size=facility_width/25), # Finer bins for more detail
        ybins=dict(start=0, end=facility_height, size=facility_height/25),
        colorbar=dict(title='Worker Density', thickness=15, len=0.75, y=0.5, tickfont_size=10, x=1.02)
    ))
    
    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height, line=dict(color=n_c, width=1.5), layer="below")]; annotations = []
    if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
        for point in config['ENTRY_EXIT_POINTS']: shapes.append(go.layout.Shape(type="circle", x0=point['coords'][0]-1, y0=point['coords'][1]-1, x1=point['coords'][0]+1, y1=point['coords'][1]+1, fillcolor=a_c, line_color=NEUTRAL_COLOR_HC if high_contrast else "black", opacity=0.6, layer="above"))
    if show_prod_lines and 'WORK_AREAS' in config:
         for area_name, area_details in config['WORK_AREAS'].items():
            if 'coords' in area_details and ('Assembly' in area_name or 'Packing' in area_name or 'Quality' in area_name or 'Warehouse' in area_name):
                (x0,y0), (x1,y1) = area_details['coords']; shapes.append(go.layout.Shape(type="rect", x0=min(x0,x1), y0=min(y0,y1), x1=max(x0,x1), y1=max(y0,y1), line=dict(color=s_c, dash="dot", width=1), fillcolor="rgba(0,0,0,0)", layer="above"))

    _apply_common_layout_settings(fig, "Aggregated Worker Density Heatmap", high_contrast, yaxis_title="Y Coordinate (m)", xaxis_title="X Coordinate (m)"); fig.update_layout(xaxis_range=[0, facility_width], yaxis_range=[0, facility_height], shapes=shapes, annotations=annotations, autosize=True); fig.update_xaxes(constrain="domain", scaleanchor="y", scaleratio=1); fig.update_yaxes(constrain="domain"); return fig

def plot_worker_wellbeing(scores, triggers, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(scores)))
    fig.add_trace(go.Scatter(x=x_vals, y=scores, mode='lines', name='Well-Being Index', line=dict(color=p_c, width=2), hovertemplate='Well-Being: %{y:.1f}%<extra></extra>'))
    avg_wellbeing = np.mean(scores) if scores else None
    if avg_wellbeing is not None: fig.add_hline(y=avg_wellbeing, line=dict(color=n_c, width=1, dash="dot"), annotation_text=f"Avg: {avg_wellbeing:.1f}%", annotation_position="bottom left", annotation_font_size=10)
    trigger_colors = {'threshold': cr_c, 'trend': a_c, 'disruption': h_c, 'work_area_general': s_c}; trigger_symbols = {'threshold': 'x', 'trend': 'triangle-down-open', 'disruption': 'star-open', 'work_area_general': 'diamond-open'} # Open symbols for better visibility
    for trigger_type, points in triggers.items():
        flat_points = []; processed_trigger_type = trigger_type
        if isinstance(points, list): flat_points = points
        elif isinstance(points, dict) and trigger_type == 'work_area': all_wa_points = set(); [all_wa_points.update(p_list) for p_list in points.values() if isinstance(p_list, list)]; flat_points = list(all_wa_points); processed_trigger_type = 'work_area_general'
        valid_points = sorted(list(set(p for p in flat_points if isinstance(p, (int, float)) and 0 <= p < len(scores))))
        if valid_points: fig.add_trace(go.Scatter(x=valid_points, y=[scores[p] for p in valid_points], mode='markers', name=f'{processed_trigger_type.replace("_", " ").title()}', marker=dict(color=trigger_colors.get(processed_trigger_type, n_c), size=10, symbol=trigger_symbols.get(processed_trigger_type, 'circle-open'), line=dict(width=1.5, color=p_c if high_contrast else "white" )), hovertemplate=f'{processed_trigger_type.replace("_", " ").title()}: %{{y:.1f}}% at Step %{{x}}<extra></extra>'))
    _apply_common_layout_settings(fig, "Worker Well-Being Index Trend", high_contrast, yaxis_title="Index (%)", yaxis_range=[0, 105]); return fig

def plot_psychological_safety(data, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name='Psych. Safety', line=dict(color=h_c, width=2), hovertemplate='Psych. Safety: %{y:.1f}%<extra></extra>'))
    avg_safety = np.mean(data) if data else None
    if avg_safety is not None: fig.add_hline(y=avg_safety, line=dict(color=n_c, width=1, dash="dot"), annotation_text=f"Avg: {avg_safety:.1f}%", annotation_position="bottom left", annotation_font_size=10)
    _apply_common_layout_settings(fig, "Psychological Safety Score Trend", high_contrast, yaxis_title="Score (%)", yaxis_range=[0, 105]); return fig

def plot_downtime_trend(downtime_events_list, interval_threshold, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); 
    downtime_durations = [event.get('duration', 0) for event in downtime_events_list]; x_vals = list(range(len(downtime_durations)))
    bar_colors = [cr_c if d > interval_threshold else pos_c for d in downtime_durations] 
    hover_texts = [f"Duration: {event.get('duration', 0):.1f} min<br>Cause: {event.get('cause', 'Unknown')}" for event in downtime_events_list]
    fig.add_trace(go.Bar(x=x_vals, y=downtime_durations, name='Downtime', marker_color=bar_colors, width=0.7, text=hover_texts, hoverinfo='text'))
    fig.add_hline(y=interval_threshold, line=dict(color=a_c, width=1.5, dash="longdash"), annotation_text=f"Alert Threshold: {interval_threshold} min", annotation_position="top right", annotation=dict(font_size=10, bgcolor="rgba(255,255,255,0.7)" if not high_contrast else "rgba(0,0,0,0.7)", borderpad=2, font_color = n_c if not high_contrast else p_c ))
    _apply_common_layout_settings(fig, "Downtime per Interval", high_contrast, yaxis_title="Downtime (minutes)"); max_y_val = max(max(downtime_durations) * 1.15 if downtime_durations else interval_threshold * 1.5, interval_threshold * 1.5, 10); fig.update_yaxes(range=[0, max_y_val]); return fig

def plot_team_cohesion(data, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name='Team Cohesion', line=dict(color=s_c, width=2), hovertemplate='Cohesion: %{y:.1f}%<extra></extra>'))
    avg_cohesion = np.mean(data) if data else None
    if avg_cohesion is not None: fig.add_hline(y=avg_cohesion, line=dict(color=n_c, width=1, dash="dot"), annotation_text=f"Avg: {avg_cohesion:.1f}%", annotation_position="top left", annotation_font_size=10)
    _apply_common_layout_settings(fig, "Team Cohesion Index Trend", high_contrast, yaxis_title="Cohesion Index (%)", yaxis_range=[0, 105]); return fig

def plot_perceived_workload(data, high_workload_threshold, very_high_workload_threshold, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(data)))
    line_color = n_c # Neutral line for workload trend
    marker_colors = [cr_c if val >= very_high_workload_threshold else a_c if val >= high_workload_threshold else pos_c for val in data] # Green for low/normal workload
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Perceived Workload', line=dict(color=line_color, width=2), marker=dict(size=6, color=marker_colors, symbol='circle'), hovertemplate='Workload: %{y:.1f}/10<extra></extra>'))
    fig.add_hline(y=high_workload_threshold, line=dict(color=a_c, width=1.5, dash="dash"), annotation_text=f"High Load ({high_workload_threshold})", annotation_position="bottom right", annotation_font=dict(color=a_c, size=10))
    fig.add_hline(y=very_high_workload_threshold, line=dict(color=cr_c, width=1.5, dash="dash"), annotation_text=f"Very High ({very_high_workload_threshold})", annotation_position="top right", annotation_font=dict(color=cr_c, size=10))
    _apply_common_layout_settings(fig, "Perceived Workload Index (0-10 Scale)", high_contrast, yaxis_title="Workload Index", yaxis_range=[0, 10.5]); return fig

def plot_downtime_causes_pie(downtime_events_list, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    causes_summary = {}; total_downtime_duration_for_pie = 0 
    for event in downtime_events_list:
        duration = event.get('duration', 0); cause = event.get('cause', 'Unknown')
        if duration > 0 and cause != "None" and cause != "Unknown": 
            # Consolidate "Equip.Fail" from combined causes
            if "Equip.Fail" in cause and cause != "Equipment Failure": cause = "Equipment Failure"
            if "HumanError" in cause and cause != "Human Error": cause = "Human Error"
            causes_summary[cause] = causes_summary.get(cause, 0) + duration
            total_downtime_duration_for_pie += duration
    if not causes_summary:
        fig = go.Figure(); _apply_common_layout_settings(fig, "Downtime by Cause (No Categorized Downtime)", high_contrast, show_legend=False)
        fig.add_annotation(text="No categorized downtime events in selected period.", showarrow=False, font_size=12); return fig
    labels = list(causes_summary.keys()); values = list(causes_summary.values()); num_causes = len(labels)
    color_sequence = px.colors.qualitative.Set3 if not high_contrast else px.colors.qualitative.T10 # Different palettes for visibility
    pie_colors = [color_sequence[i % len(color_sequence)] for i in range(num_causes)]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, pull=[0.02]*num_causes, marker_colors=pie_colors, textinfo='label+percent', insidetextorientation='auto', hovertemplate="<b>Cause:</b> %{label}<br><b>Duration:</b> %{value:.1f} min<br><b>Share:</b> %{percent}<extra></extra>", sort=True, direction='descending')]) # Sort by value for clarity
    _apply_common_layout_settings(fig, f"Downtime Distribution by Cause (Total: {total_downtime_duration_for_pie:.0f} min)", high_contrast, show_legend=False) # Legend can be redundant with pie labels
    fig.update_traces(textfont_size=11) # Make text on pie smaller if many segments
    return fig
