# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Consistent color scheme definitions
PLOTLY_TEMPLATE = "plotly_dark" # Base template
# Standard Theme Colors
PRIMARY_COLOR_STD = "#6366F1"  # Indigo-500 (Main data series)
SECONDARY_COLOR_STD = "#22D3EE" # Cyan-400 (Secondary data series, comparisons)
ALERT_COLOR_STD = "#FACC15"    # Yellow-400 (Warnings, thresholds)
CRITICAL_COLOR_STD = "#F87171" # Red-400 (Critical issues, failures)
HIGHLIGHT_COLOR_STD = "#A78BFA" # Violet-400 (Forecasts, highlights)
NEUTRAL_COLOR_STD = "#9CA3AF"  # Gray-400 (Gridlines, less important text)
POSITIVE_COLOR_STD = "#34D399" # Green-400 (Good performance, above target)

# High Contrast Theme Colors
PRIMARY_COLOR_HC = "#FFFFFF"   # White
SECONDARY_COLOR_HC = "#FFFF00" # Yellow
ALERT_COLOR_HC = "#FFBF00"    # Amber
CRITICAL_COLOR_HC = "#FF0000" # Red
HIGHLIGHT_COLOR_HC = "#00FFFF" # Cyan
NEUTRAL_COLOR_HC = "#BEBEBE"  # Gray
POSITIVE_COLOR_HC = "#00FF00" # Bright Green

def _get_colors(high_contrast=False):
    if high_contrast:
        return PRIMARY_COLOR_HC, SECONDARY_COLOR_HC, ALERT_COLOR_HC, CRITICAL_COLOR_HC, HIGHLIGHT_COLOR_HC, NEUTRAL_COLOR_HC, POSITIVE_COLOR_HC
    return PRIMARY_COLOR_STD, SECONDARY_COLOR_STD, ALERT_COLOR_STD, CRITICAL_COLOR_STD, HIGHLIGHT_COLOR_STD, NEUTRAL_COLOR_STD, POSITIVE_COLOR_STD

def _apply_common_layout_settings(fig, title_text, high_contrast=False, yaxis_title=None, xaxis_title="Time Step (Interval)", yaxis_range=None, show_legend=True):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title_text, x=0.5, font=dict(size=16, color=p_c if high_contrast else "#EAEAEA")),
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="#1F2937" if not high_contrast else "#101010", # Darker for HC plot bg
        font=dict(color=n_c if high_contrast else "#D1D5DB", size=11), 
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(31, 41, 55, 0.5)" if not high_contrast else "rgba(0,0,0,0.5)", # Semi-transparent legend BG
            bordercolor=n_c, borderwidth=0.5, font_size=10,
            traceorder="normal" # Keep legend order same as trace add order
        ) if show_legend else None,
        margin=dict(l=60, r=40, t=70, b=60), # Increased top margin for title
        xaxis=dict(
            title=xaxis_title, 
            gridcolor=n_c if high_contrast else "#2b3648", # Subtler grid
            zerolinecolor=n_c if high_contrast else "#4A5568", 
            showline=True, linewidth=1, linecolor=n_c if high_contrast else "#4A5568",
            rangemode='tozero' if xaxis_title and "Time" in xaxis_title else 'normal' # Ensure x-axis starts at 0 for time
        ),
        yaxis=dict(
            title=yaxis_title, 
            gridcolor=n_c if high_contrast else "#2b3648", 
            zerolinecolor=n_c if high_contrast else "#4A5568", 
            showline=True, linewidth=1, linecolor=n_c if high_contrast else "#4A5568",
            range=yaxis_range
        ),
        hovermode="x unified",
        dragmode='pan' # Default to pan for easier navigation on touch devices
    )

def plot_key_metrics_summary(compliance, proximity, wellbeing, downtime, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    metrics_config = [
        ("Task Compliance", compliance, 75, p_c, pos_c, cr_c, "%", False), # value, target, good_color, positive_delta_color, negative_delta_color, suffix, lower_is_better
        ("Collaboration Idx", proximity, 60, s_c, pos_c, a_c, "%", False),
        ("Well-Being Idx", wellbeing, 70, h_c, pos_c, cr_c, "%", False),
        ("Total Downtime", downtime, 30, a_c, cr_c, pos_c, " min", True) # For downtime, high delta is bad (cr_c), low delta is good (pos_c)
    ]
    figs = []

    for title, value, target, bar_color, pos_delta_c, neg_delta_c, suffix, lower_is_better in metrics_config:
        fig = go.Figure()
        delta_increasing_color = pos_delta_c if not lower_is_better else neg_delta_c
        delta_decreasing_color = neg_delta_c if not lower_is_better else pos_delta_c
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=float(value),
            delta={'reference': float(target), 'increasing': {'color': delta_increasing_color}, 'decreasing': {'color': delta_decreasing_color}, 'font': {'size': 12}},
            title={'text': title, 'font': {'size': 12, 'color': n_c}},
            number={'suffix': suffix, 'font': {'size': 20, 'color': p_c if high_contrast else "#FFFFFF"}},
            gauge={
                'axis': {'range': [0, 100 if suffix == "%" else max(target * 1.5, value * 1.2, 10)], 'tickwidth': 1, 'tickcolor': n_c},
                'bar': {'color': bar_color, 'thickness': 0.7},
                'bgcolor': "#2a3447" if not high_contrast else "#222222",
                'borderwidth': 0.5, 'bordercolor': n_c,
                'steps': [ # Define poor, fair, good regions visually
                    {'range': [0, target * (0.8 if not lower_is_better else 1.2)], 'color': cr_c if not lower_is_better else pos_c }, # Red / Green
                    {'range': [target * (0.8 if not lower_is_better else 1.2), target], 'color': a_c }, # Amber
                ],
                'threshold': {'line': {'color': p_c if high_contrast else "white", 'width': 2.5}, 'thickness': 0.8, 'value': target}
            }
        ))
        fig.update_layout(height=180, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=p_c if high_contrast else "#D1D5DB"))
        figs.append(fig)
    return figs

def plot_task_compliance_score(data, disruption_points, forecast_data=None, z_scores=None, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    fig = go.Figure()
    x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Task Compliance', line=dict(color=p_c, width=2.5), marker=dict(size=5), hovertemplate='Compliance: %{y:.1f}%<extra></extra>'))
    if forecast_data and len(forecast_data)==len(data): fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast', line=dict(color=h_c, dash='dashdot', width=1.5), hovertemplate='Forecast: %{y:.1f}%<extra></extra>'))
    
    # Add Z-scores as a secondary y-axis if available and sensible
    # if z_scores and len(z_scores) == len(data):
    #     fig.add_trace(go.Scatter(x=x_vals, y=z_scores, name='Z-Score', yaxis='y2', line=dict(color=n_c, width=1, dash='solid'), opacity=0.6, hovertemplate='Z-Score: %{y:.2f}<extra></extra>'))
    #     fig.update_layout(yaxis2=dict(title='Z-Score', overlaying='y', side='right', showgrid=False, range=[-3,3]))


    for dp_step in disruption_points: # Use unique name
        if 0 <= dp_step < len(data): 
            fig.add_vline(x=dp_step, line=dict(color=a_c, width=1.5, dash="longdash"), 
                          annotation_text="Disruption", annotation_position="top left", 
                          annotation=dict(font_size=10, bgcolor="rgba(250,204,21,0.7)", borderpad=2, textangle=0)) # Custom annotation
    _apply_common_layout_settings(fig, "Task Compliance Score Trend", high_contrast, yaxis_title="Compliance Score (%)", yaxis_range=[max(0, min(data)-15 if data else 0), min(105, max(data)+15 if data else 100)])
    return fig

def plot_collaboration_proximity_index(data, disruption_points, forecast_data=None, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    fig = go.Figure(); x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Collab. Index', line=dict(color=s_c, width=2.5), marker=dict(size=5), hovertemplate='Collab. Index: %{y:.1f}%<extra></extra>'))
    if forecast_data and len(forecast_data)==len(data): fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast', line=dict(color=h_c, dash='dashdot', width=1.5), hovertemplate='Forecast: %{y:.1f}%<extra></extra>'))
    for dp_step in disruption_points:
        if 0 <= dp_step < len(data): fig.add_vline(x=dp_step, line=dict(color=a_c, width=1.5, dash="longdash"), annotation_text="Disruption", annotation_position="bottom right", annotation=dict(font_size=10, bgcolor="rgba(250,204,21,0.7)", borderpad=2))
    _apply_common_layout_settings(fig, "Collaboration Proximity Index Trend", high_contrast, yaxis_title="Proximity Index (%)", yaxis_range=[max(0, min(data)-15 if data else 0), min(105, max(data)+15 if data else 100)])
    return fig

def plot_operational_recovery(recovery_data, productivity_loss_data, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    fig = go.Figure(); x_vals = list(range(len(recovery_data)))
    fig.add_trace(go.Scatter(x=x_vals, y=recovery_data, mode='lines', name='Operational Recovery', line=dict(color=pos_c, width=2), hovertemplate='Recovery: %{y:.1f}%<extra></extra>'))
    if productivity_loss_data and len(productivity_loss_data)==len(recovery_data): fig.add_trace(go.Scatter(x=x_vals, y=productivity_loss_data, mode='lines', name='Productivity Loss', line=dict(color=cr_c, dash='dot', width=1.5), hovertemplate='Prod. Loss: %{y:.1f}%<extra></extra>'))
    _apply_common_layout_settings(fig, "Operational Resilience: Recovery vs. Loss", high_contrast, yaxis_title="Percentage (%)", yaxis_range=[0, 105])
    return fig

def plot_operational_efficiency(efficiency_df, selected_metrics, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    fig = go.Figure()
    # Define distinct colors for OEE components
    colors = {'uptime': p_c, 'throughput': s_c, 'quality': h_c, 'oee': pos_c if not high_contrast else p_c} # OEE in green or white
    line_styles = {'uptime': 'solid', 'throughput': 'solid', 'quality': 'solid', 'oee': 'dashdot'}
    line_widths = {'uptime': 1.5, 'throughput': 1.5, 'quality': 1.5, 'oee': 2.5} # Make OEE line thicker

    for metric in selected_metrics:
        if metric in efficiency_df.columns:
            fig.add_trace(go.Scatter(x=efficiency_df.index, y=efficiency_df[metric], mode='lines', name=metric.upper(), 
                                     line=dict(color=colors.get(metric, n_c), width=line_widths.get(metric, 1.5), dash=line_styles.get(metric, 'solid')),
                                     hovertemplate=f'{metric.upper()}: %{{y:.1f}}%<extra></extra>'))
    _apply_common_layout_settings(fig, "Overall Equipment Effectiveness (OEE) & Components", high_contrast, yaxis_title="Efficiency Score (%)", yaxis_range=[0, 105])
    return fig

def plot_worker_distribution(team_positions_df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    df_step = team_positions_df[team_positions_df['step'] == selected_step].copy() # Use .copy() to avoid SettingWithCopyWarning
    if df_step.empty: return go.Figure().update_layout(title_text=f"No worker data for Step {selected_step}", template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1F2937" if not high_contrast else "#111111")

    facility_width, facility_height = facility_size
    # Create a categorical color map for zones for consistency
    zone_names = df_step['zone'].unique()
    color_map_base = px.colors.qualitative.Plotly if not high_contrast else px.colors.qualitative.Bold
    zone_color_map = {zone: color_map_base[i % len(color_map_base)] for i, zone in enumerate(zone_names)}
    
    df_step['color'] = df_step['zone'].map(zone_color_map)

    if use_3d:
        fig = px.scatter_3d(df_step, x='x', y='y', z='z', color='zone', color_discrete_map=zone_color_map,
                            hover_name='worker_id', hover_data={'zone': True, 'status': True, 'x':':.1f', 'y':':.1f', 'z':':.1f'},
                            range_x=[0, facility_width], range_y=[0, facility_height], range_z=[0,max(5, df_step['z'].max() if 'z' in df_step.columns and not df_step['z'].empty else 5)],
                            symbol='status', opacity=0.8)
        fig.update_scenes(aspectmode='cube', xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False, xaxis_backgroundcolor="rgba(0,0,0,0)", yaxis_backgroundcolor="rgba(0,0,0,0)", zaxis_backgroundcolor="rgba(0,0,0,0)")
    else:
        fig = px.scatter(df_step, x='x', y='y', color='zone', color_discrete_map=zone_color_map,
                         hover_name='worker_id', hover_data={'zone': True, 'status': True, 'x':':.1f', 'y':':.1f'},
                         range_x=[-5, facility_width + 5], range_y=[-5, facility_height + 5],
                         symbol='status', opacity=0.8, size_max=10) # Ensure markers are not too small
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False)

    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height, line=dict(color=n_c, width=1.5), fillcolor="rgba(0,0,0,0)", layer="below")]
    annotations = []
    if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
        for point in config['ENTRY_EXIT_POINTS']:
            shapes.append(go.layout.Shape(type="circle", x0=point['coords'][0]-1.5, y0=point['coords'][1]-1.5, x1=point['coords'][0]+1.5, y1=point['coords'][1]+1.5, fillcolor=a_c, line_color=p_c if high_contrast else "white", opacity=0.8, layer="above"))
            annotations.append(dict(x=point['coords'][0], y=point['coords'][1]+4, text=point['name'][:1].upper(), showarrow=False, font=dict(size=9, color=p_c if high_contrast else "#FFFFFF"), borderpad=1, bgcolor="rgba(0,0,0,0.3)"))
    if show_prod_lines and 'WORK_AREAS' in config: # Renamed from show_production_lines
         for area_name, area_details in config['WORK_AREAS'].items():
            if 'coords' in area_details:
                (x0,y0), (x1,y1) = area_details['coords']
                shapes.append(go.layout.Shape(type="rect", x0=min(x0,x1), y0=min(y0,y1), x1=max(x0,x1), y1=max(y0,y1), line=dict(color=s_c, dash="dashdot", width=1), fillcolor="rgba(34,211,238,0.05)", layer="below")) # Slight fill for areas
                annotations.append(dict(x=(x0+x1)/2, y=min(y0,y1)-3, text=area_name, showarrow=False, font=dict(size=8, color=n_c), opacity=0.8, yanchor="top"))
    _apply_common_layout_settings(fig, f"Worker Spatial Distribution (Time: {selected_step*2} min)", high_contrast, yaxis_title="Y (m)", xaxis_title="X (m)", show_legend=True)
    fig.update_layout(shapes=shapes, annotations=annotations, legend_title_text='Zone / Status')
    return fig

def plot_worker_density_heatmap(team_positions_df, facility_size, config, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    if team_positions_df.empty: return go.Figure().update_layout(title_text="No worker data for Heatmap", template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1F2937" if not high_contrast else "#111111")
    facility_width, facility_height = facility_size; p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast)
    fig = go.Figure(go.Densitymapbox(lat=team_positions_df['y'], lon=team_positions_df['x'], z=np.ones(len(team_positions_df)), # Using Mapbox for better heatmap visualization
                                     radius=10, # Adjust radius for desired smoothing
                                     colorscale='Viridis' if not high_contrast else 'Hot', 
                                     showscale=True, colorbar=dict(title='Density',thickness=10, len=0.6)))
    # If not using Mapbox, Histogram2dContour as before:
    # fig = go.Figure(go.Histogram2dContour(x=team_positions_df['x'], y=team_positions_df['y'], colorscale='Blues' if not high_contrast else 'Greys', reversescale=high_contrast, showscale=True, line=dict(width=0.5, color=n_c), contours=dict(coloring='heatmap', showlabels=False), xbins=dict(start=0, end=facility_width, size=facility_width/25), ybins=dict(start=0, end=facility_height, size=facility_height/25), colorbar=dict(title='Density', len=0.7, y=0.5, thickness=10)))
    
    fig.update_layout(mapbox_style="carto-darkmatter" if not high_contrast else "carto-positron", # Choose appropriate mapbox style
                      mapbox_center_lon=facility_width/2, mapbox_center_lat=facility_height/2,
                      mapbox_zoom=5, # This will need adjustment based on facility_size mapping to lat/lon
                      margin={"r":0,"t":40,"l":0,"b":0})

    # Map facility_size to Mapbox view. This is tricky as mapbox uses lat/lon.
    # For a non-geo plot, the Histogram2dContour or Histogram2d is simpler.
    # If using Histogram2dContour/Histogram2d:
    # _apply_common_layout_settings(fig, "Aggregated Worker Density", high_contrast, yaxis_title="Y (m)", xaxis_title="X (m)")
    # fig.update_layout(xaxis_range=[0, facility_width], yaxis_range=[0, facility_height])
    # fig.update_xaxes(scaleanchor="y", scaleratio=1); fig.update_yaxes(constrain="domain")
    
    # Using simpler title if Mapbox for now, as layout settings are different.
    fig.update_layout(title_text="Aggregated Worker Density", title_x=0.5)
    
    return fig


def plot_worker_wellbeing(scores, triggers, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(scores)))
    fig.add_trace(go.Scatter(x=x_vals, y=scores, mode='lines', name='Well-Being Index', line=dict(color=p_c, width=2), hovertemplate='Well-Being: %{y:.1f}%<extra></extra>'))
    avg_wellbeing = np.mean(scores) if scores else None
    if avg_wellbeing is not None: fig.add_hline(y=avg_wellbeing, line=dict(color=n_c, width=1, dash="dot"), annotation_text=f"Avg: {avg_wellbeing:.1f}%", annotation_position="bottom left", annotation_font_size=10)

    trigger_colors = {'threshold': cr_c, 'trend': a_c, 'disruption': h_c, 'work_area_general': s_c}
    trigger_symbols = {'threshold': 'x', 'trend': 'triangle-down', 'disruption': 'star-diamond', 'work_area_general': 'diamond-tall'}
    
    for trigger_type, points in triggers.items(): # Process each trigger type separately
        flat_points = []
        if isinstance(points, list): flat_points = points
        elif isinstance(points, dict) and trigger_type == 'work_area': # Consolidate all work_area alerts for visibility
            for area_alerts in points.values(): flat_points.extend(area_alerts)
            trigger_type = 'work_area_general' # Use a general type for legend if plotting all work area triggers together

        valid_points = sorted(list(set(p for p in flat_points if 0 <= p < len(scores)))) # Unique, sorted, valid points
        if valid_points:
            fig.add_trace(go.Scatter(x=valid_points, y=[scores[p] for p in valid_points], mode='markers', 
                                     name=f'{trigger_type.replace("_", " ").title()} Alert', 
                                     marker=dict(color=trigger_colors.get(trigger_type, n_c), size=10, symbol=trigger_symbols.get(trigger_type, 'circle-open'), line=dict(width=1, color=p_c if high_contrast else "#FFFFFF" )),
                                     hovertemplate=f'{trigger_type.replace("_", " ").title()}: %{{y:.1f}}% at Step %{{x}}<extra></extra>'))
    _apply_common_layout_settings(fig, "Worker Well-Being Index Trend", high_contrast, yaxis_title="Index (%)", yaxis_range=[0, 105])
    return fig

def plot_psychological_safety(data, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name='Psych. Safety', line=dict(color=h_c, width=2), hovertemplate='Psych. Safety: %{y:.1f}%<extra></extra>'))
    avg_safety = np.mean(data) if data else None
    if avg_safety is not None: fig.add_hline(y=avg_safety, line=dict(color=n_c, width=1, dash="dot"), annotation_text=f"Avg: {avg_safety:.1f}%", annotation_position="bottom left", annotation_font_size=10)
    _apply_common_layout_settings(fig, "Psychological Safety Score Trend", high_contrast, yaxis_title="Score (%)", yaxis_range=[0, 105])
    return fig

def plot_downtime_trend(downtime_minutes, threshold, high_contrast=False):
    p_c, s_c, a_c, cr_c, h_c, n_c, pos_c = _get_colors(high_contrast); fig = go.Figure(); x_vals = list(range(len(downtime_minutes)))
    bar_colors = [cr_c if d > threshold else pos_c for d in downtime_minutes] # Positive color for at/below threshold
    fig.add_trace(go.Bar(x=x_vals, y=downtime_minutes, name='Downtime', marker_color=bar_colors, width=0.7, hovertemplate='Downtime: %{y:.1f} min<extra></extra>'))
    fig.add_hline(y=threshold, line=dict(color=a_c, width=1.5, dash="longdash"), annotation_text=f"Alert Threshold: {threshold} min", annotation_position="top right", annotation=dict(font_size=10, bgcolor="rgba(250,204,21,0.7)", borderpad=2))
    _apply_common_layout_settings(fig, "Downtime per Interval", high_contrast, yaxis_title="Downtime (minutes)")
    max_y_val = max(max(downtime_minutes) * 1.15 if downtime_minutes else threshold * 1.5, threshold * 1.5, 10) # Ensure y-axis is sufficiently scaled
    fig.update_yaxes(range=[0, max_y_val])
    return fig
