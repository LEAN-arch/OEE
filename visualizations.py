# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# --- Color Definitions for LIGHT GRAY THEME ---

# Base Backgrounds (Light Gray)
COLOR_PAGE_BG_LIGHT = "#F0F2F6" # A common light gray for app backgrounds
COLOR_PAPER_BG_LIGHT = COLOR_PAGE_BG_LIGHT # Plot paper background same as page
COLOR_PLOT_BG_LIGHT = COLOR_PAGE_BG_LIGHT  # Plot area background same as page

# Text Colors (Dark on Light Background)
COLOR_PRIMARY_TEXT_LIGHT_THEME = "#262730" # Dark gray for main text
COLOR_SECONDARY_TEXT_LIGHT_THEME = "#5E6474" # Slightly lighter dark gray for secondary text
COLOR_WHITE_ON_DARK_BG = "#FFFFFF" # For elements that might still have a dark bg (e.g. custom tooltips)

# Grid and Axis Lines (visible on light gray)
COLOR_GRID_LIGHT_THEME = "#D0D0D0" # Medium gray grid
COLOR_AXIS_LINE_LIGHT_THEME = "#B0B0B0" # Slightly darker gray for axis lines

# Semantic Colors (chosen for visibility on light gray, may need tweaking for specific color deficiencies)
# These are often used as line/marker colors, ensure they have good contrast with each other and background.
COLOR_CRITICAL_RED_LIGHT_THEME = "#D62728" # Plotly's default red
COLOR_WARNING_AMBER_LIGHT_THEME = "#FF7F0E" # Plotly's default orange
COLOR_POSITIVE_GREEN_LIGHT_THEME = "#2CA02C" # Plotly's default green
COLOR_INFO_BLUE_LIGHT_THEME = "#1F77B4"     # Plotly's default blue
COLOR_NEUTRAL_GRAY_LIGHT_THEME = "#7F7F7F"  # Mid-gray (can be same as dark text if contrast is ok)
COLOR_ACCENT_PURPLE_LIGHT_THEME = "#9467BD" # Plotly's default purple (alternative to main.py's indigo)

# Categorical Palettes (Okabe-Ito is good for accessibility)
# These are generally good on both light and dark backgrounds
ACCESSIBLE_CATEGORICAL_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
# High contrast may need adjustment if white is used for lines on light bg.
# For light bg, we want dark, distinct colors.
HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG = ['#000000', '#004949', '#b66dff', '#007959', '#990000', '#590000', '#003366', '#492500'] # Black, Dark Teal, Purple, Dark Green, Dark Red, Maroon, Dark Blue, Brown

# Sequential Palettes (generally okay, but test visibility)
ACCESSIBLE_SEQUENTIAL_PLOTLY_SCALES_LIGHT = ["Blues", "Greens", "Oranges", "Reds", "Purples", "Greys", "Viridis", "Cividis"]
HIGH_CONTRAST_SEQUENTIAL_PLOTLY_SCALES_LIGHT = ["OrRd", "PuBuGn", "YlGnBu", "Inferno", "Magma", "Blackbody"]


PLOTLY_TEMPLATE_LIGHT = "plotly_white" # Base light theme template
EPSILON = 1e-9

def _apply_common_layout_settings(fig, title_text, high_contrast=False,
                                 yaxis_title=None, xaxis_title="Time Step (Interval)",
                                 yaxis_range=None, show_legend=True):
    
    font_main_color = COLOR_PRIMARY_TEXT_LIGHT_THEME
    grid_c = COLOR_GRID_LIGHT_THEME 
    axis_line_c = COLOR_AXIS_LINE_LIGHT_THEME
    legend_bg = "rgba(255,255,255,0.85)" # Light, slightly transparent legend
    legend_border_c = COLOR_NEUTRAL_GRAY_LIGHT_THEME

    fig.update_layout(
        template=PLOTLY_TEMPLATE_LIGHT,
        title=dict(text=title_text, x=0.5, font=dict(size=16, color=font_main_color)),
        paper_bgcolor=COLOR_PAPER_BG_LIGHT, # Should match Streamlit page bg
        plot_bgcolor=COLOR_PLOT_BG_LIGHT,   # Should match Streamlit page bg
        font=dict(color=font_main_color, size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor=legend_bg, bordercolor=legend_border_c, borderwidth=1,
            font_size=10, traceorder="normal", font=dict(color=font_main_color)
        ) if show_legend else None,
        margin=dict(l=65, r=40, t=70, b=60),
        xaxis=dict(
            title=xaxis_title, gridcolor=grid_c, zerolinecolor=grid_c, zerolinewidth=1,
            showline=True, linewidth=1.5, linecolor=axis_line_c,
            rangemode='tozero' if xaxis_title and ("Time" in xaxis_title or "Step" in xaxis_title) else 'normal',
            titlefont=dict(size=13, color=font_main_color),
            tickfont=dict(size=11, color=font_main_color)
        ),
        yaxis=dict(
            title=yaxis_title, gridcolor=grid_c, zerolinecolor=grid_c, zerolinewidth=1,
            showline=True, linewidth=1.5, linecolor=axis_line_c,
            range=yaxis_range,
            titlefont=dict(size=13, color=font_main_color),
            tickfont=dict(size=11, color=font_main_color)
        ),
        hovermode="x unified",
        dragmode='pan'
    )
    fig.update_traces(hoverlabel=dict(bgcolor="rgba(240,240,240,0.9)", # Lighter tooltip bg
                                     font_size=12,
                                     font_color=COLOR_PRIMARY_TEXT_LIGHT_THEME, # Dark text for tooltip
                                     bordercolor=COLOR_NEUTRAL_GRAY_LIGHT_THEME))

def _get_no_data_figure(title_text, high_contrast=False): # high_contrast might not be needed for text if always dark
    fig = go.Figure()
    _apply_common_layout_settings(fig, title_text, high_contrast, show_legend=False,
                                 xaxis_title="", yaxis_title="")
    fig.update_xaxes(showticklabels=False, zeroline=False, showgrid=False, showline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False, showgrid=False, showline=False)
    fig.add_annotation(text="No data available for this visualization.",
                       showarrow=False, font=dict(size=14, color=COLOR_PRIMARY_TEXT_LIGHT_THEME))
    return fig

def _get_no_data_pie_figure(title_text, high_contrast=False):
    fig = go.Figure()
    _apply_common_layout_settings(fig, title_text, high_contrast, show_legend=False,
                                  xaxis_title="", yaxis_title="")
    fig.update_xaxes(showticklabels=False, zeroline=False, showgrid=False, showline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False, showgrid=False, showline=False)
    fig.add_annotation(text="No data available.",
                       showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(size=14, color=COLOR_PRIMARY_TEXT_LIGHT_THEME))
    return fig

def plot_key_metrics_summary(compliance: float, proximity: float, wellbeing: float, downtime: float,
                             target_compliance: float, target_proximity: float, target_wellbeing: float, target_downtime: float,
                             high_contrast: bool = False,
                             color_positive: str = COLOR_POSITIVE_GREEN_LIGHT_THEME, 
                             color_warning: str = COLOR_WARNING_AMBER_LIGHT_THEME,
                             color_negative: str = COLOR_CRITICAL_RED_LIGHT_THEME, 
                             accent_color: str = COLOR_ACCENT_PURPLE_LIGHT_THEME): # Changed to purple
    figs = []
    font_color_gauge = COLOR_PRIMARY_TEXT_LIGHT_THEME
    gauge_base_bgcolor = "rgba(220, 220, 220, 0.7)" # Lighter gray for gauge base
    gauge_border_color = COLOR_NEUTRAL_GRAY_LIGHT_THEME

    metrics_config = [
        ("Task Compliance", compliance, target_compliance, accent_color, "%", False),
        ("Collaboration Metric", proximity, target_proximity, accent_color, "%", False),
        ("Well-Being Index", wellbeing, target_wellbeing, accent_color, "%", False),
        ("Total Downtime", downtime, target_downtime, accent_color, " min", True)
    ]

    for title, raw_value, raw_target, bar_color_val, suffix, lower_is_better in metrics_config:
        fig = go.Figure(); value = 0.0; target = 0.0
        try: value = float(raw_value) if pd.notna(raw_value) else 0.0
        except (ValueError, TypeError): value = 0.0
        try: target = float(raw_target) if pd.notna(raw_target) else 0.0
        except (ValueError, TypeError): target = 0.0
        
        increasing_color = color_positive 
        decreasing_color = color_negative 
        if lower_is_better: # Swap for "lower is better" metrics
            increasing_color, decreasing_color = decreasing_color, increasing_color
        
        axis_max_val = 100.0 if suffix == "%" else max(abs(value) * 1.3, abs(target) * 1.3, 10.0)
        if abs(axis_max_val) < EPSILON : axis_max_val = 10.0

        steps_config = []
        c_good, c_warn, c_crit = color_positive, color_warning, color_negative

        if lower_is_better:
            s1_upper = target; s2_upper = target * 1.35
            if abs(target) < EPSILON: s1_upper = EPSILON * 2; s2_upper = axis_max_val * 0.25 if axis_max_val > EPSILON else EPSILON * 5
            steps_config = [{'range': [0, s1_upper], 'color': c_good}, {'range': [s1_upper, s2_upper], 'color': c_warn}, {'range': [s2_upper, axis_max_val], 'color': c_crit}]
        else:
            s1_upper = target * 0.75; s2_upper = target
            if abs(target) < EPSILON: s1_upper = axis_max_val * 0.33; s2_upper = axis_max_val * 0.66
            steps_config = [{'range': [0, s1_upper], 'color': c_crit}, {'range': [s1_upper, s2_upper], 'color': c_warn}, {'range': [s2_upper, axis_max_val], 'color': c_good}]
        
        valid_steps = []; current_lower = 0.0
        for step_conf in steps_config:
            s_l_raw, s_u_raw = step_conf['range']
            s_l = max(current_lower, float(s_l_raw) if pd.notna(s_l_raw) else current_lower)
            s_u = min(axis_max_val, float(s_u_raw) if pd.notna(s_u_raw) else axis_max_val)
            if s_u > s_l + EPSILON: valid_steps.append({'range': [s_l, s_u], 'color': step_conf['color']}); current_lower = s_u
        if not valid_steps: valid_steps = [{'range': [0, axis_max_val], 'color': accent_color}]

        fig.add_trace(go.Indicator(
            mode="gauge+number+delta", value=value,
            delta={'reference': target, 'increasing': {'color': increasing_color}, 'decreasing': {'color': decreasing_color}, 'font': {'size': 12, 'color': font_color_gauge}},
            title={'text': title, 'font': {'size': 13, 'color': font_color_gauge}},
            number={'suffix': suffix, 'font': {'size': 20, 'color': font_color_gauge}},
            gauge={'axis': {'range': [0, axis_max_val], 'tickwidth': 1, 'tickcolor': gauge_border_color, 'tickfont': {'color': font_color_gauge}},
                   'bar': {'color': bar_color_val if bar_color_val else accent_color, 'thickness': 0.7},
                   'bgcolor': gauge_base_bgcolor, 'borderwidth': 1, 'bordercolor': gauge_border_color,
                   'steps': valid_steps,
                   'threshold': {'line': {'color': font_color_gauge, 'width': 3}, 'thickness': 0.85, 'value': target}}))
        fig.update_layout(height=200, margin=dict(l=15,r=15,t=40,b=15), paper_bgcolor=COLOR_PAPER_BG_LIGHT, plot_bgcolor=COLOR_PLOT_BG_LIGHT, font=dict(color=font_color_gauge))
        figs.append(fig)
    return figs

def plot_task_compliance_score(data, disruption_points=None, forecast_data=None, z_scores=None, high_contrast=False):
    if not isinstance(data, (list, np.ndarray, pd.Series)) or not any(pd.notna(x) for x in data):
        return _get_no_data_figure("Task Compliance Score Trend", high_contrast)
    
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure(); x_vals = list(range(len(data)))

    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Compliance',
                             line=dict(color=cat_palette[0 % len(cat_palette)], width=2.5),
                             marker=dict(size=6, symbol="circle", line=dict(width=1, color=cat_palette[0 % len(cat_palette)])),
                             hovertemplate='Compliance: %{y:.1f}%<extra></extra>'))
    if forecast_data and len(forecast_data)==len(data) and any(pd.notna(x) for x in forecast_data):
        fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast',
                                 line=dict(color=cat_palette[1 % len(cat_palette)], dash='dashdot', width=2),
                                 hovertemplate='Forecast: %{y:.1f}%<extra></extra>'))
    
    if disruption_points:
        for dp_step in disruption_points:
            if 0 <= dp_step < len(data):
                fig.add_vline(x=dp_step, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="longdash"),
                              annotation_text="▼", annotation_position="top",
                              annotation=dict(font_size=12, font_color=COLOR_WARNING_AMBER_LIGHT_THEME, showarrow=False, yshift=10))

    valid_data_for_range = [v for v in data if pd.notna(v)]
    if forecast_data: valid_data_for_range.extend([v for v in forecast_data if pd.notna(v)])
    min_val = min(valid_data_for_range) if valid_data_for_range else 0.0
    max_val = max(valid_data_for_range) if valid_data_for_range else 100.0
    
    _apply_common_layout_settings(fig, "Task Compliance Score Trend", high_contrast,
                                 yaxis_title="Score (%)",
                                 yaxis_range=[max(0, min_val - 10), min(105, max_val + 10)])
    return fig

def plot_collaboration_proximity_index(data, disruption_points=None, forecast_data=None, high_contrast=False):
    plot_title = "Collaboration Metric Trend"
    if not isinstance(data, (list, np.ndarray, pd.Series)) or not any(pd.notna(x) for x in data):
        return _get_no_data_figure(plot_title, high_contrast)

    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure(); x_vals = list(range(len(data)))

    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Collab. Metric',
                             line=dict(color=cat_palette[1 % len(cat_palette)], width=2.5),
                             marker=dict(size=6, symbol="diamond", line=dict(width=1, color=cat_palette[1 % len(cat_palette)])),
                             hovertemplate='Collab. Metric: %{y:.1f}%<extra></extra>'))
    if forecast_data and len(forecast_data)==len(data) and any(pd.notna(x) for x in forecast_data):
        fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast',
                                 line=dict(color=cat_palette[2 % len(cat_palette)], dash='dashdot', width=2),
                                 hovertemplate='Forecast: %{y:.1f}%<extra></extra>'))

    if disruption_points:
        for dp_step in disruption_points:
            if 0 <= dp_step < len(data):
                fig.add_vline(x=dp_step, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="longdash"),
                              annotation_text="▼", annotation_position="top",
                              annotation=dict(font_size=12, font_color=COLOR_WARNING_AMBER_LIGHT_THEME, showarrow=False, yshift=10))
                                  
    valid_data_for_range = [v for v in data if pd.notna(v)]
    if forecast_data: valid_data_for_range.extend([v for v in forecast_data if pd.notna(v)])
    min_val = min(valid_data_for_range) if valid_data_for_range else 0.0
    max_val = max(valid_data_for_range) if valid_data_for_range else 100.0

    _apply_common_layout_settings(fig, plot_title, high_contrast,
                                 yaxis_title="Score (%)",
                                 yaxis_range=[max(0, min_val - 10), min(105, max_val + 10)])
    return fig

def plot_operational_recovery(recovery_data, productivity_loss_data, disruption_points=None, high_contrast=False):
    if not isinstance(recovery_data, (list, np.ndarray, pd.Series)) or not any(pd.notna(x) for x in recovery_data):
        return _get_no_data_figure("Operational Resilience", high_contrast)
    fig = go.Figure(); x_vals = list(range(len(recovery_data)))
    fig.add_trace(go.Scatter(x=x_vals, y=recovery_data, mode='lines', name='Op. Recovery',
                             line=dict(color=COLOR_POSITIVE_GREEN_LIGHT_THEME, width=2.5, dash='solid'),
                             hovertemplate='Recovery: %{y:.1f}%<extra></extra>'))
    if productivity_loss_data and len(productivity_loss_data) == len(recovery_data) and any(pd.notna(x) for x in productivity_loss_data):
        fig.add_trace(go.Scatter(x=x_vals, y=productivity_loss_data, mode='lines', name='Prod. Loss',
                                 line=dict(color=COLOR_CRITICAL_RED_LIGHT_THEME, dash='dot', width=2.5),
                                 hovertemplate='Prod. Loss: %{y:.1f}%<extra></extra>'))
    if disruption_points:
        for dp_step in disruption_points:
            if 0 <= dp_step < len(recovery_data):
                fig.add_vline(x=dp_step, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="longdash"),
                              annotation_text="▼", annotation_position="top",
                              annotation=dict(font_size=12, font_color=COLOR_WARNING_AMBER_LIGHT_THEME, showarrow=False, yshift=10))
    _apply_common_layout_settings(fig, "Operational Resilience: Recovery vs. Loss", high_contrast,
                                 yaxis_title="Percentage (%)", yaxis_range=[0, 105])
    return fig

def plot_operational_efficiency(efficiency_df, selected_metrics, disruption_points=None, high_contrast=False):
    if not isinstance(efficiency_df, pd.DataFrame) or efficiency_df.empty:
        return _get_no_data_figure("Operational Efficiency (OEE)", high_contrast)
    
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure()
    metric_styles = {
        'uptime': {'color_idx': 0, 'dash': 'solid', 'width': 2.0},
        'throughput': {'color_idx': 1, 'dash': 'solid', 'width': 2.0},
        'quality': {'color_idx': 2, 'dash': 'solid', 'width': 2.0},
        'oee': {'color_idx': 3, 'dash': 'solid', 'width': 3.0}
    }
    for i, metric in enumerate(selected_metrics):
        if metric in efficiency_df.columns and not efficiency_df[metric].isnull().all():
            style_info = metric_styles.get(metric, {'color_idx': i % len(cat_palette), 'dash': 'solid', 'width': 2.0})
            color = cat_palette[style_info['color_idx'] % len(cat_palette)]
            fig.add_trace(go.Scatter(x=efficiency_df.index, y=efficiency_df[metric], mode='lines', name=metric.upper(),
                                     line=dict(color=color, width=style_info['width'], dash=style_info['dash']),
                                     hovertemplate=f'{metric.upper()}: %{{y:.1f}}%<extra></extra>'))
    if disruption_points:
        for dp_step in disruption_points:
             actual_x_val = dp_step
             if not isinstance(efficiency_df.index, pd.RangeIndex) and isinstance(dp_step, int) and dp_step < len(efficiency_df.index):
                 actual_x_val = efficiency_df.index[dp_step]
             if actual_x_val in efficiency_df.index or (isinstance(dp_step, int) and 0 <= dp_step < len(efficiency_df)):
                fig.add_vline(x=actual_x_val, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="longdash"),
                              annotation_text="▼", annotation_position="top",
                              annotation=dict(font_size=12, font_color=COLOR_WARNING_AMBER_LIGHT_THEME, showarrow=False, yshift=10))
    _apply_common_layout_settings(fig, "OEE & Components", high_contrast,
                                 yaxis_title="Efficiency Score (%)", yaxis_range=[0, 105])
    return fig

def plot_worker_distribution(team_positions_df, facility_size, facility_config, use_3d=False, selected_step=0, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    work_areas_cfg = facility_config.get('WORK_AREAS', {})
    entry_exit_cfg = facility_config.get('ENTRY_EXIT_POINTS', [])
    mpi = facility_config.get('MINUTES_PER_INTERVAL', 2)

    plot_title = f"Worker Distribution (Time: {selected_step * mpi} min)"
    if not isinstance(team_positions_df, pd.DataFrame) or team_positions_df.empty:
         return _get_no_data_figure(plot_title, high_contrast)

    df_step = team_positions_df[team_positions_df['step'] == selected_step].copy()
    if df_step.empty:
        return _get_no_data_figure(f"{plot_title} - No data for this step", high_contrast)

    facility_width, facility_height = facility_size
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    
    if 'zone' not in df_step.columns: df_step['zone'] = 'UnknownZone'
    else: df_step['zone'] = df_step['zone'].fillna('UnknownZone')
    
    zone_names = sorted(df_step['zone'].unique())
    zone_color_map = {zone: cat_palette[i % len(cat_palette)] for i, zone in enumerate(zone_names)}

    status_symbols_map = {"working": "circle", "idle": "square", "break":"diamond", "fatigued": "cross", "exhausted": "x-thin", "disrupted":"hourglass"}
    df_step['symbol_plotly'] = df_step['status'].map(status_symbols_map).fillna("circle")

    if use_3d and 'z' not in df_step.columns: df_step['z'] = 0.0
    
    if use_3d:
        fig = px.scatter_3d(df_step, x='x', y='y', z='z', color='zone', color_discrete_map=zone_color_map,
                            hover_name='worker_id', hover_data={'zone': True, 'status': True, 'x':':.1f', 'y':':.1f', 'z':':.1f'},
                            range_x=[0, facility_width], range_y=[0, facility_height], 
                            range_z=[0,max(5, df_step['z'].max() if 'z' in df_step.columns and not df_step['z'].empty and pd.notna(df_step['z'].max()) else 5.0)],
                            symbol='symbol_plotly', opacity=0.9, size_max=12)
        fig.update_scenes(aspectmode='data', xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False,
                          xaxis_backgroundcolor=COLOR_PLOT_BG_LIGHT, yaxis_backgroundcolor=COLOR_PLOT_BG_LIGHT, zaxis_backgroundcolor=COLOR_PLOT_BG_LIGHT)
    else:
        fig = px.scatter(df_step, x='x', y='y', color='zone', color_discrete_map=zone_color_map,
                         hover_name='worker_id', hover_data={'zone': True, 'status': True, 'x':':.1f', 'y':':.1f'},
                         range_x=[-5, facility_width + 5], range_y=[-5, facility_height + 5],
                         symbol='symbol_plotly', opacity=0.9, size_max=10)
        grid_c = COLOR_GRID_LIGHT_THEME
        fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True, xaxis_gridcolor=grid_c, yaxis_gridcolor=grid_c,
                          xaxis_zeroline=True, yaxis_zeroline=True, xaxis_zerolinecolor=grid_c, yaxis_zerolinecolor=grid_c)

    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height,
                              line=dict(color=COLOR_NEUTRAL_GRAY_LIGHT_THEME, width=1.5), fillcolor="rgba(0,0,0,0)", layer="below")]
    annotations = []
    font_c = COLOR_PRIMARY_TEXT_LIGHT_THEME
    ee_point_color = COLOR_INFO_BLUE_LIGHT_THEME
    work_area_outline_color = ACCESSIBLE_CATEGORICAL_PALETTE[4 % len(ACCESSIBLE_CATEGORICAL_PALETTE)] 

    if show_entry_exit and entry_exit_cfg:
        for point in entry_exit_cfg:
            if isinstance(point, dict) and 'coords' in point and isinstance(point['coords'], tuple) and len(point['coords']) == 2:
                px_ee, py_ee = point['coords']
                shapes.append(go.layout.Shape(type="circle", x0=px_ee-1.5, y0=py_ee-1.5, x1=px_ee+1.5, y1=py_ee+1.5,
                                            fillcolor=ee_point_color, line_color=COLOR_PRIMARY_TEXT_LIGHT_THEME, line_width=1, opacity=0.9, layer="above"))
                annotations.append(dict(x=px_ee, y=py_ee+4, text=point.get('name','EE')[:2].upper(), showarrow=False,
                                        font=dict(size=9, color=font_c), borderpad=2,
                                        bgcolor="rgba(200,200,200,0.3)" if not high_contrast else "rgba(100,100,100,0.5)"))
    if show_prod_lines and work_areas_cfg:
         for area_name, area_details in work_areas_cfg.items():
            if isinstance(area_details, dict) and 'coords' in area_details:
                coords_list = area_details['coords']
                if isinstance(coords_list, list) and len(coords_list) == 2 and \
                   isinstance(coords_list[0], tuple) and len(coords_list[0])==2 and \
                   isinstance(coords_list[1], tuple) and len(coords_list[1])==2:
                    (x0,y0), (x1,y1) = coords_list[0], coords_list[1]
                    if all(isinstance(c, (int,float)) for c_tuple in coords_list for c in c_tuple):
                        shapes.append(go.layout.Shape(type="rect", x0=min(x0,x1), y0=min(y0,y1), x1=max(x0,x1), y1=max(y0,y1),
                                                    line=dict(color=work_area_outline_color, dash="dashdot", width=1.5),
                                                    fillcolor="rgba(200,200,200,0.05)", layer="below"))
                        annotations.append(dict(x=(x0+x1)/2, y=min(y0,y1)-4, text=area_name, showarrow=False,
                                                font=dict(size=9, color=COLOR_SECONDARY_TEXT_LIGHT_THEME), opacity=0.9, yanchor="top"))

    _apply_common_layout_settings(fig, plot_title, high_contrast,
                                 yaxis_title="Y (m)", xaxis_title="X (m)", show_legend=True)
    fig.update_layout(shapes=shapes, annotations=annotations, legend_title_text='Zone')
    return fig

def plot_worker_density_heatmap(team_positions_df, facility_size, facility_config, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    work_areas_cfg = facility_config.get('WORK_AREAS', {})
    entry_exit_cfg = facility_config.get('ENTRY_EXIT_POINTS', [])
    plot_title = "Aggregated Worker Density Heatmap"

    if not isinstance(team_positions_df, pd.DataFrame) or team_positions_df.empty or 'x' not in team_positions_df.columns or 'y' not in team_positions_df.columns:
        return _get_no_data_figure(plot_title, high_contrast)

    facility_width, facility_height = facility_size
    heatmap_colorscale = HIGH_CONTRAST_SEQUENTIAL_PLOTLY_SCALES_LIGHT[0] if high_contrast else ACCESSIBLE_SEQUENTIAL_PLOTLY_SCALES_LIGHT[0]

    fig = go.Figure(go.Histogram2dContour(
        x=team_positions_df['x'], y=team_positions_df['y'],
        colorscale=heatmap_colorscale, reversescale=False, showscale=True, line=dict(width=0),
        contours=dict(coloring='heatmap', showlabels=False),
        xbins=dict(start=-facility_width*0.05, end=facility_width*1.05, size=facility_width/max(1,int(facility_width/15))), 
        ybins=dict(start=-facility_height*0.05, end=facility_height*1.05, size=facility_height/max(1,int(facility_height/12))),
        colorbar=dict(title='Density', thickness=15, len=0.75, y=0.5, tickfont_size=10, x=1.05,
                      bgcolor="rgba(255,255,255,0.1)", bordercolor=COLOR_NEUTRAL_GRAY_LIGHT_THEME,
                      titlefont_color=COLOR_PRIMARY_TEXT_LIGHT_THEME, tickfont_color=COLOR_PRIMARY_TEXT_LIGHT_THEME)
    ))
    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height,
                              line=dict(color=COLOR_NEUTRAL_GRAY_LIGHT_THEME, width=1.5), layer="below")]
    
    ee_point_color = COLOR_INFO_BLUE_LIGHT_THEME
    work_area_outline_color = TOL_BRIGHT_CATEGORICAL[5 % len(TOL_BRIGHT_CATEGORICAL)]

    if show_entry_exit and entry_exit_cfg:
        for point in entry_exit_cfg:
            if isinstance(point, dict) and 'coords' in point and isinstance(point['coords'], tuple) and len(point['coords']) == 2:
                px_ee, py_ee = point['coords']
                shapes.append(go.layout.Shape(type="circle", x0=px_ee-1, y0=py_ee-1, x1=px_ee+1, y1=py_ee+1,
                                            fillcolor=ee_point_color, line_color=COLOR_PRIMARY_TEXT_LIGHT_THEME, opacity=0.6, layer="above"))
    if show_prod_lines and work_areas_cfg:
         for area_details in work_areas_cfg.values():
            if isinstance(area_details, dict) and 'coords' in area_details:
                coords_list = area_details['coords']
                if isinstance(coords_list, list) and len(coords_list) == 2 and \
                   isinstance(coords_list[0], tuple) and isinstance(coords_list[1], tuple):
                    (x0,y0), (x1,y1) = coords_list[0], coords_list[1]
                    if all(isinstance(c, (int,float)) for c_tuple in coords_list for c in c_tuple):
                        shapes.append(go.layout.Shape(type="rect", x0=min(x0,x1), y0=min(y0,y1), x1=max(x0,x1), y1=max(y0,y1),
                                                    line=dict(color=work_area_outline_color, dash="dot", width=1.5),
                                                    fillcolor="rgba(0,0,0,0)", layer="above"))

    _apply_common_layout_settings(fig, plot_title, high_contrast,
                                 yaxis_title="Y Coordinate (m)", xaxis_title="X Coordinate (m)")
    fig.update_layout(xaxis_range=[0, facility_width], yaxis_range=[0, facility_height], shapes=shapes, autosize=True)
    fig.update_xaxes(constrain="domain"); fig.update_yaxes(constrain="domain", scaleanchor="x", scaleratio=1)
    return fig

def plot_worker_wellbeing(scores, triggers, disruption_points=None, high_contrast=False):
    if not isinstance(scores, (list, np.ndarray, pd.Series)) or not any(pd.notna(x) for x in scores):
        return _get_no_data_figure("Worker Well-Being Index Trend", high_contrast)

    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure(); x_vals = list(range(len(scores)))

    fig.add_trace(go.Scatter(x=x_vals, y=scores, mode='lines', name='Well-Being Index',
                             line=dict(color=cat_palette[0 % len(cat_palette)], width=2.5),
                             hovertemplate='Well-Being: %{y:.1f}%<extra></extra>'))
    
    valid_scores = [s for s in scores if pd.notna(s)]
    avg_wellbeing = np.mean(valid_scores) if valid_scores else None
    if avg_wellbeing is not None:
        fig.add_hline(y=avg_wellbeing, line=dict(color=COLOR_NEUTRAL_GRAY_LIGHT_THEME, width=1, dash="dot"),
                      annotation_text=f"Avg: {avg_wellbeing:.1f}%", annotation_position="bottom left",
                      annotation_font_size=10, annotation_font_color=COLOR_NEUTRAL_GRAY_LIGHT_THEME)

    if disruption_points:
        for dp_step in disruption_points:
            if 0 <= dp_step < len(scores):
                fig.add_vline(x=dp_step, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="longdash"),
                              annotation_text="▼", annotation_position="top",
                              annotation=dict(font_size=12, font_color=COLOR_WARNING_AMBER_LIGHT_THEME, showarrow=False, yshift=10))

    trigger_styles = {
        'threshold': {'color': COLOR_CRITICAL_RED_LIGHT_THEME, 'symbol': 'x-dot', 'name': 'Threshold Alert', 'size': 10},
        'trend': {'color': COLOR_WARNING_AMBER_LIGHT_THEME, 'symbol': 'triangle-down-dot', 'name': 'Trend Alert', 'size': 10},
        'disruption': {'color': COLOR_INFO_BLUE_LIGHT_THEME, 'symbol': 'star-dot', 'name': 'Disruption Link', 'size': 10},
        'work_area': {'color': cat_palette[2 % len(cat_palette)], 'symbol': 'diamond-dot', 'name': 'Work Area Alert', 'size': 9}
    }
    default_trigger_style = {'color': COLOR_NEUTRAL_GRAY_LIGHT_THEME, 'symbol': 'circle-dot', 'name': 'Other Alert', 'size': 8}
    triggers_data = triggers if isinstance(triggers, dict) else {}
    
    for trigger_type, points_data in triggers_data.items():
        relative_steps_for_type = []; style_key = trigger_type
        if trigger_type == 'work_area' and isinstance(points_data, dict):
            all_wa_rel_steps = set()
            for rel_steps_list in points_data.values():
                if isinstance(rel_steps_list, list): all_wa_rel_steps.update(p for p in rel_steps_list if isinstance(p, (int, float)))
            relative_steps_for_type = sorted([int(p) for p in list(all_wa_rel_steps) if 0 <= p < len(scores)])
        elif isinstance(points_data, list):
            relative_steps_for_type = sorted([int(p) for p in points_data if isinstance(p, (int, float)) and 0 <= p < len(scores)])
        else: continue

        if relative_steps_for_type:
            style = trigger_styles.get(style_key, default_trigger_style)
            valid_x_points = [p for p in relative_steps_for_type if pd.notna(scores[p])] # Ensure corresponding score is not NaN
            valid_y_values = [scores[p] for p in valid_x_points]

            if valid_x_points and valid_y_values: # Redundant check as valid_x_points ensures valid_y_values
                fig.add_trace(go.Scatter(
                    x=valid_x_points, y=valid_y_values, mode='markers', name=style['name'],
                    marker=dict(color=style['color'], size=style['size'], symbol=style['symbol'],
                                line=dict(width=1.5, color=COLOR_PLOT_BG_LIGHT if high_contrast else style['color'])),
                    hovertemplate=f'{style["name"]}: %{{y:.1f}}% at Step %{{x}}<extra></extra>' ))
    _apply_common_layout_settings(fig, "Worker Well-Being Index Trend", high_contrast,
                                 yaxis_title="Index (%)", yaxis_range=[0, 105])
    return fig

def plot_psychological_safety(data, disruption_points=None, high_contrast=False):
    if not isinstance(data, (list, np.ndarray, pd.Series)) or not any(pd.notna(x) for x in data):
        return _get_no_data_figure("Psychological Safety Score Trend", high_contrast)
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure(); x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name='Psych. Safety',
                             line=dict(color=cat_palette[2 % len(cat_palette)], width=2.5),
                             hovertemplate='Psych. Safety: %{y:.1f}%<extra></extra>'))
    valid_scores = [s for s in data if pd.notna(s)]
    avg_safety = np.mean(valid_scores) if valid_scores else None
    if avg_safety is not None:
        fig.add_hline(y=avg_safety, line=dict(color=COLOR_NEUTRAL_GRAY_LIGHT_THEME, width=1, dash="dot"),
                      annotation_text=f"Avg: {avg_safety:.1f}%", annotation_position="bottom left",
                      annotation_font_size=10, annotation_font_color=COLOR_NEUTRAL_GRAY_LIGHT_THEME)
    if disruption_points:
        for dp_step in disruption_points:
            if 0 <= dp_step < len(data):
                fig.add_vline(x=dp_step, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="longdash"),
                              annotation_text="▼", annotation_position="top",
                              annotation=dict(font_size=12, font_color=COLOR_WARNING_AMBER_LIGHT_THEME, showarrow=False, yshift=10))
    _apply_common_layout_settings(fig, "Psychological Safety Score Trend", high_contrast,
                                 yaxis_title="Score (%)", yaxis_range=[0, 105])
    return fig

def plot_downtime_trend(downtime_per_step_data, interval_threshold_minutes, disruption_points=None, high_contrast=False):
    if not isinstance(downtime_per_step_data, (list, np.ndarray, pd.Series)) or not any(d > EPSILON for d in downtime_per_step_data):
        return _get_no_data_figure("Downtime per Interval", high_contrast)

    fig = go.Figure(); x_vals = list(range(len(downtime_per_step_data)))
    bar_colors = [COLOR_CRITICAL_RED_LIGHT_THEME if d > interval_threshold_minutes else COLOR_POSITIVE_GREEN_LIGHT_THEME for d in downtime_per_step_data]
    fig.add_trace(go.Bar(x=x_vals, y=downtime_per_step_data, name='Downtime', marker_color=bar_colors, width=0.7,
                         hovertemplate='Downtime: %{y:.1f} min<extra></extra>'))
    fig.add_hline(y=interval_threshold_minutes, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="dash"),
                  annotation_text=f"Alert: {interval_threshold_minutes} min", annotation_position="top right",
                  annotation_font=dict(size=10, color=COLOR_WARNING_AMBER_LIGHT_THEME), # Removed bgcolor for cleaner look on light theme
                  annotation_bgcolor="rgba(255,255,255,0.6)")


    max_y_val = max(max(downtime_per_step_data) * 1.15 if downtime_per_step_data and max(downtime_per_step_data) > 0 else 10.0, interval_threshold_minutes * 1.5, 10.0)
    _apply_common_layout_settings(fig, "Downtime per Interval", high_contrast, yaxis_title="Downtime (minutes)")
    fig.update_yaxes(range=[0, max_y_val])
    return fig

def plot_team_cohesion(data, disruption_points=None, high_contrast=False):
    if not isinstance(data, (list, np.ndarray, pd.Series)) or not any(pd.notna(x) for x in data):
        return _get_no_data_figure("Team Cohesion Index Trend", high_contrast)
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure(); x_vals = list(range(len(data)))
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name='Team Cohesion',
                             line=dict(color=cat_palette[3 % len(cat_palette)], width=2.5),
                             hovertemplate='Cohesion: %{y:.1f}%<extra></extra>'))
    valid_scores = [s for s in data if pd.notna(s)]
    avg_cohesion = np.mean(valid_scores) if valid_scores else None
    if avg_cohesion is not None:
        fig.add_hline(y=avg_cohesion, line=dict(color=COLOR_NEUTRAL_GRAY_LIGHT_THEME, width=1, dash="dot"),
                      annotation_text=f"Avg: {avg_cohesion:.1f}%", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color=COLOR_NEUTRAL_GRAY_LIGHT_THEME)
    if disruption_points:
        for dp_step in disruption_points:
            if 0 <= dp_step < len(data):
                fig.add_vline(x=dp_step, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="longdash"),
                              annotation_text="▼", annotation_position="top",
                              annotation=dict(font_size=12, font_color=COLOR_WARNING_AMBER_LIGHT_THEME, showarrow=False, yshift=10))
    _apply_common_layout_settings(fig, "Team Cohesion Index Trend", high_contrast,
                                 yaxis_title="Cohesion Index (%)", yaxis_range=[0, 105])
    return fig

def plot_perceived_workload(data, high_workload_threshold, very_high_workload_threshold,
                            disruption_points=None, high_contrast=False):
    if not isinstance(data, (list, np.ndarray, pd.Series)) or not any(pd.notna(x) for x in data):
        return _get_no_data_figure("Perceived Workload Index", high_contrast)
    
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure(); x_vals = list(range(len(data)))
    line_color_main = cat_palette[4 % len(cat_palette)]
    marker_colors = [COLOR_CRITICAL_RED_LIGHT_THEME if val >= very_high_workload_threshold else
                     COLOR_WARNING_AMBER_LIGHT_THEME if val >= high_workload_threshold else
                     COLOR_POSITIVE_GREEN_LIGHT_THEME for val in data]
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Perceived Workload',
                             line=dict(color=line_color_main, width=2),
                             marker=dict(size=6, color=marker_colors, line=dict(width=1, color=COLOR_PAPER_BG_LIGHT if high_contrast else line_color_main)),
                             hovertemplate='Workload: %{y:.1f}/10<extra></extra>'))
    fig.add_hline(y=high_workload_threshold, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="dash"),
                  annotation_text=f"High ({high_workload_threshold})", annotation_position="bottom right",
                  annotation_font=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, size=10))
    fig.add_hline(y=very_high_workload_threshold, line=dict(color=COLOR_CRITICAL_RED_LIGHT_THEME, width=1.5, dash="dash"),
                  annotation_text=f"V.High ({very_high_workload_threshold})", annotation_position="top right",
                  annotation_font=dict(color=COLOR_CRITICAL_RED_LIGHT_THEME, size=10))
    if disruption_points:
        for dp_step in disruption_points:
            if 0 <= dp_step < len(data):
                 fig.add_vline(x=dp_step, line=dict(color=COLOR_WARNING_AMBER_LIGHT_THEME, width=1.5, dash="longdash"),
                              annotation_text="▼", annotation_position="top",
                              annotation=dict(font_size=12, font_color=COLOR_WARNING_AMBER_LIGHT_THEME, showarrow=False, yshift=10))
    _apply_common_layout_settings(fig, "Perceived Workload Index (0-10 Scale)", high_contrast,
                                 yaxis_title="Workload Index", yaxis_range=[0, 10.5])
    return fig

def plot_downtime_causes_pie(downtime_events_list_for_pie, high_contrast=False):
    if not isinstance(downtime_events_list_for_pie, list) or not downtime_events_list_for_pie:
        return _get_no_data_pie_figure("Downtime Distribution by Cause", high_contrast)

    causes_summary = {}; total_downtime_duration_for_pie = 0.0
    for event in downtime_events_list_for_pie:
        if not isinstance(event, dict): continue
        duration = event.get('duration', 0.0); cause = str(event.get('cause', 'Unknown')).strip()
        if not isinstance(duration, (int, float)) or duration <= EPSILON: continue
        if cause == "None" or cause == "Unknown" or not cause : cause = "Other/Unspecified"
        if "Equip" in cause and "Fail" in cause: cause = "Equipment Failure"
        elif "Human" in cause and "Error" in cause: cause = "Human Error"
        elif "Material" in cause: cause = "Material Shortage"
        elif "Process" in cause or "Bottle" in cause : cause = "Process Bottleneck"
        elif "Meeting" in cause: cause = "Meeting Overrun"
        causes_summary[cause] = causes_summary.get(cause, 0.0) + duration
        total_downtime_duration_for_pie += duration

    if not causes_summary or total_downtime_duration_for_pie < EPSILON :
        return _get_no_data_pie_figure("Downtime by Cause (No Categorized Downtime)", high_contrast)

    labels = list(causes_summary.keys()); values = list(causes_summary.values())
    pie_color_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE_LIGHT_BG if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    pie_colors = [pie_color_palette[i % len(pie_color_palette)] for i in range(len(labels))]
    slice_border_color = COLOR_GRID_LIGHT_THEME # Use grid color for border for subtlety
    text_color_pie = COLOR_PRIMARY_TEXT_LIGHT_THEME # Dark text for pie chart labels if on light background

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.45, pull=[0.02]*len(labels), marker_colors=pie_colors,
        textfont=dict(color=text_color_pie, size=11), insidetextorientation='radial',
        hovertemplate="<b>%{label}</b><br>Duration: %{value:.1f} min<br>Share: %{percent}<extra></extra>",
        sort=True, direction='clockwise' )])
    fig.update_traces(marker=dict(line=dict(color=slice_border_color, width=1.5)))
    _apply_common_layout_settings(fig, f"Downtime by Cause (Total: {total_downtime_duration_for_pie:.0f} min)",
                                 high_contrast, show_legend=(len(labels) <= 7))
    fig.update_layout(legend_title_text="Downtime Causes" if len(labels) <=7 else None,
                      uniformtext_minsize=9, uniformtext_mode='hide')
    return fig
