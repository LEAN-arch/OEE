# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import logging # For logging errors within visualization functions

logger = logging.getLogger(__name__)

# --- Accessible Color Definitions (Consistent with main.py) ---
# General purpose accessible palette for categorical data (Okabe-Ito based)
ACCESSIBLE_CATEGORICAL_PALETTE = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#CC79A7",  # Reddish Purple
    "#999999",  # Gray (for an 8th option if needed)
]

# High contrast palette for categorical data (for dark theme)
HIGH_CONTRAST_CATEGORICAL_PALETTE = [
    "#FFFFFF",  # White
    "#FFFF00",  # Bright Yellow
    "#FF69B4",  # Hot Pink
    "#00FFFF",  # Cyan
    "#39FF14",  # Neon Green
    "#FFA500",  # Orange (distinct from yellow)
    "#BEBEBE",  # Light Gray (as an alternative to white if white is too stark for some elements)
    "#FF0000",  # Bright Red
]

# Plotly built-in sequential color scales known for accessibility
ACCESSIBLE_SEQUENTIAL_PLOTLY_SCALES = ["Viridis", "Cividis", "Plasma", "Blues", "Greens", "Oranges"]
# For high contrast sequential, consider starker versions or perceptually uniform ones
HIGH_CONTRAST_SEQUENTIAL_PLOTLY_SCALES = ["Inferno", "Magma", "OrRd", "YlGnBu", "Greys"]


# Semantic colors for indicators (e.g., gauges, positive/negative trends)
COLOR_CRITICAL_RED = "#E53E3E"     # Tailwind Red 500 / Used for critical states, errors
COLOR_WARNING_AMBER = "#F59E0B"    # Tailwind Amber 500 / Used for warnings, moderate issues
COLOR_POSITIVE_GREEN = "#10B981"   # Tailwind Emerald 500 / Used for positive states, success
COLOR_INFO_BLUE = "#3B82F6"        # Tailwind Blue 500 / Used for informational elements
COLOR_NEUTRAL_GRAY = "#6B7280"     # Tailwind Gray 500 / For neutral elements, secondary text
COLOR_ACCENT_INDIGO = "#4F46E5"    # Main accent color from CSS

# Theme-dependent text and background colors
COLOR_WHITE_TEXT = "#FFFFFF"
COLOR_LIGHT_TEXT = "#EAEAEA"       # Default text on dark theme
COLOR_DARK_TEXT_ON_LIGHT_BG_HC = "#000000" # For annotations on light HC elements
COLOR_SUBTLE_GRID_STD = "#374151"  # Standard dark theme grid
COLOR_SUBTLE_GRID_HC = "#555555"   # High-contrast dark theme grid
COLOR_AXIS_LINE_STD = "#4A5568"
COLOR_AXIS_LINE_HC = "#777777"

# Default Plotly template (can be overridden by specific settings)
PLOTLY_TEMPLATE = "plotly_dark"

def _apply_common_layout_settings(fig, title_text, high_contrast=False, yaxis_title=None, xaxis_title="Time Step (Interval)", yaxis_range=None, show_legend=True):
    font_color = COLOR_WHITE_TEXT if high_contrast else COLOR_LIGHT_TEXT
    title_font_color = COLOR_WHITE_TEXT
    grid_color = COLOR_SUBTLE_GRID_HC if high_contrast else COLOR_SUBTLE_GRID_STD
    axis_line_color = COLOR_AXIS_LINE_HC if high_contrast else COLOR_AXIS_LINE_STD
    legend_bgcolor = "rgba(0,0,0,0.7)" if high_contrast else "rgba(31, 41, 55, 0.7)" # More opaque black for HC
    legend_border_color = COLOR_NEUTRAL_GRAY

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title_text, x=0.5, font=dict(size=16, color=title_font_color)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor=legend_bgcolor,
            bordercolor=legend_border_color, borderwidth=0.5, font_size=10,
            traceorder="normal",
            font=dict(color=font_color)
        ) if show_legend else None,
        margin=dict(l=60, r=40, t=70, b=60),
        xaxis=dict(
            title=xaxis_title, gridcolor=grid_color, zerolinecolor=grid_color,
            showline=True, linewidth=1, linecolor=axis_line_color,
            rangemode='tozero' if xaxis_title and "Time" in xaxis_title else 'normal',
            titlefont=dict(size=13, color=font_color), tickfont=dict(size=11, color=font_color)
        ),
        yaxis=dict(
            title=yaxis_title, gridcolor=grid_color, zerolinecolor=grid_color,
            showline=True, linewidth=1, linecolor=axis_line_color,
            range=yaxis_range,
            titlefont=dict(size=13, color=font_color), tickfont=dict(size=11, color=font_color)
        ),
        hovermode="x unified",
        dragmode='pan'
    )

def _get_no_data_figure(title_text, high_contrast=False):
    fig = go.Figure()
    _apply_common_layout_settings(fig, title_text, high_contrast, show_legend=False)
    fig.add_annotation(text="No data available for this visualization.", 
                       showarrow=False, font=dict(size=14, color=COLOR_LIGHT_TEXT if not high_contrast else COLOR_WHITE_TEXT))
    return fig

def plot_key_metrics_summary(compliance, proximity, wellbeing, downtime, high_contrast=False,
                             color_positive=COLOR_POSITIVE_GREEN,
                             color_warning=COLOR_WARNING_AMBER, # Changed from color_neutral
                             color_negative=COLOR_CRITICAL_RED,
                             accent_color=COLOR_ACCENT_INDIGO):
    figs = []
    font_color_gauge = COLOR_WHITE_TEXT if high_contrast else COLOR_LIGHT_TEXT
    gauge_base_bgcolor = "rgba(42, 52, 71, 0.7)" if not high_contrast else "rgba(17,17,17,0.7)" # Darker for HC base

    metrics_config = [
        ("Task Compliance", compliance, 75, accent_color, "%", False), # Target 75%
        ("Collaboration Idx", proximity, 60, accent_color, "%", False), # Target 60%
        ("Well-Being Idx", wellbeing, 70, accent_color, "%", False),   # Target 70%
        ("Total Downtime", downtime, 30, accent_color, " min", True) # Target < 30 min (lower is better)
    ]

    for title, value, target, bar_color_val, suffix, lower_is_better in metrics_config:
        fig = go.Figure()
        
        # Delta colors depend on whether lower is better
        increasing_is_good_color = color_positive
        decreasing_is_good_color = color_negative
        if lower_is_better:
            increasing_is_good_color = color_negative
            decreasing_is_good_color = color_positive
        
        # Gauge step thresholds and colors
        # For "higher is better"
        bad_thresh_high_better = target * 0.8
        warn_thresh_high_better = target
        # For "lower is better"
        good_thresh_low_better = target
        warn_thresh_low_better = target * 1.25 # e.g. 25% over target

        steps_config = []
        if lower_is_better:
            steps_config = [
                {'range': [0, good_thresh_low_better], 'color': color_positive},
                {'range': [good_thresh_low_better, warn_thresh_low_better], 'color': color_warning},
                {'range': [warn_thresh_low_better, max(warn_thresh_low_better * 1.5, value * 1.2 if value else warn_thresh_low_better * 1.5, 10)], 'color': color_negative}
            ]
        else: # Higher is better
            steps_config = [
                {'range': [0, bad_thresh_high_better], 'color': color_negative},
                {'range': [bad_thresh_high_better, warn_thresh_high_better], 'color': color_warning},
                {'range': [warn_thresh_high_better, 100 if suffix == "%" else max(target * 1.5, value * 1.2 if value else target * 1.5, 10)], 'color': color_positive}
            ]
        
        axis_max_val = (100 if suffix == "%" 
                        else max(target * 1.5, value * 1.2 if value is not None and value > 0 else target * 1.5, 10))


        fig.add_trace(go.Indicator(
            mode="gauge+number+delta", value=float(value) if value is not None else 0,
            delta={'reference': float(target), 
                   'increasing': {'color': increasing_is_good_color}, 
                   'decreasing': {'color': decreasing_is_good_color}, 
                   'font': {'size': 12}},
            title={'text': title, 'font': {'size': 12, 'color': font_color_gauge}},
            number={'suffix': suffix, 'font': {'size': 18, 'color': font_color_gauge}},
            gauge={
                'axis': {'range': [0, axis_max_val], 
                         'tickwidth': 1, 'tickcolor': COLOR_NEUTRAL_GRAY if not high_contrast else COLOR_WHITE_TEXT},
                'bar': {'color': bar_color_val, 'thickness': 0.65},
                'bgcolor': gauge_base_bgcolor,
                'borderwidth': 0.5, 'bordercolor': COLOR_NEUTRAL_GRAY,
                'steps': steps_config,
                'threshold': {'line': {'color': font_color_gauge, 'width': 2.5}, 'thickness': 0.8, 'value': target}
            }
        ))
        fig.update_layout(height=180, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color_gauge))
        figs.append(fig)
    return figs


def plot_task_compliance_score(data, disruption_points, forecast_data=None, z_scores=None, high_contrast=False):
    if not data: return _get_no_data_figure("Task Compliance Score Trend", high_contrast)
    
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure()
    x_vals = list(range(len(data)))

    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Compliance',
                             line=dict(color=cat_palette[0], width=2.5),
                             marker=dict(size=5, symbol="circle"), 
                             hovertemplate='Compliance: %{y:.1f}%<extra></extra>'))
    if forecast_data and len(forecast_data) == len(data):
        fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast',
                                 line=dict(color=cat_palette[1], dash='dashdot', width=2),
                                 hovertemplate='Forecast: %{y:.1f}%<extra></extra>'))
    
    disruption_line_color = COLOR_WARNING_AMBER
    disruption_annot_font_color = COLOR_WHITE_TEXT if high_contrast else COLOR_LIGHT_TEXT
    disruption_annot_bgcolor = "rgba(0,0,0,0.5)" if high_contrast else "rgba(40,40,40,0.7)"

    for dp_step in disruption_points:
        if 0 <= dp_step < len(data):
            fig.add_vline(x=dp_step, line=dict(color=disruption_line_color, width=1.5, dash="longdash"),
                          annotation_text="D", annotation_position="top left",
                          annotation=dict(font_size=10, bgcolor=disruption_annot_bgcolor, borderpad=2, textangle=0, font_color=disruption_annot_font_color))
                          
    min_val = min(data) if data else 0
    max_val = max(data) if data else 100
    if forecast_data:
        min_val = min(min_val, min(forecast_data) if forecast_data else min_val)
        max_val = max(max_val, max(forecast_data) if forecast_data else max_val)

    _apply_common_layout_settings(fig, "Task Compliance Score Trend", high_contrast, yaxis_title="Score (%)",
                                  yaxis_range=[max(0, min_val - 15), min(105, max_val + 15)])
    return fig

def plot_collaboration_proximity_index(data, disruption_points, forecast_data=None, high_contrast=False):
    if not data: return _get_no_data_figure("Collaboration Proximity Index Trend", high_contrast)

    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure()
    x_vals = list(range(len(data)))

    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Collab. Index',
                             line=dict(color=cat_palette[1], width=2.5),  # Use a different color from palette
                             marker=dict(size=5, symbol="diamond"),
                             hovertemplate='Collab. Index: %{y:.1f}%<extra></extra>'))
    if forecast_data and len(forecast_data) == len(data):
        fig.add_trace(go.Scatter(x=x_vals, y=forecast_data, mode='lines', name='Forecast',
                                 line=dict(color=cat_palette[2], dash='dashdot', width=2), # Different color
                                 hovertemplate='Forecast: %{y:.1f}%<extra></extra>'))

    disruption_line_color = COLOR_WARNING_AMBER
    disruption_annot_font_color = COLOR_WHITE_TEXT if high_contrast else COLOR_LIGHT_TEXT
    disruption_annot_bgcolor = "rgba(0,0,0,0.5)" if high_contrast else "rgba(40,40,40,0.7)"

    for dp_step in disruption_points:
        if 0 <= dp_step < len(data):
            fig.add_vline(x=dp_step, line=dict(color=disruption_line_color, width=1.5, dash="longdash"),
                          annotation_text="D", annotation_position="bottom right",
                          annotation=dict(font_size=10, bgcolor=disruption_annot_bgcolor, borderpad=2, textangle=0, font_color=disruption_annot_font_color))

    min_val = min(data) if data else 0
    max_val = max(data) if data else 100
    if forecast_data:
        min_val = min(min_val, min(forecast_data) if forecast_data else min_val)
        max_val = max(max_val, max(forecast_data) if forecast_data else max_val)

    _apply_common_layout_settings(fig, "Collaboration Proximity Index Trend", high_contrast, yaxis_title="Index (%)",
                                  yaxis_range=[max(0, min_val - 15), min(105, max_val + 15)])
    return fig

def plot_operational_recovery(recovery_data, productivity_loss_data, high_contrast=False):
    if not recovery_data: return _get_no_data_figure("Operational Resilience", high_contrast)
    
    fig = go.Figure()
    x_vals = list(range(len(recovery_data)))

    fig.add_trace(go.Scatter(x=x_vals, y=recovery_data, mode='lines', name='Op. Recovery',
                             line=dict(color=COLOR_POSITIVE_GREEN, width=2.5, dash='solid'),
                             hovertemplate='Recovery: %{y:.1f}%<extra></extra>'))
    if productivity_loss_data and len(productivity_loss_data) == len(recovery_data):
        fig.add_trace(go.Scatter(x=x_vals, y=productivity_loss_data, mode='lines', name='Prod. Loss',
                                 line=dict(color=COLOR_CRITICAL_RED, dash='dot', width=2.5),
                                 hovertemplate='Prod. Loss: %{y:.1f}%<extra></extra>'))
    _apply_common_layout_settings(fig, "Operational Resilience: Recovery vs. Loss", high_contrast, yaxis_title="Percentage (%)", yaxis_range=[0, 105])
    return fig

def plot_operational_efficiency(efficiency_df, selected_metrics, high_contrast=False):
    if efficiency_df.empty: return _get_no_data_figure("Operational Efficiency (OEE)", high_contrast)
    
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure()
    
    # Define colors and styles for each metric for consistency
    metric_styles = {
        'uptime': {'color': cat_palette[0], 'dash': 'solid', 'width': 1.5},
        'throughput': {'color': cat_palette[1], 'dash': 'solid', 'width': 1.5},
        'quality': {'color': cat_palette[2], 'dash': 'solid', 'width': 1.5},
        'oee': {'color': cat_palette[3], 'dash': 'solid', 'width': 2.5} # OEE thicker
    }
    default_style = {'color': COLOR_NEUTRAL_GRAY, 'dash': 'solid', 'width': 1.5}

    for i, metric in enumerate(selected_metrics):
        if metric in efficiency_df.columns:
            style = metric_styles.get(metric, default_style)
            # If default style is used because metric_styles doesn't have it, pick next color from palette
            if metric not in metric_styles and i < len(cat_palette):
                 style['color'] = cat_palette[i % len(cat_palette)]


            fig.add_trace(go.Scatter(x=efficiency_df.index, y=efficiency_df[metric], mode='lines', name=metric.upper(),
                                     line=dict(color=style['color'], width=style['width'], dash=style['dash']),
                                     hovertemplate=f'{metric.upper()}: %{{y:.1f}}%<extra></extra>'))
    _apply_common_layout_settings(fig, "Overall Equipment Effectiveness (OEE) & Components", high_contrast, yaxis_title="Efficiency Score (%)", yaxis_range=[0, 105])
    return fig

def plot_worker_distribution(team_positions_df, facility_size, config, use_3d=False, selected_step=0, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    df_step = team_positions_df[team_positions_df['step'] == selected_step].copy()
    if df_step.empty: return _get_no_data_figure(f"Worker Distribution (Time: {selected_step*2} min)", high_contrast)

    facility_width, facility_height = facility_size
    zone_names = df_step['zone'].unique()
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    zone_color_map = {zone: cat_palette[i % len(cat_palette)] for i, zone in enumerate(zone_names)}
    
    status_symbols_map = {"working": "circle", "idle": "square", "break":"diamond", "fatigued": "cross", "exhausted": "x-thin", "disrupted":"hourglass"}
    df_step['symbol_plotly'] = df_step['status'].map(status_symbols_map).fillna("circle")

    if use_3d:
        fig = px.scatter_3d(df_step, x='x', y='y', z='z', color='zone', color_discrete_map=zone_color_map,
                            hover_name='worker_id', hover_data={'zone': True, 'status': True, 'x':':.1f', 'y':':.1f', 'z':':.1f'},
                            range_x=[0, facility_width], range_y=[0, facility_height],
                            range_z=[0,max(5, df_step['z'].max() if 'z' in df_step.columns and not df_step['z'].empty else 5)],
                            symbol='symbol_plotly', opacity=0.9, size_max=12)
        fig.update_scenes(aspectmode='data', xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False,
                          xaxis_backgroundcolor="rgba(0,0,0,0)", yaxis_backgroundcolor="rgba(0,0,0,0)", zaxis_backgroundcolor="rgba(0,0,0,0)")
    else:
        fig = px.scatter(df_step, x='x', y='y', color='zone', color_discrete_map=zone_color_map,
                         hover_name='worker_id', hover_data={'zone': True, 'status': True, 'x':':.1f', 'y':':.1f'},
                         range_x=[-5, facility_width + 5], range_y=[-5, facility_height + 5],
                         symbol='symbol_plotly', opacity=0.9, size_max=10)
        # For 2D, explicitly set grid, zeroline based on theme, px.scatter might not inherit fully
        grid_c = COLOR_SUBTLE_GRID_HC if high_contrast else COLOR_SUBTLE_GRID_STD
        fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True, xaxis_gridcolor=grid_c, yaxis_gridcolor=grid_c,
                          xaxis_zeroline=True, yaxis_zeroline=True, xaxis_zerolinecolor=grid_c, yaxis_zerolinecolor=grid_c)

    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height,
                              line=dict(color=COLOR_NEUTRAL_GRAY, width=1.5), fillcolor="rgba(0,0,0,0)", layer="below")]
    annotations = []
    
    font_c = COLOR_WHITE_TEXT if high_contrast else COLOR_LIGHT_TEXT
    ee_point_color = COLOR_INFO_BLUE
    work_area_line_color = cat_palette[len(zone_names) % len(cat_palette)] # pick a distinct color

    if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
        for point in config['ENTRY_EXIT_POINTS']:
            shapes.append(go.layout.Shape(type="circle", x0=point['coords'][0]-1.5, y0=point['coords'][1]-1.5,
                                          x1=point['coords'][0]+1.5, y1=point['coords'][1]+1.5,
                                          fillcolor=ee_point_color, line_color=COLOR_WHITE_TEXT if high_contrast else COLOR_DARK_TEXT_ON_LIGHT_BG_HC, line_width=1, opacity=0.9, layer="above"))
            annotations.append(dict(x=point['coords'][0], y=point['coords'][1]+4, text=point['name'][:2].upper(),
                                    showarrow=False, font=dict(size=9, color=font_c),
                                    borderpad=2, bgcolor="rgba(0,0,0,0.3)" if not high_contrast else "rgba(200,200,200,0.3)"))
    if show_prod_lines and 'WORK_AREAS' in config:
         for area_name, area_details in config['WORK_AREAS'].items():
            if 'coords' in area_details:
                (x0,y0), (x1,y1) = area_details['coords']
                shapes.append(go.layout.Shape(type="rect", x0=min(x0,x1), y0=min(y0,y1), x1=max(x0,x1), y1=max(y0,y1),
                                              line=dict(color=work_area_line_color, dash="dashdot", width=1.5),
                                              fillcolor=f"rgba({int(work_area_line_color[1:3],16)},{int(work_area_line_color[3:5],16)},{int(work_area_line_color[5:7],16)},0.05)", layer="below")) # Transparent version of line color
                annotations.append(dict(x=(x0+x1)/2, y=min(y0,y1)-4, text=area_name, showarrow=False,
                                        font=dict(size=9, color=COLOR_NEUTRAL_GRAY), opacity=0.9, yanchor="top"))
    _apply_common_layout_settings(fig, f"Worker Spatial Distribution (Time: {selected_step*2} min)", high_contrast, yaxis_title="Y (m)", xaxis_title="X (m)", show_legend=True)
    fig.update_layout(shapes=shapes, annotations=annotations, legend_title_text='Zone')
    return fig


def plot_worker_density_heatmap(team_positions_df, facility_size, config, show_entry_exit=True, show_prod_lines=True, high_contrast=False):
    if team_positions_df.empty: return _get_no_data_figure("Aggregated Worker Density Heatmap", high_contrast)

    facility_width, facility_height = facility_size
    heatmap_colorscale = HIGH_CONTRAST_SEQUENTIAL_PLOTLY_SCALES[4] if high_contrast else ACCESSIBLE_SEQUENTIAL_PLOTLY_SCALES[1] # Greys for HC, Cividis for std
    
    fig = go.Figure(go.Histogram2dContour(
        x=team_positions_df['x'], y=team_positions_df['y'],
        colorscale=heatmap_colorscale, reversescale=(heatmap_colorscale=="Greys"), # Reverse Greys for dark-on-light heat
        showscale=True,
        line=dict(width=0.3, color=COLOR_NEUTRAL_GRAY if high_contrast else "#333"), # Color for contour lines
        contours=dict(coloring='heatmap', showlabels=False), # coloring='lines' with color=font_c for HC could be an option
        xbins=dict(start=0, end=facility_width, size=facility_width/max(1, facility_width/20)), # ensure size > 0
        ybins=dict(start=0, end=facility_height, size=facility_height/max(1, facility_height/20)), # ensure size > 0
        colorbar=dict(title='Density', thickness=15, len=0.75, y=0.5, tickfont_size=10, x=1.05,
                      bgcolor="rgba(0,0,0,0.1)", bordercolor=COLOR_NEUTRAL_GRAY,
                      titlefont_color=COLOR_WHITE_TEXT if high_contrast else COLOR_LIGHT_TEXT,
                      tickfont_color=COLOR_WHITE_TEXT if high_contrast else COLOR_LIGHT_TEXT)
    ))
    shapes = [go.layout.Shape(type="rect", x0=0, y0=0, x1=facility_width, y1=facility_height,
                              line=dict(color=COLOR_NEUTRAL_GRAY, width=1.5), layer="below")]
    annotations = [] # Usually not needed for pure heatmap
    
    ee_point_color = COLOR_INFO_BLUE
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    work_area_line_color = cat_palette[0] # Example

    if show_entry_exit and 'ENTRY_EXIT_POINTS' in config:
        for point in config['ENTRY_EXIT_POINTS']:
            shapes.append(go.layout.Shape(type="circle", x0=point['coords'][0]-1, y0=point['coords'][1]-1,
                                          x1=point['coords'][0]+1, y1=point['coords'][1]+1,
                                          fillcolor=ee_point_color, line_color=COLOR_WHITE_TEXT, opacity=0.6, layer="above"))
    if show_prod_lines and 'WORK_AREAS' in config:
         for area_name, area_details in config['WORK_AREAS'].items():
            if 'coords' in area_details:
                (x0,y0), (x1,y1) = area_details['coords']
                shapes.append(go.layout.Shape(type="rect", x0=min(x0,x1), y0=min(y0,y1), x1=max(x0,x1), y1=max(y0,y1),
                                              line=dict(color=work_area_line_color, dash="dot", width=1.5),
                                              fillcolor="rgba(0,0,0,0)", layer="above")) # Increased width
    _apply_common_layout_settings(fig, "Aggregated Worker Density Heatmap", high_contrast, yaxis_title="Y Coordinate (m)", xaxis_title="X Coordinate (m)")
    fig.update_layout(xaxis_range=[0, facility_width], yaxis_range=[0, facility_height], shapes=shapes, annotations=annotations, autosize=True)
    fig.update_xaxes(constrain="domain", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(constrain="domain")
    return fig

def plot_worker_wellbeing(scores, triggers, high_contrast=False):
    if not scores: return _get_no_data_figure("Worker Well-Being Index Trend", high_contrast)

    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure()
    x_vals = list(range(len(scores)))

    fig.add_trace(go.Scatter(x=x_vals, y=scores, mode='lines', name='Well-Being Index',
                             line=dict(color=cat_palette[0], width=2.5),
                             hovertemplate='Well-Being: %{y:.1f}%<extra></extra>'))
    
    avg_wellbeing = np.mean(scores) if scores else None
    if avg_wellbeing is not None:
        fig.add_hline(y=avg_wellbeing, line=dict(color=COLOR_NEUTRAL_GRAY, width=1, dash="dot"),
                      annotation_text=f"Avg: {avg_wellbeing:.1f}%", annotation_position="bottom left",
                      annotation_font_size=10, annotation_font_color=COLOR_NEUTRAL_GRAY)

    trigger_styles = {
        'threshold': {'color': COLOR_CRITICAL_RED, 'symbol': 'x-thin-open', 'name': 'Threshold Alert'},
        'trend': {'color': COLOR_WARNING_AMBER, 'symbol': 'triangle-down-open', 'name': 'Trend Alert'},
        'disruption': {'color': COLOR_INFO_BLUE, 'symbol': 'star-open', 'name': 'Disruption Link'},
        'work_area_general': {'color': cat_palette[1], 'symbol': 'diamond-open', 'name': 'Work Area Alert'}
    }
    default_trigger_style = {'color': COLOR_NEUTRAL_GRAY, 'symbol': 'circle-open', 'name': 'Other Alert'}

    for trigger_type, points in triggers.items():
        flat_points = []
        processed_trigger_type = trigger_type
        if isinstance(points, list): flat_points = points
        elif isinstance(points, dict) and trigger_type == 'work_area':
            all_wa_points = set()
            [all_wa_points.update(p_list) for p_list in points.values() if isinstance(p_list, list)]
            flat_points = list(all_wa_points)
            processed_trigger_type = 'work_area_general'
        
        valid_points = sorted(list(set(p for p in flat_points if isinstance(p, (int, float)) and 0 <= p < len(scores))))
        if valid_points:
            style = trigger_styles.get(processed_trigger_type, default_trigger_style)
            fig.add_trace(go.Scatter(
                x=valid_points, y=[scores[p] for p in valid_points], mode='markers', name=style['name'],
                marker=dict(color=style['color'], size=10, symbol=style['symbol'],
                            line=dict(width=1.5, color=COLOR_WHITE_TEXT if high_contrast else COLOR_DARK_TEXT_ON_LIGHT_BG_HC)), # Ensure marker border is visible
                hovertemplate=f'{style["name"]}: %{{y:.1f}}% at Step %{{x}}<extra></extra>'
            ))
    _apply_common_layout_settings(fig, "Worker Well-Being Index Trend", high_contrast, yaxis_title="Index (%)", yaxis_range=[0, 105])
    return fig

def plot_psychological_safety(data, high_contrast=False):
    if not data: return _get_no_data_figure("Psychological Safety Score Trend", high_contrast)
    
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure()
    x_vals = list(range(len(data)))

    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name='Psych. Safety',
                             line=dict(color=cat_palette[2], width=2.5), # Example color
                             hovertemplate='Psych. Safety: %{y:.1f}%<extra></extra>'))
    avg_safety = np.mean(data) if data else None
    if avg_safety is not None:
        fig.add_hline(y=avg_safety, line=dict(color=COLOR_NEUTRAL_GRAY, width=1, dash="dot"),
                      annotation_text=f"Avg: {avg_safety:.1f}%", annotation_position="bottom left",
                      annotation_font_size=10, annotation_font_color=COLOR_NEUTRAL_GRAY)
    _apply_common_layout_settings(fig, "Psychological Safety Score Trend", high_contrast, yaxis_title="Score (%)", yaxis_range=[0, 105])
    return fig

def plot_downtime_trend(downtime_events_list, interval_threshold, high_contrast=False):
    if not downtime_events_list: return _get_no_data_figure("Downtime per Interval", high_contrast)
    
    fig = go.Figure()
    downtime_durations = [event.get('duration', 0) for event in downtime_events_list]
    x_vals = list(range(len(downtime_durations)))
    
    bar_colors = [COLOR_CRITICAL_RED if d > interval_threshold else COLOR_POSITIVE_GREEN for d in downtime_durations]
    hover_texts = [f"Duration: {event.get('duration', 0):.1f} min<br>Cause: {event.get('cause', 'Unknown')}" for event in downtime_events_list]

    fig.add_trace(go.Bar(x=x_vals, y=downtime_durations, name='Downtime',
                         marker_color=bar_colors, width=0.7,
                         text=hover_texts, hoverinfo='text'))
    
    threshold_line_color = COLOR_WARNING_AMBER
    threshold_annot_font_color = COLOR_DARK_TEXT_ON_LIGHT_BG_HC if high_contrast else COLOR_LIGHT_TEXT
    threshold_annot_bgcolor = "rgba(245, 158, 11, 0.7)" # Amber background

    fig.add_hline(y=interval_threshold, line=dict(color=threshold_line_color, width=1.5, dash="longdash"),
                  annotation_text=f"Alert Thresh: {interval_threshold} min", annotation_position="top right",
                  annotation=dict(font_size=10, bgcolor=threshold_annot_bgcolor, borderpad=2, font_color=threshold_annot_font_color))
    _apply_common_layout_settings(fig, "Downtime per Interval", high_contrast, yaxis_title="Downtime (minutes)")
    max_y_val = max(max(downtime_durations) * 1.15 if downtime_durations else interval_threshold * 1.5, interval_threshold * 1.5, 10)
    fig.update_yaxes(range=[0, max_y_val])
    return fig

def plot_team_cohesion(data, high_contrast=False):
    if not data: return _get_no_data_figure("Team Cohesion Index Trend", high_contrast)
    
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure()
    x_vals = list(range(len(data)))

    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name='Team Cohesion',
                             line=dict(color=cat_palette[3], width=2.5), # Example color
                             hovertemplate='Cohesion: %{y:.1f}%<extra></extra>'))
    avg_cohesion = np.mean(data) if data else None
    if avg_cohesion is not None:
        fig.add_hline(y=avg_cohesion, line=dict(color=COLOR_NEUTRAL_GRAY, width=1, dash="dot"),
                      annotation_text=f"Avg: {avg_cohesion:.1f}%", annotation_position="top left",
                      annotation_font_size=10, annotation_font_color=COLOR_NEUTRAL_GRAY)
    _apply_common_layout_settings(fig, "Team Cohesion Index Trend", high_contrast, yaxis_title="Cohesion Index (%)", yaxis_range=[0, 105])
    return fig

def plot_perceived_workload(data, high_workload_threshold, very_high_workload_threshold, high_contrast=False):
    if not data: return _get_no_data_figure("Perceived Workload Index", high_contrast)
    
    cat_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    fig = go.Figure()
    x_vals = list(range(len(data)))
    
    line_color_main = cat_palette[4] # Example color for the line itself
    marker_colors = [COLOR_CRITICAL_RED if val >= very_high_workload_threshold else
                     COLOR_WARNING_AMBER if val >= high_workload_threshold else
                     COLOR_POSITIVE_GREEN for val in data]

    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines+markers', name='Perceived Workload',
                             line=dict(color=line_color_main, width=2),
                             marker=dict(size=6, color=marker_colors, line=dict(width=0.5, color=COLOR_WHITE_TEXT if high_contrast else COLOR_DARK_TEXT_ON_LIGHT_BG_HC)),
                             hovertemplate='Workload: %{y:.1f}/10<extra></extra>'))
    
    fig.add_hline(y=high_workload_threshold, line=dict(color=COLOR_WARNING_AMBER, width=1.5, dash="dash"),
                  annotation_text=f"High ({high_workload_threshold})", annotation_position="bottom right",
                  annotation_font=dict(color=COLOR_WARNING_AMBER, size=10))
    fig.add_hline(y=very_high_workload_threshold, line=dict(color=COLOR_CRITICAL_RED, width=1.5, dash="dash"),
                  annotation_text=f"Very High ({very_high_workload_threshold})", annotation_position="top right",
                  annotation_font=dict(color=COLOR_CRITICAL_RED, size=10))
    _apply_common_layout_settings(fig, "Perceived Workload Index (0-10 Scale)", high_contrast, yaxis_title="Workload Index", yaxis_range=[0, 10.5])
    return fig

def plot_downtime_causes_pie(downtime_events_list, high_contrast=False):
    if not downtime_events_list: return _get_no_data_figure("Downtime Distribution by Cause", high_contrast)

    causes_summary = {}
    total_downtime_duration_for_pie = 0
    for event in downtime_events_list:
        duration = event.get('duration', 0)
        cause = event.get('cause', 'Unknown')
        if duration > 0 and cause != "None" and cause != "Unknown":
            if "Equip.Fail" in cause and cause != "Equipment Failure": cause = "Equipment Failure"
            if "HumanError" in cause and cause != "Human Error": cause = "Human Error"
            causes_summary[cause] = causes_summary.get(cause, 0) + duration
            total_downtime_duration_for_pie += duration
    
    if not causes_summary:
        return _get_no_data_figure("Downtime by Cause (No Categorized Downtime)", high_contrast)
        
    labels = list(causes_summary.keys())
    values = list(causes_summary.values())
    num_causes = len(labels)
    
    pie_color_palette = HIGH_CONTRAST_CATEGORICAL_PALETTE if high_contrast else ACCESSIBLE_CATEGORICAL_PALETTE
    pie_colors = [pie_color_palette[i % len(pie_color_palette)] for i in range(num_causes)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.4,
        pull=[0.02]*num_causes, marker_colors=pie_colors,
        textinfo='label+percent', insidetextorientation='auto',
        hovertemplate="<b>Cause:</b> %{label}<br><b>Duration:</b> %{value:.1f} min<br><b>Share:</b> %{percent}<extra></extra>",
        sort=True, direction='descending'
    )])
    
    slice_border_color = COLOR_DARK_TEXT_ON_LIGHT_BG_HC if high_contrast else COLOR_WHITE_TEXT # Ensure border contrasts with slices
    fig.update_traces(textfont_size=10, 
                      textfont_color=COLOR_DARK_TEXT_ON_LIGHT_BG_HC, # Dark text on potentially light slices
                      marker=dict(line=dict(color=slice_border_color, width=1))) 

    _apply_common_layout_settings(fig, f"Downtime Distribution by Cause (Total: {total_downtime_duration_for_pie:.0f} min)",
                                  high_contrast, show_legend=False if num_causes > 7 else True) # Auto-hide legend if too many items
    return fig
