# main.py
import logging
import streamlit as st
import pandas as pd
import numpy as np
from config import DEFAULT_CONFIG, validate_config
from visualizations import (
    plot_key_metrics_summary, plot_task_compliance_score, plot_collaboration_proximity_index,
    plot_operational_recovery, plot_operational_efficiency, plot_worker_distribution,
    plot_worker_density_heatmap, plot_worker_wellbeing, plot_psychological_safety,
    plot_downtime_trend, plot_team_cohesion, plot_perceived_workload,
    plot_downtime_causes_pie
)
from simulation import simulate_workplace_operations
from utils import save_simulation_data, load_simulation_data, generate_pdf_report

logger = logging.getLogger(__name__)
if not logger.handlers: # Ensure logger is configured only once
    logging.basicConfig(level=logging.INFO, # INFO for production, DEBUG for development
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
                        filename='dashboard.log',
                        filemode='a') # Append to log file
logger.info("Main.py: Startup. Imports parsed, logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.3.2\nInsights for operational excellence & psychosocial well-being."})

# CSS Color constants (from your original CSS, also used by visualizations.py)
COLOR_CRITICAL_RED_CSS = "#E53E3E"; COLOR_WARNING_AMBER_CSS = "#F59E0B"; COLOR_POSITIVE_GREEN_CSS = "#10B981"; COLOR_INFO_BLUE_CSS = "#3B82F6"; COLOR_ACCENT_INDIGO_CSS = "#4F46E5"
# Themed color for Streamlit dividers, visually similar to ACCENT_INDIGO
THEMED_DIVIDER_COLOR = "violet" 

# --- UTILITY FUNCTIONS (main.py specific or simple helpers) ---
def safe_get(data_dict, path_str, default_val=None):
    current = data_dict
    is_list_like_path = False
    if isinstance(path_str, str):
        is_list_like_path = path_str.endswith(('.data', '.scores', '.triggers', '_log', 'events_list'))
    
    if default_val is None: default_return = [] if is_list_like_path else None
    else: default_return = default_val

    if not isinstance(path_str, str): return default_return # Path must be string
    if not isinstance(data_dict, dict):
        if path_str: logger.debug(f"safe_get: data_dict not dict for path '{path_str}'. Type: {type(data_dict)}.")
        return default_return
    try:
        keys = path_str.split('.')
        for key in keys:
            if isinstance(current, dict): current = current.get(key)
            elif isinstance(current, (list, pd.Series)) and key.isdigit():
                idx = int(key)
                current = current[idx] if idx < len(current) else None
            else: current = None; break # Path broken
        
        if current is None: # If any part of the path led to None
            # If the original intent was a list-like structure based on path name, return [] if default_val wasn't set
            is_list_like_final_key = keys and keys[-1] in ['data', 'scores', 'triggers', '_log', 'events_list']
            return [] if default_val is None and is_list_like_final_key else default_val
        return current
    except (ValueError, IndexError, TypeError) as e: # Catch errors during access
        logger.debug(f"safe_get failed for path '{path_str}': {e}. Returning default '{default_return}'.")
        return default_return

def safe_stat(data_list, stat_func, default_val=0.0):
    if not isinstance(data_list, (list, np.ndarray, pd.Series)): return default_val
    if isinstance(data_list, pd.Series): # Handle Pandas Series separately for robust NaN handling
        valid_data = pd.to_numeric(data_list, errors='coerce').dropna().tolist()
    else: # For lists or NumPy arrays
        valid_data = []
        for x in data_list:
            if x is None or (isinstance(x, float) and np.isnan(x)): continue # Skip None and NaN
            try: valid_data.append(float(x)) # Try to convert to float
            except (ValueError, TypeError): pass # Ignore elements that can't be converted
    if not valid_data: return default_val # No valid numeric data
    try:
        result = stat_func(np.array(valid_data)) # Perform statistical function
        return default_val if isinstance(result, (float, np.floating)) and np.isnan(result) else result
    except Exception: return default_val # Catch any error during stat_func execution


def get_actionable_insights(sim_data, current_config_dict):
    insights = []
    if not sim_data or not isinstance(sim_data, dict): 
        logger.warning("get_actionable_insights: sim_data is None or not a dict.", extra={'user_action': 'Insights - Invalid Input'})
        return insights
    
    sim_cfg_params_insights = sim_data.get('config_params', {}) # Config from the simulation run

    # Helper to get config values, prioritizing sim_cfg_params_insights
    def _get_insight_cfg(key, default):
        return sim_cfg_params_insights.get(key, current_config_dict.get(key, default))

    # Task Compliance
    compliance_data = safe_get(sim_data, 'task_compliance.data', [])
    target_compliance = float(_get_insight_cfg('TARGET_COMPLIANCE', 75.0))
    compliance_avg = safe_stat(compliance_data, np.mean, 0.0) # Default to 0 if no data
    if compliance_data and compliance_avg < target_compliance * 0.9: # Check if data was present
        insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Avg. Task Compliance ({compliance_avg:.1f}%) critically below target ({target_compliance:.0f}%). Review disruptions, complexities, training."})
    elif compliance_data and compliance_avg < target_compliance:
        insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Avg. Task Compliance at {compliance_avg:.1f}%. Review areas with lowest compliance."})

    # Worker Wellbeing
    wellbeing_scores = safe_get(sim_data, 'worker_wellbeing.scores', [])
    target_wellbeing = float(_get_insight_cfg('TARGET_WELLBEING', 70.0))
    wellbeing_avg = safe_stat(wellbeing_scores, np.mean, 0.0)
    wb_crit_factor = float(_get_insight_cfg('WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET', 0.85))
    if wellbeing_scores and wellbeing_avg < target_wellbeing * wb_crit_factor:
        insights.append({"type": "critical", "title": "Critical Worker Well-being", "text": f"Avg. Well-being ({wellbeing_avg:.1f}%) critically low (target {target_wellbeing:.0f}%). Urgent review needed."})
    
    threshold_triggers = safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])
    if wellbeing_scores and len(threshold_triggers) > max(2, len(wellbeing_scores) * 0.1):
        insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers)} low well-being instances. Investigate triggers."})

    # Total Downtime
    downtime_event_log = safe_get(sim_data, 'downtime_events_log', [])
    downtime_durations = [event.get('duration', 0.0) for event in downtime_event_log if isinstance(event, dict)]
    total_downtime = sum(downtime_durations)

    shift_mins_insights = float(sim_cfg_params_insights.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
    dt_thresh_percent_insights = float(_get_insight_cfg('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT', 0.05))
    dt_thresh_total_shift_abs_insights = shift_mins_insights * dt_thresh_percent_insights
    if total_downtime > dt_thresh_total_shift_abs_insights:
        insights.append({"type": "critical", "title": "Excessive Total Downtime", "text": f"Total downtime {total_downtime:.0f} min, exceeds guideline of {dt_thresh_total_shift_abs_insights:.0f} min ({dt_thresh_percent_insights*100:.0f}% of shift). Analyze causes."})

    # Psychological Safety
    psych_safety_scores = safe_get(sim_data, 'psychological_safety', [])
    target_psych_safety = float(_get_insight_cfg('TARGET_PSYCH_SAFETY', 70.0))
    psych_safety_avg = safe_stat(psych_safety_scores, np.mean, 0.0)
    if psych_safety_scores and psych_safety_avg < target_psych_safety * 0.9:
        insights.append({"type": "warning", "title": "Low Psychological Safety", "text": f"Avg. Psych. Safety ({psych_safety_avg:.1f}%) below target ({target_psych_safety:.0f}%). Build trust."})

    # Team Cohesion
    cohesion_scores = safe_get(sim_data, 'worker_wellbeing.team_cohesion_scores', [])
    target_cohesion = float(_get_insight_cfg('TARGET_TEAM_COHESION', 70.0))
    cohesion_avg = safe_stat(cohesion_scores, np.mean, 0.0)
    if cohesion_scores and cohesion_avg < target_cohesion * 0.9:
        insights.append({"type": "warning", "title": "Low Team Cohesion", "text": f"Avg. Team Cohesion ({cohesion_avg:.1f}%) below desired. Consider team-building."})
    
    # Perceived Workload
    workload_scores = safe_get(sim_data, 'worker_wellbeing.perceived_workload_scores', []) # Scale 0-10
    target_workload = float(_get_insight_cfg('TARGET_PERCEIVED_WORKLOAD', 6.5))
    workload_avg = safe_stat(workload_scores, np.mean, target_workload / 2) 
    workload_very_high_thresh = float(_get_insight_cfg('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5))
    workload_high_thresh = float(_get_insight_cfg('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5))

    if workload_scores:
        if workload_avg > workload_very_high_thresh:
            insights.append({"type": "critical", "title": "Very High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) critically high. Immediate review required."})
        elif workload_avg > workload_high_thresh:
            insights.append({"type": "warning", "title": "High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) exceeds high threshold. Monitor."})
        elif workload_avg > target_workload:
            insights.append({"type": "info", "title": "Elevated Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) above target ({target_workload:.1f}/10). Consider adjustments."})
    
    # Spatial Insights
    team_pos_df = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
    work_areas_effective_insight = sim_cfg_params_insights.get('WORK_AREAS_EFFECTIVE', current_config_dict.get('WORK_AREAS', {}))
    if not team_pos_df.empty and isinstance(work_areas_effective_insight, dict) and \
       all(col in team_pos_df.columns for col in ['zone', 'worker_id', 'step']):
        for zone_name, zone_details in work_areas_effective_insight.items():
            if not isinstance(zone_details, dict): continue
            workers_in_zone_series = team_pos_df[team_pos_df['zone'] == zone_name].groupby('step')['worker_id'].nunique()
            if not workers_in_zone_series.empty:
                workers_in_zone_avg = workers_in_zone_series.mean()
                intended_workers = zone_details.get('workers', 0)
                coords = zone_details.get('coords'); area_m2 = 1.0
                if coords and isinstance(coords, list) and len(coords) == 2 and \
                   all(isinstance(p, tuple) and len(p)==2 and all(isinstance(c, (int,float)) for c_tuple_val in coords for c in c_tuple_val) for p in coords):
                    (x0,y0), (x1,y1) = coords[0], coords[1]; area_m2 = abs(x1-x0) * abs(y1-y0)
                if abs(area_m2) < 1e-6: area_m2 = 1.0 
                
                avg_density = workers_in_zone_avg / area_m2
                intended_density = (intended_workers / area_m2) if intended_workers > 0 else 0

                if intended_density > 0 and avg_density > intended_density * 1.8: 
                     insights.append({"type": "warning", "title": f"Potential Overcrowding: '{zone_name}'", "text": f"Avg. density ({avg_density:.2f} w/m²) significantly > intended ({intended_density:.2f} w/m²). Review layout/paths."})
                elif intended_workers > 0 and workers_in_zone_avg < intended_workers * 0.4 and not zone_details.get("is_rest_area", False):
                     insights.append({"type": "info", "title": f"Potential Underutilization: '{zone_name}'", "text": f"Avg. workers ({workers_in_zone_avg:.1f}) <40% of assigned ({intended_workers}). Check allocation."})

    # Holistic Performance Check
    if all(s is not None for s in [compliance_data, wellbeing_scores, psych_safety_scores]) and \
       compliance_avg > target_compliance * 1.05 and \
       wellbeing_avg > target_wellbeing * 1.05 and \
       total_downtime < dt_thresh_total_shift_abs_insights * 0.5 and \
       psych_safety_avg > target_psych_safety * 1.05:
        insights.append({"type": "positive", "title": "Holistically Excellent Performance", "text": "Key operational and psychosocial metrics significantly exceed targets. Well-balanced and high-performing!"})
    
    initiative = sim_cfg_params_insights.get('TEAM_INITIATIVE', 'Standard Operations') 
    if initiative != "Standard Operations":
        insights.append({"type": "info", "title": f"Initiative Active: '{initiative}'", "text": f"The '{initiative}' initiative was simulated. Compare metrics to a 'Standard Operations' baseline run."})
    
    logger.info(f"get_actionable_insights: Generated {len(insights)} insights.", extra={'user_action': 'Actionable Insights - End'})
    return insights
    def aggregate_downtime_by_step(raw_downtime_event_log, num_total_steps_agg):
    downtime_per_step_agg = [0.0] * num_total_steps_agg
    if not isinstance(raw_downtime_event_log, list):
        logger.warning("aggregate_downtime_by_step: input is not a list.")
        return downtime_per_step_agg

    for event in raw_downtime_event_log:
        if not isinstance(event, dict): continue
        step, duration = event.get('step'), event.get('duration', 0.0)
        if isinstance(step, int) and 0 <= step < num_total_steps_agg and isinstance(duration, (int, float)) and duration > 0:
            downtime_per_step_agg[step] += float(duration)
    return downtime_per_step_agg

def _prepare_timeseries_for_export(raw_data, num_total_steps, default_val=np.nan):
    if not isinstance(raw_data, list): return [default_val] * num_total_steps
    return (raw_data + [default_val] * num_total_steps)[:num_total_steps]

def _slice_dataframe_by_step_indices(df, start_idx, end_idx):
    if not isinstance(df, pd.DataFrame) or df.empty: return pd.DataFrame()
    if isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1:
        safe_start, safe_end = max(0, start_idx), min(len(df), end_idx)
        return df.iloc[safe_start:safe_end] if safe_start < safe_end else pd.DataFrame()
    if 'step' in df.columns: return df[(df['step'] >= start_idx) & (df['step'] < end_idx)]
    if df.index.name == 'step' or df.index.is_numeric(): return df[(df.index >= start_idx) & (df.index < end_idx)]
    logger.warning(f"Could not slice DataFrame. Columns: {df.columns}, Index: {df.index.name}")
    return pd.DataFrame()

# --- CSS STYLES ---
st.markdown(f"""
    <style>
        .main {{ background-color: #121828; color: #EAEAEA; font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif; padding: 2rem; }}
        h1 {{ font-size: 2.4rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; text-align: center; margin-bottom: 2rem; color: #FFFFFF; }}
        div[data-testid="stTabs"] section[role="tabpanel"] > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div > h2 {{ 
            font-size: 1.75rem !important; font-weight: 600 !important; line-height: 1.3 !important; margin: 1.2rem 0 1rem 0 !important; 
            color: #D1D5DB !important; border-bottom: 2px solid {COLOR_ACCENT_INDIGO_CSS} !important; padding-bottom: 0.6rem !important; text-align: left !important;
        }}
        div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] .stSubheader {{ 
            font-size: 1.3rem !important; font-weight: 500 !important; line-height: 1.4 !important; margin-top: 1.8rem !important; 
            margin-bottom: 0.8rem !important; color: #C0C0C0 !important; border-bottom: 1px solid #4A5568 !important; 
            padding-bottom: 0.3rem !important; text-align: left !important;
        }}
        div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h5 {{
            font-size: 1.0rem !important; font-weight: 600 !important; line-height: 1.3 !important;
            margin: 1.5rem 0 0.5rem 0 !important; color: #C8C8C8 !important; text-align: left !important;
        }}
        div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h6 {{
            font-size: 0.95rem !important; font-weight: 500 !important; line-height: 1.3 !important;
            margin-top: 1rem !important; margin-bottom: 0.5rem !important; color: #B0B0B0 !important; text-align: left;
        }}
        [data-testid="stSidebar"] h2 {{ 
            font-size: 1.4rem !important; color: #EAEAEA !important; margin-top: 1.5rem !important; margin-bottom: 0.5rem !important;
            padding-bottom: 0.3rem !important; border-bottom: 1px solid #4A5568 !important;
        }}
        [data-testid="stSidebar"] h3 {{ 
            font-size: 1.1rem !important; text-align: center !important; margin-bottom: 1.2rem !important; color: #A0A0A0 !important; 
            border-bottom: none !important; 
        }}
        [data-testid="stSidebar"] div[data-testid="stExpander"] h5 {{ /* Parameter Group Titles like "Schedule Shift Events" */
            color: #E0E0E0 !important; text-align: left; font-size: 1.0rem !important; 
            font-weight: 600 !important; margin-top: 0.8rem !important; margin-bottom: 0.4rem !important; 
        }}
        [data-testid="stSidebar"] div[data-testid="stExpander"] h6 {{ /* Sub-titles like "Current Scheduled Events" */
            color: #D1D5DB !important; text-align: left; font-size: 0.9rem !important;
            font-weight: 600 !important; margin-top: 1rem !important; margin-bottom: 0.3rem !important;
        }}
        [data-testid="stSidebar"] .stMarkdownContainer > p, [data-testid="stSidebar"] .stCaption {{ 
             color: #B0B0B0 !important; font-size: 0.85rem !important;
             line-height: 1.3 !important; margin-top: 0.2rem !important; margin-bottom: 0.5rem !important;
        }}
        .stButton>button {{ background-color: {COLOR_ACCENT_INDIGO_CSS}; color: #FFFFFF; border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.95rem; font-weight: 500; transition: all 0.2s ease-in-out; border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stButton>button:hover, .stButton>button:focus {{ background-color: #6366F1; transform: translateY(-1px); box-shadow: 0 3px 7px rgba(0,0,0,0.2); outline: none; }}
        .stButton>button:disabled {{ background-color: #374151; color: #9CA3AF; cursor: not-allowed; box-shadow: none; }}
        [data-testid="stSidebar"] div[data-testid*="stWidgetLabel"] label p, 
        [data-testid="stSidebar"] label[data-baseweb="checkbox"] span, 
        [data-testid="stSidebar"] .stSelectbox > label, 
        [data-testid="stSidebar"] .stMultiSelect > label {{
            color: #E0E0E0 !important; font-weight: 600 !important;
            font-size: 0.92rem !important; padding-bottom: 3px !important; display: block !important; 
        }}
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"], 
        [data-testid="stSidebar"] .stNumberInput div input, 
        [data-testid="stSidebar"] .stTextInput div input,
        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {{ 
            background-color: #2D3748 !important; color: #EAEAEA !important; border-radius: 6px !important; 
            padding: 0.4rem 0.5rem !important; margin-bottom: 0.6rem !important; font-size: 0.9rem !important; 
            border: 1px solid #4A5568 !important; height: auto !important; 
        }}
        [data-testid="stSidebar"] .stNumberInput button {{ background-color: #374151 !important; color: #EAEAEA !important; border: 1px solid #4A5568 !important; }}
        [data-testid="stSidebar"] .stNumberInput button:hover {{ background-color: #4A5568 !important; }}
        [data-testid="stSidebar"] {{ background-color: #1F2937; color: #EAEAEA; padding: 1.5rem; border-right: 1px solid #374151; font-size: 0.95rem; }}
        [data-testid="stSidebar"] .stButton>button {{ background-color: {COLOR_POSITIVE_GREEN_CSS}; width: 100%; margin-bottom: 0.5rem; }}
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus {{ background-color: #6EE7B7; }}
        [data-testid="stSidebar"] .stButton button[kind="primary"] {{ background-color: {COLOR_ACCENT_INDIGO_CSS} !important; }}
        [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {{ background-color: #6366F1 !important; }}
        .stMetric {{ background-color: #1F2937; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #374151; display: flex; flex-direction: column; align-items: flex-start;}}
        .stMetric > div[data-testid="stMetricLabel"] {{ font-size: 1.0rem !important; color: #B0B0B0 !important; font-weight: 600 !important; margin-bottom: 0.3rem !important; }}
        .stMetric div[data-testid="stMetricValue"] {{ font-size: 2.2rem !important; color: #FFFFFF !important; font-weight: 700 !important; line-height: 1.1 !important; }} 
        .stMetric div[data-testid="stMetricDelta"] {{ font-size: 0.9rem !important; font-weight: 500 !important; padding-top: 0.1rem !important; }} 
        .stExpander {{ background-color: #1F2937; border-radius: 8px; margin: 1rem 0; border: 1px solid #374151; }}
        .stExpander header {{ font-size: 1rem; font-weight: 500; color: #E0E0E0; padding: 0.5rem 1rem; }}
        .stTabs [data-baseweb="tab-list"] {{ background-color: #1F2937; border-radius: 8px; padding: 0.5rem; display: flex; justify-content: center; gap: 0.5rem; border-bottom: 2px solid #374151;}}
        .stTabs [data-baseweb="tab"] {{ color: #D1D5DB; padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 500; font-size: 0.95rem; transition: all 0.2s ease-in-out; border: none; border-bottom: 2px solid transparent; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{ background-color: transparent; color: {COLOR_ACCENT_INDIGO_CSS}; border-bottom: 2px solid {COLOR_ACCENT_INDIGO_CSS}; font-weight:600; }}
        .stTabs [data-baseweb="tab"]:hover {{ background-color: #374151; color: #FFFFFF; }}
        .stPlotlyChart {{ border-radius: 6px; }} 
        .stDataFrame {{ border-radius: 8px; font-size: 0.875rem; border: 1px solid #374151; }}
        .stDataFrame thead th {{ background-color: #293344; color: #EAEAEA; font-weight: 600; }}
        .stDataFrame tbody tr:nth-child(even) {{ background-color: #222C3D; }}
        .stDataFrame tbody tr:hover {{ background-color: #374151; }}
        @media (max-width: 768px) {{ 
            .main {{ padding: 1rem; }} h1 {{ font-size: 1.8rem; }} 
            div[data-testid="stTabs"] section[role="tabpanel"] > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div > h2 {{ font-size: 1.4rem !important; }} 
            div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] .stSubheader {{ font-size: 1.1rem !important; }} 
            .stPlotlyChart {{ min-height: 300px !important; }} 
            .stTabs [data-baseweb="tab"] {{ padding: 0.5rem 0.8rem; font-size: 0.85rem; }} 
        }}
        .spinner {{ display: flex; justify-content: center; align-items: center; height: 100px; }}
        .spinner::after {{ content: ''; width: 40px; height: 40px; border: 4px solid #4A5568; border-top: 4px solid {COLOR_ACCENT_INDIGO_CSS}; border-radius: 50%; animation: spin 0.8s linear infinite; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .onboarding-modal {{ background-color: #1F2937; border: 1px solid #374151; border-radius: 8px; padding: 1.5rem; max-width: 550px; margin: 2rem auto; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }}
        .onboarding-modal h3 {{ color: #EAEAEA; margin-bottom: 1rem; text-align: center; }}
        .onboarding-modal p, .onboarding-modal ul {{ color: #D1D5DB; line-height: 1.6; margin-bottom: 1rem; font-size: 0.9rem; }}
        .onboarding-modal ul {{ list-style-position: inside; padding-left: 0.5rem; }}
        .alert-critical {{ border-left: 5px solid {COLOR_CRITICAL_RED_CSS}; background-color: rgba(229, 62, 62, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }} 
        .alert-warning {{ border-left: 5px solid {COLOR_WARNING_AMBER_CSS}; background-color: rgba(245, 158, 11, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }}
        .alert-positive {{ border-left: 5px solid {COLOR_POSITIVE_GREEN_CSS}; background-color: rgba(16, 185, 129, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }}
        .alert-info {{ border-left: 5px solid {COLOR_INFO_BLUE_CSS}; background-color: rgba(59, 130, 246, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }}
        .insight-title {{ font-weight: 600; color: #EAEAEA; margin-bottom: 0.25rem;}}
        .insight-text {{ font-size: 0.9rem; color: #D1D5DB;}}
        .event-item {{padding: 0.3rem 0.5rem; margin-bottom: 0.3rem; background-color: #2a3447; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;}}
        .event-text {{font-size: 0.85rem;}}
        .remove-event-btn button {{background-color: {COLOR_CRITICAL_RED_CSS} !important; color: white !important; padding: 0.1rem 0.4rem !important; font-size: 0.75rem !important; line-height: 1 !important; border-radius: 3px !important; min-height: auto !important; margin-left: 0.5rem !important;}}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR RENDERING ---
def render_settings_sidebar():
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem; color: #A0A0A0;'>Workplace Optimizer</h3>", unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")
        with st.expander("🧪 Simulation Parameters", expanded=True):
            st.number_input("Team Size", min_value=1, max_value=200, key="sb_team_size_num", step=1,
                            help="Adjust the number of workers in the simulated shift.")
            st.number_input("Shift Duration (min)", min_value=60, max_value=7200, key="sb_shift_duration_num", step=10,
                            help="Set the total length of the simulated work shift in minutes.")
            
            current_shift_duration_sb = st.session_state.sb_shift_duration_num
            mpi_sb = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)

            st.markdown("---"); st.markdown("<h5>🗓️ Schedule Shift Events</h5>", unsafe_allow_html=True)
            st.caption("Define disruptions, breaks, etc. Times are from shift start.")
            
            event_types_sb = ["Major Disruption", "Minor Disruption", "Scheduled Break", "Short Pause", "Team Meeting", "Maintenance", "Custom Event"]
            with st.container():
                st.session_state.form_event_type = st.selectbox("Event Type", event_types_sb, 
                    index=event_types_sb.index(st.session_state.form_event_type) if st.session_state.form_event_type in event_types_sb else 0, 
                    key="widget_form_event_type_selector")
                
                col1_form_sb, col2_form_sb = st.columns(2)
                st.session_state.form_event_start = col1_form_sb.number_input("Start (min)", min_value=0, 
                    max_value=max(0, current_shift_duration_sb - mpi_sb), step=mpi_sb, key="widget_form_event_start_input")
                st.session_state.form_event_duration = col2_form_sb.number_input("Duration (min)", min_value=mpi_sb, 
                    max_value=current_shift_duration_sb, step=mpi_sb, key="widget_form_event_duration_input")

            if st.button("➕ Add Event", key="sb_add_event_button_main", use_container_width=True):
                start_val_sb_add = st.session_state.form_event_start
                duration_val_sb_add = st.session_state.form_event_duration
                type_val_sb_add = st.session_state.form_event_type

                if start_val_sb_add + duration_val_sb_add > current_shift_duration_sb:
                    st.warning("Event end time exceeds shift duration.")
                elif start_val_sb_add < 0 : st.warning("Event start time cannot be negative.")
                elif duration_val_sb_add < mpi_sb: st.warning(f"Event duration must be at least {mpi_sb} minute(s).")
                else:
                    st.session_state.sb_scheduled_events_list.append({
                        "Event Type": type_val_sb_add,
                        "Start Time (min)": start_val_sb_add, 
                        "Duration (min)": duration_val_sb_add,
                    })
                    st.session_state.sb_scheduled_events_list.sort(key=lambda x: x.get("Start Time (min)", 0))
                    st.session_state.form_event_start = 0 
                    st.session_state.form_event_duration = max(mpi_sb, 10)
                    st.rerun()

            st.markdown("<h6>Current Scheduled Events:</h6>", unsafe_allow_html=True) 
            if not st.session_state.sb_scheduled_events_list:
                st.caption("No events scheduled yet.")
            else:
                with st.container(height=200):
                    for i_ev_disp_sb, event_disp_sb in enumerate(st.session_state.sb_scheduled_events_list):
                        ev_col1_sb, ev_col2_sb = st.columns([0.85,0.15])
                        ev_col1_sb.markdown(f"<div class='event-item'><span><b>{event_disp_sb.get('Event Type','N/A')}</b> at {event_disp_sb.get('Start Time (min)','N/A')}min ({event_disp_sb.get('Duration (min)','N/A')}min)</span></div>", unsafe_allow_html=True)
                        if ev_col2_sb.button("✖", key=f"remove_event_button_main_{i_ev_disp_sb}", help="Remove this event", type="secondary", use_container_width=True):
                            st.session_state.sb_scheduled_events_list.pop(i_ev_disp_sb); st.rerun()
            
            if st.session_state.sb_scheduled_events_list:
                if st.button("Clear All Events", key="sb_clear_all_events_button_main", type="secondary", use_container_width=True):
                    st.session_state.sb_scheduled_events_list = []; st.rerun()
            
            st.markdown("---") 
            team_initiative_options_sb = ["Standard Operations", "More frequent breaks", "Team recognition", "Increased Autonomy"]
            st.selectbox("Operational Initiative", team_initiative_options_sb, key="sb_team_initiative_selectbox", 
                         help="Apply an operational strategy to observe its impact on metrics.")
            
            run_simulation_button_sidebar = st.button("🚀 Run Simulation", key="sb_run_simulation_main_button", type="primary", use_container_width=True)
        
        with st.expander("🎨 Visualization Options"):
            st.checkbox("High Contrast Plots", key="sb_high_contrast_checkbox", help="Applies a high-contrast color theme to all charts.")
            st.checkbox("Enable 3D Worker View", key="sb_use_3d_distribution_checkbox", help="Renders worker positions in a 3D scatter plot.")
            st.checkbox("Show Debug Info", key="sb_debug_mode_checkbox", help="Display additional debug information.")
        
        with st.expander("💾 Data Management & Export"):
            load_data_button_sidebar = st.button("🔄 Load Previous Simulation", key="sb_load_data_main_button", use_container_width=True)
            can_export_data_sidebar = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            
            if st.button("📄 Download Report (.tex)", key="sb_pdf_download_button_main", disabled=not can_export_data_sidebar, use_container_width=True, help="Generates a LaTeX (.tex) file summarizing the simulation. Requires LaTeX to compile."):
                if can_export_data_sidebar:
                    try:
                        sim_res_pdf_sb = st.session_state.simulation_results
                        sim_cfg_pdf_export_sb = sim_res_pdf_sb.get('config_params', {})
                        num_steps_pdf_export_sb = sim_cfg_pdf_export_sb.get('SHIFT_DURATION_INTERVALS', 0)
                        mpi_pdf_export_sb = sim_cfg_pdf_export_sb.get('MINUTES_PER_INTERVAL', DEFAULT_CONFIG["MINUTES_PER_INTERVAL"])

                        if num_steps_pdf_export_sb == 0: st.warning("⚠️ No simulation data (0 steps) for report."); raise SystemExit
                        
                        pdf_data_dict_sb = {'step': list(range(num_steps_pdf_export_sb)), 'time_minutes': [i * mpi_pdf_export_sb for i in range(num_steps_pdf_export_sb)]}
                        export_metrics_map_pdf_sb = {
                            'task_compliance.data': 'Task Compliance (%)', 'collaboration_metric.data': 'Collaboration Metric (%)',
                            'operational_recovery': 'Operational Recovery (%)', 'worker_wellbeing.scores': 'Worker Wellbeing Index',
                            'psychological_safety': 'Psychological Safety Score', 'productivity_loss': 'Productivity Loss (%)',
                            'task_completion_rate': 'Task Completion Rate (%)',
                            'worker_wellbeing.team_cohesion_scores': 'Team Cohesion Score',
                            'worker_wellbeing.perceived_workload_scores': 'Perceived Workload (0-10)' }
                        for path_pdf_sb, col_name_pdf_sb in export_metrics_map_pdf_sb.items():
                            pdf_data_dict_sb[col_name_pdf_sb] = _prepare_timeseries_for_export(safe_get(sim_res_pdf_sb, path_pdf_sb, []), num_steps_pdf_export_sb)
                        
                        raw_downtime_log_pdf_sb = safe_get(sim_res_pdf_sb, 'downtime_events_log', [])
                        pdf_data_dict_sb['Downtime (min/interval)'] = aggregate_downtime_by_step(raw_downtime_log_pdf_sb, num_steps_pdf_export_sb)

                        df_for_report_sb = pd.DataFrame(pdf_data_dict_sb)
                        generate_pdf_report(df_for_report_sb, sim_cfg_pdf_export_sb)
                        st.success("✅ LaTeX report 'workplace_report.tex' generated.")
                    except SystemExit: pass
                    except Exception as e_pdf_sb: 
                        logger.error(f"PDF Generation Error: {e_pdf_sb}", exc_info=True, extra={'user_action': 'PDF Generation Error'})
                        st.error(f"❌ PDF Generation Error: {e_pdf_sb}")
            
            if can_export_data_sidebar:
                sim_res_csv_exp_sb = st.session_state.simulation_results
                sim_cfg_csv_exp_sb = sim_res_csv_exp_sb.get('config_params', {})
                num_steps_csv_exp_sb = sim_cfg_csv_exp_sb.get('SHIFT_DURATION_INTERVALS', 0)
                mpi_csv_exp_sb = sim_cfg_csv_exp_sb.get('MINUTES_PER_INTERVAL', DEFAULT_CONFIG["MINUTES_PER_INTERVAL"])

                if num_steps_csv_exp_sb > 0:
                    csv_data_dict_sb = {'step': list(range(num_steps_csv_exp_sb)), 'time_minutes': [i * mpi_csv_exp_sb for i in range(num_steps_csv_exp_sb)]}
                    for path_csv_sb, col_name_csv_sb in export_metrics_map_pdf_sb.items(): # Reusing PDF map
                        csv_col_name = col_name_csv_sb.replace(' (%)','_percent').replace(' (0-10)','_0_10').replace(' ','_')
                        csv_data_dict_sb[csv_col_name] = _prepare_timeseries_for_export(safe_get(sim_res_csv_exp_sb, path_csv_sb, []), num_steps_csv_exp_sb)
                    
                    raw_downtime_csv_sb = safe_get(sim_res_csv_exp_sb, 'downtime_events_log', [])
                    csv_data_dict_sb['downtime_minutes_per_interval'] = aggregate_downtime_by_step(raw_downtime_csv_sb, num_steps_csv_exp_sb)
                    
                    df_to_csv_export_sb = pd.DataFrame(csv_data_dict_sb)
                    st.download_button("📥 Download Data (CSV)", df_to_csv_export_sb.to_csv(index=False).encode('utf-8'), 
                                      "workplace_summary.csv", "text/csv", key="sb_csv_download_button_main", use_container_width=True)
                else: st.caption("No detailed data to export (0 simulation steps).")
            elif not can_export_data_sidebar: st.caption("Run simulation for export options.")
        
        if st.session_state.sb_debug_mode_checkbox:
            with st.expander("🛠️ Debug Information", expanded=False):
                st.write("**Default Config (Partial):**")
                st.json({k_dbg: DEFAULT_CONFIG.get(k_dbg) for k_dbg in ['MINUTES_PER_INTERVAL', 'WORK_AREAS', 'DEFAULT_SCHEDULED_EVENTS', 'TEAM_SIZE']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results: 
                    st.write("**Active Simulation Config (from results):**")
                    st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else: st.write("**No active simulation data.**")
        
        st.markdown("## 📋 Help & Info")
        if st.button("ℹ️ Help & Glossary", key="sb_help_button_main_sidebar", use_container_width=True): 
            st.session_state.show_help_glossary = not st.session_state.get('show_help_glossary', False)
            st.rerun()
        if st.button("🚀 Quick Tour", key="sb_tour_button_main_sidebar", use_container_width=True): 
            st.session_state.show_tour = not st.session_state.get('show_tour', False)
            st.rerun()
            
    return run_simulation_button_sidebar, load_data_button_sidebar
    # --- SIMULATION LOGIC WRAPPER (CACHED) ---
@st.cache_data(ttl=3600, show_spinner="⚙️ Simulating workplace operations...")
def run_simulation_logic(team_size_sl, shift_duration_sl, scheduled_events_from_ui_sl, team_initiative_sl):
    config_sl = DEFAULT_CONFIG.copy()
    config_sl['TEAM_SIZE'] = int(team_size_sl)
    config_sl['SHIFT_DURATION_MINUTES'] = int(shift_duration_sl)
    
    mpi_sl = _get_config_value_sl(config_sl, {}, 'MINUTES_PER_INTERVAL', 2) # Using _get_config_value_sl for consistency
    if mpi_sl <= 0: mpi_sl = 2; logger.error("MPI was <=0 in config, used 2 for calculation.")
    config_sl['SHIFT_DURATION_INTERVALS'] = config_sl['SHIFT_DURATION_MINUTES'] // mpi_sl

    processed_events_sl = []
    for event_sl_ui_item in scheduled_events_from_ui_sl:
        evt_sl_item = event_sl_ui_item.copy()
        if 'step' not in evt_sl_item and 'Start Time (min)' in evt_sl_item:
            start_time_min_evt = _get_config_value_sl(evt_sl_item, {}, 'Start Time (min)', 0) # Use helper
            evt_sl_item['step'] = int(start_time_min_evt // mpi_sl)
        processed_events_sl.append(evt_sl_item)
    config_sl['SCHEDULED_EVENTS'] = processed_events_sl
    
    # Worker Redistribution Logic
    if 'WORK_AREAS' in config_sl and isinstance(config_sl['WORK_AREAS'], dict) and config_sl['WORK_AREAS']:
        current_total_workers_cfg = sum(_get_config_value_sl(z,{},'workers',0) for z in config_sl['WORK_AREAS'].values() if isinstance(z,dict))
        target_team_size_sl = config_sl['TEAM_SIZE']

        if current_total_workers_cfg != target_team_size_sl and target_team_size_sl >= 0:
            logger.info(f"Redistributing workers. Config sum: {current_total_workers_cfg}, Target team: {target_team_size_sl}")
            
            # Prioritize distributing to non-rest areas first
            work_areas_for_dist = {k:v for k,v in config_sl['WORK_AREAS'].items() if isinstance(v,dict) and not v.get('is_rest_area',False)}
            non_dist_areas = {k:v for k,v in config_sl['WORK_AREAS'].items() if k not in work_areas_for_dist}

            if target_team_size_sl == 0: # If target is 0, set all to 0
                for zone_k_sl_zero in config_sl['WORK_AREAS']: config_sl['WORK_AREAS'][zone_k_sl_zero]['workers'] = 0
            elif work_areas_for_dist : # If there are areas to distribute to
                # Calculate sum of workers only in distributable areas for ratio if needed
                current_dist_workers_sum = sum(_get_config_value_sl(z,{},'workers',0) for z in work_areas_for_dist.values())

                if current_dist_workers_sum > 0: # Proportional redistribution among distributable areas
                    ratio_sl = target_team_size_sl / current_dist_workers_sum
                    accumulated_sl = 0
                    sorted_zone_keys_sl = sorted(work_areas_for_dist.keys())
                    for i_zone_sl, zone_k_sl in enumerate(sorted_zone_keys_sl):
                        original_workers = _get_config_value_sl(config_sl['WORK_AREAS'][zone_k_sl], {}, 'workers', 0)
                        if i_zone_sl < len(sorted_zone_keys_sl) - 1:
                            new_w_sl = int(round(original_workers * ratio_sl))
                            config_sl['WORK_AREAS'][zone_k_sl]['workers'] = new_w_sl
                            accumulated_sl += new_w_sl
                        else: 
                            config_sl['WORK_AREAS'][zone_k_sl]['workers'] = max(0, target_team_size_sl - accumulated_sl)
                else: # Distribute evenly among distributable areas if they had 0 workers initially
                    num_dist_zones = len(work_areas_for_dist)
                    base_w_sl, rem_w_sl = divmod(target_team_size_sl, num_dist_zones)
                    assign_count_sl = 0
                    for zone_k_sl_even in work_areas_for_dist: # Iterate directly over keys
                        config_sl['WORK_AREAS'][zone_k_sl_even]['workers'] = base_w_sl + (1 if assign_count_sl < rem_w_sl else 0)
                        assign_count_sl +=1
            # Ensure non-distributable (e.g. rest) areas that were not part of above logic have 0 workers
            for zone_k_sl_nd, zone_data_sl_nd in non_dist_areas.items():
                if isinstance(zone_data_sl_nd, dict): zone_data_sl_nd['workers'] = 0


    validate_config(config_sl)
    logger.info(f"Running simulation: Team={config_sl['TEAM_SIZE']}, Duration={config_sl['SHIFT_DURATION_MINUTES']}min ({config_sl['SHIFT_DURATION_INTERVALS']} intervals), Events={len(config_sl['SCHEDULED_EVENTS'])}, Initiative={team_initiative_sl}", extra={'user_action': 'Run Simulation - Start'})
    
    expected_keys_sl = [
        'team_positions_df', 'task_compliance', 'collaboration_metric',
        'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 
        'worker_wellbeing', 'psychological_safety', 'feedback_impact', 
        'downtime_events_log', 'task_completion_rate'
    ]
    sim_results_tuple_sl_run = simulate_workplace_operations(
        num_team_members=config_sl['TEAM_SIZE'], num_steps=config_sl['SHIFT_DURATION_INTERVALS'],
        scheduled_events=config_sl['SCHEDULED_EVENTS'], team_initiative=team_initiative_sl, config=config_sl
    )
    
    if not isinstance(sim_results_tuple_sl_run, tuple) or len(sim_results_tuple_sl_run) != len(expected_keys_sl):
        logger.critical(f"Simulation returned unexpected data format. Expected tuple of {len(expected_keys_sl)}, got {type(sim_results_tuple_sl_run)} of len {len(sim_results_tuple_sl_run) if isinstance(sim_results_tuple_sl_run,(list,tuple)) else 'N/A'}.", extra={'user_action':'Sim Format Error'})
        raise TypeError("Simulation returned unexpected data format.")
        
    simulation_output_dict_sl_final = dict(zip(expected_keys_sl, sim_results_tuple_sl_run))
    simulation_output_dict_sl_final['config_params'] = {
        'TEAM_SIZE': config_sl['TEAM_SIZE'], 'SHIFT_DURATION_MINUTES': config_sl['SHIFT_DURATION_MINUTES'],
        'SHIFT_DURATION_INTERVALS': config_sl['SHIFT_DURATION_INTERVALS'],
        'MINUTES_PER_INTERVAL': mpi_sl, 'SCHEDULED_EVENTS': config_sl['SCHEDULED_EVENTS'],
        'TEAM_INITIATIVE': team_initiative_sl, 'WORK_AREAS_EFFECTIVE': config_sl.get('WORK_AREAS', {}).copy()
    }
    
    disruption_steps_final_sl = [evt.get('step') for evt in config_sl['SCHEDULED_EVENTS'] if isinstance(evt,dict) and "Disruption" in evt.get("Event Type","") and isinstance(evt.get('step'),int)]
    simulation_output_dict_sl_final['config_params']['DISRUPTION_EVENT_STEPS'] = sorted(list(set(disruption_steps_final_sl)))

    save_simulation_data(simulation_output_dict_sl_final) 
    return simulation_output_dict_sl_final

def _get_config_value_sl(primary_conf, secondary_conf, key, default):
    return secondary_conf.get(key, primary_conf.get(key, default))

# --- TIME RANGE INPUT WIDGETS ---
def time_range_input_section(tab_key_prefix: str, max_minutes_for_range_ui: int, st_col_obj = st, interval_duration_min_ui: int = 2):
    start_time_key_ui = f"{tab_key_prefix}_start_time_min"
    end_time_key_ui = f"{tab_key_prefix}_end_time_min"
    if interval_duration_min_ui <=0: interval_duration_min_ui = 2 

    current_start_ui = st.session_state.get(start_time_key_ui, 0)
    current_end_ui = st.session_state.get(end_time_key_ui, max_minutes_for_range_ui)
    current_start_ui = max(0, min(current_start_ui, max_minutes_for_range_ui))
    current_end_ui = max(current_start_ui, min(current_end_ui, max_minutes_for_range_ui))
    st.session_state[start_time_key_ui], st.session_state[end_time_key_ui] = current_start_ui, current_end_ui
    
    prev_start_ui_val, prev_end_ui_val = current_start_ui, current_end_ui
    cols_ui_time = st_col_obj.columns(2)
    
    new_start_time_val_ui = cols_ui_time[0].number_input( "Start Time (min)", 0, max_minutes_for_range_ui, current_start_ui, 
        interval_duration_min_ui, key=f"widget_num_input_{start_time_key_ui}", help=f"Range: 0 to {max_minutes_for_range_ui} min.")
    st.session_state[start_time_key_ui] = new_start_time_val_ui
    
    end_time_min_for_widget_ui = st.session_state[start_time_key_ui]
    new_end_time_val_ui = cols_ui_time[1].number_input("End Time (min)", end_time_min_for_widget_ui, max_minutes_for_range_ui, current_end_ui, 
        interval_duration_min_ui, key=f"widget_num_input_{end_time_key_ui}", help=f"Range: {end_time_min_for_widget_ui} to {max_minutes_for_range_ui} min.")
    st.session_state[end_time_key_ui] = new_end_time_val_ui

    if st.session_state[end_time_key_ui] < st.session_state[start_time_key_ui]:
        st.session_state[end_time_key_ui] = st.session_state[start_time_key_ui]
    
    if prev_start_ui_val != st.session_state[start_time_key_ui] or prev_end_ui_val != st.session_state[end_time_key_ui]:
        st.rerun()
    return int(st.session_state[start_time_key_ui]), int(st.session_state[end_time_key_ui])
    # --- MAIN APPLICATION FUNCTION ---
def main():
    st.title("Workplace Shift Optimization Dashboard")
    
    mpi_global_app = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)
    if mpi_global_app <= 0: mpi_global_app = 2 
    app_state_defaults_main = {
        'simulation_results': None, 'show_tour': False, 'show_help_glossary': False,
        'sb_team_size_num': DEFAULT_CONFIG['TEAM_SIZE'], 'sb_shift_duration_num': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
        'sb_scheduled_events_list': list(DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', [])),
        'sb_team_initiative_selectbox': "Standard Operations",
        'sb_high_contrast_checkbox': False, 'sb_use_3d_distribution_checkbox': False, 'sb_debug_mode_checkbox': False,
        'form_event_type': "Major Disruption", 'form_event_start': 0, 'form_event_duration': max(mpi_global_app, 10),
    }
    default_max_mins_main_app = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - mpi_global_app if DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] > mpi_global_app else 0
    for prefix_main_app in ['op', 'ww', 'dt']:
        app_state_defaults_main[f'{prefix_main_app}_start_time_min'] = 0
        app_state_defaults_main[f'{prefix_main_app}_end_time_min'] = default_max_mins_main_app
    for key_main_app, val_main_app in app_state_defaults_main.items():
        if key_main_app not in st.session_state: st.session_state[key_main_app] = val_main_app
            
    run_simulation_button_main_app, load_data_button_main_app = render_settings_sidebar()
    
    current_high_contrast_main_app = st.session_state.sb_high_contrast_checkbox
    use_3d_main_app = st.session_state.sb_use_3d_distribution_checkbox

    active_mpi_main_app = mpi_global_app
    max_mins_ui_main_app = default_max_mins_main_app
    simulation_disruption_steps_absolute_main = []

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        sim_cfg_main_app_active = st.session_state.simulation_results.get('config_params', {})
        active_mpi_main_app = sim_cfg_main_app_active.get('MINUTES_PER_INTERVAL', mpi_global_app)
        if active_mpi_main_app <= 0 : active_mpi_main_app = 2
        sim_intervals_main_app_active = sim_cfg_main_app_active.get('SHIFT_DURATION_INTERVALS', 0)
        max_mins_ui_main_app = max(0, sim_intervals_main_app_active * active_mpi_main_app - active_mpi_main_app) if sim_intervals_main_app_active > 0 else 0
        simulation_disruption_steps_absolute_main = sim_cfg_main_app_active.get('DISRUPTION_EVENT_STEPS', [])
    else:
        shift_duration_from_sidebar_main = st.session_state.sb_shift_duration_num
        sim_intervals_main_app_active = shift_duration_from_sidebar_main // active_mpi_main_app if active_mpi_main_app > 0 else 0
        max_mins_ui_main_app = max(0, sim_intervals_main_app_active * active_mpi_main_app - active_mpi_main_app) if sim_intervals_main_app_active > 0 else 0
        for event_main_ui_item_cfg in st.session_state.sb_scheduled_events_list:
            if "Disruption" in event_main_ui_item_cfg.get("Event Type","") and isinstance(event_main_ui_item_cfg.get("Start Time (min)"), (int,float)):
                simulation_disruption_steps_absolute_main.append(int(event_main_ui_item_cfg["Start Time (min)"] // active_mpi_main_app))
        simulation_disruption_steps_absolute_main = sorted(list(set(simulation_disruption_steps_absolute_main)))
    
    for prefix_main_ui_clamp_val in ['op', 'ww', 'dt']:
        st.session_state[f"{prefix_main_ui_clamp_val}_start_time_min"] = max(0, min(st.session_state.get(f"{prefix_main_ui_clamp_val}_start_time_min",0), max_mins_ui_main_app))
        st.session_state[f"{prefix_main_ui_clamp_val}_end_time_min"] = max(st.session_state[f"{prefix_main_ui_clamp_val}_start_time_min"], min(st.session_state.get(f"{prefix_main_ui_clamp_val}_end_time_min",max_mins_ui_main_app), max_mins_ui_main_app))

    if run_simulation_button_main_app:
        with st.spinner("🚀 Simulating workplace operations... This may take a moment."):
            try:
                results_run = run_simulation_logic(st.session_state.sb_team_size_num, st.session_state.sb_shift_duration_num, 
                                               list(st.session_state.sb_scheduled_events_list), st.session_state.sb_team_initiative_selectbox)
                st.session_state.simulation_results = results_run
                new_cfg_run_main = results_run['config_params']
                new_mpi_run_main = new_cfg_run_main.get('MINUTES_PER_INTERVAL', 2)
                new_sim_intervals_run_main = new_cfg_run_main.get('SHIFT_DURATION_INTERVALS',0)
                new_max_mins_run_main = max(0, new_sim_intervals_run_main * new_mpi_run_main - new_mpi_run_main) if new_sim_intervals_run_main > 0 else 0
                for pfx_run in ['op','ww','dt']: st.session_state[f"{pfx_run}_start_time_min"]=0; st.session_state[f"{pfx_run}_end_time_min"]=new_max_mins_run_main
                st.success("✅ Simulation completed successfully!"); logger.info("Sim run success."); st.rerun()
            except Exception as e_run_main_app: logger.error(f"Sim Run Error: {e_run_main_app}", exc_info=True); st.error(f"❌ Sim failed: {e_run_main_app}"); st.session_state.simulation_results = None
    if load_data_button_main_app:
        with st.spinner("🔄 Loading saved simulation data..."):
            try:
                loaded_main = load_simulation_data()
                if loaded_main and isinstance(loaded_main, dict) and 'config_params' in loaded_main:
                    st.session_state.simulation_results = loaded_main; cfg_ld_main = loaded_main['config_params']
                    st.session_state.sb_team_size_num = cfg_ld_main.get('TEAM_SIZE', DEFAULT_CONFIG['TEAM_SIZE'])
                    st.session_state.sb_shift_duration_num = cfg_ld_main.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'])
                    st.session_state.sb_scheduled_events_list = list(cfg_ld_main.get('SCHEDULED_EVENTS', []))
                    st.session_state.sb_team_initiative_selectbox = cfg_ld_main.get('TEAM_INITIATIVE', "Standard Operations")
                    max_ld_main = max(0, cfg_ld_main.get('SHIFT_DURATION_INTERVALS',0) * cfg_ld_main.get('MINUTES_PER_INTERVAL',2) - cfg_ld_main.get('MINUTES_PER_INTERVAL',2))
                    for pfx_ld in ['op','ww','dt']: st.session_state[f"{pfx_ld}_start_time_min"]=0; st.session_state[f"{pfx_ld}_end_time_min"]=max_ld_main
                    st.success("✅ Data loaded!"); logger.info("Load success."); st.rerun()
                else: st.error("❌ Failed to load data or data invalid."); logger.warning("Load fail/invalid.")
            except Exception as e_load_main_app: logger.error(f"Load Data Error: {e_load_main_app}", exc_info=True); st.error(f"❌ Load failed: {e_load_main_app}"); st.session_state.simulation_results = None
    
    if st.session_state.get('show_tour', False): 
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Tour</h3><p>...</p></div>""", unsafe_allow_html=True) # Truncated for brevity
        if st.button("Got it!", key="tour_modal_close_main_final"): st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help_glossary', False): 
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3><p>...</p></div>""", unsafe_allow_html=True) # Truncated
        if st.button("Understood", key="help_modal_close_main_final"): st.session_state.show_help_glossary = False; st.rerun()

    tab_names_main_final = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Well-being", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs_obj_main_final = st.tabs(tab_names_main_final)
    plot_cfg_interactive_final = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}
    plot_cfg_minimal_final = {'displayModeBar': False}

    with tabs_obj_main_final[0]: # Overview Tab
        st.header("📊 Key Performance Indicators & Actionable Insights", divider=THEMED_DIVIDER_COLOR)
        if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
            sim_data_ov_main = st.session_state.simulation_results
            sim_cfg_ov_main = sim_data_ov_main.get('config_params', DEFAULT_CONFIG)
            effective_cfg_ov_main = {**DEFAULT_CONFIG, **sim_cfg_ov_main}
            # ... (Full Overview tab metric calculations and plotting logic as in previous complete response) ...
            # Ensure `get_actionable_insights` is called with `effective_cfg_ov_main`
            actionable_insights_main_ov = get_actionable_insights(sim_data_ov_main, effective_cfg_ov_main)
            # ... (Display insights) ...
            # ... (Detailed overview data table) ...
        else: st.info("ℹ️ Run a simulation or load data to view the Overview & Insights.", icon="📊")
    
    tab_configs_main_final = [ # Ensure all data_paths and extra_args_fixed/paths are correct
        {"name": "📈 Operational Metrics", "key_prefix": "op", "plots": [
             {"title": "Task Compliance", "data_path": "task_compliance.data", "plot_func": plot_task_compliance_score, "extra_args_paths": {"forecast_data": "task_compliance.forecast", "z_scores": "task_compliance.z_scores"}},
             {"title": "Collaboration Metric", "data_path": "collaboration_metric.data", "plot_func": plot_collaboration_proximity_index, "extra_args_paths": {"forecast_data": "collaboration_metric.forecast"}},
             {"is_subheader": True, "title": "Additional Operational Metrics"},
             {"title": "Operational Resilience", "data_path": "operational_recovery", "plot_func": plot_operational_recovery, "extra_args_paths": {"productivity_loss_data": "productivity_loss"}},
             {"title": "OEE & Components", "is_oee": True}], "insights_html": "<p>Insights for Operational Metrics...</p>" },
        {"name": "👥 Worker Well-being", "key_prefix": "ww", "plots": [
             {"is_subheader": True, "title": "Psychosocial & Well-being Indicators"},
             {"title": "Worker Well-Being", "data_path": "worker_wellbeing.scores", "plot_func": plot_worker_wellbeing, "extra_args_paths": {"triggers": "worker_wellbeing.triggers"}},
             {"title": "Psychological Safety", "data_path": "psychological_safety", "plot_func": plot_psychological_safety},
             {"title": "Team Cohesion", "data_path": "worker_wellbeing.team_cohesion_scores", "plot_func": plot_team_cohesion},
             {"title": "Perceived Workload", "data_path": "worker_wellbeing.perceived_workload_scores", "plot_func": plot_perceived_workload, "extra_args_fixed": {"high_workload_threshold": DEFAULT_CONFIG['PERCEIVED_WORKLOAD_THRESHOLD_HIGH'], "very_high_workload_threshold": DEFAULT_CONFIG['PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH']}},
             {"is_subheader": True, "title": "Spatial Dynamics Analysis", "is_spatial": True}], "dynamic_insights_func": "render_wellbeing_alerts", "insights_html": "<p>Insights for Well-being...</p>" },
        {"name": "⏱️ Downtime Analysis", "key_prefix": "dt", "metrics_display": True, "plots": [
            {"title": "Downtime Trend", "data_path": "downtime_events_log", "plot_func": plot_downtime_trend, "is_event_based_aggregation": True, "extra_args_fixed": {"interval_threshold_minutes": DEFAULT_CONFIG['DOWNTIME_PLOT_ALERT_THRESHOLD']}},
            {"title": "Downtime Causes", "data_path": "downtime_events_log", "plot_func": plot_downtime_causes_pie, "is_event_based_filtering": True}], "insights_html": "<p>Insights for Downtime Analysis...</p>" }
    ]

    for i_tab_final, tab_def_final in enumerate(tab_configs_main_final):
        with tabs_obj_main_final[i_tab_final+1]:
            st.header(tab_def_final["name"], divider=THEMED_DIVIDER_COLOR)
            if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
                sim_data_tab_final = st.session_state.simulation_results
                sim_cfg_tab_final_active = sim_data_tab_final.get('config_params', {})
                st.markdown("##### Select Time Range for Plots:")
                start_time_tab, end_time_tab = time_range_input_section(tab_def_final["key_prefix"], max_mins_ui_main_app, interval_duration_min_ui=active_mpi_main_app)
                start_idx_tab_final, end_idx_tab_final = (start_time_tab // active_mpi_main_app if active_mpi_main_app > 0 else 0), ((end_time_tab // active_mpi_main_app) + 1 if active_mpi_main_app > 0 else 0)
                disrupt_steps_for_plots_abs_tab_final = [s for s in simulation_disruption_steps_absolute_main if start_idx_tab_final <= s < end_idx_tab_final]
                
                # ... (Full plotting loop logic from previous response, using _final suffixed variables.
                #      This includes the specific data prep for OEE, downtime plots, spatial plots,
                #      and standard timeseries plots, along with careful handling of kwargs like
                #      `disruption_points` and `triggers` for `plot_worker_wellbeing`.)
                # Example of calling a plot (ensure all plots follow this pattern):
                # if plot_cfg.get("is_oee"): ...
                # else: ... plot_data_final = ...; fig = plot_func(plot_data_final, **kwargs); st.plotly_chart(fig)
                pass # Placeholder for detailed plotting loop for brevity
            else: st.info(f"ℹ️ Run simulation or load data to view {tab_def_final['name']}.", icon="📊")

    with tabs_obj_main_final[4]: # Glossary
        st.header("📖 Glossary of Terms", divider=THEMED_DIVIDER_COLOR)
        # ... (Full Glossary HTML content as in previous combined file) ...
        st.markdown("""<p>Defines key metrics used...</p>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
