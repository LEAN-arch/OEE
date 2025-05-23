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
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
                        filename='dashboard.log',
                        filemode='a')
logger.info("Main.py: Startup. Correcting NameError and ensuring light theme.", extra={'user_action': 'System Startup'})

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.3.16\nInsights for operational excellence & psychosocial well-being."})

# --- Light Theme Color Constants for CSS and Streamlit Elements ---
COLOR_PAGE_BACKGROUND_LIGHT = "#F0F2F6"
COLOR_SIDEBAR_BACKGROUND_LIGHT = "#EAEBED"
COLOR_CONTENT_BACKGROUND_LIGHT = "#FFFFFF"

COLOR_PRIMARY_TEXT_DARK = "#262730"
COLOR_SECONDARY_TEXT_DARK = "#5E6474"
COLOR_ACCENT_TEXT_DARK = "#0052CC" 

COLOR_CRITICAL_RED_BORDER = "#D62728"
COLOR_WARNING_AMBER_BORDER = "#FF7F0E"
COLOR_POSITIVE_GREEN_BORDER = "#2CA02C"
COLOR_INFO_BLUE_BORDER = "#1F77B4"

COLOR_CRITICAL_RED_BG_LIGHT = "rgba(214, 39, 40, 0.1)"
COLOR_WARNING_AMBER_BG_LIGHT = "rgba(255, 127, 14, 0.1)"
COLOR_POSITIVE_GREEN_BG_LIGHT = "rgba(44, 160, 44, 0.1)"
COLOR_INFO_BLUE_BG_LIGHT = "rgba(31, 119, 180, 0.1)"

COLOR_BORDER_SUBTLE_LIGHT = "#D1D5DB"
COLOR_BORDER_DARKER_LIGHT = "#A0AEC0"

COLOR_ACCENT_UI_LIGHT_THEME = "#0063BF" 
COLOR_ACCENT_BUTTON_LIGHT_THEME = COLOR_ACCENT_UI_LIGHT_THEME 
COLOR_ACCENT_BUTTON_HOVER_LIGHT_THEME = "#0052A3" 
COLOR_BUTTON_SIDEBAR_DEFAULT_BG_LIGHT = "#2CA02C"
COLOR_BUTTON_SIDEBAR_DEFAULT_HOVER_BG_LIGHT = "#228B22"
COLOR_BUTTON_SECONDARY_BG_LIGHT = "#E0E0E0"
COLOR_BUTTON_SECONDARY_HOVER_BG_LIGHT = "#BDBDBD"
COLOR_BUTTON_REMOVE_BG_LIGHT = COLOR_CRITICAL_RED_BORDER

THEMED_DIVIDER_COLOR = "gray"

def safe_get(data_dict, path_str, default_val=None):
    current = data_dict
    is_list_like_path = False
    if isinstance(path_str, str):
        is_list_like_path = path_str.endswith(('.data', '.scores', '.triggers', '_log', 'events_list'))
    
    if default_val is None: default_return = [] if is_list_like_path else None
    else: default_return = default_val

    if not isinstance(path_str, str): return default_return
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
            else: current = None; break
        if current is None:
            is_list_like_final_key = keys and keys[-1] in ['data', 'scores', 'triggers', '_log', 'events_list']
            return [] if default_val is None and is_list_like_final_key else default_val
        return current
    except (ValueError, IndexError, TypeError) as e:
        logger.debug(f"safe_get failed for path '{path_str}': {e}. Returning default '{default_return}'.")
        return default_return

def safe_stat(data_list, stat_func, default_val=0.0):
    if not isinstance(data_list, (list, np.ndarray, pd.Series)): return default_val
    if isinstance(data_list, pd.Series):
        valid_data = pd.to_numeric(data_list, errors='coerce').dropna().tolist()
    else: 
        valid_data = []
        for x in data_list:
            if x is None or (isinstance(x, float) and np.isnan(x)): continue
            try: valid_data.append(float(x))
            except (ValueError, TypeError): pass 
    if not valid_data: return default_val
    try:
        result = stat_func(np.array(valid_data))
        return default_val if isinstance(result, (float, np.floating)) and np.isnan(result) else result
    except Exception: return default_val

def _get_config_value_main(primary_conf, secondary_conf, key, default):
    return secondary_conf.get(key, primary_conf.get(key, default))

def get_actionable_insights(sim_data, current_config_dict_main):
    insights = []
    if not sim_data or not isinstance(sim_data, dict): 
        logger.warning("get_actionable_insights: sim_data is None or not a dict.", extra={'user_action': 'Insights - Invalid Input'})
        return insights
    
    sim_cfg_params_insights_main = sim_data.get('config_params', {})
    def _get_insight_cfg(key, default): return _get_config_value_main(current_config_dict_main, sim_cfg_params_insights_main, key, default)

    compliance_data_ia = safe_get(sim_data, 'task_compliance.data', [])
    target_compliance_ia = float(_get_insight_cfg('TARGET_COMPLIANCE', 75.0))
    compliance_avg_ia = safe_stat(compliance_data_ia, np.mean, 0.0)
    if compliance_data_ia and compliance_avg_ia < target_compliance_ia * 0.9: insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Avg. Task Compliance ({compliance_avg_ia:.1f}%) critically below target ({target_compliance_ia:.0f}%). Review disruptions, complexities, training."})
    elif compliance_data_ia and compliance_avg_ia < target_compliance_ia: insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Avg. Task Compliance at {compliance_avg_ia:.1f}%. Review areas with lowest compliance."})

    wellbeing_scores_ia = safe_get(sim_data, 'worker_wellbeing.scores', [])
    target_wellbeing_ia = float(_get_insight_cfg('TARGET_WELLBEING', 70.0))
    wellbeing_avg_ia = safe_stat(wellbeing_scores_ia, np.mean, 0.0)
    wb_crit_factor_ia = float(_get_insight_cfg('WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET', 0.85))
    if wellbeing_scores_ia and wellbeing_avg_ia < target_wellbeing_ia * wb_crit_factor_ia: insights.append({"type": "critical", "title": "Critical Worker Well-being", "text": f"Avg. Well-being ({wellbeing_avg_ia:.1f}%) critically low (target {target_wellbeing_ia:.0f}%). Urgent review needed."})
    
    threshold_triggers_ia = safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])
    if wellbeing_scores_ia and len(threshold_triggers_ia) > max(2, len(wellbeing_scores_ia) * 0.1): insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers_ia)} low well-being instances. Investigate triggers."})

    downtime_event_log_ia = safe_get(sim_data, 'downtime_events_log', [])
    downtime_durations_ia = [event.get('duration', 0.0) for event in downtime_event_log_ia if isinstance(event, dict)]
    total_downtime_ia = sum(downtime_durations_ia)
    shift_mins_ia = float(sim_cfg_params_insights_main.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
    dt_thresh_percent_ia = float(_get_insight_cfg('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT', 0.05))
    dt_thresh_total_shift_abs_ia = shift_mins_ia * dt_thresh_percent_ia
    if total_downtime_ia > dt_thresh_total_shift_abs_ia: insights.append({"type": "critical", "title": "Excessive Total Downtime", "text": f"Total downtime {total_downtime_ia:.0f} min, exceeds guideline of {dt_thresh_total_shift_abs_ia:.0f} min ({dt_thresh_percent_ia*100:.0f}% of shift). Analyze causes."})

    psych_safety_scores_ia = safe_get(sim_data, 'psychological_safety', [])
    target_psych_safety_ia = float(_get_insight_cfg('TARGET_PSYCH_SAFETY', 70.0))
    psych_safety_avg_ia = safe_stat(psych_safety_scores_ia, np.mean, 0.0)
    if psych_safety_scores_ia and psych_safety_avg_ia < target_psych_safety_ia * 0.9: insights.append({"type": "warning", "title": "Low Psychological Safety", "text": f"Avg. Psych. Safety ({psych_safety_avg_ia:.1f}%) below target ({target_psych_safety_ia:.0f}%). Build trust."})

    cohesion_scores_ia = safe_get(sim_data, 'worker_wellbeing.team_cohesion_scores', [])
    target_cohesion_ia = float(_get_insight_cfg('TARGET_TEAM_COHESION', 70.0))
    cohesion_avg_ia = safe_stat(cohesion_scores_ia, np.mean, 0.0)
    if cohesion_scores_ia and cohesion_avg_ia < target_cohesion_ia * 0.9: insights.append({"type": "warning", "title": "Low Team Cohesion", "text": f"Avg. Team Cohesion ({cohesion_avg_ia:.1f}%) below desired. Consider team-building."})
    
    workload_scores_ia = safe_get(sim_data, 'worker_wellbeing.perceived_workload_scores', [])
    target_workload_ia = float(_get_insight_cfg('TARGET_PERCEIVED_WORKLOAD', 6.5))
    workload_avg_ia = safe_stat(workload_scores_ia, np.mean, target_workload_ia / 2) 
    workload_very_high_thresh_ia = float(_get_insight_cfg('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5))
    workload_high_thresh_ia = float(_get_insight_cfg('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5))
    if workload_scores_ia:
        if workload_avg_ia > workload_very_high_thresh_ia: insights.append({"type": "critical", "title": "Very High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg_ia:.1f}/10) critically high. Immediate review required."})
        elif workload_avg_ia > workload_high_thresh_ia: insights.append({"type": "warning", "title": "High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg_ia:.1f}/10) exceeds high threshold. Monitor."})
        elif workload_avg_ia > target_workload_ia: insights.append({"type": "info", "title": "Elevated Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg_ia:.1f}/10) above target ({target_workload_ia:.1f}/10). Consider adjustments."})
    
    team_pos_df_ia = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
    work_areas_effective_ia = sim_cfg_params_insights_main.get('WORK_AREAS_EFFECTIVE', current_config_dict_main.get('WORK_AREAS', {}))
    if not team_pos_df_ia.empty and isinstance(work_areas_effective_ia, dict) and \
       all(col in team_pos_df_ia.columns for col in ['zone', 'worker_id', 'step']):
        for zone_name_ia, zone_details_ia in work_areas_effective_ia.items():
            if not isinstance(zone_details_ia, dict): continue
            workers_in_zone_series_ia = team_pos_df_ia[team_pos_df_ia['zone'] == zone_name_ia].groupby('step')['worker_id'].nunique()
            if not workers_in_zone_series_ia.empty:
                workers_in_zone_avg_ia = workers_in_zone_series_ia.mean()
                intended_workers_ia = zone_details_ia.get('workers', 0)
                coords_ia = zone_details_ia.get('coords'); area_m2_ia = 1.0
                if coords_ia and isinstance(coords_ia, list) and len(coords_ia) == 2 and \
                   all(isinstance(p_ia, tuple) and len(p_ia)==2 and all(isinstance(c_ia, (int,float)) for c_tuple_val_ia in coords_ia for c_ia in c_tuple_val_ia) for p_ia in coords_ia):
                    if len(coords_ia[0]) == 2 and len(coords_ia[1]) == 2:
                        (x0_ia,y0_ia), (x1_ia,y1_ia) = coords_ia[0], coords_ia[1]; area_m2_ia = abs(x1_ia-x0_ia) * abs(y1_ia-y0_ia)
                if abs(area_m2_ia) < 1e-6: area_m2_ia = 1.0 
                avg_density_ia = workers_in_zone_avg_ia / area_m2_ia if area_m2_ia > 0 else 0
                intended_density_ia = (intended_workers_ia / area_m2_ia) if area_m2_ia > 0 and intended_workers_ia > 0 else 0
                if intended_density_ia > 0 and avg_density_ia > intended_density_ia * 1.8: insights.append({"type": "warning", "title": f"Potential Overcrowding: '{zone_name_ia}'", "text": f"Avg. density ({avg_density_ia:.2f} w/m²) > intended ({intended_density_ia:.2f} w/m²). Review layout."})
                elif intended_workers_ia > 0 and workers_in_zone_avg_ia < intended_workers_ia * 0.4 and not zone_details_ia.get("is_rest_area", False): insights.append({"type": "info", "title": f"Potential Underutilization: '{zone_name_ia}'", "text": f"Avg. workers ({workers_in_zone_avg_ia:.1f}) <40% of assigned ({intended_workers_ia}). Check allocation."})

    if all(s is not None for s in [compliance_data_ia, wellbeing_scores_ia, psych_safety_scores_ia]) and \
       compliance_avg_ia > target_compliance_ia * 1.05 and wellbeing_avg_ia > target_wellbeing_ia * 1.05 and \
       total_downtime_ia < dt_thresh_total_shift_abs_ia * 0.5 and psych_safety_avg_ia > target_psych_safety_ia * 1.05:
        insights.append({"type": "positive", "title": "Holistically Excellent Performance", "text": "Key metrics significantly exceed targets. Well-balanced and high-performing!"})
    
    initiative_ia = sim_cfg_params_insights_main.get('TEAM_INITIATIVE', 'Standard Operations') 
    if initiative_ia != "Standard Operations": insights.append({"type": "info", "title": f"Initiative Active: '{initiative_ia}'", "text": f"The '{initiative_ia}' initiative simulated. Compare to 'Standard Operations' baseline."})
    
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
    logger.warning(f"Could not slice DataFrame by step indices. Columns: {df.columns}, Index: {df.index.name}")
    return pd.DataFrame()

# --- CSS STYLES FOR LIGHT THEME ---
st.markdown(f"""
    <style>
        /* Base Styles for Light Theme */
        .main {{ 
            background-color: {COLOR_PAGE_BACKGROUND_LIGHT} !important; 
            color: {COLOR_PRIMARY_TEXT_DARK} !important; 
            font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif; padding: 2rem; 
        }}
        /* Ensure all general text defaults to primary dark color */
        body, p, div, span, li {{
            color: {COLOR_PRIMARY_TEXT_DARK} !important;
        }}

        h1 {{ 
            font-size: 2.4rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; 
            text-align: center; margin-bottom: 2rem; color: {COLOR_PRIMARY_TEXT_DARK} !important; 
        }}
        
        /* Main Content Headers (Tabs) - h2 generated by st.header in tabs */
        div[data-testid="stTabs"] section[role="tabpanel"] > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div > h2 {{ 
            font-size: 1.75rem !important; font-weight: 600 !important; line-height: 1.3 !important; 
            margin: 1.2rem 0 1rem 0 !important; color: {COLOR_PRIMARY_TEXT_DARK} !important; 
            padding-bottom: 0.6rem !important; text-align: left !important;
        }}

        /* Main Content Section Subheaders - h3 generated by st.subheader */
         div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] .stSubheader {{ 
            font-size: 1.3rem !important; font-weight: 500 !important; line-height: 1.4 !important; 
            margin-top: 1.8rem !important; margin-bottom: 0.8rem !important; color: {COLOR_SECONDARY_TEXT_DARK} !important;
            border-bottom: 1px solid {COLOR_BORDER_SUBTLE_LIGHT} !important; 
            padding-bottom: 0.3rem !important; text-align: left !important;
        }}
        /* Main Content Markdown H5 (e.g. for "Select Time Range for Plots:") */
        div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h5 {{
            font-size: 1.0rem !important; font-weight: 600 !important; line-height: 1.3 !important;
            margin: 1.5rem 0 0.5rem 0 !important; color: {COLOR_PRIMARY_TEXT_DARK} !important; text-align: left !important;
        }}
        /* Main Content Markdown H6 (e.g. for individual plot titles) */
        div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h6 {{
            font-size: 0.95rem !important; font-weight: 600 !important; 
            line-height: 1.3 !important; margin-top: 1rem !important; margin-bottom: 0.5rem !important; 
            color: {COLOR_PRIMARY_TEXT_DARK} !important; /* Changed to primary for better visibility */
             text-align: left;
        }}

        /* Sidebar Specific Styles */
        [data-testid="stSidebar"] {{ 
            background-color: {COLOR_SIDEBAR_BACKGROUND_LIGHT} !important; 
            /* color: {COLOR_PRIMARY_TEXT_DARK} !important; Main color set by .main, overridden by specific selectors below */
            padding: 1.5rem; border-right: 1px solid {COLOR_BORDER_DARKER_LIGHT} !important; 
            font-size: 0.95rem; 
        }}
        [data-testid="stSidebar"] * {{ /* Ensure all text in sidebar defaults to dark */
            color: {COLOR_PRIMARY_TEXT_DARK} !important;
        }}
        [data-testid="stSidebar"] h2 {{ 
            font-size: 1.4rem !important; color: {COLOR_PRIMARY_TEXT_DARK} !important;
            margin-top: 1.5rem !important; margin-bottom: 0.5rem !important;
            padding-bottom: 0.3rem !important; border-bottom: 1px solid {COLOR_BORDER_DARKER_LIGHT} !important;
        }}
        [data-testid="stSidebar"] h3 {{ 
            font-size: 1.1rem !important; text-align: center !important; 
            margin-bottom: 1.2rem !important; color: {COLOR_SECONDARY_TEXT_DARK} !important; 
            border-bottom: none !important; 
        }}
        [data-testid="stSidebar"] div[data-testid="stExpander"] h5 {{
            color: {COLOR_PRIMARY_TEXT_DARK} !important; text-align: left; font-size: 1.0rem !important; 
            font-weight: 600 !important; margin-top: 0.8rem !important; margin-bottom: 0.4rem !important; 
        }}
        [data-testid="stSidebar"] div[data-testid="stExpander"] h6 {{
            color: {COLOR_SECONDARY_TEXT_DARK} !important; text-align: left; font-size: 0.9rem !important;
            font-weight: 600 !important; margin-top: 1rem !important; margin-bottom: 0.3rem !important;
        }}
        [data-testid="stSidebar"] .stMarkdownContainer > p, [data-testid="stSidebar"] .stCaption {{ 
             color: {COLOR_SECONDARY_TEXT_DARK} !important; font-size: 0.85rem !important;
             line-height: 1.3 !important; margin-top: 0.2rem !important; margin-bottom: 0.5rem !important;
        }}

        /* Buttons */
        .stButton>button {{ 
            background-color: {COLOR_ACCENT_BUTTON_LIGHT_THEME} !important; color: #FFFFFF !important; 
            border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.95rem; font-weight: 500; 
            transition: all 0.2s ease-in-out; border: 1px solid {COLOR_ACCENT_BUTTON_LIGHT_THEME} !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
        }}
        .stButton>button:hover, .stButton>button:focus {{ 
            background-color: {COLOR_ACCENT_BUTTON_HOVER_LIGHT_THEME} !important; 
            border-color: {COLOR_ACCENT_BUTTON_HOVER_LIGHT_THEME} !important;
            transform: translateY(-1px); box-shadow: 0 3px 7px rgba(0,0,0,0.2); outline: none; 
        }}
        .stButton>button:disabled {{ 
            background-color: #B0BEC5 !important; color: #78909C !important; 
            border-color: #B0BEC5 !important; cursor: not-allowed; box-shadow: none; 
        }}
        
        /* Sidebar Widget Labels */
        [data-testid="stSidebar"] div[data-testid*="stWidgetLabel"] label p, 
        [data-testid="stSidebar"] label[data-baseweb="checkbox"] span, 
        [data-testid="stSidebar"] .stSelectbox > label, 
        [data-testid="stSidebar"] .stMultiSelect > label {{
            color: {COLOR_PRIMARY_TEXT_DARK} !important; font-weight: 600 !important;
            font-size: 0.92rem !important; padding-bottom: 3px !important; display: block !important; 
        }}

        /* Sidebar Widget Input Fields */
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"], 
        [data-testid="stSidebar"] .stNumberInput div input, 
        [data-testid="stSidebar"] .stTextInput div input,
        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {{ 
            background-color: #FFFFFF !important; color: {COLOR_PRIMARY_TEXT_DARK} !important; 
            border-radius: 6px !important; padding: 0.4rem 0.5rem !important; 
            margin-bottom: 0.6rem !important; font-size: 0.9rem !important; 
            border: 1px solid {COLOR_BORDER_DARKER_LIGHT} !important; height: auto !important; 
        }}
        [data-testid="stSidebar"] .stNumberInput button {{ 
            background-color: #CFD8DC !important; color: {COLOR_PRIMARY_TEXT_DARK} !important; 
            border: 1px solid {COLOR_BORDER_DARKER_LIGHT} !important;
        }}
        [data-testid="stSidebar"] .stNumberInput button:hover {{ background-color: #B0BEC5 !important; }}

        /* Sidebar Buttons (specific overrides) */
        [data-testid="stSidebar"] .stButton>button {{ 
            background-color: {COLOR_BUTTON_SIDEBAR_DEFAULT_BG_LIGHT} !important; 
            color: #FFFFFF !important; border-color: {COLOR_BUTTON_SIDEBAR_DEFAULT_BG_LIGHT} !important;
        }}
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus {{ 
            background-color: {COLOR_BUTTON_SIDEBAR_DEFAULT_HOVER_BG_LIGHT} !important;
            border-color: {COLOR_BUTTON_SIDEBAR_DEFAULT_HOVER_BG_LIGHT} !important;
        }}
         [data-testid="stSidebar"] .stButton button[kind="primary"] {{ 
             background-color: {COLOR_ACCENT_UI_LIGHT_THEME} !important;
             border-color: {COLOR_ACCENT_UI_LIGHT_THEME} !important;
             color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {{
             background-color: {COLOR_ACCENT_BUTTON_HOVER_LIGHT_THEME} !important;
             border-color: {COLOR_ACCENT_BUTTON_HOVER_LIGHT_THEME} !important;
        }}
        [data-testid="stSidebar"] .stButton button[kind="secondary"] {{ 
             background-color: {COLOR_BUTTON_SECONDARY_BG_LIGHT} !important; color: {COLOR_PRIMARY_TEXT_DARK} !important;
             border: 1px solid {COLOR_BORDER_DARKER_LIGHT} !important;
        }}
         [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {{
             background-color: {COLOR_BUTTON_SECONDARY_HOVER_BG_LIGHT} !important;
        }}

        /* Metric Display */
        .stMetric {{ 
            background-color: {COLOR_CONTENT_BACKGROUND_LIGHT} !important; 
            border-radius: 8px; padding: 1rem 1.25rem; margin: 0.5rem 0; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid {COLOR_BORDER_SUBTLE_LIGHT} !important; 
        }}
        .stMetric > div[data-testid="stMetricLabel"] {{ 
            font-size: 1.0rem !important; color: {COLOR_SECONDARY_TEXT_DARK} !important; 
            font-weight: 600 !important; margin-bottom: 0.3rem !important;
        }}
        .stMetric div[data-testid="stMetricValue"] {{ 
            font-size: 2.2rem !important; color: {COLOR_PRIMARY_TEXT_DARK} !important; 
            font-weight: 700 !important; line-height: 1.1 !important;
        }} 
        
        /* Expanders and Tabs */
        .stExpander {{ 
            background-color: {COLOR_CONTENT_BACKGROUND_LIGHT} !important; 
            border-radius: 8px; margin: 1rem 0; border: 1px solid {COLOR_BORDER_SUBTLE_LIGHT} !important; 
        }}
        .stExpander header {{ font-size: 1rem; font-weight: 500; color: {COLOR_PRIMARY_TEXT_DARK} !important; padding: 0.5rem 1rem; }}
        
        .stTabs [data-baseweb="tab-list"] {{ 
            background-color: {COLOR_SIDEBAR_BACKGROUND_LIGHT} !important; 
            border-radius: 8px; padding: 0.5rem; display: flex; justify-content: center; 
            gap: 0.5rem; border-bottom: 2px solid {COLOR_BORDER_DARKER_LIGHT} !important;
        }}
        .stTabs [data-baseweb="tab"] {{ 
            color: {COLOR_SECONDARY_TEXT_DARK} !important; padding: 0.6rem 1.2rem; border-radius: 6px; 
            font-weight: 500; font-size: 0.95rem; transition: all 0.2s ease-in-out; 
            border: none; border-bottom: 2px solid transparent; 
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{ 
            background-color: transparent !important; color: {COLOR_ACCENT_UI_LIGHT_THEME} !important; 
            border-bottom: 2px solid {COLOR_ACCENT_UI_LIGHT_THEME} !important; font-weight:600; 
        }}
        .stTabs [data-baseweb="tab"]:hover {{ 
            background-color: #CFD8DC !important; color: {COLOR_PRIMARY_TEXT_DARK} !important; 
        }}

        .stPlotlyChart {{ border-radius: 6px; }} 
        .stDataFrame {{ 
            border-radius: 8px; font-size: 0.875rem; border: 1px solid {COLOR_BORDER_SUBTLE_LIGHT} !important; 
            background-color: {COLOR_CONTENT_BACKGROUND_LIGHT} !important;
        }}
        .stDataFrame thead th {{ 
            background-color: #E8EAF6 !important; color: {COLOR_PRIMARY_TEXT_DARK} !important; font-weight: 600; 
        }}
        .stDataFrame tbody tr:nth-child(even) {{ background-color: #FAFAFA !important; }}
        .stDataFrame tbody tr:hover {{ background-color: #E0E0E0 !important; }}

        .spinner::after {{ border: 4px solid #CFD8DC; border-top: 4px solid {COLOR_ACCENT_UI_LIGHT_THEME}; }}
        .onboarding-modal {{ 
            background-color: {COLOR_CONTENT_BACKGROUND_LIGHT} !important; 
            border: 1px solid {COLOR_BORDER_DARKER_LIGHT} !important; 
            color: {COLOR_PRIMARY_TEXT_DARK} !important;
        }}
        .onboarding-modal h3 {{ color: {COLOR_PRIMARY_TEXT_DARK} !important; }}
        .onboarding-modal p, .onboarding-modal ul {{ color: {COLOR_SECONDARY_TEXT_DARK} !important; }}

        .alert-critical {{ border-left: 5px solid {COLOR_CRITICAL_RED_BORDER}; background-color: {COLOR_CRITICAL_RED_BG_LIGHT}; }} 
        .alert-warning {{ border-left: 5px solid {COLOR_WARNING_AMBER_BORDER}; background-color: {COLOR_WARNING_AMBER_BG_LIGHT}; }}
        .alert-positive {{ border-left: 5px solid {COLOR_POSITIVE_GREEN_BORDER}; background-color: {COLOR_POSITIVE_GREEN_BG_LIGHT}; }}
        .alert-info {{ border-left: 5px solid {COLOR_INFO_BLUE_BORDER}; background-color: {COLOR_INFO_BLUE_BG_LIGHT}; }}
        /* Ensure text inside alerts is dark for readability on light backgrounds */
        .alert-critical .insight-title, .alert-critical .insight-text,
        .alert-warning .insight-title, .alert-warning .insight-text,
        .alert-positive .insight-title, .alert-positive .insight-text,
        .alert-info .insight-title, .alert-info .insight-text {{ color: {COLOR_PRIMARY_TEXT_DARK} !important; }}
        
        .event-item {{ background-color: #E8EAF6; border: 1px solid {COLOR_BORDER_SUBTLE_LIGHT}; }} 
        .event-text {{ color: {COLOR_PRIMARY_TEXT_DARK} !important; }}
        .remove-event-btn button {{ 
            background-color: {COLOR_BUTTON_REMOVE_BG_LIGHT} !important; color: white !important; 
            padding: 0.1rem 0.4rem !important; font-size: 0.75rem !important; line-height: 1 !important; 
            border-radius: 3px !important; min-height: auto !important; margin-left: 0.5rem !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR RENDERING ---
def render_settings_sidebar():
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Workplace Optimizer</h3>", unsafe_allow_html=True)
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
                    export_metrics_map_csv_sb = { 
                        'task_compliance.data': 'task_compliance_percent',
                        'collaboration_metric.data': 'collaboration_metric_percent',
                        'operational_recovery': 'operational_recovery_percent',
                        'worker_wellbeing.scores': 'worker_wellbeing_index',
                        'psychological_safety': 'psychological_safety_score',
                        'productivity_loss': 'productivity_loss_percent',
                        'task_completion_rate': 'task_completion_rate_percent',
                        'worker_wellbeing.team_cohesion_scores': 'team_cohesion_score',
                        'worker_wellbeing.perceived_workload_scores': 'perceived_workload_score_0_10'
                    }
                    for path_csv_sb, col_name_csv_sb in export_metrics_map_csv_sb.items():
                        clean_col_name = col_name_csv_sb.replace(' (%)','_percent').replace(' (0-10)','_0_10').replace(' ','_').lower()
                        csv_data_dict_sb[clean_col_name] = _prepare_timeseries_for_export(safe_get(sim_res_csv_exp_sb, path_csv_sb, []), num_steps_csv_exp_sb)
                    
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
    
    mpi_sl = _get_config_value_sl_main(config_sl, {}, 'MINUTES_PER_INTERVAL', 2, data_type=float) 
    if mpi_sl <= 0: mpi_sl = 2.0; logger.error("MINUTES_PER_INTERVAL was invalid (<=0), defaulted to 2.0 for calculations.")
    config_sl['SHIFT_DURATION_INTERVALS'] = int(config_sl['SHIFT_DURATION_MINUTES'] // mpi_sl)

    processed_events_sl = []
    for event_sl_ui_item in scheduled_events_from_ui_sl:
        evt_sl_item = event_sl_ui_item.copy() 
        if 'step' not in evt_sl_item and 'Start Time (min)' in evt_sl_item:
            start_time_min_evt = _get_config_value_sl_main(evt_sl_item, {}, 'Start Time (min)', 0, data_type=float)
            if mpi_sl > 0 : evt_sl_item['step'] = int(start_time_min_evt // mpi_sl)
            else: evt_sl_item['step'] = 0 
        processed_events_sl.append(evt_sl_item)
    config_sl['SCHEDULED_EVENTS'] = processed_events_sl
    
    if 'WORK_AREAS' in config_sl and isinstance(config_sl['WORK_AREAS'], dict) and config_sl['WORK_AREAS']:
        current_total_workers_in_cfg = sum(_get_config_value_sl_main(z_cfg, {}, 'workers', 0, data_type=int) for z_cfg in config_sl['WORK_AREAS'].values() if isinstance(z_cfg, dict))
        target_team_size_for_dist = config_sl['TEAM_SIZE']
        if current_total_workers_in_cfg != target_team_size_for_dist and target_team_size_for_dist >= 0:
            logger.info(f"Redistributing workers. Config sum: {current_total_workers_in_cfg}, Target team: {target_team_size_for_dist}")
            distributable_areas = {k:v for k,v in config_sl['WORK_AREAS'].items() if isinstance(v,dict) and not v.get('is_rest_area',False)}
            if not distributable_areas: 
                logger.warning("No non-rest work areas for worker redistribution. Using all areas if any.")
                distributable_areas = {k:v for k,v in config_sl['WORK_AREAS'].items() if isinstance(v,dict)}

            if target_team_size_for_dist == 0:
                for zone_k_sl_zero in config_sl['WORK_AREAS']: 
                    if isinstance(config_sl['WORK_AREAS'][zone_k_sl_zero], dict):
                        config_sl['WORK_AREAS'][zone_k_sl_zero]['workers'] = 0
            elif distributable_areas:
                sum_workers_in_dist_areas = sum(_get_config_value_sl_main(z_dist, {}, 'workers', 0, data_type=int) for z_dist in distributable_areas.values())
                if sum_workers_in_dist_areas > 0: 
                    ratio_sl = target_team_size_for_dist / sum_workers_in_dist_areas
                    accumulated_sl = 0; sorted_dist_zone_keys = sorted(distributable_areas.keys())
                    for i_zone_sl, zone_k_sl in enumerate(sorted_dist_zone_keys):
                        original_workers_dist = _get_config_value_sl_main(config_sl['WORK_AREAS'][zone_k_sl], {}, 'workers', 0, data_type=int)
                        if i_zone_sl < len(sorted_dist_zone_keys) - 1:
                            new_w_sl = int(round(original_workers_dist * ratio_sl))
                            config_sl['WORK_AREAS'][zone_k_sl]['workers'] = new_w_sl; accumulated_sl += new_w_sl
                        else: config_sl['WORK_AREAS'][zone_k_sl]['workers'] = max(0, target_team_size_for_dist - accumulated_sl)
                elif len(distributable_areas) > 0: 
                    base_w_sl, rem_w_sl = divmod(target_team_size_for_dist, len(distributable_areas))
                    assign_count_sl = 0
                    for zone_k_sl_even in distributable_areas:
                        config_sl['WORK_AREAS'][zone_k_sl_even]['workers'] = base_w_sl + (1 if assign_count_sl < rem_w_sl else 0); assign_count_sl +=1
            all_area_keys = set(config_sl['WORK_AREAS'].keys()); dist_area_keys = set(distributable_areas.keys())
            non_dist_keys = all_area_keys - dist_area_keys
            for r_zone_k in non_dist_keys:
                if isinstance(config_sl['WORK_AREAS'][r_zone_k], dict) and config_sl['WORK_AREAS'][r_zone_k].get('is_rest_area'):
                     config_sl['WORK_AREAS'][r_zone_k]['workers'] = 0

    validate_config(config_sl)
    logger.info(f"Running simulation with config: Team Size={config_sl['TEAM_SIZE']}, Duration={config_sl['SHIFT_DURATION_MINUTES']}min ({config_sl['SHIFT_DURATION_INTERVALS']} intervals of {mpi_sl}min), Scheduled Events: {len(config_sl['SCHEDULED_EVENTS'])}, Initiative: {team_initiative_sl}", extra={'user_action': 'Run Simulation - Start'})
    
    expected_keys_sl_run = ['team_positions_df', 'task_compliance', 'collaboration_metric','operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 'worker_wellbeing', 'psychological_safety', 'feedback_impact', 'downtime_events_log', 'task_completion_rate']
    sim_results_tuple_sl_final = simulate_workplace_operations(num_team_members=config_sl['TEAM_SIZE'], num_steps=config_sl['SHIFT_DURATION_INTERVALS'], scheduled_events=config_sl['SCHEDULED_EVENTS'], team_initiative=team_initiative_sl, config=config_sl)
    
    if not isinstance(sim_results_tuple_sl_final, tuple) or len(sim_results_tuple_sl_final) != len(expected_keys_sl_run):
        logger.critical(f"Simulation returned unexpected data format. Expected tuple of {len(expected_keys_sl_run)}, got {type(sim_results_tuple_sl_final)} of len {len(sim_results_tuple_sl_final) if isinstance(sim_results_tuple_sl_final,(list,tuple)) else 'N/A'}.", extra={'user_action':'Sim Format Error'})
        raise TypeError("Simulation returned unexpected data format.")
        
    simulation_output_dict_sl_final_run = dict(zip(expected_keys_sl_run, sim_results_tuple_sl_final))
    simulation_output_dict_sl_final_run['config_params'] = {
        'TEAM_SIZE': config_sl['TEAM_SIZE'], 'SHIFT_DURATION_MINUTES': config_sl['SHIFT_DURATION_MINUTES'],
        'SHIFT_DURATION_INTERVALS': config_sl['SHIFT_DURATION_INTERVALS'], 'MINUTES_PER_INTERVAL': mpi_sl, 
        'SCHEDULED_EVENTS': config_sl['SCHEDULED_EVENTS'], 'TEAM_INITIATIVE': team_initiative_sl, 
        'WORK_AREAS_EFFECTIVE': config_sl.get('WORK_AREAS', {}).copy(),
        'ENTRY_EXIT_POINTS': config_sl.get('ENTRY_EXIT_POINTS', []).copy(),
        'FACILITY_SIZE': config_sl.get('FACILITY_SIZE', (100,80))
    }
    
    disruption_steps_final_sl_run = [evt.get('step') for evt in config_sl['SCHEDULED_EVENTS'] if isinstance(evt,dict) and "Disruption" in evt.get("Event Type","") and isinstance(evt.get('step'),int)]
    simulation_output_dict_sl_final_run['config_params']['DISRUPTION_EVENT_STEPS'] = sorted(list(set(disruption_steps_final_sl_run)))

    save_simulation_data(simulation_output_dict_sl_final_run) 
    return simulation_output_dict_sl_final_run

# Helper specifically for run_simulation_logic's config access
def _get_config_value_sl_main(primary_conf, secondary_conf, key, default, data_type=None):
    val = secondary_conf.get(key, primary_conf.get(key, default))
    if data_type:
        try:
            if data_type == float: return float(val)
            if data_type == int: return int(val)
        except (ValueError, TypeError): return default
    return val

# --- TIME RANGE INPUT WIDGETS ---
def time_range_input_section(tab_key_prefix: str, max_minutes_for_range_ui: int, st_col_obj = st, interval_duration_min_ui: int = 2):
    start_time_key_ui = f"{tab_key_prefix}_start_time_min"
    end_time_key_ui = f"{tab_key_prefix}_end_time_min"

    interval_duration_min_ui = float(interval_duration_min_ui) if isinstance(interval_duration_min_ui, (int, float)) and interval_duration_min_ui > 0 else 2.0
    max_minutes_for_range_ui = float(max_minutes_for_range_ui) if isinstance(max_minutes_for_range_ui, (int, float)) and max_minutes_for_range_ui >=0 else 0.0
    
    current_start_ui_val = float(st.session_state.get(start_time_key_ui, 0.0))
    current_end_ui_val = float(st.session_state.get(end_time_key_ui, max_minutes_for_range_ui))
    current_start_ui_val = max(0.0, min(current_start_ui_val, max_minutes_for_range_ui))
    current_end_ui_val = max(current_start_ui_val, min(current_end_ui_val, max_minutes_for_range_ui))
    
    st.session_state[start_time_key_ui], st.session_state[end_time_key_ui] = current_start_ui_val, current_end_ui_val
    
    prev_start_ui_val_state, prev_end_ui_val_state = current_start_ui_val, current_end_ui_val
    cols_ui_time_range = st_col_obj.columns(2)
    
    new_start_time_val_ui_widget = cols_ui_time_range[0].number_input( "Start Time (min)", min_value=0.0, max_value=max_minutes_for_range_ui, value=current_start_ui_val, step=interval_duration_min_ui, key=f"widget_num_input_{start_time_key_ui}", help=f"Range: 0 to {int(max_minutes_for_range_ui)} min.")
    st.session_state[start_time_key_ui] = float(new_start_time_val_ui_widget)
    
    end_time_min_for_widget_val_ui = st.session_state[start_time_key_ui]
    new_end_time_val_ui_widget = cols_ui_time_range[1].number_input("End Time (min)", min_value=end_time_min_for_widget_val_ui, max_value=max_minutes_for_range_ui, value=current_end_ui_val, step=interval_duration_min_ui, key=f"widget_num_input_{end_time_key_ui}", help=f"Range: {int(end_time_min_for_widget_val_ui)} to {int(max_minutes_for_range_ui)} min.")
    st.session_state[end_time_key_ui] = float(new_end_time_val_ui_widget)

    if st.session_state[end_time_key_ui] < st.session_state[start_time_key_ui]: st.session_state[end_time_key_ui] = st.session_state[start_time_key_ui]
    
    if abs(prev_start_ui_val_state - st.session_state[start_time_key_ui]) > 1e-6 or \
       abs(prev_end_ui_val_state - st.session_state[end_time_key_ui]) > 1e-6:
        st.rerun()
        
    return int(st.session_state[start_time_key_ui]), int(st.session_state[end_time_key_ui])

# --- MAIN APPLICATION FUNCTION ---
def main():
    st.title("Workplace Shift Optimization Dashboard")
    
    mpi_global_app_main = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)
    if not isinstance(mpi_global_app_main, (int, float)) or mpi_global_app_main <= 0: mpi_global_app_main = 2.0 
    else: mpi_global_app_main = float(mpi_global_app_main)

    app_state_defaults_main_app = {
        'simulation_results': None, 'show_tour': False, 'show_help_glossary': False,
        'sb_team_size_num': DEFAULT_CONFIG['TEAM_SIZE'], 'sb_shift_duration_num': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
        'sb_scheduled_events_list': list(DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', [])),
        'sb_team_initiative_selectbox': "Standard Operations",
        'sb_high_contrast_checkbox': False, 'sb_use_3d_distribution_checkbox': False, 'sb_debug_mode_checkbox': False,
        'form_event_type': "Major Disruption", 'form_event_start': 0, 'form_event_duration': max(mpi_global_app_main, 10.0),
    }
    default_max_mins_main_app_init = float(DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - mpi_global_app_main) if DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] > mpi_global_app_main else 0.0
    for prefix_main_app_init in ['op', 'ww', 'dt']:
        app_state_defaults_main_app[f'{prefix_main_app_init}_start_time_min'] = 0.0
        app_state_defaults_main_app[f'{prefix_main_app_init}_end_time_min'] = default_max_mins_main_app_init
    for key_main_app_init, val_main_app_init in app_state_defaults_main_app.items():
        if key_main_app_init not in st.session_state: st.session_state[key_main_app_init] = val_main_app_init
            
    run_simulation_button_main_app_call, load_data_button_main_app_call = render_settings_sidebar()
    current_high_contrast_main_app_val = st.session_state.sb_high_contrast_checkbox
    use_3d_main_app_val = st.session_state.sb_use_3d_distribution_checkbox

    active_mpi_main_app_val = mpi_global_app_main
    max_mins_ui_main_app_val = default_max_mins_main_app_init
    simulation_disruption_steps_absolute_main_val = []

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        sim_cfg_main_app_active = st.session_state.simulation_results.get('config_params', {})
        active_mpi_main_app_val = float(sim_cfg_main_app_active.get('MINUTES_PER_INTERVAL', mpi_global_app_main))
        if active_mpi_main_app_val <= 0 : active_mpi_main_app_val = 2.0
        sim_intervals_main_app_active_val = sim_cfg_main_app_active.get('SHIFT_DURATION_INTERVALS', 0)
        max_mins_ui_main_app_val = float(max(0, sim_intervals_main_app_active_val * active_mpi_main_app_val - active_mpi_main_app_val)) if sim_intervals_main_app_active_val > 0 else 0.0
        simulation_disruption_steps_absolute_main_val = sim_cfg_main_app_active.get('DISRUPTION_EVENT_STEPS', [])
    else:
        shift_duration_from_sidebar_main_val = st.session_state.sb_shift_duration_num
        sim_intervals_main_app_active_val = int(shift_duration_from_sidebar_main_val // active_mpi_main_app_val) if active_mpi_main_app_val > 0 else 0
        max_mins_ui_main_app_val = float(max(0, sim_intervals_main_app_active_val * active_mpi_main_app_val - active_mpi_main_app_val)) if sim_intervals_main_app_active_val > 0 else 0.0
        if active_mpi_main_app_val > 0:
            for event_main_ui_item_cfg_val in st.session_state.sb_scheduled_events_list:
                if "Disruption" in event_main_ui_item_cfg_val.get("Event Type","") and isinstance(event_main_ui_item_cfg_val.get("Start Time (min)"), (int,float)):
                    simulation_disruption_steps_absolute_main_val.append(int(event_main_ui_item_cfg_val["Start Time (min)"] // active_mpi_main_app_val))
            simulation_disruption_steps_absolute_main_val = sorted(list(set(simulation_disruption_steps_absolute_main_val)))
    
    for prefix_main_ui_clamp_val_final in ['op', 'ww', 'dt']:
        st.session_state[f"{prefix_main_ui_clamp_val_final}_start_time_min"] = max(0.0, min(float(st.session_state.get(f"{prefix_main_ui_clamp_val_final}_start_time_min",0.0)), max_mins_ui_main_app_val))
        st.session_state[f"{prefix_main_ui_clamp_val_final}_end_time_min"] = max(st.session_state[f"{prefix_main_ui_clamp_val_final}_start_time_min"], min(float(st.session_state.get(f"{prefix_main_ui_clamp_val_final}_end_time_min",max_mins_ui_main_app_val)), max_mins_ui_main_app_val))

    if run_simulation_button_main_app_call:
        with st.spinner("🚀 Simulating workplace operations... This may take a moment."):
            try:
                results_run_main_final = run_simulation_logic(st.session_state.sb_team_size_num, st.session_state.sb_shift_duration_num, list(st.session_state.sb_scheduled_events_list), st.session_state.sb_team_initiative_selectbox)
                st.session_state.simulation_results = results_run_main_final
                new_cfg_run_main_final = results_run_main_final['config_params']
                new_mpi_run_main_final = new_cfg_run_main_final.get('MINUTES_PER_INTERVAL', 2.0)
                new_sim_intervals_run_main_final = new_cfg_run_main_final.get('SHIFT_DURATION_INTERVALS',0)
                new_max_mins_run_main_final = float(max(0, new_sim_intervals_run_main_final * new_mpi_run_main_final - new_mpi_run_main_final)) if new_sim_intervals_run_main_final > 0 else 0.0
                for pfx_run_final in ['op','ww','dt']: st.session_state[f"{pfx_run_final}_start_time_min"]=0.0; st.session_state[f"{pfx_run_final}_end_time_min"]=new_max_mins_run_main_final
                st.success("✅ Simulation completed successfully!"); logger.info("Sim run success."); st.rerun()
            except Exception as e_run_main_final: logger.error(f"Sim Run Error: {e_run_main_final}", exc_info=True); st.error(f"❌ Sim failed: {e_run_main_final}"); st.session_state.simulation_results = None
    if load_data_button_main_app_call:
        with st.spinner("🔄 Loading saved simulation data..."):
            try:
                loaded_main_final = load_simulation_data()
                if loaded_main_final and isinstance(loaded_main_final, dict) and 'config_params' in loaded_main_final:
                    st.session_state.simulation_results = loaded_main_final; cfg_ld_main_final = loaded_main_final['config_params']
                    st.session_state.sb_team_size_num = cfg_ld_main_final.get('TEAM_SIZE', DEFAULT_CONFIG['TEAM_SIZE'])
                    st.session_state.sb_shift_duration_num = cfg_ld_main_final.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'])
                    st.session_state.sb_scheduled_events_list = list(cfg_ld_main_final.get('SCHEDULED_EVENTS', []))
                    st.session_state.sb_team_initiative_selectbox = cfg_ld_main_final.get('TEAM_INITIATIVE', "Standard Operations")
                    mpi_loaded = cfg_ld_main_final.get('MINUTES_PER_INTERVAL', 2.0)
                    intervals_loaded = cfg_ld_main_final.get('SHIFT_DURATION_INTERVALS',0)
                    max_ld_main_final = float(max(0, intervals_loaded * mpi_loaded - mpi_loaded)) if intervals_loaded > 0 else 0.0
                    for pfx_ld_final in ['op','ww','dt']: st.session_state[f"{pfx_ld_final}_start_time_min"]=0.0; st.session_state[f"{pfx_ld_final}_end_time_min"]=max_ld_main_final
                    st.success("✅ Data loaded!"); logger.info("Load success."); st.rerun()
                else: st.error("❌ Failed to load data or data invalid."); logger.warning("Load fail/invalid.")
            except Exception as e_load_main_final: logger.error(f"Load Data Error: {e_load_main_final}", exc_info=True); st.error(f"❌ Load failed: {e_load_main_final}"); st.session_state.simulation_results = None
    
    if st.session_state.get('show_tour', False): 
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3><p>Welcome! This dashboard helps you monitor and analyze workplace shift operations. Use the sidebar to configure simulations and navigate. The main area displays results across several tabs: Overview, Operational Metrics, Worker Well-being (including psychosocial factors and spatial dynamics), Downtime Analysis, and a Glossary. Interactive charts and actionable insights will guide you in optimizing operations.</p><p>Start by running a new simulation or loading previous data from the sidebar!</p></div>""", unsafe_allow_html=True)
        if st.button("Got it!", key="tour_modal_close_button_main_area_final"): st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help_glossary', False): 
        with st.container(): st.markdown(""" <div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3> <p>This dashboard provides insights into simulated workplace operations. Use the sidebar to configure and run simulations or load previously saved data. Navigate through the analysis using the main tabs above.</p><h4>Metric Definitions:</h4> <ul style="font-size: 0.85rem; list-style-type: disc; padding-left: 20px;"> <li><b>Task Compliance Score:</b> Percentage of tasks completed correctly and on time.</li><li><b>Collaboration Metric:</b> A score indicating teamwork potential and interaction levels.</li><li><b>Operational Recovery Score:</b> Ability to maintain output after disruptions.</li><li><b>Worker Well-Being Index:</b> Composite score of fatigue, stress, and satisfaction.</li><li><b>Psychological Safety Score:</b> Comfort level in reporting issues or suggesting improvements.</li><li><b>Team Cohesion Index:</b> Measure of bonds and sense of belonging within a team.</li><li><b>Perceived Workload Index:</b> Indicator of task demand (0-10 scale).</li><li><b>Uptime:</b> Percentage of time equipment is operational.</li><li><b>Throughput:</b> Percentage of maximum production rate achieved.</li><li><b>Quality Rate:</b> Percentage of products meeting quality standards.</li><li><b>OEE (Overall Equipment Effectiveness):</b> Combined score of Uptime, Throughput, and Quality.</li><li><b>Productivity Loss:</b> Percentage of potential output lost.</li><li><b>Downtime Events Log:</b> A raw log of individual downtime occurrences, each with step, duration, and cause. Aggregated for trend plots.</li><li><b>Task Completion Rate:</b> Percentage of tasks completed per time interval.</li></ul><p>For further assistance, refer to documentation or contact support.</p></div> """, unsafe_allow_html=True) 
        if st.button("Understood", key="help_modal_close_button_main_area_final"): st.session_state.show_help_glossary = False; st.rerun()

    tab_names_ui_final = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Well-being", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs_st_objs_final = st.tabs(tab_names_ui_final)
    plot_cfg_interactive_final_ui = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}
    plot_cfg_minimal_final_ui = {'displayModeBar': False}

    with tabs_st_objs_final[0]: # Overview Tab
        st.header("📊 Key Performance Indicators & Actionable Insights", divider=THEMED_DIVIDER_COLOR)
        if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
            sim_data_ov_final = st.session_state.simulation_results
            sim_cfg_ov_final = sim_data_ov_final.get('config_params', DEFAULT_CONFIG)
            effective_cfg_ov_final = {**DEFAULT_CONFIG, **sim_cfg_ov_final}
            target_compliance_ov_final = float(effective_cfg_ov_final.get('TARGET_COMPLIANCE', 75.0))
            target_collab_ov_final = float(effective_cfg_ov_final.get('TARGET_COLLABORATION', 65.0))
            target_wellbeing_ov_final = float(effective_cfg_ov_final.get('TARGET_WELLBEING', 75.0))
            downtime_log_ov_final = safe_get(sim_data_ov_final, 'downtime_events_log', [])
            downtime_durations_ov_final = [evt.get('duration', 0.0) for evt in downtime_log_ov_final if isinstance(evt, dict)]
            compliance_val_ov_final = safe_stat(safe_get(sim_data_ov_final, 'task_compliance.data', []), np.mean, 0.0)
            collab_val_ov_final = safe_stat(safe_get(sim_data_ov_final, 'collaboration_metric.data', []), np.mean, 0.0)
            wellbeing_val_ov_final = safe_stat(safe_get(sim_data_ov_final, 'worker_wellbeing.scores', []), np.mean, 0.0)
            downtime_total_ov_final = sum(downtime_durations_ov_final)
            shift_duration_ov_final = float(effective_cfg_ov_final.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
            dt_target_percent_ov_final = float(effective_cfg_ov_final.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT', 0.05))
            dt_target_abs_ov_final = shift_duration_ov_final * dt_target_percent_ov_final
            cols_metrics_ov_final = st.columns(4)
            cols_metrics_ov_final[0].metric("Task Compliance", f"{compliance_val_ov_final:.1f}%", f"{compliance_val_ov_final-target_compliance_ov_final:.1f}% vs Target")
            cols_metrics_ov_final[1].metric("Collaboration Metric", f"{collab_val_ov_final:.1f}%", f"{collab_val_ov_final-target_collab_ov_final:.1f}% vs Target")
            cols_metrics_ov_final[2].metric("Worker Well-Being", f"{wellbeing_val_ov_final:.1f}%", f"{wellbeing_val_ov_final-target_wellbeing_ov_final:.1f}% vs Target")
            cols_metrics_ov_final[3].metric("Total Downtime", f"{downtime_total_ov_final:.1f} min", f"{downtime_total_ov_final-dt_target_abs_ov_final:.1f} min vs Target ({dt_target_abs_ov_final:.0f}min)", delta_color="inverse")
            try:
                summary_figs_ov_final = plot_key_metrics_summary(compliance=compliance_val_ov_final, proximity=collab_val_ov_final, wellbeing=wellbeing_val_ov_final, downtime=downtime_total_ov_final, target_compliance=target_compliance_ov_final, target_proximity=target_collab_ov_final, target_wellbeing=target_wellbeing_ov_final, target_downtime=dt_target_abs_ov_final, high_contrast=current_high_contrast_main_app_val)
                if summary_figs_ov_final and isinstance(summary_figs_ov_final, list):
                    cols_gauges_ov_final = st.columns(min(len(summary_figs_ov_final), 4) or 1)
                    for i_gauge_ov_final, fig_gauge_ov_final in enumerate(summary_figs_ov_final): 
                        if fig_gauge_ov_final: cols_gauges_ov_final[i_gauge_ov_final % len(cols_gauges_ov_final)].plotly_chart(fig_gauge_ov_final, use_container_width=True, config=plot_cfg_minimal_final_ui)
                else: st.caption("Overview gauge charts could not be generated.")
            except Exception as e_gauge_ov_final: logger.error(f"Overview Gauges Plot Error: {e_gauge_ov_final}", exc_info=True); st.error(f"⚠️ Error rendering gauges: {e_gauge_ov_final}")
            st.markdown("---"); st.subheader("💡 Key Insights & Leadership Actions")
            actionable_insights_ov_final = get_actionable_insights(sim_data_ov_final, effective_cfg_ov_final)
            if actionable_insights_ov_final:
                for insight_ov in actionable_insights_ov_final: st.markdown(f'<div class="alert-{insight_ov["type"]}"><p class="insight-title">{insight_ov["title"]}</p><p class="insight-text">{insight_ov["text"]}</p></div>', unsafe_allow_html=True)
            else: st.info("✅ No critical alerts or specific insights identified based on current thresholds.", icon="👍")
            with st.expander("View Detailed Overview Data Table", expanded=False):
                num_steps_ov_table_final = effective_cfg_ov_final.get('SHIFT_DURATION_INTERVALS', 0)
                mpi_ov_table_final = effective_cfg_ov_final.get('MINUTES_PER_INTERVAL', mpi_global_app_main)
                if num_steps_ov_table_final > 0:
                    df_data_ov_table_final = {'Time (min)': [i * mpi_ov_table_final for i in range(num_steps_ov_table_final)]}
                    df_data_ov_table_final['Task Compliance (%)'] = _prepare_timeseries_for_export(safe_get(sim_data_ov_final, 'task_compliance.data', []), num_steps_ov_table_final)
                    df_data_ov_table_final['Collaboration Metric (%)'] = _prepare_timeseries_for_export(safe_get(sim_data_ov_final, 'collaboration_metric.data', []), num_steps_ov_table_final)
                    df_data_ov_table_final['Well-Being (%)'] = _prepare_timeseries_for_export(safe_get(sim_data_ov_final, 'worker_wellbeing.scores', []), num_steps_ov_table_final)
                    downtime_log_ov_table_final = safe_get(sim_data_ov_final, 'downtime_events_log', [])
                    df_data_ov_table_final['Downtime (min/interval)'] = aggregate_downtime_by_step(downtime_log_ov_table_final, num_steps_ov_table_final)
                    st.dataframe(pd.DataFrame(df_data_ov_table_final).style.format("{:.1f}", na_rep="-").set_table_styles([{'selector': 'th', 'props': [('background-color', '#E8EAF6'), ('color', COLOR_PRIMARY_TEXT_DARK)]}]), use_container_width=True, height=300)
                else: st.caption("No detailed overview data available (0 simulation steps).")
        else: st.info("ℹ️ Run a simulation or load data to view the Overview & Insights.", icon="📊")
    
    op_insights_html_main_final = f"<div class='alert-info insight-text' style='margin-top:1rem; background-color: {COLOR_INFO_BLUE_BG_LIGHT}; border-left-color: {COLOR_INFO_BLUE_BORDER};'><p class='insight-title' style='color: {COLOR_PRIMARY_TEXT_DARK};'>Review Operational Bottlenecks:</p><ul style='color: {COLOR_SECONDARY_TEXT_DARK};'><li><b>Low Compliance/OEE:</b> Investigate root causes for low Task Compliance or OEE components.</li><li><b>Recovery Performance:</b> Slow recovery post-disruption may need better contingency plans.</li><li><b>Collaboration Impact:</b> Low Collaboration Metric might indicate communication issues.</li></ul><p class='insight-title' style='color: {COLOR_PRIMARY_TEXT_DARK};'>Strategic Considerations:</p><p style='color: {COLOR_SECONDARY_TEXT_DARK};'>Use 'Operational Initiative' to simulate changes and compare against baseline.</p></div>"
    ww_static_insights_html_main_final = f"<h6 style='margin-top:1.5rem; color:{COLOR_PRIMARY_TEXT_DARK};'>💡 Considerations for Psychosocial Well-being:</h6><ul style='font-size:0.9rem; color: {COLOR_SECONDARY_TEXT_DARK}; padding-left:20px; margin-bottom:0;'><li><strong>Monitor Risk Factors:</strong> Review Well-being, Psych. Safety, Cohesion, Workload.</li><li><strong>Spatial Awareness:</strong> Correlate density/isolation with well-being.</li><li><strong>Evaluate Initiatives:</strong> Test strategies via 'Operational Initiative'.</li><li><strong>Empowerment & Control:</strong> Assess 'Increased Autonomy' impact.</li><li><strong>Prevent Burnout:</strong> Address sustained high workload/low well-being.</li></ul>" 
    dt_insights_html_main_final = f"<div class='alert-info insight-text' style='margin-top:1rem; background-color: {COLOR_INFO_BLUE_BG_LIGHT}; border-left-color: {COLOR_INFO_BLUE_BORDER};'><p class='insight-title' style='color: {COLOR_PRIMARY_TEXT_DARK};'>Focus Areas for Downtime Reduction:</p><ul style='color: {COLOR_SECONDARY_TEXT_DARK};'><li><strong>Prioritize by Cause:</strong> Use pie chart to find primary downtime reasons.</li><li><strong>Analyze Trend for Patterns:</strong> Look for recurring high downtime in trend plot.</li><li><strong>Incident Frequency vs. Severity:</strong> Address both systemic minor issues and major ones.</li><li><strong>Disruption Correlation:</strong> Check if downtime spikes correlate with operational metric drops.</li></ul></div>"

    tab_configs_main_final_app = [
        {"name": "📈 Operational Metrics", "key_prefix": "op", "plots": [
             {"title": "Task Compliance Score", "data_path": "task_compliance.data", "plot_func": plot_task_compliance_score, "extra_args_paths": {"forecast_data": "task_compliance.forecast", "z_scores": "task_compliance.z_scores"}},
             {"title": "Collaboration Metric", "data_path": "collaboration_metric.data", "plot_func": plot_collaboration_proximity_index, "extra_args_paths": {"forecast_data": "collaboration_metric.forecast"}},
             {"is_subheader": True, "title": "Additional Operational Metrics"},
             {"title": "Operational Resilience", "data_path": "operational_recovery", "plot_func": plot_operational_recovery, "extra_args_paths": {"productivity_loss_data": "productivity_loss"}},
             {"title": "OEE & Components", "is_oee": True}], "insights_html": op_insights_html_main_final },
        {"name": "👥 Worker Well-being", "key_prefix": "ww", "plots": [
             {"is_subheader": True, "title": "Psychosocial & Well-being Indicators"},
             {"title": "Worker Well-Being Index", "data_path": "worker_wellbeing.scores", "plot_func": plot_worker_wellbeing, "extra_args_paths": {"triggers": "worker_wellbeing.triggers"}},
             {"title": "Psychological Safety Score", "data_path": "psychological_safety", "plot_func": plot_psychological_safety},
             {"title": "Team Cohesion Index", "data_path": "worker_wellbeing.team_cohesion_scores", "plot_func": plot_team_cohesion},
             {"title": "Perceived Workload Index (0-10)", "data_path": "worker_wellbeing.perceived_workload_scores", "plot_func": plot_perceived_workload, "extra_args_fixed": {"high_workload_threshold": DEFAULT_CONFIG['PERCEIVED_WORKLOAD_THRESHOLD_HIGH'], "very_high_workload_threshold": DEFAULT_CONFIG['PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH']}},
             {"is_subheader": True, "title": "Spatial Dynamics Analysis", "is_spatial": True}], "dynamic_insights_func": "render_wellbeing_alerts", "insights_html": ww_static_insights_html_main_final },
        {"name": "⏱️ Downtime Analysis", "key_prefix": "dt", "metrics_display": True, "plots": [
            {"title": "Downtime Trend (per Interval)", "data_path": "downtime_events_log", "plot_func": plot_downtime_trend, "is_event_based_aggregation": True, "extra_args_fixed": {"interval_threshold_minutes": DEFAULT_CONFIG['DOWNTIME_PLOT_ALERT_THRESHOLD']}},
            {"title": "Downtime Distribution by Cause", "data_path": "downtime_events_log", "plot_func": plot_downtime_causes_pie, "is_event_based_filtering": True}], "insights_html": dt_insights_html_main_final }
    ]

    for i_tab_main_final_loop, tab_def_main_final_loop in enumerate(tab_configs_main_final_app):
        with tabs_st_objs_final[i_tab_main_final_loop+1]:
            st.header(tab_def_main_final_loop["name"], divider=THEMED_DIVIDER_COLOR)
            if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
                sim_data_tab_final_loop = st.session_state.simulation_results
                sim_cfg_tab_final_active_loop = sim_data_tab_final_loop.get('config_params', {})
                st.markdown("##### Select Time Range for Plots:")
                start_time_ui_tab_loop, end_time_ui_tab_loop = time_range_input_section(tab_def_main_final_loop["key_prefix"], max_mins_ui_main_app_val, interval_duration_min_ui=active_mpi_main_app_val)
                start_idx_tab_final_loop = int(start_time_ui_tab_loop // active_mpi_main_app_val) if active_mpi_main_app_val > 0 else 0
                end_idx_tab_final_loop = int(end_time_ui_tab_loop // active_mpi_main_app_val) + 1 if active_mpi_main_app_val > 0 else 0
                disrupt_steps_for_plots_abs_tab_final_loop = [s for s in simulation_disruption_steps_absolute_main_val if start_idx_tab_final_loop <= s < end_idx_tab_final_loop]

                if tab_def_main_final_loop.get("metrics_display"):
                    downtime_log_tab_metrics_display = safe_get(sim_data_tab_final_loop, 'downtime_events_log', [])
                    downtime_events_in_range_tab_display = [evt for evt in downtime_log_tab_metrics_display if isinstance(evt, dict) and start_idx_tab_final_loop <= evt.get('step', -1) < end_idx_tab_final_loop]
                    downtime_durations_in_range_tab_display = [evt.get('duration',0.0) for evt in downtime_events_in_range_tab_display]
                    if downtime_events_in_range_tab_display: 
                        total_dt_period_disp = sum(downtime_durations_in_range_tab_display); num_incidents_disp = len([d for d in downtime_durations_in_range_tab_display if d > 0])
                        avg_dur_incident_disp = total_dt_period_disp / num_incidents_disp if num_incidents_disp > 0 else 0.0
                        dt_cols_disp = st.columns(3)
                        dt_cols_disp[0].metric("Total Downtime in Period", f"{total_dt_period_disp:.1f} min")
                        dt_cols_disp[1].metric("Number of Incidents", f"{num_incidents_disp}")
                        dt_cols_disp[2].metric("Avg. Duration / Incident", f"{avg_dur_incident_disp:.1f} min")

                plot_col_container_tab_final = st.container() 
                num_plots_in_row_tab_final = 0
                for plot_cfg_tab_item_final in tab_def_main_final_loop["plots"]:
                    if plot_cfg_tab_item_final.get("is_subheader"):
                        st.subheader(plot_cfg_tab_item_final["title"]) 
                        if plot_cfg_tab_item_final.get("is_spatial"):
                            facility_config_spatial_tab_final = {
                                'FACILITY_SIZE': sim_cfg_tab_final_active_loop.get('FACILITY_SIZE', DEFAULT_CONFIG['FACILITY_SIZE']),
                                'WORK_AREAS': sim_cfg_tab_final_active_loop.get('WORK_AREAS_EFFECTIVE', DEFAULT_CONFIG['WORK_AREAS']),
                                'ENTRY_EXIT_POINTS': sim_cfg_tab_final_active_loop.get('ENTRY_EXIT_POINTS', DEFAULT_CONFIG.get('ENTRY_EXIT_POINTS',[])),
                                'MINUTES_PER_INTERVAL': active_mpi_main_app_val }
                            with st.container(border=True):
                                team_pos_df_all_spatial = safe_get(sim_data_tab_final_loop, 'team_positions_df', pd.DataFrame())
                                zones_dist_spatial = ["All"] + list(facility_config_spatial_tab_final.get('WORK_AREAS', {}).keys())
                                zone_sel_key_spatial = f"{tab_def_main_final_loop['key_prefix']}_zone_sel_spatial_dist_final"
                                if zone_sel_key_spatial not in st.session_state: st.session_state[zone_sel_key_spatial] = "All"
                                zone_sel_dist_final = st.selectbox("Filter by Zone:", zones_dist_spatial, key=zone_sel_key_spatial)
                                filt_team_pos_df_spatial_time_final = pd.DataFrame()
                                if not team_pos_df_all_spatial.empty and 'step' in team_pos_df_all_spatial.columns: filt_team_pos_df_spatial_time_final = team_pos_df_all_spatial[(team_pos_df_all_spatial['step'] >= start_idx_tab_final_loop) & (team_pos_df_all_spatial['step'] < end_idx_tab_final_loop)]
                                filt_team_pos_df_spatial_loop_final = filt_team_pos_df_spatial_time_final
                                if zone_sel_dist_final != "All" and not filt_team_pos_df_spatial_loop_final.empty and 'zone' in filt_team_pos_df_spatial_loop_final.columns : filt_team_pos_df_spatial_loop_final = filt_team_pos_df_spatial_loop_final[filt_team_pos_df_spatial_loop_final['zone'] == zone_sel_dist_final]
                                show_ee_key_final = f'{tab_def_main_final_loop["key_prefix"]}_show_ee_spatial_cb_final'; 
                                if show_ee_key_final not in st.session_state: st.session_state[show_ee_key_final] = True
                                show_ee_exp_final = st.checkbox("Show E/E Points", key=show_ee_key_final) 
                                show_pl_key_final = f'{tab_def_main_final_loop["key_prefix"]}_show_pl_spatial_cb_final'; 
                                if show_pl_key_final not in st.session_state: st.session_state[show_pl_key_final] = True
                                show_pl_exp_final = st.checkbox("Show Area Outlines", key=show_pl_key_final)
                                spatial_plot_cols_final = st.columns(2)
                                with spatial_plot_cols_final[0]: 
                                    st.markdown("<h6>Worker Positions (Snapshot)</h6>", unsafe_allow_html=True)
                                    min_s_val_slider = int(start_idx_tab_final_loop)
                                    max_s_val_slider = max(min_s_val_slider, int(end_idx_tab_final_loop - 1))
                                    snap_slider_key_final_widget = f"{tab_def_main_final_loop['key_prefix']}_snap_step_slider_final"
                                    current_slider_val_state_widget = st.session_state.get(snap_slider_key_final_widget, min_s_val_slider)
                                    clamped_value_for_slider_widget = max(min_s_val_slider, min(current_slider_val_state_widget, max_s_val_slider))
                                    if min_s_val_slider == max_s_val_slider: clamped_value_for_slider_widget = min_s_val_slider
                                    st.session_state[snap_slider_key_final_widget] = clamped_value_for_slider_widget
                                    slider_is_disabled_widget = (min_s_val_slider >= max_s_val_slider)
                                    if max_mins_ui_main_app_val < active_mpi_main_app_val :
                                        st.caption("Not enough data for time step snapshot selector.")
                                        snap_step_val_final_widget_val = min_s_val_slider 
                                    else:
                                        snap_step_val_final_widget_val = st.slider("Time Step for Snapshot:", min_value=min_s_val_slider, max_value=max_s_val_slider, value=clamped_value_for_slider_widget, key=f"widget_actual_render_{snap_slider_key_final_widget}", step=1, disabled=slider_is_disabled_widget)
                                        if st.session_state[snap_slider_key_final_widget] != snap_step_val_final_widget_val : st.session_state[snap_slider_key_final_widget] = snap_step_val_final_widget_val
                                    if not team_pos_df_all_spatial.empty and max_s_val_slider >= min_s_val_slider:
                                        try: 
                                            fig_dist_final = plot_worker_distribution(team_pos_df_all_spatial, facility_config_spatial_tab_final.get('FACILITY_SIZE',(100,80)), facility_config_spatial_tab_final, use_3d_main_app_val, int(snap_step_val_final_widget_val), show_ee_exp_final, show_pl_exp_final, current_high_contrast_main_app_val)
                                            if fig_dist_final: st.plotly_chart(fig_dist_final, use_container_width=True, config=plot_cfg_interactive_final_ui)
                                            else: st.caption("Worker distribution plot error."); logger.warning("plot_worker_distribution returned None.")
                                        except Exception as e_dist_final: logger.error(f"Spatial Dist Plot Error: {e_dist_final}", exc_info=True); st.error(f"⚠️ Error plotting Worker Positions: {e_dist_final}.")
                                    else: st.caption("No data for positions snapshot or invalid time range.")
                                with spatial_plot_cols_final[1]: 
                                    st.markdown("<h6>Worker Density Heatmap</h6>", unsafe_allow_html=True)
                                    if not filt_team_pos_df_spatial_loop_final.empty:
                                        try: 
                                            fig_heat_final = plot_worker_density_heatmap(filt_team_pos_df_spatial_loop_final, facility_config_spatial_tab_final.get('FACILITY_SIZE',(100,80)), facility_config_spatial_tab_final, show_ee_exp_final, show_pl_exp_final, current_high_contrast_main_app_val)
                                            if fig_heat_final: st.plotly_chart(fig_heat_final, use_container_width=True, config=plot_cfg_interactive_final_ui)
                                            else: st.caption("Density heatmap error."); logger.warning("plot_worker_density_heatmap returned None.")
                                        except Exception as e_heat_final: logger.error(f"Spatial Heatmap Plot Error: {e_heat_final}", exc_info=True); st.error(f"⚠️ Error plotting Density Heatmap: {e_heat_final}.")
                                    else: st.caption("No data for density heatmap in this time range/zone.")
                        num_plots_in_row_tab_final = 0; continue
                    
                    if num_plots_in_row_tab_final == 0: plot_columns_tab_final = plot_col_container_tab_final.columns(2)
                    current_plot_col_tab_final = plot_columns_tab_final[num_plots_in_row_tab_final % 2]
                    with current_plot_col_tab_final:
                        st.markdown(f"<h6>{plot_cfg_tab_item_final['title']}</h6>", unsafe_allow_html=True)
                        with st.container(border=True):
                            plot_data_to_render_final = None; plot_kwargs_final = {"high_contrast": current_high_contrast_main_app_val}
                            try:
                                if plot_cfg_tab_item_final.get("is_oee"):
                                    eff_df_full_oee = safe_get(sim_data_tab_final_loop, 'efficiency_metrics_df', pd.DataFrame())
                                    if not eff_df_full_oee.empty:
                                        oee_ms_key_final = f"{tab_def_main_final_loop['key_prefix']}_oee_metrics_ms_final"
                                        if oee_ms_key_final not in st.session_state: st.session_state[oee_ms_key_final] = ['uptime', 'throughput', 'quality', 'oee']
                                        sel_metrics_oee = st.multiselect("Select OEE Metrics:", ['uptime', 'throughput', 'quality', 'oee'],default=st.session_state[oee_ms_key_final], key=oee_ms_key_final)
                                        filt_eff_df_oee = _slice_dataframe_by_step_indices(eff_df_full_oee, start_idx_tab_final_loop, end_idx_tab_final_loop)
                                        if "disruption_points" in plot_operational_efficiency.__code__.co_varnames: plot_kwargs_final["disruption_points"] = [s - start_idx_tab_final_loop for s in disrupt_steps_for_plots_abs_tab_final_loop if s - start_idx_tab_final_loop >=0]
                                        if not filt_eff_df_oee.empty:
                                            fig_oee_final = plot_operational_efficiency(filt_eff_df_oee, sel_metrics_oee, **plot_kwargs_final)
                                            if fig_oee_final: st.plotly_chart(fig_oee_final, use_container_width=True, config=plot_cfg_interactive_final_ui)
                                            else: st.caption("OEE plot error."); logger.warning("plot_operational_efficiency returned None.")
                                        else: st.caption("No OEE data for this time range.")
                                    else: st.caption("No OEE data available.")
                                else: 
                                    raw_plot_data_tab_final = safe_get(sim_data_tab_final_loop, plot_cfg_tab_item_final["data_path"], [])
                                    if "extra_args_paths" in plot_cfg_tab_item_final:
                                        for arg_n_final, arg_p_final in plot_cfg_tab_item_final["extra_args_paths"].items():
                                            extra_d_final = safe_get(sim_data_tab_final_loop, arg_p_final, [])
                                            if plot_cfg_tab_item_final["plot_func"] == plot_worker_wellbeing and arg_n_final == "triggers":
                                                filt_trigs_final = {}; 
                                                if isinstance(extra_d_final, dict):
                                                    for tr_type, tr_steps_abs in extra_d_final.items():
                                                        if tr_type == 'work_area' and isinstance(tr_steps_abs, dict):
                                                            filt_trigs_final[tr_type] = {zn: [s - start_idx_tab_final_loop for s in (s_list if isinstance(s_list,list) else []) if start_idx_tab_final_loop <= s < end_idx_tab_final_loop and s - start_idx_tab_final_loop >=0] for zn,s_list in tr_steps_abs.items()}
                                                            filt_trigs_final[tr_type] = {k_trig:v_trig for k_trig,v_trig in filt_trigs_final[tr_type].items() if v_trig}
                                                        elif isinstance(tr_steps_abs, list): filt_trigs_final[tr_type] = [s - start_idx_tab_final_loop for s in tr_steps_abs if start_idx_tab_final_loop <= s < end_idx_tab_final_loop and s - start_idx_tab_final_loop >=0]
                                                plot_kwargs_final[arg_n_final] = filt_trigs_final
                                            elif isinstance(extra_d_final, list): plot_kwargs_final[arg_n_final] = extra_d_final[start_idx_tab_final_loop:min(end_idx_tab_final_loop, len(extra_d_final))] if start_idx_tab_final_loop < len(extra_d_final) else []
                                            else: plot_kwargs_final[arg_n_final] = extra_d_final
                                    if "extra_args_fixed" in plot_cfg_tab_item_final: plot_kwargs_final.update(plot_cfg_tab_item_final["extra_args_fixed"])
                                    if "disruption_points" in plot_cfg_tab_item_final["plot_func"].__code__.co_varnames: plot_kwargs_final["disruption_points"] = [s - start_idx_tab_final_loop for s in disrupt_steps_for_plots_abs_tab_final_loop if s - start_idx_tab_final_loop >=0]

                                    if plot_cfg_tab_item_final.get("is_event_based_aggregation"):
                                        num_steps_in_range_agg_final = end_idx_tab_final_loop - start_idx_tab_final_loop
                                        aggregated_data_agg_final = [0.0] * num_steps_in_range_agg_final if num_steps_in_range_agg_final > 0 else []
                                        for evt_agg_final in raw_plot_data_tab_final:
                                            if isinstance(evt_agg_final,dict) and start_idx_tab_final_loop <= evt_agg_final.get('step',-1) < end_idx_tab_final_loop:
                                                rel_step_agg_final = evt_agg_final['step'] - start_idx_tab_final_loop
                                                if 0 <= rel_step_agg_final < num_steps_in_range_agg_final: aggregated_data_agg_final[rel_step_agg_final] += evt_agg_final.get('duration',0)
                                        plot_data_to_render_final = aggregated_data_agg_final
                                    elif plot_cfg_tab_item_final.get("is_event_based_filtering"):
                                        plot_data_to_render_final = [evt_filt_final for evt_filt_final in raw_plot_data_tab_final if isinstance(evt_filt_final,dict) and start_idx_tab_final_loop <= evt_filt_final.get('step',-1) < end_idx_tab_final_loop]
                                        if "disruption_points" in plot_kwargs_final: del plot_kwargs_final["disruption_points"]
                                    elif isinstance(raw_plot_data_tab_final, list): plot_data_to_render_final = raw_plot_data_tab_final[start_idx_tab_final_loop:min(end_idx_tab_final_loop, len(raw_plot_data_tab_final))] if start_idx_tab_final_loop < len(raw_plot_data_tab_final) else []
                                    elif isinstance(raw_plot_data_tab_final, pd.DataFrame): plot_data_to_render_final = _slice_dataframe_by_step_indices(raw_plot_data_tab_final, start_idx_tab_final_loop, end_idx_tab_final_loop)
                                    
                                    data_exists_for_plot_final = False
                                    if isinstance(plot_data_to_render_final, (list, pd.Series)) and len(plot_data_to_render_final) > 0: data_exists_for_plot_final = True
                                    elif isinstance(plot_data_to_render_final, pd.DataFrame) and not plot_data_to_render_final.empty: data_exists_for_plot_final = True
                                    elif plot_cfg_tab_item_final.get("is_event_based_filtering") and isinstance(plot_data_to_render_final, list): data_exists_for_plot_final = True 
                                    elif plot_cfg_tab_item_final.get("is_event_based_aggregation") and isinstance(plot_data_to_render_final, list) and (any(x > 1e-6 for x in plot_data_to_render_final) or (end_idx_tab_final_loop - start_idx_tab_final_loop > 0)): data_exists_for_plot_final = True

                                    if data_exists_for_plot_final:
                                        fig_obj_main_final = plot_cfg_tab_item_final["plot_func"](plot_data_to_render_final, **plot_kwargs_final)
                                        if fig_obj_main_final: st.plotly_chart(fig_obj_main_final, use_container_width=True, config=plot_cfg_interactive_final_ui)
                                        else: st.caption(f"Plot for '{plot_cfg_tab_item_final['title']}' could not be generated (returned None)."); logger.warning(f"Plot func for '{plot_cfg_tab_item_final['title']}' returned None.")
                                    else: st.caption(f"No data for '{plot_cfg_tab_item_final['title']}' in selected range.")
                            except Exception as e_plot_render_final: logger.error(f"Error rendering plot '{plot_cfg_tab_item_final['title']}': {e_plot_render_final}", exc_info=True); st.error(f"⚠️ Error for plot '{plot_cfg_tab_item_final['title']}': {e_plot_render_final}")
                    num_plots_in_row_tab_final += 1
                
                st.markdown("<hr style='margin-top:2rem;'><h3 style='text-align:center; margin-top:1rem;'>🏛️ Leadership Actionable Insights</h3>", unsafe_allow_html=True)
                if tab_def_main_final_loop.get("dynamic_insights_func") == "render_wellbeing_alerts":
                    with st.container(border=True):
                        st.markdown("<h6>Well-Being Alerts (within selected time range):</h6>", unsafe_allow_html=True); insights_count_wb_final = 0 
                        ww_trigs_disp_raw_final = safe_get(sim_data_tab_final_loop, 'worker_wellbeing.triggers', {})
                        for alert_type_final, alert_steps_raw_final in ww_trigs_disp_raw_final.items():
                            if alert_type_final == 'work_area' and isinstance(alert_steps_raw_final, dict):
                                wa_alert_found_final = False; wa_details_html_final = ""
                                for zone_final, zone_steps_raw_list_final in alert_steps_raw_final.items():
                                    zone_steps_in_range_final = [s for s in (zone_steps_raw_list_final if isinstance(zone_steps_raw_list_final, list) else []) if start_idx_tab_final_loop <= s < end_idx_tab_final_loop]
                                    if zone_steps_in_range_final: wa_alert_found_final = True; wa_details_html_final += f"  - {zone_final}: {len(zone_steps_in_range_final)} alerts at steps {zone_steps_in_range_final}<br>"
                                if wa_alert_found_final: st.markdown(f"<div class='alert-warning insight-text'><strong>Work Area Specific Alerts:</strong><br>{wa_details_html_final}</div>", unsafe_allow_html=True); insights_count_wb_final +=1
                            elif isinstance(alert_steps_raw_final, list):
                                alert_steps_in_range_final = [s for s in alert_steps_raw_final if start_idx_tab_final_loop <= s < end_idx_tab_final_loop]
                                if alert_steps_in_range_final:
                                    alert_class_final = "alert-critical" if alert_type_final == "threshold" else "alert-warning" if alert_type_final == "trend" else "alert-info"
                                    alert_title_text_final = alert_type_final.replace("_", " ").title()
                                    st.markdown(f"<div class='{alert_class_final} insight-text'><strong>{alert_title_text_final} Alerts ({len(alert_steps_in_range_final)}x):</strong> Steps {alert_steps_in_range_final}.</div>", unsafe_allow_html=True); insights_count_wb_final += 1
                        if insights_count_wb_final == 0: st.markdown(f"<p class='insight-text' style='color: {COLOR_POSITIVE_GREEN_BORDER};'>✅ No specific well-being alerts triggered in selected period.</p>", unsafe_allow_html=True)
                if tab_def_main_final_loop.get("insights_html"): st.markdown(tab_def_main_final_loop["insights_html"], unsafe_allow_html=True) 
            else: st.info(f"ℹ️ Run simulation or load data to view {tab_def_main_final_loop['name']}.", icon="📊")

    with tabs_st_objs_final[4]:
        st.header("📖 Glossary of Terms", divider=THEMED_DIVIDER_COLOR)
        st.markdown("""
            <div style="font-size: 0.95rem; line-height: 1.7;">
            <p>This glossary defines key metrics used throughout the dashboard to help you understand the operational insights provided. For a combined view with general help, click the "ℹ️ Help & Glossary" button in the sidebar.</p>
            <details><summary><strong>Task Compliance Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of tasks completed correctly and within the allocated time. It measures adherence to operational protocols and standards. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Collaboration Metric</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A simulated score indicating teamwork potential and interaction levels based on factors like team cohesion, workload, and disruptions. It is not a direct measure of physical proximity in the current simulation. <em>Range: 0-100%. Higher is generally better, reflecting positive interaction dynamics.</em></p></details>
            <details><summary><strong>Operational Recovery Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A measure of the system's ability to return to and maintain target output levels (based on OEE) after experiencing disruptions. It reflects operational resilience. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Worker Well-Being Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A composite score derived from simulated factors such as fatigue, stress levels (related to workload and control), and job satisfaction (related to leadership, safety, cohesion). It provides an indicator of overall worker health and morale. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Psychological Safety Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">An estimate of the perceived comfort level among workers to report issues, voice concerns, or suggest improvements without fear of negative consequences. Influenced by leadership, communication, disruptions, and team cohesion. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Team Cohesion Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A measure of the strength of bonds and sense of belonging within a team. Impacted by disruptions, workload, psychological safety, and collaboration levels. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Perceived Workload Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">An indicator of how demanding workers perceive their tasks and overall workload on a scale (0-10). Influenced by task backlog and time pressure. Persistently high scores can lead to stress and burnout. <em>Lower is generally better (closer to target).</em></p></details>
            <details><summary><strong>Uptime</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of scheduled operational time that equipment or a system is available and functioning correctly. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Throughput</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The rate at which a system processes work or produces output, expressed as a percentage of its theoretical maximum potential. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Quality Rate</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of products or outputs that meet predefined quality standards, free of defects. Influenced by task compliance. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>OEE (Overall Equipment Effectiveness)</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A comprehensive metric calculated as (Uptime × Throughput × Quality Rate). It provides a holistic view of operational performance and efficiency. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Productivity Loss</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of potential output or operational time lost due to inefficiencies, disruptions, downtime, or substandard performance. Calculated as 100 - Operational Recovery Score. <em>Range: 0-100%. Lower is better.</em></p></details>
            <details><summary><strong>Downtime Events Log / Downtime per Interval</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The simulation outputs a log of individual downtime events (`downtime_events_log`), each with a start step, duration, and cause. For trend plots and summary tables, these are aggregated to show total downtime duration per simulation interval. <em>Lower is better.</em></p></details>
            <details><summary><strong>Task Completion Rate</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of assigned tasks that are successfully completed within a given time interval. In this simulation, it's currently an alias for Throughput. <em>Range: 0-100%. Higher is better.</em></p></details>
            <hr>
            <p><strong>Simulation Step / Interval:</strong> The simulation progresses in discrete time steps. The duration of each interval (e.g., 2 minutes) is defined by `MINUTES_PER_INTERVAL` in the configuration. Many metrics are reported per interval.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    main()

