# main.py
import logging
import streamlit as st
import pandas as pd
import numpy as np
import math # Ensure math is imported here if used in utility functions before main()
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
    logging.basicConfig(level=logging.INFO, # Changed to INFO for production, DEBUG for dev
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
                        filename='dashboard.log',
                        filemode='a')
logger.info("Main.py: Parsed imports and logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.2.1\nInsights for operational excellence & psychosocial well-being."})

# --- Accessible Color Definitions ---
COLOR_CRITICAL_RED = "#E53E3E"
COLOR_WARNING_AMBER = "#F59E0B"
COLOR_POSITIVE_GREEN = "#10B981"
COLOR_INFO_BLUE = "#3B82F6"
COLOR_ACCENT_INDIGO = "#4F46E5"

# --- UTILITY FUNCTIONS (DEFINED GLOBALLY BEFORE MAIN) ---
def safe_get(data_dict, path_str, default_val=None):
    current = data_dict
    is_list_like_path = False
    # Ensure path_str is a string before using string methods
    if isinstance(path_str, str):
        is_list_like_path = path_str.endswith(('.data', '.scores', '.triggers', 'minutes', 'events_list'))
    
    if default_val is None: 
        default_return = [] if is_list_like_path else None
    else: 
        default_return = default_val

    if not isinstance(path_str, str):
        logger.warning(f"safe_get: path_str is not a string: {path_str}. Returning default '{default_return}'.", extra={'user_action': 'Safe Get Internal Warning'})
        return default_return

    if not isinstance(data_dict, dict):
        # Log only if path_str is not empty, to avoid noise for initial checks on non-dict sim_data
        if path_str: 
            logger.warning(f"safe_get: data_dict is not a dictionary for path '{path_str}'. Type: {type(data_dict)}. Returning default '{default_return}'.", extra={'user_action': 'Safe Get Internal Warning'})
        return default_return

    try:
        keys = path_str.split('.')
        for i, key in enumerate(keys):
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, pd.Series)) and key.isdigit():
                idx = int(key)
                current = current[idx] if idx < len(current) else None
            else:
                current = None # Path broken
                break
        
        if current is None:
            # Re-check is_list_like_path based on the actual keys if path_str was complex and default_val was None
            # This helps ensure [] is returned if the final key indicates a list structure.
            is_list_like_final_key = keys and keys[-1] in ['data', 'scores', 'triggers', 'minutes', 'events_list']
            return [] if default_val is None and is_list_like_final_key else default_val
        return current
    except (ValueError, IndexError, TypeError) as e:
        logger.debug(f"safe_get failed for path '{path_str}': {e}. Returning default '{default_return}'.", extra={'user_action': 'Safe Get Internal Error'})
        return default_return


def safe_stat(data_list, stat_func, default_val=0.0):
    log_data_list_repr = str(data_list)
    if len(log_data_list_repr) > 150: 
        log_data_list_repr = log_data_list_repr[:147] + "..."
    # logger.debug(f"safe_stat: Input data (preview): {log_data_list_repr}, func: {stat_func.__name__}, default: {default_val}", extra={'user_action': 'Safe Stat Call'})

    if not isinstance(data_list, (list, np.ndarray, pd.Series)):
        # logger.debug(f"safe_stat: data_list is not a list/array/series, type: {type(data_list)}. Returning default_val: {default_val}", extra={'user_action': 'Safe Stat Type Check Fail'})
        return default_val
    
    # More robust conversion to numeric, handling potential strings that are numbers, None, and existing NaNs
    if isinstance(data_list, pd.Series):
        valid_data_series = pd.to_numeric(data_list, errors='coerce').dropna()
        valid_data = valid_data_series.tolist()
    else: # list or np.ndarray
        valid_data = [float(x) for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x)) and isinstance(x, (int, float, str)) and str(x).strip()]
        try:
            valid_data = [float(x) for x in valid_data] # Final conversion after basic filtering
        except ValueError: # If string cannot be converted after all
            valid_data = [x for x in valid_data if isinstance(x, (int, float))]


    # logger.debug(f"safe_stat: Valid data (count: {len(valid_data)}, preview after float conversion): {str(valid_data)[:100]}", extra={'user_action': 'Safe Stat Valid Data'})

    if not valid_data:
        # logger.debug(f"safe_stat: No valid numeric data after filtering. Returning default_val: {default_val}", extra={'user_action': 'Safe Stat No Valid Data'})
        return default_val
    
    try:
        result = stat_func(np.array(valid_data)) 
        if isinstance(result, (float, np.floating)) and np.isnan(result): 
            # logger.debug(f"safe_stat: stat_func returned NaN. Returning default_val: {default_val}", extra={'user_action': 'Safe Stat NaN Result'})
            return default_val
        # logger.debug(f"safe_stat: stat_func returned: {result}. Type: {type(result)}", extra={'user_action': 'Safe Stat Success'})
        return result
    except Exception as e: 
        logger.warning(f"safe_stat: Error in stat_func {stat_func.__name__} on data (preview): {str(valid_data)[:50]}: {e}. Returning default_val: {default_val}", exc_info=True, extra={'user_action': 'Safe Stat Error'})
        return default_val

def get_actionable_insights(sim_data, current_config):
    insights = []
    if not sim_data or not isinstance(sim_data, dict): 
        logger.warning("get_actionable_insights: sim_data is None or not a dict.", extra={'user_action': 'Actionable Insights - Invalid Input'})
        return insights
    
    # logger.debug(f"get_actionable_insights: Generating insights. Sim_data available.", extra={'user_action': 'Actionable Insights - Start'})

    compliance_data = safe_get(sim_data, 'task_compliance.data', [])
    target_compliance = float(current_config.get('TARGET_COMPLIANCE', 75.0))
    compliance_avg = safe_stat(compliance_data, np.mean, default_val=0.0) # Default to 0 if no data
    if compliance_avg < target_compliance * 0.9:
        insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Avg. Task Compliance ({compliance_avg:.1f}%) significantly below target ({target_compliance:.0f}%). Review disruption impacts, task complexities, and training."})
    elif compliance_avg < target_compliance:
        insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Avg. Task Compliance at {compliance_avg:.1f}%. Identify intervals or areas with lowest compliance for process review."})

    wellbeing_scores = safe_get(sim_data, 'worker_wellbeing.scores', [])
    target_wellbeing = float(current_config.get('TARGET_WELLBEING', 70.0))
    wellbeing_avg = safe_stat(wellbeing_scores, np.mean, default_val=0.0)
    wellbeing_critical_factor = float(current_config.get('WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET', 0.85))
    if wellbeing_avg < target_wellbeing * wellbeing_critical_factor:
        insights.append({"type": "critical", "title": "Critical Worker Well-being", "text": f"Avg. Well-being ({wellbeing_avg:.1f}%) critically low (target {target_wellbeing:.0f}%). Urgent review of work conditions, load, and stress factors needed."})
    
    threshold_triggers = safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])
    if wellbeing_scores and len(threshold_triggers) > max(2, len(wellbeing_scores) * 0.1): # Check if wellbeing_scores is not empty
        insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers)} instances of well-being dropping below threshold. Investigate specific triggers."})

    # downtime_minutes is expected to be a list of event dictionaries
    downtime_events_list = safe_get(sim_data, 'downtime_minutes', []) 
    downtime_durations = [event.get('duration', 0.0) for event in downtime_events_list if isinstance(event, dict)]
    total_downtime = safe_stat(downtime_durations, np.sum, default_val=0.0)
    
    sim_cfg_params = sim_data.get('config_params', {})
    shift_mins = float(sim_cfg_params.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
    dt_thresh_percent = float(current_config.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT', 0.05))
    dt_thresh_total_shift = shift_mins * dt_thresh_percent
    if total_downtime > dt_thresh_total_shift:
        insights.append({"type": "critical", "title": "Excessive Total Shift Downtime", "text": f"Total shift downtime is {total_downtime:.0f} minutes, exceeding the guideline of {dt_thresh_total_shift:.0f} min ({dt_thresh_percent*100:.0f}% of shift). Deep dive into disruption causes and recovery protocols."})
    
    psych_safety_scores = safe_get(sim_data, 'psychological_safety', [])
    target_psych_safety = float(current_config.get('TARGET_PSYCH_SAFETY', 70.0))
    psych_safety_avg = safe_stat(psych_safety_scores, np.mean, default_val=0.0)
    if psych_safety_avg < target_psych_safety * 0.9:
        insights.append({"type": "warning", "title": "Low Psychological Safety", "text": f"Avg. Psych. Safety ({psych_safety_avg:.1f}%) is below target ({target_psych_safety:.0f}%). Consider initiatives to build trust and open communication."})

    cohesion_scores = safe_get(sim_data, 'worker_wellbeing.team_cohesion_scores', [])
    target_cohesion = float(current_config.get('TARGET_TEAM_COHESION', 70.0))
    cohesion_avg = safe_stat(cohesion_scores, np.mean, default_val=0.0)
    if cohesion_avg < target_cohesion * 0.9:
        insights.append({"type": "warning", "title": "Low Team Cohesion", "text": f"Avg. Team Cohesion ({cohesion_avg:.1f}%) is below desired levels. Consider team-building or structural reviews."})
    
    workload_scores = safe_get(sim_data, 'worker_wellbeing.perceived_workload_scores', [])
    target_workload = float(current_config.get('TARGET_PERCEIVED_WORKLOAD', 6.5)) # Scale 0-10
    workload_avg = safe_stat(workload_scores, np.mean, default_val=target_workload / 2) # Default to a mid-low value
    workload_very_high_thresh = float(current_config.get('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5))
    workload_high_thresh = float(current_config.get('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5))
    if workload_avg > workload_very_high_thresh:
        insights.append({"type": "critical", "title": "Very High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) is critically high. Immediate review of task distribution, staffing, and efficiencies required."})
    elif workload_avg > workload_high_thresh:
        insights.append({"type": "warning", "title": "High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) exceeds high threshold. Monitor closely and identify bottlenecks."})
    elif workload_avg > target_workload:
        insights.append({"type": "info", "title": "Elevated Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) is above target ({target_workload:.1f}/10). Consider proactive adjustments."})
    
    team_pos_df = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
    if not team_pos_df.empty and 'zone' in team_pos_df.columns and 'worker_id' in team_pos_df.columns:
        work_areas_config_insight = current_config.get('WORK_AREAS', {})
        for zone_name, zone_details in work_areas_config_insight.items():
            if not isinstance(zone_details, dict): continue
            workers_in_zone_series = team_pos_df[team_pos_df['zone'] == zone_name].groupby('step')['worker_id'].nunique()
            if not workers_in_zone_series.empty:
                workers_in_zone_avg = workers_in_zone_series.mean()
                intended_workers = zone_details.get('workers', 0)
                coords = zone_details.get('coords'); area_m2 = 1.0 # Default to 1 to avoid division by zero
                if coords and isinstance(coords, list) and len(coords) == 2 and \
                   all(isinstance(p, tuple) and len(p)==2 for p in coords):
                    (x0,y0), (x1,y1) = coords; area_m2 = abs(x1-x0) * abs(y1-y0)
                if area_m2 == 0: area_m2 = 1.0 # Avoid division by zero if area is malformed
                
                avg_density = workers_in_zone_avg / area_m2 if area_m2 > 0 else 0 # Ensure area_m2 > 0
                
                # Define density based on intended workers for comparison
                intended_density = (intended_workers / area_m2) if area_m2 > 0 and intended_workers > 0 else 0

                if intended_density > 0 and avg_density > intended_density * 1.8: 
                     insights.append({"type": "warning", "title": f"Potential Overcrowding in '{zone_name}'", "text": f"Average worker density ({avg_density:.2f} w/m²) significantly higher than based on assigned workers ({intended_density:.2f} w/m²). Review layout or worker paths."})
                elif intended_workers > 0 and workers_in_zone_avg < intended_workers * 0.4: 
                     insights.append({"type": "info", "title": f"Potential Underutilization of '{zone_name}'", "text": f"Average workers observed ({workers_in_zone_avg:.1f}) is less than 40% of assigned ({intended_workers}). Check task allocation or if workers are congregating elsewhere."})

    # Holistic check should consider if data was available for averages
    if compliance_avg > target_compliance * 1.05 and wellbeing_scores and \
       wellbeing_avg > target_wellbeing * 1.05 and psych_safety_scores and \
       total_downtime < dt_thresh_total_shift * 0.5 and \
       psych_safety_avg > target_psych_safety * 1.05:
        insights.append({"type": "positive", "title": "Holistically Excellent Performance", "text": "Key operational and psychosocial metrics significantly exceed targets. A well-balanced and high-performing shift! Leadership should identify and replicate success factors."})
    
    initiative = sim_cfg_params.get('TEAM_INITIATIVE', 'Standard Operations') 
    if initiative != "Standard Operations":
        insights.append({"type": "info", "title": f"Initiative Active: '{initiative}'", "text": f"The '{initiative}' initiative was simulated. Its impact can be assessed by comparing metrics to a 'Standard Operations' baseline run."})
    
    logger.info(f"get_actionable_insights: Generated {len(insights)} insights.", extra={'user_action': 'Actionable Insights - End'})
    return insights

# --- Helper function for PDF/CSV data aggregation ---
def aggregate_downtime_by_step(downtime_events_list, num_total_steps):
    """
    Aggregates downtime durations per simulation step.
    Assumes downtime_events_list is a list of dictionaries,
    each with 'step' (int) and 'duration' (float/int).
    """
    downtime_per_step_agg = [0.0] * num_total_steps
    if not isinstance(downtime_events_list, list):
        logger.warning("aggregate_downtime_by_step: downtime_events_list is not a list.")
        return downtime_per_step_agg

    for event in downtime_events_list:
        if not isinstance(event, dict):
            # logger.debug(f"aggregate_downtime_by_step: Skipping non-dict event: {event}")
            continue
        
        step = event.get('step') 
        duration = event.get('duration', 0.0)

        if not isinstance(step, int):
            # Fallback if 'step' is not int, try 'Start Time (min)' if this is from sidebar event structure
            # This part might be too specific if 'downtime_minutes' from sim always has 'step'
            start_time_min_val = event.get('Start Time (min)') 
            if isinstance(start_time_min_val, (int, float)):
                step = int(start_time_min_val // 2) # Assuming 2 min per step
            else:
                # logger.debug(f"aggregate_downtime_by_step: Skipping event with invalid step/start_time: {event}")
                continue
        
        if 0 <= step < num_total_steps and isinstance(duration, (int, float)) and duration > 0:
            downtime_per_step_agg[step] += float(duration)
        # else:
            # logger.debug(f"aggregate_downtime_by_step: Event step {step} out of range [0, {num_total_steps-1}] or invalid duration for event: {event}")
    return downtime_per_step_agg

def _prepare_timeseries_for_export(raw_data, num_total_steps, default_val=np.nan):
    """Helper to ensure a list is of length num_total_steps, padding or truncating."""
    if not isinstance(raw_data, list):
        raw_data = [] # Default to empty list if not a list
    
    # Pad with default_val if shorter, truncate if longer
    prepared_data = (raw_data + [default_val] * num_total_steps)[:num_total_steps]
    return prepared_data

st.markdown(f"""
    <style>
        /* Base Styles */
        .main {{ background-color: #121828; color: #EAEAEA; font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif; padding: 2rem; }}
        h1 {{ font-size: 2.4rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; text-align: center; margin-bottom: 2rem; color: #FFFFFF; }}
        
        /* Main Content Headers (Tabs) - Targets h2 generated by st.header in tabs */
        div[data-testid="stTabs"] section[role="tabpanel"] > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div > h2 {{ 
            font-size: 1.75rem !important; 
            font-weight: 600 !important; 
            line-height: 1.3 !important; 
            margin: 1.2rem 0 1rem 0 !important; 
            color: #D1D5DB !important; 
            border-bottom: 2px solid {COLOR_ACCENT_INDIGO} !important; 
            padding-bottom: 0.6rem !important;
            text-align: left !important;
        }}

        /* Main Content Section Subheaders (e.g., "Additional Operational Metrics") - Targets h3 generated by st.subheader */
         div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] .stSubheader {{ 
            font-size: 1.3rem !important; 
            font-weight: 500 !important; 
            line-height: 1.4 !important; 
            margin-top: 1.8rem !important; 
            margin-bottom: 0.8rem !important; 
            color: #C0C0C0 !important;
            border-bottom: 1px solid #4A5568 !important; 
            padding-bottom: 0.3rem !important;
            text-align: left !important;
        }}
        /* Main Content Markdown H5 (e.g. for "Select Time Range for Plots:") */
        div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h5 {{
            font-size: 1.0rem !important; 
            font-weight: 600 !important; 
            line-height: 1.3 !important;
            margin: 1.5rem 0 0.5rem 0 !important; 
            color: #C8C8C8 !important; 
            text-align: left !important;
        }}
        div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] h6 {{
            font-size: 0.95rem !important; 
            font-weight: 500 !important; 
            line-height: 1.3 !important;
            margin-top: 1rem !important; margin-bottom: 0.5rem !important; 
            color: #B0B0B0 !important; text-align: left;
        }}

        /* Sidebar Specific Headers */
        [data-testid="stSidebar"] h2 {{ 
            font-size: 1.4rem !important; color: #EAEAEA !important;
            margin-top: 1.5rem !important; margin-bottom: 0.5rem !important;
            padding-bottom: 0.3rem !important; border-bottom: 1px solid #4A5568 !important;
        }}
        [data-testid="stSidebar"] h3 {{ 
            font-size: 1.1rem !important; text-align: center !important; 
            margin-bottom: 1.2rem !important; color: #A0A0A0 !important; 
            border-bottom: none !important; 
        }}
        [data-testid="stSidebar"] div[data-testid="stExpander"] h5 {{
            color: #E0E0E0 !important; text-align: left; font-size: 1.0rem !important; 
            font-weight: 600 !important; margin-top: 0.8rem !important; margin-bottom: 0.4rem !important; 
        }}
        [data-testid="stSidebar"] div[data-testid="stExpander"] h6 {{
            color: #D1D5DB !important; text-align: left; font-size: 0.9rem !important;
            font-weight: 600 !important; margin-top: 1rem !important; margin-bottom: 0.3rem !important;
        }}
        [data-testid="stSidebar"] .stMarkdownContainer > p, 
        [data-testid="stSidebar"] .stCaption {{ 
             color: #B0B0B0 !important; font-size: 0.85rem !important;
             line-height: 1.3 !important; margin-top: 0.2rem !important; margin-bottom: 0.5rem !important;
        }}
        /* Specifically for the "Add New Event:" prompt using Markdown P tag */
        [data-testid="stSidebar"] div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] > div > div > div > p {{ /* May need to adjust selector if this was for st.markdown("Add New Event:") directly */
            color: #E0E0E0 !important; font-weight: 600 !important;
            font-size:0.92rem !important; margin-bottom:2px !important;
            padding-bottom: 3px !important;
        }}

        .stButton>button {{ background-color: {COLOR_ACCENT_INDIGO}; color: #FFFFFF; border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.95rem; font-weight: 500; transition: all 0.2s ease-in-out; border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stButton>button:hover, .stButton>button:focus {{ background-color: #6366F1; transform: translateY(-1px); box-shadow: 0 3px 7px rgba(0,0,0,0.2); outline: none; }}
        .stButton>button:disabled {{ background-color: #374151; color: #9CA3AF; cursor: not-allowed; box-shadow: none; }}
        
        /* Sidebar Widget Label Styling */
        [data-testid="stSidebar"] div[data-testid*="stWidgetLabel"] label p, /* For st.text_input, st.number_input etc. label */
        [data-testid="stSidebar"] label[data-baseweb="checkbox"] span, /* For st.checkbox label */
        [data-testid="stSidebar"] .stSelectbox > label, /* For st.selectbox label */
        [data-testid="stSidebar"] .stMultiSelect > label {{ /* For st.multiselect label */
            color: #E0E0E0 !important; 
            font-weight: 600 !important;
            font-size: 0.92rem !important; 
            padding-bottom: 3px !important; 
            display: block !important; 
        }}

        /* Sidebar Widget INPUT FIELDS */
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"], 
        [data-testid="stSidebar"] .stNumberInput div input, 
        [data-testid="stSidebar"] .stTextInput div input,
        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] {{ 
            background-color: #2D3748 !important; color: #EAEAEA !important; 
            border-radius: 6px !important; padding: 0.4rem 0.5rem !important; 
            margin-bottom: 0.6rem !important; font-size: 0.9rem !important; 
            border: 1px solid #4A5568 !important; height: auto !important; 
        }}
        [data-testid="stSidebar"] .stNumberInput button {{ /* +/- buttons for number input */
            background-color: #374151 !important; color: #EAEAEA !important; border: 1px solid #4A5568 !important;
        }}
        [data-testid="stSidebar"] .stNumberInput button:hover {{ background-color: #4A5568 !important; }}

        [data-testid="stSidebar"] {{ background-color: #1F2937; color: #EAEAEA; padding: 1.5rem; border-right: 1px solid #374151; font-size: 0.95rem; }}
        [data-testid="stSidebar"] .stButton>button {{ background-color: {COLOR_POSITIVE_GREEN}; width: 100%; margin-bottom: 0.5rem; }}
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus {{ background-color: #6EE7B7; }}
        /* Make specific sidebar buttons different if needed, e.g., Run Simulation */
        [data-testid="stSidebar"] .stButton button[kind="primary"] {{ /* For buttons with type="primary" in sidebar */
             background-color: {COLOR_ACCENT_INDIGO} !important;
        }}
        [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {{
             background-color: #6366F1 !important;
        }}


        .stMetric {{ background-color: #1F2937; border-radius: 8px; padding: 1rem 1.25rem; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #374151; display: flex; flex-direction: column; align-items: flex-start;}}
        .stMetric > div[data-testid="stMetricLabel"] {{ 
            font-size: 1.0rem !important; color: #B0B0B0 !important; font-weight: 600 !important; margin-bottom: 0.3rem !important;
        }}
        .stMetric div[data-testid="stMetricValue"] {{ 
            font-size: 2.2rem !important; color: #FFFFFF !important; font-weight: 700 !important; line-height: 1.1 !important;
        }} 
        .stMetric div[data-testid="stMetricDelta"] {{ 
            font-size: 0.9rem !important; font-weight: 500 !important; padding-top: 0.1rem !important;
        }} 

        .stExpander {{ background-color: #1F2937; border-radius: 8px; margin: 1rem 0; border: 1px solid #374151; }}
        .stExpander header {{ font-size: 1rem; font-weight: 500; color: #E0E0E0; padding: 0.5rem 1rem; }}
        .stTabs [data-baseweb="tab-list"] {{ background-color: #1F2937; border-radius: 8px; padding: 0.5rem; display: flex; justify-content: center; gap: 0.5rem; border-bottom: 2px solid #374151;}}
        .stTabs [data-baseweb="tab"] {{ color: #D1D5DB; padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 500; font-size: 0.95rem; transition: all 0.2s ease-in-out; border: none; border-bottom: 2px solid transparent; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{ background-color: transparent; color: {COLOR_ACCENT_INDIGO}; border-bottom: 2px solid {COLOR_ACCENT_INDIGO}; font-weight:600; }}
        .stTabs [data-baseweb="tab"]:hover {{ background-color: #374151; color: #FFFFFF; }}
        .plot-container {{ background-color: #1F2937; border-radius: 8px; padding: 1rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #374151;}}
        .stPlotlyChart {{ border-radius: 6px; }} 
        .stDataFrame {{ border-radius: 8px; font-size: 0.875rem; border: 1px solid #374151; }}
        .stDataFrame thead th {{ background-color: #293344; color: #EAEAEA; font-weight: 600; }}
        .stDataFrame tbody tr:nth-child(even) {{ background-color: #222C3D; }}
        .stDataFrame tbody tr:hover {{ background-color: #374151; }}
        @media (max-width: 768px) {{ 
            .main {{ padding: 1rem; }} 
            h1 {{ font-size: 1.8rem; }} 
            div[data-testid="stTabs"] section[role="tabpanel"] > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div > h2 {{ font-size: 1.4rem !important; }} 
            div[data-testid="stTabs"] section[role="tabpanel"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] .stSubheader {{ font-size: 1.1rem !important; }} 
            .stPlotlyChart {{ min-height: 300px !important; }} 
            .stTabs [data-baseweb="tab"] {{ padding: 0.5rem 0.8rem; font-size: 0.85rem; }} 
        }}
        .spinner {{ display: flex; justify-content: center; align-items: center; height: 100px; }}
        .spinner::after {{ content: ''; width: 40px; height: 40px; border: 4px solid #4A5568; border-top: 4px solid {COLOR_ACCENT_INDIGO}; border-radius: 50%; animation: spin 0.8s linear infinite; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .onboarding-modal {{ background-color: #1F2937; border: 1px solid #374151; border-radius: 8px; padding: 1.5rem; max-width: 550px; margin: 2rem auto; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }}
        .onboarding-modal h3 {{ color: #EAEAEA; margin-bottom: 1rem; text-align: center; }}
        .onboarding-modal p, .onboarding-modal ul {{ color: #D1D5DB; line-height: 1.6; margin-bottom: 1rem; font-size: 0.9rem; }}
        .onboarding-modal ul {{ list-style-position: inside; padding-left: 0.5rem; }}
        .alert-critical {{ border-left: 5px solid {COLOR_CRITICAL_RED}; background-color: rgba(229, 62, 62, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }} 
        .alert-warning {{ border-left: 5px solid {COLOR_WARNING_AMBER}; background-color: rgba(245, 158, 11, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }}
        .alert-positive {{ border-left: 5px solid {COLOR_POSITIVE_GREEN}; background-color: rgba(16, 185, 129, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }}
        .alert-info {{ border-left: 5px solid {COLOR_INFO_BLUE}; background-color: rgba(59, 130, 246, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }}
        .insight-title {{ font-weight: 600; color: #EAEAEA; margin-bottom: 0.25rem;}}
        .insight-text {{ font-size: 0.9rem; color: #D1D5DB;}}
        .event-item {{padding: 0.3rem 0.5rem; margin-bottom: 0.3rem; background-color: #2a3447; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;}}
        .event-text {{font-size: 0.85rem;}}
        .remove-event-btn button {{background-color: #E53E3E !important; color: white !important; padding: 0.1rem 0.4rem !important; font-size: 0.75rem !important; line-height: 1 !important; border-radius: 3px !important; min-height: auto !important; margin-left: 0.5rem !important;}}
    </style>
""", unsafe_allow_html=True)

# --- render_settings_sidebar ---
def render_settings_sidebar():
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem; color: #A0A0A0;'>Workplace Optimizer</h3>", unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")
        with st.expander("🧪 Simulation Parameters", expanded=True):
            # Initialize session state for direct use by widgets
            if 'sb_team_size_num' not in st.session_state:
                st.session_state.sb_team_size_num = DEFAULT_CONFIG['TEAM_SIZE']
            st.number_input( 
                "Team Size", min_value=1, max_value=200,
                key="sb_team_size_num", # Use same key for value and widget_key
                step=1,
                help="Adjust the number of workers in the simulated shift."
            )

            if 'sb_shift_duration_num' not in st.session_state:
                st.session_state.sb_shift_duration_num = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']
            st.number_input(
                "Shift Duration (min)", min_value=60, max_value=2000, # Max value increased
                key="sb_shift_duration_num",
                step=10,
                help="Set the total length of the simulated work shift in minutes."
            )
            
            current_shift_duration_for_events = st.session_state.sb_shift_duration_num

            st.markdown("---")
            st.markdown("<h5>🗓️ Schedule Shift Events</h5>", unsafe_allow_html=True) 
            st.caption("Define disruptions, breaks, etc. Times are from shift start.")

            if 'sb_scheduled_events_list' not in st.session_state:
                st.session_state.sb_scheduled_events_list = list(DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', []))
            
            event_types = ["Major Disruption", "Minor Disruption", "Scheduled Break", "Short Pause", "Team Meeting", "Maintenance", "Custom Event"]
            
            # Event form state initialization
            if "form_event_type" not in st.session_state: st.session_state.form_event_type = event_types[0]
            if "form_event_start" not in st.session_state: st.session_state.form_event_start = 0
            if "form_event_duration" not in st.session_state: st.session_state.form_event_duration = 10

            with st.container():
                st.session_state.form_event_type = st.selectbox("Event Type", event_types, key="widget_sb_new_event_type_select_RENDER", index=event_types.index(st.session_state.form_event_type))
                
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    st.session_state.form_event_start = st.number_input("Start (min)", min_value=0, max_value=max(0, current_shift_duration_for_events - 1), step=1, key="widget_sb_new_event_start_num_RENDER", help=f"Minutes from shift start (0 to {max(0,current_shift_duration_for_events-1)})")
                with col_time2:
                    st.session_state.form_event_duration = st.number_input("Duration (min)", min_value=1, max_value=current_shift_duration_for_events, step=1, key="widget_sb_new_event_duration_num_RENDER")

            if st.button("➕ Add Event", key="sb_add_event_btn", use_container_width=True, type="primary"): # Standard type
                add_event_type_val = st.session_state.form_event_type
                add_event_start_val = st.session_state.form_event_start
                add_event_duration_val = st.session_state.form_event_duration

                if add_event_start_val + add_event_duration_val > current_shift_duration_for_events:
                    st.warning(f"Event end time ({add_event_start_val + add_event_duration_val} min) exceeds shift duration ({current_shift_duration_for_events} min).")
                elif add_event_start_val < 0 : st.warning("Event start time cannot be negative.")
                elif add_event_duration_val < 1: st.warning("Event duration must be at least 1 minute.")
                else:
                    st.session_state.sb_scheduled_events_list.append({
                        "Event Type": add_event_type_val,
                        "Start Time (min)": add_event_start_val, # Ensure these keys match what simulation expects
                        "Duration (min)": add_event_duration_val,
                        # Add 'step' if your simulation uses it directly from events
                        "step": int(add_event_start_val // (DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2))) # Assuming 2 min/interval from config
                    })
                    st.session_state.sb_scheduled_events_list.sort(key=lambda x: x.get("Start Time (min)", 0))
                    # Reset form input session state values
                    st.session_state.form_event_start = 0 
                    st.session_state.form_event_duration = 10
                    st.rerun()

            st.markdown("<h6>Current Scheduled Events:</h6>", unsafe_allow_html=True) 
            if not st.session_state.sb_scheduled_events_list:
                st.caption("No events scheduled yet.")
            else:
                # Use a container with a fixed height for scrollability
                with st.container(height=200): 
                    for i, event in enumerate(st.session_state.sb_scheduled_events_list):
                        event_col1, event_col2 = st.columns([0.85, 0.15])
                        with event_col1:
                            st.markdown(f"<div class='event-item'><span class='event-text'><b>{event.get('Event Type','N/A')}</b> at {event.get('Start Time (min)','N/A')} min (lasts {event.get('Duration (min)','N/A')} min)</span></div>", unsafe_allow_html=True)
                        with event_col2:
                            if st.button("✖", key=f"remove_event_{i}", help="Remove this event", type="secondary", use_container_width=True):
                                st.session_state.sb_scheduled_events_list.pop(i)
                                st.rerun()
            
            if st.session_state.sb_scheduled_events_list: # Show clear button only if there are events
                if st.button("Clear All Events", key="sb_clear_events_btn", type="secondary", use_container_width=True): # Standard type
                    st.session_state.sb_scheduled_events_list = []
                    st.rerun()
            
            st.markdown("---") 
            team_initiative_opts = ["Standard Operations", "More frequent breaks", "Team recognition", "Increased Autonomy"]
            if 'sb_team_initiative_selectbox' not in st.session_state:
                st.session_state.sb_team_initiative_selectbox = team_initiative_opts[0]
            
            st.selectbox("Operational Initiative", team_initiative_opts, 
                         key="sb_team_initiative_selectbox", 
                         help="Apply an operational strategy to observe its impact on metrics.")
            
            run_simulation_button = st.button("🚀 Run Simulation", key="sb_run_simulation_button", type="primary", use_container_width=True)
        
        # Visualization Options
        with st.expander("🎨 Visualization Options"):
            if 'sb_high_contrast_checkbox' not in st.session_state: st.session_state.sb_high_contrast_checkbox = False
            st.checkbox("High Contrast Plots", key="sb_high_contrast_checkbox", help="Applies a high-contrast color theme to all charts for better accessibility.")
            
            if 'sb_use_3d_distribution_checkbox' not in st.session_state: st.session_state.sb_use_3d_distribution_checkbox = False
            st.checkbox("Enable 3D Worker View", key="sb_use_3d_distribution_checkbox", help="Renders worker positions in a 3D scatter plot.")
            
            if 'sb_debug_mode_checkbox' not in st.session_state: st.session_state.sb_debug_mode_checkbox = False
            st.checkbox("Show Debug Info", key="sb_debug_mode_checkbox", help="Display additional debug information in the sidebar.")
        
        # Data Management & Export
        with st.expander("💾 Data Management & Export"):
            load_data_button = st.button("🔄 Load Previous Simulation", key="sb_load_data_button", use_container_width=True)
            
            can_gen_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("📄 Download Report (.tex)", key="sb_pdf_button", disabled=not can_gen_report, use_container_width=True, help="Generates a LaTeX (.tex) file summarizing the simulation. Requires a LaTeX distribution to compile to PDF."):
                if can_gen_report:
                    try:
                        sim_res = st.session_state.simulation_results
                        sim_cfg_params_pdf = sim_res.get('config_params', {})
                        num_total_steps_pdf = sim_cfg_params_pdf.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) // DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)

                        if num_total_steps_pdf == 0: 
                            st.warning("⚠️ No simulation data (0 steps) for report.")
                            raise SystemExit # Use SystemExit to cleanly stop processing for this button press
                        
                        pdf_data = {}
                        metrics_to_export_pdf = [
                            'operational_recovery', 'psychological_safety', 
                            'productivity_loss', 'task_completion_rate'
                        ]
                        for k in metrics_to_export_pdf:
                            pdf_data[k] = _prepare_timeseries_for_export(sim_res.get(k, []), num_total_steps_pdf)
                        
                        # Handle nested data structures
                        pdf_data['task_compliance'] = _prepare_timeseries_for_export(safe_get(sim_res, 'task_compliance.data', []), num_total_steps_pdf)
                        pdf_data['collaboration_proximity'] = _prepare_timeseries_for_export(safe_get(sim_res, 'collaboration_proximity.data', []), num_total_steps_pdf)
                        pdf_data['worker_wellbeing'] = _prepare_timeseries_for_export(safe_get(sim_res, 'worker_wellbeing.scores', []), num_total_steps_pdf)
                        
                        # Aggregate downtime_minutes (list of event dicts) per step
                        downtime_events_for_pdf = sim_res.get('downtime_minutes', [])
                        pdf_data['downtime_minutes_per_step'] = aggregate_downtime_by_step(downtime_events_for_pdf, num_total_steps_pdf)
                        
                        pdf_data['step'] = list(range(num_total_steps_pdf))
                        minutes_per_interval_pdf = sim_cfg_params_pdf.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) / num_total_steps_pdf if num_total_steps_pdf > 0 else DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)
                        pdf_data['time_minutes'] = [i * minutes_per_interval_pdf for i in range(num_total_steps_pdf)]
                        
                        df_for_pdf = pd.DataFrame(pdf_data)
                        generate_pdf_report(df_for_pdf) # Assuming generate_pdf_report expects a DataFrame
                        st.success("✅ LaTeX report (.tex) 'workplace_report.tex' generated.")
                    except SystemExit: # Catch the explicit SystemExit for 0 steps
                        pass 
                    except Exception as e: 
                        logger.error(f"PDF Gen Error: {e}", exc_info=True, extra={'user_action': 'PDF Generation Error'})
                        st.error(f"❌ PDF Gen Error: {e}")
            
            if can_gen_report: # Changed from checking simulation_results directly
                sim_res_exp = st.session_state.simulation_results
                sim_cfg_params_csv = sim_res_exp.get('config_params', {})
                num_total_steps_csv = sim_cfg_params_csv.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) // DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)

                if num_total_steps_csv > 0:
                    csv_data = {}
                    metrics_to_export_csv = [
                        'operational_recovery', 'psychological_safety', 
                        'productivity_loss', 'task_completion_rate'
                    ]
                    for k in metrics_to_export_csv:
                         csv_data[k] = _prepare_timeseries_for_export(sim_res_exp.get(k, []), num_total_steps_csv)

                    # Handle nested data structures for CSV
                    csv_data['task_compliance'] = _prepare_timeseries_for_export(safe_get(sim_res_exp, 'task_compliance.data', []), num_total_steps_csv)
                    csv_data['collaboration_proximity'] = _prepare_timeseries_for_export(safe_get(sim_res_exp, 'collaboration_proximity.data', []), num_total_steps_csv)
                    
                    ww_data_csv = sim_res_exp.get('worker_wellbeing', {})
                    csv_data['worker_wellbeing_index'] = _prepare_timeseries_for_export(ww_data_csv.get('scores', []), num_total_steps_csv)
                    csv_data['team_cohesion'] = _prepare_timeseries_for_export(ww_data_csv.get('team_cohesion_scores', []), num_total_steps_csv)
                    csv_data['perceived_workload'] = _prepare_timeseries_for_export(ww_data_csv.get('perceived_workload_scores', []), num_total_steps_csv)

                    downtime_events_for_csv = sim_res_exp.get('downtime_minutes', [])
                    csv_data['downtime_minutes_per_step'] = aggregate_downtime_by_step(downtime_events_for_csv, num_total_steps_csv)
                    
                    csv_data['step'] = list(range(num_total_steps_csv))
                    minutes_per_interval_csv = sim_cfg_params_csv.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) / num_total_steps_csv if num_total_steps_csv > 0 else DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)
                    csv_data['time_minutes'] = [i * minutes_per_interval_csv for i in range(num_total_steps_csv)]
                    
                    df_for_csv = pd.DataFrame(csv_data)
                    st.download_button("📥 Download Data (CSV)", df_for_csv.to_csv(index=False).encode('utf-8'), "workplace_summary.csv", "text/csv", key="sb_csv_dl_button", use_container_width=True)
                else: 
                    st.caption("No detailed data to export (0 simulation steps).")
            elif not can_gen_report : 
                st.caption("Run simulation for export options.")
        
        if st.session_state.get('sb_debug_mode_checkbox', False): # Use .get for safety
            with st.expander("🛠️ Debug Information", expanded=False):
                st.write("**Default Config (Partial):**")
                st.json({k: DEFAULT_CONFIG.get(k) for k in ['ENTRY_EXIT_POINTS', 'WORK_AREAS', 'DEFAULT_SCHEDULED_EVENTS', 'MINUTES_PER_INTERVAL']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results: 
                    st.write("**Active Simulation Config (from results):**")
                    st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else: st.write("**No active simulation data.**")
        
        st.markdown("## 📋 Help & Info")
        if st.button("ℹ️ Help & Glossary", key="sb_help_button", use_container_width=True): 
            st.session_state.show_help_glossary = not st.session_state.get('show_help_glossary', False)
            st.rerun() # Rerun to show/hide modal
        if st.button("🚀 Quick Tour", key="sb_tour_button", use_container_width=True): 
            st.session_state.show_tour = not st.session_state.get('show_tour', False)
            st.rerun() # Rerun to show/hide modal
            
    return run_simulation_button, load_data_button # Only return button states, other values are read from session_state

# --- run_simulation_logic (Caching re-enabled) ---
@st.cache_data(ttl=3600, show_spinner="⚙️ Running simulation model...") 
def run_simulation_logic(team_size, shift_duration_minutes, scheduled_events_list_of_dicts, team_initiative_selected):
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    # Ensure MINUTES_PER_INTERVAL is in config, default if not
    minutes_per_interval = config.get('MINUTES_PER_INTERVAL', 2) 
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // minutes_per_interval
    
    # Ensure events have a 'step' if the simulation expects it.
    # The sidebar now adds 'step' based on 'Start Time (min)' and MINUTES_PER_INTERVAL.
    processed_scheduled_events = []
    for event_dict in scheduled_events_list_of_dicts:
        new_event = event_dict.copy()
        if 'step' not in new_event and 'Start Time (min)' in new_event:
            new_event['step'] = int(new_event['Start Time (min)'] // minutes_per_interval)
        processed_scheduled_events.append(new_event)
    config['SCHEDULED_EVENTS'] = processed_scheduled_events
    
    logger.info(f"run_simulation_logic: SCHEDULED_EVENTS processed: {config['SCHEDULED_EVENTS']}", extra={'user_action': 'Process Scheduled Events'})

    # Worker distribution logic 
    if 'WORK_AREAS' in config and isinstance(config['WORK_AREAS'], dict) and config['WORK_AREAS']:
        total_workers_in_config_zones = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values() if isinstance(zone, dict)) # Added check for zone type
        if total_workers_in_config_zones != team_size and team_size > 0:
            logger.info(f"Adjusting worker distribution in config based on team size {team_size}. Configured sum was {total_workers_in_config_zones}.", extra={'user_action': 'Adjust Worker Distribution'})
            if total_workers_in_config_zones > 0: 
                ratio = team_size / total_workers_in_config_zones
                accumulated_workers = 0
                # Sort keys to ensure consistent redistribution if multiple runs with same initial config
                sorted_zone_keys = sorted([k for k, v in config['WORK_AREAS'].items() if isinstance(v, dict)]) 
                for zone_key_idx, zone_key in enumerate(sorted_zone_keys):
                    zone_data = config['WORK_AREAS'][zone_key] # Should be a dict
                    if zone_key_idx < len(sorted_zone_keys) - 1:
                        workers_prop = zone_data.get('workers', 0) * ratio
                        assigned_val = int(round(workers_prop)) 
                        zone_data['workers'] = assigned_val
                        accumulated_workers += assigned_val
                    else: # Last zone takes the remainder
                        remaining_workers_to_assign = team_size - accumulated_workers
                        zone_data['workers'] = remaining_workers_to_assign
                        if zone_data['workers'] < 0:
                            logger.warning(f"Negative workers ({zone_data['workers']}) calculated for last zone {zone_key}. Setting to 0 and attempting re-balance.", extra={'user_action': 'Worker Distribution Warning'})
                            zone_data['workers'] = 0 
                            # Attempt to add deficit to the first zone if possible
                            final_sum_check = sum(z.get('workers',0) for z_key in sorted_zone_keys for z in [config['WORK_AREAS'][z_key]] if isinstance(z, dict))
                            if final_sum_check != team_size and sorted_zone_keys:
                                deficit = team_size - final_sum_check
                                config['WORK_AREAS'][sorted_zone_keys[0]]['workers'] += deficit
                                logger.info(f"Re-balanced: Added {deficit} workers to zone {sorted_zone_keys[0]}.", extra={'user_action': 'Worker Re-balance'})
            else: # If no workers initially configured in zones, distribute as evenly as possible
                num_zones = len([k for k, v in config['WORK_AREAS'].items() if isinstance(v, dict)])
                if num_zones > 0:
                    workers_per_zone = team_size // num_zones
                    remainder_workers = team_size % num_zones
                    for i, zone_key in enumerate([k for k, v in config['WORK_AREAS'].items() if isinstance(v, dict)]):
                        config['WORK_AREAS'][zone_key]['workers'] = workers_per_zone + (1 if i < remainder_workers else 0)
        elif team_size == 0: 
             for zone_key in config['WORK_AREAS']: 
                 if isinstance(config['WORK_AREAS'][zone_key], dict):
                     config['WORK_AREAS'][zone_key]['workers'] = 0
    
    validate_config(config) 
    logger.info(f"Running simulation with: Team Size={team_size}, Duration={shift_duration_minutes}min ({config['SHIFT_DURATION_INTERVALS']} intervals), Scheduled Events: {len(config['SCHEDULED_EVENTS'])} events, Initiative: {team_initiative_selected}", extra={'user_action': 'Run Simulation - Start'})
    
    sim_results_tuple = simulate_workplace_operations(
        num_team_members=team_size,
        num_steps=config['SHIFT_DURATION_INTERVALS'],
        scheduled_events=config['SCHEDULED_EVENTS'], 
        team_initiative=team_initiative_selected,
        config=config # Pass the full config
    )
    
    # Ensure all expected keys from simulation are present
    expected_keys = ['team_positions_df', 'task_compliance', 'collaboration_proximity', 
                     'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 
                     'worker_wellbeing', 'psychological_safety', 'feedback_impact', 
                     'downtime_minutes', 'task_completion_rate']
    
    if not isinstance(sim_results_tuple, tuple) or len(sim_results_tuple) != len(expected_keys):
        err_msg = f"Simulation returned unexpected data format. Expected tuple of length {len(expected_keys)}, got {type(sim_results_tuple)} of length {len(sim_results_tuple) if isinstance(sim_results_tuple, (tuple,list)) else 'N/A'}." 
        logger.critical(err_msg, extra={'user_action': 'Run Simulation - CRITICAL Data Format Error'})
        # Optionally, log parts of the sim_results_tuple if it's not too large
        # logger.critical(f"Received data (partial): {str(sim_results_tuple)[:200]}", extra={'user_action': 'Run Simulation - CRITICAL Data Format Error'})
        raise TypeError(err_msg)
        
    simulation_output_dict = dict(zip(expected_keys, sim_results_tuple))
    
    # Store key config parameters used for this simulation run
    # This helps in reloading and understanding the context of the results
    simulation_output_dict['config_params'] = {
        'TEAM_SIZE': team_size,
        'SHIFT_DURATION_MINUTES': shift_duration_minutes,
        'SHIFT_DURATION_INTERVALS': config['SHIFT_DURATION_INTERVALS'],
        'MINUTES_PER_INTERVAL': minutes_per_interval,
        'SCHEDULED_EVENTS': config['SCHEDULED_EVENTS'], # Store the processed events
        'TEAM_INITIATIVE': team_initiative_selected,
        # Store a copy of WORK_AREAS as it was used in the simulation
        'WORK_AREAS_EFFECTIVE': config.get('WORK_AREAS', {}).copy() 
    }
    
    # Derive disruption event steps from the *actual* scheduled events used
    disruption_event_steps_derived = []
    for event in config['SCHEDULED_EVENTS']: # Use events from the config dict
        if isinstance(event, dict) and "Disruption" in event.get("Event Type",""):
            # Use 'step' if available, otherwise calculate from 'Start Time (min)'
            event_step = event.get('step')
            if event_step is None: # Fallback if 'step' wasn't in the event dict
                 start_time = event.get("Start Time (min)")
                 if isinstance(start_time, (int, float)) and start_time >=0:
                    event_step = int(start_time // minutes_per_interval)
            
            if isinstance(event_step, int):
                 disruption_event_steps_derived.append(event_step)

    simulation_output_dict['config_params']['DISRUPTION_EVENT_STEPS'] = sorted(list(set(disruption_event_steps_derived)))

    save_simulation_data(simulation_output_dict) 
    return simulation_output_dict

# --- time_range_input_section ---
def time_range_input_section(tab_key_prefix: str, max_minutes: int, st_col_obj = st, interval_duration_min: int = 2):
    start_time_key = f"{tab_key_prefix}_start_time_min"
    end_time_key = f"{tab_key_prefix}_end_time_min"

    # Initialize session state for these keys if they don't exist
    if start_time_key not in st.session_state:
        st.session_state[start_time_key] = 0
    if end_time_key not in st.session_state:
        st.session_state[end_time_key] = max_minutes
    
    # Ensure current values in session state are valid against current max_minutes
    st.session_state[start_time_key] = min(st.session_state[start_time_key], max_minutes)
    st.session_state[start_time_key] = max(0, st.session_state[start_time_key])
    st.session_state[end_time_key] = min(st.session_state[end_time_key], max_minutes)
    st.session_state[end_time_key] = max(st.session_state[start_time_key], st.session_state[end_time_key]) 
    
    cols = st_col_obj.columns(2)
    
    # Store original values to detect change
    prev_start_time = st.session_state[start_time_key]
    prev_end_time = st.session_state[end_time_key]

    # Use on_change or direct key manipulation for simplicity here
    # The widgets will update their respective session_state keys directly.
    cols[0].number_input(
        "Start Time (min)", 
        min_value=0, 
        max_value=max_minutes, 
        # value=st.session_state[start_time_key], # Not needed if key is same
        step=interval_duration_min, # Use interval duration for step
        key=start_time_key, 
        help=f"Select the start of the time range (in minutes from shift start, 0 to {max_minutes})."
    )
    # Ensure end_time_min_value is at least the current start_time_key
    end_time_min_value = st.session_state[start_time_key]
    cols[1].number_input( 
        "End Time (min)", 
        min_value=end_time_min_value, 
        max_value=max_minutes, 
        # value=st.session_state[end_time_key], # Not needed if key is same
        step=interval_duration_min,
        key=end_time_key,
        help=f"Select the end of the time range (in minutes from shift start, {end_time_min_value} to {max_minutes})."
    )
    
    # Post-widget interaction: ensure end time is not less than start time
    if st.session_state[end_time_key] < st.session_state[start_time_key]:
        st.session_state[end_time_key] = st.session_state[start_time_key]
    
    # Check if values actually changed to trigger a rerun
    # This prevents reruns if the user just clicks away without changing values
    # or if programmatic corrections didn't change the final state.
    if prev_start_time != st.session_state[start_time_key] or \
       prev_end_time != st.session_state[end_time_key]:
        st.rerun()

    return int(st.session_state[start_time_key]), int(st.session_state[end_time_key])

# --- MAIN FUNCTION ---
def main():
    st.title("Workplace Shift Optimization Dashboard")
    
    # Centralized session state initialization
    app_state_defaults = {
        'simulation_results': None,
        'show_tour': False,
        'show_help_glossary': False,
        'sb_team_size_num': DEFAULT_CONFIG['TEAM_SIZE'],
        'sb_shift_duration_num': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
        'sb_scheduled_events_list': list(DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', [])),
        'sb_team_initiative_selectbox': "Standard Operations",
        'sb_high_contrast_checkbox': False,
        'sb_use_3d_distribution_checkbox': False,
        'sb_debug_mode_checkbox': False,
        'form_event_type': ["Major Disruption", "Minor Disruption", "Scheduled Break", "Short Pause", "Team Meeting", "Maintenance", "Custom Event"][0], # Default for event form
        'form_event_start': 0, # Default for event form
        'form_event_duration': 10, # Default for event form
        # Time range selectors for tabs (will be updated based on sim duration)
        'op_start_time_min': 0, 'op_end_time_min': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
        'ww_start_time_min': 0, 'ww_end_time_min': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
        'dt_start_time_min': 0, 'dt_end_time_min': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
    }
    for key, default_value in app_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            
    # Call render_settings_sidebar to display UI and get interactive element states
    sb_run_sim_btn, sb_load_data_btn = render_settings_sidebar() 
    
    # Parameters for simulation are now read directly from session_state
    current_team_size = st.session_state.sb_team_size_num
    current_shift_duration = st.session_state.sb_shift_duration_num
    current_scheduled_events = list(st.session_state.sb_scheduled_events_list) # Ensure it's a list copy
    current_team_initiative = st.session_state.sb_team_initiative_selectbox
    
    # Visualization options from session_state
    current_high_contrast_setting = st.session_state.sb_high_contrast_checkbox
    sb_use_3d_val = st.session_state.sb_use_3d_distribution_checkbox
    # sb_debug_mode_val = st.session_state.sb_debug_mode_checkbox # Used in sidebar directly

    minutes_per_interval = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)
    disruption_steps_for_plots = [] 

    # Determine max minutes for time range inputs and disruption steps
    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        sim_cfg = st.session_state.simulation_results.get('config_params', {})
        # Use actual shift duration from THAT sim for max minutes
        sim_shift_duration_cfg = sim_cfg.get('SHIFT_DURATION_MINUTES', current_shift_duration)
        sim_intervals_cfg = sim_cfg.get('SHIFT_DURATION_INTERVALS', sim_shift_duration_cfg // minutes_per_interval)
        minutes_per_interval = sim_cfg.get('MINUTES_PER_INTERVAL', minutes_per_interval)

        current_max_minutes_for_inputs = max(0, (sim_intervals_cfg) * minutes_per_interval - minutes_per_interval) if sim_intervals_cfg > 0 else 0
        
        # Use pre-calculated disruption steps if available, otherwise derive
        disruption_steps_for_plots = sim_cfg.get('DISRUPTION_EVENT_STEPS', [])
        if not disruption_steps_for_plots: # Fallback if not in stored config
            loaded_scheduled_events = sim_cfg.get('SCHEDULED_EVENTS', [])
            for event in loaded_scheduled_events: 
                if isinstance(event, dict) and "Disruption" in event.get("Event Type", ""):
                    step = event.get('step') # Prefer 'step' if available
                    if step is None:
                        start_time = event.get("Start Time (min)")
                        if isinstance(start_time, (int, float)) and start_time >= 0:
                            step = int(start_time // minutes_per_interval)
                    if isinstance(step, int): disruption_steps_for_plots.append(step)
            disruption_steps_for_plots = sorted(list(set(disruption_steps_for_plots)))
        # logger.debug(f"Derived disruption_steps_for_plots from loaded config: {disruption_steps_for_plots}", extra={'user_action': 'Derive Disrupt Steps'})
    else: # No simulation results yet, use sidebar settings
        num_intervals_from_sidebar = current_shift_duration // minutes_per_interval
        current_max_minutes_for_inputs = max(0, num_intervals_from_sidebar * minutes_per_interval - minutes_per_interval) if num_intervals_from_sidebar > 0 else 0
        
        for event in current_scheduled_events: 
            if isinstance(event, dict) and "Disruption" in event.get("Event Type", ""):
                step = event.get('step')
                if step is None:
                    start_time = event.get("Start Time (min)")
                    if isinstance(start_time, (int, float)) and start_time >= 0:
                        step = int(start_time // minutes_per_interval)
                if isinstance(step, int): disruption_steps_for_plots.append(step)
        disruption_steps_for_plots = sorted(list(set(disruption_steps_for_plots)))
    
    current_max_minutes_for_inputs = max(0, current_max_minutes_for_inputs) # Final sanity check
    # logger.debug(f"Main: current_max_minutes_for_inputs final value for UI: {current_max_minutes_for_inputs}", extra={'user_action': 'Set Max Minutes'})

    # --- Simulation run and load logic ---
    if sb_run_sim_btn:
        with st.spinner("🚀 Simulating workplace operations..."): # spinner class in CSS
            try:
                logger.info(f"Events passed to run_simulation_logic: {current_scheduled_events}", extra={'user_action': 'Prepare Simulation Run'})
                sim_results = run_simulation_logic(
                    current_team_size, current_shift_duration, current_scheduled_events, current_team_initiative
                )
                st.session_state.simulation_results = sim_results
                
                # Update time range selectors based on the new simulation's duration
                new_sim_cfg = sim_results['config_params']
                new_sim_intervals = new_sim_cfg.get('SHIFT_DURATION_INTERVALS',0)
                new_minutes_per_interval = new_sim_cfg.get('MINUTES_PER_INTERVAL', 2)
                new_max_mins = max(0, new_sim_intervals * new_minutes_per_interval - new_minutes_per_interval) if new_sim_intervals > 0 else 0
                
                current_max_minutes_for_inputs = new_max_mins # Update for immediate use
                for prefix in ['op', 'ww', 'dt']:
                    st.session_state[f"{prefix}_start_time_min"] = 0
                    st.session_state[f"{prefix}_end_time_min"] = new_max_mins

                st.success("✅ Simulation completed!")
                logger.info("Simulation run successful.", extra={'user_action': 'Run Simulation - Success'})
                st.rerun() 
            except Exception as e:
                logger.error(f"Simulation Run Error: {e}", exc_info=True, extra={'user_action': 'Run Simulation - Error'})
                st.error(f"❌ Simulation failed: {str(e)}") 
                st.session_state.simulation_results = None # Clear results on failure

    if sb_load_data_btn:
        with st.spinner("🔄 Loading saved simulation data..."): # spinner class in CSS
            try:
                loaded_data = load_simulation_data()
                if loaded_data and isinstance(loaded_data, dict) and 'config_params' in loaded_data:
                    st.session_state.simulation_results = loaded_data
                    cfg = loaded_data['config_params'] # Must have config_params
                    
                    # Update sidebar controls to reflect loaded simulation's parameters
                    st.session_state.sb_team_size_num = cfg.get('TEAM_SIZE', DEFAULT_CONFIG['TEAM_SIZE'])
                    st.session_state.sb_shift_duration_num = cfg.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'])
                    st.session_state.sb_scheduled_events_list = list(cfg.get('SCHEDULED_EVENTS', DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', [])))
                    st.session_state.sb_team_initiative_selectbox = cfg.get('TEAM_INITIATIVE', "Standard Operations")
                     
                    # Update time range selectors based on loaded simulation's duration
                    loaded_sim_intervals = cfg.get('SHIFT_DURATION_INTERVALS', 0)
                    loaded_minutes_per_interval = cfg.get('MINUTES_PER_INTERVAL', 2)
                    new_max_mins_load = max(0, loaded_sim_intervals * loaded_minutes_per_interval - loaded_minutes_per_interval) if loaded_sim_intervals > 0 else 0
                    
                    current_max_minutes_for_inputs = new_max_mins_load # Update global max
                    for prefix in ['op', 'ww', 'dt']:
                        st.session_state[f"{prefix}_start_time_min"] = 0
                        st.session_state[f"{prefix}_end_time_min"] = new_max_mins_load
                    
                    st.success("✅ Data loaded successfully!")
                    logger.info("Saved data loaded successfully.", extra={'user_action': 'Load Data - Success'})
                    st.rerun()
                else:
                    st.error("❌ Failed to load data or data is incomplete/invalid.")
                    logger.warning("Load data failed or invalid format (missing 'config_params').", extra={'user_action': 'Load Data - Fail/Invalid'})
                    st.session_state.simulation_results = None # Clear if load fails
            except Exception as e:
                logger.error(f"Load Data Error: {e}", exc_info=True, extra={'user_action': 'Load Data - Error'})
                st.error(f"❌ Failed to load data: {e}")
                st.session_state.simulation_results = None

    # --- Modals for Tour & Help ---
    if st.session_state.get('show_tour'): 
        with st.container(): # Using st.container for modal structure
             st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3><p>Welcome! This dashboard helps you monitor and analyze workplace shift operations. Use the sidebar to configure simulations and navigate. The main area displays results across several tabs: Overview, Operational Metrics, Worker Well-being (including psychosocial factors and spatial dynamics), Downtime Analysis, and a Glossary. Interactive charts and actionable insights will guide you in optimizing operations.</p><p>Start by running a new simulation or loading previous data from the sidebar!</p></div>""", unsafe_allow_html=True)
        if st.button("Got it!", key="tour_modal_close_btn"): 
            st.session_state.show_tour = False
            st.rerun()
    if st.session_state.get('show_help_glossary'): 
        with st.container():
            st.markdown(""" <div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3> <p>This dashboard provides insights into simulated workplace operations. Use the sidebar to configure and run simulations or load previously saved data. Navigate through the analysis using the main tabs above.</p><h4>Metric Definitions:</h4> <ul style="font-size: 0.85rem; list-style-type: disc; padding-left: 20px;"> <li><b>Task Compliance Score:</b> Percentage of tasks completed correctly and on time.</li><li><b>Collaboration Proximity Index:</b> Percentage of workers near colleagues, indicating teamwork potential.</li><li><b>Operational Recovery Score:</b> Ability to maintain output after disruptions.</li><li><b>Worker Well-Being Index:</b> Composite score of fatigue, stress levels, and job satisfaction.</li><li><b>Psychological Safety Score:</b> Comfort level in reporting issues or suggesting improvements.</li><li><b>Team Cohesion Index:</b> Measure of bonds and sense of belonging within a team.</li><li><b>Perceived Workload Index:</b> Indicator of how demanding workers perceive their tasks (0-10 scale).</li><li><b>Uptime:</b> Percentage of time equipment is operational.</li><li><b>Throughput:</b> Percentage of maximum production rate achieved.</li><li><b>Quality Rate:</b> Percentage of products meeting quality standards.</li><li><b>OEE (Overall Equipment Effectiveness):</b> Combined score of Uptime, Throughput, and Quality Rate.</li><li><b>Productivity Loss:</b> Percentage of potential output lost due to inefficiencies.</li><li><b>Downtime (per interval):</b> Total minutes of unplanned operational stops. Assumed simulation output `downtime_minutes` is a list of event dicts, each with `step`, `duration`, `type`.</li><li><b>Task Completion Rate:</b> Percentage of tasks completed per time interval.</li></ul><p>For further assistance, please refer to the detailed documentation or contact support@example.com.</p></div> """, unsafe_allow_html=True) 
        if st.button("Understood", key="help_modal_close_btn"): 
            st.session_state.show_help_glossary = False
            st.rerun()

    # --- Main Content Tabs ---
    tabs_main_names = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Well-being", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs = st.tabs(tabs_main_names)
    plot_config_interactive = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'pan2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}
    plot_config_minimal = {'displayModeBar': False} # For simple gauges
    
    # --- Overview Tab ---
    with tabs[0]: 
        st.header("📊 Key Performance Indicators & Actionable Insights", divider="blue")
        if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
            sim_data = st.session_state.simulation_results
            # Ensure effective_config uses DEFAULT_CONFIG as base, then overrides with sim_data's config_params
            effective_config_base = DEFAULT_CONFIG.copy()
            effective_config_base.update(sim_data.get('config_params', {})) # Update with sim specific config
            effective_config = effective_config_base

            compliance_target = float(effective_config.get('TARGET_COMPLIANCE', 75.0))
            collab_target = float(effective_config.get('TARGET_COLLABORATION', 60.0))
            wb_target = float(effective_config.get('TARGET_WELLBEING', 70.0))
            
            # For total downtime, sum durations from the list of event dicts
            downtime_events_overview = safe_get(sim_data, 'downtime_minutes', [])
            downtime_durations_overview = [event.get('duration', 0.0) for event in downtime_events_overview if isinstance(event, dict)]
            
            compliance_val = safe_stat(safe_get(sim_data, 'task_compliance.data', []), np.mean, default_val=0.0)
            proximity_val = safe_stat(safe_get(sim_data, 'collaboration_proximity.data', []), np.mean, default_val=0.0)
            wellbeing_val = safe_stat(safe_get(sim_data, 'worker_wellbeing.scores', []), np.mean, default_val=0.0)
            downtime_total_overview = safe_stat(downtime_durations_overview, np.sum, default_val=0.0)

            sim_duration_minutes_cfg = float(effective_config.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
            dt_target_total_shift_percentage = float(effective_config.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT', 0.05)) 
            dt_target_total_shift = sim_duration_minutes_cfg * dt_target_total_shift_percentage
            
            cols_metrics = st.columns(4)
            cols_metrics[0].metric("Task Compliance", f"{compliance_val:.1f}%", f"{compliance_val-compliance_target:.1f}% vs Target {compliance_target:.0f}%")
            cols_metrics[1].metric("Collaboration Index", f"{proximity_val:.1f}%", f"{proximity_val-collab_target:.1f}% vs Target {collab_target:.0f}%")
            cols_metrics[2].metric("Worker Well-Being", f"{wellbeing_val:.1f}%", f"{wellbeing_val-wb_target:.1f}% vs Target {wb_target:.0f}%")
            cols_metrics[3].metric("Total Downtime", f"{downtime_total_overview:.1f} min", f"{downtime_total_overview-dt_target_total_shift:.1f} min vs Target {dt_target_total_shift:.0f}min", delta_color="inverse")
            
            try:
                summary_figs = plot_key_metrics_summary(
                    compliance=compliance_val, proximity=proximity_val, wellbeing=wellbeing_val, 
                    downtime=downtime_total_overview, target_compliance=compliance_target, 
                    target_proximity=collab_target, target_wellbeing=wb_target, 
                    target_downtime=dt_target_total_shift, high_contrast=current_high_contrast_setting,
                    color_positive=COLOR_POSITIVE_GREEN, color_warning=COLOR_WARNING_AMBER,
                    color_negative=COLOR_CRITICAL_RED, accent_color=COLOR_ACCENT_INDIGO
                )
                if summary_figs: # Check if figs were generated
                    cols_gauges = st.columns(min(len(summary_figs), 4) or 1) # Ensure at least 1 column
                    for i_gauge, fig_gauge in enumerate(summary_figs): 
                        if fig_gauge: # Check if individual figure is not None
                            cols_gauges[i_gauge % len(cols_gauges)].plotly_chart(fig_gauge, use_container_width=True, config=plot_config_minimal)
                else: st.caption("Gauge charts could not be generated (no data or error).")
            except Exception as e: 
                logger.error(f"Overview Gauges Plot Error: {e}", exc_info=True, extra={'user_action': 'Gauge Plot Error'})
                st.error(f"⚠️ Error rendering overview gauges: {str(e)}")

            st.markdown("---"); st.subheader("💡 Key Insights & Leadership Actions")
            actionable_insights = get_actionable_insights(sim_data, effective_config)
            if actionable_insights:
                for insight in actionable_insights: 
                    st.markdown(f'<div class="alert-{insight["type"]}"><p class="insight-title">{insight["title"]}</p><p class="insight-text">{insight["text"]}</p></div>', unsafe_allow_html=True)
            else: st.info("✅ No critical alerts or specific insights identified based on current thresholds.", icon="👍")
            
            with st.expander("View Detailed Overview Data Table", expanded=False):
                # For table, use aggregated downtime per step
                num_total_steps_overview = effective_config.get('SHIFT_DURATION_INTERVALS', 0)
                minutes_per_interval_overview = effective_config.get('MINUTES_PER_INTERVAL', 2)

                if num_total_steps_overview > 0:
                    df_data_overview = {'Time (min)': [i * minutes_per_interval_overview for i in range(num_total_steps_overview)]}
                    df_data_overview['Task Compliance (%)'] = _prepare_timeseries_for_export(safe_get(sim_data, 'task_compliance.data', []), num_total_steps_overview)
                    df_data_overview['Collaboration (%)'] = _prepare_timeseries_for_export(safe_get(sim_data, 'collaboration_proximity.data', []), num_total_steps_overview)
                    df_data_overview['Well-Being (%)'] = _prepare_timeseries_for_export(safe_get(sim_data, 'worker_wellbeing.scores', []), num_total_steps_overview)
                    
                    # Aggregate downtime_minutes for the table
                    downtime_events_for_table = safe_get(sim_data, 'downtime_minutes', [])
                    df_data_overview['Downtime (min/interval)'] = aggregate_downtime_by_step(downtime_events_for_table, num_total_steps_overview)
                    
                    st.dataframe(pd.DataFrame(df_data_overview).style.format("{:.1f}", na_rep="-").set_table_styles([{'selector': 'th', 'props': [('background-color', '#293344'), ('color', '#EAEAEA')]}]), use_container_width=True, height=300)
                else: st.caption("No detailed overview data (0 simulation steps).")
        else: st.info("ℹ️ Run a simulation or load data to view the Overview & Insights.", icon="📊")
    
    # --- Tab Definitions & Rendering Loop ---
    op_insights_html = """<div class='alert-info insight-text' style='margin-top:1rem;'><p class="insight-title">Review Operational Bottlenecks:</p><ul><li><b>Low Compliance/OEE:</b> If Task Compliance or OEE components (Uptime, Throughput, Quality) are consistently low or dip significantly, investigate the root causes. Are these correlated with disruptions, high workload periods, or specific zones?</li><li><b>Recovery Performance:</b> Evaluate how quickly Operational Recovery returns to target after disruptions. Slow recovery indicates a need for improved contingency plans or resource flexibility.</li><li><b>Collaboration Impact:</b> If Collaboration Index is low and operational metrics suffer, it may indicate communication breakdowns or poor team synergy affecting task handoffs. Consider targeted team interventions or process clarifications.</li></ul><p class="insight-title">Strategic Considerations:</p><p>Use the "Operational Initiative" setting in the sidebar to simulate changes (e.g., new break policies, recognition programs). Compare these scenarios against a "Standard Operations" baseline to quantify the ROI and impact of leadership decisions on operational KPIs and worker well-being.</p></div>"""
    ww_static_insights_html = """ 
            <h6 style='margin-top:1.5rem;'>💡 Considerations for Psychosocial Well-being:</h6>
            <ul style="font-size:0.9rem; color: #D1D5DB; padding-left:20px; margin-bottom:0;">
                <li><strong>Monitor Psychosocial Risk Factors:</strong> Regularly review Well-being, Psychological Safety, Team Cohesion, and Perceived Workload indices. Dips or consistently low scores require proactive investigation.</li>
                <li><strong>Spatial Awareness:</strong> Correlate high-density zones or areas with isolated workers (from Distribution and Heatmap plots) with well-being or productivity metrics. Overcrowding can increase stress, while isolation can reduce cohesion.</li>
                <li><strong>Evaluate Initiatives:</strong> Actively use the "Operational Initiative" setting in the sidebar to test strategies like 'more frequent breaks' or 'team recognition'. Compare results against a 'Standard Operations' baseline to quantify the ROI and impact of leadership decisions on workplace policies.</li>
                <li><strong>Empowerment & Control:</strong> The "Increased Autonomy" initiative's impact on psychological safety and well-being can guide decisions on job design and worker empowerment.</li>
                <li><strong>Prevent Burnout:</strong> Address sustained high workload or low well-being proactively to prevent burnout, which severely impacts long-term productivity and retention.</li>
            </ul>""" 
    dt_insights_html = """<div class='alert-info insight-text' style='margin-top:1rem;'><p class="insight-title">Focus Areas for Downtime Reduction:</p><ul><li><strong>Prioritize by Cause:</strong> Use the 'Downtime by Cause' pie chart to pinpoint the primary reasons for lost time. Allocate resources to address the largest segments first. If 'Equipment Failure' dominates, schedule reliability assessments and enhance preventive maintenance. If 'Material Shortage' is prevalent, review supply chain and inventory management.</li><li><strong>Analyze Trend Plot for Patterns:</strong> Look for patterns in the 'Downtime Trend' bar chart. Are there specific times of day or intervals with recurring high downtime? This might point to shift change issues, inadequate handovers, or processes that are more failure-prone under certain conditions.</li><li><strong>Incident Frequency vs. Severity:</strong> A high number of short downtime incidents can be as damaging as a few long ones due to the cumulative effect and the effort of restarting. Address both systemic minor issues and prepare for less frequent major ones.</li><li><strong>Disruption Correlation:</strong> Are downtime spikes often preceded or accompanied by events on the 'Operational Metrics' tab (e.g., drops in compliance, OEE)? Understanding these correlations can help in developing more resilient operational plans.</li></ul></div>"""

    tab_configs = [
        {"name": "📈 Operational Metrics", "key_prefix": "op", 
         "plots": [
             {"title": "Task Compliance Score Over Time", "data_path": "task_compliance.data", "plot_func": plot_task_compliance_score, "extra_args_paths": {"forecast_data": "task_compliance.forecast", "z_scores": "task_compliance.z_scores"}},
             {"title": "Collaboration Proximity Index Over Time", "data_path": "collaboration_proximity.data", "plot_func": plot_collaboration_proximity_index, "extra_args_paths": {"forecast_data": "collaboration_proximity.forecast"}},
             {"is_subheader": True, "title": "Additional Operational Metrics"}, 
             {"title": "Operational Recovery vs. Loss", "data_path": "operational_recovery", "plot_func": plot_operational_recovery, "extra_args_paths": {"productivity_loss_data": "productivity_loss"}},
             {"title": "OEE & Components", "is_oee": True} 
         ],
         "insights_html": op_insights_html
        },
        {"name": "👥 Worker Well-being", "key_prefix": "ww", 
         "plots": [
             {"is_subheader": True, "title": "Psychosocial & Well-being Indicators"},
             {"title": "Worker Well-Being Index", "data_path": "worker_wellbeing.scores", "plot_func": plot_worker_wellbeing, "extra_args_paths": {"triggers": "worker_wellbeing.triggers"}}, # Triggers are passed whole, plot func handles step filtering
             {"title": "Psychological Safety Score", "data_path": "psychological_safety", "plot_func": plot_psychological_safety},
             {"title": "Team Cohesion Index", "data_path": "worker_wellbeing.team_cohesion_scores", "plot_func": plot_team_cohesion},
             {"title": "Perceived Workload Index (0-10)", "data_path": "worker_wellbeing.perceived_workload_scores", "plot_func": plot_perceived_workload, "extra_args_fixed": {"high_workload_threshold": DEFAULT_CONFIG.get('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5), "very_high_workload_threshold": DEFAULT_CONFIG.get('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5)}},
             {"is_subheader": True, "title": "Spatial Dynamics Analysis", "is_spatial": True}
         ],
         "dynamic_insights_func": "render_wellbeing_alerts", 
         "insights_html": ww_static_insights_html 
        },
        {"name": "⏱️ Downtime Analysis", "key_prefix": "dt", 
         "metrics_display": True,
         "plots": [
            {"title": "Downtime Trend (per Interval)", "data_path": "downtime_minutes", "plot_func": plot_downtime_trend, "is_event_based_aggregation": True, "extra_args_fixed": {"interval_threshold": DEFAULT_CONFIG.get('DOWNTIME_PLOT_ALERT_THRESHOLD', 10)}},
            {"title": "Downtime Distribution by Cause", "data_path": "downtime_minutes", "plot_func": plot_downtime_causes_pie, "is_event_based_filtering": True}
         ],
         "insights_html": dt_insights_html
        }
    ]
     # --- Tab Rendering Loop ---
    for i, tab_config in enumerate(tab_configs): # Loop through defined tabs after Overview
        with tabs[i+1]: 
            st.header(tab_config["name"], divider="blue") 
            if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
                sim_data = st.session_state.simulation_results
                sim_cfg_tab = sim_data.get('config_params', {})
                minutes_per_interval_tab = sim_cfg_tab.get('MINUTES_PER_INTERVAL', 2)
                
                st.markdown("##### Select Time Range for Plots:") 
                # current_max_minutes_for_inputs is already correctly set based on loaded/run sim
                start_time_min, end_time_min = time_range_input_section(
                    tab_config["key_prefix"], current_max_minutes_for_inputs, interval_duration_min=minutes_per_interval_tab
                )
                start_idx, end_idx = start_time_min // minutes_per_interval_tab, (end_time_min // minutes_per_interval_tab) + 1
                
                # logger.debug(f"Tab '{tab_config['name']}': Time range {start_time_min}-{end_time_min} min. Indices {start_idx}-{end_idx-1}. Max mins: {current_max_minutes_for_inputs}", extra={'user_action': f'Tab {tab_config["name"]} Time Range'})
                
                # Filter global disruption_steps_for_plots for this tab's specific time window
                filt_disrupt_steps_for_tab_plots = [s for s in disruption_steps_for_plots if start_idx <= s < end_idx]

                if tab_config.get("metrics_display"): 
                    downtime_events_list_all = safe_get(sim_data, 'downtime_minutes', []) # List of event dicts
                    # Filter events by step for the current time range
                    downtime_events_in_range = [
                        event for event in downtime_events_list_all 
                        if isinstance(event, dict) and start_idx <= event.get('step', -1) < end_idx
                    ]
                    downtime_durations_in_range = [event.get('duration',0.0) for event in downtime_events_in_range]
                    
                    if downtime_events_in_range: 
                        total_downtime_period = sum(downtime_durations_in_range)
                        num_incidents = len([d for d in downtime_durations_in_range if d > 0]) # Count actual incidents
                        avg_duration_per_incident = total_downtime_period / num_incidents if num_incidents > 0 else 0.0
                        
                        dt_cols_metrics = st.columns(3)
                        dt_cols_metrics[0].metric("Total Downtime in Period", f"{total_downtime_period:.1f} min")
                        dt_cols_metrics[1].metric("Number of Incidents", f"{num_incidents}")
                        dt_cols_metrics[2].metric("Avg. Duration / Incident", f"{avg_duration_per_incident:.1f} min")

                plot_col_container = st.container() 
                num_plots_in_row = 0

                for plot_idx, plot_info in enumerate(tab_config["plots"]):
                    if plot_info.get("is_subheader"):
                        st.subheader(plot_info["title"]) 
                        if plot_info.get("is_spatial"):
                            # Spatial plots need team_positions_df and facility/work_area config
                            facility_config_for_spatial = DEFAULT_CONFIG.copy()
                            facility_config_for_spatial.update(sim_cfg_tab.get('WORK_AREAS_EFFECTIVE', DEFAULT_CONFIG.get('WORK_AREAS', {})))
                            
                            with st.container(border=True):
                                team_pos_df_all = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
                                zones_dist = ["All"] + list(facility_config_for_spatial.get('WORK_AREAS', {}).keys())
                                # Ensure unique key for selectbox
                                zone_sel_key = f"{tab_config['key_prefix']}_zone_sel_spatial_dist"
                                if zone_sel_key not in st.session_state: st.session_state[zone_sel_key] = "All"
                                zone_sel_dist = st.selectbox("Filter by Zone:", zones_dist, key=zone_sel_key)
                                
                                filt_team_pos_df_spatial_time = team_pos_df_all
                                if not filt_team_pos_df_spatial_time.empty: 
                                    filt_team_pos_df_spatial_time = filt_team_pos_df_spatial_time[(filt_team_pos_df_spatial_time['step'] >= start_idx) & (filt_team_pos_df_spatial_time['step'] < end_idx)]
                                
                                filt_team_pos_df_spatial_final = filt_team_pos_df_spatial_time
                                if zone_sel_dist != "All" and not filt_team_pos_df_spatial_final.empty : 
                                    filt_team_pos_df_spatial_final = filt_team_pos_df_spatial_final[filt_team_pos_df_spatial_final['zone'] == zone_sel_dist]

                                show_ee_key = f'{tab_config["key_prefix"]}_show_ee_spatial_cb'
                                if show_ee_key not in st.session_state: st.session_state[show_ee_key] = True
                                show_ee_exp = st.checkbox("Show E/E Points", key=show_ee_key) 
                                
                                show_pl_key = f'{tab_config["key_prefix"]}_show_pl_spatial_cb'
                                if show_pl_key not in st.session_state: st.session_state[show_pl_key] = True
                                show_pl_exp = st.checkbox("Show Area Outlines", key=show_pl_key)
                                
                                spatial_plot_cols = st.columns(2)
                                with spatial_plot_cols[0]: # Worker Distribution (Snapshot)
                                    min_snap_step, max_snap_step = start_idx, max(start_idx, end_idx -1) 
                                    snap_slider_key = f"{tab_config['key_prefix']}_snap_step_slider"
                                    if snap_slider_key not in st.session_state or not (min_snap_step <= st.session_state[snap_slider_key] <= max_snap_step):
                                        st.session_state[snap_slider_key] = min_snap_step if min_snap_step <= max_snap_step else max_snap_step
                                    
                                    snap_step_val = st.slider("Snapshot Time Step (Worker Positions):", min_snap_step, max_snap_step, key=snap_slider_key, step=1, disabled=(max_snap_step < min_snap_step))
                                    
                                    if not team_pos_df_all.empty and max_snap_step >= min_snap_step:
                                        try: 
                                            st.plotly_chart(plot_worker_distribution(
                                                team_pos_df_all, DEFAULT_CONFIG['FACILITY_SIZE'], facility_config_for_spatial, 
                                                sb_use_3d_val, snap_step_val, show_ee_exp, show_pl_exp, 
                                                current_high_contrast_setting, title_text=None # Pass title if function supports it
                                            ), use_container_width=True, config=plot_config_interactive)
                                        except Exception as e: 
                                            logger.error(f"Spatial Dist Plot Error: {e}", exc_info=True, extra={'user_action': 'Plot Error'})
                                            st.error(f"⚠️ Error plotting Worker Positions: {str(e)}.")
                                    else: st.caption("No data for positions snapshot or invalid time range.")
                                with spatial_plot_cols[1]: # Worker Density Heatmap (Aggregated over selected time)
                                    if not filt_team_pos_df_spatial_final.empty: # Use time and zone filtered data
                                        try: 
                                            st.plotly_chart(plot_worker_density_heatmap(
                                                filt_team_pos_df_spatial_final, DEFAULT_CONFIG['FACILITY_SIZE'], facility_config_for_spatial, 
                                                show_ee_exp, show_pl_exp, current_high_contrast_setting, title_text=None
                                            ), use_container_width=True, config=plot_config_interactive)
                                        except Exception as e: 
                                            logger.error(f"Spatial Heatmap Plot Error: {e}", exc_info=True, extra={'user_action': 'Plot Error'})
                                            st.error(f"⚠️ Error plotting Density Heatmap: {str(e)}.")
                                    else: st.caption("No data for density heatmap in this time range/zone.")
                        num_plots_in_row = 0 # Reset for next row of plots after subheader/spatial section
                        continue

                    if num_plots_in_row == 0: # Start a new row of 2 plots
                       plot_columns = plot_col_container.columns(2)
                    
                    current_plot_col = plot_columns[num_plots_in_row % 2]
                    with current_plot_col:
                        with st.container(border=True): # Each plot in its own bordered container
                            try:
                                data_to_plot_final = None
                                plot_data_raw = safe_get(sim_data, plot_info["data_path"], [])
                                kwargs_for_plot = {"high_contrast": current_high_contrast_setting, "title_text": plot_info["title"]}

                                if "extra_args_paths" in plot_info:
                                    for arg_name, arg_path in plot_info["extra_args_paths"].items():
                                        extra_data_raw = safe_get(sim_data, arg_path, [])
                                        # For triggers, pass the whole structure; plot function will filter by step
                                        if plot_info["plot_func"] == plot_worker_wellbeing and arg_name == "triggers":
                                            kwargs_for_plot[arg_name] = extra_data_raw
                                            kwargs_for_plot["start_step"] = start_idx # Pass range for filtering
                                            kwargs_for_plot["end_step"] = end_idx
                                        elif isinstance(extra_data_raw, list) and start_idx < len(extra_data_raw):
                                            kwargs_for_plot[arg_name] = extra_data_raw[start_idx:min(end_idx, len(extra_data_raw))]
                                        elif isinstance(extra_data_raw, list): # extra_data_raw is list but empty or start_idx too large
                                            kwargs_for_plot[arg_name] = []
                                        else: # Not a list, pass as is
                                            kwargs_for_plot[arg_name] = extra_data_raw
                                if "extra_args_fixed" in plot_info: 
                                    kwargs_for_plot.update(plot_info["extra_args_fixed"])
                                
                                # Pass relevant disruption points to the plot function if it expects them
                                func_params = plot_info["plot_func"].__code__.co_varnames
                                if "disruption_points" in func_params:
                                    # For most plots, disruption points are absolute step numbers within the time window
                                    kwargs_for_plot["disruption_points"] = filt_disrupt_steps_for_tab_plots

                                # --- Data preparation based on plot type ---
                                if plot_info.get("is_oee"):
                                    eff_df_full = safe_get(sim_data, 'efficiency_metrics_df', pd.DataFrame())
                                    if not eff_df_full.empty:
                                        oee_ms_key = f"{tab_config['key_prefix']}_oee_metrics_ms"
                                        if oee_ms_key not in st.session_state: st.session_state[oee_ms_key] = ['uptime', 'throughput', 'quality', 'oee']
                                        sel_metrics = st.multiselect("Select OEE Metrics:", ['uptime', 'throughput', 'quality', 'oee'], key=oee_ms_key)
                                        
                                        filt_eff_df = pd.DataFrame() # Initialize
                                        if isinstance(eff_df_full.index, pd.RangeIndex) and start_idx < len(eff_df_full):
                                            filt_eff_df = eff_df_full.iloc[start_idx:min(end_idx, len(eff_df_full))]
                                        elif not isinstance(eff_df_full.index, pd.RangeIndex): # If index is not simple RangeIndex
                                            if 'step' in eff_df_full.columns: 
                                                filt_eff_df = eff_df_full[(eff_df_full['step'] >= start_idx) & (eff_df_full['step'] < end_idx)]
                                            elif eff_df_full.index.name == 'step' or (isinstance(eff_df_full.index, pd.Index) and eff_df_full.index.is_numeric()):
                                                filt_eff_df = eff_df_full[(eff_df_full.index >= start_idx) & (eff_df_full.index < end_idx)]
                                        
                                        if not filt_eff_df.empty:
                                            st.plotly_chart(plot_operational_efficiency(filt_eff_df, sel_metrics, **kwargs_for_plot), use_container_width=True, config=plot_config_interactive)
                                        else: st.caption("No OEE data for this time range.")
                                    else: st.caption("No OEE data available.")
                                    num_plots_in_row +=1 # Increment here as OEE is a full plot item
                                    continue # Skip to next plot_info

                                elif plot_info.get("is_event_based_aggregation"): # e.g., plot_downtime_trend
                                    all_events = plot_data_raw if isinstance(plot_data_raw, list) else []
                                    num_steps_in_range = end_idx - start_idx
                                    aggregated_data_in_range = [0.0] * num_steps_in_range
                                    for event in all_events:
                                        if isinstance(event, dict) and 'step' in event and 'duration' in event:
                                            step = event['step']
                                            if start_idx <= step < end_idx:
                                                relative_step = step - start_idx
                                                if 0 <= relative_step < num_steps_in_range: # Boundary check
                                                    aggregated_data_in_range[relative_step] += float(event['duration'])
                                    data_to_plot_final = aggregated_data_in_range
                                    # Adjust disruption points to be relative to this new aggregated data's range
                                    if "disruption_points" in kwargs_for_plot:
                                        kwargs_for_plot["disruption_points"] = [s - start_idx for s in filt_disrupt_steps_for_tab_plots]

                                elif plot_info.get("is_event_based_filtering"): # e.g., plot_downtime_causes_pie
                                    all_events = plot_data_raw if isinstance(plot_data_raw, list) else []
                                    filtered_events_for_plot = []
                                    for event in all_events:
                                        if isinstance(event, dict) and 'step' in event: # Check for 'step'
                                            if start_idx <= event['step'] < end_idx:
                                                filtered_events_for_plot.append(event)
                                    data_to_plot_final = filtered_events_for_plot
                                    # This type of plot might not use disruption_points, or use them differently
                                    if "disruption_points" in kwargs_for_plot: 
                                        del kwargs_for_plot["disruption_points"] # Or handle as needed

                                else: # Standard time-series data (list or DataFrame)
                                    if isinstance(plot_data_raw, list):
                                        if start_idx < len(plot_data_raw): 
                                            data_to_plot_final = plot_data_raw[start_idx:min(end_idx, len(plot_data_raw))]
                                        else: data_to_plot_final = []
                                    elif isinstance(plot_data_raw, pd.DataFrame) and not plot_data_raw.empty:
                                        # Similar DataFrame slicing as OEE
                                        if isinstance(plot_data_raw.index, pd.RangeIndex) and start_idx < len(plot_data_raw):
                                            data_to_plot_final = plot_data_raw.iloc[start_idx:min(end_idx, len(plot_data_raw))]
                                        # Add more specific DataFrame filtering if needed, similar to OEE
                                        else: data_to_plot_final = pd.DataFrame() # Fallback
                                    else:
                                        data_to_plot_final = [] # Default for unknown or empty raw data

                                # --- Final check and plot rendering ---
                                final_check_has_data = False
                                if isinstance(data_to_plot_final, list) and data_to_plot_final: final_check_has_data = True
                                elif isinstance(data_to_plot_final, pd.DataFrame) and not data_to_plot_final.empty: final_check_has_data = True
                                # For event-based filtering, an empty list of events is valid "no data"
                                elif plot_info.get("is_event_based_filtering") and isinstance(data_to_plot_final, list):
                                    final_check_has_data = True # Plot function will handle empty list of events

                                if not final_check_has_data:
                                    st.caption(f"No data for '{plot_info['title']}' in this time range.")
                                else:
                                    st.plotly_chart(plot_info["plot_func"](data_to_plot_final, **kwargs_for_plot), use_container_width=True, config=plot_config_interactive)
                            
                            except Exception as e:
                                logger.error(f"Tab '{tab_config['name']}', Plot '{plot_info['title']}' Error: {e}", exc_info=True, extra={'user_action': f'Plot Error - {plot_info["title"]}'})
                                st.error(f"⚠️ Error plotting {plot_info['title']}: {str(e)}")
                    num_plots_in_row += 1
                
                # --- Insights Section for the Tab ---
                st.markdown("<hr style='margin-top:2rem;'><h3 style='text-align:center; margin-top:1rem;'>🏛️ Leadership Actionable Insights</h3>", unsafe_allow_html=True)
                if tab_config.get("dynamic_insights_func") == "render_wellbeing_alerts":
                    with st.container(border=True): # Bordered container for these alerts
                        st.markdown("<h6>Well-Being Alerts (within selected time range):</h6>", unsafe_allow_html=True)
                        ww_trigs_disp_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {}) # This is a dict like {'threshold': [steps], 'trend': [steps], ...}
                        
                        insights_count_wb = 0 
                        for alert_type, alert_steps_raw in ww_trigs_disp_raw.items():
                            if alert_type == 'work_area' and isinstance(alert_steps_raw, dict): # Nested dict for work_area
                                wa_alert_found_in_range = False
                                wa_details_html = ""
                                for zone, zone_steps_raw in alert_steps_raw.items():
                                    zone_steps_in_range = [s for s in (zone_steps_raw if isinstance(zone_steps_raw, list) else []) if start_idx <= s < end_idx]
                                    if zone_steps_in_range:
                                        wa_alert_found_in_range = True
                                        wa_details_html += f"&nbsp;&nbsp;- {zone}: {len(zone_steps_in_range)} alerts at steps {zone_steps_in_range}<br>"
                                if wa_alert_found_in_range:
                                    st.markdown(f"<div class='alert-warning insight-text'><strong>Work Area Specific Alerts:</strong><br>{wa_details_html}</div>", unsafe_allow_html=True)
                                    insights_count_wb +=1
                            elif isinstance(alert_steps_raw, list): # Standard list of steps for other alert types
                                alert_steps_in_range = [s for s in alert_steps_raw if start_idx <= s < end_idx]
                                if alert_steps_in_range:
                                    alert_class = "alert-critical" if alert_type == "threshold" else "alert-warning" if alert_type == "trend" else "alert-info"
                                    alert_title_text = alert_type.replace("_", " ").title()
                                    st.markdown(f"<div class='{alert_class} insight-text'><strong>{alert_title_text} Alerts ({len(alert_steps_in_range)} times):</strong> Steps {alert_steps_in_range}.</div>", unsafe_allow_html=True)
                                    insights_count_wb += 1
                        
                        if insights_count_wb == 0: 
                            st.markdown(f"<p class='insight-text' style='color: {COLOR_POSITIVE_GREEN};'>✅ No specific well-being alerts triggered in the selected period.</p>", unsafe_allow_html=True)
                    
                if tab_config.get("insights_html"): # Static HTML insights for the tab
                     st.markdown(tab_config["insights_html"], unsafe_allow_html=True) 
            else: # No simulation results
                st.info(f"ℹ️ Run a simulation or load data to view {tab_config['name']}.", icon="📊")
    
    # --- Glossary Tab ---
    with tabs[4]: # Index for Glossary tab
        st.header("📖 Glossary of Terms", divider="blue") 
        st.markdown("""
            <div style="font-size: 0.95rem; line-height: 1.7;">
            <p>This glossary defines key metrics used throughout the dashboard to help you understand the operational insights provided. For a combined view with general help, click the "ℹ️ Help & Glossary" button in the sidebar.</p>
            <details><summary><strong>Task Compliance Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of tasks completed correctly and within the allocated time. It measures adherence to operational protocols and standards. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Collaboration Proximity Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of workers observed within a defined proximity (e.g., 5 meters) of their colleagues. This index suggests opportunities for teamwork, communication, and knowledge sharing. <em>Range: 0-100%. Optimal levels vary.</em></p></details>
            <details><summary><strong>Operational Recovery Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A measure of the system's ability to return to and maintain target output levels after experiencing disruptions. It reflects operational resilience. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Worker Well-Being Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A composite score derived from simulated factors such as fatigue, stress levels, and job satisfaction. It provides an indicator of overall worker health and morale. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Psychological Safety Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">An estimate of the perceived comfort level among workers to report issues, voice concerns, or suggest improvements without fear of negative consequences. It indicates a supportive and open work environment. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Team Cohesion Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A measure of the strength of bonds and sense of belonging within a team. Higher cohesion often correlates with better communication and mutual support. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Perceived Workload Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">An indicator of how demanding workers perceive their tasks and overall workload on a scale (e.g., 0-10). Persistently high scores can lead to stress and burnout. <em>Lower is generally better.</em></p></details>
            <details><summary><strong>Uptime</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of scheduled operational time that equipment or a system is available and functioning correctly. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Throughput</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The rate at which a system processes work or produces output, often expressed as a percentage of its maximum potential or target rate. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Quality Rate</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of products or outputs that meet predefined quality standards, free of defects. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>OEE (Overall Equipment Effectiveness)</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A comprehensive metric calculated as (Uptime × Throughput × Quality Rate). It provides a holistic view of operational performance and efficiency. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Productivity Loss</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of potential output or operational time lost due to inefficiencies, disruptions, downtime, or substandard performance. <em>Range: 0-100%. Lower is better.</em></p></details>
            <details><summary><strong>Downtime (per interval)</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The simulation output for `downtime_minutes` is a list of event dictionaries, each including `step`, `duration`, and `type`. For trend plots and summary tables, these are aggregated to show total downtime duration per simulation interval. <em>Lower is better.</em></p></details>
            <details><summary><strong>Task Completion Rate</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of assigned tasks that are successfully completed within a given time interval. Measures task throughput and efficiency over time. <em>Range: 0-100%. Higher is better.</em></p></details>
            <hr>
            <p><strong>Simulation Step / Interval:</strong> The simulation progresses in discrete time steps or intervals. The duration of each interval (e.g., 2 minutes) is defined in `DEFAULT_CONFIG['MINUTES_PER_INTERVAL']`. Many metrics are reported per interval.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
