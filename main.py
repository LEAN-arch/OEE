# main.py
import logging
import streamlit as st
import pandas as pd
import numpy as np
import math
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
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
                        filename='dashboard.log',
                        filemode='a')
logger.info("Main.py: Parsed imports and logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.2.4\nInsights for operational excellence & psychosocial well-being."})

# --- Accessible Color Definitions ---
COLOR_CRITICAL_RED = "#E53E3E"
COLOR_WARNING_AMBER = "#F59E0B"
COLOR_POSITIVE_GREEN = "#10B981"
COLOR_INFO_BLUE = "#3B82F6"
COLOR_ACCENT_INDIGO = "#4F46E5"

# --- UTILITY FUNCTIONS ---
def safe_get(data_dict, path_str, default_val=None):
    current = data_dict
    is_list_like_path = False
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
        if path_str: 
            logger.debug(f"safe_get: data_dict is not a dictionary for path '{path_str}'. Type: {type(data_dict)}. Returning default '{default_return}'.", extra={'user_action': 'Safe Get Internal Warning'})
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
                current = None 
                break
        
        if current is None:
            is_list_like_final_key = keys and keys[-1] in ['data', 'scores', 'triggers', 'minutes', 'events_list']
            return [] if default_val is None and is_list_like_final_key else default_val
        return current
    except (ValueError, IndexError, TypeError) as e:
        logger.debug(f"safe_get failed for path '{path_str}': {e}. Returning default '{default_return}'.", extra={'user_action': 'Safe Get Internal Error'})
        return default_return

def safe_stat(data_list, stat_func, default_val=0.0):
    if not isinstance(data_list, (list, np.ndarray, pd.Series)):
        return default_val
    
    if isinstance(data_list, pd.Series):
        valid_data_series = pd.to_numeric(data_list, errors='coerce').dropna()
        valid_data = valid_data_series.tolist()
    else: 
        valid_data = []
        for x in data_list:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            try:
                valid_data.append(float(x))
            except (ValueError, TypeError): pass 

    if not valid_data: return default_val
    
    try:
        result = stat_func(np.array(valid_data)) 
        return default_val if isinstance(result, (float, np.floating)) and np.isnan(result) else result
    except Exception as e: 
        logger.warning(f"safe_stat: Error in {stat_func.__name__} (preview): {str(valid_data)[:50]}: {e}. Default: {default_val}", exc_info=False, extra={'user_action': 'Safe Stat Error'})
        return default_val

def get_actionable_insights(sim_data, current_config):
    # ... (get_actionable_insights function remains the same as previous correct version)
    insights = []
    if not sim_data or not isinstance(sim_data, dict): 
        logger.warning("get_actionable_insights: sim_data is None or not a dict.", extra={'user_action': 'Actionable Insights - Invalid Input'})
        return insights
    
    compliance_data = safe_get(sim_data, 'task_compliance.data', [])
    target_compliance = float(current_config.get('TARGET_COMPLIANCE', 75.0))
    compliance_avg = safe_stat(compliance_data, np.mean, default_val=0.0) 
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
    if wellbeing_scores and len(threshold_triggers) > max(2, len(wellbeing_scores) * 0.1): 
        insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers)} instances of well-being dropping below threshold. Investigate specific triggers."})

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
    target_workload = float(current_config.get('TARGET_PERCEIVED_WORKLOAD', 6.5)) 
    workload_avg = safe_stat(workload_scores, np.mean, default_val=target_workload / 2) 
    workload_very_high_thresh = float(current_config.get('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5))
    workload_high_thresh = float(current_config.get('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5))
    if workload_avg > workload_very_high_thresh:
        insights.append({"type": "critical", "title": "Very High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) is critically high. Immediate review of task distribution, staffing, and efficiencies required."})
    elif workload_avg > workload_high_thresh:
        insights.append({"type": "warning", "title": "High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) exceeds high threshold. Monitor closely and identify bottlenecks."})
    elif workload_avg > target_workload:
        insights.append({"type": "info", "title": "Elevated Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) is above target ({target_workload:.1f}/10). Consider proactive adjustments."})
    
    team_pos_df = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
    if not team_pos_df.empty and 'zone' in team_pos_df.columns and 'worker_id' in team_pos_df.columns and 'step' in team_pos_df.columns:
        work_areas_config_insight = current_config.get('WORK_AREAS', {})
        if isinstance(work_areas_config_insight, dict): # Ensure it's a dict
            for zone_name, zone_details in work_areas_config_insight.items():
                if not isinstance(zone_details, dict): continue
                workers_in_zone_series = team_pos_df[team_pos_df['zone'] == zone_name].groupby('step')['worker_id'].nunique()
                if not workers_in_zone_series.empty:
                    workers_in_zone_avg = workers_in_zone_series.mean()
                    intended_workers = zone_details.get('workers', 0)
                    coords = zone_details.get('coords'); area_m2 = 1.0 
                    if coords and isinstance(coords, list) and len(coords) == 2 and \
                       all(isinstance(p, tuple) and len(p)==2 for p in coords):
                        (x0,y0), (x1,y1) = coords; area_m2 = abs(x1-x0) * abs(y1-y0)
                    if area_m2 == 0: area_m2 = 1.0 
                    
                    avg_density = workers_in_zone_avg / area_m2 if area_m2 > 0 else 0 
                    intended_density = (intended_workers / area_m2) if area_m2 > 0 and intended_workers > 0 else 0

                    if intended_density > 0 and avg_density > intended_density * 1.8: 
                         insights.append({"type": "warning", "title": f"Potential Overcrowding in '{zone_name}'", "text": f"Average worker density ({avg_density:.2f} w/m²) significantly higher than based on assigned workers ({intended_density:.2f} w/m²). Review layout or worker paths."})
                    elif intended_workers > 0 and workers_in_zone_avg < intended_workers * 0.4: 
                         insights.append({"type": "info", "title": f"Potential Underutilization of '{zone_name}'", "text": f"Average workers observed ({workers_in_zone_avg:.1f}) is less than 40% of assigned ({intended_workers}). Check task allocation or if workers are congregating elsewhere."})

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


def aggregate_downtime_by_step(downtime_events_list, num_total_steps, minutes_per_interval):
    # ... (aggregate_downtime_by_step function remains the same)
    downtime_per_step_agg = [0.0] * num_total_steps
    if not isinstance(downtime_events_list, list):
        logger.warning("aggregate_downtime_by_step: downtime_events_list is not a list.")
        return downtime_per_step_agg

    for event in downtime_events_list:
        if not isinstance(event, dict):
            continue
        
        step = event.get('step') 
        duration = event.get('duration', 0.0)

        if not isinstance(step, int): 
            start_time_min_val = event.get('Start Time (min)') 
            if isinstance(start_time_min_val, (int, float)) and minutes_per_interval > 0:
                step = int(start_time_min_val // minutes_per_interval)
            else:
                continue
        
        if 0 <= step < num_total_steps and isinstance(duration, (int, float)) and duration > 0:
            downtime_per_step_agg[step] += float(duration)
    return downtime_per_step_agg

def _prepare_timeseries_for_export(raw_data, num_total_steps, default_val=np.nan):
    # ... (_prepare_timeseries_for_export function remains the same)
    if not isinstance(raw_data, list):
        logger.debug(f"_prepare_timeseries_for_export: raw_data is not a list (type: {type(raw_data)}), returning default list.")
        return [default_val] * num_total_steps
    
    if len(raw_data) == num_total_steps:
        return raw_data
    elif len(raw_data) < num_total_steps:
        return raw_data + [default_val] * (num_total_steps - len(raw_data))
    else: 
        return raw_data[:num_total_steps]


def _slice_dataframe_by_step_indices(df, start_idx, end_idx):
    # ... (_slice_dataframe_by_step_indices function remains the same)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if isinstance(df.index, pd.RangeIndex):
        safe_start_idx = max(0, start_idx)
        safe_end_idx = min(len(df), end_idx)
        return df.iloc[safe_start_idx:safe_end_idx] if safe_start_idx < safe_end_idx else pd.DataFrame()

    if 'step' in df.columns:
        return df[(df['step'] >= start_idx) & (df['step'] < end_idx)]
    
    if df.index.name == 'step' or (isinstance(df.index, pd.Index) and df.index.is_numeric()):
        return df[(df.index >= start_idx) & (df.index < end_idx)]
        
    logger.warning(f"_slice_dataframe_by_step_indices: Could not slice DF. Cols: {df.columns}, Idx: {df.index.name}. Empty DF returned.")
    return pd.DataFrame()

# --- CSS ---
st.markdown(f""" <style> {/* ... CSS remains the same ... */} </style> """, unsafe_allow_html=True) # Shortened for brevity

# --- render_settings_sidebar ---
def render_settings_sidebar():
    # ... (render_settings_sidebar function remains the same as previous correct version)
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem; color: #A0A0A0;'>Workplace Optimizer</h3>", unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")
        with st.expander("🧪 Simulation Parameters", expanded=True):
            st.number_input( 
                "Team Size", min_value=1, max_value=200,
                key="sb_team_size_num", 
                step=1,
                help="Adjust the number of workers in the simulated shift."
            )
            st.number_input(
                "Shift Duration (min)", min_value=60, max_value=7200, 
                key="sb_shift_duration_num",
                step=10,
                help="Set the total length of the simulated work shift in minutes."
            )
            
            current_shift_duration_for_events = st.session_state.sb_shift_duration_num
            minutes_per_interval_local = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)


            st.markdown("---")
            st.markdown("<h5>🗓️ Schedule Shift Events</h5>", unsafe_allow_html=True) 
            st.caption("Define disruptions, breaks, etc. Times are from shift start.")
            
            event_types = ["Major Disruption", "Minor Disruption", "Scheduled Break", "Short Pause", "Team Meeting", "Maintenance", "Custom Event"]
            
            with st.container():
                st.session_state.form_event_type = st.selectbox("Event Type", event_types, 
                                                                index=event_types.index(st.session_state.form_event_type) if st.session_state.form_event_type in event_types else 0, 
                                                                key="widget_sb_new_event_type_RENDER") 
                
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    st.session_state.form_event_start = st.number_input("Start (min)", min_value=0, 
                                                                         max_value=max(0, current_shift_duration_for_events - minutes_per_interval_local), 
                                                                         step=minutes_per_interval_local, 
                                                                         key="widget_sb_new_event_start_RENDER",  
                                                                         help=f"Minutes from shift start (0 to {max(0,current_shift_duration_for_events-minutes_per_interval_local)}).")
                with col_time2:
                    st.session_state.form_event_duration = st.number_input("Duration (min)", min_value=minutes_per_interval_local, 
                                                                            max_value=current_shift_duration_for_events, 
                                                                            step=minutes_per_interval_local, 
                                                                            key="widget_sb_new_event_duration_RENDER")

            if st.button("➕ Add Event", key="sb_add_event_btn", use_container_width=True): 
                add_event_type_val = st.session_state.form_event_type
                add_event_start_val = st.session_state.form_event_start
                add_event_duration_val = st.session_state.form_event_duration

                if add_event_start_val + add_event_duration_val > current_shift_duration_for_events:
                    st.warning(f"Event end time ({add_event_start_val + add_event_duration_val} min) exceeds shift duration ({current_shift_duration_for_events} min).")
                elif add_event_start_val < 0 : st.warning("Event start time cannot be negative.")
                elif add_event_duration_val < minutes_per_interval_local: st.warning(f"Event duration must be at least {minutes_per_interval_local} minute(s).")
                else:
                    st.session_state.sb_scheduled_events_list.append({
                        "Event Type": add_event_type_val,
                        "Start Time (min)": add_event_start_val, 
                        "Duration (min)": add_event_duration_val,
                        "step": int(add_event_start_val // minutes_per_interval_local) 
                    })
                    st.session_state.sb_scheduled_events_list.sort(key=lambda x: x.get("Start Time (min)", 0))
                    st.session_state.form_event_start = 0 
                    st.session_state.form_event_duration = max(minutes_per_interval_local, 10) 
                    st.rerun()

            st.markdown("<h6>Current Scheduled Events:</h6>", unsafe_allow_html=True) 
            if not st.session_state.sb_scheduled_events_list:
                st.caption("No events scheduled yet.")
            else:
                with st.container(height=200): 
                    for i, event in enumerate(st.session_state.sb_scheduled_events_list):
                        event_col1, event_col2 = st.columns([0.85, 0.15])
                        with event_col1:
                            st.markdown(f"<div class='event-item'><span class='event-text'><b>{event.get('Event Type','N/A')}</b> at {event.get('Start Time (min)','N/A')} min (lasts {event.get('Duration (min)','N/A')} min)</span></div>", unsafe_allow_html=True)
                        with event_col2:
                            if st.button("✖", key=f"remove_event_{i}", help="Remove this event", type="secondary", use_container_width=True):
                                st.session_state.sb_scheduled_events_list.pop(i)
                                st.rerun()
            
            if st.session_state.sb_scheduled_events_list: 
                if st.button("Clear All Events", key="sb_clear_events_btn", type="secondary", use_container_width=True): 
                    st.session_state.sb_scheduled_events_list = []
                    st.rerun()
            
            st.markdown("---") 
            team_initiative_opts = ["Standard Operations", "More frequent breaks", "Team recognition", "Increased Autonomy"]
            st.selectbox("Operational Initiative", team_initiative_opts, 
                         key="sb_team_initiative_selectbox", 
                         help="Apply an operational strategy to observe its impact on metrics.")
            
            run_simulation_button = st.button("🚀 Run Simulation", key="sb_run_simulation_button", type="primary", use_container_width=True)
        
        with st.expander("🎨 Visualization Options"):
            st.checkbox("High Contrast Plots", key="sb_high_contrast_checkbox", help="Applies a high-contrast color theme to all charts for better accessibility.")
            st.checkbox("Enable 3D Worker View", key="sb_use_3d_distribution_checkbox", help="Renders worker positions in a 3D scatter plot.")
            st.checkbox("Show Debug Info", key="sb_debug_mode_checkbox", help="Display additional debug information in the sidebar.")
        
        with st.expander("💾 Data Management & Export"):
            load_data_button = st.button("🔄 Load Previous Simulation", key="sb_load_data_button", use_container_width=True)
            
            can_gen_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("📄 Download Report (.tex)", key="sb_pdf_button", disabled=not can_gen_report, use_container_width=True, help="Generates a LaTeX (.tex) file summarizing the simulation."):
                if can_gen_report:
                    try:
                        sim_res = st.session_state.simulation_results
                        sim_cfg_params_pdf = sim_res.get('config_params', {})
                        num_total_steps_pdf = sim_cfg_params_pdf.get('SHIFT_DURATION_INTERVALS', 0)
                        minutes_per_interval_pdf = sim_cfg_params_pdf.get('MINUTES_PER_INTERVAL', DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2))

                        if num_total_steps_pdf == 0: 
                            st.warning("⚠️ No simulation data (0 steps) for report.")
                            raise SystemExit 
                        
                        pdf_data = {}
                        metrics_to_export_pdf = ['operational_recovery', 'psychological_safety', 'productivity_loss', 'task_completion_rate']
                        for k in metrics_to_export_pdf:
                            pdf_data[k] = _prepare_timeseries_for_export(sim_res.get(k, []), num_total_steps_pdf)
                        
                        pdf_data['task_compliance'] = _prepare_timeseries_for_export(safe_get(sim_res, 'task_compliance.data', []), num_total_steps_pdf)
                        pdf_data['collaboration_proximity'] = _prepare_timeseries_for_export(safe_get(sim_res, 'collaboration_proximity.data', []), num_total_steps_pdf)
                        pdf_data['worker_wellbeing'] = _prepare_timeseries_for_export(safe_get(sim_res, 'worker_wellbeing.scores', []), num_total_steps_pdf)
                        
                        downtime_events_for_pdf = sim_res.get('downtime_minutes', [])
                        pdf_data['downtime_minutes_per_step'] = aggregate_downtime_by_step(downtime_events_for_pdf, num_total_steps_pdf, minutes_per_interval_pdf)
                        
                        pdf_data['step'] = list(range(num_total_steps_pdf))
                        pdf_data['time_minutes'] = [i * minutes_per_interval_pdf for i in range(num_total_steps_pdf)]
                        
                        df_for_pdf = pd.DataFrame(pdf_data)
                        generate_pdf_report(df_for_pdf) 
                        st.success("✅ LaTeX report (.tex) 'workplace_report.tex' generated.")
                    except SystemExit: pass 
                    except Exception as e: 
                        logger.error(f"PDF Gen Error: {e}", exc_info=True, extra={'user_action': 'PDF Generation Error'})
                        st.error(f"❌ PDF Gen Error: {e}")
            
            if can_gen_report: 
                sim_res_exp = st.session_state.simulation_results
                sim_cfg_params_csv = sim_res_exp.get('config_params', {})
                num_total_steps_csv = sim_cfg_params_csv.get('SHIFT_DURATION_INTERVALS', 0)
                minutes_per_interval_csv = sim_cfg_params_csv.get('MINUTES_PER_INTERVAL', DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2))


                if num_total_steps_csv > 0:
                    csv_data = {}
                    metrics_to_export_csv = ['operational_recovery', 'psychological_safety', 'productivity_loss', 'task_completion_rate']
                    for k in metrics_to_export_csv:
                         csv_data[k] = _prepare_timeseries_for_export(sim_res_exp.get(k, []), num_total_steps_csv)

                    csv_data['task_compliance'] = _prepare_timeseries_for_export(safe_get(sim_res_exp, 'task_compliance.data', []), num_total_steps_csv)
                    csv_data['collaboration_proximity'] = _prepare_timeseries_for_export(safe_get(sim_res_exp, 'collaboration_proximity.data', []), num_total_steps_csv)
                    
                    ww_data_csv = sim_res_exp.get('worker_wellbeing', {})
                    csv_data['worker_wellbeing_index'] = _prepare_timeseries_for_export(ww_data_csv.get('scores', []), num_total_steps_csv)
                    csv_data['team_cohesion'] = _prepare_timeseries_for_export(ww_data_csv.get('team_cohesion_scores', []), num_total_steps_csv)
                    csv_data['perceived_workload'] = _prepare_timeseries_for_export(ww_data_csv.get('perceived_workload_scores', []), num_total_steps_csv)

                    downtime_events_for_csv = sim_res_exp.get('downtime_minutes', [])
                    csv_data['downtime_minutes_per_step'] = aggregate_downtime_by_step(downtime_events_for_csv, num_total_steps_csv, minutes_per_interval_csv)
                    
                    csv_data['step'] = list(range(num_total_steps_csv))
                    csv_data['time_minutes'] = [i * minutes_per_interval_csv for i in range(num_total_steps_csv)]
                    
                    df_for_csv = pd.DataFrame(csv_data)
                    st.download_button("📥 Download Data (CSV)", df_for_csv.to_csv(index=False).encode('utf-8'), "workplace_summary.csv", "text/csv", key="sb_csv_dl_button", use_container_width=True)
                else: 
                    st.caption("No detailed data to export (0 simulation steps).")
            elif not can_gen_report : 
                st.caption("Run simulation for export options.")
        
        if st.session_state.sb_debug_mode_checkbox:
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
            st.rerun() 
        if st.button("🚀 Quick Tour", key="sb_tour_button", use_container_width=True): 
            st.session_state.show_tour = not st.session_state.get('show_tour', False)
            st.rerun() 
            
    return run_simulation_button, load_data_button

# --- run_simulation_logic ---
@st.cache_data(ttl=3600, show_spinner="⚙️ Running simulation model...") 
def run_simulation_logic(team_size, shift_duration_minutes, scheduled_events_list_of_dicts, team_initiative_selected):
    # ... (run_simulation_logic function remains the same as previous correct version)
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    minutes_per_interval = config.get('MINUTES_PER_INTERVAL', 2) 
    if minutes_per_interval <= 0: 
        logger.error(f"MINUTES_PER_INTERVAL is {minutes_per_interval}, must be > 0. Using default 2.")
        minutes_per_interval = 2
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // minutes_per_interval
    
    processed_scheduled_events = []
    for event_dict in scheduled_events_list_of_dicts:
        new_event = event_dict.copy()
        if 'step' not in new_event and 'Start Time (min)' in new_event:
            new_event['step'] = int(new_event['Start Time (min)'] // minutes_per_interval)
        processed_scheduled_events.append(new_event)
    config['SCHEDULED_EVENTS'] = processed_scheduled_events
    
    logger.info(f"run_simulation_logic: SCHEDULED_EVENTS processed: {config['SCHEDULED_EVENTS']}", extra={'user_action': 'Process Scheduled Events'})

    if 'WORK_AREAS' in config and isinstance(config['WORK_AREAS'], dict) and config['WORK_AREAS']:
        total_workers_in_config_zones = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values() if isinstance(zone, dict))
        if total_workers_in_config_zones != team_size and team_size > 0:
            logger.info(f"Adjusting worker distribution for team size {team_size}. Configured sum was {total_workers_in_config_zones}.", extra={'user_action': 'Adjust Worker Distribution'})
            if total_workers_in_config_zones > 0: 
                ratio = team_size / total_workers_in_config_zones
                accumulated_workers = 0
                sorted_zone_keys = sorted([k for k, v in config['WORK_AREAS'].items() if isinstance(v, dict)]) 
                for zone_key_idx, zone_key in enumerate(sorted_zone_keys):
                    zone_data = config['WORK_AREAS'][zone_key] 
                    if zone_key_idx < len(sorted_zone_keys) - 1:
                        workers_prop = zone_data.get('workers', 0) * ratio
                        assigned_val = int(round(workers_prop)) 
                        zone_data['workers'] = assigned_val
                        accumulated_workers += assigned_val
                    else: 
                        remaining_workers_to_assign = team_size - accumulated_workers
                        zone_data['workers'] = remaining_workers_to_assign
                        if zone_data['workers'] < 0:
                            logger.warning(f"Negative workers ({zone_data['workers']}) for last zone {zone_key}. Setting to 0 and re-balancing.", extra={'user_action': 'Worker Distribution Warning'})
                            zone_data['workers'] = 0 
                            current_sum_after_neg_fix = sum(z.get('workers',0) for z_key_inner in sorted_zone_keys for z in [config['WORK_AREAS'][z_key_inner]] if isinstance(z, dict))
                            if current_sum_after_neg_fix != team_size and sorted_zone_keys: 
                                deficit_or_surplus = team_size - current_sum_after_neg_fix
                                config['WORK_AREAS'][sorted_zone_keys[0]]['workers'] = max(0, config['WORK_AREAS'][sorted_zone_keys[0]].get('workers',0) + deficit_or_surplus) 
                                logger.info(f"Re-balanced: Adjusted workers in zone {sorted_zone_keys[0]} by {deficit_or_surplus}.", extra={'user_action': 'Worker Re-balance'})
            else: 
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
    logger.info(f"Running simulation with: Team Size={team_size}, Duration={shift_duration_minutes}min ({config['SHIFT_DURATION_INTERVALS']} intervals of {minutes_per_interval}min), Scheduled Events: {len(config['SCHEDULED_EVENTS'])}, Initiative: {team_initiative_selected}", extra={'user_action': 'Run Simulation - Start'})
    
    sim_results_tuple = simulate_workplace_operations(
        num_team_members=team_size,
        num_steps=config['SHIFT_DURATION_INTERVALS'],
        scheduled_events=config['SCHEDULED_EVENTS'], 
        team_initiative=team_initiative_selected,
        config=config 
    )
    
    expected_keys = ['team_positions_df', 'task_compliance', 'collaboration_proximity', 
                     'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 
                     'worker_wellbeing', 'psychological_safety', 'feedback_impact', 
                     'downtime_minutes', 'task_completion_rate']
    
    if not isinstance(sim_results_tuple, tuple) or len(sim_results_tuple) != len(expected_keys):
        err_msg = f"Simulation returned unexpected data format. Expected tuple of length {len(expected_keys)}, got {type(sim_results_tuple)} of length {len(sim_results_tuple) if isinstance(sim_results_tuple, (tuple,list)) else 'N/A'}." 
        logger.critical(err_msg, extra={'user_action': 'Run Simulation - CRITICAL Data Format Error'})
        logger.critical(f"Received data (partial): {str(sim_results_tuple)[:500]}", extra={'user_action': 'Run Simulation - CRITICAL Data Content'})
        raise TypeError(err_msg)
        
    simulation_output_dict = dict(zip(expected_keys, sim_results_tuple))
    
    simulation_output_dict['config_params'] = {
        'TEAM_SIZE': team_size,
        'SHIFT_DURATION_MINUTES': shift_duration_minutes,
        'SHIFT_DURATION_INTERVALS': config['SHIFT_DURATION_INTERVALS'],
        'MINUTES_PER_INTERVAL': minutes_per_interval,
        'SCHEDULED_EVENTS': config['SCHEDULED_EVENTS'], 
        'TEAM_INITIATIVE': team_initiative_selected,
        'WORK_AREAS_EFFECTIVE': config.get('WORK_AREAS', {}).copy() 
    }
    
    disruption_event_steps_derived = []
    for event in config['SCHEDULED_EVENTS']: 
        if isinstance(event, dict) and "Disruption" in event.get("Event Type",""):
            event_step = event.get('step')
            if event_step is None: 
                 start_time = event.get("Start Time (min)")
                 if isinstance(start_time, (int, float)) and start_time >=0 and minutes_per_interval > 0:
                    event_step = int(start_time // minutes_per_interval)
            if isinstance(event_step, int):
                 disruption_event_steps_derived.append(event_step)

    simulation_output_dict['config_params']['DISRUPTION_EVENT_STEPS'] = sorted(list(set(disruption_event_steps_derived)))

    save_simulation_data(simulation_output_dict) 
    return simulation_output_dict

# --- time_range_input_section ---
def time_range_input_section(tab_key_prefix: str, max_minutes_for_range: int, st_col_obj = st, interval_duration_min: int = 2):
    # ... (time_range_input_section function remains the same as previous correct version)
    start_time_key = f"{tab_key_prefix}_start_time_min"
    end_time_key = f"{tab_key_prefix}_end_time_min"

    if interval_duration_min <=0: interval_duration_min = 2 

    if start_time_key not in st.session_state:
        st.session_state[start_time_key] = 0
    if end_time_key not in st.session_state:
        st.session_state[end_time_key] = max_minutes_for_range
    
    st.session_state[start_time_key] = max(0, min(st.session_state[start_time_key], max_minutes_for_range))
    st.session_state[end_time_key] = max(st.session_state[start_time_key], min(st.session_state[end_time_key], max_minutes_for_range))
    
    prev_start_time = st.session_state[start_time_key]
    prev_end_time = st.session_state[end_time_key]

    cols = st_col_obj.columns(2)
    
    new_start_time = cols[0].number_input(
        "Start Time (min)", 
        min_value=0, 
        max_value=max_minutes_for_range, 
        value=st.session_state[start_time_key],
        step=interval_duration_min, 
        key=f"widget_{start_time_key}", 
        help=f"Select the start of the time range (0 to {max_minutes_for_range})."
    )
    st.session_state[start_time_key] = new_start_time 
    
    end_time_min_widget_val = st.session_state[start_time_key]
    new_end_time = cols[1].number_input( 
        "End Time (min)", 
        min_value=end_time_min_widget_val, 
        max_value=max_minutes_for_range, 
        value=st.session_state[end_time_key],
        step=interval_duration_min,
        key=f"widget_{end_time_key}", 
        help=f"Select the end of the time range ({end_time_min_widget_val} to {max_minutes_for_range})."
    )
    st.session_state[end_time_key] = new_end_time

    if st.session_state[end_time_key] < st.session_state[start_time_key]:
        st.session_state[end_time_key] = st.session_state[start_time_key]
    
    if prev_start_time != st.session_state[start_time_key] or \
       prev_end_time != st.session_state[end_time_key]:
        logger.debug(f"Time range changed for {tab_key_prefix}: {st.session_state[start_time_key]}-{st.session_state[end_time_key]}. Rerunning.")
        st.rerun()

    return int(st.session_state[start_time_key]), int(st.session_state[end_time_key])

# --- MAIN FUNCTION ---
def main():
    st.title("Workplace Shift Optimization Dashboard")
    
    minutes_per_interval_global = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)
    if minutes_per_interval_global <= 0: minutes_per_interval_global = 2

    app_state_defaults = {
        'simulation_results': None, 'show_tour': False, 'show_help_glossary': False,
        'sb_team_size_num': DEFAULT_CONFIG['TEAM_SIZE'],
        'sb_shift_duration_num': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
        'sb_scheduled_events_list': list(DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', [])),
        'sb_team_initiative_selectbox': "Standard Operations",
        'sb_high_contrast_checkbox': False, 'sb_use_3d_distribution_checkbox': False, 'sb_debug_mode_checkbox': False,
        'form_event_type': ["Major Disruption", "Minor Disruption", "Scheduled Break", "Short Pause", "Team Meeting", "Maintenance", "Custom Event"][0],
        'form_event_start': 0, 
        'form_event_duration': max(minutes_per_interval_global, 10), 
    }
    default_max_mins = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - minutes_per_interval_global if DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] > minutes_per_interval_global else 0
    for prefix in ['op', 'ww', 'dt']:
        app_state_defaults[f'{prefix}_start_time_min'] = 0
        app_state_defaults[f'{prefix}_end_time_min'] = default_max_mins

    for key, default_value in app_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            
    sb_run_sim_btn, sb_load_data_btn = render_settings_sidebar() 
    
    current_team_size = st.session_state.sb_team_size_num
    current_shift_duration = st.session_state.sb_shift_duration_num
    current_scheduled_events = list(st.session_state.sb_scheduled_events_list) 
    current_team_initiative = st.session_state.sb_team_initiative_selectbox
    
    current_high_contrast_setting = st.session_state.sb_high_contrast_checkbox
    sb_use_3d_val = st.session_state.sb_use_3d_distribution_checkbox

    disruption_steps_for_plots = [] 
    current_max_minutes_for_inputs = default_max_mins
    minutes_per_interval_active = minutes_per_interval_global 

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        sim_cfg = st.session_state.simulation_results.get('config_params', {})
        minutes_per_interval_active = sim_cfg.get('MINUTES_PER_INTERVAL', minutes_per_interval_global)
        if minutes_per_interval_active <= 0: minutes_per_interval_active = 2 

        sim_intervals_cfg = sim_cfg.get('SHIFT_DURATION_INTERVALS', 0)
        current_max_minutes_for_inputs = max(0, sim_intervals_cfg * minutes_per_interval_active - minutes_per_interval_active) if sim_intervals_cfg > 0 else 0
        
        disruption_steps_for_plots = sim_cfg.get('DISRUPTION_EVENT_STEPS', [])
        if not disruption_steps_for_plots and 'SCHEDULED_EVENTS' in sim_cfg:
            temp_disruption_steps = []
            for event in sim_cfg['SCHEDULED_EVENTS']:
                if isinstance(event, dict) and "Disruption" in event.get("Event Type", ""):
                    step = event.get('step')
                    if step is None: 
                        start_time = event.get("Start Time (min)")
                        if isinstance(start_time, (int, float)) and start_time >= 0 and minutes_per_interval_active > 0:
                            step = int(start_time // minutes_per_interval_active)
                    if isinstance(step, int): temp_disruption_steps.append(step)
            disruption_steps_for_plots = sorted(list(set(temp_disruption_steps)))
        logger.debug(f"Main: Using disruption_steps_for_plots from sim_data: {disruption_steps_for_plots}")
    else: 
        num_intervals_from_sidebar = current_shift_duration // minutes_per_interval_active if minutes_per_interval_active > 0 else 0
        current_max_minutes_for_inputs = max(0, num_intervals_from_sidebar * minutes_per_interval_active - minutes_per_interval_active) if num_intervals_from_sidebar > 0 else 0
        temp_disruption_steps = []
        for event in current_scheduled_events: 
            if isinstance(event, dict) and "Disruption" in event.get("Event Type", ""):
                step = event.get('step') 
                if step is None: 
                    start_time = event.get("Start Time (min)")
                    if isinstance(start_time, (int, float)) and start_time >= 0 and minutes_per_interval_active > 0:
                        step = int(start_time // minutes_per_interval_active)
                if isinstance(step, int): temp_disruption_steps.append(step)
        disruption_steps_for_plots = sorted(list(set(temp_disruption_steps)))
        logger.debug(f"Main: Using disruption_steps_for_plots from sidebar: {disruption_steps_for_plots}")

    for prefix in ['op', 'ww', 'dt']:
        max_val_for_prefix = current_max_minutes_for_inputs
        # Ensure start is not greater than end, and both are within max_val
        st.session_state[f"{prefix}_start_time_min"] = max(0, min(st.session_state.get(f"{prefix}_start_time_min", 0), max_val_for_prefix))
        st.session_state[f"{prefix}_end_time_min"] = max(st.session_state[f"{prefix}_start_time_min"], min(st.session_state.get(f"{prefix}_end_time_min", max_val_for_prefix), max_val_for_prefix))


    if sb_run_sim_btn:
        # ... (sb_run_sim_btn logic remains the same)
        with st.spinner("🚀 Simulating workplace operations..."): 
            try:
                logger.info(f"Events passed to run_simulation_logic: {current_scheduled_events}", extra={'user_action': 'Prepare Simulation Run'})
                sim_results = run_simulation_logic(
                    current_team_size, current_shift_duration, current_scheduled_events, current_team_initiative
                )
                st.session_state.simulation_results = sim_results
                
                new_sim_cfg = sim_results['config_params']
                new_minutes_per_interval = new_sim_cfg.get('MINUTES_PER_INTERVAL', 2)
                new_sim_intervals = new_sim_cfg.get('SHIFT_DURATION_INTERVALS',0)
                new_max_mins = max(0, new_sim_intervals * new_minutes_per_interval - new_minutes_per_interval) if new_sim_intervals > 0 else 0
                
                for prefix in ['op', 'ww', 'dt']: 
                    st.session_state[f"{prefix}_start_time_min"] = 0
                    st.session_state[f"{prefix}_end_time_min"] = new_max_mins

                st.success("✅ Simulation completed!")
                logger.info("Simulation run successful.", extra={'user_action': 'Run Simulation - Success'})
                st.rerun() 
            except Exception as e:
                logger.error(f"Simulation Run Error: {e}", exc_info=True, extra={'user_action': 'Run Simulation - Error'})
                st.error(f"❌ Simulation failed: {str(e)}") 
                st.session_state.simulation_results = None 


    if sb_load_data_btn:
        # ... (sb_load_data_btn logic remains the same)
        with st.spinner("🔄 Loading saved simulation data..."): 
            try:
                loaded_data = load_simulation_data()
                if loaded_data and isinstance(loaded_data, dict) and 'config_params' in loaded_data:
                    st.session_state.simulation_results = loaded_data
                    cfg = loaded_data['config_params'] 
                    
                    st.session_state.sb_team_size_num = cfg.get('TEAM_SIZE', DEFAULT_CONFIG['TEAM_SIZE'])
                    st.session_state.sb_shift_duration_num = cfg.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'])
                    st.session_state.sb_scheduled_events_list = list(cfg.get('SCHEDULED_EVENTS', DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', [])))
                    st.session_state.sb_team_initiative_selectbox = cfg.get('TEAM_INITIATIVE', "Standard Operations")
                     
                    loaded_minutes_per_interval = cfg.get('MINUTES_PER_INTERVAL', 2)
                    loaded_sim_intervals = cfg.get('SHIFT_DURATION_INTERVALS', 0)
                    new_max_mins_load = max(0, loaded_sim_intervals * loaded_minutes_per_interval - loaded_minutes_per_interval) if loaded_sim_intervals > 0 else 0
                    
                    for prefix in ['op', 'ww', 'dt']: 
                        st.session_state[f"{prefix}_start_time_min"] = 0
                        st.session_state[f"{prefix}_end_time_min"] = new_max_mins_load
                    
                    st.success("✅ Data loaded successfully!")
                    logger.info("Saved data loaded successfully.", extra={'user_action': 'Load Data - Success'})
                    st.rerun()
                else:
                    st.error("❌ Failed to load data or data is incomplete/invalid.")
                    logger.warning("Load data failed or invalid format (missing 'config_params').", extra={'user_action': 'Load Data - Fail/Invalid'})
                    st.session_state.simulation_results = None 
            except Exception as e:
                logger.error(f"Load Data Error: {e}", exc_info=True, extra={'user_action': 'Load Data - Error'})
                st.error(f"❌ Failed to load data: {e}")
                st.session_state.simulation_results = None


    # Modals and Overview Tab ... (remain the same as previous correct version)
    if st.session_state.show_tour: 
        with st.container():
             st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3><p>Welcome! ...</p></div>""", unsafe_allow_html=True)
        if st.button("Got it!", key="tour_modal_close_btn"): 
            st.session_state.show_tour = False; st.rerun()
    if st.session_state.show_help_glossary: 
        with st.container():
            st.markdown(""" <div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3> <p>This dashboard provides insights...</p></div> """, unsafe_allow_html=True) 
        if st.button("Understood", key="help_modal_close_btn"): 
            st.session_state.show_help_glossary = False; st.rerun()

    tabs_main_names = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Well-being", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs = st.tabs(tabs_main_names)
    plot_config_interactive = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'pan2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}
    plot_config_minimal = {'displayModeBar': False} 
    
    with tabs[0]: 
        # ... (Overview Tab content remains the same as previous correct version)
        st.header("📊 Key Performance Indicators & Actionable Insights", divider="blue")
        if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
            sim_data = st.session_state.simulation_results
            effective_config_base = DEFAULT_CONFIG.copy()
            effective_config_base.update(sim_data.get('config_params', {})) 
            effective_config = effective_config_base

            compliance_target = float(effective_config.get('TARGET_COMPLIANCE', 75.0))
            collab_target = float(effective_config.get('TARGET_COLLABORATION', 60.0))
            wb_target = float(effective_config.get('TARGET_WELLBEING', 70.0))
            
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
                if summary_figs and isinstance(summary_figs, list): 
                    cols_gauges = st.columns(min(len(summary_figs), 4) or 1) 
                    for i_gauge, fig_gauge in enumerate(summary_figs): 
                        if fig_gauge: 
                            cols_gauges[i_gauge % len(cols_gauges)].plotly_chart(fig_gauge, use_container_width=True, config=plot_config_minimal)
                        else:
                            logger.warning(f"Overview: Gauge fig {i_gauge} is None.")
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
                num_total_steps_overview = effective_config.get('SHIFT_DURATION_INTERVALS', 0)
                minutes_per_interval_overview = effective_config.get('MINUTES_PER_INTERVAL', minutes_per_interval_global)

                if num_total_steps_overview > 0:
                    df_data_overview = {'Time (min)': [i * minutes_per_interval_overview for i in range(num_total_steps_overview)]}
                    df_data_overview['Task Compliance (%)'] = _prepare_timeseries_for_export(safe_get(sim_data, 'task_compliance.data', []), num_total_steps_overview)
                    df_data_overview['Collaboration (%)'] = _prepare_timeseries_for_export(safe_get(sim_data, 'collaboration_proximity.data', []), num_total_steps_overview)
                    df_data_overview['Well-Being (%)'] = _prepare_timeseries_for_export(safe_get(sim_data, 'worker_wellbeing.scores', []), num_total_steps_overview)
                    
                    downtime_events_for_table = safe_get(sim_data, 'downtime_minutes', [])
                    df_data_overview['Downtime (min/interval)'] = aggregate_downtime_by_step(downtime_events_for_table, num_total_steps_overview, minutes_per_interval_overview)
                    
                    st.dataframe(pd.DataFrame(df_data_overview).style.format("{:.1f}", na_rep="-").set_table_styles([{'selector': 'th', 'props': [('background-color', '#293344'), ('color', '#EAEAEA')]}]), use_container_width=True, height=300)
                else: st.caption("No detailed overview data (0 simulation steps).")
        else: st.info("ℹ️ Run a simulation or load data to view the Overview & Insights.", icon="📊")


    op_insights_html = """<div class='alert-info insight-text' style='margin-top:1rem;'><p class="insight-title">Review Operational Bottlenecks:</p><ul><li><b>Low Compliance/OEE:</b> Investigate root causes...</li></ul></div>"""
    ww_static_insights_html = """<h6 style='margin-top:1.5rem;'>💡 Considerations for Psychosocial Well-being:</h6><ul style="font-size:0.9rem; color: #D1D5DB; padding-left:20px; margin-bottom:0;"><li><strong>Monitor Risk Factors:</strong> ...</li></ul>""" 
    dt_insights_html = """<div class='alert-info insight-text' style='margin-top:1rem;'><p class="insight-title">Focus Areas for Downtime Reduction:</p><ul><li><strong>Prioritize by Cause:</strong> ...</li></ul></div>"""

    tab_configs = [
        {"name": "📈 Operational Metrics", "key_prefix": "op", 
         "plots": [
             {"title": "Task Compliance Score Over Time", "data_path": "task_compliance.data", "plot_func": plot_task_compliance_score, "extra_args_paths": {"forecast_data": "task_compliance.forecast", "z_scores": "task_compliance.z_scores"}},
             {"title": "Collaboration Proximity Index Over Time", "data_path": "collaboration_proximity.data", "plot_func": plot_collaboration_proximity_index, "extra_args_paths": {"forecast_data": "collaboration_proximity.forecast"}},
             {"is_subheader": True, "title": "Additional Operational Metrics"}, 
             {"title": "Operational Recovery vs. Loss", "data_path": "operational_recovery", "plot_func": plot_operational_recovery, "extra_args_paths": {"productivity_loss_data": "productivity_loss"}},
             {"title": "OEE & Components", "is_oee": True} # No data_path for OEE as it's special
         ], "insights_html": op_insights_html},
        {"name": "👥 Worker Well-being", "key_prefix": "ww", 
         "plots": [
             {"is_subheader": True, "title": "Psychosocial & Well-being Indicators"},
             {"title": "Worker Well-Being Index", "data_path": "worker_wellbeing.scores", "plot_func": plot_worker_wellbeing, "extra_args_paths": {"triggers": "worker_wellbeing.triggers"}}, # Triggers path
             {"title": "Psychological Safety Score", "data_path": "psychological_safety", "plot_func": plot_psychological_safety},
             {"title": "Team Cohesion Index", "data_path": "worker_wellbeing.team_cohesion_scores", "plot_func": plot_team_cohesion},
             {"title": "Perceived Workload Index (0-10)", "data_path": "worker_wellbeing.perceived_workload_scores", "plot_func": plot_perceived_workload, "extra_args_fixed": {"high_workload_threshold": DEFAULT_CONFIG.get('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5), "very_high_workload_threshold": DEFAULT_CONFIG.get('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5)}},
             {"is_subheader": True, "title": "Spatial Dynamics Analysis", "is_spatial": True}
         ], "dynamic_insights_func": "render_wellbeing_alerts", "insights_html": ww_static_insights_html },
        {"name": "⏱️ Downtime Analysis", "key_prefix": "dt", "metrics_display": True,
         "plots": [
            {"title": "Downtime Trend (per Interval)", "data_path": "downtime_minutes", "plot_func": plot_downtime_trend, "is_event_based_aggregation": True, "extra_args_fixed": {"interval_threshold": DEFAULT_CONFIG.get('DOWNTIME_PLOT_ALERT_THRESHOLD', 10)}},
            {"title": "Downtime Distribution by Cause", "data_path": "downtime_minutes", "plot_func": plot_downtime_causes_pie, "is_event_based_filtering": True}
         ], "insights_html": dt_insights_html }
    ]

    for i, tab_config in enumerate(tab_configs): 
        with tabs[i+1]: 
            st.header(tab_config["name"], divider="blue") 
            if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
                sim_data = st.session_state.simulation_results
                sim_cfg_tab = sim_data.get('config_params', {})
                
                st.markdown(f"##### Select Time Range for Plots:")
                start_time_min, end_time_min = time_range_input_section(
                    tab_config["key_prefix"], current_max_minutes_for_inputs, 
                    interval_duration_min=minutes_per_interval_active
                )
                start_idx = start_time_min // minutes_per_interval_active if minutes_per_interval_active > 0 else 0
                end_idx = (end_time_min // minutes_per_interval_active) + 1 if minutes_per_interval_active > 0 else 0 # end_idx is exclusive for slicing
                
                logger.debug(f"Tab '{tab_config['name']}': Time range {start_time_min}-{end_time_min} min (Indices {start_idx}-{end_idx-1}). MPI_active: {minutes_per_interval_active}")
                
                filt_disrupt_steps_for_tab_plots_abs = [s for s in disruption_steps_for_plots if start_idx <= s < end_idx] # Absolute steps in current window
                logger.debug(f"Tab '{tab_config['name']}': Absolute disruption steps in window: {filt_disrupt_steps_for_tab_plots_abs}")

                if tab_config.get("metrics_display"): 
                    # ... (metrics display logic remains the same)
                    downtime_events_list_all = safe_get(sim_data, 'downtime_minutes', []) 
                    downtime_events_in_range = [
                        event for event in downtime_events_list_all 
                        if isinstance(event, dict) and start_idx <= event.get('step', -1) < end_idx
                    ]
                    downtime_durations_in_range = [event.get('duration',0.0) for event in downtime_events_in_range]
                    
                    if downtime_events_in_range: 
                        total_downtime_period = sum(downtime_durations_in_range)
                        num_incidents = len([d for d in downtime_durations_in_range if d > 0]) 
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
                            # ... (spatial plotting logic remains the same as previous correct version)
                            facility_config_for_spatial = DEFAULT_CONFIG.copy() 
                            sim_work_areas = sim_cfg_tab.get('WORK_AREAS_EFFECTIVE', DEFAULT_CONFIG.get('WORK_AREAS', {}))
                            facility_config_for_spatial['WORK_AREAS'] = sim_work_areas
                            facility_config_for_spatial['FACILITY_SIZE'] = DEFAULT_CONFIG['FACILITY_SIZE'] 
                            facility_config_for_spatial['ENTRY_EXIT_POINTS'] = DEFAULT_CONFIG.get('ENTRY_EXIT_POINTS', [])

                            with st.container(border=True):
                                team_pos_df_all = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
                                zones_dist = ["All"] + list(facility_config_for_spatial.get('WORK_AREAS', {}).keys())
                                zone_sel_key = f"{tab_config['key_prefix']}_zone_sel_spatial_dist"
                                if zone_sel_key not in st.session_state: st.session_state[zone_sel_key] = "All"
                                zone_sel_dist = st.selectbox("Filter by Zone:", zones_dist, key=zone_sel_key)
                                
                                filt_team_pos_df_spatial_time = pd.DataFrame()
                                if not team_pos_df_all.empty and 'step' in team_pos_df_all.columns:
                                    filt_team_pos_df_spatial_time = team_pos_df_all[(team_pos_df_all['step'] >= start_idx) & (team_pos_df_all['step'] < end_idx)]
                                
                                filt_team_pos_df_spatial_final = filt_team_pos_df_spatial_time
                                if zone_sel_dist != "All" and not filt_team_pos_df_spatial_final.empty and 'zone' in filt_team_pos_df_spatial_final.columns : 
                                    filt_team_pos_df_spatial_final = filt_team_pos_df_spatial_final[filt_team_pos_df_spatial_final['zone'] == zone_sel_dist]

                                show_ee_key = f'{tab_config["key_prefix"]}_show_ee_spatial_cb'
                                if show_ee_key not in st.session_state: st.session_state[show_ee_key] = True
                                show_ee_exp = st.checkbox("Show E/E Points", key=show_ee_key) 
                                
                                show_pl_key = f'{tab_config["key_prefix"]}_show_pl_spatial_cb'
                                if show_pl_key not in st.session_state: st.session_state[show_pl_key] = True
                                show_pl_exp = st.checkbox("Show Area Outlines", key=show_pl_key)
                                
                                spatial_plot_cols = st.columns(2)
                                with spatial_plot_cols[0]: 
                                    st.markdown("<h6>Worker Positions (Snapshot)</h6>", unsafe_allow_html=True)
                                    min_snap_step, max_snap_step = start_idx, max(start_idx, end_idx -1) 
                                    snap_slider_key = f"{tab_config['key_prefix']}_snap_step_slider"
                                    if snap_slider_key not in st.session_state or not (min_snap_step <= st.session_state[snap_slider_key] <= max_snap_step):
                                        st.session_state[snap_slider_key] = min_snap_step if min_snap_step <= max_snap_step else (max_snap_step if max_snap_step >= min_snap_step else 0)
                                    
                                    snap_step_val = st.slider("Time Step for Snapshot:", min_snap_step, max_snap_step, key=snap_slider_key, step=1, disabled=(max_snap_step < min_snap_step))
                                    
                                    if not team_pos_df_all.empty and max_snap_step >= min_snap_step:
                                        fig_dist = None
                                        try: 
                                            fig_dist = plot_worker_distribution(
                                                team_pos_df_all, facility_config_for_spatial.get('FACILITY_SIZE', [100,100]), 
                                                facility_config_for_spatial, 
                                                sb_use_3d_val, snap_step_val, show_ee_exp, show_pl_exp, 
                                                current_high_contrast_setting
                                            )
                                            if fig_dist: st.plotly_chart(fig_dist, use_container_width=True, config=plot_config_interactive)
                                            else: st.caption("Worker distribution plot could not be generated."); logger.warning("plot_worker_distribution returned None.")
                                        except Exception as e: 
                                            logger.error(f"Spatial Dist Plot Error: {e}", exc_info=True, extra={'user_action': 'Plot Error'})
                                            st.error(f"⚠️ Error plotting Worker Positions: {str(e)}.")
                                    else: st.caption("No data for positions snapshot or invalid time range.")
                                with spatial_plot_cols[1]: 
                                    st.markdown("<h6>Worker Density Heatmap</h6>", unsafe_allow_html=True)
                                    if not filt_team_pos_df_spatial_final.empty: 
                                        fig_heat = None
                                        try: 
                                            fig_heat = plot_worker_density_heatmap(
                                                filt_team_pos_df_spatial_final, facility_config_for_spatial.get('FACILITY_SIZE', [100,100]), 
                                                facility_config_for_spatial, 
                                                show_ee_exp, show_pl_exp, current_high_contrast_setting
                                            )
                                            if fig_heat: st.plotly_chart(fig_heat, use_container_width=True, config=plot_config_interactive)
                                            else: st.caption("Density heatmap could not be generated."); logger.warning("plot_worker_density_heatmap returned None.")
                                        except Exception as e: 
                                            logger.error(f"Spatial Heatmap Plot Error: {e}", exc_info=True, extra={'user_action': 'Plot Error'})
                                            st.error(f"⚠️ Error plotting Density Heatmap: {str(e)}.")
                                    else: st.caption("No data for density heatmap in this time range/zone.")

                        num_plots_in_row = 0 
                        continue # Skip to next plot_info in tab_config["plots"]

                    # --- This section is for individual plots (not subheaders/spatial) ---
                    if num_plots_in_row == 0: 
                       plot_columns = plot_col_container.columns(2)
                    
                    current_plot_col = plot_columns[num_plots_in_row % 2]
                    
                    with current_plot_col:
                        st.markdown(f"<h6>{plot_info['title']}</h6>", unsafe_allow_html=True) # Plot title
                        with st.container(border=True): 
                            try:
                                data_to_plot_final = None
                                kwargs_for_plot = {"high_contrast": current_high_contrast_setting} 

                                # Handle OEE plot separately as it doesn't use a single 'data_path'
                                if plot_info.get("is_oee"):
                                    eff_df_full = safe_get(sim_data, 'efficiency_metrics_df', pd.DataFrame())
                                    if not eff_df_full.empty:
                                        oee_ms_key = f"{tab_config['key_prefix']}_oee_metrics_ms"
                                        if oee_ms_key not in st.session_state: st.session_state[oee_ms_key] = ['uptime', 'throughput', 'quality', 'oee']
                                        sel_metrics = st.multiselect("Select OEE Metrics:", ['uptime', 'throughput', 'quality', 'oee'], key=oee_ms_key)
                                        
                                        filt_eff_df = _slice_dataframe_by_step_indices(eff_df_full, start_idx, end_idx)
                                        
                                        if "disruption_points" in plot_operational_efficiency.__code__.co_varnames:
                                            kwargs_for_plot["disruption_points"] = [s - start_idx for s in filt_disrupt_steps_for_tab_plots_abs if s - start_idx >=0]


                                        if not filt_eff_df.empty:
                                            fig_oee = None
                                            try:
                                                fig_oee = plot_operational_efficiency(filt_eff_df, sel_metrics, **kwargs_for_plot)
                                                if fig_oee: st.plotly_chart(fig_oee, use_container_width=True, config=plot_config_interactive)
                                                else: st.caption("OEE plot could not be generated."); logger.warning("plot_operational_efficiency returned None.")
                                            except Exception as e_oee:
                                                logger.error(f"OEE plot error: {e_oee}", exc_info=True); st.error(f"Error plotting OEE: {e_oee}")
                                        else: st.caption("No OEE data for this time range.")
                                    else: st.caption("No OEE data available.")
                                    num_plots_in_row +=1 
                                    continue # Move to next plot_info in the loop for this tab

                                # --- Generic data fetching for other plots ---
                                plot_data_raw = safe_get(sim_data, plot_info["data_path"], []) # This was the source of KeyError for OEE
                                
                                # Prepare kwargs (extra_args_paths, extra_args_fixed, disruption_points)
                                if "extra_args_paths" in plot_info:
                                    for arg_name, arg_path in plot_info["extra_args_paths"].items():
                                        extra_data_raw = safe_get(sim_data, arg_path, [])
                                        
                                        if plot_info["plot_func"] == plot_worker_wellbeing and arg_name == "triggers":
                                            # Filter and relativize triggers for plot_worker_wellbeing
                                            filtered_triggers = {}
                                            if isinstance(extra_data_raw, dict):
                                                for trig_type, trig_steps_abs in extra_data_raw.items():
                                                    if trig_type == 'work_area' and isinstance(trig_steps_abs, dict):
                                                        filtered_triggers[trig_type] = {
                                                            zone: [s - start_idx for s in (steps_list if isinstance(steps_list, list) else []) if start_idx <= s < end_idx and s - start_idx >= 0]
                                                            for zone, steps_list in trig_steps_abs.items()
                                                        }
                                                        # Remove empty zone lists
                                                        filtered_triggers[trig_type] = {k:v for k,v in filtered_triggers[trig_type].items() if v}

                                                    elif isinstance(trig_steps_abs, list):
                                                        filtered_triggers[trig_type] = [s - start_idx for s in trig_steps_abs if start_idx <= s < end_idx and s - start_idx >=0]
                                            kwargs_for_plot[arg_name] = filtered_triggers
                                            logger.debug(f"Wellbeing plot: Processed triggers for plot func: {filtered_triggers}")
                                        
                                        elif isinstance(extra_data_raw, list) and start_idx < len(extra_data_raw): # Slicing for other list-based extra args
                                            kwargs_for_plot[arg_name] = extra_data_raw[start_idx:min(end_idx, len(extra_data_raw))]
                                        elif isinstance(extra_data_raw, list): 
                                            kwargs_for_plot[arg_name] = []
                                        else: 
                                            kwargs_for_plot[arg_name] = extra_data_raw # Pass as is if not a list (e.g. a DF)

                                if "extra_args_fixed" in plot_info: 
                                    kwargs_for_plot.update(plot_info["extra_args_fixed"])
                                
                                func_params = plot_info["plot_func"].__code__.co_varnames
                                if "disruption_points" in func_params:
                                    # Disruption points should be relative to the start of the sliced data
                                    kwargs_for_plot["disruption_points"] = [s - start_idx for s in filt_disrupt_steps_for_tab_plots_abs if s - start_idx >=0]
                                    logger.debug(f"Plot '{plot_info['title']}': disruption_points for plot func: {kwargs_for_plot.get('disruption_points')}")

                                # --- Data preparation for the main plot data ---
                                if plot_info.get("is_event_based_aggregation"): 
                                    all_events = plot_data_raw if isinstance(plot_data_raw, list) else []
                                    num_steps_in_range = end_idx - start_idx
                                    aggregated_data_in_range = [0.0] * num_steps_in_range if num_steps_in_range > 0 else []
                                    for event in all_events: 
                                        if isinstance(event, dict) and 'step' in event and 'duration' in event:
                                            step = event['step']
                                            if start_idx <= step < end_idx: 
                                                relative_step = step - start_idx
                                                if 0 <= relative_step < num_steps_in_range: 
                                                    aggregated_data_in_range[relative_step] += float(event['duration'])
                                    data_to_plot_final = aggregated_data_in_range
                                elif plot_info.get("is_event_based_filtering"): 
                                    all_events = plot_data_raw if isinstance(plot_data_raw, list) else []
                                    filtered_events_for_plot = []
                                    for event in all_events:
                                        if isinstance(event, dict) and 'step' in event: 
                                            if start_idx <= event['step'] < end_idx:
                                                filtered_events_for_plot.append(event)
                                    data_to_plot_final = filtered_events_for_plot
                                    if "disruption_points" in kwargs_for_plot: del kwargs_for_plot["disruption_points"] 
                                else: # Standard list or DataFrame slicing
                                    if isinstance(plot_data_raw, list):
                                        data_to_plot_final = plot_data_raw[start_idx:min(end_idx, len(plot_data_raw))] if start_idx < len(plot_data_raw) and start_idx < end_idx else []
                                    elif isinstance(plot_data_raw, pd.DataFrame) and not plot_data_raw.empty:
                                        data_to_plot_final = _slice_dataframe_by_step_indices(plot_data_raw, start_idx, end_idx)
                                    else: data_to_plot_final = []
                                
                                # --- Final check and plot rendering ---
                                final_check_has_data = False
                                if isinstance(data_to_plot_final, list) and data_to_plot_final: final_check_has_data = True
                                elif isinstance(data_to_plot_final, pd.DataFrame) and not data_to_plot_final.empty: final_check_has_data = True
                                elif plot_info.get("is_event_based_filtering") and isinstance(data_to_plot_final, list): final_check_has_data = True 
                                elif plot_info.get("is_event_based_aggregation") and isinstance(data_to_plot_final, list) and (any(x > 0 for x in data_to_plot_final) or (end_idx - start_idx > 0)): final_check_has_data = True


                                if not final_check_has_data:
                                    st.caption(f"No data to plot for '{plot_info['title']}' in this time range.")
                                    logger.debug(f"Plot '{plot_info['title']}': No final data to plot. Sliced data type: {type(data_to_plot_final)}")
                                else:
                                    fig_to_show = None
                                    try:
                                        logger.debug(f"Calling {plot_info['plot_func'].__name__} for '{plot_info['title']}'. Data preview: {str(data_to_plot_final)[:100]}. Kwargs: {kwargs_for_plot}")
                                        fig_to_show = plot_info["plot_func"](data_to_plot_final, **kwargs_for_plot)
                                        
                                        if fig_to_show: 
                                            st.plotly_chart(fig_to_show, use_container_width=True, config=plot_config_interactive)
                                        else:
                                            st.caption(f"Plot for '{plot_info['title']}' could not be generated (function returned None).")
                                            logger.warning(f"{plot_info['plot_func'].__name__} for '{plot_info['title']}' returned None. Data: {str(data_to_plot_final)[:100]}")
                                    except Exception as e_plot_func:
                                        logger.error(f"Error in {plot_info['plot_func'].__name__} for '{plot_info['title']}': {e_plot_func}", exc_info=True)
                                        st.error(f"⚠️ Error generating plot '{plot_info['title']}': {e_plot_func}")
                            
                            except Exception as e_outer: # Catch errors in data prep before calling plot func
                                logger.error(f"Tab '{tab_config['name']}', Plot '{plot_info['title']}' Data Prep Error: {e_outer}", exc_info=True)
                                st.error(f"⚠️ Error setting up plot {plot_info['title']}: {str(e_outer)}")
                    num_plots_in_row += 1
                
                # Insights section for the tab
                st.markdown("<hr style='margin-top:2rem;'><h3 style='text-align:center; margin-top:1rem;'>🏛️ Leadership Actionable Insights</h3>", unsafe_allow_html=True)
                if tab_config.get("dynamic_insights_func") == "render_wellbeing_alerts":
                    # ... (dynamic insights logic for wellbeing remains the same)
                    with st.container(border=True): 
                        st.markdown("<h6>Well-Being Alerts (within selected time range):</h6>", unsafe_allow_html=True)
                        ww_trigs_disp_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {}) 
                        
                        insights_count_wb = 0 
                        for alert_type, alert_steps_raw in ww_trigs_disp_raw.items():
                            if alert_type == 'work_area' and isinstance(alert_steps_raw, dict): 
                                wa_alert_found_in_range = False; wa_details_html = ""
                                for zone, zone_steps_raw_list in alert_steps_raw.items():
                                    zone_steps_in_range = [s for s in (zone_steps_raw_list if isinstance(zone_steps_raw_list, list) else []) if start_idx <= s < end_idx]
                                    if zone_steps_in_range:
                                        wa_alert_found_in_range = True
                                        wa_details_html += f"  - {zone}: {len(zone_steps_in_range)} alerts at steps {zone_steps_in_range}<br>"
                                if wa_alert_found_in_range:
                                    st.markdown(f"<div class='alert-warning insight-text'><strong>Work Area Specific Alerts:</strong><br>{wa_details_html}</div>", unsafe_allow_html=True)
                                    insights_count_wb +=1
                            elif isinstance(alert_steps_raw, list): 
                                alert_steps_in_range = [s for s in alert_steps_raw if start_idx <= s < end_idx]
                                if alert_steps_in_range:
                                    alert_class = "alert-critical" if alert_type == "threshold" else "alert-warning" if alert_type == "trend" else "alert-info"
                                    alert_title_text = alert_type.replace("_", " ").title()
                                    st.markdown(f"<div class='{alert_class} insight-text'><strong>{alert_title_text} Alerts ({len(alert_steps_in_range)} times):</strong> Steps {alert_steps_in_range}.</div>", unsafe_allow_html=True)
                                    insights_count_wb += 1
                        
                        if insights_count_wb == 0: 
                            st.markdown(f"<p class='insight-text' style='color: {COLOR_POSITIVE_GREEN};'>✅ No specific well-being alerts triggered in the selected period.</p>", unsafe_allow_html=True)

                if tab_config.get("insights_html"): 
                     st.markdown(tab_config["insights_html"], unsafe_allow_html=True) 
            else: 
                st.info(f"ℹ️ Run a simulation or load data to view {tab_config['name']}.", icon="📊")
    
    # Glossary Tab
    with tabs[4]: 
        # ... (Glossary content remains the same)
        st.header("📖 Glossary of Terms", divider="blue") 
        st.markdown(""" <div style="font-size: 0.95rem; line-height: 1.7;"> <p>This glossary defines key metrics...</p> </div> """, unsafe_allow_html=True) 


if __name__ == "__main__":
    main()
