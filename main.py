# main.py
import logging
import streamlit as st
import pandas as pd
import numpy as np
# import math # Not directly used in main, simulation handles its own math needs
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

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.3.1\nInsights for operational excellence & psychosocial well-being."})

# CSS Color constants (from your original CSS, also used by visualizations.py)
COLOR_CRITICAL_RED_CSS = "#E53E3E"; COLOR_WARNING_AMBER_CSS = "#F59E0B"; COLOR_POSITIVE_GREEN_CSS = "#10B981"; COLOR_INFO_BLUE_CSS = "#3B82F6"; COLOR_ACCENT_INDIGO_CSS = "#4F46E5"

# --- UTILITY FUNCTIONS (main.py specific or simple helpers) ---
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
            elif isinstance(current, (list, pd.Series)) and key.isdigit(): current = current[int(key)] if int(key) < len(current) else None
            else: current = None; break
        if current is None:
            is_list_like_final_key = keys and keys[-1] in ['data', 'scores', 'triggers', '_log', 'events_list']
            return [] if default_val is None and is_list_like_final_key else default_val
        return current
    except (ValueError, IndexError, TypeError): return default_return

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

def get_actionable_insights(sim_data, current_config_dict):
    insights = []
    if not sim_data or not isinstance(sim_data, dict): 
        logger.warning("get_actionable_insights: sim_data is None or not a dict.", extra={'user_action': 'Insights - Invalid Input'})
        return insights
    
    sim_cfg_params_insights = sim_data.get('config_params', {}) # Use config from the simulation run

    # Task Compliance
    compliance_data = safe_get(sim_data, 'task_compliance.data', [])
    target_compliance = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'TARGET_COMPLIANCE', 75.0))
    compliance_avg = safe_stat(compliance_data, np.mean, 0.0)
    if compliance_avg < target_compliance * 0.9:
        insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Avg. Task Compliance ({compliance_avg:.1f}%) critically below target ({target_compliance:.0f}%). Review disruptions, complexities, training."})
    elif compliance_avg < target_compliance:
        insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Avg. Task Compliance at {compliance_avg:.1f}%. Review areas with lowest compliance."})

    # Worker Wellbeing
    wellbeing_scores = safe_get(sim_data, 'worker_wellbeing.scores', [])
    target_wellbeing = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'TARGET_WELLBEING', 70.0))
    wellbeing_avg = safe_stat(wellbeing_scores, np.mean, 0.0)
    wb_crit_factor = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET', 0.85))
    if wellbeing_scores and wellbeing_avg < target_wellbeing * wb_crit_factor: # Check if wellbeing_scores is not empty
        insights.append({"type": "critical", "title": "Critical Worker Well-being", "text": f"Avg. Well-being ({wellbeing_avg:.1f}%) critically low (target {target_wellbeing:.0f}%). Urgent review needed."})
    
    threshold_triggers = safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])
    if wellbeing_scores and len(threshold_triggers) > max(2, len(wellbeing_scores) * 0.1):
        insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers)} low well-being instances. Investigate triggers."})

    # Total Downtime
    downtime_event_log = safe_get(sim_data, 'downtime_events_log', []) # Use the raw log
    downtime_durations = [event.get('duration', 0.0) for event in downtime_event_log if isinstance(event, dict)]
    total_downtime = sum(downtime_durations)

    shift_mins_insights = float(sim_cfg_params_insights.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
    dt_thresh_percent_insights = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT', 0.05))
    dt_thresh_total_shift_abs_insights = shift_mins_insights * dt_thresh_percent_insights
    if total_downtime > dt_thresh_total_shift_abs_insights:
        insights.append({"type": "critical", "title": "Excessive Total Downtime", "text": f"Total downtime {total_downtime:.0f} min, exceeds guideline of {dt_thresh_total_shift_abs_insights:.0f} min ({dt_thresh_percent_insights*100:.0f}% of shift). Analyze causes."})

    # Psychological Safety
    psych_safety_scores = safe_get(sim_data, 'psychological_safety', [])
    target_psych_safety = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'TARGET_PSYCH_SAFETY', 70.0))
    psych_safety_avg = safe_stat(psych_safety_scores, np.mean, 0.0)
    if psych_safety_scores and psych_safety_avg < target_psych_safety * 0.9:
        insights.append({"type": "warning", "title": "Low Psychological Safety", "text": f"Avg. Psych. Safety ({psych_safety_avg:.1f}%) below target ({target_psych_safety:.0f}%). Build trust."})

    # Team Cohesion
    cohesion_scores = safe_get(sim_data, 'worker_wellbeing.team_cohesion_scores', [])
    target_cohesion = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'TARGET_TEAM_COHESION', 70.0))
    cohesion_avg = safe_stat(cohesion_scores, np.mean, 0.0)
    if cohesion_scores and cohesion_avg < target_cohesion * 0.9:
        insights.append({"type": "warning", "title": "Low Team Cohesion", "text": f"Avg. Team Cohesion ({cohesion_avg:.1f}%) below desired. Consider team-building."})
    
    # Perceived Workload
    workload_scores = safe_get(sim_data, 'worker_wellbeing.perceived_workload_scores', []) # Scale 0-10
    target_workload = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'TARGET_PERCEIVED_WORKLOAD', 6.5))
    workload_avg = safe_stat(workload_scores, np.mean, target_workload / 2) # Default to mid-low
    workload_very_high_thresh = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5))
    workload_high_thresh = float(_get_config_value(current_config_dict, sim_cfg_params_insights, 'PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5))

    if workload_scores: # Only generate workload insights if data exists
        if workload_avg > workload_very_high_thresh:
            insights.append({"type": "critical", "title": "Very High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) critically high. Immediate review required."})
        elif workload_avg > workload_high_thresh:
            insights.append({"type": "warning", "title": "High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) exceeds high threshold. Monitor."})
        elif workload_avg > target_workload:
            insights.append({"type": "info", "title": "Elevated Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) above target ({target_workload:.1f}/10). Consider adjustments."})
    
    # Spatial Insights (Overcrowding/Underutilization)
    team_pos_df = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
    work_areas_effective = sim_cfg_params_insights.get('WORK_AREAS_EFFECTIVE', current_config_dict.get('WORK_AREAS', {}))
    if not team_pos_df.empty and isinstance(work_areas_effective, dict) and \
       all(col in team_pos_df.columns for col in ['zone', 'worker_id', 'step']):
        for zone_name, zone_details in work_areas_effective.items():
            if not isinstance(zone_details, dict): continue
            workers_in_zone_series = team_pos_df[team_pos_df['zone'] == zone_name].groupby('step')['worker_id'].nunique()
            if not workers_in_zone_series.empty:
                workers_in_zone_avg = workers_in_zone_series.mean()
                intended_workers = zone_details.get('workers', 0)
                coords = zone_details.get('coords'); area_m2 = 1.0
                if coords and isinstance(coords, list) and len(coords) == 2 and \
                   all(isinstance(p, tuple) and len(p)==2 and all(isinstance(c, (int,float)) for c_tuple in coords for c in c_tuple) for p in coords): # Check coord format
                    (x0,y0), (x1,y1) = coords[0], coords[1]; area_m2 = abs(x1-x0) * abs(y1-y0)
                if abs(area_m2) < 1e-6: area_m2 = 1.0 # Avoid division by zero or tiny area
                
                avg_density = workers_in_zone_avg / area_m2
                intended_density = (intended_workers / area_m2) if intended_workers > 0 else 0

                if intended_density > 0 and avg_density > intended_density * 1.8: 
                     insights.append({"type": "warning", "title": f"Potential Overcrowding: '{zone_name}'", "text": f"Avg. density ({avg_density:.2f} w/m²) significantly > intended ({intended_density:.2f} w/m²). Review layout/paths."})
                elif intended_workers > 0 and workers_in_zone_avg < intended_workers * 0.4 and not zone_details.get("is_rest_area"): # Don't flag underutilization for rest areas
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

def _get_config_value(primary_conf, secondary_conf, key, default):
    """Helper to get config value, prioritizing secondary_conf (e.g., from sim_results)."""
    return secondary_conf.get(key, primary_conf.get(key, default))
def aggregate_downtime_by_step(raw_downtime_event_log, num_total_steps_agg):
    """Aggregates durations from a raw log of downtime events for each simulation step."""
    downtime_per_step_agg = [0.0] * num_total_steps_agg # Initialize with zeros
    if not isinstance(raw_downtime_event_log, list):
        logger.warning("aggregate_downtime_by_step: input is not a list.")
        return downtime_per_step_agg # Return list of zeros

    for event in raw_downtime_event_log:
        if not isinstance(event, dict): continue
        step, duration = event.get('step'), event.get('duration', 0.0)
        # Ensure step is a valid index and duration is a positive number
        if isinstance(step, int) and 0 <= step < num_total_steps_agg and isinstance(duration, (int, float)) and duration > 0:
            downtime_per_step_agg[step] += float(duration)
    return downtime_per_step_agg

def _prepare_timeseries_for_export(raw_data, num_total_steps, default_val=np.nan):
    if not isinstance(raw_data, list): return [default_val] * num_total_steps
    # Pad if shorter, truncate if longer
    return (raw_data + [default_val] * num_total_steps)[:num_total_steps]

def _slice_dataframe_by_step_indices(df, start_idx, end_idx):
    if not isinstance(df, pd.DataFrame) or df.empty: return pd.DataFrame()
    # Prioritize iloc if index is simple range, for performance
    if isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1:
        safe_start = max(0, start_idx)
        safe_end = min(len(df), end_idx)
        return df.iloc[safe_start:safe_end] if safe_start < safe_end else pd.DataFrame()
    # Fallback to other methods
    if 'step' in df.columns: return df[(df['step'] >= start_idx) & (df['step'] < end_idx)]
    if df.index.name == 'step' or df.index.is_numeric(): return df[(df.index >= start_idx) & (df.index < end_idx)]
    logger.warning(f"Could not determine how to slice DataFrame by step indices. Columns: {df.columns}, Index: {df.index.name}")
    return pd.DataFrame() # Return empty if no suitable slicing method found

# --- CSS STYLES ---
# (Full CSS block as provided in the original problem description)
st.markdown(f"""
    <style>
        /* Base Styles */
        .main {{ background-color: #121828; color: #EAEAEA; font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif; padding: 2rem; }}
        h1 {{ font-size: 2.4rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; text-align: center; margin-bottom: 2rem; color: #FFFFFF; }}
        
        /* Main Content Headers (Tabs) - Targets h2 generated by st.header in tabs */
        div[data-testid="stTabs"] section[role="tabpanel"] > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div[data-testid="stVerticalBlock"] > div:nth-child(1) > div > h2 {{ 
            font-size: 1.75rem !important; font-weight: 600 !important; line-height: 1.3 !important; margin: 1.2rem 0 1rem 0 !important; 
            color: #D1D5DB !important; border-bottom: 2px solid {COLOR_ACCENT_INDIGO_CSS} !important; padding-bottom: 0.6rem !important; text-align: left !important;
        }}
        /* ... (Rest of the CSS block from your original file) ... */
        .remove-event-btn button {{background-color: {COLOR_CRITICAL_RED_CSS} !important; color: white !important; padding: 0.1rem 0.4rem !important; font-size: 0.75rem !important; line-height: 1 !important; border-radius: 3px !important; min-height: auto !important; margin-left: 0.5rem !important;}}
    </style>
""", unsafe_allow_html=True)


# --- SIDEBAR RENDERING ---
def render_settings_sidebar():
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem; color: #A0A0A0;'>Workplace Optimizer</h3>", unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")
        with st.expander("🧪 Simulation Parameters", expanded=True):
            # Use st.session_state directly for widget values
            st.number_input("Team Size", min_value=1, max_value=200, key="sb_team_size_num", step=1,
                            help="Adjust the number of workers in the simulated shift.")
            st.number_input("Shift Duration (min)", min_value=60, max_value=7200, key="sb_shift_duration_num", step=10,
                            help="Set the total length of the simulated work shift in minutes.")
            
            current_shift_duration_sb = st.session_state.sb_shift_duration_num
            mpi_sb = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)

            st.markdown("---"); st.markdown("<h5>🗓️ Schedule Shift Events</h5>", unsafe_allow_html=True)
            st.caption("Define disruptions, breaks, etc. Times are from shift start.")
            
            event_types_sb = ["Major Disruption", "Minor Disruption", "Scheduled Break", "Short Pause", "Team Meeting", "Maintenance", "Custom Event"]
            with st.container(): # Form for new event
                st.session_state.form_event_type = st.selectbox("Event Type", event_types_sb, 
                    index=event_types_sb.index(st.session_state.form_event_type) if st.session_state.form_event_type in event_types_sb else 0, 
                    key="widget_form_event_type_selector") # Ensure unique key
                
                col1_form, col2_form = st.columns(2)
                # Use session state keys that match the ones being modified by the widget
                st.session_state.form_event_start = col1_form.number_input("Start (min)", min_value=0, 
                                                                            max_value=max(0, current_shift_duration_sb - mpi_sb), # Prevent event starting in last interval if it has duration
                                                                            step=mpi_sb, key="widget_form_event_start_input")
                st.session_state.form_event_duration = col2_form.number_input("Duration (min)", min_value=mpi_sb, 
                                                                                max_value=current_shift_duration_sb, 
                                                                                step=mpi_sb, key="widget_form_event_duration_input")

            if st.button("➕ Add Event", key="sb_add_event_button", use_container_width=True):
                # Values are already in st.session_state due to widget keying
                start_val = st.session_state.form_event_start
                duration_val = st.session_state.form_event_duration
                type_val = st.session_state.form_event_type

                if start_val + duration_val > current_shift_duration_sb:
                    st.warning("Event end time exceeds shift duration.")
                elif start_val < 0 : st.warning("Event start time cannot be negative.")
                elif duration_val < mpi_sb: st.warning(f"Event duration must be at least {mpi_sb} minute(s).")
                else:
                    st.session_state.sb_scheduled_events_list.append({
                        "Event Type": type_val,
                        "Start Time (min)": start_val, 
                        "Duration (min)": duration_val,
                        # 'step' can be derived later or here: 'step': int(start_val // mpi_sb)
                    })
                    st.session_state.sb_scheduled_events_list.sort(key=lambda x: x.get("Start Time (min)", 0))
                    # Optionally reset form fields
                    st.session_state.form_event_start = 0 
                    st.session_state.form_event_duration = max(mpi_sb, 10) # Sensible default duration
                    st.rerun()

            st.markdown("<h6>Current Scheduled Events:</h6>", unsafe_allow_html=True) 
            if not st.session_state.sb_scheduled_events_list:
                st.caption("No events scheduled yet.")
            else:
                with st.container(height=200): # Scrollable container for events
                    for i_ev_disp, event_disp in enumerate(st.session_state.sb_scheduled_events_list):
                        ev_col1, ev_col2 = st.columns([0.85,0.15])
                        ev_col1.markdown(f"<div class='event-item'><span><b>{event_disp.get('Event Type','N/A')}</b> at {event_disp.get('Start Time (min)','N/A')}min ({event_disp.get('Duration (min)','N/A')}min)</span></div>", unsafe_allow_html=True)
                        if ev_col2.button("✖", key=f"remove_event_button_{i_ev_disp}", help="Remove this event", type="secondary", use_container_width=True):
                            st.session_state.sb_scheduled_events_list.pop(i_ev_disp)
                            st.rerun()
            
            if st.session_state.sb_scheduled_events_list: # Show clear button only if there are events
                if st.button("Clear All Events", key="sb_clear_all_events_button", type="secondary", use_container_width=True):
                    st.session_state.sb_scheduled_events_list = []
                    st.rerun()
            
            st.markdown("---") 
            team_initiative_options_sb = ["Standard Operations", "More frequent breaks", "Team recognition", "Increased Autonomy"]
            st.selectbox("Operational Initiative", team_initiative_options_sb, 
                         key="sb_team_initiative_selectbox", 
                         help="Apply an operational strategy to observe its impact on metrics.")
            
            run_simulation_button_sb = st.button("🚀 Run Simulation", key="sb_run_simulation_main_button", type="primary", use_container_width=True)
        
        with st.expander("🎨 Visualization Options"):
            st.checkbox("High Contrast Plots", key="sb_high_contrast_checkbox", help="Applies a high-contrast color theme to all charts.")
            st.checkbox("Enable 3D Worker View", key="sb_use_3d_distribution_checkbox", help="Renders worker positions in a 3D scatter plot.")
            st.checkbox("Show Debug Info", key="sb_debug_mode_checkbox", help="Display additional debug information.")
        
        with st.expander("💾 Data Management & Export"):
            load_data_button_sb = st.button("🔄 Load Previous Simulation", key="sb_load_data_main_button", use_container_width=True)
            
            can_export_data = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            
            # PDF Export Button
            if st.button("📄 Download Report (.tex)", key="sb_pdf_download_button", disabled=not can_export_data, use_container_width=True, help="Generates a LaTeX (.tex) file summarizing the simulation. Requires LaTeX to compile."):
                if can_export_data:
                    try:
                        sim_res_pdf = st.session_state.simulation_results
                        sim_cfg_pdf_export = sim_res_pdf.get('config_params', {})
                        num_steps_pdf_export = sim_cfg_pdf_export.get('SHIFT_DURATION_INTERVALS', 0)
                        mpi_pdf_export = sim_cfg_pdf_export.get('MINUTES_PER_INTERVAL', DEFAULT_CONFIG["MINUTES_PER_INTERVAL"])

                        if num_steps_pdf_export == 0: st.warning("⚠️ No simulation data (0 steps) for report."); raise SystemExit
                        
                        pdf_data_dict = {'step': list(range(num_steps_pdf_export)), 'time_minutes': [i * mpi_pdf_export for i in range(num_steps_pdf_export)]}
                        export_metrics_map_pdf = { # path_in_sim_res : col_name_in_df
                            'task_compliance.data': 'Task Compliance (%)',
                            'collaboration_metric.data': 'Collaboration Metric (%)', # Updated key
                            'operational_recovery': 'Operational Recovery (%)',
                            'worker_wellbeing.scores': 'Worker Wellbeing Index',
                            'psychological_safety': 'Psychological Safety Score',
                            'productivity_loss': 'Productivity Loss (%)',
                            'task_completion_rate': 'Task Completion Rate (%)',
                            'worker_wellbeing.team_cohesion_scores': 'Team Cohesion Score',
                            'worker_wellbeing.perceived_workload_scores': 'Perceived Workload (0-10)'
                        }
                        for path, col_name in export_metrics_map_pdf.items():
                            pdf_data_dict[col_name] = _prepare_timeseries_for_export(safe_get(sim_res_pdf, path, []), num_steps_pdf_export)
                        
                        raw_downtime_log_for_pdf = safe_get(sim_res_pdf, 'downtime_events_log', [])
                        pdf_data_dict['Downtime (min/interval)'] = aggregate_downtime_by_step(raw_downtime_log_for_pdf, num_steps_pdf_export)

                        df_for_report = pd.DataFrame(pdf_data_dict)
                        generate_pdf_report(df_for_report, sim_cfg_pdf_export) # Pass sim_config_params
                        st.success("✅ LaTeX report 'workplace_report.tex' generated.")
                    except SystemExit: pass # Cleanly exit if no data
                    except Exception as e_pdf: 
                        logger.error(f"PDF Generation Error: {e_pdf}", exc_info=True, extra={'user_action': 'PDF Generation Error'})
                        st.error(f"❌ PDF Generation Error: {e_pdf}")
            
            # CSV Export Button
            if can_export_data:
                sim_res_csv_exp = st.session_state.simulation_results
                sim_cfg_csv_exp = sim_res_csv_exp.get('config_params', {})
                num_steps_csv_exp = sim_cfg_csv_exp.get('SHIFT_DURATION_INTERVALS', 0)
                mpi_csv_exp = sim_cfg_csv_exp.get('MINUTES_PER_INTERVAL', DEFAULT_CONFIG["MINUTES_PER_INTERVAL"])

                if num_steps_csv_exp > 0:
                    csv_data_dict = {'step': list(range(num_steps_csv_exp)), 'time_minutes': [i * mpi_csv_exp for i in range(num_steps_csv_exp)]}
                    # Reuse map from PDF or define specific for CSV if needed
                    for path, col_name in export_metrics_map_pdf.items(): # Using same map as PDF
                        csv_data_dict[col_name.replace(' (%)','_percent').replace(' (0-10)','_0_10')] = _prepare_timeseries_for_export(safe_get(sim_res_csv_exp, path, []), num_steps_csv_exp)
                    
                    raw_downtime_log_for_csv = safe_get(sim_res_csv_exp, 'downtime_events_log', [])
                    csv_data_dict['downtime_minutes_per_interval'] = aggregate_downtime_by_step(raw_downtime_log_for_csv, num_steps_csv_exp)
                    
                    df_to_csv_export = pd.DataFrame(csv_data_dict)
                    st.download_button("📥 Download Data (CSV)", df_to_csv_export.to_csv(index=False).encode('utf-8'), 
                                      "workplace_summary.csv", "text/csv", key="sb_csv_download_button", use_container_width=True)
                else: st.caption("No detailed data to export (0 simulation steps).")
            elif not can_export_data : st.caption("Run simulation for export options.")
        
        if st.session_state.sb_debug_mode_checkbox:
            with st.expander("🛠️ Debug Information", expanded=False):
                st.write("**Default Config (Partial):**")
                st.json({k: DEFAULT_CONFIG.get(k) for k in ['MINUTES_PER_INTERVAL', 'WORK_AREAS', 'DEFAULT_SCHEDULED_EVENTS']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results: 
                    st.write("**Active Simulation Config (from results):**")
                    st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else: st.write("**No active simulation data.**")
        
        st.markdown("## 📋 Help & Info")
        if st.button("ℹ️ Help & Glossary", key="sb_help_button_main", use_container_width=True): 
            st.session_state.show_help_glossary = not st.session_state.get('show_help_glossary', False)
            st.rerun()
        if st.button("🚀 Quick Tour", key="sb_tour_button_main", use_container_width=True): 
            st.session_state.show_tour = not st.session_state.get('show_tour', False)
            st.rerun()
            
    return run_simulation_button_sb, load_data_button_sb
# --- SIMULATION LOGIC WRAPPER (CACHED) ---
@st.cache_data(ttl=3600, show_spinner="⚙️ Simulating workplace operations...")
def run_simulation_logic(team_size_sl, shift_duration_sl, scheduled_events_from_ui_sl, team_initiative_sl):
    # Start with a fresh copy of default config for each run
    config_sl = DEFAULT_CONFIG.copy()
    config_sl['TEAM_SIZE'] = int(team_size_sl) # Ensure int
    config_sl['SHIFT_DURATION_MINUTES'] = int(shift_duration_sl) # Ensure int
    
    mpi_sl = _get_config_value_sl(config_sl, {}, 'MINUTES_PER_INTERVAL', 2) # Use helper for this initial fetch
    if mpi_sl <= 0: mpi_sl = 2; logger.error("MPI was <=0 in config, used 2 for calculation.")
    config_sl['SHIFT_DURATION_INTERVALS'] = config_sl['SHIFT_DURATION_MINUTES'] // mpi_sl

    # Process events from UI: ensure 'step' is derived
    processed_events_sl = []
    for event_sl_ui_item in scheduled_events_from_ui_sl:
        evt_sl_item = event_sl_ui_item.copy()
        if 'step' not in evt_sl_item and 'Start Time (min)' in evt_sl_item:
            start_time_min_evt = _get_config_value_sl(evt_sl_item, {}, 'Start Time (min)', 0)
            evt_sl_item['step'] = int(start_time_min_evt // mpi_sl)
        processed_events_sl.append(evt_sl_item)
    config_sl['SCHEDULED_EVENTS'] = processed_events_sl
    
    # Worker Redistribution Logic
    if 'WORK_AREAS' in config_sl and isinstance(config_sl['WORK_AREAS'], dict) and config_sl['WORK_AREAS']:
        current_total_workers_cfg = sum(_get_config_value_sl(z,{},'workers',0) for z in config_sl['WORK_AREAS'].values() if isinstance(z,dict))
        target_team_size_sl = config_sl['TEAM_SIZE']

        if current_total_workers_cfg != target_team_size_sl and target_team_size_sl >= 0: # Allow team size 0
            logger.info(f"Redistributing workers. Config sum: {current_total_workers_cfg}, Target team: {target_team_size_sl}")
            work_areas_for_dist = {k:v for k,v in config_sl['WORK_AREAS'].items() if isinstance(v,dict) and not v.get('is_rest_area',False)}
            if not work_areas_for_dist : work_areas_for_dist = config_sl['WORK_AREAS'] # Fallback to all if no non-rest areas

            if target_team_size_sl == 0:
                for zone_k_sl_zero in work_areas_for_dist: config_sl['WORK_AREAS'][zone_k_sl_zero]['workers'] = 0
            elif current_total_workers_cfg > 0 and work_areas_for_dist: # Proportional redistribution
                ratio_sl = target_team_size_sl / current_total_workers_cfg
                accumulated_sl = 0
                sorted_zone_keys_sl = sorted(work_areas_for_dist.keys())
                for i_zone_sl, zone_k_sl in enumerate(sorted_zone_keys_sl):
                    original_workers = _get_config_value_sl(config_sl['WORK_AREAS'][zone_k_sl], {}, 'workers', 0)
                    if i_zone_sl < len(sorted_zone_keys_sl) - 1:
                        new_w_sl = int(round(original_workers * ratio_sl))
                        config_sl['WORK_AREAS'][zone_k_sl]['workers'] = new_w_sl
                        accumulated_sl += new_w_sl
                    else: # Last zone gets remainder to ensure sum matches target_team_size_sl
                        config_sl['WORK_AREAS'][zone_k_sl]['workers'] = max(0, target_team_size_sl - accumulated_sl)
            elif work_areas_for_dist : # Distribute evenly if no workers configured initially or target_team_size_sl is 0
                num_dist_zones = len(work_areas_for_dist)
                base_w_sl, rem_w_sl = divmod(target_team_size_sl, num_dist_zones)
                assign_count_sl = 0
                for zone_k_sl_even, zone_data_sl_even in work_areas_for_dist.items():
                    zone_data_sl_even['workers'] = base_w_sl + (1 if assign_count_sl < rem_w_sl else 0)
                    assign_count_sl +=1
            # Ensure rest areas always have 0 workers unless explicitly set otherwise for some reason
            for zone_k_sl_rest, zone_data_sl_rest in config_sl['WORK_AREAS'].items():
                if isinstance(zone_data_sl_rest, dict) and zone_data_sl_rest.get('is_rest_area'):
                    # If rest area workers were not part of initial sum, set to 0 unless TEAM_SIZE is very small
                    # This logic might need refinement based on how 'is_rest_area' workers are meant to be handled.
                    # For now, assume they are not primary work assignments unless explicitly part of team_size distribution.
                     if zone_k_sl_rest not in work_areas_for_dist: # If it wasn't part of the distribution pool
                         zone_data_sl_rest['workers'] = 0


    validate_config(config_sl) # Validate the final config to be used by simulation
    
    logger.info(f"Running simulation: Team={config_sl['TEAM_SIZE']}, Duration={config_sl['SHIFT_DURATION_MINUTES']}min ({config_sl['SHIFT_DURATION_INTERVALS']} intervals), Events={len(config_sl['SCHEDULED_EVENTS'])}, Initiative={team_initiative_sl}", extra={'user_action': 'Run Simulation - Start'})
    
    expected_keys_sl = [
        'team_positions_df', 'task_compliance', 'collaboration_metric',
        'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 
        'worker_wellbeing', 'psychological_safety', 'feedback_impact', 
        'downtime_events_log', 'task_completion_rate'
    ]
    sim_results_tuple_sl_run = simulate_workplace_operations(
        num_team_members=config_sl['TEAM_SIZE'],
        num_steps=config_sl['SHIFT_DURATION_INTERVALS'],
        scheduled_events=config_sl['SCHEDULED_EVENTS'], 
        team_initiative=team_initiative_sl,
        config=config_sl
    )
    
    if not isinstance(sim_results_tuple_sl_run, tuple) or len(sim_results_tuple_sl_run) != len(expected_keys_sl):
        logger.critical("Simulation returned unexpected data format or length.", extra={'user_action':'Sim Format Error'})
        raise TypeError("Simulation returned unexpected data format.")
        
    simulation_output_dict_sl_final = dict(zip(expected_keys_sl, sim_results_tuple_sl_run))
    
    # Store key config parameters with the results for reproducibility and context
    simulation_output_dict_sl_final['config_params'] = {
        'TEAM_SIZE': config_sl['TEAM_SIZE'], 
        'SHIFT_DURATION_MINUTES': config_sl['SHIFT_DURATION_MINUTES'],
        'SHIFT_DURATION_INTERVALS': config_sl['SHIFT_DURATION_INTERVALS'],
        'MINUTES_PER_INTERVAL': mpi_sl, # Crucial for interpreting steps vs time
        'SCHEDULED_EVENTS': config_sl['SCHEDULED_EVENTS'], # The actual events used
        'TEAM_INITIATIVE': team_initiative_sl,
        'WORK_AREAS_EFFECTIVE': config_sl.get('WORK_AREAS', {}).copy() # The work areas used
    }
    
    # Derive disruption event steps from the *actual* scheduled events used for THIS run
    disruption_steps_final_sl = [evt.get('step') for evt in config_sl['SCHEDULED_EVENTS'] if isinstance(evt,dict) and "Disruption" in evt.get("Event Type","") and isinstance(evt.get('step'),int)]
    simulation_output_dict_sl_final['config_params']['DISRUPTION_EVENT_STEPS'] = sorted(list(set(disruption_steps_final_sl)))

    save_simulation_data(simulation_output_dict_sl_final) 
    return simulation_output_dict_sl_final

def _get_config_value_sl(primary_conf, secondary_conf, key, default): # Helper for run_simulation_logic context
    return secondary_conf.get(key, primary_conf.get(key, default))


# --- TIME RANGE INPUT WIDGETS ---
def time_range_input_section(tab_key_prefix: str, max_minutes_for_range_ui: int, st_col_obj = st, interval_duration_min_ui: int = 2):
    start_time_key_ui = f"{tab_key_prefix}_start_time_min"
    end_time_key_ui = f"{tab_key_prefix}_end_time_min"
    if interval_duration_min_ui <=0: interval_duration_min_ui = 2 

    # Ensure session state keys exist and are valid before rendering widgets
    current_start = st.session_state.get(start_time_key_ui, 0)
    current_end = st.session_state.get(end_time_key_ui, max_minutes_for_range_ui)
    
    current_start = max(0, min(current_start, max_minutes_for_range_ui))
    current_end = max(current_start, min(current_end, max_minutes_for_range_ui))
    
    # Update session state if clamping changed values, to ensure widgets show correct clamped value
    st.session_state[start_time_key_ui] = current_start
    st.session_state[end_time_key_ui] = current_end
    
    prev_start_ui = current_start
    prev_end_ui = current_end

    cols_ui = st_col_obj.columns(2)
    
    # Use unique keys for widgets to avoid conflicts if this function is called multiple times
    # The session state keys (start_time_key_ui, end_time_key_ui) are the source of truth.
    new_start_time_ui = cols_ui[0].number_input( "Start Time (min)", min_value=0, max_value=max_minutes_for_range_ui, 
        value=current_start, step=interval_duration_min_ui, key=f"widget_num_{start_time_key_ui}", 
        help=f"Range: 0 to {max_minutes_for_range_ui} min.")
    
    # Update session state immediately after widget interaction
    st.session_state[start_time_key_ui] = new_start_time_ui 
    
    end_time_min_widget_val_ui = st.session_state[start_time_key_ui] # Min for end_time is current start_time
    new_end_time_ui = cols_ui[1].number_input("End Time (min)", min_value=end_time_min_widget_val_ui, max_value=max_minutes_for_range_ui, 
        value=current_end, step=interval_duration_min_ui, key=f"widget_num_{end_time_key_ui}",
        help=f"Range: {end_time_min_widget_val_ui} to {max_minutes_for_range_ui} min.")
    st.session_state[end_time_key_ui] = new_end_time_ui

    # Final clamping and check for rerun
    if st.session_state[end_time_key_ui] < st.session_state[start_time_key_ui]:
        st.session_state[end_time_key_ui] = st.session_state[start_time_key_ui]
    
    if prev_start_ui != st.session_state[start_time_key_ui] or prev_end_ui != st.session_state[end_time_key_ui]:
        st.rerun()

    return int(st.session_state[start_time_key_ui]), int(st.session_state[end_time_key_ui])
# --- MAIN APPLICATION FUNCTION ---
def main():
    st.title("Workplace Shift Optimization Dashboard")
    
    # Centralized session state initialization
    mpi_global_app = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)
    if mpi_global_app <= 0: mpi_global_app = 2 # Safety check
    
    app_state_defaults_app = {
        'simulation_results': None, 'show_tour': False, 'show_help_glossary': False,
        'sb_team_size_num': DEFAULT_CONFIG['TEAM_SIZE'],
        'sb_shift_duration_num': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
        'sb_scheduled_events_list': list(DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', [])), # Deep copy for lists/dicts
        'sb_team_initiative_selectbox': "Standard Operations",
        'sb_high_contrast_checkbox': False, 
        'sb_use_3d_distribution_checkbox': False, 
        'sb_debug_mode_checkbox': False,
        'form_event_type': "Major Disruption", # Default for sidebar event adder
        'form_event_start': 0, 
        'form_event_duration': max(mpi_global_app, 10), # Sensible default duration for event adder
    }
    # Default time ranges for plot tabs - these will be updated based on sim duration
    default_max_mins_app = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - mpi_global_app if DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] > mpi_global_app else 0
    for prefix_app in ['op', 'ww', 'dt']: # Operational, WorkerWellbeing, DownTime tabs
        app_state_defaults_app[f'{prefix_app}_start_time_min'] = 0
        app_state_defaults_app[f'{prefix_app}_end_time_min'] = default_max_mins_app

    for key_app, val_app in app_state_defaults_app.items():
        if key_app not in st.session_state:
            st.session_state[key_app] = val_app
            
    # Render sidebar and get button states
    run_simulation_button_main, load_data_button_main = render_settings_sidebar()
    
    # Get current visualization settings from session state (updated by sidebar checkboxes)
    current_high_contrast_main_app = st.session_state.sb_high_contrast_checkbox
    use_3d_main_app = st.session_state.sb_use_3d_distribution_checkbox

    # Determine active MINUTES_PER_INTERVAL and max_minutes for UI time range inputs
    active_mpi_main_app = mpi_global_app
    max_mins_ui_main_app = default_max_mins_app
    simulation_disruption_steps_absolute = [] # Absolute step numbers of disruptions

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        sim_cfg_main_app = st.session_state.simulation_results.get('config_params', {})
        active_mpi_main_app = sim_cfg_main_app.get('MINUTES_PER_INTERVAL', mpi_global_app)
        if active_mpi_main_app <= 0 : active_mpi_main_app = 2 # Safety
        sim_intervals_main_app = sim_cfg_main_app.get('SHIFT_DURATION_INTERVALS', 0)
        max_mins_ui_main_app = max(0, sim_intervals_main_app * active_mpi_main_app - active_mpi_main_app) if sim_intervals_main_app > 0 else 0
        simulation_disruption_steps_absolute = sim_cfg_main_app.get('DISRUPTION_EVENT_STEPS', [])
    else: # No simulation results, derive from current sidebar settings
        shift_duration_from_sidebar = st.session_state.sb_shift_duration_num
        sim_intervals_main_app = shift_duration_from_sidebar // active_mpi_main_app if active_mpi_main_app > 0 else 0
        max_mins_ui_main_app = max(0, sim_intervals_main_app * active_mpi_main_app - active_mpi_main_app) if sim_intervals_main_app > 0 else 0
        for event_main_ui_item in st.session_state.sb_scheduled_events_list: # Use current list from sidebar
            if "Disruption" in event_main_ui_item.get("Event Type","") and isinstance(event_main_ui_item.get("Start Time (min)"), (int,float)):
                simulation_disruption_steps_absolute.append(int(event_main_ui_item["Start Time (min)"] // active_mpi_main_app))
        simulation_disruption_steps_absolute = sorted(list(set(simulation_disruption_steps_absolute)))
    
    # Ensure UI time range selectors are clamped to the current max_mins_ui_main_app
    for prefix_main_ui_clamp in ['op', 'ww', 'dt']:
        st.session_state[f"{prefix_main_ui_clamp}_start_time_min"] = max(0, min(st.session_state.get(f"{prefix_main_ui_clamp}_start_time_min",0), max_mins_ui_main_app))
        st.session_state[f"{prefix_main_ui_clamp}_end_time_min"] = max(st.session_state[f"{prefix_main_ui_clamp}_start_time_min"], min(st.session_state.get(f"{prefix_main_ui_clamp}_end_time_min",max_mins_ui_main_app), max_mins_ui_main_app))


    # --- Simulation Run & Load Logic ---
    if run_simulation_button_main:
        with st.spinner("🚀 Simulating workplace operations... This may take a moment."):
            try:
                # Pass current values from session_state to the simulation logic
                simulation_results_run = run_simulation_logic(
                    st.session_state.sb_team_size_num, 
                    st.session_state.sb_shift_duration_num, 
                    list(st.session_state.sb_scheduled_events_list), # Pass a copy
                    st.session_state.sb_team_initiative_selectbox
                )
                st.session_state.simulation_results = simulation_results_run
                
                # Update UI time range selectors based on the new simulation's duration
                new_sim_cfg_run = simulation_results_run['config_params']
                new_mpi_run = new_sim_cfg_run.get('MINUTES_PER_INTERVAL', 2)
                new_sim_intervals_run = new_sim_cfg_run.get('SHIFT_DURATION_INTERVALS',0)
                new_max_mins_run = max(0, new_sim_intervals_run * new_mpi_run - new_mpi_run) if new_sim_intervals_run > 0 else 0
                
                for prefix_run_ui in ['op', 'ww', 'dt']: # Reset time ranges for new simulation
                    st.session_state[f"{prefix_run_ui}_start_time_min"] = 0
                    st.session_state[f"{prefix_run_ui}_end_time_min"] = new_max_mins_run

                st.success("✅ Simulation completed successfully!")
                logger.info("Simulation run successful via UI button.", extra={'user_action': 'Run Simulation - Success'})
                st.rerun() # Rerun to update plots with new data and clamped time ranges
            except Exception as e_run_main:
                logger.error(f"Simulation Run Error from UI: {e_run_main}", exc_info=True, extra={'user_action': 'Run Simulation - Error'})
                st.error(f"❌ Simulation failed: {str(e_run_main)}") 
                st.session_state.simulation_results = None # Clear results on failure

    if load_data_button_main:
        with st.spinner("🔄 Loading saved simulation data..."):
            try:
                loaded_data_main = load_simulation_data()
                if loaded_data_main and isinstance(loaded_data_main, dict) and 'config_params' in loaded_data_main:
                    st.session_state.simulation_results = loaded_data_main
                    cfg_loaded_main = loaded_data_main['config_params']
                    
                    # Update sidebar controls to reflect loaded simulation's parameters
                    st.session_state.sb_team_size_num = cfg_loaded_main.get('TEAM_SIZE', DEFAULT_CONFIG['TEAM_SIZE'])
                    st.session_state.sb_shift_duration_num = cfg_loaded_main.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'])
                    st.session_state.sb_scheduled_events_list = list(cfg_loaded_main.get('SCHEDULED_EVENTS', [])) # Deep copy
                    st.session_state.sb_team_initiative_selectbox = cfg_loaded_main.get('TEAM_INITIATIVE', "Standard Operations")
                     
                    # Update UI time range selectors based on loaded simulation's duration
                    loaded_mpi_main = cfg_loaded_main.get('MINUTES_PER_INTERVAL', 2)
                    loaded_sim_intervals_main = cfg_loaded_main.get('SHIFT_DURATION_INTERVALS', 0)
                    new_max_mins_load_main = max(0, loaded_sim_intervals_main * loaded_mpi_main - loaded_mpi_main) if loaded_sim_intervals_main > 0 else 0
                    
                    for prefix_load_ui in ['op', 'ww', 'dt']: # Reset time ranges for loaded simulation
                        st.session_state[f"{prefix_load_ui}_start_time_min"] = 0
                        st.session_state[f"{prefix_load_ui}_end_time_min"] = new_max_mins_load_main
                    
                    st.success("✅ Data loaded successfully!")
                    logger.info("Saved data loaded successfully via UI button.", extra={'user_action': 'Load Data - Success'})
                    st.rerun() # Rerun to reflect loaded data and updated controls
                else:
                    st.error("❌ Failed to load data or data is incomplete/invalid (e.g., missing 'config_params').")
                    logger.warning("Load data failed or invalid format from UI button.", extra={'user_action': 'Load Data - Fail/Invalid'})
                    st.session_state.simulation_results = None # Clear if load fails
            except Exception as e_load_main:
                logger.error(f"Load Data Error from UI: {e_load_main}", exc_info=True, extra={'user_action': 'Load Data - Error'})
                st.error(f"❌ Failed to load data: {e_load_main}")
                st.session_state.simulation_results = None
    
    # --- Modals for Tour & Help ---
    if st.session_state.get('show_tour', False): 
        with st.container():
             st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3><p>Welcome! This dashboard helps you monitor and analyze workplace shift operations. Use the sidebar to configure simulations and navigate. The main area displays results across several tabs: Overview, Operational Metrics, Worker Well-being (including psychosocial factors and spatial dynamics), Downtime Analysis, and a Glossary. Interactive charts and actionable insights will guide you in optimizing operations.</p><p>Start by running a new simulation or loading previous data from the sidebar!</p></div>""", unsafe_allow_html=True)
        if st.button("Got it!", key="tour_modal_close_button_main_area"): 
            st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help_glossary', False): 
        with st.container():
            st.markdown(""" <div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3> <p>This dashboard provides insights into simulated workplace operations. Use the sidebar to configure and run simulations or load previously saved data. Navigate through the analysis using the main tabs above.</p><h4>Metric Definitions:</h4> <ul style="font-size: 0.85rem; list-style-type: disc; padding-left: 20px;"> <li><b>Task Compliance Score:</b> Percentage of tasks completed correctly and on time.</li><li><b>Collaboration Metric:</b> A score indicating teamwork potential and interaction levels.</li><li><b>Operational Recovery Score:</b> Ability to maintain output after disruptions.</li><li><b>Worker Well-Being Index:</b> Composite score of fatigue, stress, and satisfaction.</li><li><b>Psychological Safety Score:</b> Comfort level in reporting issues or suggesting improvements.</li><li><b>Team Cohesion Index:</b> Measure of bonds and sense of belonging within a team.</li><li><b>Perceived Workload Index:</b> Indicator of task demand (0-10 scale).</li><li><b>Uptime:</b> Percentage of time equipment is operational.</li><li><b>Throughput:</b> Percentage of maximum production rate achieved.</li><li><b>Quality Rate:</b> Percentage of products meeting quality standards.</li><li><b>OEE (Overall Equipment Effectiveness):</b> Combined score of Uptime, Throughput, and Quality.</li><li><b>Productivity Loss:</b> Percentage of potential output lost.</li><li><b>Downtime Events Log:</b> A raw log of individual downtime occurrences, each with step, duration, and cause. Aggregated for trend plots.</li><li><b>Task Completion Rate:</b> Percentage of tasks completed per time interval.</li></ul><p>For further assistance, refer to documentation or contact support.</p></div> """, unsafe_allow_html=True) 
        if st.button("Understood", key="help_modal_close_button_main_area"): 
            st.session_state.show_help_glossary = False; st.rerun()

    # --- MAIN CONTENT TABS ---
    tab_names_main_ui = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Well-being", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs_streamlit_objs_main = st.tabs(tab_names_main_ui)
    plot_config_interactive_main_ui = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}} # Keep zoom/pan
    plot_config_minimal_main_ui = {'displayModeBar': False}

    # --- Overview Tab ---
    with tabs_streamlit_objs_main[0]:
        st.header("📊 Key Performance Indicators & Actionable Insights", divider=COLOR_ACCENT_INDIGO_CSS)
        if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
            sim_data_overview = st.session_state.simulation_results
            # Effective config for this tab comes from the simulation results if available
            sim_cfg_overview = sim_data_overview.get('config_params', DEFAULT_CONFIG)
            effective_config_overview = {**DEFAULT_CONFIG, **sim_cfg_overview}


            target_compliance_ov = float(effective_config_overview.get('TARGET_COMPLIANCE', 75.0))
            target_collab_ov = float(effective_config_overview.get('TARGET_COLLABORATION', 65.0)) # For collaboration_metric
            target_wellbeing_ov = float(effective_config_overview.get('TARGET_WELLBEING', 75.0))
            
            downtime_log_ov = safe_get(sim_data_overview, 'downtime_events_log', [])
            downtime_durations_ov = [evt.get('duration', 0.0) for evt in downtime_log_ov if isinstance(evt, dict)]
            
            compliance_val_ov = safe_stat(safe_get(sim_data_overview, 'task_compliance.data', []), np.mean, 0.0)
            collab_val_ov = safe_stat(safe_get(sim_data_overview, 'collaboration_metric.data', []), np.mean, 0.0) # Updated key
            wellbeing_val_ov = safe_stat(safe_get(sim_data_overview, 'worker_wellbeing.scores', []), np.mean, 0.0)
            downtime_total_ov = sum(downtime_durations_ov)

            shift_duration_ov = float(effective_config_overview.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
            dt_target_percent_ov = float(effective_config_overview.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT', 0.05))
            dt_target_abs_ov = shift_duration_ov * dt_target_percent_ov
            
            cols_metrics_ov = st.columns(4)
            cols_metrics_ov[0].metric("Task Compliance", f"{compliance_val_ov:.1f}%", f"{compliance_val_ov-target_compliance_ov:.1f}% vs Target")
            cols_metrics_ov[1].metric("Collaboration Metric", f"{collab_val_ov:.1f}%", f"{collab_val_ov-target_collab_ov:.1f}% vs Target")
            cols_metrics_ov[2].metric("Worker Well-Being", f"{wellbeing_val_ov:.1f}%", f"{wellbeing_val_ov-target_wellbeing_ov:.1f}% vs Target")
            cols_metrics_ov[3].metric("Total Downtime", f"{downtime_total_ov:.1f} min", f"{downtime_total_ov-dt_target_abs_ov:.1f} min vs Target ({dt_target_abs_ov:.0f}min)", delta_color="inverse")
            
            try:
                summary_figs_ov = plot_key_metrics_summary(
                    compliance=compliance_val_ov, proximity=collab_val_ov, wellbeing=wellbeing_val_ov, 
                    downtime=downtime_total_ov, target_compliance=target_compliance_ov, 
                    target_proximity=target_collab_ov, target_wellbeing=target_wellbeing_ov, 
                    target_downtime=dt_target_abs_ov, high_contrast=current_high_contrast_main_app
                ) # Visualizations.py handles its own internal semantic colors
                if summary_figs_ov and isinstance(summary_figs_ov, list):
                    cols_gauges_ov = st.columns(min(len(summary_figs_ov), 4) or 1)
                    for i_gauge_ov, fig_gauge_ov in enumerate(summary_figs_ov): 
                        if fig_gauge_ov: cols_gauges_ov[i_gauge_ov % len(cols_gauges_ov)].plotly_chart(fig_gauge_ov, use_container_width=True, config=plot_cfg_minimal_main_ui)
                else: st.caption("Overview gauge charts could not be generated.")
            except Exception as e_gauge_ov: 
                logger.error(f"Overview Gauges Plot Error: {e_gauge_ov}", exc_info=True); st.error(f"⚠️ Error rendering gauges: {e_gauge_ov}")

            st.markdown("---"); st.subheader("💡 Key Insights & Leadership Actions")
            actionable_insights_ov = get_actionable_insights(sim_data_overview, effective_config_overview) # Pass effective config
            if actionable_insights_ov:
                for insight in actionable_insights_ov: 
                    st.markdown(f'<div class="alert-{insight["type"]}"><p class="insight-title">{insight["title"]}</p><p class="insight-text">{insight["text"]}</p></div>', unsafe_allow_html=True)
            else: st.info("✅ No critical alerts or specific insights identified based on current thresholds.", icon="👍")
            
            with st.expander("View Detailed Overview Data Table", expanded=False):
                num_steps_ov_table = effective_config_overview.get('SHIFT_DURATION_INTERVALS', 0)
                mpi_ov_table = effective_config_overview.get('MINUTES_PER_INTERVAL', mpi_global_app)

                if num_steps_ov_table > 0:
                    df_data_ov_table = {'Time (min)': [i * mpi_ov_table for i in range(num_steps_ov_table)]}
                    df_data_ov_table['Task Compliance (%)'] = _prepare_timeseries_for_export(safe_get(sim_data_overview, 'task_compliance.data', []), num_steps_ov_table)
                    df_data_ov_table['Collaboration Metric (%)'] = _prepare_timeseries_for_export(safe_get(sim_data_overview, 'collaboration_metric.data', []), num_steps_ov_table) # Updated
                    df_data_ov_table['Well-Being (%)'] = _prepare_timeseries_for_export(safe_get(sim_data_overview, 'worker_wellbeing.scores', []), num_steps_ov_table)
                    
                    downtime_log_ov_table = safe_get(sim_data_overview, 'downtime_events_log', [])
                    df_data_ov_table['Downtime (min/interval)'] = aggregate_downtime_by_step(downtime_log_ov_table, num_steps_ov_table)
                    
                    st.dataframe(pd.DataFrame(df_data_ov_table).style.format("{:.1f}", na_rep="-").set_table_styles([{'selector': 'th', 'props': [('background-color', '#293344'), ('color', '#EAEAEA')]}]), use_container_width=True, height=300)
                else: st.caption("No detailed overview data available (0 simulation steps).")
        else: st.info("ℹ️ Run a simulation or load data to view the Overview & Insights.", icon="📊")
    
    # --- Tab Definitions (Ensure data_paths and plot function argument names are correct) ---
    # (Full tab_configs_main_app definitions as in previous combined response, ensure correctness)
    # For brevity, insights_html will be simple strings here
    tab_configs_main_app = [
        {"name": "📈 Operational Metrics", "key_prefix": "op", 
         "plots": [
             {"title": "Task Compliance Score", "data_path": "task_compliance.data", "plot_func": plot_task_compliance_score, "extra_args_paths": {"forecast_data": "task_compliance.forecast", "z_scores": "task_compliance.z_scores"}},
             {"title": "Collaboration Metric", "data_path": "collaboration_metric.data", "plot_func": plot_collaboration_proximity_index, "extra_args_paths": {"forecast_data": "collaboration_metric.forecast"}},
             {"is_subheader": True, "title": "Additional Operational Metrics"}, 
             {"title": "Operational Resilience", "data_path": "operational_recovery", "plot_func": plot_operational_recovery, "extra_args_paths": {"productivity_loss_data": "productivity_loss"}},
             {"title": "OEE & Components", "is_oee": True} 
         ], "insights_html": "<p>Review operational trends and OEE components.</p>" },
        {"name": "👥 Worker Well-being", "key_prefix": "ww", 
         "plots": [
             {"is_subheader": True, "title": "Psychosocial & Well-being Indicators"},
             {"title": "Worker Well-Being Index", "data_path": "worker_wellbeing.scores", "plot_func": plot_worker_wellbeing, "extra_args_paths": {"triggers": "worker_wellbeing.triggers"}},
             {"title": "Psychological Safety Score", "data_path": "psychological_safety", "plot_func": plot_psychological_safety},
             {"title": "Team Cohesion Index", "data_path": "worker_wellbeing.team_cohesion_scores", "plot_func": plot_team_cohesion},
             {"title": "Perceived Workload Index (0-10)", "data_path": "worker_wellbeing.perceived_workload_scores", "plot_func": plot_perceived_workload, "extra_args_fixed": {"high_workload_threshold": DEFAULT_CONFIG['PERCEIVED_WORKLOAD_THRESHOLD_HIGH'], "very_high_workload_threshold": DEFAULT_CONFIG['PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH']}},
             {"is_subheader": True, "title": "Spatial Dynamics Analysis", "is_spatial": True}
         ], "dynamic_insights_func": "render_wellbeing_alerts", "insights_html": "<p>Monitor psychosocial factors and worker distribution.</p>" },
        {"name": "⏱️ Downtime Analysis", "key_prefix": "dt", "metrics_display": True, 
         "plots": [
            {"title": "Downtime Trend (per Interval)", "data_path": "downtime_events_log", "plot_func": plot_downtime_trend, "is_event_based_aggregation": True, "extra_args_fixed": {"interval_threshold_minutes": DEFAULT_CONFIG['DOWNTIME_PLOT_ALERT_THRESHOLD']}}, # Changed arg name
            {"title": "Downtime Distribution by Cause", "data_path": "downtime_events_log", "plot_func": plot_downtime_causes_pie, "is_event_based_filtering": True}
         ], "insights_html": "<p>Analyze downtime causes and trends to improve uptime.</p>" }
    ]

    # --- Tab Rendering Loop ---
    for i_tab_main_app, tab_def_main_app in enumerate(tab_configs_main_app):
        with tabs_streamlit_objs_main[i_tab_main_app+1]: # Start from index 1 (after Overview)
            st.header(tab_def_main_app["name"], divider=COLOR_ACCENT_INDIGO_CSS)
            if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
                sim_data_main_tab_loop = st.session_state.simulation_results
                sim_cfg_main_tab_active = sim_data_main_tab_loop.get('config_params', {}) # Config for this specific sim run

                st.markdown("##### Select Time Range for Plots:")
                start_time_ui_tab, end_time_ui_tab = time_range_input_section(
                    tab_def_main_app["key_prefix"], max_mins_ui_main_app, # Use max_mins_ui_main_app calculated above
                    interval_duration_min_ui=active_mpi_main_app # Use active_mpi_main_app
                )
                # Convert time (minutes) to step indices for slicing
                start_idx_tab_ui = start_time_ui_tab // active_mpi_main_app if active_mpi_main_app > 0 else 0
                end_idx_tab_ui = (end_time_ui_tab // active_mpi_main_app) + 1 if active_mpi_main_app > 0 else 0 # end_idx is exclusive for slicing
                
                # Filter absolute disruption steps for the current tab's selected time window
                disrupt_steps_for_plots_abs_tab = [s for s in simulation_disruption_steps_absolute if start_idx_tab_ui <= s < end_idx_tab_ui]

                if tab_def_main_app.get("metrics_display"): # For Downtime tab summary metrics
                    downtime_log_tab_metrics = safe_get(sim_data_main_tab_loop, 'downtime_events_log', [])
                    downtime_events_in_range_tab = [evt for evt in downtime_log_tab_metrics if isinstance(evt, dict) and start_idx_tab_ui <= evt.get('step', -1) < end_idx_tab_ui]
                    downtime_durations_in_range_tab = [evt.get('duration',0.0) for evt in downtime_events_in_range_tab]
                    
                    if downtime_events_in_range_tab: 
                        total_dt_period = sum(downtime_durations_in_range_tab); num_incidents = len([d for d in downtime_durations_in_range_tab if d > 0])
                        avg_dur_incident = total_dt_period / num_incidents if num_incidents > 0 else 0.0
                        dt_cols = st.columns(3)
                        dt_cols[0].metric("Total Downtime in Period", f"{total_dt_period:.1f} min")
                        dt_cols[1].metric("Number of Incidents", f"{num_incidents}")
                        dt_cols[2].metric("Avg. Duration / Incident", f"{avg_dur_incident:.1f} min")

                plot_col_container_tab = st.container() 
                num_plots_in_row_tab = 0

                for plot_cfg_tab_item in tab_def_main_app["plots"]:
                    if plot_cfg_tab_item.get("is_subheader"):
                        st.subheader(plot_cfg_tab_item["title"]) 
                        if plot_cfg_tab_item.get("is_spatial"):
                            # Construct facility_config for spatial plots
                            facility_config_spatial_tab = {
                                'FACILITY_SIZE': sim_cfg_main_tab_active.get('FACILITY_SIZE', DEFAULT_CONFIG['FACILITY_SIZE']),
                                'WORK_AREAS': sim_cfg_main_tab_active.get('WORK_AREAS_EFFECTIVE', DEFAULT_CONFIG['WORK_AREAS']),
                                'ENTRY_EXIT_POINTS': sim_cfg_main_tab_active.get('ENTRY_EXIT_POINTS', DEFAULT_CONFIG['ENTRY_EXIT_POINTS']),
                                'MINUTES_PER_INTERVAL': active_mpi_main_app # Pass MPI for titles in plot func
                            }
                            # ... (Full spatial plot rendering logic as in previous combined file) ...
                            pass # Placeholder for brevity
                        num_plots_in_row_tab = 0; continue # Reset for next plot row

                    if num_plots_in_row_tab == 0: plot_columns_tab = plot_col_container_tab.columns(2)
                    current_plot_col_tab = plot_columns_tab[num_plots_in_row_tab % 2]
                    
                    with current_plot_col_tab:
                        st.markdown(f"<h6>{plot_cfg_tab_item['title']}</h6>", unsafe_allow_html=True) # Plot title
                        with st.container(border=True):
                            plot_data_final_tab = None; plot_kwargs_tab = {"high_contrast": current_high_contrast_main_app}
                            try:
                                if plot_cfg_tab_item.get("is_oee"):
                                    # ... (OEE plotting logic as in previous file, ensuring disruption_points are relative)
                                    # disruption_points_oee = [s - start_idx_tab_ui for s in disrupt_steps_for_plots_abs_tab if s - start_idx_tab_ui >=0]
                                    # plot_kwargs_tab["disruption_points"] = disruption_points_oee
                                    pass # Placeholder for brevity
                                else: # Generic plot data preparation
                                    raw_plot_data_tab = safe_get(sim_data_main_tab_loop, plot_cfg_tab_item["data_path"], [])
                                    
                                    if "extra_args_paths" in plot_cfg_tab_item:
                                        for arg_n, arg_p in plot_cfg_tab_item["extra_args_paths"].items():
                                            extra_d = safe_get(sim_data_main_tab_loop, arg_p, [])
                                            if plot_cfg_tab_item["plot_func"] == plot_worker_wellbeing and arg_n == "triggers":
                                                # Filter & relativize triggers for plot_worker_wellbeing
                                                filt_trigs = {}
                                                if isinstance(extra_d, dict):
                                                    for tr_type, tr_steps_abs in extra_d.items():
                                                        if tr_type == 'work_area' and isinstance(tr_steps_abs, dict):
                                                            filt_trigs[tr_type] = {zn: [s - start_idx_tab_ui for s in (s_list if isinstance(s_list,list) else []) if start_idx_tab_ui <= s < end_idx_tab_ui and s - start_idx_tab_ui >=0] for zn,s_list in tr_steps_abs.items()}
                                                            filt_trigs[tr_type] = {k:v for k,v in filt_trigs[tr_type].items() if v} # Remove empty zone lists
                                                        elif isinstance(tr_steps_abs, list):
                                                            filt_trigs[tr_type] = [s - start_idx_tab_ui for s in tr_steps_abs if start_idx_tab_ui <= s < end_idx_tab_ui and s - start_idx_tab_ui >=0]
                                                plot_kwargs_tab[arg_n] = filt_trigs
                                            elif isinstance(extra_d, list): plot_kwargs_tab[arg_n] = extra_d[start_idx_tab_ui:min(end_idx_tab_ui, len(extra_d))] if start_idx_tab_ui < len(extra_d) else []
                                            else: plot_kwargs_tab[arg_n] = extra_d # Pass DFs as is
                                    if "extra_args_fixed" in plot_cfg_tab_item: plot_kwargs_tab.update(plot_cfg_tab_item["extra_args_fixed"])

                                    if "disruption_points" in plot_cfg_tab_item["plot_func"].__code__.co_varnames:
                                        plot_kwargs_tab["disruption_points"] = [s - start_idx_tab_ui for s in disrupt_steps_for_plots_abs_tab if s - start_idx_tab_ui >=0]

                                    if plot_cfg_tab_item.get("is_event_based_aggregation"): # For plot_downtime_trend
                                        num_steps_in_range_agg = end_idx_tab_ui - start_idx_tab_ui
                                        aggregated_data_agg = [0.0] * num_steps_in_range_agg if num_steps_in_range_agg > 0 else []
                                        for evt_agg in raw_plot_data_tab: # raw_plot_data_tab is downtime_events_log
                                            if isinstance(evt_agg,dict) and start_idx_tab_ui <= evt_agg.get('step',-1) < end_idx_tab_ui:
                                                rel_step_agg = evt_agg['step'] - start_idx_tab_ui
                                                if 0 <= rel_step_agg < num_steps_in_range_agg: aggregated_data_agg[rel_step_agg] += evt_agg.get('duration',0)
                                        plot_data_final_tab = aggregated_data_agg
                                    elif plot_cfg_tab_item.get("is_event_based_filtering"): # For plot_downtime_causes_pie
                                        plot_data_final_tab = [evt_filt for evt_filt in raw_plot_data_tab if isinstance(evt_filt,dict) and start_idx_tab_ui <= evt_filt.get('step',-1) < end_idx_tab_ui]
                                        if "disruption_points" in plot_kwargs_tab: del plot_kwargs_tab["disruption_points"] # Pie doesn't use them
                                    elif isinstance(raw_plot_data_tab, list):
                                        plot_data_final_tab = raw_plot_data_tab[start_idx_tab_ui:min(end_idx_tab_ui, len(raw_plot_data_tab))] if start_idx_tab_ui < len(raw_plot_data_tab) else []
                                    elif isinstance(raw_plot_data_tab, pd.DataFrame):
                                        plot_data_final_tab = _slice_dataframe_by_step_indices(raw_plot_data_tab, start_idx_tab_ui, end_idx_tab_ui)
                                
                                # Render plot if data exists
                                data_exists_for_plot = False
                                if isinstance(plot_data_final_tab, (list, pd.Series)) and len(plot_data_final_tab) > 0: data_exists_for_plot = True
                                elif isinstance(plot_data_final_tab, pd.DataFrame) and not plot_data_final_tab.empty: data_exists_for_plot = True
                                elif plot_cfg_tab_item.get("is_event_based_filtering") and isinstance(plot_data_final_tab, list): data_exists_for_plot = True # Plot even if list is empty for pie

                                if data_exists_for_plot:
                                    fig_obj_main = plot_cfg_tab_item["plot_func"](plot_data_final_tab, **plot_kwargs_tab)
                                    if fig_obj_main: st.plotly_chart(fig_obj_main, use_container_width=True, config=plot_config_interactive_main_ui)
                                    else: st.caption(f"Plot for '{plot_cfg_tab_item['title']}' could not be generated (returned None).")
                                else: st.caption(f"No data for '{plot_cfg_tab_item['title']}' in selected range.")
                            except Exception as e_plot_render:
                                logger.error(f"Error rendering plot '{plot_cfg_tab_item['title']}': {e_plot_render}", exc_info=True)
                                st.error(f"⚠️ Error for plot '{plot_cfg_tab_item['title']}': {e_plot_render}")
                    num_plots_in_row_tab += 1
                
                # --- Insights Section for Tab ---
                st.markdown("<hr style='margin-top:2rem;'><h3 style='text-align:center; margin-top:1rem;'>🏛️ Leadership Actionable Insights</h3>", unsafe_allow_html=True)
                # ... (Full dynamic and static insights rendering logic as in previous combined file) ...
                if tab_def_main_app.get("insights_html"): st.markdown(tab_def_main_app["insights_html"], unsafe_allow_html=True)

            else: # No simulation_results
                st.info(f"ℹ️ Run a simulation or load data to view {tab_def_main_app['name']}.", icon="📊")

    # --- Glossary Tab ---
    with tabs_streamlit_objs_main[4]:
        st.header("📖 Glossary of Terms", divider=COLOR_ACCENT_INDIGO_CSS)
        # ... (Full Glossary HTML content as in previous combined file) ...
        st.markdown("""<p>Defines key metrics used...</p>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
