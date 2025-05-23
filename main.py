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
logger.info("Main.py: Startup. Imports parsed, logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.3.1\nInsights for operational excellence & psychosocial well-being."})

# CSS Color constants (from your original CSS, for reference or direct use if needed outside visualizations.py)
COLOR_CRITICAL_RED_CSS = "#E53E3E"; COLOR_WARNING_AMBER_CSS = "#F59E0B"; COLOR_POSITIVE_GREEN_CSS = "#10B981"; COLOR_INFO_BLUE_CSS = "#3B82F6"; COLOR_ACCENT_INDIGO_CSS = "#4F46E5"

# --- UTILITY FUNCTIONS (main.py specific) ---
def safe_get(data_dict, path_str, default_val=None):
    current = data_dict
    is_list_like_path = False
    if isinstance(path_str, str):
        is_list_like_path = path_str.endswith(('.data', '.scores', '.triggers', '_log', 'events_list')) # '_log' for downtime_events_log
    
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
        valid_data = [float(x) for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x)) and isinstance(x, (int, float, str)) and str(x).strip()]
        try: valid_data = [float(x) for x in valid_data]
        except ValueError: valid_data = [x for x in valid_data if isinstance(x, (int, float))]
    if not valid_data: return default_val
    try:
        result = stat_func(np.array(valid_data))
        return default_val if isinstance(result, (float, np.floating)) and np.isnan(result) else result
    except Exception: return default_val

def get_actionable_insights(sim_data, current_config_dict):
    insights = []
    if not sim_data or not isinstance(sim_data, dict): return insights
    
    compliance_data = safe_get(sim_data, 'task_compliance.data', [])
    target_compliance = float(current_config_dict.get('TARGET_COMPLIANCE', 75.0))
    compliance_avg = safe_stat(compliance_data, np.mean, 0.0)
    if compliance_avg < target_compliance * 0.9:
        insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Avg. Task Compliance ({compliance_avg:.1f}%) critically below target ({target_compliance:.0f}%). Review disruptions, complexities, training."})
    elif compliance_avg < target_compliance:
        insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Avg. Task Compliance at {compliance_avg:.1f}%. Review areas with lowest compliance."})

    wellbeing_scores = safe_get(sim_data, 'worker_wellbeing.scores', [])
    target_wellbeing = float(current_config_dict.get('TARGET_WELLBEING', 70.0))
    wellbeing_avg = safe_stat(wellbeing_scores, np.mean, 0.0)
    wb_crit_factor = float(current_config_dict.get('WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET', 0.85))
    if wellbeing_avg < target_wellbeing * wb_crit_factor:
        insights.append({"type": "critical", "title": "Critical Worker Well-being", "text": f"Avg. Well-being ({wellbeing_avg:.1f}%) critically low (target {target_wellbeing:.0f}%). Urgent review needed."})
    
    threshold_triggers = safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])
    if wellbeing_scores and len(threshold_triggers) > max(2, len(wellbeing_scores) * 0.1):
        insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers)} low well-being instances. Investigate triggers."})

    downtime_event_log = safe_get(sim_data, 'downtime_events_log', []) # Use the raw log
    downtime_durations = [event.get('duration', 0.0) for event in downtime_event_log if isinstance(event, dict)]
    total_downtime = sum(downtime_durations)

    sim_cfg_params_insights = sim_data.get('config_params', {})
    shift_mins_insights = float(sim_cfg_params_insights.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
    dt_thresh_percent_insights = float(current_config_dict.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT_PERCENT', 0.05))
    dt_thresh_total_shift_abs_insights = shift_mins_insights * dt_thresh_percent_insights
    if total_downtime > dt_thresh_total_shift_abs_insights:
        insights.append({"type": "critical", "title": "Excessive Total Downtime", "text": f"Total downtime {total_downtime:.0f} min, exceeds guideline of {dt_thresh_total_shift_abs_insights:.0f} min ({dt_thresh_percent_insights*100:.0f}% of shift). Analyze causes."})

    # ... (Rest of insights logic as per previous `get_actionable_insights` version,
    #      ensuring correct data paths like `collaboration_metric.data` if needed for insights)
    logger.info(f"get_actionable_insights: Generated {len(insights)} insights.", extra={'user_action': 'Actionable Insights - End'})
    return insights

def aggregate_downtime_by_step(raw_downtime_event_log, num_total_steps_agg):
    downtime_per_step_agg = [0.0] * num_total_steps_agg
    if not isinstance(raw_downtime_event_log, list): return downtime_per_step_agg
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
    if isinstance(df.index, pd.RangeIndex):
        s_start, s_end = max(0, start_idx), min(len(df), end_idx)
        return df.iloc[s_start:s_end] if s_start < s_end else pd.DataFrame()
    if 'step' in df.columns: return df[(df['step'] >= start_idx) & (df['step'] < end_idx)]
    if df.index.name == 'step' or df.index.is_numeric(): return df[(df.index >= start_idx) & (df.index < end_idx)]
    return pd.DataFrame()

# --- CSS ---
st.markdown(f""" <style> {/* ... CSS from previous full main.py (ensure color vars are defined if used directly in CSS) ... */} </style> """, unsafe_allow_html=True)

# --- SIDEBAR ---
def render_settings_sidebar():
    # ... (Full sidebar rendering logic as in previous combined file,
    #      using unique keys for widgets like `widget_form_event_type`, `widget_form_event_start` etc.
    #      and directly using st.session_state for their values.
    #      Returns run_sim_btn, load_data_btn)
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem; color: #A0A0A0;'>Workplace Optimizer</h3>", unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")
        with st.expander("🧪 Simulation Parameters", expanded=True):
            st.number_input("Team Size", min_value=1, max_value=200, key="sb_team_size_num", step=1)
            st.number_input("Shift Duration (min)", min_value=60, max_value=7200, key="sb_shift_duration_num", step=10)
            
            current_shift_duration_sb = st.session_state.sb_shift_duration_num
            mpi_sb = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)

            st.markdown("---"); st.markdown("<h5>🗓️ Schedule Shift Events</h5>", unsafe_allow_html=True)
            event_types_sb = ["Major Disruption", "Minor Disruption", "Scheduled Break", "Short Pause", "Team Meeting", "Maintenance", "Custom Event"]
            with st.container():
                st.session_state.form_event_type = st.selectbox("Event Type", event_types_sb, 
                    index=event_types_sb.index(st.session_state.form_event_type) if st.session_state.form_event_type in event_types_sb else 0, 
                    key="widget_event_type_form")
                c1_sb,c2_sb = st.columns(2)
                st.session_state.form_event_start = c1_sb.number_input("Start (min)", 0, max(0, current_shift_duration_sb - mpi_sb), mpi_sb, key="widget_event_start_form")
                st.session_state.form_event_duration = c2_sb.number_input("Duration (min)", mpi_sb, current_shift_duration_sb, mpi_sb, key="widget_event_duration_form")

            if st.button("➕ Add Event", key="sb_add_event_btn_main", use_container_width=True):
                if st.session_state.form_event_start + st.session_state.form_event_duration > current_shift_duration_sb:
                    st.warning("Event end exceeds shift.")
                else:
                    st.session_state.sb_scheduled_events_list.append({
                        "Event Type": st.session_state.form_event_type,
                        "Start Time (min)": st.session_state.form_event_start, 
                        "Duration (min)": st.session_state.form_event_duration,
                    })
                    st.session_state.sb_scheduled_events_list.sort(key=lambda x: x.get("Start Time (min)", 0))
                    st.session_state.form_event_start = 0 # Reset form
                    st.session_state.form_event_duration = max(mpi_sb, 10)
                    st.rerun()

            if not st.session_state.sb_scheduled_events_list: st.caption("No events scheduled.")
            else:
                with st.container(height=200):
                    for i_ev, event_ev in enumerate(st.session_state.sb_scheduled_events_list):
                        evc1, evc2 = st.columns([0.85,0.15])
                        evc1.markdown(f"<div class='event-item'><span><b>{event_ev.get('Event Type','N/A')}</b> at {event_ev.get('Start Time (min)','N/A')}min ({event_ev.get('Duration (min)','N/A')}min)</span></div>", unsafe_allow_html=True)
                        if evc2.button("✖", key=f"rem_ev_main_{i_ev}", type="secondary", use_container_width=True):
                            st.session_state.sb_scheduled_events_list.pop(i_ev); st.rerun()
            if st.session_state.sb_scheduled_events_list and st.button("Clear All Events", key="sb_clear_events_main", type="secondary", use_container_width=True):
                st.session_state.sb_scheduled_events_list = []; st.rerun()

            st.markdown("---")
            st.selectbox("Operational Initiative", ["Standard Operations", "More frequent breaks", "Team recognition", "Increased Autonomy"], key="sb_team_initiative_selectbox")
            run_button = st.button("🚀 Run Simulation", key="sb_run_button_main", type="primary", use_container_width=True)
        
        with st.expander("🎨 Visualization Options"): # ... (as before) ...
             st.checkbox("High Contrast Plots", key="sb_high_contrast_checkbox") # ...
        with st.expander("💾 Data Management & Export"): # ... (as before, but with complete CSV logic)
            load_button = st.button("🔄 Load Previous Simulation", key="sb_load_button_main", use_container_width=True)
            can_export_main = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("📄 Download Report (.tex)", key="sb_tex_dl_main", disabled=not can_export_main, use_container_width=True):
                # ... (Full PDF generation logic from previous complete main.py, using updated data keys) ...
                pass # Placeholder for brevity
            
            if can_export_main:
                sim_res_csv = st.session_state.simulation_results
                sim_cfg_csv = sim_res_csv.get('config_params', {})
                num_steps_csv = sim_cfg_csv.get('SHIFT_DURATION_INTERVALS', 0)
                mpi_csv = sim_cfg_csv.get('MINUTES_PER_INTERVAL', DEFAULT_CONFIG["MINUTES_PER_INTERVAL"])

                if num_steps_csv > 0:
                    csv_data = {'step': list(range(num_steps_csv)), 'time_minutes': [i * mpi_csv for i in range(num_steps_csv)]}
                    export_metrics_csv = {
                        'task_compliance.data': 'task_compliance_percent',
                        'collaboration_metric.data': 'collaboration_metric_percent', # Updated
                        'operational_recovery': 'operational_recovery_percent',
                        'worker_wellbeing.scores': 'worker_wellbeing_index',
                        'psychological_safety': 'psychological_safety_score',
                        'productivity_loss': 'productivity_loss_percent',
                        'task_completion_rate': 'task_completion_rate_percent',
                        'worker_wellbeing.team_cohesion_scores': 'team_cohesion_score',
                        'worker_wellbeing.perceived_workload_scores': 'perceived_workload_score_0_10'
                    }
                    for path, col_name in export_metrics_csv.items():
                        csv_data[col_name] = _prepare_timeseries_for_export(safe_get(sim_res_csv, path, []), num_steps_csv)
                    
                    raw_downtime_csv = safe_get(sim_res_csv, 'downtime_events_log', [])
                    csv_data['downtime_minutes_per_interval'] = aggregate_downtime_by_step(raw_downtime_csv, num_steps_csv)
                    
                    df_to_csv = pd.DataFrame(csv_data)
                    st.download_button("📥 Download Data (CSV)", df_to_csv.to_csv(index=False).encode('utf-8'), 
                                      "workplace_summary.csv", "text/csv", key="sb_csv_dl_main", use_container_width=True)
                else: st.caption("No detailed data to export (0 steps).")
            elif not can_export_main: st.caption("Run simulation for export options.")
        # ... (Debug info, Help buttons as before) ...
    return run_button, load_button


# --- run_simulation_logic (main.py) ---
@st.cache_data(ttl=3600, show_spinner="⚙️ Simulating workplace operations...")
def run_simulation_logic(team_size_sl, shift_duration_sl, scheduled_events_sl, team_initiative_sl):
    config_sl = DEFAULT_CONFIG.copy()
    config_sl['TEAM_SIZE'] = team_size_sl
    config_sl['SHIFT_DURATION_MINUTES'] = shift_duration_sl
    
    mpi_sl = config_sl.get('MINUTES_PER_INTERVAL', 2)
    if mpi_sl <= 0: mpi_sl = 2; logger.error("MPI was <=0, used 2.")
    config_sl['SHIFT_DURATION_INTERVALS'] = shift_duration_sl // mpi_sl

    processed_events_sl = []
    for event_sl_ui in scheduled_events_sl:
        evt_sl = event_sl_ui.copy()
        if 'step' not in evt_sl and 'Start Time (min)' in evt_sl: # Derive step if not present
            evt_sl['step'] = int(evt_sl['Start Time (min)'] // mpi_sl)
        processed_events_sl.append(evt_sl)
    config_sl['SCHEDULED_EVENTS'] = processed_events_sl
    
    # Worker Redistribution Logic (ensure sum of workers in WORK_AREAS matches TEAM_SIZE)
    if 'WORK_AREAS' in config_sl and isinstance(config_sl['WORK_AREAS'], dict):
        current_total_workers_cfg = sum(z.get('workers',0) for z in config_sl['WORK_AREAS'].values() if isinstance(z,dict))
        if current_total_workers_cfg != team_size_sl and team_size_sl > 0:
            logger.info(f"Redistributing workers. Config sum: {current_total_workers_cfg}, Target team: {team_size_sl}")
            if current_total_workers_cfg > 0 : # Proportional redistribution
                ratio = team_size_sl / current_total_workers_cfg
                accumulated = 0; sorted_zones = sorted([k for k,v in config_sl['WORK_AREAS'].items() if isinstance(v,dict)])
                for i_zone, zone_k in enumerate(sorted_zones):
                    zone_data_sl = config_sl['WORK_AREAS'][zone_k]
                    if i_zone < len(sorted_zones) -1:
                        new_w = int(round(zone_data_sl.get('workers',0) * ratio))
                        zone_data_sl['workers'] = new_w; accumulated += new_w
                    else: # Last zone gets remainder
                        zone_data_sl['workers'] = max(0, team_size_sl - accumulated) # Ensure non-negative
            else: # Distribute evenly if no workers configured
                num_assignable_zones = len([k for k,v in config_sl['WORK_AREAS'].items() if isinstance(v,dict) and not v.get('is_rest_area')])
                if num_assignable_zones == 0: num_assignable_zones = len(config_sl['WORK_AREAS']) # Fallback to all if no non-rest
                if num_assignable_zones > 0:
                    base_w, rem_w = divmod(team_size_sl, num_assignable_zones)
                    assign_count = 0
                    for zone_k, zone_data_sl in config_sl['WORK_AREAS'].items():
                        if isinstance(zone_data_sl, dict) and not zone_data_sl.get('is_rest_area', False):
                            zone_data_sl['workers'] = base_w + (1 if assign_count < rem_w else 0)
                            assign_count +=1
    
    validate_config(config_sl)
    logger.info(f"Running sim: Team={team_size_sl}, Duration={shift_duration_sl}min, Events={len(config_sl['SCHEDULED_EVENTS'])}, Initiative={team_initiative_sl}", extra={'user_action': 'Run Simulation - Start'})
    
    expected_keys = [ # MUST match simulation.py return tuple order
        'team_positions_df', 'task_compliance', 'collaboration_metric',
        'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 
        'worker_wellbeing', 'psychological_safety', 'feedback_impact', 
        'downtime_events_log', 'task_completion_rate'
    ]
    sim_results_tuple_sl = simulate_workplace_operations(
        num_team_members=team_size_sl, num_steps=config_sl['SHIFT_DURATION_INTERVALS'],
        scheduled_events=config_sl['SCHEDULED_EVENTS'], team_initiative=team_initiative_sl, config=config_sl
    )
    
    if not isinstance(sim_results_tuple_sl, tuple) or len(sim_results_tuple_sl) != len(expected_keys):
        logger.critical("Simulation returned unexpected data format or length.", extra={'user_action':'Sim Format Error'})
        raise TypeError("Simulation returned unexpected data format.")
        
    simulation_output_dict_sl = dict(zip(expected_keys, sim_results_tuple_sl))
    simulation_output_dict_sl['config_params'] = {
        'TEAM_SIZE': team_size_sl, 'SHIFT_DURATION_MINUTES': shift_duration_sl,
        'SHIFT_DURATION_INTERVALS': config_sl['SHIFT_DURATION_INTERVALS'],
        'MINUTES_PER_INTERVAL': mpi_sl, 'SCHEDULED_EVENTS': config_sl['SCHEDULED_EVENTS'],
        'TEAM_INITIATIVE': team_initiative_sl, 'WORK_AREAS_EFFECTIVE': config_sl.get('WORK_AREAS', {}).copy()
    }
    
    disruption_steps_sl = [evt.get('step') for evt in config_sl['SCHEDULED_EVENTS'] if isinstance(evt,dict) and "Disruption" in evt.get("Event Type","") and isinstance(evt.get('step'),int)]
    simulation_output_dict_sl['config_params']['DISRUPTION_EVENT_STEPS'] = sorted(list(set(disruption_steps_sl)))

    save_simulation_data(simulation_output_dict_sl) 
    return simulation_output_dict_sl


# --- MAIN APPLICATION ---
def main():
    st.title("Workplace Shift Optimization Dashboard")
    
    mpi_global_main = DEFAULT_CONFIG.get("MINUTES_PER_INTERVAL", 2)
    app_state_defaults_main = {
        'simulation_results': None, 'show_tour': False, 'show_help_glossary': False,
        'sb_team_size_num': DEFAULT_CONFIG['TEAM_SIZE'], 'sb_shift_duration_num': DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'],
        'sb_scheduled_events_list': list(DEFAULT_CONFIG.get('DEFAULT_SCHEDULED_EVENTS', [])),
        'sb_team_initiative_selectbox': "Standard Operations",
        'sb_high_contrast_checkbox': False, 'sb_use_3d_distribution_checkbox': False, 'sb_debug_mode_checkbox': False,
        'form_event_type': "Major Disruption", 'form_event_start': 0, 'form_event_duration': max(mpi_global_main, 10),
    }
    default_max_mins_main = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - mpi_global_main if DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] > mpi_global_main else 0
    for prefix_main in ['op', 'ww', 'dt']:
        app_state_defaults_main[f'{prefix_main}_start_time_min'] = 0
        app_state_defaults_main[f'{prefix_main}_end_time_min'] = default_max_mins_main
    for key_main, val_main in app_state_defaults_main.items():
        if key_main not in st.session_state: st.session_state[key_main] = val_main
            
    sb_run_btn, sb_load_btn = render_settings_sidebar()
    
    current_high_contrast_main = st.session_state.sb_high_contrast_checkbox
    use_3d_main = st.session_state.sb_use_3d_distribution_checkbox

    active_mpi_main = mpi_global_main
    max_mins_ui_main = default_max_mins_main
    sim_disruption_steps_main = []

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        sim_cfg_main = st.session_state.simulation_results.get('config_params', {})
        active_mpi_main = sim_cfg_main.get('MINUTES_PER_INTERVAL', mpi_global_main)
        sim_intervals_main = sim_cfg_main.get('SHIFT_DURATION_INTERVALS', 0)
        max_mins_ui_main = max(0, sim_intervals_main * active_mpi_main - active_mpi_main) if sim_intervals_main > 0 else 0
        sim_disruption_steps_main = sim_cfg_main.get('DISRUPTION_EVENT_STEPS', [])
    else:
        sim_intervals_main = st.session_state.sb_shift_duration_num // active_mpi_main if active_mpi_main > 0 else 0
        max_mins_ui_main = max(0, sim_intervals_main * active_mpi_main - active_mpi_main) if sim_intervals_main > 0 else 0
        for event_main_ui in st.session_state.sb_scheduled_events_list:
            if "Disruption" in event_main_ui.get("Event Type","") and isinstance(event_main_ui.get("Start Time (min)"), (int,float)):
                sim_disruption_steps_main.append(int(event_main_ui["Start Time (min)"] // active_mpi_main))
        sim_disruption_steps_main = sorted(list(set(sim_disruption_steps_main)))
    
    for prefix_main_ui in ['op', 'ww', 'dt']: # Ensure UI time ranges are clamped
        st.session_state[f"{prefix_main_ui}_start_time_min"] = max(0, min(st.session_state.get(f"{prefix_main_ui}_start_time_min",0), max_mins_ui_main))
        st.session_state[f"{prefix_main_ui}_end_time_min"] = max(st.session_state[f"{prefix_main_ui}_start_time_min"], min(st.session_state.get(f"{prefix_main_ui}_end_time_min",max_mins_ui_main), max_mins_ui_main))

    if sb_run_btn:
        with st.spinner("🚀 Simulating workplace operations..."):
            try:
                results = run_simulation_logic(st.session_state.sb_team_size_num, st.session_state.sb_shift_duration_num, 
                                               st.session_state.sb_scheduled_events_list, st.session_state.sb_team_initiative_selectbox)
                st.session_state.simulation_results = results
                new_cfg = results['config_params']
                new_max = max(0, new_cfg.get('SHIFT_DURATION_INTERVALS',0) * new_cfg.get('MINUTES_PER_INTERVAL',2) - new_cfg.get('MINUTES_PER_INTERVAL',2))
                for pfx in ['op','ww','dt']: st.session_state[f"{pfx}_start_time_min"]=0; st.session_state[f"{pfx}_end_time_min"]=new_max
                st.success("✅ Simulation completed!"); logger.info("Sim run success."); st.rerun()
            except Exception as e_run: logger.error(f"Sim Run Error: {e_run}", exc_info=True); st.error(f"❌ Sim failed: {e_run}"); st.session_state.simulation_results = None
    if sb_load_btn:
        with st.spinner("🔄 Loading saved simulation data..."):
            try:
                loaded = load_simulation_data()
                if loaded and isinstance(loaded, dict) and 'config_params' in loaded:
                    st.session_state.simulation_results = loaded; cfg_ld = loaded['config_params']
                    st.session_state.sb_team_size_num = cfg_ld.get('TEAM_SIZE', DEFAULT_CONFIG['TEAM_SIZE'])
                    st.session_state.sb_shift_duration_num = cfg_ld.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'])
                    st.session_state.sb_scheduled_events_list = list(cfg_ld.get('SCHEDULED_EVENTS', [])) # list() for copy
                    st.session_state.sb_team_initiative_selectbox = cfg_ld.get('TEAM_INITIATIVE', "Standard Operations")
                    max_ld = max(0, cfg_ld.get('SHIFT_DURATION_INTERVALS',0) * cfg_ld.get('MINUTES_PER_INTERVAL',2) - cfg_ld.get('MINUTES_PER_INTERVAL',2))
                    for pfx in ['op','ww','dt']: st.session_state[f"{pfx}_start_time_min"]=0; st.session_state[f"{pfx}_end_time_min"]=max_ld
                    st.success("✅ Data loaded!"); logger.info("Load success."); st.rerun()
                else: st.error("❌ Failed to load data or data invalid."); logger.warning("Load fail/invalid.")
            except Exception as e_load: logger.error(f"Load Data Error: {e_load}", exc_info=True); st.error(f"❌ Load failed: {e_load}"); st.session_state.simulation_results = None
    
    # --- Modals (as before) ---
    if st.session_state.show_tour: # ... Tour modal ...
        pass
    if st.session_state.show_help_glossary: # ... Help modal ...
        pass

    # --- MAIN TABS ---
    tab_names_main = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Well-being", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs_obj_main = st.tabs(tab_names_main)
    plot_cfg_interactive_main = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}
    plot_cfg_minimal_main = {'displayModeBar': False}

    with tabs_obj_main[0]: # Overview
        # ... (Full Overview Tab logic as in previous complete file, ensuring correct data paths like 'downtime_events_log' and 'collaboration_metric.data')
        pass

    # Tab Definitions (ensure data_path and extra_args_paths are correct for the simulation output)
    tab_configs_main = [
        {"name": "📈 Operational Metrics", "key_prefix": "op", "plots": [
             {"title": "Task Compliance", "data_path": "task_compliance.data", "plot_func": plot_task_compliance_score, "extra_args_paths": {"forecast_data": "task_compliance.forecast", "z_scores": "task_compliance.z_scores"}},
             {"title": "Collaboration Metric", "data_path": "collaboration_metric.data", "plot_func": plot_collaboration_proximity_index, "extra_args_paths": {"forecast_data": "collaboration_metric.forecast"}},
             {"is_subheader": True, "title": "Additional Operational Metrics"},
             {"title": "Operational Resilience", "data_path": "operational_recovery", "plot_func": plot_operational_recovery, "extra_args_paths": {"productivity_loss_data": "productivity_loss"}},
             {"title": "OEE & Components", "is_oee": True}]},
        {"name": "👥 Worker Well-being", "key_prefix": "ww", "plots": [
             {"is_subheader": True, "title": "Psychosocial & Well-being Indicators"},
             {"title": "Worker Well-Being", "data_path": "worker_wellbeing.scores", "plot_func": plot_worker_wellbeing, "extra_args_paths": {"triggers": "worker_wellbeing.triggers"}},
             {"title": "Psychological Safety", "data_path": "psychological_safety", "plot_func": plot_psychological_safety},
             # ... (other wellbeing plots as before) ...
             {"is_subheader": True, "title": "Spatial Dynamics Analysis", "is_spatial": True}]},
        {"name": "⏱️ Downtime Analysis", "key_prefix": "dt", "metrics_display": True, "plots": [
            {"title": "Downtime Trend", "data_path": "downtime_events_log", "plot_func": plot_downtime_trend, "is_event_based_aggregation": True, "extra_args_fixed": {"interval_threshold_minutes": DEFAULT_CONFIG['DOWNTIME_PLOT_ALERT_THRESHOLD']}},
            {"title": "Downtime Causes", "data_path": "downtime_events_log", "plot_func": plot_downtime_causes_pie, "is_event_based_filtering": True}]}
    ]
    # (For brevity, insights_html and dynamic_insights_func are omitted from tab_configs_main here, but should be included)

    for i_tab_main, tab_def_main in enumerate(tab_configs_main):
        with tabs_obj_main[i_tab_main+1]:
            st.header(tab_def_main["name"], divider="blue")
            if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
                sim_data_main_tab = st.session_state.simulation_results
                sim_cfg_main_tab_loop = sim_data_main_tab.get('config_params', {}) # Config for this specific sim run

                st.markdown("##### Select Time Range for Plots:")
                start_time_ui, end_time_ui = time_range_input_section(
                    tab_def_main["key_prefix"], max_mins_ui_main, interval_duration_min=active_mpi_main
                )
                start_idx_ui = start_time_ui // active_mpi_main if active_mpi_main > 0 else 0
                end_idx_ui = (end_time_ui // active_mpi_main) + 1 if active_mpi_main > 0 else 0
                
                disrupt_steps_tab_abs_ui = [s for s in sim_disruption_steps_main if start_idx_ui <= s < end_idx_ui]

                # ... (Metrics display for Downtime tab as before) ...

                plot_container_main_loop = st.container()
                plots_in_row_main = 0
                for plot_cfg_main_loop in tab_def_main["plots"]:
                    # ... (Full plotting loop logic from previous complete main.py, ensuring:
                    #      - correct data slicing for lists and DataFrames
                    #      - correct processing for event_based_aggregation/filtering (downtime)
                    #      - correct handling of `extra_args_paths` (especially `triggers` for wellbeing)
                    #      - `disruption_points` passed to plot functions are relative to the sliced data
                    #      - `facility_config` for spatial plots is correctly constructed from `WORK_AREAS_EFFECTIVE`
                    #      - All plot functions are called with `high_contrast=current_high_contrast_main`
                    # )
                    pass # Placeholder for brevity
                # ... (Insights section for tab as before) ...
            else: st.info(f"ℹ️ Run simulation or load data to view {tab_def_main['name']}.", icon="📊")

    with tabs_obj_main[4]: # Glossary
        # ... (Glossary content as before) ...
        pass

if __name__ == "__main__":
    main()
