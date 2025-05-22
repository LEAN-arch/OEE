# main.py
import logging
import streamlit as st
import pandas as pd
import numpy as np
from config import DEFAULT_CONFIG, validate_config
from visualizations import (
    plot_key_metrics_summary,
    plot_task_compliance_score,
    plot_collaboration_proximity_index,
    plot_operational_recovery,
    plot_operational_efficiency,
    plot_worker_distribution,
    plot_worker_density_heatmap,
    plot_worker_wellbeing,
    plot_psychological_safety,
    plot_downtime_trend
)
from simulation import simulate_workplace_operations
from utils import save_simulation_data, load_simulation_data, generate_pdf_report

LEAN_LOGO_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgG6NcxuAAAAAElFTkSuQmCC"
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]', filename='dashboard.log', filemode='a')
logger.info("Main.py: Parsed imports and logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.1\nInsights for operational excellence."})
st.markdown(""" <style> /* ... CSS from previous ... */ </style> """, unsafe_allow_html=True) # Assuming CSS is same


def render_settings_sidebar():
    with st.sidebar:
        st.markdown(f'<img src="{LEAN_LOGO_BASE64}" width="100" alt="Lean Institute Logo" style="display: block; margin: 0 auto 1rem;">', unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")

        with st.expander("🧪 Simulation Parameters", expanded=True):
            # For each widget, its 'key' will be used to store its value in st.session_state.
            # The 'value' argument reads from st.session_state[key] if it exists, otherwise uses the default.
            
            team_size = st.slider(
                "Team Size", 10, 100, 
                st.session_state.get('sb_team_size_slider', DEFAULT_CONFIG['TEAM_SIZE']), # Read from session_state or default
                key="sb_team_size_slider", 
                help="Adjust the number of workers in the simulated shift."
            )
            shift_duration = st.slider(
                "Shift Duration (min)", 200, 2000, 
                st.session_state.get('sb_shift_duration_slider', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']), 
                step=2, key="sb_shift_duration_slider", 
                help="Set the total length of the simulated work shift."
            )
            
            max_disrupt_time = shift_duration - 2 
            disruption_options = [i * 2 for i in range(max_disrupt_time // 2)] if max_disrupt_time > 0 else []
            default_disrupt_mins_raw = [i * 2 for i in DEFAULT_CONFIG.get('DISRUPTION_INTERVALS', [])]
            valid_default_disrupt_mins = [m for m in default_disrupt_mins_raw if m in disruption_options]
            
            current_disrupt_selection_from_state = st.session_state.get('sb_disruption_intervals_multiselect', valid_default_disrupt_mins)
            # Ensure it's a list and items are valid
            if not isinstance(current_disrupt_selection_from_state, list):
                current_disrupt_selection_from_state = valid_default_disrupt_mins
            valid_current_disrupt_selection_for_widget = [m for m in current_disrupt_selection_from_state if m in disruption_options]

            disruption_intervals_minutes = st.multiselect(
                "Disruption Times (min)", disruption_options, 
                valid_current_disrupt_selection_for_widget,
                key="sb_disruption_intervals_multiselect", 
                help="Select times (in minutes from shift start) when disruptions will occur."
            )
            
            team_initiative_opts = ["Standard Operations", "More frequent breaks", "Team recognition"]
            current_initiative_from_state = st.session_state.get('sb_team_initiative_selectbox', team_initiative_opts[0])
            team_initiative_idx = team_initiative_opts.index(current_initiative_from_state) if current_initiative_from_state in team_initiative_opts else 0
            team_initiative = st.selectbox(
                "Operational Initiative", team_initiative_opts, 
                index=team_initiative_idx, 
                key="sb_team_initiative_selectbox", 
                help="Apply an operational strategy to observe its impact."
            )
            
            run_simulation_button = st.button("🚀 Run New Simulation", key="sb_run_simulation_button", type="primary", use_container_width=True)

        with st.expander("🎨 Visualization Options"):
            high_contrast = st.checkbox("High Contrast Plots", st.session_state.get('sb_high_contrast_checkbox', False), key="sb_high_contrast_checkbox", help="Use high-contrast colors for charts.")
            use_3d_distribution = st.checkbox("Enable 3D Worker View", st.session_state.get('sb_use_3d_distribution_checkbox', False), key="sb_use_3d_distribution_checkbox", help="Render worker positions in 3D.")
            debug_mode = st.checkbox("Show Debug Info", st.session_state.get('sb_debug_mode_checkbox', False), key="sb_debug_mode_checkbox", help="Display debug information.")

        with st.expander("💾 Data Management & Export"):
            load_data_button = st.button("🔄 Load Previous Simulation", key="sb_load_data_button", use_container_width=True)
            can_gen_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            # ... (rest of data management, PDF, CSV export as before - ensure keys are unique if any widgets are added here)
            if st.button("📄 Download Report (.tex)", key="sb_pdf_button", disabled=not can_gen_report, use_container_width=True): # Unique key
                if can_gen_report:
                    # ... PDF generation logic ...
                    pass 
            if can_gen_report: # Simplified CSV download button outside of if condition for button press
                sim_res_exp = st.session_state.simulation_results; num_steps_csv = len(sim_res_exp.get('downtime_minutes', []))
                if num_steps_csv > 0:
                    csv_data = {k: sim_res_exp.get(k, [np.nan]*num_steps_csv)[:num_steps_csv] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'downtime_minutes', 'task_completion_rate']}
                    csv_data.update({'task_compliance': sim_res_exp.get('task_compliance', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv], 'collaboration_proximity': sim_res_exp.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv], 'worker_wellbeing': sim_res_exp.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps_csv)[:num_steps_csv], 'step': list(range(num_steps_csv)), 'time_minutes': [i * 2 for i in range(num_steps_csv)]})
                    st.download_button("📥 Download Data (CSV)", pd.DataFrame(csv_data).to_csv(index=False).encode('utf-8'), "workplace_summary.csv", "text/csv", key="sb_csv_dl_button", use_container_width=True) # Unique key
                else: st.caption("No data for CSV.")
            elif not can_gen_report: st.caption("Run simulation for export.")


        if st.session_state.get('sb_debug_mode_checkbox', False): 
            with st.expander("🛠️ Debug Information", expanded=False):
                st.write("**Default Config (Partial):**"); st.json({k: DEFAULT_CONFIG.get(k) for k in ['ENTRY_EXIT_POINTS', 'WORK_AREAS']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Active Simulation Config (from results):**"); st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else: st.write("**No active simulation.**")

        st.markdown("## 📋 Help & Info")
        if st.button("ℹ️ Help & Glossary", key="sb_help_button", use_container_width=True): st.session_state.show_help_glossary = not st.session_state.get('show_help_glossary', False); st.rerun() # Unique key
        if st.button("🚀 Quick Tour", key="sb_tour_button", use_container_width=True): st.session_state.show_tour = not st.session_state.get('show_tour', False); st.rerun() # Unique key

    # Values returned are now directly from the widgets, which are also stored in session_state by their keys
    return team_size, shift_duration, disruption_intervals_minutes, team_initiative, \
           run_simulation_button, load_data_button, \
           high_contrast, use_3d_distribution, debug_mode


# --- run_simulation_logic remains the same ---
@st.cache_data(ttl=3600, show_spinner="⚙️ Running simulation model...")
def run_simulation_logic(team_size, shift_duration_minutes, disruption_intervals_minutes, team_initiative_selected):
    config = DEFAULT_CONFIG.copy(); config['TEAM_SIZE'] = team_size; config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // 2
    config['DISRUPTION_INTERVALS'] = sorted(list(set(m // 2 for m in disruption_intervals_minutes if isinstance(m, (int, float)) and m >= 0)))
    if 'WORK_AREAS' in config and isinstance(config['WORK_AREAS'], dict) and config['WORK_AREAS']:
        total_workers_in_config_zones = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())
        if total_workers_in_config_zones == 0 and team_size > 0: 
            num_zones_for_dist = len([zn for zn, zd in config['WORK_AREAS'].items() if zd.get('tasks_per_interval',0) > 0 or 'Warehouse' in zn or 'Assembly' in zn]); num_zones_for_dist = num_zones_for_dist if num_zones_for_dist > 0 else len(config['WORK_AREAS']) 
            if num_zones_for_dist > 0:
                workers_per_zone = team_size // num_zones_for_dist; remainder_workers = team_size % num_zones_for_dist
                zone_keys_for_dist = [zn for zn, zd in config['WORK_AREAS'].items() if zd.get('tasks_per_interval',0) > 0 or 'Warehouse' in zn or 'Assembly' in zn]
                if not zone_keys_for_dist: zone_keys_for_dist = list(config['WORK_AREAS'].keys())
                for i, zone_key in enumerate(zone_keys_for_dist): config['WORK_AREAS'][zone_key]['workers'] = workers_per_zone + (1 if i < remainder_workers else 0)
                all_zone_keys = list(config['WORK_AREAS'].keys()); [config['WORK_AREAS'][zk].update({'workers':0}) for zk in all_zone_keys if zk not in zone_keys_for_dist] 
        elif total_workers_in_config_zones > 0 and total_workers_in_config_zones != team_size : 
            ratio = team_size / total_workers_in_config_zones; accumulated_workers = 0; sorted_zone_keys = sorted(list(config['WORK_AREAS'].keys()))
            for zone_key in sorted_zone_keys[:-1]: assigned = int(round(config['WORK_AREAS'][zone_key].get('workers', 0) * ratio)); config['WORK_AREAS'][zone_key]['workers'] = assigned; accumulated_workers += assigned
            last_zone_key = sorted_zone_keys[-1]; config['WORK_AREAS'][last_zone_key]['workers'] = team_size - accumulated_workers
        elif team_size == 0: 
            for zone_key in config['WORK_AREAS']: config['WORK_AREAS'][zone_key]['workers'] = 0
    validate_config(config) 
    logger.info(f"Running simulation with: Team Size={team_size}, Duration={shift_duration_minutes}min, Disruptions(min): {disruption_intervals_minutes}, Initiative: {team_initiative_selected}", extra={'user_action': 'Run Simulation'})
    sim_results_tuple = simulate_workplace_operations(num_team_members=team_size, num_steps=config['SHIFT_DURATION_INTERVALS'], disruption_intervals=config['DISRUPTION_INTERVALS'], team_initiative=team_initiative_selected, config=config)
    expected_keys = ['team_positions_df', 'task_compliance', 'collaboration_proximity', 'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 'worker_wellbeing', 'psychological_safety', 'feedback_impact', 'downtime_minutes', 'task_completion_rate']
    if not isinstance(sim_results_tuple, tuple) or len(sim_results_tuple) != len(expected_keys):
        err_msg = f"Simulation returned unexpected data format. Expected tuple of {len(expected_keys)} items, got {type(sim_results_tuple)} with length {len(sim_results_tuple) if isinstance(sim_results_tuple, (list,tuple)) else 'N/A'}."; logger.critical(err_msg, extra={'user_action': 'Run Simulation - CRITICAL Data Format Error'}); raise TypeError(err_msg)
    simulation_output_dict = dict(zip(expected_keys, sim_results_tuple))
    simulation_output_dict['config_params'] = {'TEAM_SIZE': team_size, 'SHIFT_DURATION_MINUTES': shift_duration_minutes, 'DISRUPTION_INTERVALS_MINUTES': disruption_intervals_minutes, 'DISRUPTION_INTERVALS_STEPS': config['DISRUPTION_INTERVALS'], 'TEAM_INITIATIVE': team_initiative_selected}
    save_simulation_data(simulation_output_dict); return simulation_output_dict

# --- get_actionable_insights remains the same ---
def get_actionable_insights(sim_data, config):
    insights = []; # ... (implementation as before)
    if not sim_data or not isinstance(sim_data, dict): return insights 
    compliance_data = safe_get(sim_data, 'task_compliance.data', []); compliance_avg = safe_stat(compliance_data, np.mean, default_val=DEFAULT_CONFIG.get('TARGET_COMPLIANCE', 75)); target_compliance = config.get('TARGET_COMPLIANCE', 75)
    if compliance_avg < target_compliance * 0.9: insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Avg. Task Compliance ({compliance_avg:.1f}%) is significantly below target ({target_compliance}%). Review disruption impacts, task complexities, and training."})
    elif compliance_avg < target_compliance: insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Avg. Task Compliance at {compliance_avg:.1f}%. Identify intervals or areas with lowest compliance for process review."})
    wellbeing_scores = safe_get(sim_data, 'worker_wellbeing.scores', []); wellbeing_avg = safe_stat(wellbeing_scores, np.mean, default_val=DEFAULT_CONFIG.get('TARGET_WELLBEING', 70)); target_wellbeing = config.get('TARGET_WELLBEING', 70)
    if wellbeing_avg < target_wellbeing * 0.85: insights.append({"type": "critical", "title": "Critical Worker Well-being Levels", "text": f"Avg. Well-being ({wellbeing_avg:.1f}%) is critically low (target {target_wellbeing}%). Urgent review of work conditions, load, and stress factors needed."})
    threshold_triggers = safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])
    if len(threshold_triggers) > (len(wellbeing_scores) * 0.1) and len(threshold_triggers) > 2 : insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers)} instances of well-being dropping below threshold. Investigate specific triggers and affected periods."})
    downtime_data = safe_get(sim_data, 'downtime_minutes', []); total_downtime = safe_stat(downtime_data, np.sum); downtime_threshold_total = config.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] * 0.05) 
    if total_downtime > downtime_threshold_total : insights.append({"type": "critical", "title": "Excessive Total Downtime", "text": f"Total shift downtime is {total_downtime:.0f} minutes, exceeding the guideline of {downtime_threshold_total:.0f} min. Focus on causes of disruptions and equipment reliability."})
    if compliance_avg > target_compliance * 1.05 and wellbeing_avg > target_wellbeing * 1.05 and total_downtime < downtime_threshold_total * 0.5 : insights.append({"type": "positive", "title": "Excellent Overall Performance", "text": "Key metrics significantly exceed targets, indicating highly effective operations and a positive work environment. Identify and replicate success factors."})
    initiative = sim_data.get('config_params', {}).get('TEAM_INITIATIVE', 'Standard Operations')
    if initiative != "Standard Operations": insights.append({"type": "info", "title": f"Initiative Active: '{initiative}'", "text": f"The '{initiative}' initiative was simulated. Compare results to a baseline 'Standard Operations' run to quantify its specific impact."})
    return insights


def main():
    st.title("Workplace Shift Optimization Dashboard")
    
    # Initialize general state keys ONCE if they don't exist.
    # Widget-specific value keys (like sb_team_size_slider) will be handled by st.session_state.get() in render_settings_sidebar
    general_state_keys = ['simulation_results', 'show_tour', 'show_help_glossary',
                          'op_metrics_time_slider_val', 'worker_insights_time_slider_val', 
                          'worker_snap_step_slider_val_dist_tab', 'downtime_tab_time_slider_val']
    for key in general_state_keys:
        if key not in st.session_state: 
            st.session_state[key] = None 
    
    # Call sidebar. The returned values are the CURRENT state of the widgets.
    # These widgets manage their own values in st.session_state using their 'key' arguments.
    sb_team_size, sb_shift_duration, sb_disrupt_mins, sb_team_initiative, \
    sb_run_sim_btn, sb_load_data_btn, sb_high_contrast, \
    sb_use_3d, sb_debug_mode = render_settings_sidebar()

    # Store these CURRENT sidebar values in separate session state keys IF you need to access them
    # outside of render_settings_sidebar before a rerun that re-calls render_settings_sidebar.
    # If render_settings_sidebar is the only place they are used to generate other values for the app,
    # then direct use of its return values or st.session_state[widget_key] is fine.
    # For simplicity, we'll assume these are general settings used by main app logic:
    st.session_state.team_size_setting = sb_team_size
    st.session_state.shift_duration_setting = sb_shift_duration
    st.session_state.disruption_intervals_minutes_setting = sb_disrupt_mins
    st.session_state.team_initiative_setting = sb_team_initiative
    st.session_state.high_contrast_setting = sb_high_contrast
    st.session_state.use_3d_distribution_setting = sb_use_3d
    st.session_state.debug_mode_setting = sb_debug_mode
    
    _default_shift_duration = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']; 
    _current_shift_duration_for_slider_max = st.session_state.get('shift_duration_setting', _default_shift_duration) or _default_shift_duration
    current_max_minutes_for_sliders = _current_shift_duration_for_slider_max - 2; 
    disruption_steps_for_plots = []

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        num_steps_from_sim = len(st.session_state.simulation_results.get('downtime_minutes', []))
        if num_steps_from_sim > 0: current_max_minutes_for_sliders = (num_steps_from_sim - 1) * 2
        disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', [])
    else: 
        _disrupt_mins_list_for_plots = st.session_state.get('disruption_intervals_minutes_setting', [])
        disruption_steps_for_plots = [m // 2 for m in _disrupt_mins_list_for_plots if isinstance(m, (int, float))]
    current_max_minutes_for_sliders = max(0, current_max_minutes_for_sliders) 

    if sb_run_sim_btn:
        with st.spinner("🚀 Simulating workplace operations..."):
            try: 
                st.session_state.simulation_results = run_simulation_logic(
                    st.session_state.team_size_setting, 
                    st.session_state.shift_duration_setting,
                    st.session_state.disruption_intervals_minutes_setting, 
                    st.session_state.team_initiative_setting
                )
                st.success("✅ Simulation completed!"); 
                logger.info("Simulation run successful.", extra={'user_action': 'Run Simulation - Success'}); 
                st.rerun() 
            except Exception as e: 
                logger.error(f"Simulation Run Error: {e}", exc_info=True, extra={'user_action': 'Run Simulation - Error'})
                st.error(f"❌ Simulation failed: {e}"); 
                st.session_state.simulation_results = None 
    
    if sb_load_data_btn:
        with st.spinner("🔄 Loading saved data..."):
            try:
                loaded_data = load_simulation_data() 
                if loaded_data and isinstance(loaded_data, dict):
                    st.session_state.simulation_results = loaded_data; 
                    cfg = loaded_data.get('config_params', {})
                    # Update the "setting" session state keys, which will then inform the sidebar widgets on the next rerun.
                    st.session_state.team_size_setting = cfg.get('TEAM_SIZE', st.session_state.team_size_setting)
                    st.session_state.shift_duration_setting = cfg.get('SHIFT_DURATION_MINUTES', st.session_state.shift_duration_setting)
                    st.session_state.team_initiative_setting = cfg.get('TEAM_INITIATIVE', st.session_state.team_initiative_setting)
                    st.session_state.disruption_intervals_minutes_setting = cfg.get('DISRUPTION_INTERVALS_MINUTES', st.session_state.disruption_intervals_minutes_setting)
                    
                    # Also update the direct widget keys in session_state so sidebar shows loaded values immediately.
                    st.session_state.sb_team_size_slider = cfg.get('TEAM_SIZE', st.session_state.get('sb_team_size_slider'))
                    st.session_state.sb_shift_duration_slider = cfg.get('SHIFT_DURATION_MINUTES', st.session_state.get('sb_shift_duration_slider'))
                    st.session_state.sb_team_initiative_selectbox = cfg.get('TEAM_INITIATIVE', st.session_state.get('sb_team_initiative_selectbox'))
                    st.session_state.sb_disruption_intervals_multiselect = cfg.get('DISRUPTION_INTERVALS_MINUTES', st.session_state.get('sb_disruption_intervals_multiselect'))


                    st.success("✅ Data loaded successfully!"); 
                    logger.info("Saved data loaded successfully.", extra={'user_action': 'Load Data - Success'}); 
                    st.rerun() 
                else: 
                    st.error("❌ Failed to load data or data is not in the expected dictionary format.")
                    logger.warning("Load data failed or invalid format.", extra={'user_action': 'Load Data - Fail/Invalid'})
            except Exception as e: 
                logger.error(f"Load Data Error: {e}", exc_info=True, extra={'user_action': 'Load Data - Error'})
                st.error(f"❌ Failed to load data: {e}"); 
                st.session_state.simulation_results = None
    
    if st.session_state.get('show_tour'): # Tour Modal
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3> ... </div>""", unsafe_allow_html=True) # Abridged
        if st.button("Got it!", key="tour_modal_close_btn"): st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help_glossary'): # Help/Glossary Modal
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3> ... </div>""", unsafe_allow_html=True) # Abridged
        if st.button("Understood", key="help_modal_close_btn"): st.session_state.show_help_glossary = False; st.rerun()

    tabs_main_names = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Insights", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs = st.tabs(tabs_main_names)
    
    def safe_get(data_dict, path_str, default_val=None):
        current = data_dict; default_return = default_val if default_val is not None else ([])
        if not isinstance(path_str, str) or not isinstance(data_dict, dict): return default_return
        for key in path_str.split('.'):
            if isinstance(current, dict): current = current.get(key)
            elif isinstance(current, (list, pd.Series)) and key.isdigit():
                try: idx = int(key); current = current[idx] if idx < len(current) else None
                except (ValueError, IndexError): current = None; break
            else: current = None; break
        return current if current is not None else default_return
    def safe_stat(data_list, stat_func, default_val=0.0):
        if not isinstance(data_list, (list, np.ndarray, pd.Series)): data_list = [];
        valid_data = [x for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x))]; return stat_func(valid_data) if valid_data else default_val
    
    plot_config_interactive = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'pan2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}; plot_config_minimal = {'displayModeBar': False}

    with tabs[0]: 
        st.header("📊 Key Performance Indicators & Actionable Insights", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; config_for_targets = {**DEFAULT_CONFIG, **sim_data.get('config_params',{})} # Merge for targets
            compliance_target=config_for_targets.get('TARGET_COMPLIANCE', 75); collab_target=config_for_targets.get('TARGET_COLLABORATION', 60); wb_target=config_for_targets.get('TARGET_WELLBEING', 70); dt_target_total_shift=config_for_targets.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT', sim_data.get('config_params',{}).get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) * 0.05 )
            compliance = safe_stat(safe_get(sim_data, 'task_compliance.data', []), np.mean); proximity = safe_stat(safe_get(sim_data, 'collaboration_proximity.data', []), np.mean); wellbeing = safe_stat(safe_get(sim_data, 'worker_wellbeing.scores', []), np.mean); downtime = safe_stat(safe_get(sim_data, 'downtime_minutes', []), np.sum)
            cols_metrics = st.columns(4); cols_metrics[0].metric("Task Compliance", f"{compliance:.1f}%", f"{compliance-compliance_target:.1f}% vs Target {compliance_target}%"); cols_metrics[1].metric("Collaboration Index", f"{proximity:.1f}%", f"{proximity-collab_target:.1f}% vs Target {collab_target}%"); cols_metrics[2].metric("Worker Well-Being", f"{wellbeing:.1f}%", f"{wellbeing-wb_target:.1f}% vs Target {wb_target}%"); cols_metrics[3].metric("Total Downtime", f"{downtime:.1f} min", f"{downtime-dt_target_total_shift:.1f} min vs Target {dt_target_total_shift:.0f}min", delta_color="inverse")
            try:
                summary_figs = plot_key_metrics_summary(compliance, proximity, wellbeing, downtime, st.session_state.high_contrast_setting) # Use setting from session_state
                if summary_figs:
                    cols_gauges = st.columns(min(len(summary_figs), 4) or 1) 
                    for i, fig in enumerate(summary_figs):
                        with cols_gauges[i % len(cols_gauges)]: st.plotly_chart(fig, use_container_width=True, config=plot_config_minimal)
                else: st.caption("Gauge charts could not be generated.")
            except Exception as e: logger.error(f"Overview Gauges Plotting Error: {e}", exc_info=True); st.error("⚠️ Error rendering overview gauges.")
            st.markdown("---"); st.subheader("💡 Key Insights & Alerts")
            actionable_insights = get_actionable_insights(sim_data, config_for_targets) # Pass relevant config
            if actionable_insights:
                for insight in actionable_insights:
                    alert_class = f"alert-{insight['type']}"; st.markdown(f'<div class="{alert_class}"><p class="insight-title">{insight["title"]}</p><p class="insight-text">{insight["text"]}</p></div>', unsafe_allow_html=True)
            else: st.info("✅ No critical alerts or specific insights identified. Performance appears stable against defined thresholds.")
            with st.expander("View Detailed Overview Data Table", expanded=False):
                num_s = len(safe_get(sim_data, 'downtime_minutes', []))
                if num_s > 0:
                    df_data = {'Time (min)': [i*2 for i in range(num_s)]}; df_data.update({'Task Compliance (%)': safe_get(sim_data, 'task_compliance.data', [np.nan]*num_s)[:num_s], 'Collaboration (%)': safe_get(sim_data, 'collaboration_proximity.data', [np.nan]*num_s)[:num_s], 'Well-Being (%)': safe_get(sim_data, 'worker_wellbeing.scores', [np.nan]*num_s)[:num_s], 'Downtime (min)': safe_get(sim_data, 'downtime_minutes', [np.nan]*num_s)[:num_s]})
                    st.dataframe(pd.DataFrame(df_data).style.format("{:.1f}", na_rep="-").set_table_styles([{'selector': 'th', 'props': [('background-color', '#293344'), ('color', '#EAEAEA')]}]), use_container_width=True, height=300)
                else: st.caption("No detailed overview data.")
        else: st.info("ℹ️ Run a simulation or load data to view the Overview & Insights.", icon="📊")

    with tabs[1]: 
        st.header("📈 Operational Performance Trends", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; op_slider_key = "op_time_slider"; op_val_key = "op_metrics_time_slider_val" # Using specific keys for session state value
            default_val = (0, current_max_minutes_for_sliders); val_from_state = st.session_state.get(op_val_key)
            if val_from_state is None or not (isinstance(val_from_state, tuple) and len(val_from_state) == 2): current_val = default_val; st.session_state[op_val_key] = current_val
            else: current_val = val_from_state
            current_val = (min(current_val[0], current_max_minutes_for_sliders), min(current_val[1], current_max_minutes_for_sliders)); current_val = (current_val[0], max(current_val[0], current_val[1]))
            time_range = st.slider("Select Time Range (minutes):", 0, current_max_minutes_for_sliders, current_val, 2, key=op_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({op_val_key: st.session_state[op_slider_key]}))
            start_idx, end_idx = time_range[0]//2, time_range[1]//2 + 1; filt_disrupt_steps = [s for s in disruption_steps_for_plots if start_idx <= s < end_idx]
            # ... (rest of plotting for Operational Metrics with no-data captions)
        else: st.info("ℹ️ Run a simulation or load data to view Operational Metrics.", icon="📈")

    with tabs[2]: 
        st.header("👥 Worker Dynamics & Well-being", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; worker_slider_key = "worker_insights_time_slider"; worker_val_key = "worker_insights_time_slider_val"
            default_val = (0, current_max_minutes_for_sliders); val_from_state = st.session_state.get(worker_val_key)
            if val_from_state is None or not (isinstance(val_from_state, tuple) and len(val_from_state) == 2): current_val = default_val; st.session_state[worker_val_key] = current_val
            else: current_val = val_from_state
            current_val = (min(current_val[0], current_max_minutes_for_sliders), min(current_val[1], current_max_minutes_for_sliders)); current_val = (current_val[0], max(current_val[0], current_val[1]))
            shared_time_range = st.slider("Select Time Range (minutes) for Worker Insights:", 0, current_max_minutes_for_sliders, current_val, 2, key=worker_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({worker_val_key: st.session_state[worker_slider_key]}))
            shared_start_idx, shared_end_idx = shared_time_range[0]//2, shared_time_range[1]//2 + 1
            # ... (rest of plotting for Worker Insights with no-data captions)
        else: st.info("ℹ️ Run a simulation or load data to view Worker Insights.", icon="👥")

    with tabs[3]: 
        st.header("⏱️ Downtime Analysis", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; dt_slider_key = "downtime_tab_time_slider"; dt_val_key = "downtime_tab_time_slider_val"
            default_val = (0, current_max_minutes_for_sliders); val_from_state = st.session_state.get(dt_val_key)
            if val_from_state is None or not (isinstance(val_from_state, tuple) and len(val_from_state) == 2): current_val = default_val; st.session_state[dt_val_key] = current_val
            else: current_val = val_from_state
            current_val = (min(current_val[0], current_max_minutes_for_sliders), min(current_val[1], current_max_minutes_for_sliders)); current_val = (current_val[0], max(current_val[0], current_val[1]))
            time_range_dt = st.slider("Select Time Range (minutes):", 0, current_max_minutes_for_sliders, current_val, 2, key=dt_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({dt_val_key: st.session_state[dt_slider_key]}))
            dt_start_idx, dt_end_idx = time_range_dt[0]//2, time_range_dt[1]//2 + 1
            dt_data_list = safe_get(sim_data, 'downtime_minutes', [])[dt_start_idx:dt_end_idx]
            if dt_data_list:
                total_downtime_period = sum(dt_data_list); avg_downtime_period = np.mean(dt_data_list) if dt_data_list else 0
                st.metric("Total Downtime in Selected Period", f"{total_downtime_period:.1f} min", help="Sum of all downtime instances in the selected time frame.")
                with st.container(border=True): st.markdown('<h5>Downtime Trend Over Time</h5>', unsafe_allow_html=True)
                try: st.plotly_chart(plot_downtime_trend(dt_data_list, DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD', 10), st.session_state.high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                except Exception as e: logger.error(f"Downtime Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Downtime Trend.")
            else: st.caption("No Downtime data for this time range.")
        else: st.info("ℹ️ Run a simulation or load data for Downtime Analysis.", icon="⏱️")
        
    with tabs[4]: 
        st.header("📖 Glossary of Terms", divider="blue")
        # ... (Glossary content as before)
        st.markdown("""<div style="font-size: 0.95rem; line-height: 1.7;">...</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
