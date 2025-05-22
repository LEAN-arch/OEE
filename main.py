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

# Placeholder base64 logo
LEAN_LOGO_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgG6NcxuAAAAAElFTkSuQmCC"

logger = logging.getLogger(__name__)
if not logger.handlers: 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
        filename='dashboard.log',
        filemode='a'
    )
logger.info("Main.py: Parsed imports and logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(
    page_title="Workplace Shift Optimization Dashboard", # More action-oriented title
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@example.com',
        'Report a bug': "mailto:bugs@example.com",
        'About': "# Workplace Shift Optimization Dashboard\nVersion 1.1\nInsights for operational excellence."
    }
)

# Optimized CSS (largely the same, ensure consistency with visualization styling)
st.markdown("""
    <style>
        /* ... (CSS from previous optimized version remains largely the same) ... */
        /* Additions for actionable insights */
        .alert-critical { border-left: 5px solid #F87171; background-color: rgba(248, 113, 113, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }
        .alert-warning { border-left: 5px solid #FACC15; background-color: rgba(250, 204, 21, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }
        .alert-positive { border-left: 5px solid #22D3EE; background-color: rgba(34, 211, 238, 0.1); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }
        .insight-title { font-weight: 600; color: #EAEAEA; margin-bottom: 0.25rem;}
        .insight-text { font-size: 0.9rem; color: #D1D5DB;}
    </style>
""", unsafe_allow_html=True)

def render_settings_sidebar(): # No major changes needed for this optimization pass, sidebar is for input.
    with st.sidebar:
        st.markdown(f'<img src="{LEAN_LOGO_BASE64}" width="100" alt="Lean Institute Logo" style="display: block; margin: 0 auto 1rem;">', unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")

        with st.expander("🧪 Simulation Parameters", expanded=True):
            team_size_val = st.session_state.get('team_size', DEFAULT_CONFIG['TEAM_SIZE'])
            shift_duration_val = st.session_state.get('shift_duration', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'])
            team_size = st.slider("Team Size", 10, 100, team_size_val, key="sb_team_size_slider", help="Adjust the number of workers in the simulated shift.")
            shift_duration = st.slider("Shift Duration (min)", 200, 2000, shift_duration_val, step=2, key="sb_shift_duration_slider", help="Set the total length of the simulated work shift.")
            max_disrupt_time = shift_duration - 2 
            disruption_options = [i * 2 for i in range(max_disrupt_time // 2)] if max_disrupt_time > 0 else []
            default_disrupt_mins_raw = [i * 2 for i in DEFAULT_CONFIG.get('DISRUPTION_INTERVALS', [])]
            valid_default_disrupt_mins = [m for m in default_disrupt_mins_raw if m in disruption_options]
            _current_disrupt_selection_from_state = st.session_state.get('disruption_intervals_minutes')
            current_disrupt_selection_for_widget = []
            if _current_disrupt_selection_from_state is None: current_disrupt_selection_for_widget = valid_default_disrupt_mins
            elif not isinstance(_current_disrupt_selection_from_state, list): logger.warning(f"Session 'disruption_intervals_minutes' not a list. Resetting."); current_disrupt_selection_for_widget = valid_default_disrupt_mins
            else: current_disrupt_selection_for_widget = _current_disrupt_selection_from_state
            valid_current_disrupt_selection_for_widget = [m for m in current_disrupt_selection_for_widget if m in disruption_options]
            disruption_intervals_minutes = st.multiselect("Disruption Times (min)", disruption_options, valid_current_disrupt_selection_for_widget, key="sb_disruption_intervals_multiselect", help="Select specific times (in minutes from shift start) when disruptions will occur in the simulation.")
            team_initiative_opts = ["Standard Operations", "More frequent breaks", "Team recognition"] # Added "Standard Operations"
            current_initiative = st.session_state.get('team_initiative', team_initiative_opts[0])
            team_initiative_idx = team_initiative_opts.index(current_initiative) if current_initiative in team_initiative_opts else 0
            team_initiative = st.selectbox("Operational Initiative", team_initiative_opts, index=team_initiative_idx, key="sb_team_initiative_selectbox", help="Apply an operational strategy to observe its impact on metrics.")
            run_simulation_button = st.button("🚀 Run New Simulation", key="sb_run_simulation_button", type="primary", use_container_width=True)

        with st.expander("🎨 Visualization & Debug"):
            high_contrast = st.checkbox("High Contrast Plots", st.session_state.get('high_contrast', False), key="sb_high_contrast_checkbox", help="Use a high-contrast color theme for all charts to improve readability.")
            use_3d_distribution = st.checkbox("Enable 3D Worker View", st.session_state.get('use_3d_distribution', False), key="sb_use_3d_distribution_checkbox", help="Render worker positions in a 3D space (can impact performance on large datasets).")
            debug_mode = st.checkbox("Show Debug Info", st.session_state.get('debug_mode', False), key="sb_debug_mode_checkbox", help="Display raw configuration and internal state information for troubleshooting.")

        with st.expander("💾 Data Management & Export"):
            load_data_button = st.button("🔄 Load Previous Simulation", key="sb_load_data_button", use_container_width=True)
            can_gen_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("📄 Download Report (.tex)", key="sb_download_report_button", disabled=not can_gen_report, use_container_width=True, help="Generate a LaTeX (.tex) file summarizing key simulation results. Requires a LaTeX distribution (e.g., MiKTeX, TeX Live) to compile to PDF."):
                if can_gen_report:
                    try:
                        sim_res = st.session_state.simulation_results; num_steps = len(sim_res.get('downtime_minutes', []))
                        if num_steps == 0: st.warning("⚠️ No simulation data available to generate a report."); raise SystemExit 
                        pdf_data = {k: sim_res.get(k, [np.nan]*num_steps)[:num_steps] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'downtime_minutes', 'task_completion_rate']}
                        pdf_data.update({'task_compliance': sim_res.get('task_compliance', {}).get('data', [np.nan]*num_steps)[:num_steps], 'collaboration_proximity': sim_res.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps)[:num_steps], 'worker_wellbeing': sim_res.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps)[:num_steps], 'step': list(range(num_steps)), 'time_minutes': [i * 2 for i in range(num_steps)]})
                        generate_pdf_report(pd.DataFrame(pdf_data)); st.success("✅ LaTeX report (.tex) file 'workplace_report.tex' has been generated.")
                    except SystemExit: pass 
                    except Exception as e: logger.error(f"PDF Generation Error: {e}", exc_info=True); st.error(f"❌ PDF Generation Error: {e}")
            
            if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                sim_res_exp = st.session_state.simulation_results; num_steps_csv = len(sim_res_exp.get('downtime_minutes', []))
                if num_steps_csv > 0:
                    csv_data = {k: sim_res_exp.get(k, [np.nan]*num_steps_csv)[:num_steps_csv] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'downtime_minutes', 'task_completion_rate']}
                    csv_data.update({'task_compliance': sim_res_exp.get('task_compliance', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv], 'collaboration_proximity': sim_res_exp.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv], 'worker_wellbeing': sim_res_exp.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps_csv)[:num_steps_csv], 'step': list(range(num_steps_csv)), 'time_minutes': [i * 2 for i in range(num_steps_csv)]})
                    st.download_button("📥 Download Data (CSV)", pd.DataFrame(csv_data).to_csv(index=False).encode('utf-8'), "workplace_summary.csv", "text/csv", key="sb_download_csv_button", use_container_width=True)
                else: st.caption("No detailed data to export for CSV.") # Changed to caption for less intrusion
            else: st.caption("Run a simulation to enable data export.")

        if st.session_state.get('sb_debug_mode_checkbox', False): 
            with st.expander("🛠️ Debug Information", expanded=False): # Keep debug info collapsed by default
                st.write("**Default Config (Partial):**"); st.json({k: DEFAULT_CONFIG.get(k) for k in ['ENTRY_EXIT_POINTS', 'WORK_AREAS']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Active Simulation Config (from results):**"); st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else: st.write("**No active simulation data to show config from.**")

        st.markdown("## 📋 Help & Info")
        if st.button("ℹ️ Help & Glossary", key="sb_help_toggle_button", use_container_width=True): st.session_state.show_help_glossary = not st.session_state.get('show_help_glossary', False); st.rerun()
        if st.button("🚀 Quick Tour", key="sb_tour_toggle_button", use_container_width=True): st.session_state.show_tour = not st.session_state.get('show_tour', False); st.rerun()

    return team_size, shift_duration, disruption_intervals_minutes, team_initiative, run_simulation_button, load_data_button, high_contrast, use_3d_distribution, debug_mode

@st.cache_data(ttl=3600, show_spinner="⚙️ Running simulation model...") # Changed spinner text
def run_simulation_logic(team_size, shift_duration_minutes, disruption_intervals_minutes, team_initiative_selected):
    config = DEFAULT_CONFIG.copy(); config['TEAM_SIZE'] = team_size; config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // 2
    config['DISRUPTION_INTERVALS'] = sorted(list(set(m // 2 for m in disruption_intervals_minutes if isinstance(m, (int, float)) and m >= 0)))
    # Worker distribution logic (robust version from previous iteration)
    if 'WORK_AREAS' in config and isinstance(config['WORK_AREAS'], dict) and config['WORK_AREAS']:
        total_workers_in_config_zones = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())
        if total_workers_in_config_zones == 0 and team_size > 0: 
            num_zones_for_dist = len([zn for zn, zd in config['WORK_AREAS'].items() if zd.get('tasks_per_interval',0) > 0 or 'Warehouse' in zn or 'Assembly' in zn]); num_zones_for_dist = num_zones_for_dist if num_zones_for_dist > 0 else len(config['WORK_AREAS']) 
            if num_zones_for_dist > 0:
                workers_per_zone = team_size // num_zones_for_dist; remainder_workers = team_size % num_zones_for_dist
                zone_keys_for_dist = [zn for zn, zd in config['WORK_AREAS'].items() if zd.get('tasks_per_interval',0) > 0 or 'Warehouse' in zn or 'Assembly' in zn]
                if not zone_keys_for_dist: zone_keys_for_dist = list(config['WORK_AREAS'].keys())
                for i, zone_key in enumerate(zone_keys_for_dist): config['WORK_AREAS'][zone_key]['workers'] = workers_per_zone + (1 if i < remainder_workers else 0)
                all_zone_keys = list(config['WORK_AREAS'].keys()); [config['WORK_AREAS'][zk].update({'workers':0}) for zk in all_zone_keys if zk not in zone_keys_for_dist] # ensure others are 0
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

def get_actionable_insights(sim_data, config):
    insights = []
    # Compliance Insights
    compliance_avg = np.mean(safe_get(sim_data, 'task_compliance.data', [75])) # Default to target if no data
    if compliance_avg < 70: insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Average Task Compliance is {compliance_avg:.1f}%, significantly below target. Review disruption impacts and task complexities."})
    elif compliance_avg < 80: insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Average Task Compliance at {compliance_avg:.1f}%. Consider reviewing processes in low-performing intervals."})

    # Wellbeing Insights
    wellbeing_avg = np.mean(safe_get(sim_data, 'worker_wellbeing.scores', [70]))
    if wellbeing_avg < 60: insights.append({"type": "critical", "title": "Critical Worker Well-being", "text": f"Average Well-being is {wellbeing_avg:.1f}%. Urgent review of work conditions, break schedules, and stress factors needed."})
    elif len(safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])) > 3: insights.append({"type": "warning", "title": "Frequent Well-being Alerts", "text": "Multiple low well-being threshold alerts detected. Investigate specific triggers."})

    # Downtime Insights
    total_downtime = np.sum(safe_get(sim_data, 'downtime_minutes', [0]))
    downtime_threshold = config.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT', 60) # New config: total acceptable downtime
    if total_downtime > downtime_threshold : insights.append({"type": "critical", "title": "Excessive Total Downtime", "text": f"Total shift downtime is {total_downtime:.0f} minutes, exceeding the target of {downtime_threshold} min. Focus on disruption mitigation."})
    
    # Positive Insights
    if compliance_avg > 90 and wellbeing_avg > 80 and total_downtime < downtime_threshold / 2 :
        insights.append({"type": "positive", "title": "Excellent Operational Performance", "text": "Key metrics indicate a highly efficient and healthy shift. Current strategies are effective."})
    
    # Initiative Impact (Simple example)
    initiative = sim_data.get('config_params', {}).get('TEAM_INITIATIVE', 'Standard Operations')
    if initiative != "Standard Operations" and wellbeing_avg > 75 :
         insights.append({"type": "positive", "title": f"Positive Impact from '{initiative}'", "text": f"The '{initiative}' initiative appears to correlate with good worker well-being ({wellbeing_avg:.1f}%)."})

    return insights

def main():
    st.title("Workplace Shift Optimization Dashboard")
    # Simplified initialization - let widgets handle their defaults initially based on session_state.get(key, DEFAULT)
    for key in ['simulation_results', 'show_tour', 'show_help_glossary']:
        if key not in st.session_state: st.session_state[key] = None 
    
    sb_team_size, sb_shift_duration, sb_disrupt_mins, sb_team_initiative, sb_run_sim_btn, sb_load_data_btn, sb_high_contrast, sb_use_3d, sb_debug_mode = render_settings_sidebar()
    st.session_state.update(team_size=sb_team_size, shift_duration=sb_shift_duration, disruption_intervals_minutes=sb_disrupt_mins, team_initiative=sb_team_initiative, high_contrast=sb_high_contrast, use_3d_distribution=sb_use_3d, debug_mode=sb_debug_mode)
    
    _default_shift_duration = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']; _current_shift_duration = st.session_state.get('shift_duration', _default_shift_duration) or _default_shift_duration
    current_max_minutes_for_sliders = _current_shift_duration - 2; disruption_steps_for_plots = []
    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        num_steps_from_sim = len(st.session_state.simulation_results.get('downtime_minutes', []))
        if num_steps_from_sim > 0: current_max_minutes_for_sliders = (num_steps_from_sim - 1) * 2
        disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', [])
    else: _disrupt_mins_list = st.session_state.get('disruption_intervals_minutes', []); disruption_steps_for_plots = [m // 2 for m in _disrupt_mins_list if isinstance(m, (int, float))]
    current_max_minutes_for_sliders = max(0, current_max_minutes_for_sliders) 

    if sb_run_sim_btn: # Action based on button state
        with st.spinner("🚀 Simulating workplace operations... Please wait."):
            try: st.session_state.simulation_results = run_simulation_logic(st.session_state.team_size, st.session_state.shift_duration, st.session_state.disruption_intervals_minutes, st.session_state.team_initiative); st.success("✅ Simulation completed successfully!"); logger.info("Simulation run successful.", extra={'user_action': 'Run Simulation - Success'}); st.rerun() 
            except Exception as e: logger.error(f"Simulation Run Error: {e}", exc_info=True, extra={'user_action': 'Run Simulation - Error'}); st.error(f"❌ Simulation failed: {e}"); st.session_state.simulation_results = None 
    if sb_load_data_btn: # Action based on button state
        with st.spinner("🔄 Loading saved simulation data..."):
            try:
                loaded_data = load_simulation_data() 
                if loaded_data and isinstance(loaded_data, dict):
                    st.session_state.simulation_results = loaded_data; cfg = loaded_data.get('config_params', {})
                    # Update session state to reflect loaded simulation's parameters
                    st.session_state.update(team_size=cfg.get('TEAM_SIZE',st.session_state.team_size), shift_duration=cfg.get('SHIFT_DURATION_MINUTES',st.session_state.shift_duration), team_initiative=cfg.get('TEAM_INITIATIVE',st.session_state.team_initiative), disruption_intervals_minutes=cfg.get('DISRUPTION_INTERVALS_MINUTES',st.session_state.disruption_intervals_minutes))
                    st.success("✅ Data loaded successfully!"); logger.info("Saved data loaded successfully.", extra={'user_action': 'Load Data - Success'}); st.rerun() 
                else: st.error("❌ Failed to load data or data is not in the expected dictionary format."); logger.warning("Load data failed or invalid format.", extra={'user_action': 'Load Data - Fail/Invalid'})
            except Exception as e: logger.error(f"Load Data Error: {e}", exc_info=True, extra={'user_action': 'Load Data - Error'}); st.error(f"❌ Failed to load data: {e}"); st.session_state.simulation_results = None
    
    if st.session_state.get('show_tour'): # Tour Modal
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3><p>Welcome! ...</p></div>""", unsafe_allow_html=True) # Keep content brief for brevity
        if st.button("Got it!", key="end_tour_btn_modal", use_container_width=True): st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help_glossary'): # Help/Glossary Modal
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3><p>This dashboard ...</p><h4>Metric Definitions:</h4>...</div>""", unsafe_allow_html=True) # Keep content brief for brevity
        if st.button("Understood", key="close_help_glossary_btn_modal", use_container_width=True): st.session_state.show_help_glossary = False; st.rerun()

    tabs_main_names = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Insights", "⏱️ Downtime Analysis", "📖 Glossary"] # Updated Overview tab name
    tabs = st.tabs(tabs_main_names)
    
    def safe_get(data_dict, path_str, default_val=None): current = data_dict; default_return = default_val if default_val is not None else ([]) ; 
        if not isinstance(path_str, str) or not isinstance(data_dict, dict): return default_return
        for key in path_str.split('.'):
            if isinstance(current, dict): current = current.get(key)
            elif isinstance(current, (list, pd.Series)) and key.isdigit():
                try: idx = int(key); current = current[idx] if idx < len(current) else None
                except (ValueError, IndexError): current = None; break
            else: current = None; break
        return current if current is not None else default_return
    def safe_stat(data_list, stat_func, default_val=0.0):
        if not isinstance(data_list, (list, np.ndarray, pd.Series)): data_list = []
        valid_data = [x for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x))]; return stat_func(valid_data) if valid_data else default_val
    
    plot_config_interactive = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'pan2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}; plot_config_minimal = {'displayModeBar': False}

    # --- Overview & Insights Tab ---
    with tabs[0]: 
        st.header("📊 Key Performance Indicators & Actionable Insights", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; compliance_target=75; collab_target=60; wb_target=70; dt_target=30 # Define targets
            compliance = safe_stat(safe_get(sim_data, 'task_compliance.data', []), np.mean); proximity = safe_stat(safe_get(sim_data, 'collaboration_proximity.data', []), np.mean); wellbeing = safe_stat(safe_get(sim_data, 'worker_wellbeing.scores', []), np.mean); downtime = safe_stat(safe_get(sim_data, 'downtime_minutes', []), np.sum)
            
            cols_metrics = st.columns(4)
            cols_metrics[0].metric("Task Compliance", f"{compliance:.1f}%", f"{compliance-compliance_target:.1f}% vs Target {compliance_target}%")
            cols_metrics[1].metric("Collaboration Index", f"{proximity:.1f}%", f"{proximity-collab_target:.1f}% vs Target {collab_target}%")
            cols_metrics[2].metric("Worker Well-Being", f"{wellbeing:.1f}%", f"{wellbeing-wb_target:.1f}% vs Target {wb_target}%")
            cols_metrics[3].metric("Total Downtime", f"{downtime:.1f} min", f"{downtime-dt_target:.1f} min vs Target {dt_target}min", delta_color="inverse")

            try: # Gauge Charts
                summary_figs = plot_key_metrics_summary(compliance, proximity, wellbeing, downtime, st.session_state.get('high_contrast'))
                if summary_figs:
                    cols_gauges = st.columns(min(len(summary_figs), 4) or 1) 
                    for i, fig in enumerate(summary_figs):
                        with cols_gauges[i % len(cols_gauges)]: st.plotly_chart(fig, use_container_width=True, config=plot_config_minimal)
                else: st.caption("Gauge charts could not be generated for the overview metrics.")
            except Exception as e: logger.error(f"Overview Gauges Plotting Error: {e}", exc_info=True); st.error("⚠️ Error rendering overview gauges.")

            st.markdown("---") # Divider
            st.subheader("💡 Key Insights & Alerts")
            actionable_insights = get_actionable_insights(sim_data, DEFAULT_CONFIG) # Pass relevant config like thresholds
            if actionable_insights:
                for insight in actionable_insights:
                    alert_class = f"alert-{insight['type']}" # e.g., alert-critical
                    st.markdown(f'<div class="{alert_class}"><p class="insight-title">{insight["title"]}</p><p class="insight-text">{insight["text"]}</p></div>', unsafe_allow_html=True)
            else:
                st.info("✅ No critical alerts or specific insights identified based on current thresholds. Overall performance appears stable.")

            with st.expander("View Detailed Overview Data Table", expanded=False):
                num_s = len(safe_get(sim_data, 'downtime_minutes', []))
                if num_s > 0:
                    df_data = {'Time (min)': [i*2 for i in range(num_s)]}; df_data.update({'Task Compliance (%)': safe_get(sim_data, 'task_compliance.data', [np.nan]*num_s)[:num_s], 'Collaboration (%)': safe_get(sim_data, 'collaboration_proximity.data', [np.nan]*num_s)[:num_s], 'Well-Being (%)': safe_get(sim_data, 'worker_wellbeing.scores', [np.nan]*num_s)[:num_s], 'Downtime (min)': safe_get(sim_data, 'downtime_minutes', [np.nan]*num_s)[:num_s]})
                    st.dataframe(pd.DataFrame(df_data).style.format("{:.1f}", na_rep="-").set_table_styles([{'selector': 'th', 'props': [('background-color', '#293344'), ('color', '#EAEAEA')]}]), use_container_width=True, height=300)
                else: st.caption("No detailed overview data available to display.")
        else: st.info("ℹ️ Run a simulation or load data to view the Overview & Insights.", icon="📊")

    # --- Operational Metrics Tab --- (Slider initialization pattern repeated)
    with tabs[1]: 
        st.header("📈 Operational Performance Trends", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; op_time_slider_key = "op_metrics_time_slider"; op_time_val_key = "op_metrics_time_slider_val"
            default_op_time_val = (0, current_max_minutes_for_sliders); op_time_val_from_state = st.session_state.get(op_time_val_key)
            if op_time_val_from_state is None or not (isinstance(op_time_val_from_state, tuple) and len(op_time_val_from_state) == 2): op_time_val = default_op_time_val; st.session_state[op_time_val_key] = op_time_val
            else: op_time_val = op_time_val_from_state
            op_time_val = (min(op_time_val[0], current_max_minutes_for_sliders), min(op_time_val[1], current_max_minutes_for_sliders)); op_time_val = (op_time_val[0], max(op_time_val[0], op_time_val[1])) # Ensure end >= start
            time_range = st.slider("Select Time Range (minutes):", 0, current_max_minutes_for_sliders, op_time_val, 2, key=op_time_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({op_time_val_key: st.session_state[op_time_slider_key]}))
            start_idx, end_idx = time_range[0]//2, time_range[1]//2 + 1; filt_disrupt_steps = [s for s in disruption_steps_for_plots if start_idx <= s < end_idx]

            # ... (Rest of plotting for Operational Metrics tab remains structurally similar but uses plot_config_interactive) ...
            # Example for Task Compliance Plot:
            tc_data_list = safe_get(sim_data, 'task_compliance.data', [])[start_idx:end_idx]
            if tc_data_list:
                with st.container(border=True): 
                    st.markdown('<h5>Task Compliance Score Over Time</h5>', unsafe_allow_html=True)
                    try: 
                        tc_z = safe_get(sim_data, 'task_compliance.z_scores', [])[start_idx:end_idx]
                        tc_f = safe_get(sim_data, 'task_compliance.forecast', [])[start_idx:end_idx] if safe_get(sim_data, 'task_compliance.forecast', []) else None
                        st.plotly_chart(plot_task_compliance_score(tc_data_list, filt_disrupt_steps, tc_f, tc_z, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                    except Exception as e: logger.error(f"Op Tab TC Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Task Compliance.")
            else: st.caption("No Task Compliance data for this time range.") # Using caption for no data message

            cp_data_list = safe_get(sim_data, 'collaboration_proximity.data', [])[start_idx:end_idx]
            if cp_data_list:
                with st.container(border=True):
                    st.markdown('<h5>Collaboration Proximity Index Over Time</h5>', unsafe_allow_html=True)
                    try:
                        cp_f = safe_get(sim_data, 'collaboration_proximity.forecast', [])[start_idx:end_idx] if safe_get(sim_data, 'collaboration_proximity.forecast', []) else None
                        st.plotly_chart(plot_collaboration_proximity_index(cp_data_list, filt_disrupt_steps, cp_f, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                    except Exception as e: logger.error(f"Op Tab CP Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Collaboration Index.")
            else: st.caption("No Collaboration Index data for this time range.")

            with st.expander("Show Additional Operational Metrics", expanded=False):
                or_data_list = safe_get(sim_data, 'operational_recovery', [])[start_idx:end_idx]
                if or_data_list:
                    with st.container(border=True):
                        st.markdown('<h5>Operational Recovery & Resilience</h5>', unsafe_allow_html=True)
                        try: 
                            pl_data = safe_get(sim_data, 'productivity_loss', [])[start_idx:end_idx]
                            st.plotly_chart(plot_operational_recovery(or_data_list, pl_data, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                        except Exception as e: logger.error(f"Op Tab OR Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Operational Recovery.")
                else: st.caption("No Operational Recovery data for this time range.")
                
                eff_df_full = safe_get(sim_data, 'efficiency_metrics_df', pd.DataFrame())
                if not eff_df_full.empty:
                    with st.container(border=True):
                        st.markdown('<h5>Overall Equipment Effectiveness (OEE) & Components</h5>', unsafe_allow_html=True)
                        try:
                            sel_metrics = st.multiselect("Select Efficiency Metrics:", ['uptime', 'throughput', 'quality', 'oee'], default=['uptime', 'throughput', 'quality', 'oee'], key="eff_metrics_multiselect_op_tab")
                            filt_eff_df = eff_df_full.iloc[start_idx:end_idx] if isinstance(eff_df_full.index, pd.RangeIndex) and end_idx <= len(eff_df_full) else eff_df_full 
                            if not filt_eff_df.empty: 
                                st.plotly_chart(plot_operational_efficiency(filt_eff_df, sel_metrics, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            else: st.info("No OEE data available for the selected time range.")
                        except Exception as e: logger.error(f"Op Tab OEE Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting OEE.")
                else: st.caption("No OEE data available.")
        else: st.info("ℹ️ Run a simulation or load data to view Operational Metrics.", icon="📈")

    # --- Worker Insights Tab --- (Apply similar slider init and no-data captions)
    with tabs[2]: 
        st.header("👥 Worker Dynamics & Well-being", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; team_pos_df_all = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
            worker_time_slider_key = "worker_insights_time_slider"; worker_time_val_key = "worker_insights_time_slider_val"
            default_worker_time_val = (0, current_max_minutes_for_sliders); worker_time_val_from_state = st.session_state.get(worker_time_val_key)
            if worker_time_val_from_state is None or not (isinstance(worker_time_val_from_state, tuple) and len(worker_time_val_from_state) == 2): worker_time_val = default_worker_time_val; st.session_state[worker_time_val_key] = worker_time_val
            else: worker_time_val = worker_time_val_from_state
            worker_time_val = (min(worker_time_val[0], current_max_minutes_for_sliders), min(worker_time_val[1], current_max_minutes_for_sliders)); worker_time_val = (worker_time_val[0], max(worker_time_val[0], worker_time_val[1]))
            shared_time_range = st.slider("Select Time Range (minutes) for Worker Insights:", 0, current_max_minutes_for_sliders, worker_time_val, 2, key=worker_time_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({worker_time_val_key: st.session_state[worker_time_slider_key]}))
            shared_start_idx, shared_end_idx = shared_time_range[0]//2, shared_time_range[1]//2 + 1
            # ... (rest of Worker Insights plotting with no-data captions and try-except)
            with st.expander("Worker Distribution Analysis", expanded=True):
                zones = ["All"] + list(DEFAULT_CONFIG.get('WORK_AREAS', {}).keys()); zone_sel = st.selectbox("Filter by Zone (for Distribution & Density):", zones, key="worker_zone_selectbox_dist_tab")
                filt_team_pos_df = team_pos_df_all
                if not filt_team_pos_df.empty: filt_team_pos_df = filt_team_pos_df[(filt_team_pos_df['step'] >= shared_start_idx) & (filt_team_pos_df['step'] < shared_end_idx)]; 
                if zone_sel != "All" and not filt_team_pos_df.empty : filt_team_pos_df = filt_team_pos_df[filt_team_pos_df['zone'] == zone_sel]
                show_ee = st.checkbox("Show Entry/Exit Points on plots", True, key="worker_show_ee_checkbox_dist_tab"); show_pl = st.checkbox("Show Production Lines on plots", True, key="worker_show_pl_checkbox_dist_tab")
                cols_dist = st.columns(2)
                with cols_dist[0]:
                    st.markdown("<h5>Worker Positions (Time Snapshot)</h5>", unsafe_allow_html=True)
                    min_step, max_step = shared_start_idx, max(shared_start_idx, shared_end_idx -1)
                    snap_slider_key="wsnap_slider"; snap_val_key="wsnap_val"; default_snap=min_step
                    snap_val=st.session_state.get(snap_val_key,default_snap)
                    if snap_val is None or not isinstance(snap_val, int): snap_val = default_snap; st.session_state[snap_val_key] = snap_val
                    snap_val = max(min_step, min(snap_val, max_step)) # clamp
                    snap_step = st.slider("Select Time Step for Snapshot:", min_step, max_step, snap_val, 1, key=snap_slider_key, disabled=max_step < min_step, on_change=lambda:st.session_state.update({snap_val_key: st.session_state[snap_slider_key]}))
                    if not team_pos_df_all.empty and max_step >= min_step :
                        with st.container(border=True):
                            try: st.plotly_chart(plot_worker_distribution(team_pos_df_all, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, st.session_state.get('use_3d_distribution', False), snap_step, show_ee, show_pl, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            except Exception as e: logger.error(f"Worker Dist Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Worker Positions.")
                    else: st.caption("No data for positions snapshot with current filters.")
                with cols_dist[1]:
                    st.markdown("<h5>Worker Density Heatmap (Aggregated)</h5>", unsafe_allow_html=True)
                    if not filt_team_pos_df.empty:
                        with st.container(border=True):
                            try: st.plotly_chart(plot_worker_density_heatmap(filt_team_pos_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, show_ee, show_pl, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            except Exception as e: logger.error(f"Worker Heatmap Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Density Heatmap.")
                    else: st.caption("No data for density heatmap with current filters.")
            with st.expander("Worker Well-Being & Safety Analysis", expanded=True):
                cols_well = st.columns(2)
                with cols_well[0]:
                    st.markdown("<h5>Worker Well-Being Index</h5>", unsafe_allow_html=True)
                    ww_scores_list = safe_get(sim_data, 'worker_wellbeing.scores', [])[shared_start_idx:shared_end_idx]
                    if ww_scores_list:
                        with st.container(border=True):
                            try: ww_trigs_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {}); ww_trigs_filt = {k: [t for t in v if shared_start_idx <= t < shared_end_idx] for k, v in ww_trigs_raw.items() if isinstance(v, list)}; ww_trigs_filt['work_area'] = {wk: [t for t in wv if shared_start_idx <= t < shared_end_idx] for wk, wv in ww_trigs_raw.get('work_area',{}).items()}; st.plotly_chart(plot_worker_wellbeing(ww_scores_list, ww_trigs_filt, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            except Exception as e: logger.error(f"Wellbeing Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Well-Being Index.")
                    else: st.caption("No Well-Being Index data for this range.")
                with cols_well[1]:
                    st.markdown("<h5>Psychological Safety Score</h5>", unsafe_allow_html=True)
                    ps_scores_list = safe_get(sim_data, 'psychological_safety', [])[shared_start_idx:shared_end_idx]
                    if ps_scores_list:
                        with st.container(border=True):
                            try: st.plotly_chart(plot_psychological_safety(ps_scores_list, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            except Exception as e: logger.error(f"Psych Safety Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Psychological Safety.")
                    else: st.caption("No Psych. Safety data for this range.")
                st.markdown("<h6>Well-Being Triggers (within selected time range):</h6>", unsafe_allow_html=True)
                ww_trigs_disp_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {}); ww_trigs_disp_filt = {k: [t for t in v if shared_start_idx <= t < shared_end_idx] for k, v in ww_trigs_disp_raw.items() if isinstance(v, list)}; ww_trigs_disp_filt['work_area'] = {wk: [t for t in wv if shared_start_idx <= t < shared_end_idx] for wk, wv in ww_trigs_disp_raw.get('work_area', {}).items()}
                st.caption(f"**Threshold Alerts (< {DEFAULT_CONFIG.get('WELLBEING_THRESHOLD',0)*100}%):** {ww_trigs_disp_filt.get('threshold', 'None') or 'None'}"); st.caption(f"**Trend Alerts (Declining):** {ww_trigs_disp_filt.get('trend', 'None')or 'None'}"); st.caption(f"**Disruption-Related Alerts:** {ww_trigs_disp_filt.get('disruption', 'None') or 'None'}")
                st.caption("**Work Area Specific Alerts:**"); wa_alert_found = False
                for zone, trigs in ww_trigs_disp_filt.get('work_area', {}).items():
                    if trigs: st.caption(f"    {zone}: {trigs}"); wa_alert_found = True
                if not wa_alert_found: st.caption("    None")
        else: st.info("ℹ️ Run a simulation or load data to view Worker Insights.", icon="👥")


    # --- Downtime Tab --- (Apply similar slider init and no-data captions)
    with tabs[3]: 
        st.header("⏱️ Downtime Analysis", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results
            dt_time_slider_key = "downtime_tab_time_slider"; dt_time_val_key = "downtime_tab_time_slider_val"
            default_dt_time_val = (0, current_max_minutes_for_sliders); dt_time_val_from_state = st.session_state.get(dt_time_val_key)
            if dt_time_val_from_state is None or not (isinstance(dt_time_val_from_state, tuple) and len(dt_time_val_from_state) == 2): dt_time_val = default_dt_time_val; st.session_state[dt_time_val_key] = dt_time_val
            else: dt_time_val = dt_time_val_from_state
            dt_time_val = (min(dt_time_val[0], current_max_minutes_for_sliders), min(dt_time_val[1], current_max_minutes_for_sliders)); dt_time_val = (dt_time_val[0], max(dt_time_val[0], dt_time_val[1]))
            time_range_dt = st.slider("Select Time Range (minutes):", 0, current_max_minutes_for_sliders, dt_time_val, 2, key=dt_time_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({dt_time_val_key: st.session_state[dt_time_slider_key]}))
            dt_start_idx, dt_end_idx = time_range_dt[0]//2, time_range_dt[1]//2 + 1
            dt_data_list = safe_get(sim_data, 'downtime_minutes', [])[dt_start_idx:dt_end_idx]

            if dt_data_list:
                total_downtime_period = sum(dt_data_list)
                avg_downtime_period = np.mean(dt_data_list) if dt_data_list else 0
                st.metric("Total Downtime in Period", f"{total_downtime_period:.1f} min", help="Sum of all downtime instances in the selected time frame.")
                
                with st.container(border=True): 
                    st.markdown('<h5>Downtime Trend Over Time</h5>', unsafe_allow_html=True)
                    try: 
                        st.plotly_chart(plot_downtime_trend(dt_data_list, DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD', 10), st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                    except Exception as e: logger.error(f"Downtime Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Downtime Trend.")
            else: st.caption("No Downtime data available for the selected time range.")
        else: st.info("ℹ️ Run a simulation or load data to view Downtime Analysis.", icon="⏱️")
        
    with tabs[4]: 
        st.header("📖 Glossary of Terms", divider="blue")
        st.markdown("""
            <div style="font-size: 0.95rem; line-height: 1.7;">
            <p>This glossary defines key metrics used throughout the dashboard to help you understand the operational insights provided. For a combined view with general help, click the "ℹ️ Help & Glossary" button in the sidebar.</p>
            <details><summary><strong>Task Compliance Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of tasks completed correctly and within the allocated time. It measures adherence to operational protocols and standards. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Collaboration Proximity Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of workers observed within a defined proximity (e.g., 5 meters) of their colleagues. This index suggests opportunities for teamwork, communication, and knowledge sharing. <em>Range: 0-100%. Optimal levels vary.</em></p></details>
            <details><summary><strong>Operational Recovery Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A measure of the system's ability to return to and maintain target output levels after experiencing disruptions. It reflects operational resilience. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Worker Well-Being Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A composite score derived from simulated factors such as fatigue, stress levels, and job satisfaction. It provides an indicator of overall worker health and morale. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Psychological Safety Score</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">An estimate of the perceived comfort level among workers to report issues, voice concerns, or suggest improvements without fear of negative consequences. It indicates a supportive and open work environment. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Uptime</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of scheduled operational time that equipment or a system is available and functioning correctly. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Throughput</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The rate at which a system processes work or produces output, often expressed as a percentage of its maximum potential or target rate. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Quality Rate</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of products or outputs that meet predefined quality standards, free of defects. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>OEE (Overall Equipment Effectiveness)</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A comprehensive metric calculated as (Uptime × Throughput × Quality Rate). It provides a holistic view of operational performance and efficiency. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Productivity Loss</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of potential output or operational time lost due to inefficiencies, disruptions, downtime, or substandard performance. <em>Range: 0-100%. Lower is better.</em></p></details>
            <details><summary><strong>Downtime (per interval)</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The total duration (in minutes) of unplanned operational stops or non-productive time within each measured time interval. Tracks interruptions to workflow. <em>Lower is better.</em></p></details>
            <details><summary><strong>Task Completion Rate</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">The percentage of assigned tasks that are successfully completed within a given time interval. Measures task throughput and efficiency over time. <em>Range: 0-100%. Higher is better.</em></p></details>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
