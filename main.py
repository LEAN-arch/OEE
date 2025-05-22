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
    plot_downtime_trend, plot_team_cohesion, plot_perceived_workload # Added new imports
)
from simulation import simulate_workplace_operations
from utils import save_simulation_data, load_simulation_data, generate_pdf_report

LEAN_LOGO_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgG6NcxuAAAAAElFTkSuQmCC"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]', filename='dashboard.log', filemode='a')
logger.info("Main.py: Parsed imports and logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.1\nInsights for operational excellence."})

st.markdown("""
    <style>
        /* Base Styles */
        .main { background-color: #121828; color: #EAEAEA; font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif; padding: 2rem; }
        h1 { font-size: 2.4rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; text-align: center; margin-bottom: 2rem; color: #FFFFFF; }
        h2 { /* Tab Headers */ font-size: 1.75rem; font-weight: 600; line-height: 1.3; margin: 1.5rem 0 1rem; color: #D0D0D0; border-bottom: 1px solid #4A5568; padding-bottom: 0.5rem;}
        h3 { /* Expander Titles / Section Subtitles */ font-size: 1.3rem; font-weight: 500; line-height: 1.4; margin-bottom: 0.75rem; color: #C0C0C0;}
        h5 { /* Plot Titles inside containers */ font-size: 1.05rem; font-weight: 500; line-height: 1.3; margin: 0.25rem 0 0.75rem; color: #B0B0B0; text-align: center;}
        h6 { /* Sub-notes or trigger list titles */ font-size: 0.9rem; font-weight: 500; line-height: 1.3; margin: 0.75rem 0 0.25rem; color: #A0A0A0;}
        .stButton>button { background-color: #4F46E5; color: #FFFFFF; border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.95rem; font-weight: 500; transition: all 0.2s ease-in-out; border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stButton>button:hover, .stButton>button:focus { background-color: #6366F1; transform: translateY(-1px); box-shadow: 0 3px 7px rgba(0,0,0,0.2); outline: none; }
        .stButton>button:disabled { background-color: #374151; color: #9CA3AF; cursor: not-allowed; box-shadow: none; }
        .stSelectbox div[data-baseweb="select"], .stSlider div[data-testid="stTickBar"], .stMultiSelect div[data-baseweb="select"] { background-color: #1F2937; color: #EAEAEA; border-radius: 6px; padding: 0.5rem; margin-bottom: 1rem; font-size: 0.95rem; border: 1px solid #374151; }
        .stSlider div[data-testid="stTickBar"] > div { background-color: #4A5568; }
        .stSlider div[role="slider"] { background-color: #4F46E5; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.3); }
        [data-testid="stSidebar"] { background-color: #1F2937; color: #EAEAEA; padding: 1.5rem; border-right: 1px solid #374151; font-size: 0.95rem; }
        [data-testid="stSidebar"] .stButton>button { background-color: #34D399; width: 100%; margin-bottom: 0.5rem; }
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus { background-color: #6EE7B7; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #EAEAEA; border-bottom: 1px solid #4A5568; margin-top:1rem;}
        .stMetric { background-color: #1F2937; border-radius: 8px; padding: 1.25rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 1.05rem; border: 1px solid #374151;}
        .stMetric > div > div > div { font-size: 1.8rem !important; color: #FFFFFF !important; } 
        .stMetric > div > div > p { font-size: 0.9rem !important; color: #A0A0A0 !important; } 
        .stMetric > div:nth-child(2) > div { font-size: 0.85rem !important; } 
        .stExpander { background-color: #1F2937; border-radius: 8px; margin: 1rem 0; border: 1px solid #374151; }
        .stExpander header { font-size: 1rem; font-weight: 500; color: #E0E0E0; padding: 0.5rem 1rem; }
        .stExpander div[role="button"] { padding: 0.75rem !important; }
        .stTabs [data-baseweb="tab-list"] { background-color: #1F2937; border-radius: 8px; padding: 0.5rem; display: flex; justify-content: center; gap: 0.5rem; border-bottom: 2px solid #374151;}
        .stTabs [data-baseweb="tab"] { color: #D1D5DB; padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 500; font-size: 0.95rem; transition: all 0.2s ease-in-out; border: none; border-bottom: 2px solid transparent; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: transparent; color: #4F46E5; border-bottom: 2px solid #4F46E5; font-weight:600; }
        .stTabs [data-baseweb="tab"]:hover { background-color: #374151; color: #FFFFFF; }
        .plot-container { background-color: #1F2937; border-radius: 8px; padding: 1rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #374151;}
        .stPlotlyChart { border-radius: 6px; } 
        .stDataFrame { border-radius: 8px; font-size: 0.875rem; border: 1px solid #374151; }
        .stDataFrame thead th { background-color: #293344; color: #EAEAEA; font-weight: 600; }
        .stDataFrame tbody tr:nth-child(even) { background-color: #222C3D; }
        .stDataFrame tbody tr:hover { background-color: #374151; }
        @media (max-width: 768px) { .main { padding: 1rem; } h1 { font-size: 1.8rem; } h2 { font-size: 1.4rem; } h3 { font-size: 1.1rem; } .stPlotlyChart { min-height: 300px !important; } .stTabs [data-baseweb="tab"] { padding: 0.5rem 0.8rem; font-size: 0.85rem; } }
        .spinner { display: flex; justify-content: center; align-items: center; height: 100px; }
        .spinner::after { content: ''; width: 40px; height: 40px; border: 4px solid #4A5568; border-top: 4px solid #4F46E5; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .onboarding-modal { background-color: #1F2937; border: 1px solid #374151; border-radius: 8px; padding: 1.5rem; max-width: 550px; margin: 2rem auto; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .onboarding-modal h3 { color: #EAEAEA; margin-bottom: 1rem; text-align: center; }
        .onboarding-modal p, .onboarding-modal ul { color: #D1D5DB; line-height: 1.6; margin-bottom: 1rem; font-size: 0.9rem; }
        .onboarding-modal ul { list-style-position: inside; padding-left: 0.5rem; }
        .alert-critical { border-left: 5px solid #F87171; background-color: rgba(248, 113, 113, 0.05); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; } 
        .alert-warning { border-left: 5px solid #FACC15; background-color: rgba(250, 204, 21, 0.05); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }
        .alert-positive { border-left: 5px solid #22D3EE; background-color: rgba(34, 211, 238, 0.05); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }
        .alert-info { border-left: 5px solid #60A5FA; background-color: rgba(96, 165, 250, 0.05); padding: 0.75rem; margin-bottom: 1rem; border-radius: 4px; }
        .insight-title { font-weight: 600; color: #EAEAEA; margin-bottom: 0.25rem;}
        .insight-text { font-size: 0.9rem; color: #D1D5DB;}
    </style>
""", unsafe_allow_html=True)

def render_settings_sidebar():
    with st.sidebar:
        st.markdown(f'<img src="{LEAN_LOGO_BASE64}" width="100" alt="Lean Institute Logo" style="display: block; margin: 0 auto 1rem;">', unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")
        with st.expander("🧪 Simulation Parameters", expanded=True):
            team_size = st.slider("Team Size", 10, 100, st.session_state.get('sb_team_size_slider', DEFAULT_CONFIG['TEAM_SIZE']), key="sb_team_size_slider", help="Adjust the number of workers in the simulated shift.")
            shift_duration = st.slider("Shift Duration (min)", 200, 2000, st.session_state.get('sb_shift_duration_slider', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']), step=2, key="sb_shift_duration_slider", help="Set the total length of the simulated work shift.")
            max_disrupt_time = shift_duration - 2 
            disruption_options = [i * 2 for i in range(max_disrupt_time // 2)] if max_disrupt_time > 0 else []
            default_disrupt_mins_raw = DEFAULT_CONFIG.get('DISRUPTION_TIMES_MINUTES', []) 
            valid_default_disrupt_mins = [m for m in default_disrupt_mins_raw if m in disruption_options]
            session_value_for_disruptions = st.session_state.get('sb_disruption_intervals_multiselect')
            current_selection_for_multiselect = [] 
            if session_value_for_disruptions is None: default_for_multiselect = valid_default_disrupt_mins
            elif not isinstance(session_value_for_disruptions, list): logger.warning(f"Sidebar: 'sb_disruption_intervals_multiselect' in session_state was {type(session_value_for_disruptions)}. Expected list. Using default."); default_for_multiselect = valid_default_disrupt_mins
            else: default_for_multiselect = [m for m in session_value_for_disruptions if m in disruption_options]
            disruption_intervals_minutes = st.multiselect("Disruption Times (min)", disruption_options, default=default_for_multiselect, key="sb_disruption_intervals_multiselect", help="Select specific times (in minutes from shift start) when disruptions will occur in the simulation.")
            team_initiative_opts = ["Standard Operations", "More frequent breaks", "Team recognition", "Increased Autonomy"] # Added new initiative
            current_initiative = st.session_state.get('sb_team_initiative_selectbox', team_initiative_opts[0]) 
            team_initiative_idx = team_initiative_opts.index(current_initiative) if current_initiative in team_initiative_opts else 0
            team_initiative = st.selectbox("Operational Initiative", team_initiative_opts, index=team_initiative_idx, key="sb_team_initiative_selectbox", help="Apply an operational strategy to observe its impact on metrics.")
            run_simulation_button = st.button("🚀 Run New Simulation", key="sb_run_simulation_button", type="primary", use_container_width=True)
        with st.expander("🎨 Visualization Options"):
            high_contrast = st.checkbox("High Contrast Plots", st.session_state.get('sb_high_contrast_checkbox', False), key="sb_high_contrast_checkbox", help="Applies a high-contrast color theme to all charts for better accessibility.")
            use_3d_distribution = st.checkbox("Enable 3D Worker View", st.session_state.get('sb_use_3d_distribution_checkbox', False), key="sb_use_3d_distribution_checkbox", help="Renders worker positions in a 3D scatter plot.")
            debug_mode = st.checkbox("Show Debug Info", st.session_state.get('sb_debug_mode_checkbox', False), key="sb_debug_mode_checkbox", help="Display additional debug information in the sidebar.")
        with st.expander("💾 Data Management & Export"):
            load_data_button = st.button("🔄 Load Previous Simulation", key="sb_load_data_button", use_container_width=True)
            can_gen_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("📄 Download Report (.tex)", key="sb_pdf_button", disabled=not can_gen_report, use_container_width=True, help="Generates a LaTeX (.tex) file summarizing the simulation. Requires a LaTeX distribution to compile to PDF."):
                if can_gen_report:
                    try:
                        sim_res = st.session_state.simulation_results; num_steps = len(sim_res.get('downtime_minutes', []))
                        if num_steps == 0: st.warning("⚠️ No simulation data available to generate a report."); raise SystemExit 
                        pdf_data = {k: sim_res.get(k, [np.nan]*num_steps)[:num_steps] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'downtime_minutes', 'task_completion_rate']}
                        pdf_data.update({'task_compliance': sim_res.get('task_compliance', {}).get('data', [np.nan]*num_steps)[:num_steps], 'collaboration_proximity': sim_res.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps)[:num_steps], 'worker_wellbeing': sim_res.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps)[:num_steps], 'step': list(range(num_steps)), 'time_minutes': [i * 2 for i in range(num_steps)]})
                        generate_pdf_report(pd.DataFrame(pdf_data)); st.success("✅ LaTeX report (.tex) file 'workplace_report.tex' has been generated.")
                    except SystemExit: pass 
                    except Exception as e: logger.error(f"PDF Generation Error: {e}", exc_info=True); st.error(f"❌ PDF Generation Error: {e}")
            if 'simulation_results' in st.session_state and st.session_state.simulation_results :
                sim_res_exp = st.session_state.simulation_results; num_steps_csv = len(sim_res_exp.get('downtime_minutes', []))
                if num_steps_csv > 0:
                    csv_data = {k: sim_res_exp.get(k, [np.nan]*num_steps_csv)[:num_steps_csv] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'downtime_minutes', 'task_completion_rate']}
                    # Include new psychosocial metrics from worker_wellbeing dict if they exist
                    ww_data = sim_res_exp.get('worker_wellbeing', {})
                    csv_data['team_cohesion'] = ww_data.get('team_cohesion_scores', [np.nan]*num_steps_csv)[:num_steps_csv]
                    csv_data['perceived_workload'] = ww_data.get('perceived_workload_scores', [np.nan]*num_steps_csv)[:num_steps_csv]
                    
                    csv_data.update({'task_compliance': sim_res_exp.get('task_compliance', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv], 'collaboration_proximity': sim_res_exp.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv], 'worker_wellbeing_index': ww_data.get('scores', [np.nan]*num_steps_csv)[:num_steps_csv], 'step': list(range(num_steps_csv)), 'time_minutes': [i * 2 for i in range(num_steps_csv)]})
                    st.download_button("📥 Download Data (CSV)", pd.DataFrame(csv_data).to_csv(index=False).encode('utf-8'), "workplace_summary.csv", "text/csv", key="sb_csv_dl_button", use_container_width=True)
                else: st.caption("No detailed data to export for CSV.") 
            elif not can_gen_report : st.caption("Run simulation for export.")
        if st.session_state.get('sb_debug_mode_checkbox', False): 
            with st.expander("🛠️ Debug Information", expanded=False): 
                st.write("**Default Config (Partial):**"); st.json({k: DEFAULT_CONFIG.get(k) for k in ['ENTRY_EXIT_POINTS', 'WORK_AREAS', 'DISRUPTION_TIMES_MINUTES']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Active Simulation Config (from results):**"); st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else: st.write("**No active simulation data to show config from.**")
        st.markdown("## 📋 Help & Info")
        if st.button("ℹ️ Help & Glossary", key="sb_help_button", use_container_width=True): st.session_state.show_help_glossary = not st.session_state.get('show_help_glossary', False); st.rerun() 
        if st.button("🚀 Quick Tour", key="sb_tour_button", use_container_width=True): st.session_state.show_tour = not st.session_state.get('show_tour', False); st.rerun() 
    return (st.session_state.sb_team_size_slider, st.session_state.sb_shift_duration_slider, 
            st.session_state.sb_disruption_intervals_multiselect, st.session_state.sb_team_initiative_selectbox,
            run_simulation_button, load_data_button, 
            st.session_state.sb_high_contrast_checkbox, st.session_state.sb_use_3d_distribution_checkbox, 
            st.session_state.sb_debug_mode_checkbox)

@st.cache_data(ttl=3600, show_spinner="⚙️ Running simulation model...")
def run_simulation_logic(team_size, shift_duration_minutes, disruption_intervals_minutes_param, team_initiative_selected):
    config = DEFAULT_CONFIG.copy(); config['TEAM_SIZE'] = team_size; config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // 2
    if not isinstance(disruption_intervals_minutes_param, list): logger.error(f"run_simulation_logic received non-list for disruption_intervals_minutes_param: {type(disruption_intervals_minutes_param)}. Value: {disruption_intervals_minutes_param}. Defaulting to empty list."); disruption_intervals_minutes_param = []
    config['DISRUPTION_EVENT_STEPS'] = sorted(list(set(m // 2 for m in disruption_intervals_minutes_param if isinstance(m, (int, float)) and m >= 0)))
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
            if sorted_zone_keys: last_zone_key = sorted_zone_keys[-1]; config['WORK_AREAS'][last_zone_key]['workers'] = team_size - accumulated_workers
            elif team_size > 0: logger.warning("Team size > 0 but no work zones defined for worker assignment after ratio.")
        elif team_size == 0: 
            for zone_key in config['WORK_AREAS']: config['WORK_AREAS'][zone_key]['workers'] = 0
    validate_config(config) 
    logger.info(f"Running simulation with: Team Size={team_size}, Duration={shift_duration_minutes}min, Disruptions(min): {disruption_intervals_minutes_param}, Disruption Steps: {config['DISRUPTION_EVENT_STEPS']}, Initiative: {team_initiative_selected}", extra={'user_action': 'Run Simulation'})
    sim_results_tuple = simulate_workplace_operations(num_team_members=team_size, num_steps=config['SHIFT_DURATION_INTERVALS'], disruption_event_steps=config['DISRUPTION_EVENT_STEPS'], team_initiative=team_initiative_selected, config=config)
    expected_keys = ['team_positions_df', 'task_compliance', 'collaboration_proximity', 'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 'worker_wellbeing', 'psychological_safety', 'feedback_impact', 'downtime_minutes', 'task_completion_rate']
    if not isinstance(sim_results_tuple, tuple) or len(sim_results_tuple) != len(expected_keys):
        err_msg = f"Simulation returned unexpected data format. Expected tuple of {len(expected_keys)} items, got {type(sim_results_tuple)} with length {len(sim_results_tuple) if isinstance(sim_results_tuple, (list,tuple)) else 'N/A'}."; logger.critical(err_msg, extra={'user_action': 'Run Simulation - CRITICAL Data Format Error'}); raise TypeError(err_msg)
    simulation_output_dict = dict(zip(expected_keys, sim_results_tuple))
    simulation_output_dict['config_params'] = {'TEAM_SIZE': team_size, 'SHIFT_DURATION_MINUTES': shift_duration_minutes, 'DISRUPTION_INTERVALS_MINUTES': disruption_intervals_minutes_param, 'DISRUPTION_EVENT_STEPS': config['DISRUPTION_EVENT_STEPS'], 'TEAM_INITIATIVE': team_initiative_selected}
    save_simulation_data(simulation_output_dict); return simulation_output_dict

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
    if not isinstance(data_list, (list, np.ndarray, pd.Series)): data_list = []
    valid_data = [x for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x))]; return stat_func(valid_data) if valid_data else default_val

def get_actionable_insights(sim_data, current_config): 
    insights = []; 
    if not sim_data or not isinstance(sim_data, dict): return insights 
    compliance_data = safe_get(sim_data, 'task_compliance.data', []); target_compliance_from_config = current_config.get('TARGET_COMPLIANCE', 75); compliance_avg = safe_stat(compliance_data, np.mean, default_val=target_compliance_from_config)
    if compliance_avg < target_compliance_from_config * 0.9: insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Avg. Task Compliance ({compliance_avg:.1f}%) significantly below target ({target_compliance_from_config}%). Review disruption impacts, task complexities, and training."})
    elif compliance_avg < target_compliance_from_config: insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Avg. Task Compliance at {compliance_avg:.1f}%. Identify intervals or areas with lowest compliance for process review."})
    wellbeing_scores = safe_get(sim_data, 'worker_wellbeing.scores', []); target_wellbeing_from_config = current_config.get('TARGET_WELLBEING', 70); wellbeing_avg = safe_stat(wellbeing_scores, np.mean, default_val=target_wellbeing_from_config)
    if wellbeing_avg < target_wellbeing_from_config * current_config.get('WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET', 0.85) : insights.append({"type": "critical", "title": "Critical Worker Well-being", "text": f"Avg. Well-being ({wellbeing_avg:.1f}%) critically low (target {target_wellbeing_from_config}%). Urgent review of work conditions, load, and stress factors needed."})
    threshold_triggers = safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])
    if wellbeing_scores and len(threshold_triggers) > max(2, len(wellbeing_scores) * 0.1): insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers)} instances of well-being dropping below threshold. Investigate specific triggers."})
    downtime_data = safe_get(sim_data, 'downtime_minutes', []); total_downtime = safe_stat(downtime_data, np.sum); sim_cfg = sim_data.get('config_params', {}); shift_mins = sim_cfg.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']); dt_thresh_total = current_config.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT', shift_mins * 0.05) 
    if total_downtime > dt_thresh_total : insights.append({"type": "critical", "title": "Excessive Total Downtime", "text": f"Total shift downtime is {total_downtime:.0f} min, exceeding guideline of {dt_thresh_total:.0f} min. Focus on disruption causes."})
    psych_safety_scores = safe_get(sim_data, 'psychological_safety', []); target_psych_safety = current_config.get('TARGET_PSYCH_SAFETY', 70); psych_safety_avg = safe_stat(psych_safety_scores, np.mean, default_val=target_psych_safety)
    if psych_safety_avg < target_psych_safety * 0.9: insights.append({"type": "warning", "title": "Low Psychological Safety", "text": f"Avg. Psych. Safety ({psych_safety_avg:.1f}%) is below target ({target_psych_safety}%). Consider initiatives to build trust and open communication."})
    cohesion_scores = safe_get(sim_data, 'worker_wellbeing.team_cohesion_scores', []); target_cohesion = current_config.get('TARGET_TEAM_COHESION', 70) 
    cohesion_avg = safe_stat(cohesion_scores, np.mean, default_val=target_cohesion)
    if cohesion_avg < target_cohesion * 0.9: insights.append({"type": "warning", "title": "Low Team Cohesion", "text": f"Avg. Team Cohesion ({cohesion_avg:.1f}%) is below desired levels. Consider team-building activities or structural reviews for collaboration."})
    workload_scores = safe_get(sim_data, 'worker_wellbeing.perceived_workload_scores', []); target_workload = current_config.get('TARGET_PERCEIVED_WORKLOAD', 6.5) 
    workload_avg = safe_stat(workload_scores, np.mean, default_val=target_workload)
    if workload_avg > current_config.get('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5): insights.append({"type": "critical", "title": "High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) is critically high. Review task distribution, staffing, and process efficiencies."})
    elif workload_avg > target_workload: insights.append({"type": "warning", "title": "Elevated Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) is above target ({target_workload}/10). Monitor closely and identify bottlenecks."})
    if compliance_avg > target_compliance_from_config * 1.05 and wellbeing_avg > target_wellbeing_from_config * 1.05 and total_downtime < dt_thresh_total * 0.5 and psych_safety_avg > target_psych_safety * 1.05: insights.append({"type": "positive", "title": "Holistically Excellent Performance", "text": "Key operational and psychosocial metrics significantly exceed targets. A well-balanced and high-performing shift!"})
    initiative = sim_data.get('config_params', {}).get('TEAM_INITIATIVE', 'Standard Operations')
    if initiative != "Standard Operations": insights.append({"type": "info", "title": f"Initiative Active: '{initiative}'", "text": f"'{initiative}' was simulated. Its impact can be assessed by comparing metrics to a 'Standard Operations' baseline run."})
    return insights

def main():
    st.title("Workplace Shift Optimization Dashboard")
    app_state_keys = ['simulation_results', 'show_tour', 'show_help_glossary', 
                      'op_metrics_time_slider_val', 'worker_insights_time_slider_val', 
                      'worker_snap_step_slider_val_dist_tab', 'downtime_tab_time_slider_val']
    for key in app_state_keys:
        if key not in st.session_state: st.session_state[key] = None 
    
    sb_team_size, sb_shift_duration, sb_disrupt_mins, sb_team_initiative, \
    sb_run_sim_btn, sb_load_data_btn, sb_high_contrast, \
    sb_use_3d, sb_debug_mode = render_settings_sidebar()
        
    _default_shift_duration = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']
    _current_shift_duration_for_slider_max = sb_shift_duration if sb_shift_duration is not None else _default_shift_duration
    current_max_minutes_for_sliders = _current_shift_duration_for_slider_max - 2
    disruption_steps_for_plots = []

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        num_steps_from_sim = len(st.session_state.simulation_results.get('downtime_minutes', []))
        if num_steps_from_sim > 0: current_max_minutes_for_sliders = (num_steps_from_sim - 1) * 2
        disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_EVENT_STEPS', [])
    else: 
        _disrupt_mins_list_for_plots = sb_disrupt_mins if isinstance(sb_disrupt_mins, list) else [] # Ensure list for iteration
        disruption_steps_for_plots = [m // 2 for m in _disrupt_mins_list_for_plots if isinstance(m, (int, float))]
    current_max_minutes_for_sliders = max(0, current_max_minutes_for_sliders) 

    if sb_run_sim_btn:
        with st.spinner("🚀 Simulating workplace operations..."):
            try: 
                final_disrupt_mins_for_sim = sb_disrupt_mins
                if not isinstance(final_disrupt_mins_for_sim, list):
                    logger.error(f"CRITICAL in main: sb_disrupt_mins for simulation was {type(final_disrupt_mins_for_sim)}. Defaulting to empty list.")
                    final_disrupt_mins_for_sim = []
                st.session_state.simulation_results = run_simulation_logic(sb_team_size, sb_shift_duration, final_disrupt_mins_for_sim, sb_team_initiative)
                st.success("✅ Simulation completed!"); logger.info("Simulation run successful.", extra={'user_action': 'Run Simulation - Success'}); st.rerun() 
            except Exception as e: logger.error(f"Simulation Run Error: {e}", exc_info=True, extra={'user_action': 'Run Simulation - Error'}); st.error(f"❌ Simulation failed: {e}"); st.session_state.simulation_results = None 
    
    if sb_load_data_btn:
        with st.spinner("🔄 Loading saved simulation data..."):
            try:
                loaded_data = load_simulation_data() 
                if loaded_data and isinstance(loaded_data, dict):
                    st.session_state.simulation_results = loaded_data; cfg = loaded_data.get('config_params', {})
                    loaded_disrupt_mins = cfg.get('DISRUPTION_INTERVALS_MINUTES', []) 
                    if not isinstance(loaded_disrupt_mins, list): logger.warning(f"Loaded DISRUPTION_INTERVALS_MINUTES was {type(loaded_disrupt_mins)}. Defaulting to []."); loaded_disrupt_mins = []
                    st.session_state.sb_team_size_slider = cfg.get('TEAM_SIZE', st.session_state.get('sb_team_size_slider'))
                    st.session_state.sb_shift_duration_slider = cfg.get('SHIFT_DURATION_MINUTES', st.session_state.get('sb_shift_duration_slider'))
                    st.session_state.sb_team_initiative_selectbox = cfg.get('TEAM_INITIATIVE', st.session_state.get('sb_team_initiative_selectbox'))
                    st.session_state.sb_disruption_intervals_multiselect = loaded_disrupt_mins
                    st.success("✅ Data loaded successfully!"); logger.info("Saved data loaded successfully.", extra={'user_action': 'Load Data - Success'}); st.rerun() 
                else: st.error("❌ Failed to load data or data is not in the expected dictionary format."); logger.warning("Load data failed or invalid format.", extra={'user_action': 'Load Data - Fail/Invalid'})
            except Exception as e: logger.error(f"Load Data Error: {e}", exc_info=True, extra={'user_action': 'Load Data - Error'}); st.error(f"❌ Failed to load data: {e}"); st.session_state.simulation_results = None
    
    if st.session_state.get('show_tour'): 
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3><p>Welcome! ...</p></div>""", unsafe_allow_html=True) 
        if st.button("Got it!", key="tour_modal_close_btn"): st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help_glossary'): 
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3><p>This dashboard ...</p><h4>Metric Definitions:</h4>...</div>""", unsafe_allow_html=True) 
        if st.button("Understood", key="help_modal_close_btn"): st.session_state.show_help_glossary = False; st.rerun()

    tabs_main_names = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Well-being", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs = st.tabs(tabs_main_names)
    plot_config_interactive = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'pan2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}; plot_config_minimal = {'displayModeBar': False}
    current_high_contrast_setting = st.session_state.get('sb_high_contrast_checkbox', False)

    with tabs[0]: 
        st.header("📊 Key Performance Indicators & Actionable Insights", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; effective_config_for_targets = {**DEFAULT_CONFIG, **sim_data.get('config_params', {})} 
            compliance_target=effective_config_for_targets.get('TARGET_COMPLIANCE', 75); collab_target=effective_config_for_targets.get('TARGET_COLLABORATION', 60); wb_target=effective_config_for_targets.get('TARGET_WELLBEING', 70); dt_target_total_shift=effective_config_for_targets.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT', sim_data.get('config_params',{}).get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) * 0.05 )
            compliance = safe_stat(safe_get(sim_data, 'task_compliance.data', []), np.mean); proximity = safe_stat(safe_get(sim_data, 'collaboration_proximity.data', []), np.mean); wellbeing = safe_stat(safe_get(sim_data, 'worker_wellbeing.scores', []), np.mean); downtime = safe_stat(safe_get(sim_data, 'downtime_minutes', []), np.sum)
            cols_metrics = st.columns(4); cols_metrics[0].metric("Task Compliance", f"{compliance:.1f}%", f"{compliance-compliance_target:.1f}% vs Target {compliance_target}%"); cols_metrics[1].metric("Collaboration Index", f"{proximity:.1f}%", f"{proximity-collab_target:.1f}% vs Target {collab_target}%"); cols_metrics[2].metric("Worker Well-Being", f"{wellbeing:.1f}%", f"{wellbeing-wb_target:.1f}% vs Target {wb_target}%"); cols_metrics[3].metric("Total Downtime", f"{downtime:.1f} min", f"{downtime-dt_target_total_shift:.1f} min vs Target {dt_target_total_shift:.0f}min", delta_color="inverse")
            try:
                summary_figs = plot_key_metrics_summary(compliance, proximity, wellbeing, downtime, current_high_contrast_setting) 
                if summary_figs:
                    cols_gauges = st.columns(min(len(summary_figs), 4) or 1) 
                    for i, fig in enumerate(summary_figs):
                        with cols_gauges[i % len(cols_gauges)]: st.plotly_chart(fig, use_container_width=True, config=plot_config_minimal)
                else: st.caption("Gauge charts could not be generated.")
            except Exception as e: logger.error(f"Overview Gauges Plotting Error: {e}", exc_info=True); st.error("⚠️ Error rendering overview gauges.")
            st.markdown("---"); st.subheader("💡 Key Insights & Alerts")
            actionable_insights = get_actionable_insights(sim_data, effective_config_for_targets) 
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
            sim_data = st.session_state.simulation_results; op_slider_key = "op_time_slider"; op_val_key = "op_metrics_time_slider_val"
            default_val = (0, current_max_minutes_for_sliders); val_from_state = st.session_state.get(op_val_key)
            if val_from_state is None or not (isinstance(val_from_state, tuple) and len(val_from_state) == 2): current_val = default_val; st.session_state[op_val_key] = current_val
            else: current_val = val_from_state
            current_val = (min(current_val[0], current_max_minutes_for_sliders), min(current_val[1], current_max_minutes_for_sliders)); current_val = (current_val[0], max(current_val[0], current_val[1]))
            time_range = st.slider("Select Time Range (minutes):", 0, current_max_minutes_for_sliders, current_val, 2, key=op_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({op_val_key: st.session_state[op_slider_key]}))
            start_idx, end_idx = time_range[0]//2, time_range[1]//2 + 1; filt_disrupt_steps = [s for s in disruption_steps_for_plots if start_idx <= s < end_idx]
            tc_data_list = safe_get(sim_data, 'task_compliance.data', [])[start_idx:end_idx]
            if tc_data_list:
                with st.container(border=True): st.markdown('<h5>Task Compliance Score Over Time</h5>', unsafe_allow_html=True)
                try: tc_z = safe_get(sim_data, 'task_compliance.z_scores', [])[start_idx:end_idx]; tc_f = safe_get(sim_data, 'task_compliance.forecast', [])[start_idx:end_idx] if safe_get(sim_data, 'task_compliance.forecast', []) else None; st.plotly_chart(plot_task_compliance_score(tc_data_list, filt_disrupt_steps, tc_f, tc_z, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                except Exception as e: logger.error(f"Op Tab TC Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Task Compliance.")
            else: st.caption("No Task Compliance data for this time range.")
            cp_data_list = safe_get(sim_data, 'collaboration_proximity.data', [])[start_idx:end_idx]
            if cp_data_list:
                with st.container(border=True): st.markdown('<h5>Collaboration Proximity Index Over Time</h5>', unsafe_allow_html=True)
                try: cp_f = safe_get(sim_data, 'collaboration_proximity.forecast', [])[start_idx:end_idx] if safe_get(sim_data, 'collaboration_proximity.forecast', []) else None; st.plotly_chart(plot_collaboration_proximity_index(cp_data_list, filt_disrupt_steps, cp_f, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                except Exception as e: logger.error(f"Op Tab CP Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Collaboration Index.")
            else: st.caption("No Collaboration Index data for this time range.")
            with st.expander("Show Additional Operational Metrics", expanded=False):
                or_data_list = safe_get(sim_data, 'operational_recovery', [])[start_idx:end_idx]
                if or_data_list:
                    with st.container(border=True): st.markdown('<h5>Operational Recovery & Resilience</h5>', unsafe_allow_html=True)
                    try: pl_data = safe_get(sim_data, 'productivity_loss', [])[start_idx:end_idx]; st.plotly_chart(plot_operational_recovery(or_data_list, pl_data, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                    except Exception as e: logger.error(f"Op Tab OR Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Operational Recovery.")
                else: st.caption("No Operational Recovery data for this time range.")
                eff_df_full = safe_get(sim_data, 'efficiency_metrics_df', pd.DataFrame())
                if not eff_df_full.empty:
                    with st.container(border=True): st.markdown('<h5>Overall Equipment Effectiveness (OEE) & Components</h5>', unsafe_allow_html=True)
                    try:
                        sel_metrics = st.multiselect("Select Efficiency Metrics:", ['uptime', 'throughput', 'quality', 'oee'], default=['uptime', 'throughput', 'quality', 'oee'], key="eff_metrics_multiselect_op_tab")
                        filt_eff_df = eff_df_full.iloc[start_idx:end_idx] if isinstance(eff_df_full.index, pd.RangeIndex) and end_idx <= len(eff_df_full) else eff_df_full 
                        if not filt_eff_df.empty: st.plotly_chart(plot_operational_efficiency(filt_eff_df, sel_metrics, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                        else: st.caption("No OEE data for this time range after filtering.")
                    except Exception as e: logger.error(f"Op Tab OEE Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting OEE.")
                else: st.caption("No OEE data available.")
        else: st.info("ℹ️ Run a simulation or load data to view Operational Metrics.", icon="📈")

    with tabs[2]: 
        st.header("👥 Worker Well-being & Psychosocial Factors", divider="blue") # Updated tab name
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; team_pos_df_all = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
            slider_key = "worker_insights_time_slider"; value_key = "worker_insights_time_slider_val"
            default_value = (0, current_max_minutes_for_sliders); value_from_state = st.session_state.get(value_key)
            if value_from_state is None or not (isinstance(value_from_state, tuple) and len(value_from_state) == 2): current_slider_value = default_value; st.session_state[value_key] = current_slider_value
            else: current_slider_value = value_from_state
            current_slider_value = (min(current_slider_value[0], current_max_minutes_for_sliders), min(current_slider_value[1], current_max_minutes_for_sliders)); current_slider_value = (current_slider_value[0], max(current_slider_value[0], current_slider_value[1]))
            shared_time_range = st.slider("Select Time Range (minutes) for Worker Insights:", 0, current_max_minutes_for_sliders, current_slider_value, 2, key=slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({value_key: st.session_state[slider_key]}))
            shared_start_idx, shared_end_idx = shared_time_range[0]//2, shared_time_range[1]//2 + 1
            
            st.subheader("Core Well-being & Safety Indicators")
            cols_core_wellbeing = st.columns(2)
            with cols_core_wellbeing[0]:
                st.markdown("<h5>Worker Well-Being Index</h5>", unsafe_allow_html=True)
                ww_scores_list = safe_get(sim_data, 'worker_wellbeing.scores', [])[shared_start_idx:shared_end_idx]
                if ww_scores_list:
                    with st.container(border=True):
                        try: ww_trigs_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {}); ww_trigs_filt = {k: [t for t in v if shared_start_idx <= t < shared_end_idx] for k, v in ww_trigs_raw.items() if isinstance(v, list)}; ww_trigs_filt['work_area'] = {wk: [t for t in wv if shared_start_idx <= t < shared_end_idx] for wk, wv in ww_trigs_raw.get('work_area',{}).items()}; st.plotly_chart(plot_worker_wellbeing(ww_scores_list, ww_trigs_filt, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                        except Exception as e: logger.error(f"Wellbeing Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Well-Being Index.")
                else: st.caption("No Well-Being Index data for this time range.")
            with cols_core_wellbeing[1]:
                st.markdown("<h5>Psychological Safety Score</h5>", unsafe_allow_html=True)
                ps_scores_list = safe_get(sim_data, 'psychological_safety', [])[shared_start_idx:shared_end_idx]
                if ps_scores_list:
                    with st.container(border=True):
                        try: st.plotly_chart(plot_psychological_safety(ps_scores_list, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                        except Exception as e: logger.error(f"Psych Safety Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Psychological Safety.")
                else: st.caption("No Psych. Safety data for this time range.")

            st.markdown("---"); st.subheader("Additional Psychosocial Risk Factors")
            cols_psychosocial_factors = st.columns(2)
            with cols_psychosocial_factors[0]:
                st.markdown("<h5>Team Cohesion Index</h5>", unsafe_allow_html=True)
                tc_scores_list = safe_get(sim_data, 'worker_wellbeing.team_cohesion_scores', [])[shared_start_idx:shared_end_idx] 
                if tc_scores_list:
                    with st.container(border=True):
                        try: st.plotly_chart(plot_team_cohesion(tc_scores_list, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                        except Exception as e: logger.error(f"Team Cohesion Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Team Cohesion.")
                else: st.caption("No Team Cohesion data for this time range.")
            with cols_psychosocial_factors[1]:
                st.markdown("<h5>Perceived Workload Index (0-10)</h5>", unsafe_allow_html=True)
                pw_scores_list = safe_get(sim_data, 'worker_wellbeing.perceived_workload_scores', [])[shared_start_idx:shared_end_idx]
                if pw_scores_list:
                    with st.container(border=True):
                        try: st.plotly_chart(plot_perceived_workload(pw_scores_list, DEFAULT_CONFIG.get('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5), DEFAULT_CONFIG.get('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5), current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                        except Exception as e: logger.error(f"Perceived Workload Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Perceived Workload.")
                else: st.caption("No Perceived Workload data for this time range.")

            st.markdown("---")
            with st.container(border=True):
                st.subheader("Psychosocial Risk Summary & Recommendations")
                st.markdown("<h6>Well-Being Alerts (within selected time range):</h6>", unsafe_allow_html=True)
                ww_trigs_disp_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {}); ww_trigs_disp_filt = {k: [t for t in v if shared_start_idx <= t < shared_end_idx] for k, v in ww_trigs_disp_raw.items() if isinstance(v, list)}; ww_trigs_disp_filt['work_area'] = {wk: [t for t in wv if shared_start_idx <= t < shared_end_idx] for wk, wv in ww_trigs_disp_raw.get('work_area', {}).items()}
                insights_count = 0
                if ww_trigs_disp_filt.get('threshold'): st.markdown(f"<div class='alert-critical insight-text'><strong>Threshold Alerts Met ({len(ww_trigs_disp_filt['threshold'])} times):</strong> Steps {ww_trigs_disp_filt['threshold']}. Acute stress/fatigue likely.</div>", unsafe_allow_html=True); insights_count+=1
                if ww_trigs_disp_filt.get('trend'): st.markdown(f"<div class='alert-warning insight-text'><strong>Declining Trend Alerts ({len(ww_trigs_disp_filt['trend'])} times):</strong> Steps {ww_trigs_disp_filt['trend']}. Accumulating stress/fatigue.</div>", unsafe_allow_html=True); insights_count+=1
                if ww_trigs_disp_filt.get('disruption'): st.markdown(f"<div class='alert-info insight-text'><strong>Disruption-linked Alerts ({len(ww_trigs_disp_filt['disruption'])} times):</strong> Steps {ww_trigs_disp_filt['disruption']}. Support post-disruption.</div>", unsafe_allow_html=True); insights_count+=1
                wa_alerts = ww_trigs_disp_filt.get('work_area', {}); wa_alert_found = any(wa_alerts.values())
                if wa_alert_found: 
                    st.markdown(f"<div class='alert-warning insight-text'><strong>Work Area Specific Alerts:</strong>", unsafe_allow_html=True)
                    for zone, trigs in wa_alerts.items():
                        if trigs: st.markdown(f"  - {zone}: {len(trigs)} alerts at steps {trigs}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True); insights_count+=1
                if insights_count == 0: st.markdown("<p class='insight-text' style='color: #22D3EE;'>✅ No specific well-being alerts triggered in the selected period.</p>", unsafe_allow_html=True)
                st.markdown("<h6 style='margin-top:1.5rem;'>💡 Actionable Considerations:</h6>", unsafe_allow_html=True)
                st.markdown("""<ul style="font-size:0.9rem; color: #D1D5DB; padding-left:20px; margin-bottom:0;"><li><strong>High Workload:</strong> ...</li><li><strong>Low Cohesion/Safety:</strong> ...</li></ul>""", unsafe_allow_html=True)
            
            with st.expander("Worker Distribution & Density Analysis (Spatial)", expanded=False):
                # ... (Worker Distribution plotting logic copied from above - use shared_start_idx etc.)
                # This section will plot worker positions and density based on the SAME time range slider
                # as the well-being and psychosocial plots for correlated analysis.
                # zones_dist = ["All"] + list(DEFAULT_CONFIG.get('WORK_AREAS', {}).keys()); zone_sel_dist = st.selectbox("Zone for Distribution:", zones_dist, key="worker_zone_selectbox_dist_tab_inner")
                # ... (Full plotting logic for distribution and heatmap using shared_start_idx, shared_end_idx, and specific selectbox if needed) ...
                st.caption("Spatial analysis helps correlate worker density and movement with psychosocial indicators.")
        else: st.info("ℹ️ Run a simulation or load data to view Worker Insights.", icon="👥")

    with tabs[3]: 
        st.header("⏱️ Downtime Analysis", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results
            slider_key = "downtime_tab_time_slider"; value_key = "downtime_tab_time_slider_val"
            default_value = (0, current_max_minutes_for_sliders); value_from_state = st.session_state.get(value_key)
            if value_from_state is None or not (isinstance(value_from_state, tuple) and len(value_from_state) == 2): current_slider_value = default_value; st.session_state[value_key] = current_slider_value
            else: current_slider_value = value_from_state
            current_slider_value = (min(current_slider_value[0], current_max_minutes_for_sliders), min(current_slider_value[1], current_max_minutes_for_sliders)); current_slider_value = (current_slider_value[0], max(current_slider_value[0], current_slider_value[1]))
            time_range_dt = st.slider("Select Time Range (minutes):", 0, current_max_minutes_for_sliders, current_slider_value, 2, key=slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({value_key: st.session_state[slider_key]}))
            dt_start_idx, dt_end_idx = time_range_dt[0]//2, time_range_dt[1]//2 + 1
            dt_data_list = safe_get(sim_data, 'downtime_minutes', [])[dt_start_idx:dt_end_idx]
            if dt_data_list:
                total_downtime_period = sum(dt_data_list); avg_downtime_period = np.mean(dt_data_list) if dt_data_list else 0
                st.metric("Total Downtime in Selected Period", f"{total_downtime_period:.1f} min", help=f"Sum of all downtime instances from minute {time_range_dt[0]} to {time_range_dt[1]}. Target for entire shift: < {DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] * 0.05):.0f} min.")
                with st.container(border=True): st.markdown('<h5>Downtime Trend Over Time (per Interval)</h5>', unsafe_allow_html=True)
                try: st.plotly_chart(plot_downtime_trend(dt_data_list, DEFAULT_CONFIG.get('DOWNTIME_PLOT_ALERT_THRESHOLD', 10), current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                except Exception as e: logger.error(f"Downtime Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Downtime Trend.")
            else: st.caption("No Downtime data for this time range.")
        else: st.info("ℹ️ Run a simulation or load data for Downtime Analysis.", icon="⏱️")
        
    with tabs[4]: 
        st.header("📖 Glossary of Terms", divider="blue")
        st.markdown("""<div style="font-size: 0.95rem; line-height: 1.7;"> ... (Full Glossary HTML content from previous answer) ... </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
