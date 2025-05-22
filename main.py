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
# Ensure logger is configured only once
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
        filename='dashboard.log',
        filemode='a'  # Append to log file
    )
logger.info("Main.py: Parsed imports and logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(
    page_title="Workplace Shift Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@example.com',
        'Report a bug': "mailto:bugs@example.com",
        'About': "# Workplace Shift Monitoring Dashboard\nVersion 1.0\nThis dashboard simulates and visualizes workplace operations."
    }
)

# Optimized CSS for a professional look and feel
st.markdown("""
    <style>
        /* Base Styles */
        .main { background-color: #121828; color: #EAEAEA; font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif; padding: 2rem; }
        h1 { font-size: 2.4rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; text-align: center; margin-bottom: 2rem; color: #FFFFFF; }
        h2 { /* Tab Headers */ font-size: 1.75rem; font-weight: 600; line-height: 1.3; margin: 1.5rem 0 1rem; color: #D0D0D0; border-bottom: 1px solid #4A5568; padding-bottom: 0.5rem;}
        h3 { /* Expander Titles / Section Subtitles */ font-size: 1.3rem; font-weight: 500; line-height: 1.4; margin-bottom: 0.75rem; color: #C0C0C0;}
        h5 { /* Plot Titles inside containers */ font-size: 1.05rem; font-weight: 500; line-height: 1.3; margin: 0.25rem 0 0.75rem; color: #B0B0B0; text-align: center;}
        h6 { /* Sub-notes or trigger list titles */ font-size: 0.9rem; font-weight: 500; line-height: 1.3; margin: 0.75rem 0 0.25rem; color: #A0A0A0;}
        
        /* Buttons */
        .stButton>button { background-color: #4F46E5; color: #FFFFFF; border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.95rem; font-weight: 500; transition: all 0.2s ease-in-out; border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stButton>button:hover, .stButton>button:focus { background-color: #6366F1; transform: translateY(-1px); box-shadow: 0 3px 7px rgba(0,0,0,0.2); outline: none; }
        .stButton>button:disabled { background-color: #374151; color: #9CA3AF; cursor: not-allowed; box-shadow: none; }
        
        /* Input Widgets */
        .stSelectbox div[data-baseweb="select"], .stSlider div[data-testid="stTickBar"], .stMultiSelect div[data-baseweb="select"] { 
            background-color: #1F2937; color: #EAEAEA; border-radius: 6px; padding: 0.5rem; margin-bottom: 1rem; font-size: 0.95rem; border: 1px solid #374151; 
        }
        /* Slider track adjustment */
        .stSlider div[data-testid="stTickBar"] > div { background-color: #4A5568; }
        .stSlider div[role="slider"] { background-color: #4F46E5; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.3); }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] { background-color: #1F2937; color: #EAEAEA; padding: 1.5rem; border-right: 1px solid #374151; font-size: 0.95rem; }
        [data-testid="stSidebar"] .stButton>button { background-color: #34D399; width: 100%; margin-bottom: 0.5rem; } /* Primary action color for sidebar */
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus { background-color: #6EE7B7; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #EAEAEA; border-bottom: 1px solid #4A5568; margin-top:1rem;}
        
        /* Metric Cards */
        .stMetric { background-color: #1F2937; border-radius: 8px; padding: 1.25rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 1.05rem; border: 1px solid #374151;}
        .stMetric > div > div > div { font-size: 1.8rem !important; color: #FFFFFF !important; } /* Metric Value */
        .stMetric > div > div > p { font-size: 0.9rem !important; color: #A0A0A0 !important; } /* Metric Label */
        .stMetric > div:nth-child(2) > div { font-size: 0.85rem !important; } /* Delta */

        /* Expanders */
        .stExpander { background-color: #1F2937; border-radius: 8px; margin: 1rem 0; border: 1px solid #374151; }
        .stExpander header { font-size: 1rem; font-weight: 500; color: #E0E0E0; padding: 0.5rem 1rem; }
        .stExpander div[role="button"] { padding: 0.75rem !important; }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] { background-color: #1F2937; border-radius: 8px; padding: 0.5rem; display: flex; justify-content: center; gap: 0.5rem; border-bottom: 2px solid #374151;}
        .stTabs [data-baseweb="tab"] { color: #D1D5DB; padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 500; font-size: 0.95rem; transition: all 0.2s ease-in-out; border: none; border-bottom: 2px solid transparent; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: transparent; color: #4F46E5; border-bottom: 2px solid #4F46E5; font-weight:600; }
        .stTabs [data-baseweb="tab"]:hover { background-color: #374151; color: #FFFFFF; }
        
        /* Plot Containers */
        .plot-container { background-color: #1F2937; border-radius: 8px; padding: 1rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #374151;}
        .stPlotlyChart { border-radius: 6px; } /* Plotly chart itself if it needs specific styling */
        
        /* DataFrames */
        .stDataFrame { border-radius: 8px; font-size: 0.875rem; border: 1px solid #374151; }
        .stDataFrame thead th { background-color: #293344; color: #EAEAEA; font-weight: 600; }
        .stDataFrame tbody tr:nth-child(even) { background-color: #222C3D; }
        .stDataFrame tbody tr:hover { background-color: #374151; }

        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .main { padding: 1rem; } h1 { font-size: 1.8rem; } h2 { font-size: 1.4rem; } h3 { font-size: 1.1rem; }
            .stPlotlyChart { min-height: 300px !important; }
            .stTabs [data-baseweb="tab"] { padding: 0.5rem 0.8rem; font-size: 0.85rem; }
        }
        /* Spinner */
        .spinner { display: flex; justify-content: center; align-items: center; height: 100px; }
        .spinner::after { content: ''; width: 40px; height: 40px; border: 4px solid #4A5568; border-top: 4px solid #4F46E5; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        /* Modals */
        .onboarding-modal { background-color: #1F2937; border: 1px solid #374151; border-radius: 8px; padding: 1.5rem; max-width: 550px; margin: 2rem auto; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .onboarding-modal h3 { color: #EAEAEA; margin-bottom: 1rem; text-align: center; }
        .onboarding-modal p, .onboarding-modal ul { color: #D1D5DB; line-height: 1.6; margin-bottom: 1rem; font-size: 0.9rem; }
        .onboarding-modal ul { list-style-position: inside; padding-left: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

def render_settings_sidebar():
    with st.sidebar:
        st.markdown(f'<img src="{LEAN_LOGO_BASE64}" width="100" alt="Lean Institute Logo" style="display: block; margin: 0 auto 1rem;">', unsafe_allow_html=True)
        st.markdown("## ⚙️ Settings")

        with st.expander("🧪 Simulation Parameters", expanded=True):
            team_size_val = st.session_state.get('team_size', DEFAULT_CONFIG['TEAM_SIZE'])
            shift_duration_val = st.session_state.get('shift_duration', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'])
            
            team_size = st.slider("Team Size", 10, 100, team_size_val, key="sb_team_size_slider")
            shift_duration = st.slider("Shift Duration (min)", 200, 2000, shift_duration_val, step=2, key="sb_shift_duration_slider")
            
            max_disrupt_time = shift_duration - 2 
            disruption_options = [i * 2 for i in range(max_disrupt_time // 2)] if max_disrupt_time > 0 else []
            
            default_disrupt_mins_raw = [i * 2 for i in DEFAULT_CONFIG.get('DISRUPTION_INTERVALS', [])]
            valid_default_disrupt_mins = [m for m in default_disrupt_mins_raw if m in disruption_options]
            
            _current_disrupt_selection_from_state = st.session_state.get('disruption_intervals_minutes')
            current_disrupt_selection_for_widget = [] 

            if _current_disrupt_selection_from_state is None:
                current_disrupt_selection_for_widget = valid_default_disrupt_mins
            elif not isinstance(_current_disrupt_selection_from_state, list):
                logger.warning(f"Session 'disruption_intervals_minutes' not a list (type: {type(_current_disrupt_selection_from_state)}). Resetting.")
                current_disrupt_selection_for_widget = valid_default_disrupt_mins
            else:
                current_disrupt_selection_for_widget = _current_disrupt_selection_from_state
            
            valid_current_disrupt_selection_for_widget = [
                m for m in current_disrupt_selection_for_widget if m in disruption_options
            ]

            disruption_intervals_minutes = st.multiselect(
                "Disruption Times (min)", 
                disruption_options, 
                valid_current_disrupt_selection_for_widget,
                key="sb_disruption_intervals_multiselect", 
                help="Select times (in minutes from start) when disruptions occur."
            )
            
            team_initiative_opts = ["More frequent breaks", "Team recognition"]
            current_initiative = st.session_state.get('team_initiative', team_initiative_opts[0])
            team_initiative_idx = team_initiative_opts.index(current_initiative) if current_initiative in team_initiative_opts else 0
            team_initiative = st.selectbox("Team Initiative Strategy", team_initiative_opts, index=team_initiative_idx, key="sb_team_initiative_selectbox", help="Select a strategy to potentially improve well-being and safety.")
            
            run_simulation_button = st.button("🚀 Run Simulation", key="sb_run_simulation_button", type="primary", use_container_width=True)

        with st.expander("🎨 Visualization Options"):
            high_contrast = st.checkbox("High Contrast Mode (Plots)", st.session_state.get('high_contrast', False), key="sb_high_contrast_checkbox", help="Applies a high-contrast theme to all plots for better accessibility.")
            use_3d_distribution = st.checkbox("3D Worker Distribution Plot", st.session_state.get('use_3d_distribution', False), key="sb_use_3d_distribution_checkbox", help="Renders worker positions in a 3D scatter plot.")
            debug_mode = st.checkbox("Enable Debug Mode", st.session_state.get('debug_mode', False), key="sb_debug_mode_checkbox", help="Shows additional debug information in the sidebar.")

        with st.expander("💾 Data & Reporting"):
            load_data_button = st.button("🔄 Load Saved Simulation", key="sb_load_data_button", use_container_width=True)
            can_gen_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("📄 Download PDF Report (.tex)", key="sb_download_report_button", disabled=not can_gen_report, use_container_width=True, help="Generates a LaTeX (.tex) file summarizing the simulation. Requires a LaTeX distribution to compile to PDF."):
                if can_gen_report:
                    try:
                        sim_res = st.session_state.simulation_results; num_steps = len(sim_res.get('downtime_minutes', []))
                        if num_steps == 0: st.warning("⚠️ No data available for PDF report generation."); raise SystemExit 
                        pdf_data = {k: sim_res.get(k, [np.nan]*num_steps)[:num_steps] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'downtime_minutes', 'task_completion_rate']}
                        pdf_data.update({'task_compliance': sim_res.get('task_compliance', {}).get('data', [np.nan]*num_steps)[:num_steps], 'collaboration_proximity': sim_res.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps)[:num_steps], 'worker_wellbeing': sim_res.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps)[:num_steps], 'step': list(range(num_steps)), 'time_minutes': [i * 2 for i in range(num_steps)]})
                        generate_pdf_report(pd.DataFrame(pdf_data)); st.success("✅ LaTeX report (.tex) generation initiated. Check 'workplace_report.tex'.")
                    except SystemExit: pass 
                    except Exception as e: logger.error(f"PDF Generation Error: {e}", exc_info=True); st.error(f"❌ PDF Generation Error: {e}")
        
        with st.expander("📊 Export Simulation Data"):
            if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                st.caption("Note: Individual plots can be exported as PNG using the camera icon on the plot.")
                sim_res_exp = st.session_state.simulation_results; num_steps_csv = len(sim_res_exp.get('downtime_minutes', []))
                if num_steps_csv > 0:
                    csv_data = {k: sim_res_exp.get(k, [np.nan]*num_steps_csv)[:num_steps_csv] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'downtime_minutes', 'task_completion_rate']}
                    csv_data.update({'task_compliance': sim_res_exp.get('task_compliance', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv], 'collaboration_proximity': sim_res_exp.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv], 'worker_wellbeing': sim_res_exp.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps_csv)[:num_steps_csv], 'step': list(range(num_steps_csv)), 'time_minutes': [i * 2 for i in range(num_steps_csv)]})
                    st.download_button("📥 Download Summary (CSV)", pd.DataFrame(csv_data).to_csv(index=False).encode('utf-8'), "workplace_summary.csv", "text/csv", key="sb_download_csv_button", use_container_width=True)
                else: st.info("No data available for CSV export.")
            else: st.info("Run a simulation to enable data export options.", icon="ℹ️")

        if st.session_state.get('sb_debug_mode_checkbox', False): 
            with st.expander("🛠️ Debug Information", expanded=False):
                st.write("**Default Config (Partial):**"); st.json({k: DEFAULT_CONFIG.get(k) for k in ['ENTRY_EXIT_POINTS', 'WORK_AREAS']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Active Simulation Config (from results):**"); st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else: st.write("**No active simulation data to show config from.**")

        st.markdown("## 📋 Help & Info")
        if st.button("ℹ️ Help & Glossary", key="sb_help_toggle_button", use_container_width=True): st.session_state.show_help_glossary = not st.session_state.get('show_help_glossary', False); st.rerun()
        if st.button("🚀 Quick Tour", key="sb_tour_toggle_button", use_container_width=True): st.session_state.show_tour = not st.session_state.get('show_tour', False); st.rerun()

    return team_size, shift_duration, disruption_intervals_minutes, team_initiative, run_simulation_button, load_data_button, high_contrast, use_3d_distribution, debug_mode

@st.cache_data(ttl=3600, show_spinner="⚙️ Optimizing simulation parameters and running model...")
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
                all_zone_keys = list(config['WORK_AREAS'].keys())
                for zk in all_zone_keys:
                    if zk not in zone_keys_for_dist: config['WORK_AREAS'][zk]['workers'] = 0
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

def main():
    st.title("Workplace Shift Monitoring Dashboard")
    ui_state_keys_to_init_none = ['team_size', 'shift_duration', 'disruption_intervals_minutes', 'team_initiative', 'high_contrast', 'use_3d_distribution', 'debug_mode', 'simulation_results', 'show_tour', 'show_help_glossary']
    for key in ui_state_keys_to_init_none:
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

    if sb_run_sim_btn:
        with st.spinner("🚀 Simulating workplace operations... Please wait."):
            try: st.session_state.simulation_results = run_simulation_logic(st.session_state.team_size, st.session_state.shift_duration, st.session_state.disruption_intervals_minutes, st.session_state.team_initiative); st.success("✅ Simulation completed successfully!"); logger.info("Simulation run successful.", extra={'user_action': 'Run Simulation - Success'}); st.rerun() 
            except Exception as e: logger.error(f"Simulation Run Error: {e}", exc_info=True, extra={'user_action': 'Run Simulation - Error'}); st.error(f"❌ Simulation failed: {e}"); st.session_state.simulation_results = None 
    if sb_load_data_btn:
        with st.spinner("🔄 Loading saved simulation data..."):
            try:
                loaded_data = load_simulation_data() 
                if loaded_data and isinstance(loaded_data, dict):
                    st.session_state.simulation_results = loaded_data; cfg = loaded_data.get('config_params', {})
                    st.session_state.update(team_size=cfg.get('TEAM_SIZE',st.session_state.team_size), shift_duration=cfg.get('SHIFT_DURATION_MINUTES',st.session_state.shift_duration), team_initiative=cfg.get('TEAM_INITIATIVE',st.session_state.team_initiative), disruption_intervals_minutes=cfg.get('DISRUPTION_INTERVALS_MINUTES',st.session_state.disruption_intervals_minutes))
                    st.success("✅ Data loaded successfully!"); logger.info("Saved data loaded successfully.", extra={'user_action': 'Load Data - Success'}); st.rerun() 
                else: st.error("❌ Failed to load data or data is not in the expected dictionary format."); logger.warning("Load data failed or invalid format.", extra={'user_action': 'Load Data - Fail/Invalid'})
            except Exception as e: logger.error(f"Load Data Error: {e}", exc_info=True, extra={'user_action': 'Load Data - Error'}); st.error(f"❌ Failed to load data: {e}"); st.session_state.simulation_results = None
    
    if st.session_state.get('show_tour'):
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3><p>Welcome! This dashboard helps you monitor and analyze workplace shift operations...</p><ul><li><b>Sidebar Controls:</b> Adjust simulation parameters...</li><li><b>Main Tabs:</b> Navigate through different views...</li><li><b>Interactive Charts:</b> Hover for details...</li></ul><p>Start by running a new simulation or loading previous data!</p></div>""", unsafe_allow_html=True)
        if st.button("Got it, let's explore!", key="end_tour_btn_modal", use_container_width=True): st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help_glossary'):
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3><p>This dashboard provides insights...</p><h4>Metric Definitions:</h4><ul><li><b>Task Compliance Score:</b> ...</li></ul><p>Contact support@example.com.</p></div>""", unsafe_allow_html=True)
        if st.button("Understood", key="close_help_glossary_btn_modal", use_container_width=True): st.session_state.show_help_glossary = False; st.rerun()

    tabs_main_names = ["📊 Overview", "📈 Operational Metrics", "👥 Worker Insights", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs = st.tabs(tabs_main_names)
    
    def safe_get(data_dict, path_str, default_val=None):
        current = data_dict
        default_return = default_val if default_val is not None else ([]) 
        
        # Corrected: Check for invalid inputs at the beginning
        if not isinstance(path_str, str) or not isinstance(data_dict, dict): 
            return default_return
        
        # Corrected: Indentation for the loop and subsequent logic
        for key in path_str.split('.'):
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, pd.Series)) and key.isdigit():
                try:
                    idx = int(key)
                    current = current[idx] if idx < len(current) else None
                except (ValueError, IndexError):
                    current = None 
                    break
            else: 
                current = None
                break
        return current if current is not None else default_return

    def safe_stat(data_list, stat_func, default_val=0.0):
        if not isinstance(data_list, (list, np.ndarray, pd.Series)): data_list = []
        valid_data = [x for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x))]; return stat_func(valid_data) if valid_data else default_val
    
    plot_config_interactive = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'pan2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}; plot_config_minimal = {'displayModeBar': False}

    with tabs[0]: 
        st.header("📊 Key Performance Indicators Overview", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; compliance = safe_stat(safe_get(sim_data, 'task_compliance.data', []), np.mean); proximity = safe_stat(safe_get(sim_data, 'collaboration_proximity.data', []), np.mean); wellbeing = safe_stat(safe_get(sim_data, 'worker_wellbeing.scores', []), np.mean); downtime = safe_stat(safe_get(sim_data, 'downtime_minutes', []), np.sum)
            cols_metrics = st.columns(4); cols_metrics[0].metric("Task Compliance", f"{compliance:.1f}%", f"{compliance-75:.1f}% vs Target"); cols_metrics[1].metric("Collaboration Index", f"{proximity:.1f}%", f"{proximity-60:.1f}% vs Target"); cols_metrics[2].metric("Worker Well-Being", f"{wellbeing:.1f}%", f"{wellbeing-70:.1f}% vs Target"); cols_metrics[3].metric("Total Downtime", f"{downtime:.1f} min", f"{downtime-30:.1f} min vs Target", delta_color="inverse")
            try:
                summary_figs = plot_key_metrics_summary(compliance, proximity, wellbeing, downtime, st.session_state.get('high_contrast'))
                if summary_figs:
                    cols_gauges = st.columns(min(len(summary_figs), 4) or 1) 
                    for i, fig in enumerate(summary_figs):
                        with cols_gauges[i % len(cols_gauges)]: st.plotly_chart(fig, use_container_width=True, config=plot_config_minimal)
                else: st.info("Gauge charts could not be generated for the overview metrics.")
            except Exception as e: logger.error(f"Overview Gauges Plotting Error: {e}", exc_info=True); st.error("⚠️ Error rendering overview gauges.")
            with st.expander("View Detailed Overview Data Table", expanded=False):
                num_s = len(safe_get(sim_data, 'downtime_minutes', []))
                if num_s > 0:
                    df_data = {'Time (min)': [i*2 for i in range(num_s)]}; df_data.update({'Task Compliance (%)': safe_get(sim_data, 'task_compliance.data', [np.nan]*num_s)[:num_s], 'Collaboration (%)': safe_get(sim_data, 'collaboration_proximity.data', [np.nan]*num_s)[:num_s], 'Well-Being (%)': safe_get(sim_data, 'worker_wellbeing.scores', [np.nan]*num_s)[:num_s], 'Downtime (min)': safe_get(sim_data, 'downtime_minutes', [np.nan]*num_s)[:num_s]})
                    st.dataframe(pd.DataFrame(df_data).style.format("{:.1f}", na_rep="-").set_table_styles([{'selector': 'th', 'props': [('background-color', '#293344'), ('color', '#EAEAEA')]}]), use_container_width=True, height=300)
                else: st.info("No detailed overview data available to display.")
        else: st.info("ℹ️ Run a simulation or load data to view the Overview.", icon="📊")

    with tabs[1]: 
        st.header("📈 Operational Performance Trends", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; op_time_slider_key = "op_metrics_time_slider"; op_time_val_key = "op_metrics_time_slider_val"
            default_op_time_val = (0, current_max_minutes_for_sliders); op_time_val_from_state = st.session_state.get(op_time_val_key)
            if op_time_val_from_state is None or not (isinstance(op_time_val_from_state, tuple) and len(op_time_val_from_state) == 2): op_time_val = default_op_time_val; st.session_state[op_time_val_key] = op_time_val
            else: op_time_val = op_time_val_from_state
            op_time_val = (min(op_time_val[0], current_max_minutes_for_sliders), min(op_time_val[1], current_max_minutes_for_sliders))
            if op_time_val[0] > op_time_val[1]: op_time_val = (op_time_val[1], op_time_val[1])
            time_range = st.slider("Select Time Range (minutes):", 0, current_max_minutes_for_sliders, op_time_val, 2, key=op_time_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({op_time_val_key: st.session_state[op_time_slider_key]}))
            start_idx, end_idx = time_range[0]//2, time_range[1]//2 + 1; filt_disrupt_steps = [s for s in disruption_steps_for_plots if start_idx <= s < end_idx]
            tc_data_list = safe_get(sim_data, 'task_compliance.data', [])[start_idx:end_idx]
            if tc_data_list:
                with st.container(border=True): st.markdown('<h5>Task Compliance Score Over Time</h5>', unsafe_allow_html=True)
                try: tc_z = safe_get(sim_data, 'task_compliance.z_scores', [])[start_idx:end_idx]; tc_f = safe_get(sim_data, 'task_compliance.forecast', [])[start_idx:end_idx] if safe_get(sim_data, 'task_compliance.forecast', []) else None; st.plotly_chart(plot_task_compliance_score(tc_data_list, filt_disrupt_steps, tc_f, tc_z, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                except Exception as e: logger.error(f"Op Tab TC Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Task Compliance.")
            cp_data_list = safe_get(sim_data, 'collaboration_proximity.data', [])[start_idx:end_idx]
            if cp_data_list:
                with st.container(border=True): st.markdown('<h5>Collaboration Proximity Index Over Time</h5>', unsafe_allow_html=True)
                try: cp_f = safe_get(sim_data, 'collaboration_proximity.forecast', [])[start_idx:end_idx] if safe_get(sim_data, 'collaboration_proximity.forecast', []) else None; st.plotly_chart(plot_collaboration_proximity_index(cp_data_list, filt_disrupt_steps, cp_f, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                except Exception as e: logger.error(f"Op Tab CP Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Collaboration Index.")
            with st.expander("Show Additional Operational Metrics", expanded=False):
                or_data_list = safe_get(sim_data, 'operational_recovery', [])[start_idx:end_idx]
                if or_data_list:
                    with st.container(border=True): st.markdown('<h5>Operational Recovery & Resilience</h5>', unsafe_allow_html=True)
                    try: pl_data = safe_get(sim_data, 'productivity_loss', [])[start_idx:end_idx]; st.plotly_chart(plot_operational_recovery(or_data_list, pl_data, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                    except Exception as e: logger.error(f"Op Tab OR Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Operational Recovery.")
                eff_df_full = safe_get(sim_data, 'efficiency_metrics_df', pd.DataFrame())
                if not eff_df_full.empty:
                    with st.container(border=True): st.markdown('<h5>Overall Equipment Effectiveness (OEE) & Components</h5>', unsafe_allow_html=True)
                    try:
                        sel_metrics = st.multiselect("Select Efficiency Metrics:", ['uptime', 'throughput', 'quality', 'oee'], default=['uptime', 'throughput', 'quality', 'oee'], key="eff_metrics_multiselect_op_tab")
                        filt_eff_df = eff_df_full.iloc[start_idx:end_idx] if isinstance(eff_df_full.index, pd.RangeIndex) and end_idx <= len(eff_df_full) else eff_df_full 
                        if not filt_eff_df.empty: st.plotly_chart(plot_operational_efficiency(filt_eff_df, sel_metrics, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                        else: st.info("No OEE data available for the selected time range.")
                    except Exception as e: logger.error(f"Op Tab OEE Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting OEE.")
        else: st.info("ℹ️ Run a simulation or load data to view Operational Metrics.", icon="📈")

    with tabs[2]: 
        st.header("👥 Worker Dynamics & Well-being", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results; team_pos_df_all = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
            worker_time_slider_key = "worker_insights_time_slider"; worker_time_val_key = "worker_insights_time_slider_val"
            default_worker_time_val = (0, current_max_minutes_for_sliders); worker_time_val_from_state = st.session_state.get(worker_time_val_key)
            if worker_time_val_from_state is None or not (isinstance(worker_time_val_from_state, tuple) and len(worker_time_val_from_state) == 2): worker_time_val = default_worker_time_val; st.session_state[worker_time_val_key] = worker_time_val
            else: worker_time_val = worker_time_val_from_state
            worker_time_val = (min(worker_time_val[0], current_max_minutes_for_sliders), min(worker_time_val[1], current_max_minutes_for_sliders))
            if worker_time_val[0] > worker_time_val[1]: worker_time_val = (worker_time_val[1], worker_time_val[1])
            shared_time_range = st.slider("Select Time Range (minutes) for Worker Insights:", 0, current_max_minutes_for_sliders, worker_time_val, 2, key=worker_time_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({worker_time_val_key: st.session_state[worker_time_slider_key]}))
            shared_start_idx, shared_end_idx = shared_time_range[0]//2, shared_time_range[1]//2 + 1
            with st.expander("Worker Distribution Analysis", expanded=True):
                zones = ["All"] + list(DEFAULT_CONFIG.get('WORK_AREAS', {}).keys()); zone_sel = st.selectbox("Filter by Zone (for Distribution & Density):", zones, key="worker_zone_selectbox_dist_tab")
                filt_team_pos_df = team_pos_df_all
                if not filt_team_pos_df.empty:
                    filt_team_pos_df = filt_team_pos_df[(filt_team_pos_df['step'] >= shared_start_idx) & (filt_team_pos_df['step'] < shared_end_idx)]
                    if zone_sel != "All": filt_team_pos_df = filt_team_pos_df[filt_team_pos_df['zone'] == zone_sel]
                show_ee = st.checkbox("Show Entry/Exit Points on plots", True, key="worker_show_ee_checkbox_dist_tab"); show_pl = st.checkbox("Show Production Lines on plots", True, key="worker_show_pl_checkbox_dist_tab")
                cols_dist = st.columns(2)
                with cols_dist[0]:
                    st.markdown("<h5>Worker Positions (Time Snapshot)</h5>", unsafe_allow_html=True)
                    min_step, max_step = shared_start_idx, max(shared_start_idx, shared_end_idx -1)
                    snap_step_slider_key = "worker_snap_step_slider_dist_tab"; snap_step_val_key = "worker_snap_step_slider_val_dist_tab"
                    snap_step_val_from_state = st.session_state.get(snap_step_val_key)
                    if snap_step_val_from_state is None: snap_step_val = min_step; st.session_state[snap_step_val_key] = snap_step_val
                    else: snap_step_val = snap_step_val_from_state
                    snap_step_val = max(min_step, min(snap_step_val, max_step))
                    if snap_step_val < min_step : snap_step_val = min_step 
                    snap_step = st.slider("Select Time Step for Snapshot:", min_step, max_step, snap_step_val, 1, key=snap_step_slider_key, disabled=max_step < min_step, on_change=lambda: st.session_state.update({snap_step_val_key: st.session_state[snap_step_slider_key]}))
                    if not team_pos_df_all.empty and max_step >= min_step :
                        with st.container(border=True):
                            try: st.plotly_chart(plot_worker_distribution(team_pos_df_all, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, st.session_state.get('use_3d_distribution', False), snap_step, show_ee, show_pl, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            except Exception as e: logger.error(f"Worker Dist Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Worker Positions.")
                    else: st.info("No data available for worker positions snapshot with current filters.")
                with cols_dist[1]:
                    st.markdown("<h5>Worker Density Heatmap (Aggregated)</h5>", unsafe_allow_html=True)
                    if not filt_team_pos_df.empty:
                        with st.container(border=True):
                            try: st.plotly_chart(plot_worker_density_heatmap(filt_team_pos_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, show_ee, show_pl, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            except Exception as e: logger.error(f"Worker Heatmap Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Density Heatmap.")
                    else: st.info("No data available for density heatmap with current filters.")
            with st.expander("Worker Well-Being & Safety Analysis", expanded=True):
                cols_well = st.columns(2)
                with cols_well[0]:
                    st.markdown("<h5>Worker Well-Being Index</h5>", unsafe_allow_html=True)
                    ww_scores_list = safe_get(sim_data, 'worker_wellbeing.scores', [])[shared_start_idx:shared_end_idx]
                    if ww_scores_list:
                        with st.container(border=True):
                            try: ww_trigs_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {}); ww_trigs_filt = {k: [t for t in v if shared_start_idx <= t < shared_end_idx] for k, v in ww_trigs_raw.items() if isinstance(v, list)}; ww_trigs_filt['work_area'] = {wk: [t for t in wv if shared_start_idx <= t < shared_end_idx] for wk, wv in ww_trigs_raw.get('work_area',{}).items()}; st.plotly_chart(plot_worker_wellbeing(ww_scores_list, ww_trigs_filt, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            except Exception as e: logger.error(f"Wellbeing Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Well-Being Index.")
                    else: st.info("No Well-Being Index data for the selected time range.")
                with cols_well[1]:
                    st.markdown("<h5>Psychological Safety Score</h5>", unsafe_allow_html=True)
                    ps_scores_list = safe_get(sim_data, 'psychological_safety', [])[shared_start_idx:shared_end_idx]
                    if ps_scores_list:
                        with st.container(border=True):
                            try: st.plotly_chart(plot_psychological_safety(ps_scores_list, st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                            except Exception as e: logger.error(f"Psych Safety Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Psychological Safety.")
                    else: st.info("No Psychological Safety data for the selected time range.")
                st.markdown("<h6>Well-Being Triggers (within selected time range):</h6>", unsafe_allow_html=True)
                ww_trigs_disp_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {}); ww_trigs_disp_filt = {k: [t for t in v if shared_start_idx <= t < shared_end_idx] for k, v in ww_trigs_disp_raw.items() if isinstance(v, list)}; ww_trigs_disp_filt['work_area'] = {wk: [t for t in wv if shared_start_idx <= t < shared_end_idx] for wk, wv in ww_trigs_disp_raw.get('work_area', {}).items()}
                st.caption(f"**Threshold Alerts (< {DEFAULT_CONFIG.get('WELLBEING_THRESHOLD',0)*100}%):** {ww_trigs_disp_filt.get('threshold', 'None')}"); st.caption(f"**Trend Alerts (Declining):** {ww_trigs_disp_filt.get('trend', 'None')}"); st.caption(f"**Disruption-Related Alerts:** {ww_trigs_disp_filt.get('disruption', 'None')}")
                st.caption("**Work Area Specific Alerts:**"); wa_alert_found = False
                for zone, trigs in ww_trigs_disp_filt.get('work_area', {}).items():
                    if trigs: st.caption(f"    {zone}: {trigs}"); wa_alert_found = True
                if not wa_alert_found: st.caption("    None")
        else: st.info("ℹ️ Run a simulation or load data to view Worker Insights.", icon="👥")

    with tabs[3]: 
        st.header("⏱️ Downtime Analysis", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results
            dt_time_slider_key = "downtime_tab_time_slider"; dt_time_val_key = "downtime_tab_time_slider_val"
            default_dt_time_val = (0, current_max_minutes_for_sliders); dt_time_val_from_state = st.session_state.get(dt_time_val_key)
            if dt_time_val_from_state is None or not (isinstance(dt_time_val_from_state, tuple) and len(dt_time_val_from_state) == 2): dt_time_val = default_dt_time_val; st.session_state[dt_time_val_key] = dt_time_val
            else: dt_time_val = dt_time_val_from_state
            dt_time_val = (min(dt_time_val[0], current_max_minutes_for_sliders), min(dt_time_val[1], current_max_minutes_for_sliders))
            if dt_time_val[0] > dt_time_val[1]: dt_time_val = (dt_time_val[1], dt_time_val[1])
            time_range_dt = st.slider("Select Time Range (minutes):", 0, current_max_minutes_for_sliders, dt_time_val, 2, key=dt_time_slider_key, disabled=current_max_minutes_for_sliders == 0, on_change=lambda: st.session_state.update({dt_time_val_key: st.session_state[dt_time_slider_key]}))
            dt_start_idx, dt_end_idx = time_range_dt[0]//2, time_range_dt[1]//2 + 1
            dt_data_list = safe_get(sim_data, 'downtime_minutes', [])[dt_start_idx:dt_end_idx]
            if dt_data_list:
                with st.container(border=True): st.markdown('<h5>Downtime Trend Over Time</h5>', unsafe_allow_html=True)
                try: st.plotly_chart(plot_downtime_trend(dt_data_list, DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD', 10), st.session_state.get('high_contrast')), use_container_width=True, config=plot_config_interactive)
                except Exception as e: logger.error(f"Downtime Plot Error: {e}", exc_info=True); st.error("⚠️ Error plotting Downtime Trend.")
            else: st.info("No Downtime data available for the selected time range.")
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
