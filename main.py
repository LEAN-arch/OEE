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

# LEAN_LOGO_BASE64 = "..." # REMOVED LOGO

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
                        filename='dashboard.log',
                        filemode='a')
logger.info("Main.py: Parsed imports and logger configured.", extra={'user_action': 'System Startup'})

st.set_page_config(page_title="Workplace Shift Optimization Dashboard", layout="wide", initial_sidebar_state="expanded", menu_items={'Get Help': 'mailto:support@example.com', 'Report a bug': "mailto:bugs@example.com", 'About': "# Workplace Shift Optimization Dashboard\nVersion 1.2\nInsights for operational excellence & psychosocial well-being."})

# --- Accessible Color Definitions ---
COLOR_CRITICAL_RED = "#E53E3E"
COLOR_WARNING_AMBER = "#F59E0B"
COLOR_POSITIVE_GREEN = "#10B981"
COLOR_INFO_BLUE = "#3B82F6"
COLOR_ACCENT_INDIGO = "#4F46E5"

# CSS (ensure stMetric styling is sufficient, or add specific overrides if needed)
st.markdown(f"""
    <style>
        /* Base Styles */
        .main {{ background-color: #121828; color: #EAEAEA; font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif; padding: 2rem; }}
        h1 {{ font-size: 2.4rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; text-align: center; margin-bottom: 2rem; color: #FFFFFF; }}
        h2 {{ /* Tab Headers */ font-size: 1.75rem; font-weight: 600; line-height: 1.3; margin: 1.5rem 0 1rem; color: #D0D0D0; border-bottom: 1px solid #4A5568; padding-bottom: 0.5rem;}}
        h3 {{ /* Expander Titles / Section Subtitles */ font-size: 1.3rem; font-weight: 500; line-height: 1.4; margin-bottom: 0.75rem; color: #C0C0C0;}}
        h5 {{ /* Plot Titles inside containers */ font-size: 1.05rem; font-weight: 500; line-height: 1.3; margin: 0.25rem 0 0.75rem; color: #B0B0B0; text-align: center;}}
        h6 {{ /* Sub-notes or trigger list titles */ font-size: 0.9rem; font-weight: 500; line-height: 1.3; margin: 0.75rem 0 0.25rem; color: #A0A0A0;}}
        .stButton>button {{ background-color: {COLOR_ACCENT_INDIGO}; color: #FFFFFF; border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.95rem; font-weight: 500; transition: all 0.2s ease-in-out; border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stButton>button:hover, .stButton>button:focus {{ background-color: #6366F1; transform: translateY(-1px); box-shadow: 0 3px 7px rgba(0,0,0,0.2); outline: none; }}
        .stButton>button:disabled {{ background-color: #374151; color: #9CA3AF; cursor: not-allowed; box-shadow: none; }}
        .stSelectbox div[data-baseweb="select"], .stNumberInput div input, .stMultiSelect div[data-baseweb="select"] {{ 
            background-color: #1F2937 !important; 
            color: #EAEAEA !important; 
            border-radius: 6px !important; 
            padding: 0.5rem !important; 
            margin-bottom: 1rem !important; 
            font-size: 0.95rem !important; 
            border: 1px solid #374151 !important; 
        }}
        /* Ensure number input buttons are also themed if possible, though harder with pure CSS */
         .stNumberInput button {{
            background-color: #374151 !important;
            color: #EAEAEA !important;
            border: 1px solid #4A5568 !important;
        }}
        .stNumberInput button:hover {{
            background-color: #4A5568 !important;
        }}

        [data-testid="stSidebar"] {{ background-color: #1F2937; color: #EAEAEA; padding: 1.5rem; border-right: 1px solid #374151; font-size: 0.95rem; }}
        [data-testid="stSidebar"] .stButton>button {{ background-color: {COLOR_POSITIVE_GREEN}; width: 100%; margin-bottom: 0.5rem; }}
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus {{ background-color: #6EE7B7; }}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: #EAEAEA; border-bottom: 1px solid #4A5568; margin-top:1rem;}}
        
        /* st.metric styling - Ensure high contrast */
        .stMetric {{ background-color: #1F2937; border-radius: 8px; padding: 1.25rem; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 1.05rem; border: 1px solid #374151;}}
        .stMetric > div:nth-child(1) > div:nth-child(1) {{ /* Label */ font-size: 0.95rem !important; color: #A0A0A0 !important; font-weight: 500; margin-bottom: 0.25rem; }} 
        .stMetric > div:nth-child(1) > div:nth-child(2) {{ /* Value */ font-size: 2rem !important; color: #FFFFFF !important; font-weight: 700; line-height: 1;}} 
        .stMetric > div:nth-child(2) > div {{ /* Delta */ font-size: 0.85rem !important; }} 
        /* Explicitly style delta up/down colors if needed, st.metric usually handles this well */
        /* .stMetric > div:nth-child(2) > div[data-delta-direction="up"] {{ color: {COLOR_POSITIVE_GREEN} !important; }} */
        /* .stMetric > div:nth-child(2) > div[data-delta-direction="down"] {{ color: {COLOR_CRITICAL_RED} !important; }} */


        .stExpander {{ background-color: #1F2937; border-radius: 8px; margin: 1rem 0; border: 1px solid #374151; }}
        .stExpander header {{ font-size: 1rem; font-weight: 500; color: #E0E0E0; padding: 0.5rem 1rem; }}
        .stExpander div[role="button"] {{ padding: 0.75rem !important; }}
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
        @media (max-width: 768px) {{ .main {{ padding: 1rem; }} h1 {{ font-size: 1.8rem; }} h2 {{ font-size: 1.4rem; }} h3 {{ font-size: 1.1rem; }} .stPlotlyChart {{ min-height: 300px !important; }} .stTabs [data-baseweb="tab"] {{ padding: 0.5rem 0.8rem; font-size: 0.85rem; }} }}
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
    </style>
""", unsafe_allow_html=True)

def render_settings_sidebar():
    with st.sidebar:
        # REMOVED LOGO: st.markdown(f'<div style="text-align: center;"><img src="{LEAN_LOGO_BASE64}" width="80" alt="Logo"></div>', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem; color: #A0A0A0;'>LEAN 2.0 Institute</h3>", unsafe_allow_html=True)
        st.markdown("## ⚙️ Simulation Controls")
        with st.expander("🧪 Simulation Parameters", expanded=True):
            team_size = st.slider("Team Size", 10, 100, st.session_state.get('sb_team_size_slider', DEFAULT_CONFIG['TEAM_SIZE']), key="sb_team_size_slider", help="Adjust the number of workers in the simulated shift.") # Keeping this slider for now
            shift_duration = st.slider("Shift Duration (min)", 200, 2000, st.session_state.get('sb_shift_duration_slider', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']), step=2, key="sb_shift_duration_slider", help="Set the total length of the simulated work shift.") # Keeping this slider
            
            max_disrupt_time = shift_duration - 2
            disruption_options = [i * 2 for i in range(max_disrupt_time // 2)] if max_disrupt_time > 0 else []
            default_disrupt_mins_raw = DEFAULT_CONFIG.get('DISRUPTION_TIMES_MINUTES', [])
            valid_default_disrupt_mins = [m for m in default_disrupt_mins_raw if m in disruption_options]
            session_value_for_disruptions = st.session_state.get('sb_disruption_intervals_multiselect')
            default_for_multiselect = valid_default_disrupt_mins
            if session_value_for_disruptions is not None:
                if not isinstance(session_value_for_disruptions, list):
                    logger.warning(f"Sidebar: 'sb_disruption_intervals_multiselect' in session_state was {type(session_value_for_disruptions)}. Expected list. Using default.")
                    default_for_multiselect = valid_default_disrupt_mins
                else:
                    default_for_multiselect = [m for m in session_value_for_disruptions if m in disruption_options]
            disruption_intervals_minutes = st.multiselect("Disruption Times (min)", disruption_options, default=default_for_multiselect, key="sb_disruption_intervals_multiselect", help="Select specific times (in minutes from shift start) when disruptions will occur in the simulation.")
            team_initiative_opts = ["Standard Operations", "More frequent breaks", "Team recognition", "Increased Autonomy"]
            current_initiative = st.session_state.get('sb_team_initiative_selectbox', team_initiative_opts[0])
            team_initiative_idx = team_initiative_opts.index(current_initiative) if current_initiative in team_initiative_opts else 0
            team_initiative = st.selectbox("Operational Initiative", team_initiative_opts, index=team_initiative_idx, key="sb_team_initiative_selectbox", help="Apply an operational strategy to observe its impact on metrics.")
            run_simulation_button = st.button("🚀 Run New Simulation", key="sb_run_simulation_button", type="primary", use_container_width=True)
        
        with st.expander("🎨 Visualization Options"):
            st.checkbox("High Contrast Plots", st.session_state.get('sb_high_contrast_checkbox', False), key="sb_high_contrast_checkbox", help="Applies a high-contrast color theme to all charts for better accessibility.")
            st.checkbox("Enable 3D Worker View", st.session_state.get('sb_use_3d_distribution_checkbox', False), key="sb_use_3d_distribution_checkbox", help="Renders worker positions in a 3D scatter plot.")
            st.checkbox("Show Debug Info", st.session_state.get('sb_debug_mode_checkbox', False), key="sb_debug_mode_checkbox", help="Display additional debug information in the sidebar.")
        
        with st.expander("💾 Data Management & Export"): # Same as before
            load_data_button = st.button("🔄 Load Previous Simulation", key="sb_load_data_button", use_container_width=True)
            can_gen_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("📄 Download Report (.tex)", key="sb_pdf_button", disabled=not can_gen_report, use_container_width=True, help="Generates a LaTeX (.tex) file summarizing the simulation. Requires a LaTeX distribution to compile to PDF."):
                if can_gen_report:
                    try:
                        sim_res = st.session_state.simulation_results
                        downtime_events_for_pdf = sim_res.get('downtime_minutes', [])
                        downtime_durations_for_pdf = [event.get('duration',0) for event in downtime_events_for_pdf if isinstance(event, dict)]
                        num_steps = len(downtime_durations_for_pdf)

                        if num_steps == 0:
                            st.warning("⚠️ No simulation data available to generate a report.")
                            raise SystemExit 

                        pdf_data = {k: sim_res.get(k, [np.nan]*num_steps)[:num_steps] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'task_completion_rate']}
                        pdf_data['downtime_minutes'] = downtime_durations_for_pdf[:num_steps]

                        pdf_data.update({
                            'task_compliance': sim_res.get('task_compliance', {}).get('data', [np.nan]*num_steps)[:num_steps],
                            'collaboration_proximity': sim_res.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps)[:num_steps],
                            'worker_wellbeing': sim_res.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps)[:num_steps],
                            'step': list(range(num_steps)),
                            'time_minutes': [i * 2 for i in range(num_steps)]
                        })
                        generate_pdf_report(pd.DataFrame(pdf_data))
                        st.success("✅ LaTeX report (.tex) file 'workplace_report.tex' has been generated.")
                    except SystemExit:
                        pass 
                    except Exception as e:
                        logger.error(f"PDF Generation Error: {e}", exc_info=True, extra={'user_action': 'Generate PDF - Error'})
                        st.error(f"❌ PDF Generation Error: {e}")
            
            if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                sim_res_exp = st.session_state.simulation_results
                downtime_events_for_csv = sim_res_exp.get('downtime_minutes', [])
                downtime_durations_for_csv = [event.get('duration',0) for event in downtime_events_for_csv if isinstance(event, dict)]
                num_steps_csv = len(downtime_durations_for_csv)

                if num_steps_csv > 0:
                    csv_data = {k: sim_res_exp.get(k, [np.nan]*num_steps_csv)[:num_steps_csv] for k in ['operational_recovery', 'psychological_safety', 'productivity_loss', 'task_completion_rate']}
                    csv_data['downtime_minutes'] = downtime_durations_for_csv
                    
                    ww_data_csv = sim_res_exp.get('worker_wellbeing', {})
                    csv_data.update({
                        'task_compliance': sim_res_exp.get('task_compliance', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv],
                        'collaboration_proximity': sim_res_exp.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv],
                        'worker_wellbeing_index': ww_data_csv.get('scores', [np.nan]*num_steps_csv)[:num_steps_csv],
                        'team_cohesion': ww_data_csv.get('team_cohesion_scores', [np.nan]*num_steps_csv)[:num_steps_csv],
                        'perceived_workload': ww_data_csv.get('perceived_workload_scores', [np.nan]*num_steps_csv)[:num_steps_csv],
                        'step': list(range(num_steps_csv)),
                        'time_minutes': [i * 2 for i in range(num_steps_csv)]
                    })
                    st.download_button("📥 Download Data (CSV)", pd.DataFrame(csv_data).to_csv(index=False).encode('utf-8'), "workplace_summary.csv", "text/csv", key="sb_csv_dl_button", use_container_width=True)
                else:
                    st.caption("No detailed data to export for CSV.")
            elif not can_gen_report:
                st.caption("Run simulation for export.")
        
        if st.session_state.get('sb_debug_mode_checkbox', False):
            with st.expander("🛠️ Debug Information", expanded=False):
                st.write("**Default Config (Partial):**")
                st.json({k: DEFAULT_CONFIG.get(k) for k in ['ENTRY_EXIT_POINTS', 'WORK_AREAS', 'DISRUPTION_TIMES_MINUTES']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Active Simulation Config (from results):**")
                    st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else:
                    st.write("**No active simulation data to show config from.**")
        
        st.markdown("## 📋 Help & Info")
        if st.button("ℹ️ Help & Glossary", key="sb_help_button", use_container_width=True):
            st.session_state.show_help_glossary = not st.session_state.get('show_help_glossary', False)
            st.rerun()
        if st.button("🚀 Quick Tour", key="sb_tour_button", use_container_width=True):
            st.session_state.show_tour = not st.session_state.get('show_tour', False)
            st.rerun()
            
    return (st.session_state.sb_team_size_slider, st.session_state.sb_shift_duration_slider,
            st.session_state.sb_disruption_intervals_multiselect, st.session_state.sb_team_initiative_selectbox,
            run_simulation_button, load_data_button,
            st.session_state.sb_high_contrast_checkbox, st.session_state.sb_use_3d_distribution_checkbox,
            st.session_state.sb_debug_mode_checkbox)

@st.cache_data(ttl=3600, show_spinner="⚙️ Running simulation model...")
def run_simulation_logic(team_size, shift_duration_minutes, disruption_intervals_minutes_param, team_initiative_selected): # Same as previous
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // 2
    
    if not isinstance(disruption_intervals_minutes_param, list):
        logger.error(f"run_simulation_logic received non-list for disruption_intervals_minutes_param: {type(disruption_intervals_minutes_param)}. Value: {disruption_intervals_minutes_param}. Defaulting to empty list.")
        disruption_intervals_minutes_param = []
    config['DISRUPTION_EVENT_STEPS'] = sorted(list(set(int(m // 2) for m in disruption_intervals_minutes_param if isinstance(m, (int, float)) and m >= 0)))

    if 'WORK_AREAS' in config and isinstance(config['WORK_AREAS'], dict) and config['WORK_AREAS']:
        total_workers_in_config_zones = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())
        if total_workers_in_config_zones != team_size and team_size > 0:
            logger.info(f"Adjusting worker distribution in config based on team size {team_size}.")
            if total_workers_in_config_zones > 0:
                ratio = team_size / total_workers_in_config_zones
                accumulated_workers = 0
                sorted_zone_keys = sorted(list(config['WORK_AREAS'].keys()))
                for zone_key in sorted_zone_keys[:-1]:
                    assigned = int(round(config['WORK_AREAS'][zone_key].get('workers', 0) * ratio))
                    config['WORK_AREAS'][zone_key]['workers'] = assigned
                    accumulated_workers += assigned
                if sorted_zone_keys: 
                    last_zone_key = sorted_zone_keys[-1]
                    config['WORK_AREAS'][last_zone_key]['workers'] = team_size - accumulated_workers
            else: 
                num_zones = len(config['WORK_AREAS'])
                if num_zones > 0:
                    workers_per_zone = team_size // num_zones
                    remainder_workers = team_size % num_zones
                    for i, zone_key in enumerate(config['WORK_AREAS'].keys()):
                        config['WORK_AREAS'][zone_key]['workers'] = workers_per_zone + (1 if i < remainder_workers else 0)
        elif team_size == 0:
             for zone_key in config['WORK_AREAS']: config['WORK_AREAS'][zone_key]['workers'] = 0
    
    validate_config(config)
    logger.info(f"Running simulation with: Team Size={team_size}, Duration={shift_duration_minutes}min, Disruptions(min): {disruption_intervals_minutes_param}, Disruption Steps: {config['DISRUPTION_EVENT_STEPS']}, Initiative: {team_initiative_selected}", extra={'user_action': 'Run Simulation - Start'})
    
    sim_results_tuple = simulate_workplace_operations(
        num_team_members=team_size,
        num_steps=config['SHIFT_DURATION_INTERVALS'],
        disruption_event_steps=config['DISRUPTION_EVENT_STEPS'],
        team_initiative=team_initiative_selected,
        config=config
    )
    
    expected_keys = ['team_positions_df', 'task_compliance', 'collaboration_proximity', 'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 'worker_wellbeing', 'psychological_safety', 'feedback_impact', 'downtime_minutes', 'task_completion_rate']
    if not isinstance(sim_results_tuple, tuple) or len(sim_results_tuple) != len(expected_keys):
        err_msg = f"Simulation returned unexpected data format. Expected tuple of {len(expected_keys)} items, got {type(sim_results_tuple)} with length {len(sim_results_tuple) if isinstance(sim_results_tuple, (list,tuple)) else 'N/A'}."
        logger.critical(err_msg, extra={'user_action': 'Run Simulation - CRITICAL Data Format Error'})
        raise TypeError(err_msg)
        
    simulation_output_dict = dict(zip(expected_keys, sim_results_tuple))
    simulation_output_dict['config_params'] = {
        'TEAM_SIZE': team_size,
        'SHIFT_DURATION_MINUTES': shift_duration_minutes,
        'DISRUPTION_INTERVALS_MINUTES': disruption_intervals_minutes_param,
        'DISRUPTION_EVENT_STEPS': config['DISRUPTION_EVENT_STEPS'],
        'TEAM_INITIATIVE': team_initiative_selected
    }
    
    save_simulation_data(simulation_output_dict)
    return simulation_output_dict

def safe_get(data_dict, path_str, default_val=None): # Same as previous
    current = data_dict
    default_return = default_val if default_val is not None else [] 
    if not isinstance(path_str, str) or not isinstance(data_dict, dict):
        return default_return
    try:
        for key in path_str.split('.'):
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, (list, pd.Series)) and key.isdigit():
                idx = int(key)
                current = current[idx] if idx < len(current) else None
            else: 
                current = None
                break
        return current if current is not None else default_return
    except (ValueError, IndexError, TypeError) as e:
        logger.debug(f"safe_get failed for path '{path_str}': {e}")
        return default_return

def safe_stat(data_list, stat_func, default_val=0.0): # Same as previous with enhanced logging
    log_data_list_repr = str(data_list)
    if len(log_data_list_repr) > 200: 
        log_data_list_repr = log_data_list_repr[:197] + "..."
    logger.debug(f"safe_stat: Input data (preview): {log_data_list_repr}, func: {stat_func.__name__}, default: {default_val}")

    if not isinstance(data_list, (list, np.ndarray, pd.Series)):
        logger.debug(f"safe_stat: data_list is not a list/array/series, type: {type(data_list)}. Returning default_val: {default_val}")
        return default_val
    
    valid_data = [x for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x))]
    logger.debug(f"safe_stat: Valid data (count: {len(valid_data)}, preview): {str(valid_data)[:100]}")

    if not valid_data:
        logger.debug(f"safe_stat: No valid data after filtering. Returning default_val: {default_val}")
        return default_val
    
    try:
        result = stat_func(valid_data)
        if isinstance(result, float) and np.isnan(result):
            logger.debug(f"safe_stat: stat_func returned NaN. Returning default_val: {default_val}")
            return default_val
        logger.debug(f"safe_stat: stat_func returned: {result}. Type: {type(result)}")
        return result
    except Exception as e: 
        logger.warning(f"safe_stat: Error in stat_func {stat_func.__name__}: {e}. Returning default_val: {default_val}", exc_info=True)
        return default_val

def get_actionable_insights(sim_data, current_config): # Same as previous
    insights = []
    if not sim_data or not isinstance(sim_data, dict): return insights
    
    compliance_data = safe_get(sim_data, 'task_compliance.data', [])
    target_compliance_from_config = float(current_config.get('TARGET_COMPLIANCE', 75.0))
    compliance_avg = safe_stat(compliance_data, np.mean, default_val=target_compliance_from_config)

    if compliance_avg < target_compliance_from_config * 0.9:
        insights.append({"type": "critical", "title": "Low Task Compliance", "text": f"Avg. Task Compliance ({compliance_avg:.1f}%) significantly below target ({target_compliance_from_config:.0f}%). Review disruption impacts, task complexities, and training."})
    elif compliance_avg < target_compliance_from_config:
        insights.append({"type": "warning", "title": "Suboptimal Task Compliance", "text": f"Avg. Task Compliance at {compliance_avg:.1f}%. Identify intervals or areas with lowest compliance for process review."})

    wellbeing_scores = safe_get(sim_data, 'worker_wellbeing.scores', [])
    target_wellbeing_from_config = float(current_config.get('TARGET_WELLBEING', 70.0))
    wellbeing_avg = safe_stat(wellbeing_scores, np.mean, default_val=target_wellbeing_from_config)
    
    wellbeing_critical_threshold_factor = float(current_config.get('WELLBEING_CRITICAL_THRESHOLD_PERCENT_OF_TARGET', 0.85))
    if wellbeing_avg < target_wellbeing_from_config * wellbeing_critical_threshold_factor:
        insights.append({"type": "critical", "title": "Critical Worker Well-being", "text": f"Avg. Well-being ({wellbeing_avg:.1f}%) critically low (target {target_wellbeing_from_config:.0f}%). Urgent review of work conditions, load, and stress factors needed."})
    
    threshold_triggers = safe_get(sim_data, 'worker_wellbeing.triggers.threshold', [])
    if wellbeing_scores and len(threshold_triggers) > max(2, len(wellbeing_scores) * 0.1):
        insights.append({"type": "warning", "title": "Frequent Low Well-being Alerts", "text": f"{len(threshold_triggers)} instances of well-being dropping below threshold. Investigate specific triggers."})

    downtime_events_list = safe_get(sim_data, 'downtime_minutes', [])
    downtime_durations = [event.get('duration', 0.0) for event in downtime_events_list if isinstance(event, dict)]
    total_downtime = safe_stat(downtime_durations, np.sum, default_val=0.0)
    
    sim_cfg_params = sim_data.get('config_params', {})
    shift_mins = float(sim_cfg_params.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
    dt_thresh_total_shift = float(current_config.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT', shift_mins * 0.05))
    
    if total_downtime > dt_thresh_total_shift:
        insights.append({"type": "critical", "title": "Excessive Total Shift Downtime", "text": f"Total shift downtime is {total_downtime:.0f} minutes, exceeding the guideline of {dt_thresh_total_shift:.0f} min. Deep dive into disruption causes, equipment reliability, and recovery protocols. Analyze downtime causes pie chart."})
    
    psych_safety_scores = safe_get(sim_data, 'psychological_safety', [])
    target_psych_safety = float(current_config.get('TARGET_PSYCH_SAFETY', 70.0))
    psych_safety_avg = safe_stat(psych_safety_scores, np.mean, default_val=target_psych_safety)
    if psych_safety_avg < target_psych_safety * 0.9:
        insights.append({"type": "warning", "title": "Low Psychological Safety", "text": f"Avg. Psych. Safety ({psych_safety_avg:.1f}%) is below target ({target_psych_safety:.0f}%). Consider initiatives to build trust and open communication."})

    cohesion_scores = safe_get(sim_data, 'worker_wellbeing.team_cohesion_scores', [])
    target_cohesion = float(current_config.get('TARGET_TEAM_COHESION', 70.0))
    cohesion_avg = safe_stat(cohesion_scores, np.mean, default_val=target_cohesion)
    if cohesion_avg < target_cohesion * 0.9:
        insights.append({"type": "warning", "title": "Low Team Cohesion", "text": f"Avg. Team Cohesion ({cohesion_avg:.1f}%) is below desired levels. Consider team-building activities or structural reviews for collaboration."})
    
    workload_scores = safe_get(sim_data, 'worker_wellbeing.perceived_workload_scores', [])
    target_workload = float(current_config.get('TARGET_PERCEIVED_WORKLOAD', 6.5))
    workload_avg = safe_stat(workload_scores, np.mean, default_val=target_workload)
    if workload_avg > float(current_config.get('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5)):
        insights.append({"type": "critical", "title": "Very High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) is critically high. Immediate review of task distribution, staffing, and process efficiencies is required."})
    elif workload_avg > float(current_config.get('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5)):
        insights.append({"type": "warning", "title": "High Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) exceeds high threshold. Monitor closely and identify bottlenecks."})
    elif workload_avg > target_workload:
        insights.append({"type": "info", "title": "Elevated Perceived Workload", "text": f"Avg. Perceived Workload ({workload_avg:.1f}/10) is above target ({target_workload:.1f}/10). Consider proactive adjustments."})
    
    if compliance_avg > target_compliance_from_config * 1.05 and \
       wellbeing_avg > target_wellbeing_from_config * 1.05 and \
       total_downtime < dt_thresh_total_shift * 0.5 and \
       psych_safety_avg > target_psych_safety * 1.05:
        insights.append({"type": "positive", "title": "Holistically Excellent Performance", "text": "Key operational and psychosocial metrics significantly exceed targets. A well-balanced and high-performing shift! Leadership should identify and replicate success factors."})
    
    initiative = sim_data.get('config_params', {}).get('TEAM_INITIATIVE', 'Standard Operations')
    if initiative != "Standard Operations":
        insights.append({"type": "info", "title": f"Initiative Active: '{initiative}'", "text": f"The '{initiative}' initiative was simulated. Its impact can be assessed by comparing metrics to a 'Standard Operations' baseline run."})
    
    return insights

# Helper function for time range inputs
def time_range_input_section(tab_key_prefix: str, max_minutes: int, st_col_obj = st):
    """Creates start and end time number inputs and returns the selected range."""
    # Use unique keys for session state based on tab_key_prefix
    start_time_key = f"{tab_key_prefix}_start_time_min"
    end_time_key = f"{tab_key_prefix}_end_time_min"

    # Initialize session state if not present
    if start_time_key not in st.session_state:
        st.session_state[start_time_key] = 0
    if end_time_key not in st.session_state:
        st.session_state[end_time_key] = max_minutes
    
    # Ensure current values are within new max_minutes bounds
    st.session_state[start_time_key] = min(st.session_state[start_time_key], max_minutes)
    st.session_state[end_time_key] = min(st.session_state[end_time_key], max_minutes)
    st.session_state[start_time_key] = max(0, st.session_state[start_time_key]) # Ensure start >= 0
    st.session_state[end_time_key] = max(st.session_state[start_time_key], st.session_state[end_time_key]) # Ensure end >= start

    cols = st_col_obj.columns(2)
    start_time = cols[0].number_input(
        "Start Time (min)", 
        min_value=0, 
        max_value=max_minutes, 
        value=st.session_state[start_time_key], 
        step=2, 
        key=f"num_{start_time_key}", # Add prefix for widget key uniqueness
        help="Select the start of the time range (in minutes from shift start)."
    )
    end_time = cols[1].number_input(
        "End Time (min)", 
        min_value=start_time, # Dynamic min_value based on start_time
        max_value=max_minutes, 
        value=st.session_state[end_time_key], 
        step=2, 
        key=f"num_{end_time_key}",
        help="Select the end of the time range (in minutes from shift start)."
    )
    
    # Update session state if values change
    if st.session_state[f"num_{start_time_key}"] != st.session_state[start_time_key]:
         st.session_state[start_time_key] = st.session_state[f"num_{start_time_key}"]
         # If start time changed, ensure end time is not less than new start time
         if st.session_state[end_time_key] < st.session_state[start_time_key]:
             st.session_state[end_time_key] = st.session_state[start_time_key]
         st.rerun() # Rerun to update dependent inputs/plots

    if st.session_state[f"num_{end_time_key}"] != st.session_state[end_time_key]:
         st.session_state[end_time_key] = st.session_state[f"num_{end_time_key}"]
         st.rerun()

    return int(start_time), int(end_time)


def main():
    st.title("Workplace Shift Optimization Dashboard")
    app_state_keys = ['simulation_results', 'show_tour', 'show_help_glossary',
                      'op_start_time_min', 'op_end_time_min', # For Operational Metrics Tab
                      'ww_start_time_min', 'ww_end_time_min',   # For Worker Wellbeing Tab
                      'dt_start_time_min', 'dt_end_time_min']   # For Downtime Analysis Tab
                      # Note: Snapshot slider is separate and its state key is used directly.
    for key in app_state_keys:
        if key not in st.session_state:
            st.session_state[key] = None # Initialize if not present

    sb_team_size, sb_shift_duration, sb_disrupt_mins_from_sidebar, sb_team_initiative, \
    sb_run_sim_btn, sb_load_data_btn, sb_high_contrast_checkbox_val, \
    sb_use_3d_val, sb_debug_mode_val = render_settings_sidebar()

    _default_shift_duration = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']
    current_max_minutes_for_sliders = _default_shift_duration - 2 # Default before sim results
    disruption_steps_for_plots = []

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        num_intervals_from_sim = len(safe_get(st.session_state.simulation_results, 'downtime_minutes', []))
        if num_intervals_from_sim > 0:
            # Max minutes is (num_intervals - 1) * 2 because time starts at 0.
            # If num_intervals is 1 (0 to 2 min), max_minutes should be 0 for slider (0 to 0 range)
            # If num_intervals is 2 (0 to 4 min), max_minutes should be 2 (0 to 2 range)
            current_max_minutes_for_sliders = max(0, (num_intervals_from_sim - 1) * 2)
        else: # No steps in sim results
            current_max_minutes_for_sliders = 0

        disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_EVENT_STEPS', [])
    else: # No simulation results yet
        _disrupt_mins_list_for_plots = sb_disrupt_mins_from_sidebar if isinstance(sb_disrupt_mins_from_sidebar, list) else []
        disruption_steps_for_plots = [int(m // 2) for m in _disrupt_mins_list_for_plots if isinstance(m, (int, float))]
        # current_max_minutes_for_sliders already set based on sidebar shift_duration before sim run
        current_max_minutes_for_sliders = sb_shift_duration -2 if sb_shift_duration else _default_shift_duration - 2
    
    current_max_minutes_for_sliders = max(0, current_max_minutes_for_sliders) # Ensure non-negative
    logger.debug(f"Main: current_max_minutes_for_sliders set to {current_max_minutes_for_sliders}")


    # ... (Simulation run and load logic - same as previous) ...
    if sb_run_sim_btn:
        with st.spinner("🚀 Simulating workplace operations..."):
            try:
                final_disrupt_mins_for_sim = sb_disrupt_mins_from_sidebar
                if not isinstance(final_disrupt_mins_for_sim, list):
                    logger.error(f"CRITICAL in main run_sim: sb_disrupt_mins for simulation was {type(final_disrupt_mins_for_sim)}. Defaulting to empty list.")
                    final_disrupt_mins_for_sim = []
                st.session_state.simulation_results = run_simulation_logic(sb_team_size, sb_shift_duration, final_disrupt_mins_for_sim, sb_team_initiative)
                # Reset time range inputs to full new range after new simulation
                for prefix in ['op', 'ww', 'dt']:
                    st.session_state[f"{prefix}_start_time_min"] = 0
                    # Calculate new max based on the *new* simulation's duration
                    new_sim_intervals = st.session_state.simulation_results['config_params']['SHIFT_DURATION_MINUTES'] // 2
                    new_max_mins = max(0, (new_sim_intervals -1) * 2)
                    st.session_state[f"{prefix}_end_time_min"] = new_max_mins

                st.success("✅ Simulation completed!")
                logger.info("Simulation run successful.", extra={'user_action': 'Run Simulation - Success'})
                st.rerun()
            except Exception as e:
                logger.error(f"Simulation Run Error: {e}", exc_info=True, extra={'user_action': 'Run Simulation - Error'})
                st.error(f"❌ Simulation failed: {e}")
                st.session_state.simulation_results = None

    if sb_load_data_btn:
        with st.spinner("🔄 Loading saved simulation data..."):
            try:
                loaded_data = load_simulation_data()
                if loaded_data and isinstance(loaded_data, dict):
                    st.session_state.simulation_results = loaded_data
                    cfg = loaded_data.get('config_params', {})
                    loaded_disrupt_mins = cfg.get('DISRUPTION_INTERVALS_MINUTES', [])
                    if not isinstance(loaded_disrupt_mins, list):
                        logger.warning(f"Loaded DISRUPTION_INTERVALS_MINUTES was {type(loaded_disrupt_mins)}. Defaulting to [].")
                        loaded_disrupt_mins = []
                    st.session_state.sb_team_size_slider = cfg.get('TEAM_SIZE', st.session_state.get('sb_team_size_slider'))
                    st.session_state.sb_shift_duration_slider = cfg.get('SHIFT_DURATION_MINUTES', st.session_state.get('sb_shift_duration_slider'))
                    st.session_state.sb_team_initiative_selectbox = cfg.get('TEAM_INITIATIVE', st.session_state.get('sb_team_initiative_selectbox'))
                    st.session_state.sb_disruption_intervals_multiselect = loaded_disrupt_mins
                     # Reset time range inputs to full new range after loading
                    for prefix in ['op', 'ww', 'dt']:
                        st.session_state[f"{prefix}_start_time_min"] = 0
                        new_sim_intervals = cfg.get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) // 2
                        new_max_mins = max(0, (new_sim_intervals -1) * 2)
                        st.session_state[f"{prefix}_end_time_min"] = new_max_mins
                    st.success("✅ Data loaded successfully!")
                    logger.info("Saved data loaded successfully.", extra={'user_action': 'Load Data - Success'})
                    st.rerun()
                else:
                    st.error("❌ Failed to load data or data is not in the expected dictionary format.")
                    logger.warning("Load data failed or invalid format.", extra={'user_action': 'Load Data - Fail/Invalid'})
            except Exception as e:
                logger.error(f"Load Data Error: {e}", exc_info=True, extra={'user_action': 'Load Data - Error'})
                st.error(f"❌ Failed to load data: {e}")
                st.session_state.simulation_results = None


    if st.session_state.get('show_tour'): # Same as before
        with st.container(): st.markdown("""<div class="onboarding-modal"><h3>🚀 Quick Dashboard Tour</h3><p>Welcome! This dashboard helps you monitor and analyze workplace shift operations. Use the sidebar to configure simulations and navigate. The main area displays results across several tabs: Overview, Operational Metrics, Worker Well-being (including psychosocial factors and spatial dynamics), Downtime Analysis, and a Glossary. Interactive charts and actionable insights will guide you in optimizing operations.</p><p>Start by running a new simulation or loading previous data from the sidebar!</p></div>""", unsafe_allow_html=True)
        if st.button("Got it!", key="tour_modal_close_btn_main"): st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help_glossary'): # Same as before
        with st.container(): st.markdown(""" <div class="onboarding-modal"><h3>ℹ️ Help & Glossary</h3> <p>This dashboard provides insights into simulated workplace operations. Use the sidebar to configure and run simulations or load previously saved data. Navigate through the analysis using the main tabs above.</p><h4>Metric Definitions:</h4> <ul style="font-size: 0.85rem; list-style-type: disc; padding-left: 20px;"> <li><b>Task Compliance Score:</b> Percentage of tasks completed correctly and on time.</li><li><b>Collaboration Proximity Index:</b> Percentage of workers near colleagues, indicating teamwork potential.</li><li><b>Operational Recovery Score:</b> Ability to maintain output after disruptions.</li><li><b>Worker Well-Being Index:</b> Composite score of fatigue, stress levels, and job satisfaction.</li><li><b>Psychological Safety Score:</b> Comfort level in reporting issues or suggesting improvements.</li><li><b>Team Cohesion Index:</b> Measure of bonds and sense of belonging within a team.</li><li><b>Perceived Workload Index:</b> Indicator of how demanding workers perceive their tasks (0-10 scale).</li><li><b>Uptime:</b> Percentage of time equipment is operational.</li><li><b>Throughput:</b> Percentage of maximum production rate achieved.</li><li><b>Quality Rate:</b> Percentage of products meeting quality standards.</li><li><b>OEE (Overall Equipment Effectiveness):</b> Combined score of Uptime, Throughput, and Quality Rate.</li><li><b>Productivity Loss:</b> Percentage of potential output lost due to inefficiencies.</li><li><b>Downtime (per interval):</b> Total minutes of unplanned operational stops.</li><li><b>Task Completion Rate:</b> Percentage of tasks completed per time interval.</li></ul><p>For further assistance, please refer to the detailed documentation or contact support@example.com.</p></div> """, unsafe_allow_html=True)
        if st.button("Understood", key="help_modal_close_btn_main"): st.session_state.show_help_glossary = False; st.rerun()

    tabs_main_names = ["📊 Overview & Insights", "📈 Operational Metrics", "👥 Worker Well-being", "⏱️ Downtime Analysis", "📖 Glossary"]
    tabs = st.tabs(tabs_main_names)
    plot_config_interactive = {'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d', 'pan2d'], 'toImageButtonOptions': {'format': 'png', 'filename': 'plot_export', 'scale': 2}}
    plot_config_minimal = {'displayModeBar': False}
    current_high_contrast_setting = sb_high_contrast_checkbox_val

    with tabs[0]: # Overview Tab
        st.header("📊 Key Performance Indicators & Actionable Insights", divider="blue")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results
            effective_config = {**DEFAULT_CONFIG, **sim_data.get('config_params', {})}
            
            compliance_target = float(effective_config.get('TARGET_COMPLIANCE', 75.0))
            collab_target = float(effective_config.get('TARGET_COLLABORATION', 60.0))
            wb_target = float(effective_config.get('TARGET_WELLBEING', 70.0))
            
            downtime_events_overview = safe_get(sim_data, 'downtime_minutes', [])
            downtime_durations_overview = [event.get('duration', 0.0) for event in downtime_events_overview if isinstance(event, dict)]
            
            compliance_val_raw = safe_stat(safe_get(sim_data, 'task_compliance.data', []), np.mean, default_val=compliance_target)
            proximity_val_raw = safe_stat(safe_get(sim_data, 'collaboration_proximity.data', []), np.mean, default_val=collab_target)
            wellbeing_val_raw = safe_stat(safe_get(sim_data, 'worker_wellbeing.scores', []), np.mean, default_val=wb_target)
            downtime_total_overview_raw = safe_stat(downtime_durations_overview, np.sum, default_val=0.0)

            try:
                compliance = float(compliance_val_raw) if pd.notna(compliance_val_raw) else 0.0
                proximity = float(proximity_val_raw) if pd.notna(proximity_val_raw) else 0.0
                wellbeing = float(wellbeing_val_raw) if pd.notna(wellbeing_val_raw) else 0.0
                downtime_total_overview = float(downtime_total_overview_raw) if pd.notna(downtime_total_overview_raw) else 0.0
            except (ValueError, TypeError) as e:
                logger.error(f"Overview Metrics - Float Conversion Error: {e}. Raw values: C={compliance_val_raw}, P={proximity_val_raw}, W={wellbeing_val_raw}, DT={downtime_total_overview_raw}")
                compliance, proximity, wellbeing, downtime_total_overview = 0.0, 0.0, 0.0, 0.0 

            sim_duration_minutes = float(sim_data.get('config_params', {}).get('SHIFT_DURATION_MINUTES', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']))
            dt_target_total_shift = float(effective_config.get('DOWNTIME_THRESHOLD_TOTAL_SHIFT', sim_duration_minutes * 0.05))

            cols_metrics = st.columns(4)
            cols_metrics[0].metric("Task Compliance", f"{compliance:.1f}%", f"{compliance-compliance_target:.1f}% vs Target {compliance_target:.0f}%")
            cols_metrics[1].metric("Collaboration Index", f"{proximity:.1f}%", f"{proximity-collab_target:.1f}% vs Target {collab_target:.0f}%")
            cols_metrics[2].metric("Worker Well-Being", f"{wellbeing:.1f}%", f"{wellbeing-wb_target:.1f}% vs Target {wb_target:.0f}%")
            cols_metrics[3].metric("Total Downtime", f"{downtime_total_overview:.1f} min", f"{downtime_total_overview-dt_target_total_shift:.1f} min vs Target {dt_target_total_shift:.0f}min", delta_color="inverse")
            
            logger.info(f"Overview Gauges - Input Values: Compliance={compliance}, Proximity={proximity}, Wellbeing={wellbeing}, Downtime={downtime_total_overview}")
            logger.info(f"Overview Gauges - Input Targets: CompT={compliance_target}, ProxT={collab_target}, WBT={wb_target}, DTT={dt_target_total_shift}")

            try:
                summary_figs = plot_key_metrics_summary(
                    compliance=compliance, proximity=proximity, wellbeing=wellbeing, downtime=downtime_total_overview,
                    target_compliance=compliance_target, target_proximity=collab_target, 
                    target_wellbeing=wb_target, target_downtime=dt_target_total_shift,
                    high_contrast=current_high_contrast_setting,
                    color_positive=COLOR_POSITIVE_GREEN, color_warning=COLOR_WARNING_AMBER,
                    color_negative=COLOR_CRITICAL_RED, accent_color=COLOR_ACCENT_INDIGO
                )
                if summary_figs:
                    cols_gauges = st.columns(min(len(summary_figs), 4) or 1)
                    for i, fig_gauge in enumerate(summary_figs):
                        with cols_gauges[i % len(cols_gauges)]:
                            st.plotly_chart(fig_gauge, use_container_width=True, config=plot_config_minimal)
                else:
                    st.caption("Gauge charts could not be generated (plot_key_metrics_summary returned None/empty).")
            except Exception as e:
                logger.error(f"Overview Gauges Plotting Error in main.py: {e}", exc_info=True)
                st.error(f"⚠️ Error rendering overview gauges: {str(e)}")
            
            st.markdown("---"); st.subheader("💡 Key Insights & Leadership Actions")
            actionable_insights = get_actionable_insights(sim_data, effective_config)
            if actionable_insights:
                for insight in actionable_insights:
                    alert_class = f"alert-{insight['type']}"
                    st.markdown(f'<div class="{alert_class}"><p class="insight-title">{insight["title"]}</p><p class="insight-text">{insight["text"]}</p></div>', unsafe_allow_html=True)
            else:
                st.info("✅ No critical alerts or specific insights identified. Performance appears stable against defined thresholds.", icon="👍")
            
            with st.expander("View Detailed Overview Data Table", expanded=False):
                downtime_durations_for_table = [event.get('duration', np.nan) for event in safe_get(sim_data, 'downtime_minutes', []) if isinstance(event, dict)]
                num_s = len(downtime_durations_for_table)
                if num_s > 0:
                    df_data = {'Time (min)': [i*2 for i in range(num_s)]}
                    df_data.update({
                        'Task Compliance (%)': safe_get(sim_data, 'task_compliance.data', [np.nan]*num_s)[:num_s],
                        'Collaboration (%)': safe_get(sim_data, 'collaboration_proximity.data', [np.nan]*num_s)[:num_s],
                        'Well-Being (%)': safe_get(sim_data, 'worker_wellbeing.scores', [np.nan]*num_s)[:num_s],
                        'Downtime (min)': downtime_durations_for_table
                    })
                    st.dataframe(pd.DataFrame(df_data).style.format("{:.1f}", na_rep="-").set_table_styles([{'selector': 'th', 'props': [('background-color', '#293344'), ('color', '#EAEAEA')]}]), use_container_width=True, height=300)
                else:
                    st.caption("No detailed overview data.")
        else:
            st.info("ℹ️ Run a simulation or load data to view the Overview & Insights.", icon="📊")

    # --- Tab Content with Number Inputs for Time Range ---
    tab_configs = [
        {"name": "📈 Operational Metrics", "key_prefix": "op", 
         "plots": [
             {"title": "Task Compliance Score Over Time", "data_path": "task_compliance.data", "plot_func": plot_task_compliance_score, "extra_args_paths": {"forecast_data": "task_compliance.forecast", "z_scores": "task_compliance.z_scores"}},
             {"title": "Collaboration Proximity Index Over Time", "data_path": "collaboration_proximity.data", "plot_func": plot_collaboration_proximity_index, "extra_args_paths": {"forecast_data": "collaboration_proximity.forecast"}},
             {"title": "Additional Operational Metrics", "is_subheader": True},
             {"title": "Operational Recovery", "data_path": "operational_recovery", "plot_func": plot_operational_recovery, "extra_args_paths": {"productivity_loss_data": "productivity_loss"}},
             {"title": "OEE", "is_oee": True} # Special case for OEE
         ],
         "insights_html": """<div class='alert-info insight-text' style='margin-top:1rem;'><p class="insight-title">Review Operational Bottlenecks:</p><ul><li><b>Low Compliance/OEE:</b> ...</li></ul></div>"""
        },
        {"name": "👥 Worker Well-being", "key_prefix": "ww", 
         "plots": [
             {"is_subheader": True, "title": "Psychosocial & Well-being Indicators"},
             {"title": "Worker Well-Being Index", "data_path": "worker_wellbeing.scores", "plot_func": plot_worker_wellbeing, "extra_args_paths": {"triggers": "worker_wellbeing.triggers"}},
             {"title": "Psychological Safety Score", "data_path": "psychological_safety", "plot_func": plot_psychological_safety},
             {"title": "Team Cohesion Index", "data_path": "worker_wellbeing.team_cohesion_scores", "plot_func": plot_team_cohesion},
             {"title": "Perceived Workload Index (0-10)", "data_path": "worker_wellbeing.perceived_workload_scores", "plot_func": plot_perceived_workload, "extra_args_fixed": {"high_workload_threshold": DEFAULT_CONFIG.get('PERCEIVED_WORKLOAD_THRESHOLD_HIGH', 7.5), "very_high_workload_threshold": DEFAULT_CONFIG.get('PERCEIVED_WORKLOAD_THRESHOLD_VERY_HIGH', 8.5)}},
             {"is_subheader": True, "title": "Spatial Dynamics Analysis", "is_spatial": True}
         ],
          "insights_html": """...""" # Well-being insights HTML
        },
        {"name": "⏱️ Downtime Analysis", "key_prefix": "dt", 
         "metrics_display": True,
         "plots": [
            {"title": "Downtime Trend (per Interval)", "data_path": "downtime_minutes", "plot_func": plot_downtime_trend, "extra_args_fixed": {"interval_threshold": DEFAULT_CONFIG.get('DOWNTIME_PLOT_ALERT_THRESHOLD', 10)}},
            {"title": "Downtime Distribution by Cause", "data_path": "downtime_minutes", "plot_func": plot_downtime_causes_pie}
         ],
         "insights_html": """<div class='alert-info insight-text' style='margin-top:1rem;'><p class="insight-title">Focus Areas for Downtime Reduction:</p><ul><li>...</li></ul></div>"""
        }
    ]

    for i, tab_config in enumerate(tab_configs):
        with tabs[i+1]: # Start from tabs[1] as tabs[0] is Overview
            st.header(tab_config["name"], divider="blue")
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                
                st.markdown("##### Select Time Range for Plots:")
                start_time_min, end_time_min = time_range_input_section(
                    tab_config["key_prefix"], current_max_minutes_for_sliders
                )
                start_idx, end_idx = start_time_min // 2, end_time_min // 2 + 1
                
                logger.debug(f"Tab '{tab_config['name']}': Time range {start_time_min}-{end_time_min} min. Indices {start_idx}-{end_idx}. Max mins: {current_max_minutes_for_sliders}")
                
                filt_disrupt_steps = [s for s in disruption_steps_for_plots if start_idx <= s < end_idx]

                if tab_config.get("metrics_display"): # For Downtime Tab
                    downtime_events_list_all = safe_get(sim_data, 'downtime_minutes', [])
                    downtime_events_filtered = []
                    if start_idx < len(downtime_events_list_all):
                        downtime_events_filtered = downtime_events_list_all[start_idx:min(end_idx, len(downtime_events_list_all))]
                    
                    downtime_durations_filtered = [event.get('duration',0.0) for event in downtime_events_filtered if isinstance(event, dict)]
                    if downtime_events_filtered:
                        total_downtime_period = sum(downtime_durations_filtered)
                        num_incidents = len([d for d in downtime_durations_filtered if d > 0])
                        avg_duration_per_incident = total_downtime_period / num_incidents if num_incidents > 0 else 0.0
                        
                        dt_cols_metrics = st.columns(3)
                        dt_cols_metrics[0].metric("Total Downtime in Period", f"{total_downtime_period:.1f} min")
                        dt_cols_metrics[1].metric("Number of Incidents", f"{num_incidents}")
                        dt_cols_metrics[2].metric("Avg. Duration / Incident", f"{avg_duration_per_incident:.1f} min")

                # Plot rendering loop
                plot_columns = st.columns(2) # Max 2 plots per row for better layout
                plot_col_idx = 0

                for plot_info in tab_config["plots"]:
                    if plot_info.get("is_subheader"):
                        # If there was an active column, reset for next plots to be full width
                        if plot_col_idx % 2 != 0: plot_col_idx = 0 
                        st.subheader(plot_info["title"])
                        if plot_info.get("is_spatial"): # Spatial plots might need their own controls
                            with st.container(border=True):
                                team_pos_df_all = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
                                zones_dist = ["All"] + list(DEFAULT_CONFIG.get('WORK_AREAS', {}).keys())
                                zone_sel_dist = st.selectbox("Filter by Zone:", zones_dist, key=f"{tab_config['key_prefix']}_zone_sel_spatial")
                                
                                filt_team_pos_df_spatial = team_pos_df_all
                                if not filt_team_pos_df_spatial.empty:
                                    filt_team_pos_df_spatial = filt_team_pos_df_spatial[(filt_team_pos_df_spatial['step'] >= start_idx) & (filt_team_pos_df_spatial['step'] < end_idx)]
                                if zone_sel_dist != "All" and not filt_team_pos_df_spatial.empty:
                                    filt_team_pos_df_spatial = filt_team_pos_df_spatial[filt_team_pos_df_spatial['zone'] == zone_sel_dist]

                                show_ee_exp = st.checkbox("Show E/E Points", value=st.session_state.get(f'{tab_config["key_prefix"]}_show_ee_spatial', True), key=f"{tab_config['key_prefix']}_show_ee_spatial_cb")
                                show_pl_exp = st.checkbox("Show Area Outlines", value=st.session_state.get(f'{tab_config["key_prefix"]}_show_pl_spatial', True), key=f"{tab_config['key_prefix']}_show_pl_spatial_cb")
                                
                                spatial_plot_cols = st.columns(2)
                                with spatial_plot_cols[0]:
                                    st.markdown("<h6>Worker Positions (Snapshot)</h6>", unsafe_allow_html=True)
                                    min_snap_step, max_snap_step = start_idx, max(start_idx, end_idx -1)
                                    snap_key = f"{tab_config['key_prefix']}_snap_step"
                                    
                                    default_snap = min_snap_step
                                    if snap_key not in st.session_state: st.session_state[snap_key] = default_snap
                                    # Validate current session state value for snapshot slider
                                    if not (min_snap_step <= st.session_state[snap_key] <= max_snap_step):
                                        st.session_state[snap_key] = default_snap

                                    snap_step_val = st.slider("Snapshot Time Step:", min_snap_step, max_snap_step, st.session_state[snap_key], 1, key=f"num_{snap_key}", disabled=(max_snap_step <= min_snap_step))
                                    if st.session_state[f"num_{snap_key}"] != st.session_state[snap_key]:
                                        st.session_state[snap_key] = st.session_state[f"num_{snap_key}"]
                                        # No rerun needed for slider typically, plot redraws based on value

                                    if not team_pos_df_all.empty and max_snap_step >= min_snap_step:
                                        try:
                                            st.plotly_chart(plot_worker_distribution(team_pos_df_all, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, sb_use_3d_val, snap_step_val, show_ee_exp, show_pl_exp, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                                        except Exception as e: logger.error(f"Spatial Dist Plot Error: {e}", exc_info=True); st.error(f"⚠️ Error plotting Worker Positions: {str(e)}.")
                                    else: st.caption("No data for positions snapshot.")
                                with spatial_plot_cols[1]:
                                    st.markdown("<h6>Worker Density (Aggregated)</h6>", unsafe_allow_html=True)
                                    if not filt_team_pos_df_spatial.empty:
                                        try:
                                            st.plotly_chart(plot_worker_density_heatmap(filt_team_pos_df_spatial, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, show_ee_exp, show_pl_exp, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                                        except Exception as e: logger.error(f"Spatial Heatmap Plot Error: {e}", exc_info=True); st.error(f"⚠️ Error plotting Density Heatmap: {str(e)}.")
                                    else: st.caption("No data for density heatmap.")
                        continue # Move to next plot_info after handling subheader/spatial

                    current_plot_col = plot_columns[plot_col_idx % 2]
                    with current_plot_col:
                        with st.container(border=True):
                            st.markdown(f'<h5>{plot_info["title"]}</h5>', unsafe_allow_html=True)
                            try:
                                if plot_info.get("is_oee"):
                                    eff_df_full = safe_get(sim_data, 'efficiency_metrics_df', pd.DataFrame())
                                    if not eff_df_full.empty:
                                        sel_metrics = st.multiselect("Select OEE Metrics:", ['uptime', 'throughput', 'quality', 'oee'], default=['uptime', 'throughput', 'quality', 'oee'], key=f"{tab_config['key_prefix']}_oee_metrics_ms")
                                        filt_eff_df = eff_df_full.iloc[start_idx:end_idx] if start_idx < end_idx and start_idx < len(eff_df_full) and end_idx <= len(eff_df_full) else pd.DataFrame()
                                        if not filt_eff_df.empty:
                                            st.plotly_chart(plot_operational_efficiency(filt_eff_df, sel_metrics, current_high_contrast_setting), use_container_width=True, config=plot_config_interactive)
                                        else: st.caption("No OEE data for this time range.")
                                    else: st.caption("No OEE data available.")
                                else:
                                    plot_data_raw = safe_get(sim_data, plot_info["data_path"], [])
                                    plot_data_list = []
                                    if isinstance(plot_data_raw, list): # Ensure it's a list before slicing
                                         plot_data_list = plot_data_raw[start_idx:end_idx] if start_idx < end_idx and start_idx < len(plot_data_raw) else []
                                    elif isinstance(plot_data_raw, pd.DataFrame) and not plot_data_raw.empty : # Handle if data path leads to a DataFrame directly
                                        plot_data_list = plot_data_raw.iloc[start_idx:end_idx] if start_idx < end_idx and start_idx < len(plot_data_raw) and end_idx <= len(plot_data_raw) else pd.DataFrame()

                                    if (isinstance(plot_data_list, list) and plot_data_list) or \
                                       (isinstance(plot_data_list, pd.DataFrame) and not plot_data_list.empty):
                                        
                                        kwargs = {}
                                        if "extra_args_paths" in plot_info:
                                            for arg_name, arg_path in plot_info["extra_args_paths"].items():
                                                extra_data_raw = safe_get(sim_data, arg_path, [])
                                                if isinstance(extra_data_raw, list):
                                                    kwargs[arg_name] = extra_data_raw[start_idx:end_idx] if start_idx < end_idx and start_idx < len(extra_data_raw) else []
                                                else: # Assuming it could be a dict like 'triggers'
                                                    kwargs[arg_name] = extra_data_raw 
                                        if "extra_args_fixed" in plot_info:
                                            kwargs.update(plot_info["extra_args_fixed"])
                                        
                                        # Always pass disruption points if the plot function expects it (most line charts do)
                                        if "disruption_points" in plot_info["plot_func"].__code__.co_varnames:
                                            kwargs["disruption_points"] = filt_disrupt_steps

                                        st.plotly_chart(plot_info["plot_func"](plot_data_list, high_contrast=current_high_contrast_setting, **kwargs), use_container_width=True, config=plot_config_interactive)
                                    else:
                                        st.caption(f"No data for {plot_info['title']} in this time range.")
                            except Exception as e:
                                logger.error(f"Tab '{tab_config['name']}', Plot '{plot_info['title']}' Error: {e}", exc_info=True)
                                st.error(f"⚠️ Error plotting {plot_info['title']}: {str(e)}")
                    plot_col_idx +=1
                
                # Insights section for the tab
                if tab_config.get("insights_html"):
                    if plot_col_idx % 2 != 0: plot_col_idx = 0 # Reset for full width
                    st.markdown("<hr><h3 style='text-align:center;'>🏛️ Leadership Actionable Insights</h3>", unsafe_allow_html=True)
                    st.markdown(tab_config["insights_html"], unsafe_allow_html=True)

            else: # No simulation results
                st.info(f"ℹ️ Run a simulation or load data to view {tab_config['name']}.", icon="📊")
    
    # Glossary Tab (tabs[4]) - Same as previous
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
            <details><summary><strong>Team Cohesion Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">A measure of the strength of bonds and sense of belonging within a team. Higher cohesion often correlates with better communication and mutual support. <em>Range: 0-100%. Higher is better.</em></p></details>
            <details><summary><strong>Perceived Workload Index</strong></summary><p style="padding-left: 20px; font-size:0.9rem;">An indicator of how demanding workers perceive their tasks and overall workload on a scale (e.g., 0-10). Persistently high scores can lead to stress and burnout. <em>Lower is generally better.</em></p></details>
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
