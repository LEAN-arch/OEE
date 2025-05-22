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

# Placeholder base64 logo (replace with actual base64 string in production)
LEAN_LOGO_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgG6NcxuAAAAAElFTkSuQmCC"

# Debug log to confirm file parsing
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log',
    filemode='a' # Append to log file
)
logger.info("Successfully parsed main.py imports", extra={'user_action': 'Parse File'})

# Streamlit page config
st.set_page_config(
    page_title="Workplace Shift Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, accessible design with improved typography
st.markdown("""
    <style>
        /* Base Styles */
        .main {
            background-color: #1E2A44;
            color: #F5F7FA;
            font-family: 'Roboto', 'Open Sans', 'Helvetica Neue', sans-serif;
            padding: 32px;
        }
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: -0.02em;
            text-align: center;
            margin-bottom: 2rem;
        }
        h2 {
            font-size: 1.875rem;
            font-weight: 600;
            line-height: 1.3;
            margin: 1.5rem 0 1rem;
        }
        h3 {
            font-size: 1.375rem;
            font-weight: 500;
            line-height: 1.4;
            margin-bottom: 0.75rem;
        }
        h5 { /* For plot titles inside containers */
            font-size: 1.1rem;
            font-weight: 500;
            line-height: 1.3;
            margin: 0.5rem 0 0.5rem;
            color: #E0E0E0; /* Lighter color for subplot titles */
        }
        h6 { /* For smaller titles or notes */
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.3;
            margin: 0.5rem 0 0.25rem;
            color: #C0C0C0;
        }
        .stButton>button {
            background-color: #4F46E5;
            color: #F5F7FA;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover, .stButton>button:focus {
            background-color: #EC4899;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            outline: none;
        }
        .stSelectbox, .stSlider, .stMultiSelect {
            background-color: #2D3B55;
            color: #F5F7FA;
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 16px;
            font-size: 1rem;
        }
        .tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            margin-left: 8px;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 300px;
            background-color: #2D3B55;
            color: #F5F7FA;
            text-align: left;
            border-radius: 8px;
            padding: 16px;
            position: absolute;
            z-index: 10;
            top: 100%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            font-size: 0.875rem;
            line-height: 1.5;
        }
        .tooltip:hover .tooltiptext, .tooltip:focus .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stSidebar"] {
            background-color: #2D3B55;
            color: #F5F7FA;
            padding: 24px;
            border-right: 1px solid #4B5EAA;
            font-size: 1rem;
        }
        [data-testid="stSidebar"] .stButton>button {
            background-color: #10B981; /* Sidebar specific button color */
            width: 100%; /* Make sidebar buttons full width */
            margin-bottom: 8px; /* Add some spacing */
        }
        [data-testid="stSidebar"] .stButton>button:hover, 
        [data-testid="stSidebar"] .stButton>button:focus {
            background-color: #EC4899; /* Hover for sidebar buttons */
        }
        .stMetric {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            font-size: 1.125rem;
        }
        .stExpander {
            background-color: #2D3B55;
            border-radius: 8px;
            margin: 16px 0;
            border: 1px solid #4B5EAA;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 10px;
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #D1D5DB;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #4F46E5;
            color: #F5F7FA;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #6B7280;
            color: #F5F7FA;
        }
        .plot-container { /* Applied to st.container(border=True) for plots */
            background-color: #222E45; /* Slightly different from main plot BG for distinction */
            border-radius: 8px;
            padding: 16px; /* Inner padding for the plot content */
            margin: 16px 0; /* Outer margin */
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .stPlotlyChart { /* Targets the plotly chart element itself */
            background-color: #2D3B55; /* Actual plot background */
            border-radius: 6px; /* Inner radius for the plot bg */
            padding: 8px; /* Minimal padding if needed around the plot itself */
        }
        .stDataFrame {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 16px;
            font-size: 0.875rem;
        }
        @media (max-width: 768px) {
            .main { padding: 16px; }
            h1 { font-size: 2rem; }
            h2 { font-size: 1.5rem; }
            h3 { font-size: 1.25rem; }
            .stColumn { width: 100% !important; margin-bottom: 1.5rem; }
            .stPlotlyChart { height: 360px !important; }
            .stTabs [data-baseweb="tab"] { padding: 10px 16px; font-size: 0.875rem; }
        }
        .spinner { display: flex; justify-content: center; align-items: center; height: 120px; }
        .spinner::after { content: ''; width: 48px; height: 48px; border: 5px solid #D1D5DB; border-top: 5px solid #4F46E5; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .onboarding-modal { background-color: #2D3B55; border-radius: 8px; padding: 24px; max-width: 600px; margin: 24px auto; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .onboarding-modal h3 { color: #F5F7FA; margin-bottom: 16px; }
        .onboarding-modal p { color: #D1D5DB; line-height: 1.6; margin-bottom: 16px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for settings
def render_settings_sidebar():
    with st.sidebar:
        st.markdown(
            f'<img src="{LEAN_LOGO_BASE64}" width="120" alt="Lean Institute Logo" aria-label="Lean Institute Logo" style="display: block; margin: 0 auto 16px;">',
            unsafe_allow_html=True
        )
        st.header("⚙️ Settings", divider="grey")

        with st.expander("🧪 Simulation", expanded=True):
            team_size = st.slider(
                "Team Size", 10, 100, st.session_state.get('team_size', DEFAULT_CONFIG['TEAM_SIZE']),
                key="team_size_slider"
            )
            shift_duration = st.slider(
                "Shift Duration (minutes)", 200, 2000, st.session_state.get('shift_duration', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']), 
                step=2, key="shift_duration_slider"
            )
            max_disruption_time = shift_duration -2 # Disruptions cannot happen at the very end
            disruption_options = [i * 2 for i in range(max_disruption_time // 2)] if max_disruption_time > 0 else []
            
            default_disrupt_mins_raw = [i * 2 for i in DEFAULT_CONFIG.get('DISRUPTION_INTERVALS', [])]
            # Filter default disruptions to be valid for current shift_duration
            valid_default_disrupt_mins = [m for m in default_disrupt_mins_raw if m in disruption_options]
            
            current_disrupt_selection = st.session_state.get('disruption_intervals_minutes', valid_default_disrupt_mins)
            # Further filter current selection based on current options (if shift_duration changed)
            valid_current_disrupt_selection = [m for m in current_disrupt_selection if m in disruption_options]

            disruption_intervals_minutes = st.multiselect(
                "Disruption Times (minutes)", disruption_options, valid_current_disrupt_selection,
                key="disruption_intervals_multiselect"
            )
            team_initiative_opts = ["More frequent breaks", "Team recognition"]
            current_initiative = st.session_state.get('team_initiative', team_initiative_opts[0])
            team_initiative_idx = team_initiative_opts.index(current_initiative) if current_initiative in team_initiative_opts else 0
            team_initiative = st.selectbox(
                "Team Initiative", team_initiative_opts, index=team_initiative_idx,
                key="team_initiative_selectbox"
            )
            run_simulation_button = st.button("Run Simulation", key="run_simulation_button", type="primary")

        with st.expander("🎨 Visualizations"):
            high_contrast = st.checkbox("High Contrast Mode", st.session_state.get('high_contrast', False), key="high_contrast_checkbox")
            use_3d_distribution = st.checkbox("3D Team Distribution", st.session_state.get('use_3d_distribution', False), key="use_3d_distribution_checkbox")
            debug_mode = st.checkbox("Debug Mode", st.session_state.get('debug_mode', False), key="debug_mode_checkbox")

        with st.expander("💾 Data"):
            load_data_button = st.button("Load Saved Data", key="load_data_button")
            can_gen_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("Download PDF Report", key="download_report_button", disabled=not can_gen_report):
                if can_gen_report:
                    try:
                        sim_res = st.session_state.simulation_results
                        num_steps = len(sim_res.get('downtime_minutes', []))
                        if num_steps == 0: st.warning("No data for PDF report."); raise SystemExit # Abort if no data
                        
                        pdf_data = {k: sim_res.get(k, [np.nan]*num_steps)[:num_steps] for k in [
                            'operational_recovery', 'psychological_safety', 'productivity_loss', 
                            'downtime_minutes', 'task_completion_rate']}
                        pdf_data.update({
                            'task_compliance': sim_res.get('task_compliance', {}).get('data', [np.nan]*num_steps)[:num_steps],
                            'collaboration_proximity': sim_res.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps)[:num_steps],
                            'worker_wellbeing': sim_res.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps)[:num_steps],
                            'step': list(range(num_steps)),
                            'time_minutes': [i * 2 for i in range(num_steps)]
                        })
                        generate_pdf_report(pd.DataFrame(pdf_data))
                        st.success("PDF report generation started (workplace_report.tex).")
                    except SystemExit: pass # Handled by st.warning
                    except Exception as e: logger.error(f"PDF Gen Error: {e}", exc_info=True); st.error(f"PDF Gen Error: {e}")
        
        with st.expander("📊 Export Options"):
            if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                st.info("Plot export (PNG) via camera icon on charts.", icon="ℹ️")
                sim_res_exp = st.session_state.simulation_results
                num_steps_csv = len(sim_res_exp.get('downtime_minutes', []))
                if num_steps_csv > 0:
                    csv_data = {k: sim_res_exp.get(k, [np.nan]*num_steps_csv)[:num_steps_csv] for k in [
                        'operational_recovery', 'psychological_safety', 'productivity_loss', 
                        'downtime_minutes', 'task_completion_rate']}
                    csv_data.update({
                        'task_compliance': sim_res_exp.get('task_compliance', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv],
                        'collaboration_proximity': sim_res_exp.get('collaboration_proximity', {}).get('data', [np.nan]*num_steps_csv)[:num_steps_csv],
                        'worker_wellbeing': sim_res_exp.get('worker_wellbeing', {}).get('scores', [np.nan]*num_steps_csv)[:num_steps_csv],
                        'step': list(range(num_steps_csv)),
                        'time_minutes': [i * 2 for i in range(num_steps_csv)]
                    })
                    st.download_button("Download Summary CSV", pd.DataFrame(csv_data).to_csv(index=False).encode('utf-8'),
                                       "workplace_summary.csv", "text/csv", key="download_csv_button")
                else: st.info("No data for CSV export.")
            else: st.info("Run simulation for export options.", icon="ℹ️")

        if st.session_state.get('debug_mode_checkbox', False):
            with st.expander("🛠️ Debug Info"):
                st.write("**Default Config (Partial):**")
                st.json({k: DEFAULT_CONFIG.get(k) for k in ['ENTRY_EXIT_POINTS', 'WORK_AREAS']}, expanded=False)
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Active Simulation Config:**")
                    st.json(st.session_state.simulation_results.get('config_params', {}), expanded=False)
                else: st.write("**No active simulation.**")

        st.header("📋 Navigation", divider="grey")
        # These buttons are for conceptual navigation; st.tabs manages its own state.
        # for i, tab_name in enumerate(["Overview", "Operational", "Worker", "Downtime", "Glossary"]):
        #     if st.button(tab_name, key=f"nav_{tab_name.lower()}_btn"): st.session_state.active_tab_idx_sidebar = i

        if st.button("ℹ️ Help", key="help_toggle_button"): st.session_state.show_help = not st.session_state.get('show_help', False); st.rerun()
        if st.button("🚀 Take a Tour", key="tour_toggle_button"): st.session_state.show_tour = not st.session_state.get('show_tour', False); st.rerun()

    return team_size, shift_duration, disruption_intervals_minutes, team_initiative, \
           run_simulation_button, load_data_button, high_contrast, use_3d_distribution, debug_mode


@st.cache_data(ttl=3600) # Cache simulation runs for an hour
def run_simulation_logic(team_size, shift_duration_minutes, disruption_intervals_minutes, team_initiative_selected):
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // 2
    config['DISRUPTION_INTERVALS'] = sorted(list(set([t // 2 for t in disruption_intervals_minutes])))

    if 'WORK_AREAS' in config and isinstance(config['WORK_AREAS'], dict) and config['WORK_AREAS']:
        total_current_workers = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())
        if total_current_workers == 0 and team_size > 0: # Initial distribution if all zones are 0
            num_zones = len(config['WORK_AREAS'])
            if num_zones > 0:
                workers_per_zone = team_size // num_zones
                rem = team_size % num_zones
                zone_keys = list(config['WORK_AREAS'].keys())
                for i, zk in enumerate(zone_keys): config['WORK_AREAS'][zk]['workers'] = workers_per_zone + (1 if i < rem else 0)
        elif total_current_workers > 0 and total_current_workers != team_size: # Rescale if sum is non-zero but not matching team_size
            ratio = team_size / total_current_workers
            for zk in config['WORK_AREAS']: config['WORK_AREAS'][zk]['workers'] = int(config['WORK_AREAS'][zk].get('workers', 0) * ratio)
            current_sum_after_ratio = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())
            diff = team_size - current_sum_after_ratio # Distribute remainder
            if diff != 0:
                zk_add = 'Assembly Line' if 'Assembly Line' in config['WORK_AREAS'] else (next(iter(config['WORK_AREAS'])) if config['WORK_AREAS'] else None)
                if zk_add: config['WORK_AREAS'][zk_add]['workers'] = config['WORK_AREAS'][zk_add].get('workers', 0) + diff
        elif team_size == 0: # If target is 0, set all to 0
            for zk in config['WORK_AREAS']: config['WORK_AREAS'][zk]['workers'] = 0
    
    validate_config(config)
    logger.info(f"Running sim: Size={team_size}, Duration={shift_duration_minutes}m, Disruptions(m): {disruption_intervals_minutes}, Initiative: {team_initiative_selected}",
                extra={'user_action': 'Run Simulation'})

    # CRITICAL ASSUMPTION: simulate_workplace_operations returns a TUPLE of 11 elements in a specific order.
    # This order MUST match how data is structured and accessed later.
    sim_results_tuple = simulate_workplace_operations(
        num_team_members=team_size,
        num_steps=config['SHIFT_DURATION_INTERVALS'],
        disruption_intervals=config['DISRUPTION_INTERVALS'],
        team_initiative=team_initiative_selected,
        config=config
    )

    # Define the expected keys for the dictionary, matching the tuple order.
    # This list MUST match the order of elements returned by simulate_workplace_operations
    # and the keys used throughout the dashboard.
    expected_keys = [
        'team_positions_df', 'task_compliance', 'collaboration_proximity', 
        'operational_recovery', 'efficiency_metrics_df', 'productivity_loss', 
        'worker_wellbeing', 'psychological_safety', 'feedback_impact', 
        'downtime_minutes', 'task_completion_rate'
    ]
    
    if not isinstance(sim_results_tuple, tuple) or len(sim_results_tuple) != len(expected_keys):
        err_msg = (f"Simulation returned unexpected data format. "
                   f"Expected a tuple of {len(expected_keys)} items, "
                   f"got {type(sim_results_tuple)} with {len(sim_results_tuple) if isinstance(sim_results_tuple, (list,tuple)) else 'N/A'} items.")
        logger.error(err_msg, extra={'user_action': 'Run Simulation - Data Format Error'})
        raise TypeError(err_msg)

    # Construct the dictionary from the tuple
    simulation_output_dict = dict(zip(expected_keys, sim_results_tuple))

    # Add config_params to this new dictionary
    simulation_output_dict['config_params'] = {
        'TEAM_SIZE': team_size,
        'SHIFT_DURATION_MINUTES': shift_duration_minutes,
        'DISRUPTION_INTERVALS_MINUTES': disruption_intervals_minutes,
        'DISRUPTION_INTERVALS_STEPS': config['DISRUPTION_INTERVALS'],
        'TEAM_INITIATIVE': team_initiative_selected
    }
    
    save_simulation_data(simulation_output_dict) # save_simulation_data should handle a dict
    return simulation_output_dict


def main():
    st.title("Workplace Shift Monitoring Dashboard")

    # Initialize session state keys if they don't exist
    for key in ['team_size', 'shift_duration', 'disruption_intervals_minutes', 'team_initiative',
                'high_contrast', 'use_3d_distribution', 'debug_mode',
                'simulation_results', 'show_tour', 'show_help']:
        if key not in st.session_state:
            st.session_state[key] = None # Will be populated by defaults or user actions

    # Call sidebar and get current user inputs/button states
    sb_team_size, sb_shift_duration, sb_disrupt_mins, sb_team_initiative, \
    sb_run_sim_btn, sb_load_data_btn, sb_high_contrast, \
    sb_use_3d, sb_debug_mode = render_settings_sidebar()

    # Update session state from sidebar inputs (makes them sticky)
    st.session_state.team_size = sb_team_size
    st.session_state.shift_duration = sb_shift_duration
    st.session_state.disruption_intervals_minutes = sb_disrupt_mins
    st.session_state.team_initiative = sb_team_initiative
    st.session_state.high_contrast = sb_high_contrast
    st.session_state.use_3d_distribution = sb_use_3d
    st.session_state.debug_mode = sb_debug_mode
    
    # Determine max minutes for sliders based on current data or settings
    current_max_minutes_for_sliders = (st.session_state.get('shift_duration', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) or DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) - 2
    disruption_steps_for_plots = []

    if st.session_state.simulation_results and isinstance(st.session_state.simulation_results, dict):
        num_steps = len(st.session_state.simulation_results.get('downtime_minutes', []))
        if num_steps > 0:
            current_max_minutes_for_sliders = (num_steps - 1) * 2
        disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', [])
    else: # No sim results, use sidebar settings for disruption steps for plots
        disruption_steps_for_plots = [m // 2 for m in st.session_state.get('disruption_intervals_minutes', [])]
    current_max_minutes_for_sliders = max(0, current_max_minutes_for_sliders)


    if sb_run_sim_btn:
        with st.spinner("Running simulation..."):
            try:
                st.session_state.simulation_results = run_simulation_logic(
                    st.session_state.team_size, st.session_state.shift_duration,
                    st.session_state.disruption_intervals_minutes, st.session_state.team_initiative
                )
                st.success("Simulation completed!")
                st.rerun() # Rerun to update UI based on new results (e.g., slider ranges)
            except Exception as e:
                logger.error(f"Simulation Run Error: {e}", exc_info=True)
                st.error(f"Simulation failed: {e}")
                st.session_state.simulation_results = None

    if sb_load_data_btn:
        with st.spinner("Loading saved data..."):
            try:
                loaded_data = load_simulation_data()
                if loaded_data and isinstance(loaded_data, dict):
                    st.session_state.simulation_results = loaded_data
                    # Update sidebar controls from loaded config
                    cfg = loaded_data.get('config_params', {})
                    for k, v_default in [('team_size', DEFAULT_CONFIG['TEAM_SIZE']), 
                                         ('shift_duration', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']),
                                         ('team_initiative', "More frequent breaks")]: # Default for initiative
                        st.session_state[k] = cfg.get(k.upper(), st.session_state.get(k, v_default)) # k.upper for cfg keys
                    st.session_state.disruption_intervals_minutes = cfg.get('DISRUPTION_INTERVALS_MINUTES', [])
                    st.success("Data loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load data or data is not in expected dictionary format.")
            except Exception as e:
                logger.error(f"Load Data Error: {e}", exc_info=True)
                st.error(f"Failed to load data: {e}")
                st.session_state.simulation_results = None
    
    # Modals
    if st.session_state.get('show_tour'):
        # ... (modal content as before, ensure unique keys for buttons)
        with st.container(): st.markdown('<div class="onboarding-modal"><h3>Welcome Tour</h3>...</div>', unsafe_allow_html=True)
        if st.button("End Tour", key="end_tour_btn"): st.session_state.show_tour = False; st.rerun()
    if st.session_state.get('show_help'):
        # ... (modal content as before, ensure unique keys for buttons)
        with st.container(): st.markdown('<div class="onboarding-modal"><h3>Help</h3>...</div>', unsafe_allow_html=True)
        if st.button("Close Help", key="close_help_btn_modal"): st.session_state.show_help = False; st.rerun()


    tabs_main_names = ["Overview", "Operational Metrics", "Worker Insights", "Downtime", "Glossary"]
    tabs = st.tabs(tabs_main_names)

    # Helper for safe data access and mean/sum calculation
    def safe_get(data_dict, path, default_val=0.0):
        current = data_dict
        for key in path.split('.'):
            if isinstance(current, dict): current = current.get(key)
            elif isinstance(current, (list, pd.Series)) and key.isdigit() and int(key) < len(current): current = current[int(key)] # Basic list indexing
            else: return default_val if not isinstance(default_val, list) else [] # Return empty list if default is list
        return current if current is not None else (default_val if not isinstance(default_val, list) else [])

    def safe_stat(data_list, stat_func, default_val=0.0):
        if not isinstance(data_list, (list, np.ndarray, pd.Series)): data_list = []
        valid_data = [x for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return stat_func(valid_data) if valid_data else default_val

    # --- Overview Tab ---
    with tabs[0]:
        st.header("Overview", divider="grey")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results
            compliance = safe_stat(safe_get(sim_data, 'task_compliance.data', []), np.mean)
            proximity = safe_stat(safe_get(sim_data, 'collaboration_proximity.data', []), np.mean)
            wellbeing = safe_stat(safe_get(sim_data, 'worker_wellbeing.scores', []), np.mean)
            downtime = safe_stat(safe_get(sim_data, 'downtime_minutes', []), np.sum)
            
            cols_metrics = st.columns(4)
            cols_metrics[0].metric("Task Compliance", f"{compliance:.1f}%", f"{compliance-75:.1f}%")
            cols_metrics[1].metric("Collaboration", f"{proximity:.1f}%", f"{proximity-60:.1f}%")
            cols_metrics[2].metric("Well-Being", f"{wellbeing:.1f}%", f"{wellbeing-70:.1f}%")
            cols_metrics[3].metric("Downtime", f"{downtime:.1f} min", f"{downtime-30:.1f} min", delta_color="inverse")

            try:
                summary_figs = plot_key_metrics_summary(compliance, proximity, wellbeing, downtime)
                cols_gauges = st.columns(min(len(summary_figs), 2) or 1) # Ensure at least 1 column
                for i, fig in enumerate(summary_figs):
                    with cols_gauges[i % len(cols_gauges)]:
                        with st.container(border=True): 
                            st.markdown('<div class="plot-container">', unsafe_allow_html=True) # No title for gauges
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e: logger.error(f"Overview Gauges Error: {e}", exc_info=True); st.error("Error rendering overview gauges.")

            with st.expander("View Detailed Overview Data"):
                num_s = len(safe_get(sim_data, 'downtime_minutes', []))
                if num_s > 0:
                    df_data = {'Time (min)': [i*2 for i in range(num_s)]}
                    df_data.update({
                        'Task Compliance (%)': safe_get(sim_data, 'task_compliance.data', [np.nan]*num_s)[:num_s],
                        'Collaboration (%)': safe_get(sim_data, 'collaboration_proximity.data', [np.nan]*num_s)[:num_s],
                        'Well-Being (%)': safe_get(sim_data, 'worker_wellbeing.scores', [np.nan]*num_s)[:num_s],
                        'Downtime (min)': safe_get(sim_data, 'downtime_minutes', [np.nan]*num_s)[:num_s]
                    })
                    st.dataframe(pd.DataFrame(df_data), use_container_width=True, height=300)
                else: st.info("No detailed overview data.")
        else: st.info("Run simulation or load data for Overview.", icon="ℹ️")

    # --- Operational Metrics Tab ---
    with tabs[1]:
        st.header("Operational Metrics", divider="grey")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results
            op_time_val = st.session_state.get("op_time_slider_val", (0, current_max_minutes_for_sliders))
            op_time_val = (min(op_time_val[0], current_max_minutes_for_sliders), min(op_time_val[1], current_max_minutes_for_sliders))
            
            time_range = st.slider("Time Range (minutes)", 0, current_max_minutes_for_sliders, op_time_val, 2, 
                                   key="op_time_slider", disabled=current_max_minutes_for_sliders == 0,
                                   on_change=lambda: st.session_state.update(op_time_slider_val=st.session_state.op_time_slider))
            start_idx, end_idx = time_range[0]//2, time_range[1]//2 + 1
            filt_disrupt_steps = [s for s in disruption_steps_for_plots if start_idx <= s < end_idx]

            # Task Compliance
            tc_data = safe_get(sim_data, 'task_compliance.data', [])[start_idx:end_idx]
            if tc_data:
                with st.container(border=True):
                    st.markdown('<div class="plot-container"><h5>Task Compliance Score</h5>', unsafe_allow_html=True)
                    try:
                        tc_z = safe_get(sim_data, 'task_compliance.z_scores', [])[start_idx:end_idx]
                        tc_f = safe_get(sim_data, 'task_compliance.forecast', [])[start_idx:end_idx] if safe_get(sim_data, 'task_compliance.forecast', []) else None
                        st.plotly_chart(plot_task_compliance_score(tc_data, filt_disrupt_steps, tc_f, tc_z), use_container_width=True)
                    except Exception as e: logger.error(f"Op Tab TC Plot Error: {e}", exc_info=True); st.error("Error: Task Compliance plot.")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Collaboration
            cp_data = safe_get(sim_data, 'collaboration_proximity.data', [])[start_idx:end_idx]
            if cp_data:
                with st.container(border=True):
                    st.markdown('<div class="plot-container"><h5>Collaboration Proximity Index</h5>', unsafe_allow_html=True)
                    try:
                        cp_f = safe_get(sim_data, 'collaboration_proximity.forecast', [])[start_idx:end_idx] if safe_get(sim_data, 'collaboration_proximity.forecast', []) else None
                        st.plotly_chart(plot_collaboration_proximity_index(cp_data, filt_disrupt_steps, cp_f), use_container_width=True)
                    except Exception as e: logger.error(f"Op Tab CP Plot Error: {e}", exc_info=True); st.error("Error: Collaboration plot.")
                    st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Additional Operational Metrics"):
                # Recovery & Resilience
                or_data = safe_get(sim_data, 'operational_recovery', [])[start_idx:end_idx]
                if or_data:
                    with st.container(border=True):
                        st.markdown('<div class="plot-container"><h5>Operational Recovery & Resilience</h5>', unsafe_allow_html=True)
                        try:
                            pl_data = safe_get(sim_data, 'productivity_loss', [])[start_idx:end_idx]
                            st.plotly_chart(plot_operational_recovery(or_data, pl_data), use_container_width=True)
                        except Exception as e: logger.error(f"Op Tab OR Plot Error: {e}", exc_info=True); st.error("Error: Recovery plot.")
                        st.markdown('</div>', unsafe_allow_html=True)
                # OEE
                eff_df = safe_get(sim_data, 'efficiency_metrics_df', pd.DataFrame())
                if not eff_df.empty:
                    with st.container(border=True):
                        st.markdown('<div class="plot-container"><h5>Operational Efficiency (OEE)</h5>', unsafe_allow_html=True)
                        try:
                            sel_metrics = st.multiselect("Select Efficiency Metrics", ['uptime', 'throughput', 'quality', 'oee'], 
                                                        default=['uptime', 'throughput', 'quality', 'oee'], key="eff_metrics_ms_op")
                            filt_eff_df = eff_df.iloc[start_idx:end_idx] if isinstance(eff_df.index, pd.RangeIndex) and end_idx <= len(eff_df) else eff_df
                            if not filt_eff_df.empty:
                                st.plotly_chart(plot_operational_efficiency(filt_eff_df, sel_metrics), use_container_width=True)
                            else: st.info("No OEE data for this range.")
                        except Exception as e: logger.error(f"Op Tab OEE Plot Error: {e}", exc_info=True); st.error("Error: OEE plot.")
                        st.markdown('</div>', unsafe_allow_html=True)
        else: st.info("Run simulation or load data for Operational Metrics.", icon="ℹ️")

    # --- Worker Insights Tab ---
    with tabs[2]:
        st.header("Worker Insights", divider="grey")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results
            team_pos_df_all = safe_get(sim_data, 'team_positions_df', pd.DataFrame())
            
            worker_time_val = st.session_state.get("worker_time_slider_val", (0, current_max_minutes_for_sliders))
            worker_time_val = (min(worker_time_val[0], current_max_minutes_for_sliders), min(worker_time_val[1], current_max_minutes_for_sliders))

            with st.expander("Worker Distribution Analysis", expanded=True):
                dist_time_range = st.slider("Time Range (minutes) for Distribution", 0, current_max_minutes_for_sliders, worker_time_val, 2,
                                            key="worker_dist_time_slider", disabled=current_max_minutes_for_sliders == 0,
                                            on_change=lambda: st.session_state.update(worker_time_slider_val=st.session_state.worker_dist_time_slider))
                dist_start_idx, dist_end_idx = dist_time_range[0]//2, dist_time_range[1]//2 + 1
                
                zones = ["All"] + list(DEFAULT_CONFIG.get('WORK_AREAS', {}).keys())
                zone_sel = st.selectbox("Filter by Zone", zones, key="worker_zone_sel")
                
                filt_team_pos_df = team_pos_df_all
                if not filt_team_pos_df.empty:
                    filt_team_pos_df = filt_team_pos_df[(filt_team_pos_df['step'] >= dist_start_idx) & (filt_team_pos_df['step'] < dist_end_idx)]
                    if zone_sel != "All": filt_team_pos_df = filt_team_pos_df[filt_team_pos_df['zone'] == zone_sel]

                show_ee = st.checkbox("Show Entry/Exit", True, key="worker_show_ee")
                show_pl = st.checkbox("Show Prod. Lines", True, key="worker_show_pl")

                cols_dist = st.columns(2)
                with cols_dist[0]:
                    st.markdown("<h5>Worker Positions (Snapshot)</h5>", unsafe_allow_html=True)
                    min_step, max_step = dist_start_idx, max(dist_start_idx, dist_end_idx -1)
                    snap_step_val = st.session_state.get("worker_snap_step_slider_val", min_step)
                    snap_step_val = max(min_step, min(snap_step_val, max_step)) # Ensure in bounds

                    snap_step = st.slider("Snapshot Time Step", min_step, max_step, snap_step_val, 1, 
                                          key="worker_snap_step_slider", disabled=max_step < min_step,
                                          on_change=lambda: st.session_state.update(worker_snap_step_slider_val=st.session_state.worker_snap_step_slider))
                    if not team_pos_df_all.empty and max_step >= min_step:
                        with st.container(border=True):
                            try: st.plotly_chart(plot_worker_distribution(team_pos_df_all, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, 
                                                                        st.session_state.get('use_3d_distribution', False), snap_step, show_ee, show_pl), use_container_width=True)
                            except Exception as e: logger.error(f"Worker Dist Plot Error: {e}", exc_info=True); st.error("Error: Worker Positions plot.")
                    else: st.info("No data for positions snapshot.")
                with cols_dist[1]:
                    st.markdown("<h5>Worker Density Heatmap</h5>", unsafe_allow_html=True)
                    if not filt_team_pos_df.empty:
                        with st.container(border=True):
                            try: st.plotly_chart(plot_worker_density_heatmap(filt_team_pos_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, show_ee, show_pl), use_container_width=True)
                            except Exception as e: logger.error(f"Worker Heatmap Plot Error: {e}", exc_info=True); st.error("Error: Density Heatmap plot.")
                    else: st.info("No data for density heatmap.")
            
            with st.expander("Worker Well-Being & Safety Analysis", expanded=True):
                well_time_range = st.slider("Time Range (minutes) for Well-Being/Safety", 0, current_max_minutes_for_sliders, worker_time_val, 2,
                                            key="worker_well_time_slider", disabled=current_max_minutes_for_sliders == 0,
                                            on_change=lambda: st.session_state.update(worker_time_slider_val=st.session_state.worker_well_time_slider)) # reuse worker_time_slider_val
                well_start_idx, well_end_idx = well_time_range[0]//2, well_time_range[1]//2 + 1
                
                cols_well = st.columns(2)
                with cols_well[0]: # Well-Being Index
                    st.markdown("<h5>Worker Well-Being Index</h5>", unsafe_allow_html=True)
                    ww_scores = safe_get(sim_data, 'worker_wellbeing.scores', [])[well_start_idx:well_end_idx]
                    if ww_scores:
                        with st.container(border=True):
                            try:
                                ww_trigs_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {})
                                ww_trigs_filt = {
                                    k: [t for t in v if well_start_idx <= t < well_end_idx] 
                                    for k, v in ww_trigs_raw.items() if isinstance(v, list)
                                }
                                ww_trigs_filt['work_area'] = {
                                    wk: [t for t in wv if well_start_idx <= t < well_end_idx]
                                    for wk, wv in ww_trigs_raw.get('work_area', {}).items()
                                }
                                st.plotly_chart(plot_worker_wellbeing(ww_scores, ww_trigs_filt), use_container_width=True)
                            except Exception as e: logger.error(f"Wellbeing Plot Error: {e}", exc_info=True); st.error("Error: Well-Being Index plot.")
                    else: st.info("No Well-Being scores for this range.")
                with cols_well[1]: # Psychological Safety
                    st.markdown("<h5>Psychological Safety Score</h5>", unsafe_allow_html=True)
                    ps_scores = safe_get(sim_data, 'psychological_safety', [])[well_start_idx:well_end_idx]
                    if ps_scores:
                        with st.container(border=True):
                            try: st.plotly_chart(plot_psychological_safety(ps_scores), use_container_width=True)
                            except Exception as e: logger.error(f"Psych Safety Plot Error: {e}", exc_info=True); st.error("Error: Psych. Safety plot.")
                    else: st.info("No Psych. Safety scores for this range.")

                st.markdown("<h6>Well-Being Triggers (within selected time range)</h6>", unsafe_allow_html=True)
                ww_trigs_disp_raw = safe_get(sim_data, 'worker_wellbeing.triggers', {})
                ww_trigs_disp_filt = {
                    k: [t for t in v if well_start_idx <= t < well_end_idx] 
                    for k, v in ww_trigs_disp_raw.items() if isinstance(v, list)
                }
                ww_trigs_disp_filt['work_area'] = {
                    wk: [t for t in wv if well_start_idx <= t < well_end_idx]
                    for wk, wv in ww_trigs_disp_raw.get('work_area', {}).items()
                }
                st.write(f"**Threshold Alerts:** {ww_trigs_disp_filt.get('threshold', 'N/A')}")
                st.write(f"**Trend Alerts:** {ww_trigs_disp_filt.get('trend', 'N/A')}")
                st.write(f"**Disruption Alerts:** {ww_trigs_disp_filt.get('disruption', 'N/A')}")
                st.write("**Work Area Alerts:**")
                wa_alert_found = False
                for zone, trigs in ww_trigs_disp_filt.get('work_area', {}).items():
                    if trigs: st.write(f"  {zone}: {trigs}"); wa_alert_found = True
                if not wa_alert_found: st.write("  None")
        else: st.info("Run simulation or load data for Worker Insights.", icon="ℹ️")

    # --- Downtime Tab ---
    with tabs[3]:
        st.header("Downtime Analysis", divider="grey")
        if st.session_state.simulation_results:
            sim_data = st.session_state.simulation_results
            dt_time_val = st.session_state.get("dt_time_slider_val", (0, current_max_minutes_for_sliders))
            dt_time_val = (min(dt_time_val[0], current_max_minutes_for_sliders), min(dt_time_val[1], current_max_minutes_for_sliders))

            time_range_dt = st.slider("Time Range (minutes)", 0, current_max_minutes_for_sliders, dt_time_val, 2,
                                      key="dt_time_slider", disabled=current_max_minutes_for_sliders == 0,
                                      on_change=lambda: st.session_state.update(dt_time_slider_val=st.session_state.dt_time_slider))
            dt_start_idx, dt_end_idx = time_range_dt[0]//2, time_range_dt[1]//2 + 1
            
            dt_data = safe_get(sim_data, 'downtime_minutes', [])[dt_start_idx:dt_end_idx]
            if dt_data:
                with st.container(border=True):
                    st.markdown('<div class="plot-container"><h5>Downtime Trend</h5>', unsafe_allow_html=True)
                    try: st.plotly_chart(plot_downtime_trend(dt_data, DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD', 10)), use_container_width=True)
                    except Exception as e: logger.error(f"Downtime Plot Error: {e}", exc_info=True); st.error("Error: Downtime Trend plot.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else: st.info("No Downtime data for this range.")
        else: st.info("Run simulation or load data for Downtime Analysis.", icon="ℹ️")
        
    # --- Glossary Tab ---
    with tabs[4]:
        st.header("Glossary", divider="grey")
        st.markdown("""
            <h5>Metric Definitions</h5>
            <ul>
                <li><b>Task Compliance Score</b>: Percentage of tasks completed correctly and on time (0–100%). Measures adherence to operational protocols.</li>
                <li><b>Collaboration Proximity Index</b>: Percentage of workers within 5 meters of colleagues (0–100%). Indicates teamwork and communication opportunities.</li>
                <li><b>Operational Recovery Score</b>: Ability to maintain output after disruptions (0–100%). Reflects resilience to unexpected events.</li>
                <li><b>Worker Well-Being Index</b>: Composite score of fatigue, stress, and satisfaction (0–100%). Tracks worker health and morale.</li>
                <li><b>Psychological Safety Score</b>: Comfort level in reporting issues or suggesting improvements (0–100%). Indicates a supportive work environment.</li>
                <li><b>Uptime</b>: Percentage of time equipment is operational (0–100%). Measures equipment reliability.</li>
                <li><b>Throughput</b>: Percentage of maximum production rate achieved (0–100%). Indicates production efficiency.</li>
                <li><b>Quality</b>: Percentage of products meeting quality standards (0–100%). Reflects output consistency.</li>
                <li><b>OEE (Overall Equipment Effectiveness)</b>: Combined score of uptime, throughput, and quality (0–100%). Holistic measure of operational performance.</li>
                <li><b>Productivity Loss</b>: Percentage of potential output lost due to inefficiencies or disruptions (0–100%).</li>
                <li><b>Downtime</b>: Total minutes of unplanned operational stops per interval. Tracks interruptions to workflow.</li>
                <li><b>Task Completion Rate</b>: Percentage of tasks completed per time interval (0–100%). Measures task efficiency over time.</li>
            </ul>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
