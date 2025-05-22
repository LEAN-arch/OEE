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
            background-color: #10B981;
        }
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus {
            background-color: #EC4899;
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
        .recommendation {
            color: #FBBF24;
            font-size: 0.875rem;
            margin-top: 12px;
            font-style: italic;
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
        .stPlotlyChart {
            margin: 16px 0;
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .summary-card {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
            display: flex;
            flex-direction: row;
            align-items: center;
            gap: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
        }
        .summary-card:hover {
            transform: translateY(-4px);
        }
        .summary-card h4 {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0;
        }
        .summary-card p {
            font-size: 1.75rem;
            font-weight: 700;
            color: #FBBF24;
            margin: 0;
        }
        .plot-container {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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
            .stColumn {
                width: 100% !important;
                margin-bottom: 1.5rem;
            }
            .stPlotlyChart {
                height: 360px !important;
            }
            .summary-card {
                flex-direction: column;
                text-align: center;
                gap: 12px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 10px 16px;
                font-size: 0.875rem;
            }
        }
        .spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 120px;
        }
        .spinner::after {
            content: '';
            width: 48px;
            height: 48px;
            border: 5px solid #D1D5DB;
            border-top: 5px solid #4F46E5;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .onboarding-modal {
            background-color: #2D3B55;
            border-radius: 8px;
            padding: 24px;
            max-width: 600px;
            margin: 24px auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .onboarding-modal h3 {
            color: #F5F7FA;
            margin-bottom: 16px;
        }
        .onboarding-modal p {
            color: #D1D5DB;
            line-height: 1.6;
            margin-bottom: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to display a loading spinner
def display_loading(message):
    with st.container():
        st.markdown(f'<div class="spinner"></div><p style="text-align: center; color: #F5F7FA;">{message}</p>', unsafe_allow_html=True)

# Sidebar for settings
def render_settings_sidebar():
    with st.sidebar:
        st.markdown(
            f'<img src="{LEAN_LOGO_BASE64}" width="120" alt="Lean 2.0 Institute Logo" aria-label="Lean 2.0 Institute Logo" style="display: block; margin: 0 auto 16px;">',
            unsafe_allow_html=True
        )
        st.header("⚙️ Settings", divider="grey")

        # Simulation Controls
        with st.expander("🧪 Simulation", expanded=True):
            team_size = st.slider(
                "Team Size",
                min_value=10, max_value=100, value=st.session_state.get('team_size', DEFAULT_CONFIG['TEAM_SIZE']),
                help="Number of workers in the simulation.",
                key="team_size_slider" # Use unique key
            )
            shift_duration = st.slider(
                "Shift Duration (minutes)",
                min_value=200, max_value=2000, value=st.session_state.get('shift_duration', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']), step=2,
                help="Shift duration in minutes (2-minute intervals).",
                key="shift_duration_slider" # Use unique key
            )
            
            max_step_for_disruption = shift_duration // 2
            default_disruption_steps = DEFAULT_CONFIG.get('DISRUPTION_INTERVALS', [])
            valid_default_disruption_steps = [s for s in default_disruption_steps if s < max_step_for_disruption]
            default_disruption_minutes = [i * 2 for i in valid_default_disruption_steps]
            
            current_disruption_minutes = st.session_state.get('disruption_intervals_minutes', default_disruption_minutes)
            # Filter current_disruption_minutes to ensure they are valid options
            valid_current_disruption_minutes = [m for m in current_disruption_minutes if m < shift_duration]


            disruption_intervals_minutes = st.multiselect(
                "Disruption Times (minutes)",
                options=[i * 2 for i in range(max_step_for_disruption)],
                default=valid_current_disruption_minutes,
                help="Times when disruptions occur.",
                key="disruption_intervals_multiselect" # Use unique key
            )
            team_initiative_options = ["More frequent breaks", "Team recognition"]
            team_initiative_default_index = team_initiative_options.index(st.session_state.get('team_initiative', team_initiative_options[0]))

            team_initiative = st.selectbox(
                "Team Initiative",
                options=team_initiative_options,
                index=team_initiative_default_index,
                help="Strategy to improve well-being and safety.",
                key="team_initiative_selectbox" # Use unique key
            )
            run_simulation_button = st.button(
                "Run Simulation", 
                key="run_simulation_button", 
                help="Start a new simulation.",
                type="primary"
            )

        # Visualization Settings
        with st.expander("🎨 Visualizations"):
            high_contrast = st.checkbox(
                "High Contrast Mode", 
                value=st.session_state.get('high_contrast', False),
                help="Enable high-contrast colors for accessibility.",
                key="high_contrast_checkbox" # Use unique key
            )
            use_3d_distribution = st.checkbox(
                "3D Team Distribution", 
                value=st.session_state.get('use_3d_distribution', False),
                help="Use 3D scatter plot for team distribution.",
                key="use_3d_distribution_checkbox" # Use unique key
            )
            debug_mode = st.checkbox(
                "Debug Mode", 
                value=st.session_state.get('debug_mode', False),
                help="Show configuration data for debugging.",
                key="debug_mode_checkbox" # Use unique key
            )

        # Data Management
        with st.expander("💾 Data"):
            load_data_button = st.button(
                "Load Saved Data", 
                key="load_data_button", 
                help="Load previously saved simulation data."
            )
            
            can_generate_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("Download PDF Report", key="download_report_button", help="Generate a PDF report.", disabled=not can_generate_report):
                if can_generate_report: 
                    try:
                        sim_results = st.session_state.simulation_results
                        num_sim_steps = len(sim_results.get('downtime_minutes', [])) 
                        if num_sim_steps == 0:
                            st.warning("Cannot generate report, simulation data is empty for key metrics.")
                        else:
                            summary_data = {
                                'step': list(range(num_sim_steps)),
                                'time_minutes': [i * 2 for i in range(num_sim_steps)],
                                'task_compliance': sim_results.get('task_compliance', {}).get('data', [np.nan]*num_sim_steps)[:num_sim_steps],
                                'collaboration_proximity': sim_results.get('collaboration_proximity', {}).get('data', [np.nan]*num_sim_steps)[:num_sim_steps],
                                'operational_recovery': sim_results.get('operational_recovery', [np.nan]*num_sim_steps)[:num_sim_steps],
                                'worker_wellbeing': sim_results.get('worker_wellbeing', {}).get('scores', [np.nan]*num_sim_steps)[:num_sim_steps],
                                'psychological_safety': sim_results.get('psychological_safety', [np.nan]*num_sim_steps)[:num_sim_steps],
                                'productivity_loss': sim_results.get('productivity_loss', [np.nan]*num_sim_steps)[:num_sim_steps],
                                'downtime_minutes': sim_results.get('downtime_minutes', [np.nan]*num_sim_steps)[:num_sim_steps],
                                'task_completion_rate': sim_results.get('task_completion_rate', [np.nan]*num_sim_steps)[:num_sim_steps]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            generate_pdf_report(summary_df) 
                            st.success("PDF report generation started (workplace_report.tex). Compile with LaTeX to view.")
                    except Exception as e:
                        logger.error(f"Failed to generate report: {str(e)}", extra={'user_action': 'Download PDF Report'})
                        st.error(f"Failed to generate report: {str(e)}.")
                else:
                     st.info("Run a simulation or load data to enable PDF report generation.", icon="ℹ️")


        # Export Options
        with st.expander("📊 Export Options"):
            if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                if st.button("Export Plots as PNG (Info)", key="export_png_info_button"): # Changed key
                    st.info("Exporting plots as PNG is handled via the camera icon on each plot's display options bar.")
                
                sim_results_export = st.session_state.simulation_results
                num_sim_steps_csv = len(sim_results_export.get('downtime_minutes', [])) 
                if num_sim_steps_csv > 0:
                    summary_data_csv = {
                        'step': list(range(num_sim_steps_csv)),
                        'time_minutes': [i * 2 for i in range(num_sim_steps_csv)],
                        'task_compliance': sim_results_export.get('task_compliance', {}).get('data', [np.nan]*num_sim_steps_csv)[:num_sim_steps_csv],
                        'collaboration_proximity': sim_results_export.get('collaboration_proximity', {}).get('data', [np.nan]*num_sim_steps_csv)[:num_sim_steps_csv],
                        'operational_recovery': sim_results_export.get('operational_recovery', [np.nan]*num_sim_steps_csv)[:num_sim_steps_csv],
                        'worker_wellbeing': sim_results_export.get('worker_wellbeing', {}).get('scores', [np.nan]*num_sim_steps_csv)[:num_sim_steps_csv],
                        'psychological_safety': sim_results_export.get('psychological_safety', [np.nan]*num_sim_steps_csv)[:num_sim_steps_csv],
                        'productivity_loss': sim_results_export.get('productivity_loss', [np.nan]*num_sim_steps_csv)[:num_sim_steps_csv],
                        'downtime_minutes': sim_results_export.get('downtime_minutes', [np.nan]*num_sim_steps_csv)[:num_sim_steps_csv],
                        'task_completion_rate': sim_results_export.get('task_completion_rate', [np.nan]*num_sim_steps_csv)[:num_sim_steps_csv]
                    }
                    summary_df_csv = pd.DataFrame(summary_data_csv)
                    st.download_button(
                        label="Download Summary CSV",
                        data=summary_df_csv.to_csv(index=False).encode('utf-8'),
                        file_name="workplace_summary.csv",
                        mime="text/csv",
                        key="download_csv_button" # Use unique key
                    )
                else:
                    st.info("No data to export for CSV (simulation results are empty).")
            else:
                st.info("Run a simulation to enable export options.", icon="ℹ️")

        # Debug Information
        if st.session_state.get('debug_mode_checkbox', False): 
            with st.expander("🛠️ Debug Info"):
                st.write("**Entry/Exit Points (Default Config):**")
                st.write(DEFAULT_CONFIG.get('ENTRY_EXIT_POINTS', "Not defined"))
                st.write("**Work Areas (Default Config):**")
                st.write(DEFAULT_CONFIG.get('WORK_AREAS', "Not defined"))
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Current Simulation Config (from results):**")
                    st.json(st.session_state.simulation_results.get('config_params', {}))
                else:
                    st.write("**No active simulation results to show config from.**")


        # Navigation and Help
        st.header("📋 Navigation", divider="grey")
        tab_names_nav = ["Overview", "Operational Metrics", "Worker Insights", "Downtime", "Glossary"]
        for i, tab_name in enumerate(tab_names_nav):
            if st.button(tab_name, key=f"nav_{tab_name.lower().replace(' ', '_')}_button", help=f"Go to {tab_name}"): # Unique keys
                st.session_state.active_tab_index_from_sidebar = i 
                # Note: This doesn't directly control st.tabs, which has its own state.
                # It's more of a conceptual navigation hint or for other logic.

        if st.button("ℹ️ Help", key="help_button"): # Keep this key if it's fine
            st.session_state.show_help = not st.session_state.get('show_help', False)


        if st.button("🚀 Take a Tour", key="tour_button"): # Keep this key
            st.session_state.show_tour = not st.session_state.get('show_tour', False)


    return team_size, shift_duration, disruption_intervals_minutes, team_initiative, \
           run_simulation_button, load_data_button, high_contrast, use_3d_distribution, debug_mode

# Simulation logic with caching
@st.cache_data # type: ignore
def run_simulation_logic(team_size, shift_duration_minutes, disruption_intervals_minutes, team_initiative_selected):
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // 2
    config['DISRUPTION_INTERVALS'] = sorted(list(set([t // 2 for t in disruption_intervals_minutes]))) # Ensure unique, sorted steps
    
    if 'WORK_AREAS' in config and isinstance(config['WORK_AREAS'], dict) and config['WORK_AREAS']:
        total_current_workers = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())

        if total_current_workers == 0 and team_size > 0:
            num_zones = len(config['WORK_AREAS'])
            if num_zones > 0:
                workers_per_zone = team_size // num_zones
                remainder_workers = team_size % num_zones
                zone_keys = list(config['WORK_AREAS'].keys())
                for i, zone_key in enumerate(zone_keys):
                    config['WORK_AREAS'][zone_key]['workers'] = workers_per_zone + (1 if i < remainder_workers else 0)
            total_current_workers = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())


        if total_current_workers > 0 and total_current_workers != team_size:
            ratio = team_size / total_current_workers
            for zone_key in config['WORK_AREAS']:
                config['WORK_AREAS'][zone_key]['workers'] = int(config['WORK_AREAS'][zone_key].get('workers', 0) * ratio)
            
            current_sum_after_ratio = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())
            if current_sum_after_ratio != team_size:
                diff = team_size - current_sum_after_ratio
                # Distribute difference, trying 'Assembly Line' first, then any other zone
                first_zone_key_to_add = None
                if 'Assembly Line' in config['WORK_AREAS']:
                    first_zone_key_to_add = 'Assembly Line'
                elif config['WORK_AREAS']: 
                    first_zone_key_to_add = next(iter(config['WORK_AREAS']))
                
                if first_zone_key_to_add:
                    config['WORK_AREAS'][first_zone_key_to_add]['workers'] = config['WORK_AREAS'][first_zone_key_to_add].get('workers', 0) + diff
        elif team_size == 0: 
            for zone_key in config['WORK_AREAS']:
                config['WORK_AREAS'][zone_key]['workers'] = 0
    
    validate_config(config) 
    logger.info(
        f"Running simulation with team_size={team_size}, shift_duration={shift_duration_minutes} min, disruptions (minutes): {disruption_intervals_minutes}, initiative: {team_initiative_selected}",
        extra={'user_action': 'Run Simulation'}
    )
    
    simulation_results_dict = simulate_workplace_operations(
        num_team_members=team_size,
        num_steps=config['SHIFT_DURATION_INTERVALS'],
        disruption_intervals=config['DISRUPTION_INTERVALS'], 
        team_initiative=team_initiative_selected,
        config=config
    )
    simulation_results_dict['config_params'] = {
        'TEAM_SIZE': team_size,
        'SHIFT_DURATION_MINUTES': shift_duration_minutes,
        'DISRUPTION_INTERVALS_MINUTES': disruption_intervals_minutes, 
        'DISRUPTION_INTERVALS_STEPS': config['DISRUPTION_INTERVALS'],
        'TEAM_INITIATIVE': team_initiative_selected
    }

    save_simulation_data(simulation_results_dict) 
    return simulation_results_dict

# Main content
def main():
    st.title("Workplace Shift Monitoring Dashboard")

    # Initialize session state for UI elements if not already present
    # This helps preserve their state across reruns if not directly tied to a button press
    # Sidebar controls
    for key in ['team_size_slider', 'shift_duration_slider', 'disruption_intervals_multiselect', 
                'team_initiative_selectbox', 'high_contrast_checkbox', 
                'use_3d_distribution_checkbox', 'debug_mode_checkbox']:
        if key not in st.session_state:
            # Default values will be set by the widgets themselves using st.session_state.get()
            pass 
            
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'active_tab_index_from_sidebar' not in st.session_state: 
        st.session_state.active_tab_index_from_sidebar = 0
    if 'show_tour' not in st.session_state:
        st.session_state.show_tour = False
    if 'show_help' not in st.session_state: 
        st.session_state.show_help = False

    team_size_input, shift_duration_input, disruption_intervals_input, team_initiative_input, \
    run_simulation_clicked, load_data_clicked, high_contrast_input, \
    use_3d_distribution_input, debug_mode_input = render_settings_sidebar()

    # Store sidebar inputs into session_state to make them sticky until changed by user
    st.session_state.team_size = team_size_input
    st.session_state.shift_duration = shift_duration_input
    st.session_state.disruption_intervals_minutes = disruption_intervals_input
    st.session_state.team_initiative = team_initiative_input
    st.session_state.high_contrast = high_contrast_input
    st.session_state.use_3d_distribution = use_3d_distribution_input
    st.session_state.debug_mode = debug_mode_input
    
    # Determine current simulation duration for sliders
    current_max_minutes_for_sliders = DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'] - 2 
    disruption_steps_for_plots = [t // 2 for t in DEFAULT_CONFIG.get('DISRUPTION_INTERVALS', [])] 

    if st.session_state.simulation_results and 'downtime_minutes' in st.session_state.simulation_results:
        num_steps_current_sim = len(st.session_state.simulation_results.get('downtime_minutes', []))
        current_max_minutes_for_sliders = (num_steps_current_sim - 1) * 2 if num_steps_current_sim > 0 else 0
        disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', disruption_steps_for_plots)
    elif 'shift_duration' in st.session_state : 
         current_max_minutes_for_sliders = st.session_state.shift_duration - 2
         # Use current sidebar disruption_intervals_minutes if no sim results
         disruption_steps_for_plots = [m // 2 for m in st.session_state.get('disruption_intervals_minutes', [])]


    if run_simulation_clicked:
        with st.spinner("Running simulation..."):
            try:
                st.session_state.simulation_results = run_simulation_logic(
                    st.session_state.team_size, 
                    st.session_state.shift_duration, 
                    st.session_state.disruption_intervals_minutes, 
                    st.session_state.team_initiative
                )
                if st.session_state.simulation_results and 'downtime_minutes' in st.session_state.simulation_results:
                    num_steps_new_sim = len(st.session_state.simulation_results.get('downtime_minutes', []))
                    current_max_minutes_for_sliders = (num_steps_new_sim - 1) * 2 if num_steps_new_sim > 0 else 0
                    disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', [])
                st.success("Simulation completed!", icon="✅")
                st.rerun() # Rerun to update sliders with new max values if duration changed
            except Exception as e:
                logger.error(f"Simulation failed: {str(e)}", extra={'user_action': 'Run Simulation'})
                st.error(f"Simulation failed: {str(e)}.")
                st.session_state.simulation_results = None 

    if load_data_clicked:
        with st.spinner("Loading saved data..."):
            try:
                loaded_results = load_simulation_data()
                if loaded_results and isinstance(loaded_results, dict) and 'downtime_minutes' in loaded_results:
                    st.session_state.simulation_results = loaded_results
                    
                    loaded_config = st.session_state.simulation_results.get('config_params', {})
                    if loaded_config:
                        st.session_state.team_size = loaded_config.get('TEAM_SIZE', st.session_state.team_size)
                        st.session_state.shift_duration = loaded_config.get('SHIFT_DURATION_MINUTES', st.session_state.shift_duration)
                        st.session_state.disruption_intervals_minutes = loaded_config.get('DISRUPTION_INTERVALS_MINUTES', st.session_state.disruption_intervals_minutes)
                        st.session_state.team_initiative = loaded_config.get('TEAM_INITIATIVE', st.session_state.team_initiative)
                    
                    num_steps_loaded_sim = len(st.session_state.simulation_results.get('downtime_minutes', []))
                    current_max_minutes_for_sliders = (num_steps_loaded_sim - 1) * 2 if num_steps_loaded_sim > 0 else 0
                    disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', [])
                    
                    st.success("Data loaded!", icon="✅")
                    st.rerun() # Rerun to update UI with loaded settings and data range
                else:
                    st.error("Failed to load data or data is empty/invalid.")
                    st.session_state.simulation_results = None 
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}", extra={'user_action': 'Load Data'})
                st.error(f"Failed to load data: {str(e)}.")
                st.session_state.simulation_results = None 

    # Onboarding Modal
    if st.session_state.get('show_tour', False):
        with st.container(): # Using st.dialog might be an option in newer Streamlit versions for true modals
            st.markdown("""
                <div class="onboarding-modal" role="dialog" aria-label="Onboarding Tour">
                    <h3>Welcome to the Dashboard!</h3>
                    <p>Explore key features:</p>
                    <ul style="color: #D1D5DB; line-height: 1.6;">
                        <li><b>Sidebar</b>: Adjust simulation settings and navigate sections.</li>
                        <li><b>Tabs</b>: View metrics, worker insights, and more.</li>
                        <li><b>Charts</b>: Hover for details, use sliders to filter, export as PNG.</li>
                        <li><b>Export</b>: Download data as CSV or generate PDF reports.</li>
                    </ul>
                    <p>Click below to start exploring!</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Get Started", key="close_tour_button"): # Unique key
                st.session_state.show_tour = False
                st.rerun()

    # Help Modal
    if st.session_state.get('show_help', False):
        with st.container():
            st.markdown("""
                <div class="onboarding-modal" role="dialog" aria-label="Help Section">
                    <h3>Help & Documentation</h3>
                    <p>Navigate the dashboard:</p>
                    <ul style="color: #D1D5DB; line-height: 1.6;">
                        <li><b>Overview</b>: High-level metrics with insights.</li>
                        <li><b>Operational Metrics</b>: Trends for performance.</li>
                        <li><b>Worker Insights</b>: Well-being and safety data.</li>
                        <li><b>Downtime</b>: Analyze downtime trends.</li>
                        <li><b>Glossary</b>: Metric definitions.</li>
                    </ul>
                    <p>Contact support@example.com for assistance.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Close Help", key="close_help_button"): # Unique key
                st.session_state.show_help = False
                st.rerun()

    # Tabs
    tab_names_main = ["Overview", "Operational Metrics", "Worker Insights", "Downtime", "Glossary"]
    # Use st.session_state to try and preserve active tab if sidebar navigation hint is used
    # However, st.tabs primarily controls its own state on direct click.
    active_tab_idx = st.session_state.get('active_tab_index_from_sidebar', 0)
    if active_tab_idx >= len(tab_names_main): active_tab_idx = 0 # bounds check
    
    # st.tabs does not have a parameter to set the default selected tab programmatically after initialization
    # The best way is to structure content such that it appears active based on session state,
    # or accept that sidebar clicks are hints and users must click tabs.
    # For simplicity, we'll let st.tabs manage its state.
    tabs = st.tabs(tab_names_main)


    # --- Overview Tab ---
    with tabs[0]:
        with st.container():
            st.header("Overview", divider="grey")
            st.markdown(
                '<div class="tooltip">Key Metrics<span class="tooltiptext">Summary of Task Compliance, Collaboration Proximity, Worker Well-Being, and Downtime with actionable insights.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                
                def safe_mean(data_list):
                    if data_list and isinstance(data_list, (list, np.ndarray)) and len(data_list) > 0:
                        valid_data = [x for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x))]
                        if valid_data: return np.mean(valid_data)
                    return 0.0

                def safe_sum(data_list):
                    if data_list and isinstance(data_list, (list, np.ndarray)) and len(data_list) > 0:
                        valid_data = [x for x in data_list if x is not None and not (isinstance(x, float) and np.isnan(x))]
                        if valid_data: return np.sum(valid_data)
                    return 0.0

                compliance_mean = safe_mean(sim_data.get('task_compliance', {}).get('data', []))
                proximity_mean = safe_mean(sim_data.get('collaboration_proximity', {}).get('data', []))
                wellbeing_mean = safe_mean(sim_data.get('worker_wellbeing', {}).get('scores', []))
                total_downtime = safe_sum(sim_data.get('downtime_minutes', []))

                logger.info(
                    f"Overview Metrics: Compliance={compliance_mean:.2f}, Proximity={proximity_mean:.2f}, Wellbeing={wellbeing_mean:.2f}, Downtime={total_downtime:.2f}",
                    extra={'user_action': 'Render Overview Metrics'}
                )

                summary_figs = []
                try:
                    summary_figs = plot_key_metrics_summary(compliance_mean, proximity_mean, wellbeing_mean, total_downtime)
                except Exception as e:
                    logger.error(f"Failed to plot key metrics: {str(e)}", extra={'user_action': 'Render Overview Metrics'})
                    st.error(f"Error rendering key metrics: {str(e)}.")

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Task Compliance", f"{compliance_mean:.1f}%", delta=f"{compliance_mean - 75:.1f}%")
                with col2: st.metric("Collaboration", f"{proximity_mean:.1f}%", delta=f"{proximity_mean - 60:.1f}%")
                with col3: st.metric("Well-Being", f"{wellbeing_mean:.1f}%", delta=f"{wellbeing_mean - 70:.1f}%")
                with col4: st.metric("Downtime", f"{total_downtime:.1f} min", delta=f"{total_downtime - 30:.1f} min", delta_color="inverse")
                
                with st.expander("View Detailed Data", expanded=False):
                    num_overview_steps = len(sim_data.get('downtime_minutes', []))
                    if num_overview_steps > 0:
                        overview_data = {
                            'Time (min)': [i * 2 for i in range(num_overview_steps)],
                            'Task Compliance (%)': sim_data.get('task_compliance', {}).get('data', [np.nan]*num_overview_steps)[:num_overview_steps],
                            'Collaboration (%)': sim_data.get('collaboration_proximity', {}).get('data', [np.nan]*num_overview_steps)[:num_overview_steps],
                            'Well-Being (%)': sim_data.get('worker_wellbeing', {}).get('scores', [np.nan]*num_overview_steps)[:num_overview_steps],
                            'Downtime (min)': sim_data.get('downtime_minutes', [np.nan]*num_overview_steps)[:num_overview_steps]
                        }
                        overview_df = pd.DataFrame(overview_data)
                        st.dataframe(overview_df, use_container_width=True, height=300)
                    else:
                        st.info("No detailed data available for overview.")
                
                if summary_figs:
                    cols_gauges = st.columns(min(len(summary_figs), 2)) 
                    for i, fig in enumerate(summary_figs):
                        with cols_gauges[i % len(cols_gauges)]: 
                             with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Run a simulation or load data to view metrics.", icon="ℹ️")

    # --- Operational Metrics Tab ---
    with tabs[1]:
        with st.container():
            st.header("Operational Metrics", divider="grey")
            st.markdown('<div class="tooltip">Performance Trends<span class="tooltiptext">Trends for task compliance, collaboration, recovery, and efficiency.</span></div>', unsafe_allow_html=True)
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                time_range_op_max = max(0, current_max_minutes_for_sliders) # Ensure non-negative
                
                time_range_op_val = st.session_state.get("time_range_op_slider_val", (0, time_range_op_max))
                # Ensure current value is within new max bounds
                time_range_op_val = (min(time_range_op_val[0], time_range_op_max), min(time_range_op_val[1], time_range_op_max))
                
                time_range_op = st.slider(
                    "Time Range (minutes)", 0, time_range_op_max, time_range_op_val, step=2,
                    key="time_range_op_slider", # Unique key
                    disabled=time_range_op_max == 0,
                    on_change=lambda: st.session_state.update(time_range_op_slider_val=st.session_state.time_range_op_slider)

                )
                time_indices = (time_range_op[0] // 2, time_range_op[1] // 2 + 1)
                filtered_disruption_steps = [s for s in disruption_steps_for_plots if time_indices[0] <= s < time_indices[1]]
                
                # Task Compliance
                tc_data = sim_data.get('task_compliance', {})
                filtered_compliance = tc_data.get('data', [])[time_indices[0]:time_indices[1]]
                if filtered_compliance:
                    try:
                        filtered_z_scores = tc_data.get('z_scores', [])[time_indices[0]:time_indices[1]]
                        filtered_forecast_tc = tc_data.get('forecast', [])[time_indices[0]:time_indices[1]] if tc_data.get('forecast') else None
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container"><h4>Task Compliance</h4>', unsafe_allow_html=True) # Added title
                            compliance_fig = plot_task_compliance_score(filtered_compliance, filtered_disruption_steps, filtered_forecast_tc, filtered_z_scores)
                            st.plotly_chart(compliance_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e: logger.error(f"Task Compliance Plot Error: {e}", exc_info=True); st.error("Error plotting Task Compliance.")
                else: st.info("No Task Compliance data for this range.")

                # Collaboration
                collab_data = sim_data.get('collaboration_proximity', {})
                filtered_collab = collab_data.get('data', [])[time_indices[0]:time_indices[1]]
                if filtered_collab:
                    try:
                        filtered_forecast_collab = collab_data.get('forecast', [])[time_indices[0]:time_indices[1]] if collab_data.get('forecast') else None
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container"><h4>Collaboration Proximity</h4>', unsafe_allow_html=True) # Added title
                            collaboration_fig = plot_collaboration_proximity_index(filtered_collab, filtered_disruption_steps, filtered_forecast_collab)
                            st.plotly_chart(collaboration_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e: logger.error(f"Collaboration Plot Error: {e}", exc_info=True); st.error("Error plotting Collaboration.")
                else: st.info("No Collaboration data for this range.")

                with st.expander("Additional Operational Metrics"):
                    # Operational Recovery & Loss
                    filtered_recovery = sim_data.get('operational_recovery', [])[time_indices[0]:time_indices[1]]
                    if filtered_recovery:
                        try:
                            filtered_loss = sim_data.get('productivity_loss', [])[time_indices[0]:time_indices[1]]
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container"><h4>Operational Recovery & Resilience</h4>', unsafe_allow_html=True) # Added title
                                resilience_fig = plot_operational_recovery(filtered_recovery, filtered_loss) 
                                st.plotly_chart(resilience_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e: logger.error(f"Recovery Plot Error: {e}", exc_info=True); st.error("Error plotting Recovery.")
                    else: st.info("No Operational Recovery data for this range.")

                    # Efficiency Metrics
                    efficiency_df = sim_data.get('efficiency_metrics_df', pd.DataFrame())
                    if not efficiency_df.empty:
                        try:
                            selected_metrics = st.multiselect(
                                "Select Efficiency Metrics to Display",
                                options=['uptime', 'throughput', 'quality', 'oee'],
                                default=['uptime', 'throughput', 'quality', 'oee'],
                                key="efficiency_metrics_multiselect_op" # Unique key
                            )
                            filtered_df_eff = efficiency_df.iloc[time_indices[0]:time_indices[1]] if isinstance(efficiency_df.index, pd.RangeIndex) else efficiency_df
                            if not filtered_df_eff.empty:
                                with st.container(border=True):
                                    st.markdown(f'<div class="plot-container"><h4>Operational Efficiency (OEE)</h4>', unsafe_allow_html=True) # Added title
                                    efficiency_fig = plot_operational_efficiency(filtered_df_eff, selected_metrics)
                                    st.plotly_chart(efficiency_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                    st.markdown('</div>', unsafe_allow_html=True)
                            else: st.info("No Efficiency data for this range after filtering.")
                        except Exception as e: logger.error(f"Efficiency Plot Error: {e}", exc_info=True); st.error("Error plotting Efficiency.")
                    else: st.info("No base Efficiency data available.")
            else:
                st.info("Run a simulation or load data to view operational metrics.", icon="ℹ️")
    
    # --- Worker Insights Tab ---
    with tabs[2]:
        with st.container():
            st.header("Worker Insights", divider="grey")
            st.markdown('<div class="tooltip">Worker Metrics<span class="tooltiptext">Distribution, well-being, and safety metrics.</span></div>', unsafe_allow_html=True)
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                team_positions_df = sim_data.get('team_positions_df', pd.DataFrame())
                time_range_worker_max = max(0, current_max_minutes_for_sliders)
                
                time_range_worker_val = st.session_state.get("time_range_worker_slider_val", (0, time_range_worker_max))
                time_range_worker_val = (min(time_range_worker_val[0], time_range_worker_max), min(time_range_worker_val[1], time_range_worker_max))


                with st.expander("Worker Distribution Analysis", expanded=True):
                    time_range_dist_worker = st.slider(
                        "Time Range (minutes) for Distribution Plots", 0, time_range_worker_max, time_range_worker_val, step=2,
                        key="time_range_dist_worker_slider", # Unique key
                        disabled=time_range_worker_max == 0,
                        on_change=lambda: st.session_state.update(time_range_worker_slider_val=st.session_state.time_range_dist_worker_slider)

                    )
                    time_indices_dist = (time_range_dist_worker[0] // 2, time_range_dist_worker[1] // 2 + 1)
                    
                    zone_options = ["All"] + list(DEFAULT_CONFIG.get('WORK_AREAS', {}).keys())
                    zone_filter = st.selectbox("Filter by Zone", zone_options, key="zone_filter_selectbox_dist") # Unique key
                    
                    filtered_df_dist = pd.DataFrame()
                    if not team_positions_df.empty:
                        temp_df = team_positions_df[(team_positions_df['step'] >= time_indices_dist[0]) & (team_positions_df['step'] < time_indices_dist[1])]
                        if zone_filter != "All": temp_df = temp_df[temp_df['zone'] == zone_filter]
                        filtered_df_dist = temp_df

                    show_entry_exit = st.checkbox("Show Entry/Exit Points on Plots", value=True, key="show_entry_exit_checkbox_dist") # Unique key
                    show_prod_lines = st.checkbox("Show Production Lines on Plots", value=True, key="show_prod_lines_checkbox_dist") # Unique key
                    
                    col_dist1, col_dist2 = st.columns(2)
                    with col_dist1:
                        st.markdown("<h5>Worker Positions (Snapshot)</h5>", unsafe_allow_html=True)
                        min_s, max_s = time_indices_dist[0], max(time_indices_dist[0], time_indices_dist[1] -1)
                        selected_step_val = st.session_state.get("selected_step_dist_slider_val", min_s)
                        selected_step_val = max(min_s, min(selected_step_val, max_s)) # Ensure in bounds

                        selected_step = st.slider("Select Time Step for Snapshot", min_s, max_s, selected_step_val, 
                                                  key="selected_step_dist_slider", # Unique key
                                                  disabled=max_s < min_s,
                                                  on_change=lambda: st.session_state.update(selected_step_dist_slider_val=st.session_state.selected_step_dist_slider)
                                                  )
                        if not team_positions_df.empty and max_s >= min_s:
                            try:
                                with st.container(border=True):
                                    dist_fig = plot_worker_distribution(team_positions_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, 
                                                                        use_3d=st.session_state.get('use_3d_distribution',False), selected_step=selected_step, 
                                                                        show_entry_exit=show_entry_exit, show_production_lines=show_prod_lines)
                                    st.plotly_chart(dist_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            except Exception as e: logger.error(f"Distribution Plot Error: {e}", exc_info=True); st.error("Error: Worker Positions plot.")
                        else: st.info("No data for Worker Positions snapshot with current filters.")
                    with col_dist2:
                        st.markdown("<h5>Worker Density Heatmap (Aggregated)</h5>", unsafe_allow_html=True)
                        if not filtered_df_dist.empty:
                            try:
                                with st.container(border=True):
                                    heatmap_fig = plot_worker_density_heatmap(filtered_df_dist, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, 
                                                                              show_entry_exit=show_entry_exit, show_production_lines=show_prod_lines)
                                    st.plotly_chart(heatmap_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            except Exception as e: logger.error(f"Heatmap Plot Error: {e}", exc_info=True); st.error("Error: Density Heatmap plot.")
                        else: st.info("No data for Density Heatmap with current filters.")

                with st.expander("Worker Well-Being & Safety Analysis", expanded=True):
                    time_range_well_val = st.session_state.get("time_range_well_slider_val", (0, time_range_worker_max))
                    time_range_well_val = (min(time_range_well_val[0], time_range_worker_max), min(time_range_well_val[1], time_range_worker_max))
                    
                    time_range_well_worker = st.slider(
                        "Time Range (minutes) for Well-Being/Safety Plots", 0, time_range_worker_max, time_range_well_val, step=2,
                        key="time_range_well_worker_slider", # Unique key
                        disabled=time_range_worker_max == 0,
                        on_change=lambda: st.session_state.update(time_range_well_slider_val=st.session_state.time_range_well_worker_slider)
                    )
                    time_indices_well = (time_range_well_worker[0] // 2, time_range_well_worker[1] // 2 + 1)

                    col_well1, col_well2 = st.columns(2)
                    worker_wellbeing_data = sim_data.get('worker_wellbeing', {'scores': [], 'triggers': {}})
                    with col_well1:
                        st.markdown("<h5>Worker Well-Being Index</h5>", unsafe_allow_html=True)
                        filtered_scores = worker_wellbeing_data.get('scores', [])[time_indices_well[0]:time_indices_well[1]]
                        if filtered_scores:
                            try:
                                ww_trig = worker_wellbeing_data.get('triggers', {})
                                filt_trig = {k: [t for t in v if time_indices_well[0] <= t < time_indices_well[1]] 
                                             for k, v in ww_trig.items() if isinstance(v, list)}
                                filt_trig['work_area'] = {
                                    wk: [t for t in wv if time_indices_well[0] <= t < time_indices_well[1]]
                                    for wk, wv in ww_trig.get('work_area',{}).items()
                                }
                                with st.container(border=True):
                                    wellbeing_fig = plot_worker_wellbeing(filtered_scores, filt_trig)
                                    st.plotly_chart(wellbeing_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            except Exception as e: logger.error(f"Wellbeing Plot Error: {e}", exc_info=True); st.error("Error: Well-Being Index plot.")
                        else: st.info("No Well-Being Index data for this range.")
                    
                    psychological_safety_data = sim_data.get('psychological_safety', [])
                    with col_well2:
                        st.markdown("<h5>Psychological Safety Score</h5>", unsafe_allow_html=True)
                        filtered_safety = psychological_safety_data[time_indices_well[0]:time_indices_well[1]]
                        if filtered_safety:
                            try:
                                with st.container(border=True):
                                    safety_fig = plot_psychological_safety(filtered_safety)
                                    st.plotly_chart(safety_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            except Exception as e: logger.error(f"Safety Plot Error: {e}", exc_info=True); st.error("Error: Psychological Safety plot.")
                        else: st.info("No Psychological Safety data for this range.")
                    
                    st.markdown("<h6>Well-Being Triggers (within selected time range)</h6>", unsafe_allow_html=True)
                    ww_trig_disp = worker_wellbeing_data.get('triggers', {})
                    filt_trig_disp = {k: [t for t in v if time_indices_well[0] <= t < time_indices_well[1]] 
                                        for k, v in ww_trig_disp.items() if isinstance(v, list)}
                    filt_trig_disp['work_area'] = {
                        wk: [t for t in wv if time_indices_well[0] <= t < time_indices_well[1]]
                        for wk, wv in ww_trig_disp.get('work_area',{}).items()
                    }
                    st.write(f"**Threshold Alerts (< {DEFAULT_CONFIG.get('WELLBEING_THRESHOLD',0)*100}%):** {filt_trig_disp.get('threshold', 'N/A')}")
                    st.write(f"**Trend Alerts (Declining):** {filt_trig_disp.get('trend', 'N/A')}")
                    st.write("**Work Area Alerts:**")
                    wa_trigs = filt_trig_disp.get('work_area',{})
                    if wa_trigs: [st.write(f"  {zone}: {triggers}") for zone, triggers in wa_trigs.items() if triggers]
                    else: st.write("  None")
                    st.write(f"**Disruption Alerts:** {filt_trig_disp.get('disruption', 'N/A')}")
            else:
                st.info("Run a simulation or load data to view worker insights.", icon="ℹ️")

    # --- Downtime Tab ---
    with tabs[3]:
        with st.container():
            st.header("Downtime Analysis", divider="grey")
            st.markdown('<div class="tooltip">Downtime Trends<span class="tooltiptext">Downtime with alerts for high values.</span></div>', unsafe_allow_html=True)
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                downtime_all = sim_data.get('downtime_minutes', [])
                time_range_down_max = max(0, current_max_minutes_for_sliders)

                time_range_down_val = st.session_state.get("time_range_down_slider_val", (0, time_range_down_max))
                time_range_down_val = (min(time_range_down_val[0], time_range_down_max), min(time_range_down_val[1], time_range_down_max))

                time_range_down_tab = st.slider(
                    "Time Range (minutes) for Downtime Plot", 0, time_range_down_max, time_range_down_val, step=2,
                    key="time_range_down_tab_slider", # Unique key
                    disabled=time_range_down_max == 0,
                    on_change=lambda: st.session_state.update(time_range_down_slider_val=st.session_state.time_range_down_tab_slider)
                )
                time_indices_down = (time_range_down_tab[0] // 2, time_range_down_tab[1] // 2 + 1)
                filtered_downtime = downtime_all[time_indices_down[0]:time_indices_down[1]]
                if filtered_downtime:
                    try:
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container"><h4>Downtime Trend</h4>', unsafe_allow_html=True) # Added title
                            downtime_fig = plot_downtime_trend(filtered_downtime, DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD', 10)) 
                            st.plotly_chart(downtime_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e: logger.error(f"Downtime Plot Error: {e}", exc_info=True); st.error("Error: Downtime Trend plot.")
                else: st.info("No Downtime data for this range.")
            else:
                st.info("Run a simulation or load data to view downtime analysis.", icon="ℹ️")

    # --- Glossary Tab ---
    with tabs[4]:
        with st.container():
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
