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
    filename='dashboard.log'
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
                key="team_size"
            )
            shift_duration = st.slider(
                "Shift Duration (minutes)",
                min_value=200, max_value=2000, value=st.session_state.get('shift_duration', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']), step=2,
                help="Shift duration in minutes (2-minute intervals).",
                key="shift_duration"
            )
            
            # Ensure default disruption intervals are valid for the current shift_duration
            max_step_for_disruption = shift_duration // 2
            valid_default_disruption_steps = [s for s in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if s < max_step_for_disruption]
            default_disruption_minutes = [i * 2 for i in valid_default_disruption_steps]

            disruption_intervals = st.multiselect(
                "Disruption Times (minutes)",
                options=[i * 2 for i in range(max_step_for_disruption)],
                default=st.session_state.get('disruption_intervals_minutes', default_disruption_minutes),
                help="Times when disruptions occur.",
                key="disruption_intervals_minutes" # Changed key to avoid conflict if 'disruption_intervals' is used for steps
            )
            team_initiative = st.selectbox(
                "Team Initiative",
                options=["More frequent breaks", "Team recognition"],
                index=0, # Consider storing index in session_state if needed
                help="Strategy to improve well-being and safety.",
                key="team_initiative"
            )
            run_simulation = st.button(
                "Run Simulation", 
                key="run_simulation", 
                help="Start a new simulation.",
                type="primary"
            )

        # Visualization Settings
        with st.expander("🎨 Visualizations"):
            high_contrast = st.checkbox(
                "High Contrast Mode", 
                help="Enable high-contrast colors for accessibility.",
                key="high_contrast"
            )
            use_3d_distribution = st.checkbox(
                "3D Team Distribution", 
                help="Use 3D scatter plot for team distribution.",
                key="use_3d_distribution"
            )
            debug_mode = st.checkbox(
                "Debug Mode", 
                help="Show configuration data for debugging.",
                key="debug_mode"
            )

        # Data Management
        with st.expander("💾 Data"):
            load_data = st.button(
                "Load Saved Data", 
                key="load_data", 
                help="Load previously saved simulation data."
            )
            
            can_generate_report = 'simulation_results' in st.session_state and st.session_state.simulation_results is not None
            if st.button("Download PDF Report", key="download_report", help="Generate a PDF report.", disabled=not can_generate_report):
                if can_generate_report: # Double check, though disabled should prevent click
                    try:
                        sim_results = st.session_state.simulation_results
                        num_sim_steps = len(sim_results['downtime_minutes']) # Use actual data length
                        summary_df = pd.DataFrame({
                            'step': range(num_sim_steps),
                            'time_minutes': [i * 2 for i in range(num_sim_steps)],
                            'task_compliance': sim_results['task_compliance']['data'],
                            'collaboration_proximity': sim_results['collaboration_proximity']['data'],
                            'operational_recovery': sim_results['operational_recovery'],
                            'worker_wellbeing': sim_results['worker_wellbeing']['scores'],
                            'psychological_safety': sim_results['psychological_safety'],
                            'productivity_loss': sim_results['productivity_loss'],
                            'downtime_minutes': sim_results['downtime_minutes'],
                            'task_completion_rate': sim_results['task_completion_rate']
                        })
                        generate_pdf_report(summary_df) # Assumes generate_pdf_report is adapted for this df structure
                        st.success("PDF report generated as 'workplace_report.tex'. Compile with LaTeX to view.")
                    except Exception as e:
                        logger.error(f"Failed to generate report: {str(e)}", extra={'user_action': 'Download PDF Report'})
                        st.error(f"Failed to generate report: {str(e)}.")
                else:
                     st.info("Run a simulation or load data to enable PDF report generation.", icon="ℹ️")


        # Export Options
        with st.expander("📊 Export Options"):
            if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                if st.button("Export Plots as PNG", key="export_png"):
                    st.info("Exporting plots as PNG is handled within each plot's display options (camera icon).")
                
                sim_results = st.session_state.simulation_results
                num_sim_steps_csv = len(sim_results['downtime_minutes']) # Use actual data length
                summary_df_csv = pd.DataFrame({
                    'step': range(num_sim_steps_csv),
                    'time_minutes': [i * 2 for i in range(num_sim_steps_csv)],
                    'task_compliance': sim_results['task_compliance']['data'],
                    'collaboration_proximity': sim_results['collaboration_proximity']['data'],
                    'operational_recovery': sim_results['operational_recovery'],
                    'worker_wellbeing': sim_results['worker_wellbeing']['scores'],
                    'psychological_safety': sim_results['psychological_safety'],
                    'productivity_loss': sim_results['productivity_loss'],
                    'downtime_minutes': sim_results['downtime_minutes'],
                    'task_completion_rate': sim_results['task_completion_rate']
                })
                st.download_button(
                    label="Download Summary CSV",
                    data=summary_df_csv.to_csv(index=False).encode('utf-8'),
                    file_name="workplace_summary.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            else:
                st.info("Run a simulation to enable export options.", icon="ℹ️")

        # Debug Information
        if debug_mode: # This uses st.session_state.debug_mode implicitly
            with st.expander("🛠️ Debug Info"):
                st.write("**Entry/Exit Points:**")
                st.write(DEFAULT_CONFIG.get('ENTRY_EXIT_POINTS', "Not defined"))
                st.write("**Work Areas:**")
                st.write(DEFAULT_CONFIG.get('WORK_AREAS', "Not defined"))
                if 'simulation_results' in st.session_state and st.session_state.simulation_results:
                    st.write("**Current Simulation Config (partial):**")
                    st.json({
                        'TEAM_SIZE': st.session_state.simulation_results.get('config_params', {}).get('TEAM_SIZE'),
                        'SHIFT_DURATION_MINUTES': st.session_state.simulation_results.get('config_params', {}).get('SHIFT_DURATION_MINUTES'),
                        'DISRUPTION_INTERVALS_STEPS': st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS')
                    })


        # Navigation and Help
        st.header("📋 Navigation", divider="grey")
        tab_names = ["Overview", "Operational Metrics", "Worker Insights", "Downtime", "Glossary"]
        # Note: st.tabs doesn't have a programmatic way to set active tab after initialization via session_state easily.
        # These buttons will set session_state, but won't directly switch the st.tabs view unless content inside tabs is conditional.
        # For now, they serve as quick "bookmarks" if the user wants to manually click the tab after.
        for i, tab_name in enumerate(tab_names):
            if st.button(tab_name, key=f"nav_{tab_name.lower().replace(' ', '_')}", help=f"Go to {tab_name}"):
                st.session_state.active_tab_index = i # Store index for potential future use

        if st.button("ℹ️ Help", key="help_button"):
            st.session_state.show_help = True

        if st.button("🚀 Take a Tour", key="tour_button"):
            st.session_state.show_tour = True

    return team_size, shift_duration, disruption_intervals, team_initiative, run_simulation, load_data, high_contrast, use_3d_distribution, debug_mode

# Simulation logic with caching
@st.cache_data
def run_simulation_logic(team_size, shift_duration_minutes, disruption_intervals_minutes, team_initiative_selected):
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration_minutes
    config['SHIFT_DURATION_INTERVALS'] = shift_duration_minutes // 2
    # Convert disruption intervals from minutes to steps (assuming 2 min per step)
    config['DISRUPTION_INTERVALS'] = [t // 2 for t in disruption_intervals_minutes]
    
    # Adjust worker distribution in WORK_AREAS based on team_size
    if 'WORK_AREAS' in config and isinstance(config['WORK_AREAS'], dict):
        total_current_workers = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())

        if total_current_workers == 0 and team_size > 0:
            # Distribute team_size workers if current distribution is zero
            # Fallback: assign all to 'Assembly Line' or the first available zone
            assigned = False
            if 'Assembly Line' in config['WORK_AREAS']:
                config['WORK_AREAS']['Assembly Line']['workers'] = team_size
                assigned = True
            elif config['WORK_AREAS']: # If any work areas exist
                first_zone_key = next(iter(config['WORK_AREAS']))
                config['WORK_AREAS'][first_zone_key]['workers'] = team_size
                assigned = True
            if assigned:
                 total_current_workers = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())

        if total_current_workers > 0 and total_current_workers != team_size:
            ratio = team_size / total_current_workers
            for zone_key in config['WORK_AREAS']:
                config['WORK_AREAS'][zone_key]['workers'] = int(config['WORK_AREAS'][zone_key].get('workers', 0) * ratio)
            
            current_sum_after_ratio = sum(zone.get('workers', 0) for zone in config['WORK_AREAS'].values())
            if current_sum_after_ratio != team_size:
                diff = team_size - current_sum_after_ratio
                # Distribute difference, trying 'Assembly Line' first, then any other zone
                if 'Assembly Line' in config['WORK_AREAS']:
                    config['WORK_AREAS']['Assembly Line']['workers'] = config['WORK_AREAS']['Assembly Line'].get('workers', 0) + diff
                elif config['WORK_AREAS']: # If 'Assembly Line' not there, add to the first zone
                    first_zone_key = next(iter(config['WORK_AREAS']))
                    config['WORK_AREAS'][first_zone_key]['workers'] = config['WORK_AREAS'][first_zone_key].get('workers', 0) + diff
        elif team_size == 0: # If target team size is 0, set all zone workers to 0
            for zone_key in config['WORK_AREAS']:
                config['WORK_AREAS'][zone_key]['workers'] = 0

    validate_config(config) # Ensure validate_config can handle the structure of config
    logger.info(
        f"Running simulation with team_size={team_size}, shift_duration={shift_duration_minutes} min, disruptions (minutes): {disruption_intervals_minutes}",
        extra={'user_action': 'Run Simulation'}
    )
    
    # This function is expected to return a dictionary of results
    # The save_simulation_data function will also need to handle this dictionary
    simulation_results_dict = simulate_workplace_operations(
        num_team_members=team_size,
        num_steps=config['SHIFT_DURATION_INTERVALS'],
        disruption_intervals=config['DISRUPTION_INTERVALS'], # these are steps
        team_initiative=team_initiative_selected,
        config=config
    )
    # Store key config params with results for reference
    simulation_results_dict['config_params'] = {
        'TEAM_SIZE': team_size,
        'SHIFT_DURATION_MINUTES': shift_duration_minutes,
        'DISRUPTION_INTERVALS_MINUTES': disruption_intervals_minutes, # Storing original minutes
        'DISRUPTION_INTERVALS_STEPS': config['DISRUPTION_INTERVALS'] # And steps used in sim
    }

    save_simulation_data(simulation_results_dict) # Assumes save_simulation_data saves a dictionary
    return simulation_results_dict

# Main content
def main():
    st.title("Workplace Shift Monitoring Dashboard")

    # Initialize session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'active_tab_index' not in st.session_state: # For sidebar navigation attempt
        st.session_state.active_tab_index = 0
    if 'show_tour' not in st.session_state:
        st.session_state.show_tour = False
    if 'show_help' not st.session_state:
        st.session_state.show_help = False

    # Sidebar settings
    # Note: disruption_intervals_from_sidebar is in minutes
    team_size, shift_duration, disruption_intervals_from_sidebar, team_initiative, run_simulation, \
    load_data, high_contrast, use_3d_distribution, debug_mode = render_settings_sidebar()
    
    # Determine current simulation duration for sliders, defaulting to sidebar setting or DEFAULT_CONFIG
    if st.session_state.simulation_results:
        # If results exist, base duration on them
        # Assuming 'downtime_minutes' or similar list represents steps
        num_steps_current_sim = len(st.session_state.simulation_results['downtime_minutes'])
        current_max_minutes_for_sliders = (num_steps_current_sim - 1) * 2 if num_steps_current_sim > 0 else 0
        # Also get disruption intervals used in the loaded/last sim (in steps)
        disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', [])
    else:
        # If no results, base on current sidebar shift_duration slider
        current_max_minutes_for_sliders = st.session_state.get('shift_duration', DEFAULT_CONFIG['SHIFT_DURATION_MINUTES']) -2
        # Convert sidebar disruption minutes to steps for potential default plotting
        disruption_steps_for_plots = [t // 2 for t in st.session_state.get('disruption_intervals_minutes', [])]


    # Handle simulation and data loading
    if run_simulation:
        with st.spinner("Running simulation..."):
            try:
                # Pass disruption_intervals_from_sidebar (which are in minutes)
                st.session_state.simulation_results = run_simulation_logic(
                    team_size, shift_duration, disruption_intervals_from_sidebar, team_initiative
                )
                # Update current_max_minutes_for_sliders based on new simulation
                num_steps_new_sim = len(st.session_state.simulation_results['downtime_minutes'])
                current_max_minutes_for_sliders = (num_steps_new_sim - 1) * 2 if num_steps_new_sim > 0 else 0
                disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', [])
                st.success("Simulation completed!", icon="✅")
            except Exception as e:
                logger.error(f"Simulation failed: {str(e)}", extra={'user_action': 'Run Simulation'})
                st.error(f"Simulation failed: {str(e)}.")
                st.session_state.simulation_results = None # Clear results on failure

    if load_data:
        with st.spinner("Loading saved data..."):
            try:
                # load_simulation_data is expected to return a dictionary
                st.session_state.simulation_results = load_simulation_data()
                if st.session_state.simulation_results:
                     # Update current_max_minutes_for_sliders based on loaded simulation
                    num_steps_loaded_sim = len(st.session_state.simulation_results['downtime_minutes'])
                    current_max_minutes_for_sliders = (num_steps_loaded_sim - 1) * 2 if num_steps_loaded_sim > 0 else 0
                    disruption_steps_for_plots = st.session_state.simulation_results.get('config_params', {}).get('DISRUPTION_INTERVALS_STEPS', [])
                    
                    # Update sidebar controls to reflect loaded data if params are stored
                    loaded_config = st.session_state.simulation_results.get('config_params', {})
                    if loaded_config:
                        st.session_state.team_size = loaded_config.get('TEAM_SIZE', team_size)
                        st.session_state.shift_duration = loaded_config.get('SHIFT_DURATION_MINUTES', shift_duration)
                        st.session_state.disruption_intervals_minutes = loaded_config.get('DISRUPTION_INTERVALS_MINUTES', disruption_intervals_from_sidebar)
                        # team_initiative might not be stored or easy to map back to index, handle carefully
                    st.success("Data loaded!", icon="✅")
                else:
                    st.error("Failed to load data or data is empty.")
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}", extra={'user_action': 'Load Data'})
                st.error(f"Failed to load data: {str(e)}.")
                st.session_state.simulation_results = None # Clear results on failure


    # Onboarding Modal
    if st.session_state.show_tour:
        with st.container():
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
            if st.button("Get Started", key="close_tour"):
                st.session_state.show_tour = False

    # Help Modal
    if st.session_state.show_help:
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
            if st.button("Close Help", key="close_help"):
                st.session_state.show_help = False

    # Tabs
    tab_names = ["Overview", "Operational Metrics", "Worker Insights", "Downtime", "Glossary"]
    tabs = st.tabs(tab_names)

    # Overview Tab
    with tabs[0]:
        with st.container():
            st.header("Overview", divider="grey")
            st.markdown(
                '<div class="tooltip">Key Metrics<span class="tooltiptext">Summary of Task Compliance, Collaboration Proximity, Worker Well-Being, and Downtime with actionable insights.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                
                # Validate and compute means with fallbacks
                # Ensure data exists and is not all NaN before computing mean
                def safe_mean(data_list):
                    if data_list and isinstance(data_list, (list, np.ndarray)) and len(data_list) > 0:
                        if not np.all(np.isnan(data_list)):
                            return np.mean(data_list)
                    return 0.0

                compliance_mean = safe_mean(sim_data['task_compliance']['data'])
                proximity_mean = safe_mean(sim_data['collaboration_proximity']['data'])
                wellbeing_mean = safe_mean(sim_data['worker_wellbeing']['scores'])
                
                total_downtime = 0.0
                if sim_data['downtime_minutes'] and isinstance(sim_data['downtime_minutes'], (list, np.ndarray)) and len(sim_data['downtime_minutes']) > 0:
                    if not np.all(np.isnan(sim_data['downtime_minutes'])):
                        total_downtime = np.sum(sim_data['downtime_minutes'])

                logger.info(
                    f"Overview Metrics: Compliance={compliance_mean}, Proximity={proximity_mean}, Wellbeing={wellbeing_mean}, Downtime={total_downtime}",
                    extra={'user_action': 'Render Overview Metrics'}
                )

                try:
                    summary_figs = plot_key_metrics_summary(compliance_mean, proximity_mean, wellbeing_mean, total_downtime)
                except Exception as e:
                    logger.error(f"Failed to plot key metrics: {str(e)}", extra={'user_action': 'Render Overview Metrics'})
                    st.error(f"Error rendering key metrics: {str(e)}.")
                    summary_figs = []

                # Enhanced Metrics Display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Task Compliance", f"{compliance_mean:.1f}%", delta=f"{compliance_mean - 75:.1f}%" if compliance_mean is not None else None, delta_color="normal")
                with col2:
                    st.metric("Collaboration", f"{proximity_mean:.1f}%", delta=f"{proximity_mean - 60:.1f}%" if proximity_mean is not None else None, delta_color="normal")
                with col3:
                    st.metric("Well-Being", f"{wellbeing_mean:.1f}%", delta=f"{wellbeing_mean - 70:.1f}%" if wellbeing_mean is not None else None, delta_color="normal")
                with col4:
                    st.metric("Downtime", f"{total_downtime:.1f} min", delta=f"{total_downtime - 30:.1f} min" if total_downtime is not None else None, delta_color="inverse")
                
                # Data Table
                with st.expander("View Detailed Data", expanded=False):
                    num_overview_steps = len(sim_data['task_compliance']['data'])
                    overview_df = pd.DataFrame({
                        'Time (min)': [i * 2 for i in range(num_overview_steps)],
                        'Task Compliance (%)': sim_data['task_compliance']['data'],
                        'Collaboration (%)': sim_data['collaboration_proximity']['data'],
                        'Well-Being (%)': sim_data['worker_wellbeing']['scores'],
                        'Downtime (min)': sim_data['downtime_minutes']
                    })
                    st.dataframe(overview_df, use_container_width=True, height=300)
                
                # Gauge Charts
                # Assuming summary_figs is a list of figures, display them.
                # If specific layout needed, adjust columns. For example, 2x2 grid:
                if summary_figs:
                    cols = st.columns(2) 
                    for i, fig in enumerate(summary_figs):
                        with cols[i % 2]: # Distribute into 2 columns
                             with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Run a simulation or load data to view metrics.", icon="ℹ️")

    # Operational Metrics Tab
    with tabs[1]:
        with st.container():
            st.header("Operational Metrics", divider="grey")
            st.markdown(
                '<div class="tooltip">Performance Trends<span class="tooltiptext">Trends for task compliance, collaboration, recovery, and efficiency.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                time_range_op = st.slider(
                    "Time Range (minutes)",
                    min_value=0,
                    max_value=max(0, current_max_minutes_for_sliders), # Ensure max_value is not negative
                    value=(0, max(0, current_max_minutes_for_sliders)),
                    step=2,
                    key="time_range_op"
                )
                time_indices = (time_range_op[0] // 2, time_range_op[1] // 2 + 1) # +1 for slicing upper bound

                # Filter disruption steps for the selected time range (disruptions are in steps)
                filtered_disruption_steps = [s for s in disruption_steps_for_plots if time_indices[0] <= s < time_indices[1]]
                
                try:
                    tc_data = sim_data['task_compliance']
                    filtered_compliance = tc_data['data'][time_indices[0]:time_indices[1]]
                    filtered_z_scores = tc_data['z_scores'][time_indices[0]:time_indices[1]]
                    filtered_forecast_tc = tc_data['forecast'][time_indices[0]:time_indices[1]] if tc_data.get('forecast') is not None else None
                    
                    if not filtered_compliance: # Check if list is empty
                        logger.warning("Empty task compliance data for selected range.", extra={'user_action': 'Render Operational Metrics'})
                        st.warning("No task compliance data available for the selected time range.")
                    else:
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                            compliance_fig = plot_task_compliance_score(filtered_compliance, filtered_disruption_steps, filtered_forecast_tc, filtered_z_scores)
                            st.plotly_chart(compliance_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Failed to render task compliance chart: {str(e)}", extra={'user_action': 'Render Operational Metrics'})
                    st.error(f"Error rendering task compliance chart: {str(e)}.")
                
                try:
                    collab_data = sim_data['collaboration_proximity']
                    filtered_collab = collab_data['data'][time_indices[0]:time_indices[1]]
                    filtered_forecast_collab = collab_data['forecast'][time_indices[0]:time_indices[1]] if collab_data.get('forecast') is not None else None
                    if not filtered_collab:
                         logger.warning("Empty collaboration data for selected range.", extra={'user_action': 'Render Operational Metrics'})
                         st.warning("No collaboration data available for the selected time range.")
                    else:
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                            collaboration_fig = plot_collaboration_proximity_index(filtered_collab, filtered_disruption_steps, filtered_forecast_collab)
                            st.plotly_chart(collaboration_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Failed to render collaboration chart: {str(e)}", extra={'user_action': 'Render Operational Metrics'})
                    st.error(f"Error rendering collaboration chart: {str(e)}.")
                
                with st.expander("Additional Metrics"):
                    try:
                        filtered_recovery = sim_data['operational_recovery'][time_indices[0]:time_indices[1]]
                        filtered_loss = sim_data['productivity_loss'][time_indices[0]:time_indices[1]]
                        if not filtered_recovery: # or not filtered_loss
                            logger.warning("Empty recovery/loss data for selected range.", extra={'user_action': 'Render Operational Metrics'})
                            st.warning("No operational recovery/loss data available for the selected time range.")
                        else:
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                resilience_fig = plot_operational_recovery(filtered_recovery, filtered_loss) # Assumes this plot takes filtered_disruption_steps if needed
                                st.plotly_chart(resilience_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"Failed to render operational recovery chart: {str(e)}", extra={'user_action': 'Render Operational Metrics'})
                        st.error(f"Error rendering operational recovery chart: {str(e)}.")
                    
                    try:
                        selected_metrics = st.multiselect(
                            "Efficiency Metrics",
                            options=['uptime', 'throughput', 'quality', 'oee'],
                            default=['uptime', 'throughput', 'quality', 'oee'],
                            key="efficiency_metrics_op"
                        )
                        # Ensure efficiency_metrics_df exists and has step/time index for slicing
                        efficiency_df = sim_data['efficiency_metrics_df']
                        if isinstance(efficiency_df.index, pd.RangeIndex): # if indexed by step
                            filtered_df = efficiency_df.iloc[time_indices[0]:time_indices[1]]
                        else: # Assuming it might have a time-based index, adjust as needed
                            filtered_df = efficiency_df # Fallback or implement time-based slicing
                        
                        if filtered_df.empty:
                            logger.warning("Empty efficiency data for selected range.", extra={'user_action': 'Render Operational Metrics'})
                            st.warning("No efficiency data available for the selected time range.")
                        else:
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                efficiency_fig = plot_operational_efficiency(filtered_df, selected_metrics)
                                st.plotly_chart(efficiency_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"Failed to render efficiency chart: {str(e)}", extra={'user_action': 'Render Operational Metrics'})
                        st.error(f"Error rendering efficiency chart: {str(e)}.")
            else:
                st.info("Run a simulation or load data to view metrics.", icon="ℹ️")

    # Worker Insights Tab
    with tabs[2]:
        with st.container():
            st.header("Worker Insights", divider="grey")
            st.markdown(
                '<div class="tooltip">Worker Metrics<span class="tooltiptext">Distribution, well-being, and safety metrics.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                team_positions_df = sim_data['team_positions_df']
                worker_wellbeing_data = sim_data['worker_wellbeing']
                psychological_safety_data = sim_data['psychological_safety']

                with st.expander("Worker Distribution", expanded=True):
                    time_range_dist = st.slider(
                        "Time Range (minutes) for Distribution", # Unique label for key
                        min_value=0,
                        max_value=max(0, current_max_minutes_for_sliders),
                        value=(0, max(0, current_max_minutes_for_sliders)),
                        step=2,
                        key="time_range_dist"
                    )
                    time_indices_dist = (time_range_dist[0] // 2, time_range_dist[1] // 2 + 1)
                    
                    zone_options = ["All"] + list(DEFAULT_CONFIG.get('WORK_AREAS', {}).keys())
                    zone_filter = st.selectbox(
                        "Zone", 
                        options=zone_options,
                        key="zone_filter_dist"
                    )
                    
                    # Filter DataFrame by time steps and zone
                    filtered_df_dist = team_positions_df[
                        (team_positions_df['step'] >= time_indices_dist[0]) & 
                        (team_positions_df['step'] < time_indices_dist[1])
                    ]
                    if zone_filter != "All":
                        filtered_df_dist = filtered_df_dist[filtered_df_dist['zone'] == zone_filter]

                    show_entry_exit = st.checkbox("Show Entry/Exit Points", value=True, key="show_entry_exit_dist")
                    show_production_lines = st.checkbox("Show Production Lines", value=True, key="show_production_lines_dist")
                    
                    col_dist1, col_dist2 = st.columns(2)
                    with col_dist1:
                        st.markdown("### Worker Positions")
                        # Slider for selected_step should be within the filtered time range
                        min_step_dist = time_indices_dist[0]
                        max_step_dist = max(min_step_dist, time_indices_dist[1] - 1) # Ensure max_step is not less than min_step
                        
                        selected_step = st.slider(
                            "Time Step (for Worker Positions snapshot)",
                            min_value=min_step_dist,
                            max_value=max_step_dist,
                            value=min_step_dist, # Default to the start of the filtered range
                            key="team_distribution_step"
                        )
                        try:
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                # plot_worker_distribution might need the full df and filter internally, or use filtered_df_dist
                                distribution_fig = plot_worker_distribution(
                                    team_positions_df, # Pass the full DF, selected_step will pick the slice
                                    DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, use_3d=use_3d_distribution,
                                    selected_step=selected_step, # This is the specific step to plot
                                    show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                                )
                                st.plotly_chart(distribution_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            logger.error(f"Failed to plot worker distribution: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering worker distribution: {str(e)}. Check debug mode for details.")
                    with col_dist2:
                        st.markdown("### Density Heatmap")
                        try:
                            with st.container(border=True):
                                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                                # Density heatmap uses the time-filtered DataFrame
                                heatmap_fig = plot_worker_density_heatmap(
                                    filtered_df_dist, # Use the time and zone filtered data
                                    DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG,
                                    show_entry_exit=show_entry_exit, show_production_lines=show_production_lines
                                )
                                st.plotly_chart(heatmap_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            logger.error(f"Failed to plot density heatmap: {str(e)}", extra={'user_action': 'Render Worker Insights'})
                            st.error(f"Error rendering density heatmap: {str(e)}. Check debug mode for details.")

                with st.expander("Worker Well-Being & Safety"):
                    time_range_well = st.slider(
                        "Time Range (minutes) for Well-Being", # Unique label
                        min_value=0,
                        max_value=max(0, current_max_minutes_for_sliders),
                        value=(0, max(0, current_max_minutes_for_sliders)),
                        step=2,
                        key="time_range_well"
                    )
                    time_indices_well = (time_range_well[0] // 2, time_range_well[1] // 2 + 1)
                    
                    col_well1, col_well2 = st.columns(2)
                    with col_well1:
                        st.markdown("### Well-Being Index")
                        filtered_scores = worker_wellbeing_data['scores'][time_indices_well[0]:time_indices_well[1]]
                        # Filter triggers based on time_indices_well (steps)
                        ww_triggers = worker_wellbeing_data['triggers']
                        filtered_triggers = {
                            'threshold': [t for t in ww_triggers.get('threshold', []) if time_indices_well[0] <= t < time_indices_well[1]],
                            'trend': [t for t in ww_triggers.get('trend', []) if time_indices_well[0] <= t < time_indices_well[1]],
                            'work_area': {
                                k: [t for t in v if time_indices_well[0] <= t < time_indices_well[1]] 
                                for k, v in ww_triggers.get('work_area', {}).items()
                            },
                            'disruption': [t for t in ww_triggers.get('disruption', []) if time_indices_well[0] <= t < time_indices_well[1]]
                        }
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                            wellbeing_fig = plot_worker_wellbeing(filtered_scores, filtered_triggers)
                            st.plotly_chart(wellbeing_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                    with col_well2:
                        st.markdown("### Psychological Safety")
                        filtered_safety = psychological_safety_data[time_indices_well[0]:time_indices_well[1]]
                        with st.container(border=True):
                            st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                            safety_fig = plot_psychological_safety(filtered_safety)
                            st.plotly_chart(safety_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("### Well-Being Triggers (within selected time range)")
                    st.write(f"**Threshold Alerts (< {DEFAULT_CONFIG.get('WELLBEING_THRESHOLD',0)*100}%):** {filtered_triggers['threshold']}")
                    st.write(f"**Trend Alerts (Declining):** {filtered_triggers['trend']}")
                    st.write("**Work Area Alerts:**")
                    for zone, triggers in filtered_triggers['work_area'].items():
                        if triggers: st.write(f"  {zone}: {triggers}")
                    st.write(f"**Disruption Alerts:** {filtered_triggers['disruption']}")
            else:
                st.info("Run a simulation or load data to view insights.", icon="ℹ️")

    # Downtime Tab
    with tabs[3]:
        with st.container():
            st.header("Downtime Analysis", divider="grey")
            st.markdown(
                '<div class="tooltip">Downtime Trends<span class="tooltiptext">Downtime with alerts for high values.</span></div>',
                unsafe_allow_html=True
            )
            if st.session_state.simulation_results:
                sim_data = st.session_state.simulation_results
                downtime_minutes_all = sim_data['downtime_minutes']
                time_range_down = st.slider(
                    "Time Range (minutes) for Downtime", # Unique label
                    min_value=0,
                    max_value=max(0, current_max_minutes_for_sliders),
                    value=(0, max(0, current_max_minutes_for_sliders)),
                    step=2,
                    key="time_range_down"
                )
                time_indices_down = (time_range_down[0] // 2, time_range_down[1] // 2 + 1)
                filtered_downtime = downtime_minutes_all[time_indices_down[0]:time_indices_down[1]]
                with st.container(border=True):
                    st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                    downtime_fig = plot_downtime_trend(filtered_downtime, DEFAULT_CONFIG.get('DOWNTIME_THRESHOLD', 10)) # Provide default for threshold
                    st.plotly_chart(downtime_fig, use_container_width=True, config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Run a simulation or load data to view analysis.", icon="ℹ️")

    # Glossary Tab
    with tabs[4]:
        with st.container():
            st.header("Glossary", divider="grey")
            st.markdown("""
                ### Metric Definitions
                - <b>Task Compliance Score</b>: Percentage of tasks completed correctly and on time (0–100%). Measures adherence to operational protocols.
                - <b>Collaboration Proximity Index</b>: Percentage of workers within 5 meters of colleagues (0–100%). Indicates teamwork and communication opportunities.
                - <b>Operational Recovery Score</b>: Ability to maintain output after disruptions (0–100%). Reflects resilience to unexpected events.
                - <b>Worker Well-Being Index</b>: Composite score of fatigue, stress, and satisfaction (0–100%). Tracks worker health and morale.
                - <b>Psychological Safety Score</b>: Comfort level in reporting issues or suggesting improvements (0–100%). Indicates a supportive work environment.
                - <b>Uptime</b>: Percentage of time equipment is operational (0–100%). Measures equipment reliability.
                - <b>Throughput</b>: Percentage of maximum production rate achieved (0–100%). Indicates production efficiency.
                - <b>Quality</b>: Percentage of products meeting quality standards (0–100%). Reflects output consistency.
                - <b>OEE (Overall Equipment Effectiveness)</b>: Combined score of uptime, throughput, and quality (0–100%). Holistic measure of operational performance.
                - <b>Productivity Loss</b>: Percentage of potential output lost due to inefficiencies or disruptions (0–100%).
                - <b>Downtime</b>: Total minutes of unplanned operational stops per interval. Tracks interruptions to workflow.
                - <b>Task Completion Rate</b>: Percentage of tasks completed per time interval (0–100%). Measures task efficiency over time.
            """, unsafe_allow_html=True) # Added unsafe_allow_html for <b> tags

if __name__ == "__main__":
    main()
