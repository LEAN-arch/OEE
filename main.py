"""
main.py
Streamlit dashboard for the Workplace Shift Monitoring Dashboard.
Refactored for multi-page navigation with improved display quality and no overlapping content.
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
from config import DEFAULT_CONFIG, validate_config
from visualizations import (
    plot_key_metrics_summary
)
from simulation import simulate_workplace_operations
from utils import save_simulation_data, load_simulation_data, generate_pdf_report
from assets import LEAN_LOGO_BASE64

# Configure logging with more detailed context
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log'
)

# Streamlit page config
st.set_page_config(
    page_title="Workplace Shift Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UX and display quality
st.markdown("""
    <style>
        /* Base Styles */
        .main { 
            background-color: #1A252F; 
            color: #E6ECEF; 
            font-family: 'Arial', sans-serif;
            padding: 20px;
        }
        h1 { 
            color: #E6ECEF; 
            font-size: 2.5rem; 
            margin-bottom: 1.5rem; 
        }
        h2 { 
            color: #E6ECEF; 
            font-size: 1.8rem; 
            margin-top: 1.5rem; 
            margin-bottom: 1rem; 
        }
        .stButton>button { 
            background-color: #3B82F6; 
            color: #E6ECEF; 
            border-radius: 8px; 
            border: 2px solid #E6ECEF; 
            padding: 8px 16px; 
            transition: background-color 0.3s; 
        }
        .stButton>button:hover, .stButton>button:focus { 
            background-color: #EC4899; 
            border-color: #EC4899; 
            outline: none; 
        }
        .stSelectbox, .stSlider, .stMultiSelect { 
            background-color: #2D3748; 
            color: #E6ECEF; 
            border-radius: 8px; 
            padding: 5px; 
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 250px;
            background-color: #2D3748;
            color: #E6ECEF;
            text-align: center;
            border-radius: 8px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext, .tooltip:focus .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stSidebar"] { 
            background-color: #2D3748; 
            color: #E6ECEF; 
            padding: 10px;
        }
        [data-testid="stSidebar"] .stButton>button { 
            background-color: #10B981; 
            border-color: #E6ECEF; 
        }
        [data-testid="stSidebar"] .stButton>button:hover, [data-testid="stSidebar"] .stButton>button:focus { 
            background-color: #EC4899; 
            border-color: #EC4899; 
        }
        .stMetric { 
            background-color: #2D3748; 
            border-radius: 8px; 
            padding: 10px; 
            margin: 10px 0;
        }
        .stExpander { 
            background-color: #2D3748; 
            border-radius: 8px; 
            margin: 10px 0;
        }
        .recommendation {
            color: #FBBF24;
            font-size: 14px;
            margin-top: 5px;
        }
        /* Responsive Design */
        @media (max-width: 768px) {
            .stColumn {
                width: 100% !important;
                margin-bottom: 1rem;
            }
            .stPlotlyChart {
                height: 400px !important;
            }
            h1 { font-size: 2rem; }
            h2 { font-size: 1.5rem; }
        }
        /* Loading Spinner */
        .spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }
        .spinner::after {
            content: '';
            width: 40px;
            height: 40px;
            border: 4px solid #E6ECEF;
            border-top: 4px solid #3B82F6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# Function to display a loading spinner
def display_loading(message):
    with st.container():
        st.markdown(f'<div class="spinner"></div><p style="text-align: center; color: #E6ECEF;">{message}</p>', unsafe_allow_html=True)

# Sidebar for settings
def render_settings_sidebar():
    with st.sidebar:
        st.header("Dashboard Settings")
        
        # Company logo with accessibility
        st.markdown(
            f'<img src="{LEAN_LOGO_BASE64}" width="150" alt="Lean 2.0 Institute Logo" aria-label="Lean 2.0 Institute Logo">',
            unsafe_allow_html=True
        )
        
        with st.expander("Simulation Controls", expanded=True):
            team_size = st.slider(
                "Team Size",
                min_value=10, max_value=100, value=DEFAULT_CONFIG['TEAM_SIZE'],
                help="Number of workers in the simulation.",
                key="team_size"
            )
            
            shift_duration = st.slider(
                "Shift Duration (minutes)",
                min_value=200, max_value=2000, value=DEFAULT_CONFIG['SHIFT_DURATION_MINUTES'], step=2,
                help="Shift duration in minutes (2-minute intervals).",
                key="shift_duration"
            )
            
            disruption_intervals = st.multiselect(
                "Disruption Times (minutes)",
                options=[i * 2 for i in range(shift_duration // 2)],
                default=[i * 2 for i in DEFAULT_CONFIG['DISRUPTION_INTERVALS']],
                help="Times (minutes) when disruptions occur.",
                key="disruption_intervals"
            )
            
            team_initiative = st.selectbox(
                "Team Initiative",
                options=["More frequent breaks", "Team recognition"],
                index=0,
                help="Strategy to improve well-being and psychological safety.",
                key="team_initiative"
            )
            
            run_simulation = st.button(
                "Run Simulation", 
                key="run_simulation", 
                help="Start a new simulation with the selected parameters.",
                type="primary"
            )
        
        with st.expander("Visualization Settings"):
            high_contrast = st.checkbox(
                "High Contrast Mode", 
                help="Enable high-contrast colors for accessibility.",
                key="high_contrast"
            )
            use_3d_distribution = st.checkbox(
                "3D Team Distribution", 
                help="Use 3D scatter plot with time slider for team distribution.",
                key="use_3d_distribution"
            )
        
        with st.expander("Data Management"):
            load_data = st.button(
                "Load Saved Data", 
                key="load_data", 
                help="Load previously saved simulation data."
            )
            if st.button(
                "Download PDF Report", 
                key="download_report", 
                help="Generate and download a PDF report of the current data."
            ) and 'simulation_results' in st.session_state:
                try:
                    summary_df = pd.DataFrame({
                        'step': range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']),
                        'time_minutes': [i * 2 for i in range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'])],
                        'task_compliance': st.session_state.simulation_results[1]['data'],
                        'collaboration_proximity': st.session_state.simulation_results[2]['data'],
                        'operational_recovery': st.session_state.simulation_results[3],
                        'worker_wellbeing': st.session_state.simulation_results[6]['scores'],
                        'psychological_safety': st.session_state.simulation_results[7],
                        'productivity_loss': st.session_state.simulation_results[5],
                        'downtime_minutes': st.session_state.simulation_results[9],
                        'task_completion_rate': st.session_state.simulation_results[10]
                    })
                    generate_pdf_report(summary_df)
                    st.success("PDF report generated as 'workplace_report.tex'. Compile with LaTeX to view.")
                except Exception as e:
                    logger.error(f"Failed to generate report: {str(e)}", extra={'user_action': 'Download PDF Report'})
                    st.error(f"Failed to generate report: {str(e)}. See dashboard.log for details.")
        
        # First-Time User Tour
        if st.button(
            "Take a Tour", 
            key="tour_button", 
            help="Take a guided tour of the dashboard features."
        ):
            st.session_state.show_tour = True
        
        # Help Section
        if st.button(
            "Help", 
            key="help_button", 
            help="View detailed help and documentation."
        ):
            st.markdown("""
                ### Help
                Navigate through different pages to monitor workplace performance:
                - **Key Metrics Summary**: Gauge charts for key metrics.
                - **Operational Metrics**: Trends for performance metrics.
                - **Worker Distribution**: Worker positions and density heatmap.
                - **Worker Well-Being & Safety**: Well-Being and Safety scores.
                - **Downtime Analysis**: Downtime trends.
                - **Glossary**: Definitions of all metrics and terms.
                
                Use the sidebar to adjust parameters, load saved data, or download a PDF report.
                Contact support@xai.com for assistance.
            """, unsafe_allow_html=True)

    return team_size, shift_duration, disruption_intervals, team_initiative, run_simulation, load_data, high_contrast, use_3d_distribution

# Function to handle simulation and data loading
@st.cache_data
def run_simulation_logic(team_size, shift_duration, disruption_intervals, team_initiative):
    config = DEFAULT_CONFIG.copy()
    config['TEAM_SIZE'] = team_size
    config['SHIFT_DURATION_MINUTES'] = shift_duration
    config['SHIFT_DURATION_INTERVALS'] = shift_duration // 2
    config['DISRUPTION_INTERVALS'] = [t // 2 for t in disruption_intervals]
    
    # Update WORK_AREAS worker counts to match new TEAM_SIZE
    total_current_workers = sum(zone['workers'] for zone in config['WORK_AREAS'].values())
    if total_current_workers != team_size:
        ratio = team_size / total_current_workers
        for zone in config['WORK_AREAS'].values():
            zone['workers'] = int(zone['workers'] * ratio)
        current_sum = sum(zone['workers'] for zone in config['WORK_AREAS'].values())
        if current_sum != team_size:
            diff = team_size - current_sum
            config['WORK_AREAS']['Assembly Line']['workers'] += diff
    
    validate_config(config)
    logger.info(
        f"Running simulation with team_size={team_size}, shift_duration={shift_duration} min",
        extra={'user_action': 'Run Simulation'}
    )
    simulation_results = simulate_workplace_operations(
        num_team_members=team_size,
        num_steps=shift_duration // 2,
        disruption_intervals=[t // 2 for t in disruption_intervals],
        team_initiative=team_initiative,
        config=config
    )
    save_simulation_data(*simulation_results)
    return simulation_results

# Main content
def main():
    st.title("Workplace Shift Monitoring Dashboard")

    # Initialize session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'show_tour' not in st.session_state:
        st.session_state.show_tour = False

    # Render settings sidebar and get user inputs
    team_size, shift_duration, disruption_intervals, team_initiative, run_simulation, load_data, high_contrast, use_3d_distribution = render_settings_sidebar()

    # Precompute minutes for efficiency
    if st.session_state.simulation_results:
        num_steps = len(st.session_state.simulation_results[0]['step'].unique())
        minutes = [i * 2 for i in range(num_steps)]
    else:
        minutes = [i * 2 for i in range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'])]

    # Handle simulation and data loading
    if run_simulation:
        with st.spinner("Running simulation..."):
            try:
                st.session_state.simulation_results = run_simulation_logic(
                    team_size, shift_duration, disruption_intervals, team_initiative
                )
                st.success("Simulation completed successfully! Navigate to pages to view results.")
            except Exception as e:
                logger.error(f"Simulation failed: {str(e)}", extra={'user_action': 'Run Simulation'})
                st.error(f"Simulation failed: {str(e)}. Check dashboard.log for details.")

    if load_data:
        with st.spinner("Loading saved data..."):
            try:
                st.session_state.simulation_results = load_simulation_data()
                st.success("Loaded saved simulation data successfully!")
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}", extra={'user_action': 'Load Data'})
                st.error(f"Failed to load data: {str(e)}. Check dashboard.log for details.")

    # Display guided tour
    if st.session_state.show_tour:
        st.markdown("""
            <script>
                alert("Welcome to the Workplace Shift Monitoring Dashboard Tour!\\n\\n" +
                      "1. **Settings Sidebar**: Adjust simulation parameters here.\\n" +
                      "2. **Navigation**: Use the sidebar to switch between pages (Key Metrics, Operational Metrics, etc.).\\n" +
                      "3. **Interactive Charts**: Hover over charts for details and use sliders to filter data.\\n\\n" +
                      "Click OK to start exploring!");
            </script>
        """, unsafe_allow_html=True)
        st.session_state.show_tour = False

    # Display a welcome message and navigation hint
    st.write("Welcome to the dashboard! Use the sidebar to navigate to different sections or run a simulation.")

# High-contrast mode with WCAG compliance
def apply_high_contrast_mode():
    st.markdown("""
        <style>
            .main { 
                background-color: #000000; 
                color: #FFFFFF; 
            }
            h1, h2 { 
                color: #FFFFFF; 
            }
            .stButton>button { 
                background-color: #FFFFFF; 
                color: #000000; 
                border: 2px solid #FFFFFF; 
            }
            .stButton>button:hover, .stButton>button:focus { 
                background-color: #FFFF00; 
                color: #000000; 
            }
            .stSelectbox, .stSlider, .stMultiSelect { 
                background-color: #333333; 
                color: #FFFFFF; 
            }
            [data-testid="stSidebar"] { 
                background-color: #111111; 
                color: #FFFFFF; 
            }
            .stMetric, .stExpander { 
                background-color: #333333; 
            }
            .recommendation {
                color: #FFFF00;
            }
            .tooltip .tooltiptext {
                background-color: #333333;
                color: #FFFFFF;
            }
        </style>
    """, unsafe_allow_html=True)

# Developer Guide
def developer_guide():
    """
    ### Developer Guide
    This dashboard is built with Streamlit and Plotly for interactive visualizations. Key components:

    - **Structure**:
      - `main.py`: Handles the main app and sidebar settings.
      - `pages/`: Directory containing page-specific files (e.g., `key_metrics.py`).
      - `render_settings_sidebar()`: Manages sidebar controls.
      - `run_simulation_logic()`: Executes the simulation with caching.
      - `apply_high_contrast_mode()`: Applies WCAG-compliant high-contrast styling.

    - **Key Files**:
      - `visualizations.py`: Contains all Plotly chart functions.
      - `simulation.py`: Handles simulation logic.
      - `utils.py`: Utilities for saving/loading data and generating PDF reports.
      - `config.py`: Configuration settings.

    - **Extending the Dashboard**:
      1. Add new pages in the `pages` directory (e.g., `pages/new_page.py`).
      2. Update `DEFAULT_CONFIG` in `config.py` for new parameters.
      3. Enhance accessibility with ARIA labels.
      4. Use the `logger` for debugging; logs are saved to `dashboard.log`.

    - **Performance Tips**:
      - Use `@st.cache_data` for expensive computations.
      - Minimize re-renders with session state.
      - Optimize Plotly charts with efficient layouts.

    - **Testing**:
      - Run simulations with various parameters.
      - Test high-contrast mode and responsiveness.
      - Check `dashboard.log` for errors (last error at 09:32 PM PDT; current time is 09:35 PM PDT).
    """

if __name__ == "__main__":
    main()
    if st.session_state.get('high_contrast', False):
        apply_high_contrast_mode()
    developer_guide()
