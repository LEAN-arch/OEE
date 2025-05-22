"""
main.py
Streamlit dashboard for the Industrial Workplace Shift Monitoring Dashboard.
Provides an interactive UI for simulation results with tabs and controls.
"""

import logging
import streamlit as st
import pandas as pd
from simulation import simulate_workplace_operations
from visualizations import (
    plot_task_compliance_trend,
    plot_worker_collaboration_trend,
    plot_operational_resilience,
    plot_operational_efficiency,
    plot_oee_gauge,
    plot_worker_distribution,
    plot_worker_wellbeing,
    plot_psychological_safety
)
from utils import save_simulation_data, logger
from config import DEFAULT_CONFIG

# Streamlit page config
st.set_page_config(
    page_title="Workplace Shift Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and accessibility
st.markdown("""
    <style>
        .main { 
            background-color: #1E2A44; 
            color: #E6E9F0; 
        }
        .stButton>button { 
            background-color: #4C78A8; 
            color: #E6E9F0; 
            border-radius: 8px; 
            border: 1px solid #E6E9F0; 
        }
        .stButton>button:hover { 
            background-color: #F58518; 
        }
        .stSelectbox, .stSlider { 
            background-color: #2A3B5A; 
            color: #E6E9F0; 
            border-radius: 8px; 
        }
        h1, h2, h3 { 
            color: #E6E9F0; 
            font-weight: 600; 
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #2A3B5A;
            color: #E6E9F0;
            text-align: center;
            border-radius: 8px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stSidebar"] { 
            background-color: #2A3B5A; 
            color: #E6E9F0; 
        }
        [data-testid="stSidebar"] .stButton>button { 
            background-color: #54A24B; 
        }
        [data-testid="stSidebar"] .stButton>button:hover { 
            background-color: #E45756; 
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for simulation controls
with st.sidebar:
    st.header("Simulation Controls")
    
    team_size = st.slider(
        "Team Size",
        min_value=10, max_value=100, value=DEFAULT_CONFIG['TEAM_SIZE'],
        help="Number of team members in the simulation."
    )
    
    shift_duration = st.slider(
        "Shift Duration (intervals)",
        min_value=100, max_value=1000, value=DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'],
        help="Number of 2-minute intervals in the shift."
    )
    
    disruption_intervals = st.multiselect(
        "Disruption Intervals",
        options=list(range(shift_duration)),
        default=DEFAULT_CONFIG['DISRUPTION_INTERVALS'],
        help="Time steps where disruptions occur."
    )
    
    team_initiative = st.selectbox(
        "Team Initiative",
        options=["More frequent breaks", "Team recognition"],
        index=0,
        help="Strategy to improve well-being and safety."
    )
    
    run_simulation = st.button("Run Simulation", key="run_simulation")

    # Accessibility toggle
    high_contrast = st.checkbox("High Contrast Mode", help="Enable high-contrast colors for better visibility.")

    # Help modal
    if st.button("Help", key="help_button"):
        st.markdown("""
            ### Help
            This dashboard monitors workplace shift performance:
            - **Operational Efficiency**: Tracks uptime, throughput, quality, and OEE.
            - **Team Distribution**: Shows worker positions by zone.
            - **Well-Being & Safety**: Monitors team well-being and psychological safety.
            - **Compliance & Collaboration**: Analyzes task compliance and worker collaboration.
            - **Resilience**: Measures recovery from disruptions.
            
            Use the sidebar to adjust simulation parameters and run new simulations.
            Contact support@xai.com for assistance.
        """, unsafe_allow_html=True)

# Main content
st.title(f"{DEFAULT_CONFIG['FACILITY_TYPE'].capitalize()} Workplace Shift Monitoring Dashboard")

# Initialize session state for simulation results
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Run simulation if button clicked
if run_simulation:
    try:
        config = DEFAULT_CONFIG.copy()
        config['TEAM_SIZE'] = team_size
        config['SHIFT_DURATION_INTERVALS'] = shift_duration
        config['DISRUPTION_INTERVALS'] = disruption_intervals
        
        logger.info("Running simulation with team_size=%d, shift_duration=%d", team_size, shift_duration)
        simulation_results = simulate_workplace_operations(
            num_team_members=team_size,
            num_steps=shift_duration,
            disruption_intervals=disruption_intervals,
            team_initiative=team_initiative,
            config=config
        )
        
        st.session_state.simulation_results = simulation_results
        logger.info("Calling save_simulation_data")
        try:
            save_simulation_data(*simulation_results)
        except NameError as ne:
            logger.error(f"NameError in save_simulation_data call: {str(ne)}")
            raise
        st.success("Simulation completed successfully!")
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        st.error(f"Simulation failed: {str(e)}")

# Display results if available
if st.session_state.simulation_results:
    (team_positions_df, compliance_variability, collaboration_index, operational_resilience,
     efficiency_metrics_df, productivity_loss, team_wellbeing, safety, feedback_impact) = st.session_state.simulation_results

    # Tabbed navigation
    tabs = st.tabs([
        "Overview",
        "Efficiency",
        "Team Distribution",
        "Well-Being",
        "Safety",
        "Compliance",
        "Collaboration",
        "Resilience"
    ])

    with tabs[0]:
        st.header("Dashboard Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="tooltip">OEE Gauge<span class="tooltiptext">Overall Equipment Effectiveness measures production efficiency.</span></div>', unsafe_allow_html=True)
            oee_fig = plot_oee_gauge(efficiency_metrics_df['oee'].mean())
            st.plotly_chart(oee_fig, use_container_width=True)
        with col2:
            st.markdown('<div class="tooltip">Key Metrics<span class="tooltiptext">Summary of critical performance indicators.</span></div>', unsafe_allow_html=True)
            st.metric("Average Well-Being", f"{team_wellbeing['scores'].mean():.2f}", delta_color="normal")
            st.metric("Average Safety", f"{safety.mean():.2f}", delta_color="normal")
            st.metric("Productivity Loss", f"{productivity_loss.sum():.1f}%", delta_color="inverse")

    with tabs[1]:
        st.header("Operational Efficiency")
        efficiency_fig = plot_operational_efficiency(efficiency_metrics_df)
        st.plotly_chart(efficiency_fig, use_container_width=True)

    with tabs[2]:
        st.header("Team Distribution")
        zone_filter = st.selectbox("Filter by Zone", options=["All"] + list(DEFAULT_CONFIG['WORK_AREAS'].keys()))
        filtered_df = team_positions_df if zone_filter == "All" else team_positions_df[team_positions_df['zone'] == zone_filter]
        distribution_fig = plot_worker_distribution(filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG)
        st.plotly_chart(distribution_fig, use_container_width=True)

    with tabs[3]:
        st.header("Team Well-Being")
        wellbeing_fig = plot_worker_wellbeing(team_wellbeing['scores'])
        st.plotly_chart(wellbeing_fig, use_container_width=True)
        st.subheader("Well-Being Triggers")
        st.write(f"**Threshold Alerts**: {team_wellbeing['triggers']['threshold']}")
        st.write(f"**Trend Alerts**: {team_wellbeing['triggers']['trend']}")
        st.write("**Work Area Alerts**:")
        for zone, triggers in team_wellbeing['triggers']['work_area'].items():
            st.write(f"{zone}: {triggers}")
        st.write(f"**Disruption Alerts**: {team_wellbeing['triggers']['disruption']}")

    with tabs[4]:
        st.header("Psychological Safety")
        safety_fig = plot_psychological_safety(safety)
        st.plotly_chart(safety_fig, use_container_width=True)

    with tabs[5]:
        st.header("Task Compliance Variability")
        compliance_fig = plot_task_compliance_trend(compliance_variability['data'], DEFAULT_CONFIG['DISRUPTION_INTERVALS'], compliance_variability['forecast'])
        st.plotly_chart(compliance_fig, use_container_width=True)

    with tabs[6]:
        st.header("Worker Collaboration Index")
        collaboration_fig = plot_worker_collaboration_trend(collaboration_index['data'], collaboration_index['forecast'])
        st.plotly_chart(collaboration_fig, use_container_width=True)

    with tabs[7]:
        st.header("Operational Resilience")
        resilience_fig = plot_operational_resilience(operational_resilience)
        st.plotly_chart(resilience_fig, use_container_width=True)

else:
    st.info("Run a simulation using the sidebar controls to view results.")

# High-contrast mode
if high_contrast:
    st.markdown("""
        <style>
            .main { 
                background-color: #000000; 
                color: #FFFFFF; 
            }
            h1, h2, h3 { 
                color: #FFFFFF; 
            }
            .stButton>button { 
                background-color: #FFFFFF; 
                color: #000000; 
                border: 1px solid #FFFFFF; 
            }
            .stButton>button:hover { 
                background-color: #F1C40F; 
                color: #000000; 
            }
            .stSelectbox, .stSlider { 
                background-color: #333333; 
                color: #FFFFFF; 
            }
            [data-testid="stSidebar"] { 
                background-color: #111111; 
                color: #FFFFFF; 
            }
        </style>
    """, unsafe_allow_html=True)
