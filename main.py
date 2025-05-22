"""
main.py
Streamlit dashboard for the Industrial Workplace Shift Monitoring Dashboard.
Provides an interactive UI with advanced visualizations and controls.
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
    plot_worker_density_heatmap,
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

# Custom CSS for refined dark theme
st.markdown("""
    <style>
        .main { 
            background-color: #1E2A44; 
            color: #F5F6F5; 
        }
        .stButton>button { 
            background-color: #636EFA; 
            color: #F5F6F5; 
            border-radius: 10px; 
            border: 2px solid #F5F6F5; 
            padding: 8px 16px; 
            transition: background-color 0.3s; 
        }
        .stButton>button:hover { 
            background-color: #EF553B; 
            border-color: #EF553B; 
        }
        .stSelectbox, .stSlider, .stMultiSelect { 
            background-color: #2A3B5A; 
            color: #F5F6F5; 
            border-radius: 10px; 
            padding: 5px; 
        }
        h1, h2, h3 { 
            color: #F5F6F5; 
            font-weight: 700; 
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 220px;
            background-color: #2A3B5A;
            color: #F5F6F5;
            text-align: center;
            border-radius: 10px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        [data-testid="stSidebar"] { 
            background-color: #2A3B5A; 
            color: #F5F6F5; 
        }
        [data-testid="stSidebar"] .stButton>button { 
            background-color: #00CC96; 
            border-color: #F5F6F5; 
        }
        [data-testid="stSidebar"] .stButton>button:hover { 
            background-color: #F15C80; 
            border-color: #F15C80; 
        }
        .stMetric { 
            background-color: #2A3B5A; 
            border-radius: 10px; 
            padding: 10px; 
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

    # Accessibility and visualization toggles
    high_contrast = st.checkbox("High Contrast Mode", help="Enable high-contrast colors for accessibility.")
    use_3d_distribution = st.checkbox("Use 3D Team Distribution", help="Toggle 3D scatter plot for team distribution.")

    # Help modal
    if st.button("Help", key="help_button"):
        st.markdown("""
            ### Help
            This dashboard monitors workplace shift performance with advanced visualizations:
            - **Operational Efficiency**: Composite charts for uptime, throughput, quality, OEE.
            - **Team Distribution**: 2D/3D scatter or density heatmap by zone.
            - **Well-Being & Safety**: Trend lines and anomaly detection.
            - **Compliance & Collaboration**: Rolling averages, forecasts, and disruption overlays.
            - **Resilience**: Dual-axis plots with productivity loss.
            
            Use the sidebar to adjust parameters, toggle 3D views, and run simulations.
            Contact support@xai.com for assistance.
        """, unsafe_allow_html=True)

# Main content
st.title(f"{DEFAULT_CONFIG['FACILITY_TYPE'].capitalize()} Workplace Shift Monitoring Dashboard")

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Run simulation
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
        save_simulation_data(*simulation_results)
        st.success("Simulation completed successfully!")
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        st.error(f"Simulation failed: {str(e)}")

# Display results
if st.session_state.simulation_results:
    (team_positions_df, compliance_variability, collaboration_index, operational_resilience,
     efficiency_metrics_df, productivity_loss, team_wellbeing, safety, feedback_impact) = st.session_state.simulation_results

    # Interactive controls
    time_range = st.slider(
        "Select Time Range",
        min_value=0, max_value=DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']-1,
        value=(0, DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']-1),
        key="time_range"
    )
    
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
            st.markdown('<div class="tooltip">OEE Gauge<span class="tooltiptext">Compares OEE to industry benchmark (85%).</span></div>', unsafe_allow_html=True)
            oee_fig = plot_oee_gauge(efficiency_metrics_df['oee'].mean())
            st.plotly_chart(oee_fig, use_container_width=True)
        with col2:
            st.markdown('<div class="tooltip">Key Metrics<span class="tooltiptext">Summary of critical performance indicators.</span></div>', unsafe_allow_html=True)
            st.metric("Average Well-Being", f"{team_wellbeing['scores'].mean():.2f}", delta=f"{team_wellbeing['scores'].mean() - DEFAULT_CONFIG['WELLBEING_THRESHOLD']:.2f}")
            st.metric("Average Safety", f"{safety.mean():.2f}", delta=f"{safety.mean() - DEFAULT_CONFIG['SAFETY_COMPLIANCE_THRESHOLD']:.2f}")
            st.metric("Total Productivity Loss", f"{productivity_loss.sum():.1f}%", delta_color="inverse")

    with tabs[1]:
        st.header("Operational Efficiency")
        selected_metrics = st.multiselect(
            "Select Metrics",
            options=['uptime', 'throughput', 'quality', 'oee'],
            default=['uptime', 'throughput', 'quality', 'oee'],
            key="efficiency_metrics"
        )
        filtered_df = efficiency_metrics_df.iloc[time_range[0]:time_range[1]+1]
        efficiency_fig = plot_operational_efficiency(filtered_df, selected_metrics)
        st.plotly_chart(efficiency_fig, use_container_width=True)

    with tabs[2]:
        st.header("Team Distribution")
        zone_filter = st.selectbox("Filter by Zone", options=["All"] + list(DEFAULT_CONFIG['WORK_AREAS'].keys()))
        filtered_df = team_positions_df if zone_filter == "All" else team_positions_df[team_positions_df['zone'] == zone_filter]
        filtered_df = filtered_df[(filtered_df['step'] >= time_range[0]) & (filtered_df['step'] <= time_range[1])]
        
        if st.checkbox("Show Density Heatmap"):
            heatmap_fig = plot_worker_density_heatmap(filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            distribution_fig = plot_worker_distribution(filtered_df, DEFAULT_CONFIG['FACILITY_SIZE'], DEFAULT_CONFIG, use_3d=use_3d_distribution)
            st.plotly_chart(distribution_fig, use_container_width=True)

    with tabs[3]:
        st.header("Team Well-Being")
        filtered_scores = team_wellbeing['scores'][time_range[0]:time_range[1]+1]
        filtered_triggers = {
            'threshold': [t for t in team_wellbeing['triggers']['threshold'] if time_range[0] <= t <= time_range[1]],
            'trend': [t for t in team_wellbeing['triggers']['trend'] if time_range[0] <= t <= time_range[1]],
            'work_area': {k: [t for t in v if time_range[0] <= t <= time_range[1]] for k, v in team_wellbeing['triggers']['work_area'].items()},
            'disruption': [t for t in team_wellbeing['triggers']['disruption'] if time_range[0] <= t <= time_range[1]]
        }
        wellbeing_fig = plot_worker_wellbeing(filtered_scores, filtered_triggers)
        st.plotly_chart(wellbeing_fig, use_container_width=True)
        st.subheader("Well-Being Triggers")
        st.write(f"**Threshold Alerts**: {filtered_triggers['threshold']}")
        st.write(f"**Trend Alerts**: {filtered_triggers['trend']}")
        st.write("**Work Area Alerts**:")
        for zone, triggers in filtered_triggers['work_area'].items():
            st.write(f"{zone}: {triggers}")
        st.write(f"**Disruption Alerts**: {filtered_triggers['disruption']}")

    with tabs[4]:
        st.header("Psychological Safety")
        filtered_safety = safety[time_range[0]:time_range[1]+1]
        safety_fig = plot_psychological_safety(filtered_safety)
        st.plotly_chart(safety_fig, use_container_width=True)

    with tabs[5]:
        st.header("Task Compliance Variability")
        filtered_compliance = compliance_variability['data'][time_range[0]:time_range[1]+1]
        filtered_z_scores = compliance_variability['z_scores'][time_range[0]:time_range[1]+1]
        filtered_forecast = compliance_variability['forecast'][time_range[0]:time_range[1]+1] if compliance_variability['forecast'] is not None else None
        filtered_disruptions = [t for t in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if time_range[0] <= t <= time_range[1]]
        compliance_fig = plot_task_compliance_trend(filtered_compliance, filtered_disruptions, filtered_forecast, filtered_z_scores)
        st.plotly_chart(compliance_fig, use_container_width=True)

    with tabs[6]:
        st.header("Worker Collaboration Index")
        filtered_collab = collaboration_index['data'][time_range[0]:time_range[1]+1]
        filtered_forecast = collaboration_index['forecast'][time_range[0]:time_range[1]+1] if collaboration_index['forecast'] is not None else None
        filtered_disruptions = [t for t in DEFAULT_CONFIG['DISRUPTION_INTERVALS'] if time_range[0] <= t <= time_range[1]]
        collaboration_fig = plot_worker_collaboration_trend(filtered_collab, filtered_disruptions, filtered_forecast)
        st.plotly_chart(collaboration_fig, use_container_width=True)

    with tabs[7]:
        st.header("Operational Resilience")
        filtered_resilience = operational_resilience[time_range[0]:time_range[1]+1]
        filtered_loss = productivity_loss[time_range[0]:time_range[1]+1]
        resilience_fig = plot_operational_resilience(filtered_resilience, filtered_loss)
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
                border: 2px solid #FFFFFF; 
            }
            .stButton>button:hover { 
                background-color: #19D3F3; 
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
            .stMetric { 
                background-color: #333333; 
            }
        </style>
    """, unsafe_allow_html=True)
