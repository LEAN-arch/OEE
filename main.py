try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px
    import plotly.figure_factory as ff
    import sys
    import logging
    from config import CONFIG
    from industrial_workplace_simulation import (
        simulate_workplace_operations,
        plot_task_compliance_trend,
        plot_worker_collaboration_trend,
        plot_operational_resilience,
        plot_operational_efficiency,
        plot_worker_distribution,
        plot_worker_wellbeing,
        plot_psychological_safety
    )
except ImportError as e:
    st.error(f"Failed to import libraries: {str(e)}. Ensure 'industrial_workplace_simulation.py' and 'config.py' are in the same directory.")
    st.error("Run 'pip install -r requirements.txt' and verify Python 3.10.")
    st.stop()

# Configure logging for operational diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_config(config):
    """Validate configuration parameters for the dashboard."""
    try:
        if config['NUM_TEAM_MEMBERS'] < 1 or config['NUM_STEPS'] < 1 or config['WORKPLACE_SIZE'] <= 0:
            raise ValueError("Invalid config: NUM_TEAM_MEMBERS, NUM_STEPS, and WORKPLACE_SIZE must be positive.")
        if not config['WORK_AREAS'] or not all(
            "center" in ws and "label" in ws for ws in config['WORK_AREAS'].values()
        ):
            raise ValueError("Invalid WORK_AREAS: Must define center coordinates and labels.")
        for name, ws in config['WORK_AREAS'].items():
            x, y = ws["center"]
            if not (0 <= x <= config['WORKPLACE_SIZE'] and 0 <= y <= config['WORKPLACE_SIZE']):
                raise ValueError(
                    f"Invalid center for {name}: Coordinates must be within [0, {config['WORKPLACE_SIZE']}]."
                )
        if config['WELLBEING_THRESHOLD'] < 0 or config['SAFETY_THRESHOLD'] < 0:
            raise ValueError("WELLBEING_THRESHOLD and SAFETY_THRESHOLD must be non-negative.")
    except ValueError as e:
        st.error(f"Configuration error: {str(e)}. Verify configuration settings in config.py.")
        st.stop()

def main():
    """Industrial Workplace Shift Monitoring Dashboard."""
    # Warn about Python version compatibility
    if sys.version_info >= (3, 13):
        st.warning("Python 3.13 detected. Use Python 3.10 for optimal compatibility with Streamlit Cloud.")

    st.title("Industrial Workplace Shift Monitoring Dashboard", anchor="dashboard-title")

    # Transparency statement
    st.info("""
    **Transparency Notice**: This dashboard uses aggregated, synthetic data to monitor workplace performance, team well-being, and operational efficiency. Visualizations optimize area layouts and schedules to enhance productivity and team health, not to track individuals. Interventions like breaks and ergonomic adjustments improve team conditions. Feedback shapes a supportive workplace environment.
    """, icon="ℹ️")

    # Load configuration
    config = CONFIG
    validate_config(config)

    # Sidebar controls
    st.sidebar.header("Operational Controls")
    show_forecast = st.sidebar.checkbox(
        "Show Predictive Trends", value=True, key="forecast-checkbox",
        help="Display forecasted trends for compliance and collaboration."
    )
    export_data = st.sidebar.button("Export Operational Data", key="export-button")

    # Collapsible feedback section
    with st.sidebar.expander("Team Feedback & Initiatives"):
        team_feedback = st.text_area(
            "Submit feedback to improve workplace conditions",
            placeholder="E.g., request more breaks, ergonomic equipment, or wellness programs...",
            key="feedback-input"
        )
        initiative = st.selectbox(
            "Team Initiative Priority",
            ["More frequent breaks", "Task reduction", "Wellness programs", "Team recognition"],
            key="initiative-select",
            help="Select the primary team-driven initiative for this shift."
        )

    # Generate operational data
    try:
        (
            team_positions_df,
            compliance_variability,
            collaboration_index,
            operational_resilience,
            efficiency_metrics_df,
            productivity_loss,
            team_wellbeing,
            safety_scores,
            team_feedback_impact
        ) = simulate_workplace_operations(
            num_team_members=config['NUM_TEAM_MEMBERS'],
            num_steps=config['NUM_STEPS'],
            workplace_size=config['WORKPLACE_SIZE'],
            adaptation_rate=config['ADAPTATION_RATE'],
            supervisor_influence=config['SUPERVISOR_INFLUENCE'],
            disruption_steps=config['DISRUPTION_STEPS'],
            team_initiative=initiative,
            config=config
        )
    except Exception as e:
        st.error(f"Data generation failed: {str(e)}.")
        st.error("Possible cause: Invalid data or configuration. Check logs or config.py settings.")
        try:
            (
                team_positions_df,
                compliance_variability,
                collaboration_index,
                operational_resilience,
                efficiency_metrics_df,
                productivity_loss,
                team_wellbeing,
                safety_scores,
                team_feedback_impact
            ) = simulate_workplace_operations(
                num_team_members=config['NUM_TEAM_MEMBERS'],
                num_steps=config['NUM_STEPS'],
                workplace_size=config['WORKPLACE_SIZE'],
                adaptation_rate=config['ADAPTATION_RATE'],
                supervisor_influence=config['SUPERVISOR_INFLUENCE'],
                disruption_steps=config['DISRUPTION_STEPS'],
                team_initiative=initiative,
                skip_forecast=True,
                config=config
            )
            show_forecast = False
            st.warning("Running in fallback mode without predictive trends due to data issues.")
        except Exception as e2:
            st.error(f"Fallback mode failed: {str(e2)}. Contact support or verify config.py.")
            st.stop()

    # Summary section
    st.subheader("Shift Performance Summary", anchor="summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Productivity Loss (Disruptions)",
            f"{np.mean([loss for i, loss in enumerate(productivity_loss) if i in config['DISRUPTION_STEPS']]):.1f}%",
            help="Average productivity loss during disruptions (e.g., IT outages, process delays)."
        )
    with col2:
        st.metric(
            "Well-Being Improvement",
            f"{team_feedback_impact['wellbeing']:.2%}",
            help="Improvement in team well-being from feedback-driven initiatives."
        )
    with col3:
        st.metric(
            "Team Cohesion Improvement",
            f"{team_feedback_impact['cohesion']:.2%}",
            help="Improvement in team collaboration from initiatives."
        )

    # Tabbed navigation for metrics
    tab1, tab2, tab3 = st.tabs(["Operational Performance", "Team Health & Collaboration", "Facility Operations"])

    with tab1:
        st.subheader("Operational Performance Metrics")
        # Task Compliance
        with st.expander("Action: Improve Task Compliance"):
            st.markdown("""
            - **High variability**: Implement standardized operating procedures (SOPs) or additional training.
            - **Post-disruption dips**: Schedule supervisor oversight or process reviews.
            """)
        try:
            st.markdown('<div aria-label="Line plot of task compliance variability over shift intervals">', unsafe_allow_html=True)
            st.pyplot(plot_task_compliance_trend(
                compliance_variability['data'],
                config['DISRUPTION_STEPS'],
                compliance_variability['forecast'] if show_forecast else None
            ))
            st.caption("Line plot showing task compliance consistency over shift intervals. Lower values indicate uniform compliance across team members.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render task compliance plot: {str(e)}. Check simulation data.")

        # Operational Efficiency (OEE)
        with st.expander("Action: Enhance Operational Efficiency"):
            st.markdown("""
            - **Target**: Uptime >90%, Throughput >85%, Quality >97%.
            - **Low OEE**: Review equipment maintenance, training, or process bottlenecks.
            """)
        try:
            st.markdown('<div aria-label="Line plot of operational efficiency metrics (OEE) over shift intervals">', unsafe_allow_html=True)
            st.pyplot(plot_operational_efficiency(efficiency_metrics_df))
            st.caption("Line plot showing operational efficiency metrics (uptime, throughput, quality, OEE) over shift intervals.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render OEE plot: {str(e)}. Verify efficiency data.")

        # Operational Resilience
        with st.expander("Action: Strengthen Operational Resilience"):
            st.markdown("""
            - **Low resilience**: Provide resources (e.g., backup equipment) or cross-training.
            - **Post-disruption recovery**: Implement rapid response protocols.
            """)
        try:
            st.markdown('<div aria-label="Line plot of operational resilience over shift intervals">', unsafe_allow_html=True)
            st.pyplot(plot_operational_resilience(operational_resilience))
            st.caption("Line plot showing operational resilience over shift intervals, with 1 indicating full recovery from disruptions.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render resilience plot: {str(e)}. Check resilience data.")

    with tab2:
        st.subheader("Team Health & Collaboration Metrics")
        col1, col2 = st.columns(2)
        # Team Well-Being
        with col1:
            with st.expander("Action: Improve Team Well-Being"):
                st.markdown("""
                - **Low scores**: Increase break frequency, provide ergonomic equipment, or offer wellness programs.
                - **Sustained declines**: Review workload or shift schedules.
                """)
            try:
                st.markdown('<div aria-label="Line plot of team well-being over shift intervals">', unsafe_allow_html=True)
                st.pyplot(plot_worker_wellbeing(team_wellbeing['scores']))
                st.caption("Line plot showing team well-being trends over shift intervals, with 1 indicating optimal health and morale.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to render well-being plot: {str(e)}. Verify well-being data.")

        # Psychological Safety
        with col2:
            with st.expander("Action: Enhance Psychological Safety"):
                st.markdown("""
                - **Low trust**: Encourage open communication through feedback sessions.
                - **Team recognition**: Implement reward programs to boost morale.
                """)
            try:
                st.markdown('<div aria-label="Line plot of psychological safety over shift intervals">', unsafe_allow_html=True)
                st.pyplot(plot_psychological_safety(safety_scores))
                st.caption("Line plot showing psychological safety trends over shift intervals, with 1 indicating high trust and communication.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to render psychological safety plot: {str(e)}. Check safety data.")

        # Team Collaboration
        with st.expander("Action: Boost Team Collaboration"):
            st.markdown("""
            - **Low collaboration**: Organize team-building activities or cross-area training.
            - **High distances**: Optimize area layouts for better interaction.
            """)
        try:
            st.markdown('<div aria-label="Line plot of team collaboration strength over shift intervals">', unsafe_allow_html=True)
            st.pyplot(plot_worker_collaboration_trend(
                collaboration_index['data'],
                collaboration_index['forecast'] if show_forecast else None
            ))
            st.caption("Line plot showing team collaboration strength over shift intervals, with higher values indicating stronger teamwork.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render collaboration plot: {str(e)}. Verify collaboration data.")

    with tab3:
        st.subheader("Facility Operations Insights")
        # Team Distribution
        with st.expander("Action: Optimize Facility Layout"):
            st.markdown("""
            - **High density areas**: Reconfigure work areas or reassign tasks to reduce congestion.
            - **Low activity zones**: Investigate underutilization or equipment issues.
            """)
        try:
            plot_mode = st.selectbox(
                "Visualization Mode", ["Hexbin + Scatter", "Hexbin Only"], key="plot-mode",
                help="Choose between combined hexbin and scatter plot or hexbin only."
            )
            use_scatter = plot_mode == "Hexbin + Scatter"
            fig = plot_worker_distribution(team_positions_df, config['WORKPLACE_SIZE'], config=config, use_plotly=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Interactive plot showing team distribution and density in the facility. Use the dropdown to filter by work area, "
                "the slider to animate across shift intervals, and the reset zoom button to restore the view."
            )
        except Exception as e:
            st.warning(f"Interactive visualization failed: {str(e)}. Using Matplotlib fallback.")
            try:
                st.markdown('<div aria-label="Hexbin plot of team distribution and density in facility">', unsafe_allow_html=True)
                st.pyplot(plot_worker_distribution(team_positions_df, config['WORKPLACE_SIZE'], config=config, use_plotly=False))
                st.caption("Hexbin plot with scatter points showing team distribution and density, colored by team member with counts.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e2:
                st.error(f"Failed to render distribution plot: {str(e2)}. Check team position data.")

    # Team well-being opportunities
    st.subheader("Team Well-Being Opportunities", anchor="wellbeing-opportunities")
    triggers = team_wellbeing['triggers']
    if triggers['threshold']:
        steps = triggers['threshold']
        st.info(
            f"💡 Address low well-being in {len(steps)} shift intervals (Steps: {', '.join(map(str, steps[:3]))}"
            f"{', ...' if len(steps) > 3 else ''}). **Actions**: Increase break frequency, provide ergonomic equipment, or launch wellness programs."
        )
    if triggers['trend']:
        steps = triggers['trend']
        st.info(
            f"💡 Address declining well-being trends in {len(steps)} shift intervals (Steps: {', '.join(map(str, steps[:3]))}"
            f"{', ...' if len(steps) > 3 else ''}). **Actions**: Reduce task loads, offer stress management workshops."
        )
    if triggers['work_area']:
        for ws, steps in triggers['work_area'].items():
            st.info(
                f"💡 Support {config['WORK_AREAS'][ws]['label']} in {len(steps)} shift intervals (Steps: {', '.join(map(str, steps[:3]))}"
                f"{', ...' if len(steps) > 3 else ''}). **Actions**: Assign additional supervisors, improve area ergonomics."
            )
    if triggers['disruption']:
        steps = triggers['disruption']
        st.info(
            f"💡 Support recovery post-disruptions in {len(steps)} shift intervals (Steps: {', '.join(map(str, steps[:3]))}"
            f"{', ...' if len(steps) > 3 else ''}). **Actions**: Recognize team efforts, schedule additional breaks."
        )
    if not any(triggers.values()):
        st.success("Team well-being is strong! Maintain with regular breaks, ergonomic support, and recognition programs.")

    # Performance anomalies
    anomalies = [
        (i, e, c) for i, (e, c) in enumerate(zip(compliance_variability['z_scores'], collaboration_index['z_scores']))
        if abs(e) > config['ANOMALY_THRESHOLD'] or abs(c) > config['ANOMALY_THRESHOLD']
    ]
    if anomalies:
        st.info(
            f"💡 Address performance or collaboration anomalies in {len(anomalies)} shift intervals. "
            "**Actions**: Provide targeted training, enhance team-building activities."
        )

    # Data export
    if export_data:
        if not team_positions_df.empty:
            csv = team_positions_df.to_csv(index=False)
            st.download_button(
                label="Download Team Position Data",
                data=csv,
                file_name='team_position_data.csv',
                mime='text/csv',
                key="download-button"
            )
        else:
            st.error("No team position data available to export.")

    st.success("Shift monitoring dashboard loaded successfully. Use insights to optimize performance and team well-being.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Dashboard initialization failed: {str(e)}. Ensure dependencies, 'industrial_workplace_simulation.py', and 'config.py' are available.")
        st.stop()
```
