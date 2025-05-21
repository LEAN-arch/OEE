try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px
    import plotly.figure_factory as ff
    from simulation import generate_synthetic_data, plot_compliance_variability, plot_team_clustering, plot_resilience, plot_oee, plot_worker_density, plot_wellbeing, plot_psychological_safety
    from config import NUM_TEAM_MEMBERS, NUM_STEPS, WORKPLACE_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, ANOMALY_THRESHOLD, WELLBEING_THRESHOLD, WELLBEING_TREND_LENGTH, WELLBEING_DISRUPTION_WINDOW, BREAK_INTERVAL, WORKLOAD_CAP_STEPS, WORK_AREAS
except ImportError as e:
    st.error(f"Failed to import libraries: {str(e)}. Run 'uv pip install -r requirements.txt'.")
    st.error("Ensure Python 3.10 on Streamlit Cloud and verify requirements.txt.")
    st.stop()

def main():
    """Workplace Shift Monitoring Dashboard."""
    # Warn about Python version
    import sys
    if sys.version_info >= (3, 13):
        st.warning("Python 3.13 detected. Use Python 3.10 for compatibility.")

    st.title("Workplace Shift Monitoring Dashboard", anchor="dashboard-title")

    # Transparency statement
    st.info("""
    **Transparency Notice**: This dashboard uses aggregated, synthetic data to monitor work area performance and well-being, ensuring team member privacy. Activity visualizations optimize layouts and tasks to improve conditions, not track individuals. Actions like breaks enhance team health. Team suggestions shape a supportive workplace.
    """, icon="ℹ️")

    # Validate config
    try:
        if NUM_TEAM_MEMBERS < 1 or NUM_STEPS < 1 or WORKPLACE_SIZE <= 0:
            raise ValueError("Invalid config: NUM_TEAM_MEMBERS, NUM_STEPS, and WORKPLACE_SIZE must be positive.")
        if not WORK_AREAS or not all("center" in area and "label" in area for area in WORK_AREAS.values()):
            raise ValueError("Invalid WORK_AREAS: Must define center coordinates and labels.")
        for name, area in WORK_AREAS.items():
            x, y = area["center"]
            if not (0 <= x <= WORKPLACE_SIZE and 0 <= y <= WORKPLACE_SIZE):
                raise ValueError(f"Invalid center for {name}: Coordinates must be within [0, {WORKPLACE_SIZE}].")
    except ValueError as e:
        st.error(f"Configuration error: {str(e)}. Check config.py.")
        st.stop()

    # Sidebar controls
    st.sidebar.header("Shift Controls")
    show_forecast = st.sidebar.checkbox("Show Predictive Trends", value=True, key="forecast-checkbox")
    export_data = st.sidebar.button("Export Shift Data", key="export-button")

    # Collapsible feedback section
    with st.sidebar.expander("Team Suggestions & Priorities"):
        worker_feedback = st.text_area(
            "Share ideas to improve conditions",
            placeholder="E.g., more breaks, ergonomic tools...",
            key="feedback-input"
        )
        priority = st.selectbox(
            "Team priority for this shift",
            ["More frequent breaks", "Task reduction", "Wellness resources", "Team recognition"],
            key="priority-select",
            help="Reflects team preferences."
        )

    # Generate synthetic data
    try:
        history_df, compliance_entropy, clustering_index, resilience_scores, efficiency_history, productivity_loss, wellbeing_data, safety_scores, feedback_impact = generate_synthetic_data(
            NUM_TEAM_MEMBERS, NUM_STEPS, WORKPLACE_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, priority
        )
        efficiency_df = pd.DataFrame(efficiency_history)
        wellbeing_scores = wellbeing_data['scores']
        wellbeing_triggers = wellbeing_data['triggers']
    except Exception as e:
        st.error(f"Data generation failed: {str(e)}.")
        st.error("Possible cause: Invalid data (e.g., NaN). Check logs or simulation.py.")
        st.error("Try disabling 'Show Predictive Trends' or verify config.py.")
        try:
            history_df, compliance_entropy, clustering_index, resilience_scores, efficiency_history, productivity_loss, wellbeing_data, safety_scores, feedback_impact = generate_synthetic_data(
                NUM_TEAM_MEMBERS, NUM_STEPS, WORKPLACE_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, priority, skip_forecast=True
            )
            efficiency_df = pd.DataFrame(efficiency_history)
            wellbeing_scores = wellbeing_data['scores']
            wellbeing_triggers = wellbeing_data['triggers']
            show_forecast = False
            st.warning("Running in fallback mode without predictive trends.")
        except Exception as e2:
            st.error(f"Fallback failed: {str(e2)}. Contact support.")
            st.stop()

    # Summary section
    st.subheader("Shift Summary", anchor="summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Productivity Loss (Disruptions)", f"{np.mean([loss for i, loss in enumerate(productivity_loss) if i in DISRUPTION_STEPS]):.1f}%", help="Loss during disruptions.")
    with col2:
        st.metric("Well-Being Boost (Suggestions)", f"{feedback_impact['wellbeing']:.2%}", help="Impact of team suggestions.")
    with col3:
        st.metric("Collaboration Boost", f"{feedback_impact['cohesion']:.2%}", help="Impact on teamwork.")

    # Tabbed navigation
    tab1, tab2, tab3 = st.tabs(["Performance", "Team Health", "Operations"])

    with tab1:
        st.subheader("Performance Metrics")
        # Task Adherence
        with st.expander("Recommendations: Task Adherence"):
            st.write("Inconsistent adherence may suggest clearer guidelines or training.")
        try:
            st.markdown('<div aria-label="Line plot of task adherence trends over shift time">', unsafe_allow_html=True)
            st.pyplot(plot_compliance_variability(compliance_entropy['data'], DISRUPTION_STEPS, compliance_entropy['forecast'] if show_forecast else None))
            st.caption("Line plot showing task adherence trends over shift time, with lower values indicating uniform adherence.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render adherence plot: {str(e)}.")

        # Operational Efficiency
        with st.expander("Recommendations: Operational Efficiency"):
            st.write("Target: Uptime >90%, Throughput >85%, Quality >97%. Support with resources or training.")
        try:
            st.markdown('<div aria-label="Line plot of operational efficiency metrics over shift time">', unsafe_allow_html=True)
            st.pyplot(plot_oee(efficiency_df))
            st.caption("Line plot showing operational efficiency metrics (uptime, throughput, quality) over shift time.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render efficiency plot: {str(e)}.")

        # Resilience
        with st.expander("Recommendations: Resilience"):
            st.write("Strong recovery reflects resilience. Support with resources or guidance.")
        try:
            st.markdown('<div aria-label="Line plot of team resilience over shift time">', unsafe_allow_html=True)
            st.pyplot(plot_resilience(resilience_scores))
            st.caption("Line plot showing team resilience trends over shift time, with 1 indicating full recovery.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render resilience plot: {str(e)}.")

    with tab2:
        st.subheader("Team Health Metrics")
        col1, col2 = st.columns(2)
        # Well-Being
        with col1:
            with st.expander("Recommendations: Well-Being"):
                st.write("Enhance with regular breaks, ergonomic tools, or wellness programs.")
            try:
                st.markdown('<div aria-label="Line plot of team well-being trends over shift time">', unsafe_allow_html=True)
                st.pyplot(plot_wellbeing(wellbeing_scores))
                st.caption("Line plot showing team well-being trends over shift time, with 1 indicating optimal well-being.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to render well-being plot: {str(e)}.")

        # Psychological Safety
        with col2:
            with st.expander("Recommendations: Psychological Safety"):
                st.write("Encourage open feedback and team recognition.")
            try:
                st.markdown('<div aria-label="Line plot of psychological safety trends over shift time">', unsafe_allow_html=True)
                st.pyplot(plot_psychological_safety(safety_scores))
                st.caption("Line plot showing psychological safety trends over shift time, with 1 indicating high trust.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to render safety plot: {str(e)}.")

        # Collaboration
        with st.expander("Recommendations: Collaboration"):
            st.write("Foster with team-building or recognition programs.")
        try:
            st.markdown('<div aria-label="Line plot of team collaboration strength over shift time">', unsafe_allow_html=True)
            st.pyplot(plot_team_clustering(clustering_index['data'], clustering_index['forecast'] if show_forecast else None))
            st.caption("Line plot showing team collaboration strength over shift time, with higher values indicating stronger collaboration.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to render collaboration plot: {str(e)}.")

    with tab3:
        st.subheader("Operational Insights")
        # Activity & Density
        with st.expander("Recommendations: Activity & Density"):
            st.write("High density areas may need layout adjustments or task reassignments.")
        try:
            fig = plot_worker_density(history_df, WORKPLACE_SIZE, use_plotly=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Interactive hexbin plot showing workplace activity and density, with tooltips for area and activity level.")
        except Exception as e:
            st.warning(f"Plotly failed: {str(e)}. Using Matplotlib fallback.")
            try:
                st.markdown('<div aria-label="Hexbin plot of workplace activity and density">', unsafe_allow_html=True)
                st.pyplot(plot_worker_density(history_df, WORKPLACE_SIZE, use_plotly=False))
                st.caption("Hexbin plot showing workplace activity and density, with color indicating activity level from low to high.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e2:
                st.error(f"Failed to render activity plot: {str(e2)}.")

    # Well-being opportunities
    st.subheader("Team Well-Being Opportunities", anchor="wellbeing-opportunities")
    if wellbeing_triggers['threshold']:
        steps = wellbeing_triggers['threshold']
        st.info(f"💡 Support teams in {len(steps)} intervals with low well-being (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). Actions: Extend breaks, provide ergonomic tools.")
    if wellbeing_triggers['trend']:
        steps = wellbeing_triggers['trend']
        st.info(f"💡 Address well-being decline in {len(steps)} intervals (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). Actions: Reduce tasks, offer workshops.")
    if wellbeing_triggers['zone']:
        for area, steps in wellbeing_triggers['zone'].items():
            st.info(f"💡 Support {area} in {len(steps)} intervals (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). Actions: Add supervisors, adjust ergonomics.")
    if wellbeing_triggers['disruption']:
        steps = wellbeing_triggers['disruption']
        st.info(f"💡 Support teams near disruptions in {len(steps)} intervals (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). Actions: Recognize teams, schedule breaks.")
    if not any(wellbeing_triggers.values()):
        st.success("Team well-being is strong! Maintain support with breaks and recognition.")

    # Performance opportunities
    anomalies = [(i, e, c) for i, (e, c) in enumerate(zip(compliance_entropy['z_scores'], clustering_index['z_scores']))
                 if abs(e) > ANOMALY_THRESHOLD or abs(c) > ANOMALY_THRESHOLD]
    if anomalies:
        st.info(f"💡 Enhance performance or collaboration in {len(anomalies)} intervals. Actions: Offer training, foster team-building.")

    # Data export
    if export_data:
        if not history_df.empty:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download Aggregated Shift Data",
                data=csv,
                file_name='workplace_shift_data.csv',
                mime='text/csv',
                key="download-button"
            )
        else:
            st.error("No shift data available to export.")

    st.success("Shift dashboard loaded — use insights to create a healthier workplace.")

if __name__ == "__main__":
    main()
