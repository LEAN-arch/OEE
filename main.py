try:
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px
    import plotly.figure_factory as ff
    from simulation import generate_synthetic_data, plot_compliance_variability, plot_team_clustering, plot_resilience, plot_oee, plot_worker_density, plot_wellbeing, plot_psychological_safety
    from config import NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, ANOMALY_THRESHOLD, WELLBEING_THRESHOLD, WELLBEING_TREND_LENGTH, WELLBEING_DISRUPTION_WINDOW, BREAK_INTERVAL, WORKLOAD_CAP_STEPS
except ImportError as e:
    st.error(f"Failed to import required libraries: {str(e)}. Ensure all dependencies are installed using 'uv pip install -r requirements.txt'.")
    st.error("If deploying on Streamlit Cloud, use Python 3.10 and verify requirements.txt.")
    st.stop()

def main():
    """Factory Operations Dashboard for Shift Monitoring."""
    # Warn about Python version
    import sys
    if sys.version_info >= (3, 13):
        st.warning("Python 3.13 detected. Use Python 3.10 for best results.")

    st.title("Factory Shift Monitoring Dashboard", anchor="dashboard-title")

    # Transparency statement
    st.info("""
    **Transparency Notice**: This dashboard uses aggregated, synthetic data to monitor zone-level performance and well-being, ensuring worker privacy. The activity visualization optimizes layouts and tasks to improve conditions, not track individuals. Actions like breaks aim to enhance team health. Worker suggestions shape a supportive workplace.
    """, icon="ℹ️")

    # Sidebar controls
    st.sidebar.header("Shift Controls")
    show_forecast = st.sidebar.checkbox("Show Predictive Trends", value=True, key="forecast-checkbox")
    export_data = st.sidebar.button("Export Shift Data", key="export-button")

    # Collapsible worker feedback section
    with st.sidebar.expander("Worker Suggestions & Priorities"):
        worker_feedback = st.text_area(
            "Share ideas to improve conditions",
            placeholder="E.g., more breaks, ergonomic tools...",
            key="feedback-input"
        )
        priority = st.selectbox(
            "Worker priority for this shift",
            ["More frequent breaks", "Task reduction", "Wellness resources", "Team recognition"],
            key="priority-select",
            help="Reflects team preferences."
        )

    # Generate synthetic data
    try:
        history_df, compliance_entropy, clustering_index, resilience_scores, oee_history, productivity_loss, wellbeing_data, safety_scores, feedback_impact = generate_synthetic_data(
            NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, priority
        )
        oee_df = pd.DataFrame(oee_history)
        wellbeing_scores = wellbeing_data['scores']
        wellbeing_triggers = wellbeing_data['triggers']
    except Exception as e:
        st.error(f"Data generation failed: {str(e)}.")
        st.error("Possible cause: Invalid data (e.g., NaN values). Check logs or simulation.py.")
        st.error("If on Streamlit Cloud, ensure Python 3.10 and rebuild. Try disabling 'Show Predictive Trends'.")
        try:
            history_df, compliance_entropy, clustering_index, resilience_scores, oee_history, productivity_loss, wellbeing_data, safety_scores, feedback_impact = generate_synthetic_data(
                NUM_OPERATORS, NUM_STEPS, FACTORY_SIZE, ADAPTATION_RATE, SUPERVISOR_INFLUENCE, DISRUPTION_STEPS, priority, skip_forecast=True
            )
            oee_df = pd.DataFrame(oee_history)
            wellbeing_scores = wellbeing_data['scores']
            wellbeing_triggers = wellbeing_data['triggers']
            show_forecast = False
            st.warning("Running in fallback mode without predictive trends.")
        except Exception as e2:
            st.error(f"Fallback failed: {str(e2)}. Contact support.")
            st.stop()

    # Summary section with metric cards
    st.subheader("Shift Summary", anchor="summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Productivity Loss (Disruptions)", f"{np.mean([loss for i, loss in enumerate(productivity_loss) if i in DISRUPTION_STEPS]):.1f}%", help="Average loss during disruptions.")
    with col2:
        st.metric("Well-Being Boost (Suggestions)", f"{feedback_impact['wellbeing']:.2%}", help="Impact of worker suggestions on well-being.")
    with col3:
        st.metric("Collaboration Boost", f"{feedback_impact['cohesion']:.2%}", help="Impact of suggestions on teamwork.")

    # Tabbed navigation
    tab1, tab2, tab3 = st.tabs(["Performance", "Team Health", "Operations"])

    with tab1:
        st.subheader("Performance Metrics")
        # SOP Compliance
        with st.expander("Recommendations: SOP Compliance"):
            st.write("Inconsistent compliance may suggest clearer procedures or training support.")
        try:
            st.pyplot(plot_compliance_variability(compliance_entropy['data'], DISRUPTION_STEPS, compliance_entropy['forecast'] if show_forecast else None), 
                     alt="Line plot of SOP compliance trends over shift time")
        except Exception as e:
            st.error(f"Failed to render compliance plot: {str(e)}.")

        # OEE
        with st.expander("Recommendations: Equipment Efficiency"):
            st.write("Target: Availability >90%, Performance >85%, Quality >97%. Support with maintenance or training.")
        try:
            st.pyplot(plot_oee(oee_df), alt="Line plot of OEE metrics over shift time")
        except Exception as e:
            st.error(f"Failed to render OEE plot: {str(e)}.")

        # Resilience
        with st.expander("Recommendations: Resilience"):
            st.write("Strong recovery reflects resilience. Support with resources or guidance.")
        try:
            st.pyplot(plot_resilience(resilience_scores), alt="Line plot of team resilience over shift time")
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
                st.pyplot(plot_wellbeing(wellbeing_scores), alt="Line plot of team well-being trends over shift time")
            except Exception as e:
                st.error(f"Failed to render well-being plot: {str(e)}.")

        # Psychological Safety
        with col2:
            with st.expander("Recommendations: Psychological Safety"):
                st.write("Encourage open feedback and team recognition.")
            try:
                st.pyplot(plot_psychological_safety(safety_scores), alt="Line plot of psychological safety trends over shift time")
            except Exception as e:
                st.error(f"Failed to render safety plot: {str(e)}.")

        # Collaboration
        with st.expander("Recommendations: Collaboration"):
            st.write("Foster with team-building or recognition programs.")
        try:
            st.pyplot(plot_team_clustering(clustering_index['data'], clustering_index['forecast'] if show_forecast else None), 
                     alt="Line plot of team collaboration strength over shift time")
        except Exception as e:
            st.error(f"Failed to render collaboration plot: {str(e)}.")

    with tab3:
        st.subheader("Operational Insights")
        # Congestion (Plotly with Matplotlib fallback)
        with st.expander("Recommendations: Activity & Congestion"):
            st.write("High congestion areas may need layout adjustments or task reassignments.")
        try:
            fig = plot_worker_density(history_df, FACTORY_SIZE, use_plotly=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotly failed: {str(e)}. Using Matplotlib fallback.")
            try:
                st.pyplot(plot_worker_density(history_df, FACTORY_SIZE, use_plotly=False), 
                         alt="Hexbin plot of factory floor activity and congestion")
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
        for zone, steps in wellbeing_triggers['zone'].items():
            st.info(f"💡 Support {zone.capitalize()} zone in {len(steps)} intervals (Steps: {', '.join(map(str, steps[:3]))}{', ...' if len(steps) > 3 else ''}). Actions: Add supervisors, adjust ergonomics.")
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
                file_name='factory_shift_data.csv',
                mime='text/csv',
                key="download-button"
            )
        else:
            st.error("No shift data available to export.")

    st.success("Shift dashboard loaded — use insights to create a healthier workplace.")

if __name__ == "__main__":
    main()
