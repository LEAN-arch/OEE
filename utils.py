"""
utils.py
Utility functions for the Industrial Workplace Shift Monitoring Dashboard.
Handles logging, data processing, and UI helpers.
"""

import logging
import pandas as pd
from config import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dashboard.log'
)
logger = logging.getLogger(__name__)

def save_simulation_data(
    team_positions_df: pd.DataFrame,
    efficiency_metrics_df: pd.DataFrame,
    compliance_variability: dict,
    collaboration_index: dict,
    operational_resilience: list,
    team_wellbeing: dict,
    safety: list,
    productivity_loss: list,
    feedback_impact: dict
) -> None:
    """
    Save simulation outputs to CSV files.

    Args:
        team_positions_df (pd.DataFrame): Team positions data.
        efficiency_metrics_df (pd.DataFrame): Efficiency metrics data.
        compliance_variability (dict): Compliance variability data.
        collaboration_index (dict): Collaboration index data.
        operational_resilience (list): Resilience scores.
        team_wellbeing (dict): Well-being data.
        safety (list): Safety scores.
        productivity_loss (list): Productivity loss values.
        feedback_impact (dict): Feedback impact data (wellbeing, cohesion).

    Raises:
        Exception: If saving fails, with detailed logging.
    """
    try:
        # Defensive import for Streamlit reloading
        import pandas as pd
        logger.info("save_simulation_data: Pandas imported successfully")
        
        # Log number of arguments received
        logger.info("save_simulation_data: Received %d arguments", 9)
        
        logger.info("Saving team_positions.csv")
        team_positions_df.to_csv('team_positions.csv', index=False)
        logger.info("Saving efficiency_metrics.csv")
        efficiency_metrics_df.to_csv('efficiency_metrics.csv', index=False)
        logger.info("Creating summary_metrics DataFrame")
        summary_df = pd.DataFrame({
            'step': range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']),
            'compliance_entropy': compliance_variability['data'],
            'collaboration_index': collaboration_index['data'],
            'resilience': operational_resilience,
            'wellbeing': team_wellbeing['scores'],
            'safety': safety,
            'productivity_loss': productivity_loss,
            'feedback_wellbeing': [feedback_impact['wellbeing']] * DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS'],
            'feedback_cohesion': [feedback_impact['cohesion']] * DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']
        })
        logger.info("Saving summary_metrics.csv")
        summary_df.to_csv('summary_metrics.csv', index=False)
        logger.info("Simulation data saved successfully")
    except NameError as ne:
        logger.error(f"NameError in save_simulation_data: {str(ne)}")
        raise
    except Exception as e:
        logger.error(f"Failed to save simulation data: {str(e)}")
        raise
