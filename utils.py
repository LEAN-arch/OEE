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
    productivity_loss: list
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
    """
    try:
        # Defensive import to handle Streamlit reloading
        import pandas as pd
        logger.info("Starting save_simulation_data with pandas imported")
        
        team_positions_df.to_csv('team_positions.csv', index=False)
        efficiency_metrics_df.to_csv('efficiency_metrics.csv', index=False)
        pd.DataFrame({
            'step': range(DEFAULT_CONFIG['SHIFT_DURATION_INTERVALS']),
            'compliance_entropy': compliance_variability['data'],
            'collaboration_index': collaboration_index['data'],
            'resilience': operational_resilience,
            'wellbeing': team_wellbeing['scores'],
            'safety': safety,
            'productivity_loss': productivity_loss
        }).to_csv('summary_metrics.csv', index=False)
        logger.info("Simulation data saved successfully")
    except Exception as e:
        logger.error(f"Failed to save simulation data: {str(e)}")
        raise
