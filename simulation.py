"""
simulation.py
Simulate workplace operations for the Workplace Shift Monitoring Dashboard.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [User Action: %(user_action)s]',
    filename='dashboard.log'
)

def simulate_workplace_operations(num_team_members, num_steps, disruption_intervals, team_initiative, config):
    """
    Simulate workplace operations with realistic worker movement and workload status.
    
    Args:
        num_team_members (int): Number of workers.
        num_steps (int): Number of 2-minute intervals.
        disruption_intervals (list): Steps where disruptions occur.
        team_initiative (str): Initiative to improve well-being/safety.
        config (dict): Configuration settings.
    
    Returns:
        tuple: Simulation results including worker positions with workload status.
    """
    try:
        # Validate inputs
        if not isinstance(num_team_members, int) or num_team_members <= 0:
            raise ValueError(f"num_team_members must be a positive integer, got {num_team_members}")
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError(f"num_steps must be a positive integer, got {num_steps}")
        if not all(isinstance(t, int) and 0 <= t < num_steps for t in disruption_intervals):
            raise ValueError(f"disruption_intervals must be valid indices, got {disruption_intervals}")
        if team_initiative not in ["More frequent breaks", "Team recognition"]:
            raise ValueError(f"Invalid team_initiative, got {team_initiative}")

        logger.info(
            f"Starting simulation: team_size={num_team_members}, steps={num_steps}, "
            f"disruptions={disruption_intervals}, initiative={team_initiative}",
            extra={'user_action': 'Run Simulation'}
        )

        # Initialize data
        np.random.seed(42)
        steps = range(num_steps)
        
        # Simulate worker
