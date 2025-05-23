# utils.py
import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any # Added for better type hinting

logger = logging.getLogger(__name__)
SAVE_FILE_PATH = "simulation_data_cache.pkl"

def save_simulation_data(simulation_results_dict: Dict[str, Any]) -> None:
    """Saves the simulation results dictionary to a pickle file."""
    try:
        with open(SAVE_FILE_PATH, "wb") as f:
            pickle.dump(simulation_results_dict, f)
        logger.info(f"Simulation data successfully saved to {SAVE_FILE_PATH}", extra={'user_action': 'Save Simulation Data - Success'})
    except Exception as e:
        logger.error(f"Error saving simulation data to {SAVE_FILE_PATH}: {e}", exc_info=True, extra={'user_action': 'Save Simulation Data - Error'})

def load_simulation_data() -> Optional[Dict[str, Any]]:
    """Loads simulation results dictionary from a pickle file."""
    try:
        with open(SAVE_FILE_PATH, "rb") as f:
            simulation_results_dict = pickle.load(f)
        # Basic validation: check if it's a dictionary
        if not isinstance(simulation_results_dict, dict):
            logger.error(f"Loaded data from {SAVE_FILE_PATH} is not a dictionary. Data is likely corrupted or invalid.", extra={'user_action': 'Load Simulation Data - Invalid Format'})
            return None
        logger.info(f"Simulation data successfully loaded from {SAVE_FILE_PATH}", extra={'user_action': 'Load Simulation Data - Success'})
        return simulation_results_dict
    except FileNotFoundError:
        logger.warning(f"Save file {SAVE_FILE_PATH} not found. No data loaded.", extra={'user_action': 'Load Simulation Data - File Not Found'})
        return None
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling data from {SAVE_FILE_PATH}. File may be corrupted or from an incompatible version: {e}", exc_info=True, extra={'user_action': 'Load Simulation Data - Unpickling Error'})
        return None
    except Exception as e: # Catch any other unexpected errors during loading
        logger.error(f"An unexpected error occurred while loading data from {SAVE_FILE_PATH}: {e}", exc_info=True, extra={'user_action': 'Load Simulation Data - Unexpected Error'})
        return None

def generate_pdf_report(summary_df: pd.DataFrame) -> None:
    """Generates a LaTeX (.tex) report from a summary DataFrame."""
    if not isinstance(summary_df, pd.DataFrame):
        logger.error("generate_pdf_report expects a Pandas DataFrame as input.", extra={'user_action': 'Generate LaTeX Report - Invalid Input Type'})
        return
    if summary_df.empty:
        logger.warning("generate_pdf_report received an empty DataFrame. No report will be generated.", extra={'user_action': 'Generate LaTeX Report - Empty DataFrame'})
        return

    logger.info("Attempting to generate LaTeX report from summary DataFrame.", extra={'user_action': 'Generate LaTeX Report - Start'})
    try:
        df_for_latex = summary_df.copy()

        # Sanitize column names for LaTeX and improve readability
        df_for_latex.columns = [
            str(col).replace('_', ' ').replace('%', '\\%').title()
            for col in df_for_latex.columns
        ]
        
        # Round numeric columns appropriately, handling potential non-numeric data mixed in
        for col in df_for_latex.columns:
            if pd.api.types.is_numeric_dtype(df_for_latex[col]):
                # Check if column is not purely integers before rounding to avoid adding .0
                if not df_for_latex[col].dropna().apply(lambda x: isinstance(x, int) or (isinstance(x, float) and x.is_integer())).all():
                    try:
                        df_for_latex[col] = pd.to_numeric(df_for_latex[col], errors='coerce').round(2)
                    except Exception as e_round:
                        logger.warning(f"Could not round column '{col}' for LaTeX report: {e_round}", extra={'user_action': 'Generate LaTeX Report - Rounding Warning'})
            # Convert other types to string to prevent to_latex errors
            elif not pd.api.types.is_string_dtype(df_for_latex[col]):
                 df_for_latex[col] = df_for_latex[col].astype(str)


        num_cols = len(df_for_latex.columns)
        # Define column format: left-aligned for all columns
        col_fmt = '|' + 'l|' * num_cols if num_cols > 0 else '|l|'

        # Generate LaTeX table string
        latex_table = df_for_latex.to_latex(
            index=False,       # Don't write DataFrame index as a column
            escape=True,       # Escape LaTeX special characters in text
            column_format=col_fmt,
            header=True,
            longtable=True,    # Use longtable environment for tables spanning multiple pages
            na_rep='-',        # Representation for NaN values
            caption='Summary of Simulation Metrics Over Time.', # Table caption
            label='tab:simsummary' # Label for cross-referencing
        )

        # LaTeX document template
        latex_document = f"""
\\documentclass[10pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}      % For including images (plots)
\\usepackage{{booktabs}}      % For professional quality tables
\\usepackage{{longtable}}     % For tables that span multiple pages
\\usepackage[margin=1in]{{geometry}} % Set page margins
\\usepackage{{fancyhdr}}      % For custom headers and footers
\\usepackage{{array}}         % For more control over table column formatting
\\usepackage{{xcolor}}        % For using colors (if needed, though not used here for table text)

\\pagestyle{{fancy}}
\\fancyhf{{}} % Clear existing header/footer
\\fancyhead[C]{{Workplace Shift Monitoring Report}}
\\fancyfoot[C]{{\\thepage}}
\\renewcommand{{\\headrulewidth}}{{0.4pt}}
\\renewcommand{{\\footrulewidth}}{{0.4pt}}

\\title{{Workplace Shift Monitoring Report}}
\\author{{Automated Simulation System}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle
\\thispagestyle{{empty}} % No header/footer on title page
\\clearpage
\\pagenumbering{{arabic}} % Start page numbering for main content

\\section*{{Simulation Data Summary}}
The data from the dashboard provides an overview of key operational metrics from the latest simulation run.
The table below presents a time-series summary of these metrics.

% Ensure longtable caption appears correctly above the table
\\begingroup
\\centering
{latex_table}
\\endgroup

% Placeholder for further analysis and visualizations
% Example:
% \\subsection*{{Key Visualizations}}
% \\begin{{figure}}[h!]
%   \\centering
%   % \\includegraphics[width=0.8\\textwidth]{{path/to/your/plot_image.png}}
%   \\caption{{Example Plot Title.}}
%   \\label{{fig:exampleplot}}
% \\end{{figure}}

\\end{{document}}
"""
        report_filename = "workplace_report.tex"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(latex_document)
        logger.info(f"LaTeX report '{report_filename}' generated successfully.", extra={'user_action': 'Generate LaTeX Report - Success'})

    except Exception as e:
        logger.error(f"Failed to generate LaTeX report: {e}", exc_info=True, extra={'user_action': 'Generate LaTeX Report - Error'})
