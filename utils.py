# utils.py
import pickle
import pandas as pd
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)
SAVE_FILE_PATH = "simulation_data_cache.pkl"

def save_simulation_data(simulation_results_dict: Dict[str, Any]) -> None:
    """Saves the simulation results dictionary to a pickle file."""
    try:
        with open(SAVE_FILE_PATH, "wb") as f:
            pickle.dump(simulation_results_dict, f)
        logger.info(f"Simulation data successfully saved to {SAVE_FILE_PATH}", extra={'user_action': 'Save Sim Data - Success'})
    except Exception as e:
        logger.error(f"Error saving simulation data to {SAVE_FILE_PATH}: {e}", exc_info=True, extra={'user_action': 'Save Sim Data - Error'})

def load_simulation_data() -> Optional[Dict[str, Any]]:
    """Loads simulation results dictionary from a pickle file."""
    try:
        with open(SAVE_FILE_PATH, "rb") as f:
            simulation_results_dict = pickle.load(f)
        if not isinstance(simulation_results_dict, dict):
            logger.error(f"Loaded data from {SAVE_FILE_PATH} is not a dictionary. Data is likely corrupted or invalid.", extra={'user_action': 'Load Sim Data - Invalid Format'})
            return None
        
        # Basic validation for essential keys expected by main.py
        required_top_level_keys = ['config_params', 'team_positions_df', 'task_compliance', 'downtime_events_log'] # Add more if critical
        missing_keys = [key for key in required_top_level_keys if key not in simulation_results_dict]
        if missing_keys:
             logger.warning(f"Loaded data from {SAVE_FILE_PATH} is missing essential keys: {', '.join(missing_keys)}. May be incompatible.", extra={'user_action': 'Load Sim Data - Missing Keys'})
             # Decide if to return None or the partial dict based on strictness
             # return None # More strict: if essential parts are missing, treat as invalid load

        logger.info(f"Simulation data successfully loaded from {SAVE_FILE_PATH}", extra={'user_action': 'Load Sim Data - Success'})
        return simulation_results_dict
    except FileNotFoundError:
        logger.warning(f"Save file {SAVE_FILE_PATH} not found. No data loaded.", extra={'user_action': 'Load Sim Data - File Not Found'})
        return None
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling data from {SAVE_FILE_PATH}. File may be corrupted or from an incompatible version: {e}", exc_info=True, extra={'user_action': 'Load Sim Data - Unpickling Error'})
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data from {SAVE_FILE_PATH}: {e}", exc_info=True, extra={'user_action': 'Load Sim Data - Unexpected Error'})
        return None

def generate_pdf_report(summary_df: pd.DataFrame, sim_config_params: Optional[Dict[str, Any]] = None) -> None:
    """
    Generates a LaTeX (.tex) report from a summary DataFrame.
    Optionally includes simulation parameters in the report.
    """
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
            str(col).replace('_', ' ').replace('%', '\\%').title() # Basic sanitization
            for col in df_for_latex.columns
        ]
        
        # Round numeric columns, convert others to string and escape problematic characters
        for col in df_for_latex.columns:
            if pd.api.types.is_numeric_dtype(df_for_latex[col]):
                # Check if column is not purely integers before rounding
                if not df_for_latex[col].dropna().apply(lambda x: isinstance(x, int) or (isinstance(x, float) and x.is_integer())).all():
                    try:
                        df_for_latex[col] = pd.to_numeric(df_for_latex[col], errors='coerce').round(2)
                    except Exception as e_round:
                        logger.warning(f"Could not round column '{col}' for LaTeX report: {e_round}", extra={'user_action': 'Generate LaTeX Report - Rounding Warning'})
            elif not pd.api.types.is_string_dtype(df_for_latex[col]): # If not string and not numeric
                 df_for_latex[col] = df_for_latex[col].astype(str)
            
            # Escape common LaTeX special characters for all string columns (or converted to string)
            if pd.api.types.is_string_dtype(df_for_latex[col]):
                df_for_latex[col] = df_for_latex[col].str.replace('&', '\\&', regex=False) \
                                                    .str.replace('%', '\\%', regex=False) \
                                                    .str.replace('#', '\\#', regex=False) \
                                                    .str.replace('_', '\\_', regex=False) \
                                                    .str.replace('{', '\\{', regex=False) \
                                                    .str.replace('}', '\\}', regex=False) \
                                                    .str.replace('~', '\\textasciitilde{}', regex=False) \
                                                    .str.replace('^', '\\textasciicircum{}', regex=False) \
                                                    .str.replace('\\', '\\textbackslash{}', regex=False)


        num_cols = len(df_for_latex.columns)
        col_fmt = '|' + 'l|' * num_cols if num_cols > 0 else '|l|'

        latex_table = df_for_latex.to_latex(
            index=False, escape=False, # We handled escaping manually
            column_format=col_fmt, header=True, longtable=True, na_rep="-",
            caption='Summary of Simulation Metrics Over Time.', label='tab:simsummary'
        )

        config_summary_tex = ""
        if sim_config_params and isinstance(sim_config_params, dict):
            config_summary_tex += "\\subsection*{Simulation Configuration}\n\\begin{itemize}\n"
            config_items_to_report = {
                "TEAM_SIZE": "Team Size",
                "SHIFT_DURATION_MINUTES": "Shift Duration (min)",
                "MINUTES_PER_INTERVAL": "Minutes per Interval",
                "TEAM_INITIATIVE": "Team Initiative"
            }
            for key, desc in config_items_to_report.items():
                val = sim_config_params.get(key)
                if val is not None:
                    val_str = str(val).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&').replace('#', '\\#') # Basic escape
                    config_summary_tex += f"    \\item \\textbf{{{desc}:}} {val_str}\n"
            
            num_events = len(sim_config_params.get("SCHEDULED_EVENTS", []))
            config_summary_tex += f"    \\item \\textbf{{Number of Scheduled Events:}} {num_events}\n"
            config_summary_tex += "\\end{itemize}\n"

        latex_document = f"""
\\documentclass[10pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amsfonts,amssymb,graphicx,booktabs,longtable,array,xcolor}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{fancyhdr}}
\\usepackage{{hyperref}}
\\hypersetup{{
    colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan,
    pdftitle={{Workplace Shift Monitoring Report}}, pdfauthor={{Automated Simulation System}}, 
    pdfsubject={{Simulation Report}}, pdfkeywords={{simulation, workplace, optimization}}
}}

\\pagestyle{{fancy}} \\fancyhf{{}}
\\fancyhead[C]{{Workplace Shift Monitoring Report}} \\fancyfoot[C]{{\\thepage}}
\\renewcommand{{\\headrulewidth}}{{0.4pt}} \\renewcommand{{\\footrulewidth}}{{0.4pt}}

\\title{{Workplace Shift Monitoring Report}}
\\author{{Automated Simulation System}} \\date{{\\today}}

\\begin{{document}}
\\maketitle \\thispagestyle{{empty}} \\clearpage
\\pagenumbering{{arabic}}

{config_summary_tex}

\\section*{{Simulation Data Summary}}
The table below presents a time-series summary of key metrics from the simulation run.

\\begingroup \\centering
{latex_table}
\\endgroup

\\end{{document}}
"""
        report_filename = "workplace_report.tex"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(latex_document)
        logger.info(f"LaTeX report '{report_filename}' generated successfully.", extra={'user_action': 'Generate LaTeX Report - Success'})

    except Exception as e:
        logger.error(f"Failed to generate LaTeX report: {e}", exc_info=True, extra={'user_action': 'Generate LaTeX Report - Error'})
