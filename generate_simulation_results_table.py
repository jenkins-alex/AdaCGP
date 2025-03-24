#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd
import numpy as np

# Configuration paths
# Results directories
sweep_dir_adacgp = 'logs/AdaCGP/cgp_simulated/best_sweep_mc/2024-09-30/14-45-57'
sweep_dir_tiso = 'logs/TISO/cgp_simulated/best_sweep_mc/2025-03-20/13-42-55/nmse_pred_alg1'
sweep_dir_tirso = 'logs/TIRSO/cgp_simulated/best_sweep_mc/2025-03-20/13-42-48/nmse_pred_alg1'
sweep_dir_glasso = 'logs/GLasso/cgp_simulated/best_sweep_mc/2025-03-20/12-33-22/nmse_pred_alg1'
sweep_dir_glsigrep = 'logs/GLSigRep/cgp_simulated/best_sweep_mc/2025-03-21/09-48-31/nmse_pred_alg1'  # Added GL-SigRep
sweep_dir_var = 'logs/VAR/cgp_simulated/best_sweep_mc/2025-03-20/15-55-33/nmse_pred_alg1'
sweep_dir_sdsem = 'logs/SDSEM/cgp_simulated/best_sweep_mc/2025-03-20/16-47-04/nmse_pred_alg1'

# Node number parameter
N = '50'  # Can be changed as needed
best_error = 'nmse_pred_from_h'  # Optimization metric

# Helper functions for data processing
def walk_sweep_dirs(sweep_dir):
    """Extract results from a sweep directory structure."""
    results = {}
    for root, dirs, files in os.walk(sweep_dir):
        for file in files:
            if file == 'eval_results.txt':
                subdir = root[len(sweep_dir)+1:]
                parts = subdir.split('/')
                if 'subdir' not in results:
                    results['subdir'] = []
                results['subdir'].append(subdir)
                for part in parts:
                    subparts = part.split('_')
                    value = subparts[-1]
                    key = '_'.join(subparts[:-1])
                    if key == '':
                        key = 'seed'
                    if key not in results:
                        results[key] = []
                    results[key].append(value)

                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        col, result = line.split(':')
                        col = col.strip()
                        result = result.strip()
                        result = float(result)
                        if col not in results:
                            results[col] = []
                        results[col].append(result)

    df = pd.DataFrame(results)
    return df

def split_by_algorithm(df):
    """Separate results by algorithm type."""
    # Get columns ending in 'alg1' and 'alg2' into two dfs
    cols = df.columns
    alg1_cols = [col for col in cols if col.endswith('_alg1')]
    alg2_cols = [col for col in cols if col.endswith('_alg2')]
    df_alg1 = df[alg1_cols]
    df_alg2 = df[alg2_cols]

    # Remove '_alg1' and '_alg2' from column names if present
    df_alg1.columns = [col.replace('_alg1', '') for col in df_alg1.columns]
    df_alg2.columns = [col.replace('_alg2', '') for col in df_alg2.columns]
    
    # Add back cols from original df that are not alg1 or alg2
    missing_cols = [col for col in cols if col not in alg1_cols and col not in alg2_cols]
    df_alg1 = pd.concat([df[missing_cols], df_alg1], axis=1)
    df_alg2 = pd.concat([df[missing_cols], df_alg2], axis=1)

    # Add algorithm column
    df_alg1['algorithm'] = 'alg1'
    df_alg2['algorithm'] = 'alg2'

    # Merge the two dfs
    df_alg = pd.concat([df_alg1, df_alg2], ignore_index=True)
    # Drop nan
    df_alg = df_alg.dropna()
    return df_alg

def format_mean_std(mean, std):
    """Format mean and standard deviation for the table."""
    return "{:.2f}".format(mean) + "{\\scriptsize$\\pm$" + "{:.2f}".format(std) + "}"

def split_alternate(df):
    """Process the alternate algorithm results."""
    split = split_by_algorithm(df.reset_index())
    split = split[split['algorithm'] == 'alg2']
    split['algorithm'] = 'alg3'
    return split

def split_not_alternate(df):
    """Process the non-alternate algorithm results."""
    split = split_by_algorithm(df.reset_index())
    return split

def generate_improved_table(df_main_res, N, output_file='tables/improved_graph_estimation.tex'):
    """
    Generate a professionally formatted LaTeX table with graph topology results.
    Highlights the best results in bold and second best results with underline.
    
    Parameters:
    -----------
    df_main_res : DataFrame
        The processed results dataframe
    N : int or str
        The N value for the table caption
    output_file : str
        Path to save the LaTeX table
    """
    # Define metrics and their display names
    metrics = ['nmse_pred', 'nmse_w', 'p_miss', 'p_false_alarm']
    metric_names = {
        'nmse_pred': 'NMSE',
        'nmse_w': 'NMSE$(\\mathbf{W})$',
        'p_miss': 'P$_{M}$',
        'p_false_alarm': 'P$_{FA}$'
    }
    
    # Algorithm display names (shorter for table)
    alg_display = {
        'AdaCGP-P1-DB': 'P1 + DB',
        'AdaCGP-P1-ADB': 'P1 + ADB',
        'AdaCGP-P2-DB': 'P2 + DB',
        'AdaCGP-P2-ADB': 'P2 + ADB',
        'TISO': 'TISO',
        'TIRSO': 'TIRSO',
        'GLasso': 'GLasso',
        'GL-SigRep': 'GL-SigRep',  # Added GL-SigRep
        'VAR': 'VAR',
        'SD-SEM': 'SD-SEM'
    }
    
    # Graph type display names
    graph_display = {
        'ER': 'Erdos-Renyi',
        'KR': 'K-Regular',
        'RANDOM': 'Random',
        'SBM': 'Stochastic Block Model'
    }
    
    # New order for graph types
    graph_types = ['RANDOM', 'ER', 'KR', 'SBM']
    
    # Ordered list of algorithms to display (GLasso at the top, followed by GL-SigRep, then VAR, then SD-SEM)
    algorithms = ['GLasso', 'GL-SigRep', 'VAR', 'SD-SEM', 'TIRSO', 'TISO', 'AdaCGP-P2-ADB', 'AdaCGP-P2-DB', 'AdaCGP-P1-ADB', 'AdaCGP-P1-DB']
    
    # Helper function to extract numeric values from formatted strings
    def extract_value(value_str):
        if value_str == "---" or pd.isna(value_str):
            return float('inf')  # Use infinity for missing values
        
        # If it's a formatted string with mean±std
        if "{\\scriptsize$\\pm$" in value_str:
            mean_str = value_str.split("{\\scriptsize$\\pm$")[0]
            try:
                return float(mean_str)
            except:
                return float('inf')
        else:
            try:
                return float(value_str)
            except:
                return float('inf')
    
    # Function to format with bold or underline
    def format_with_emphasis(value_str, emphasis):
        if value_str == "---" or pd.isna(value_str):
            return "---"
        
        if emphasis == "bold":
            return "\\textbf{" + value_str + "}"
        elif emphasis == "underline":
            return "\\underline{" + value_str + "}"
        else:
            return value_str
    
    # Simplify uncertainty values with consistent decimal places
    def clean_value(value_str):
        # Handle NaN or missing values
        if value_str == "---" or pd.isna(value_str):
            return "---"
            
        # Extract the mean and std parts
        if "{\\scriptsize$\\pm$" in value_str:
            parts = value_str.split("{\\scriptsize$\\pm$")
            mean_str = parts[0]
            std_str = parts[1].replace("}", "")
            
            # Convert to float for processing
            mean = float(mean_str)
            std = float(std_str)
            
            # Always use 2 decimal places for consistency
            return "{:.2f}".format(mean) + "{\\scriptsize$\\pm$" + "{:.2f}".format(std) + "}"
        else:
            # If there's no uncertainty, just return the value with 2 decimal places
            try:
                return "{:.2f}".format(float(value_str))
            except:
                return value_str

    # Replace the existing find_best_indices function with this modified version
    def find_best_indices(data_list, metric):
        # Filter out "---" entries before comparison
        valid_data = [(i, x) for i, x in enumerate(data_list) if x != "---" and not pd.isna(x)]
        
        if not valid_data:
            return [], []
        
        # Extract values for comparison
        indices = [i for i, _ in valid_data]
        values = [extract_value(x) for _, x in valid_data]
        
        # For all metrics, lower is better
        sorted_pairs = sorted(zip(indices, values), key=lambda pair: pair[1])
        
        # Filter out infinity values (missing data)
        valid_indices = [i for i, v in sorted_pairs if v != float('inf')]
        
        if len(valid_indices) >= 2:
            return [valid_indices[0]], [valid_indices[1]]
        elif len(valid_indices) == 1:
            return [valid_indices[0]], []
        else:
            return [], []

    # Also update the find_best_combined_indices function to skip the methods with dashes
    def find_best_combined_indices(p_miss_list, p_fa_list, nmse_w_list):
        # Create a list of indices where neither p_miss nor p_fa is "---" or NaN
        valid_indices = [i for i, (pm, pfa) in enumerate(zip(p_miss_list, p_fa_list)) 
                    if pm != "---" and pfa != "---" and not pd.isna(pm) and not pd.isna(pfa)]
        
        if not valid_indices:
            return [], []
        
        # Extract values for P_M, P_FA and NMSE_W
        p_miss_values = [extract_value(p_miss_list[i]) for i in valid_indices]
        p_fa_values = [extract_value(p_fa_list[i]) for i in valid_indices]
        nmse_w_values = [extract_value(nmse_w_list[i]) if nmse_w_list[i] != "---" and not pd.isna(nmse_w_list[i]) 
                    else float('inf') for i in valid_indices]
        
        # Calculate combined metric (P_M + P_FA)
        combined_values = []
        for i in range(len(valid_indices)):
            if p_miss_values[i] == float('inf') or p_fa_values[i] == float('inf'):
                combined_values.append(float('inf'))
            else:
                combined_values.append(p_miss_values[i] + p_fa_values[i])
        
        # Rest of the function remains the same, but using our filtered valid_indices...
        
        # Find algorithms with lowest combined value
        min_value = min(combined_values) if not all(v == float('inf') for v in combined_values) else float('inf')
        min_indices_relative = [i for i, v in enumerate(combined_values) if v == min_value and v != float('inf')]
        min_indices = [valid_indices[i] for i in min_indices_relative]  # Convert back to original indices
        
        # If there's a tie, use NMSE_W as tiebreaker
        if len(min_indices) > 1:
            # Get NMSE_W values for tied algorithms
            tie_nmse_indices = [i for i, idx in enumerate(min_indices_relative) if nmse_w_values[idx] != float('inf')]
            
            if tie_nmse_indices:
                tie_nmse_values = [nmse_w_values[min_indices_relative[i]] for i in tie_nmse_indices]
                best_relative_idx = tie_nmse_indices[tie_nmse_values.index(min(tie_nmse_values))]
                best_indices = [valid_indices[min_indices_relative[best_relative_idx]]]
                
                # Remove the best from consideration for second best
                remaining_indices = [idx for idx in min_indices_relative if idx != min_indices_relative[best_relative_idx]]
                
                if remaining_indices:
                    remaining_nmse_indices = [i for i, idx in enumerate(remaining_indices) if nmse_w_values[idx] != float('inf')]
                    if remaining_nmse_indices:
                        remaining_nmse_values = [nmse_w_values[remaining_indices[i]] for i in remaining_nmse_indices]
                        second_best_relative_idx = remaining_nmse_indices[remaining_nmse_values.index(min(remaining_nmse_values))]
                        second_best_indices = [valid_indices[remaining_indices[second_best_relative_idx]]]
                    else:
                        second_best_indices = []
                else:
                    second_best_indices = []
            else:
                # If we can't use NMSE_W to break the tie, just pick the first one
                best_indices = [valid_indices[min_indices_relative[0]]]
                second_best_indices = [valid_indices[min_indices_relative[1]]] if len(min_indices_relative) > 1 else []
        else:
            if min_indices:
                best_indices = [min_indices[0]]
                
                # Find second best
                remaining_combined_values = combined_values.copy()
                for i, idx in enumerate(min_indices_relative):
                    remaining_combined_values[i] = float('inf')
                
                second_min = min(remaining_combined_values) if not all(v == float('inf') for v in remaining_combined_values) else float('inf')
                second_min_indices_relative = [i for i, v in enumerate(remaining_combined_values) if v == second_min and v != float('inf')]
                
                if second_min_indices_relative:
                    if len(second_min_indices_relative) > 1:
                        # Tiebreaker for second place
                        second_tie_nmse_indices = [i for i in second_min_indices_relative if nmse_w_values[i] != float('inf')]
                        if second_tie_nmse_indices:
                            second_tie_nmse_values = [nmse_w_values[i] for i in second_tie_nmse_indices]
                            second_best_relative_idx = second_tie_nmse_indices[second_tie_nmse_values.index(min(second_tie_nmse_values))]
                            second_best_indices = [valid_indices[second_best_relative_idx]]
                        else:
                            second_best_indices = [valid_indices[second_min_indices_relative[0]]]
                    else:
                        second_best_indices = [valid_indices[second_min_indices_relative[0]]]
                else:
                    second_best_indices = []
            else:
                best_indices = []
                second_best_indices = []
        
        return best_indices, second_best_indices
    
    # Generate LaTeX table
    with open(output_file, 'w') as f:
        # Write table header with improved formatting
        header = "\\begin{table*}[t]\n"
        header += "\\centering\n"
        header += "\\caption{Comparison of graph topology estimation algorithms for $N = " + str(N) + "$. Lower values are better for all metrics. \\\Best results are in \\textbf{bold}, second best are \\underline{underlined}.}\n"
        header += "\\label{tab:results_table}\n"
        header += "\\setlength{\\tabcolsep}{6pt}  % Increased horizontal spacing\n"
        header += "\\setlength{\\aboverulesep}{0pt}\n"
        header += "\\setlength{\\belowrulesep}{0pt}\n"
        header += "\\renewcommand{\\arraystretch}{1.15}\n"
        f.write(header)
        
        # Define column structure with spacer columns for better separation
        column_count = len(metrics)
        column_def = "@{}l|" + "".join(["c!{\\hspace{.5em}}" for _ in range(column_count-1)]) + "c" + "!{\\hspace{.5em}}|" + "".join(["c!{\\hspace{.5em}}" for _ in range(column_count-1)]) + "c" + "@{}}"
        f.write("\\begin{tabular}{" + column_def + "\n")
        f.write("\\toprule[1pt]\\midrule[0.3pt]\n")
        
        # Create header row for the first two graph types (RANDOM and ER)
        f.write("& \\multicolumn{" + str(column_count) + "}{c}{\\textsc{" + graph_display['RANDOM'] + "}} ")
        f.write("& \\multicolumn{" + str(column_count) + "}{c}{\\textsc{" + graph_display['ER'] + "}} \\\\\n")
        f.write("\\cmidrule(lr){2-" + str(column_count+1) + "} \\cmidrule(lr){" + str(column_count+2) + "-" + str(2*column_count+1) + "}\n")
        
        # Add metric names to column headers
        f.write("Algorithm")
        for _ in range(2):  # For both Random and ER
            for metric in metrics:
                f.write(" & " + metric_names[metric])
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Collect all values for RANDOM and ER first
        graph_values = {}
        best_indices = {}
        
        for graph in ['RANDOM', 'ER']:
            # Initialize data structures for this graph
            graph_values[graph] = {
                'nmse_pred': [],
                'nmse_w': [],
                'p_miss': [],
                'p_false_alarm': []
            }
            
            # Collect all values for this graph
            for alg in algorithms:
                graph_data = df_main_res.loc[graph].reset_index(drop=True)
                alg_data = graph_data[graph_data['algorithm'] == alg]
                
                for metric in metrics:
                    if not alg_data.empty and metric in alg_data.columns:
                        value = clean_value(str(alg_data[metric].values[0]))
                        graph_values[graph][metric].append(value)
                    else:
                        graph_values[graph][metric].append("---")
            
            # Find best and second best for each metric
            best_indices[graph] = {
                'nmse_pred': find_best_indices(graph_values[graph]['nmse_pred'], 'nmse_pred'),
                'nmse_w': find_best_indices(graph_values[graph]['nmse_w'], 'nmse_w'),
                'combined': find_best_combined_indices(
                    graph_values[graph]['p_miss'], 
                    graph_values[graph]['p_false_alarm'],
                    graph_values[graph]['nmse_w']
                )
            }
        
        # Now write rows with data from both graphs
        for alg_idx, alg in enumerate(algorithms):
            row_text = alg_display.get(alg, alg)
            
            # First add RANDOM graph data
            graph = 'RANDOM'
            graph_data = df_main_res.loc[graph].reset_index(drop=True)
            alg_data = graph_data[graph_data['algorithm'] == alg]
            
            for i, metric in enumerate(metrics):
                if not alg_data.empty and metric in alg_data.columns:
                    value = clean_value(str(alg_data[metric].values[0]))
                    
                    # Apply emphasis based on metric
                    emphasis = None
                    if metric == 'nmse_pred':
                        if alg_idx in best_indices[graph]['nmse_pred'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['nmse_pred'][1]:
                            emphasis = "underline"
                    elif metric == 'nmse_w':
                        if alg_idx in best_indices[graph]['nmse_w'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['nmse_w'][1]:
                            emphasis = "underline"
                    elif metric in ['p_miss', 'p_false_alarm']:
                        if alg_idx in best_indices[graph]['combined'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['combined'][1]:
                            emphasis = "underline"
                    
                    row_text += " & " + format_with_emphasis(value, emphasis)
                else:
                    row_text += " & ---"
            
            # Then add ER graph data
            graph = 'ER'
            graph_data = df_main_res.loc[graph].reset_index(drop=True)
            alg_data = graph_data[graph_data['algorithm'] == alg]
            
            for i, metric in enumerate(metrics):
                if not alg_data.empty and metric in alg_data.columns:
                    value = clean_value(str(alg_data[metric].values[0]))
                    
                    # Apply emphasis based on metric
                    emphasis = None
                    if metric == 'nmse_pred':
                        if alg_idx in best_indices[graph]['nmse_pred'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['nmse_pred'][1]:
                            emphasis = "underline"
                    elif metric == 'nmse_w':
                        if alg_idx in best_indices[graph]['nmse_w'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['nmse_w'][1]:
                            emphasis = "underline"
                    elif metric in ['p_miss', 'p_false_alarm']:
                        if alg_idx in best_indices[graph]['combined'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['combined'][1]:
                            emphasis = "underline"
                    
                    row_text += " & " + format_with_emphasis(value, emphasis)
                else:
                    row_text += " & ---"
            
            row_text += " \\\\\n"
            f.write(row_text)
        
        # Add vertical space between the two graph sections
        f.write("\\midrule\n")
        f.write("\\addlinespace[1em]  % Added vertical space\n")
        
        # Create header row for the second two graph types (KR and SBM)
        f.write("& \\multicolumn{" + str(column_count) + "}{c}{\\textsc{" + graph_display['KR'] + "}} ")
        f.write("& \\multicolumn{" + str(column_count) + "}{c}{\\textsc{" + graph_display['SBM'] + "}} \\\\\n")
        f.write("\\cmidrule(lr){2-" + str(column_count+1) + "} \\cmidrule(lr){" + str(column_count+2) + "-" + str(2*column_count+1) + "}\n")
        
        # Add metric names to column headers for KR and SBM
        f.write("Algorithm")
        for _ in range(2):  # For both KR and SBM
            for metric in metrics:
                f.write(" & " + metric_names[metric])
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Collect all values for KR and SBM first
        graph_values = {}
        best_indices = {}
        
        for graph in ['KR', 'SBM']:
            # Initialize data structures for this graph
            graph_values[graph] = {
                'nmse_pred': [],
                'nmse_w': [],
                'p_miss': [],
                'p_false_alarm': []
            }
            
            # Collect all values for this graph
            for alg in algorithms:
                graph_data = df_main_res.loc[graph].reset_index(drop=True)
                alg_data = graph_data[graph_data['algorithm'] == alg]
                
                for metric in metrics:
                    if not alg_data.empty and metric in alg_data.columns:
                        value = clean_value(str(alg_data[metric].values[0]))
                        graph_values[graph][metric].append(value)
                    else:
                        graph_values[graph][metric].append("---")
            
            # Find best and second best for each metric
            best_indices[graph] = {
                'nmse_pred': find_best_indices(graph_values[graph]['nmse_pred'], 'nmse_pred'),
                'nmse_w': find_best_indices(graph_values[graph]['nmse_w'], 'nmse_w'),
                'combined': find_best_combined_indices(
                    graph_values[graph]['p_miss'], 
                    graph_values[graph]['p_false_alarm'],
                    graph_values[graph]['nmse_w']
                )
            }
        
        # Now write rows with data from both graphs
        for alg_idx, alg in enumerate(algorithms):
            row_text = alg_display.get(alg, alg)
            
            # First add KR graph data
            graph = 'KR'
            graph_data = df_main_res.loc[graph].reset_index(drop=True)
            alg_data = graph_data[graph_data['algorithm'] == alg]
            
            for i, metric in enumerate(metrics):
                if not alg_data.empty and metric in alg_data.columns:
                    value = clean_value(str(alg_data[metric].values[0]))
                    
                    # Apply emphasis based on metric
                    emphasis = None
                    if metric == 'nmse_pred':
                        if alg_idx in best_indices[graph]['nmse_pred'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['nmse_pred'][1]:
                            emphasis = "underline"
                    elif metric == 'nmse_w':
                        if alg_idx in best_indices[graph]['nmse_w'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['nmse_w'][1]:
                            emphasis = "underline"
                    elif metric in ['p_miss', 'p_false_alarm']:
                        if alg_idx in best_indices[graph]['combined'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['combined'][1]:
                            emphasis = "underline"
                    
                    row_text += " & " + format_with_emphasis(value, emphasis)
                else:
                    row_text += " & ---"
            
            # Then add SBM graph data
            graph = 'SBM'
            graph_data = df_main_res.loc[graph].reset_index(drop=True)
            alg_data = graph_data[graph_data['algorithm'] == alg]
            
            for i, metric in enumerate(metrics):
                if not alg_data.empty and metric in alg_data.columns:
                    value = clean_value(str(alg_data[metric].values[0]))
                    
                    # Apply emphasis based on metric
                    emphasis = None
                    if metric == 'nmse_pred':
                        if alg_idx in best_indices[graph]['nmse_pred'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['nmse_pred'][1]:
                            emphasis = "underline"
                    elif metric == 'nmse_w':
                        if alg_idx in best_indices[graph]['nmse_w'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['nmse_w'][1]:
                            emphasis = "underline"
                    elif metric in ['p_miss', 'p_false_alarm']:
                        if alg_idx in best_indices[graph]['combined'][0]:
                            emphasis = "bold"
                        elif alg_idx in best_indices[graph]['combined'][1]:
                            emphasis = "underline"
                    
                    row_text += " & " + format_with_emphasis(value, emphasis)
                else:
                    row_text += " & ---"
            
            row_text += " \\\\\n"
            f.write(row_text)
        
        # Close the tabular environment
        f.write("\\midrule[0.3pt]\\toprule[1pt]\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}")
    
    print(f"Improved LaTeX table generated at {output_file}")
    return output_file

def main():
    """Generate the main results table comparing all algorithms."""
    # Process AdaCGP data
    df_adacgp = walk_sweep_dirs(sweep_dir_adacgp)
    df_alternate = df_adacgp[df_adacgp['alternate'] == 'True']
    df_not_alternate = df_adacgp[df_adacgp['alternate'] == 'False']
    
    # Calculate statistics for AdaCGP
    adacgp_alternate_mean = df_alternate.groupby(['graph', 'N', 'use_path_1']).median(numeric_only=True)
    adacgp_alternate_std = df_alternate.groupby(['graph', 'N', 'use_path_1']).quantile(0.75, numeric_only=True) - df_alternate.groupby(['graph', 'N', 'use_path_1']).quantile(0.25, numeric_only=True)
    adacgp_not_alternate_mean = df_not_alternate.groupby(['graph', 'N', 'use_path_1']).median(numeric_only=True)
    adacgp_not_alternate_std = df_not_alternate.groupby(['graph', 'N', 'use_path_1']).quantile(0.75, numeric_only=True) - df_not_alternate.groupby(['graph', 'N', 'use_path_1']).quantile(0.25, numeric_only=True)

    # Process TISO, TIRSO, GLasso, GL-SigRep, VAR, and SD-SEM data
    df_tirso = split_by_algorithm(walk_sweep_dirs(sweep_dir_tirso))
    df_tiso = split_by_algorithm(walk_sweep_dirs(sweep_dir_tiso))
    df_glasso = split_by_algorithm(walk_sweep_dirs(sweep_dir_glasso))
    df_glsigrep = split_by_algorithm(walk_sweep_dirs(sweep_dir_glsigrep))  # Added GL-SigRep
    df_var = split_by_algorithm(walk_sweep_dirs(sweep_dir_var))
    df_sdsem = split_by_algorithm(walk_sweep_dirs(sweep_dir_sdsem))

    tiso_mean = df_tiso.groupby(by=['graph', 'N']).median(numeric_only=True)
    tiso_std = df_tiso.groupby(by=['graph', 'N']).quantile(0.75, numeric_only=True) - df_tiso.groupby(by=['graph', 'N']).quantile(0.25, numeric_only=True)
    tirso_mean = df_tirso.groupby(by=['graph', 'N']).median(numeric_only=True)
    tirso_std = df_tirso.groupby(by=['graph', 'N']).quantile(0.75, numeric_only=True) - df_tirso.groupby(by=['graph', 'N']).quantile(0.25, numeric_only=True)
    glasso_mean = df_glasso.groupby(by=['graph', 'N']).median(numeric_only=True)
    glasso_std = df_glasso.groupby(by=['graph', 'N']).quantile(0.75, numeric_only=True) - df_glasso.groupby(by=['graph', 'N']).quantile(0.25, numeric_only=True)
    glsigrep_mean = df_glsigrep.groupby(by=['graph', 'N']).median(numeric_only=True)  # GL-SigRep mean
    glsigrep_std = df_glsigrep.groupby(by=['graph', 'N']).quantile(0.75, numeric_only=True) - df_glsigrep.groupby(by=['graph', 'N']).quantile(0.25, numeric_only=True)  # GL-SigRep std
    var_mean = df_var.groupby(by=['graph', 'N']).median(numeric_only=True)
    var_std = df_var.groupby(by=['graph', 'N']).quantile(0.75, numeric_only=True) - df_var.groupby(by=['graph', 'N']).quantile(0.25, numeric_only=True)
    sdsem_mean = df_sdsem.groupby(by=['graph', 'N']).median(numeric_only=True)
    sdsem_std = df_sdsem.groupby(by=['graph', 'N']).quantile(0.75, numeric_only=True) - df_sdsem.groupby(by=['graph', 'N']).quantile(0.25, numeric_only=True)

    # Process the AdaCGP results
    split_alternate_mean = split_alternate(adacgp_alternate_mean)
    split_alternate_std = split_alternate(adacgp_alternate_std)
    split_not_alternate_mean = split_not_alternate(adacgp_not_alternate_mean)
    split_not_alternate_std = split_not_alternate(adacgp_not_alternate_std)

    # Join the split means and stds
    split_mean = pd.concat([split_alternate_mean, split_not_alternate_mean])
    split_std = pd.concat([split_alternate_std, split_not_alternate_std])

    split_mean = split_mean.groupby(['graph', 'N', 'use_path_1', 'algorithm']).mean()
    split_std = split_std.groupby(['graph', 'N', 'use_path_1', 'algorithm']).mean()

    # Filter for N=50
    adacgp_N50_path1 = split_mean.loc[(slice(None), N, 'True', slice(None)), :]
    adacgp_N50_path2 = split_mean.loc[(slice(None), N, 'False', slice(None)), :]
    tiso_N50 = tiso_mean.loc[(slice(None), N), :]
    tirso_N50 = tirso_mean.loc[(slice(None), N), :]
    glasso_N50 = glasso_mean.loc[(slice(None), N), :]
    glsigrep_N50 = glsigrep_mean.loc[(slice(None), N), :]  # Filter GL-SigRep for N=50
    var_N50 = var_mean.loc[(slice(None), N), :]
    sdsem_N50 = sdsem_mean.loc[(slice(None), N), :]

    # Join with std data
    adacgp_N50_path1 = adacgp_N50_path1.join(split_std.loc[(slice(None), N, 'True', slice(None)), :], rsuffix='_std')
    adacgp_N50_path2 = adacgp_N50_path2.join(split_std.loc[(slice(None), N, 'False', slice(None)), :], rsuffix='_std')
    tiso_N50 = tiso_N50.join(tiso_std.loc[(slice(None), N), :], rsuffix='_std')
    tirso_N50 = tirso_N50.join(tirso_std.loc[(slice(None), N), :], rsuffix='_std')
    glasso_N50 = glasso_N50.join(glasso_std.loc[(slice(None), N), :], rsuffix='_std')
    glsigrep_N50 = glsigrep_N50.join(glsigrep_std.loc[(slice(None), N), :], rsuffix='_std')  # Join GL-SigRep with std data
    var_N50 = var_N50.join(var_std.loc[(slice(None), N), :], rsuffix='_std')
    sdsem_N50 = sdsem_N50.join(sdsem_std.loc[(slice(None), N), :], rsuffix='_std')

    # Replace NMSE values with dashes for GLasso, GL-SigRep, and SD-SEM
    if 'nmse_pred' in glasso_N50.columns:
        glasso_N50['nmse_pred'] = '---'
        
    if 'nmse_pred' in glsigrep_N50.columns:
        glsigrep_N50['nmse_pred'] = '---'
        
    if 'nmse_pred' in sdsem_N50.columns:
        sdsem_N50['nmse_pred'] = '---'

    # Apply the formatting function to all datasets
    for df in [adacgp_N50_path1, adacgp_N50_path2, tiso_N50, tirso_N50, glasso_N50, glsigrep_N50, var_N50, sdsem_N50]:
        for col in df.columns:
            if col.endswith('_std'):
                continue
            
            # Skip formatting for columns that are already set to '---'
            if df[col].astype(str).eq('---').any():
                continue
                
            std_col = col + '_std'
            if std_col in df.columns:  # Make sure the std column exists
                df[col] = df.apply(lambda row: format_mean_std(row[col], row[std_col]), axis=1)
                df.drop(columns=[std_col], inplace=True)

    # Clean up indices
    adacgp_N50_path1.index = adacgp_N50_path1.index.droplevel('use_path_1')
    adacgp_N50_path2.index = adacgp_N50_path2.index.droplevel('use_path_1')
    adacgp_N50_path1.index = adacgp_N50_path1.index.droplevel('N')
    adacgp_N50_path2.index = adacgp_N50_path2.index.droplevel('N')
    tiso_N50.index = tiso_N50.index.droplevel('N')
    tirso_N50.index = tirso_N50.index.droplevel('N')
    glasso_N50.index = glasso_N50.index.droplevel('N')
    glsigrep_N50.index = glsigrep_N50.index.droplevel('N')  # Clean up GL-SigRep index
    var_N50.index = var_N50.index.droplevel('N')
    sdsem_N50.index = sdsem_N50.index.droplevel('N')

    # Rename algorithms
    name_changes_path_1 = {'alg1': 'AdaCGP-P1', 'alg2': 'AdaCGP-P1-DB', 'alg3': 'AdaCGP-P1-ADB'}
    name_changes_path_2 = {'alg1': 'AdaCGP-P2', 'alg2': 'AdaCGP-P2-DB', 'alg3': 'AdaCGP-P2-ADB'}
    adacgp_N50_path1.index = adacgp_N50_path1.index.set_levels(adacgp_N50_path1.index.levels[-1].map(name_changes_path_1), level=-1)
    adacgp_N50_path2.index = adacgp_N50_path2.index.set_levels(adacgp_N50_path2.index.levels[-1].map(name_changes_path_2), level=-1)

    # Make the algorithm index a column
    adacgp_N50_path1.reset_index(level=-1, inplace=True)
    adacgp_N50_path2.reset_index(level=-1, inplace=True)

    # Add algorithm column to TISO/TIRSO/GLasso/GL-SigRep/VAR/SD-SEM
    tiso_N50['algorithm'] = 'TISO'
    tirso_N50['algorithm'] = 'TIRSO'
    glasso_N50['algorithm'] = 'GLasso'
    glsigrep_N50['algorithm'] = 'GL-SigRep'  # Add algorithm column for GL-SigRep
    var_N50['algorithm'] = 'VAR'
    sdsem_N50['algorithm'] = 'SD-SEM'

    # Concat all dataframes for final result
    df_main_res = pd.concat([adacgp_N50_path1, adacgp_N50_path2, tiso_N50, tirso_N50, glasso_N50, glsigrep_N50, var_N50, sdsem_N50], axis=0)  # Added GL-SigRep to concat

    # Drop unnecessary columns
    if 'pce' in df_main_res.columns:
        df_main_res = df_main_res.drop(columns=['pce'])

    # Generate formatted LaTeX table
    output_file = f'tables/graph_estimation_results_N{N}.tex'
    generate_improved_table(df_main_res, N, output_file)

    # Also save as standard LaTeX
    simple_output = f'tables/simple_graph_estimation_results_N{N}.tex'
    df_main_res.to_latex(simple_output, escape=False)
    print(f"Standard LaTeX table generated at {simple_output}")

if __name__ == '__main__':
    main()