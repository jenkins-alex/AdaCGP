import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def read_complexity_results(results_dir):
    # Initialize basic configuration data
    data = {
        'cfg_path': [],
        'model': [],
        'graph_type': [],
        'N': [],
        'frac_non_zero': [],
        'seed': []
    }
    
    # First pass: discover all metrics and time steps
    all_metrics = set()
    all_time_steps = set()
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'eval_results.txt':
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if ':' in line:
                            key, _ = line.strip().split(':', 1)
                            key = key.strip()
                            
                            # Extract algorithm and time step using regex
                            match = re.match(r'(iteration_\w+)_(\w+)_T=(\d+)', key)
                            if match:
                                metric_type, alg, time_step = match.groups()
                                metric_key = f"{metric_type}_{alg}"
                                all_metrics.add(metric_key)
                                all_time_steps.add(int(time_step))
    
    # Sort time steps for better organization
    all_time_steps = sorted(all_time_steps)
    
    # Initialize all flattened metric columns
    for metric in all_metrics:
        for time_step in all_time_steps:
            column_name = f"{metric}_T{time_step}"
            data[column_name] = []
    
    # Second pass: collect data
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == 'eval_results.txt':
                cfg_path = os.path.join(root, '.hydra', 'config.yaml')
                data['cfg_path'].append(cfg_path)

                # Remove the results_dir from root
                _root = root.replace(results_dir, '')
                _root = _root[1:]
                
                # Parse directory structure
                parts = _root.split('/')
                data['model'].append(parts[0])
                data['graph_type'].append(parts[2])
                data['N'].append(parts[3].split('_')[-1])
                data['frac_non_zero'].append(parts[4].split('_')[-1])
                data['seed'].append(parts[-1].split('_')[-1])
                
                # Initialize metric values dictionary
                metric_values = {}
                
                # Read the eval_results.txt file
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            key = key.strip()
                            value = float(value.strip())
                            
                            # Extract algorithm and time step using regex
                            match = re.match(r'(iteration_\w+)_(\w+)_T=(\d+)', key)
                            if match:
                                metric_type, alg, time_step = match.groups()
                                time_step = int(time_step)
                                metric_key = f"{metric_type}_{alg}"
                                column_name = f"{metric_key}_T{time_step}"
                                metric_values[column_name] = value
                
                # Add values to the appropriate columns (with NaN for missing values)
                for metric in all_metrics:
                    for time_step in all_time_steps:
                        column_name = f"{metric}_T{time_step}"
                        data[column_name].append(metric_values.get(column_name, np.nan))
    
    # Convert to a DataFrame
    df = pd.DataFrame(data)
    
    # Convert N to numeric
    df['N'] = pd.to_numeric(df['N'])    
    return df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

def plot_unified_complexity_analysis(results_df, output_path='figures/unified_complexity_analysis.svg', 
                                     time_step=9000, selected_N=50):
    """
    Creates a unified 2x2 grid showing:
    - Top left: Iteration time vs N (log-log)
    - Top right: Memory usage vs N (log-log)
    - Bottom left: Iteration time vs time steps T (log-linear)
    - Bottom right: Memory usage vs time steps T (log-linear)
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing complexity results
    output_path : str
        Path to save the output figure
    time_step : int
        Time step to use for the top row analysis
    selected_N : int
        N value to use for the bottom row analysis
    """
    # Create figure with 2 rows, 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    
    # Get unique models
    models = results_df['model'].unique()
    models = [model for model in models if model not in ['GLSigRep']]  # remove GLSigRep and GLasso to avoid clutter

    # set font
    # plt.rcParams.update({'font.size': 12})
    # plt.rcParams.update({'font.family': 'serif'})

    # cmaps and markers
    cmap_blue = plt.get_cmap('Blues')
    cmap_red = plt.get_cmap('Reds')
    
    # Updated markers and colors
    markers = {'AdaCGP_P1': 's', 'AdaCGP_P2': '^', 'SDSEM': 'o', 'TISO': 'D', 'TIRSO': 'D', 
               'VAR': 'X', 'GrangerVAR': 'p', 'GLasso': 'v'}
    
    colors = {
        'AdaCGP_P1': cmap_blue((50+10) / 90), 
        'AdaCGP_P2': cmap_red((50+10) / 90),
        'SDSEM': '#FF8C00',       # Orange
        'TISO': '#9370DB',        # Medium purple
        'TIRSO': '#4B0082',       # Indigo
        'VAR': '#75B87F',         # Light green
        'GrangerVAR': '#8B4513',  # Dark green (Sea Green)
        'GLasso': '#FF69B4',      # Light pink (Hot Pink)
        'GLSigRep': '#2E8B57'     # Brown (Saddle Brown)
    }
    
    display_labels = {
        'AdaCGP_P1': 'AdaCGP (P1)',
        'AdaCGP_P2': 'AdaCGP (P2)',
        'SDSEM': 'SD-SEM',
        'TISO': 'TISO',
        'TIRSO': 'TIRSO',
        'VAR': 'VAR',
        'GrangerVAR': 'VAR + Granger',
        'GLasso': 'GLasso',
        'GLSigRep': 'GL-SigRep'
    }
    
    line_styles = {
        'AdaCGP_P1': '-',
        'AdaCGP_P2': '-',
        'SDSEM': '-',
        'TISO': '-',
        'TIRSO': '-',
        'VAR': '--',
        'GrangerVAR': '--',
        'GLasso': '--',
        'GLSigRep': '--'
    }
    
    # PART 1: Top row - Complexity vs N
    for model in models:
        model_data = results_df[results_df['model'] == model]
        
        # Group by N and average over seeds
        grouped = model_data.groupby('N')
        
        # Extract time and memory metrics
        time_col = f'iteration_time_alg1_T{time_step}'
        memory_col = f'iteration_memory_alg1_T{time_step}'
        
        if time_col not in model_data.columns or memory_col not in model_data.columns:
            print(f"Warning: Metrics {time_col} or {memory_col} not found for model {model}")
            continue
            
        # Calculate means (excluding NaN values)
        n_values = []
        time_means = []
        memory_means = []
        
        for n, group in grouped:
            time_values = group[time_col].dropna()
            memory_values = group[memory_col].dropna()
            
            if len(time_values) > 0 and len(memory_values) > 0:
                n_values.append(n)
                time_means.append(time_values.mean())
                memory_means.append(memory_values.mean())
        
        # Sort by N for proper line plotting
        sort_indices = np.argsort(n_values)
        n_values = np.array(n_values)[sort_indices]
        time_means = np.array(time_means)[sort_indices]
        memory_means = np.array(memory_means)[sort_indices]
        
        # Plot iteration time vs N
        axs[0, 0].plot(n_values, time_means, label=display_labels[model], linestyle=line_styles[model],
                   marker=markers.get(model, 'o'), color=colors[model],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, linewidth=1.5, alpha=1 if 'Ada' in model else 0.5)

        # Plot iteration memory vs N
        axs[1, 0].plot(n_values, memory_means, label=display_labels[model], linestyle=line_styles[model],
                   marker=markers.get(model, 'o'), color=colors[model],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, linewidth=1.5, alpha=1 if 'Ada' in model else 0.5)

    # PART 2: Bottom row - Complexity vs time steps
    # Get all time steps (from column names)
    time_steps = []
    for col in results_df.columns:
        if col.startswith('iteration_time_alg1_T'):
            step = int(col.split('T')[1])
            time_steps.append(step)
    time_steps = sorted(time_steps)
    
    # For each model, extract time and memory metrics for each timestep
    for model in models:
        model_data = results_df[(results_df['model'] == model) & (results_df['N'] == selected_N)]
        
        if len(model_data) == 0:
            print(f"Warning: No data for model {model} with N={selected_N}")
            continue
        
        # Initialize arrays for time and memory means
        time_means = []
        memory_means = []
        
        # Calculate average over seeds for each time step
        for step in time_steps:
            time_col = f'iteration_time_alg1_T{step}'
            memory_col = f'iteration_memory_alg1_T{step}'
            
            time_values = model_data[time_col].dropna()
            memory_values = model_data[memory_col].dropna()
            
            if len(time_values) > 0 and len(memory_values) > 0:
                time_means.append(time_values.mean())
                memory_means.append(memory_values.mean())
            else:
                time_means.append(np.nan)
                memory_means.append(np.nan)
        
        # Plot iteration time vs time steps
        axs[0, 1].plot(time_steps, time_means, label=f"{model.upper()}", linestyle=line_styles[model],
                   marker=markers.get(model, 'o'), color=colors[model],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, linewidth=1.5, alpha=1 if 'Ada' in model else 0.5)

        # Plot iteration memory vs time steps
        axs[1, 1].plot(time_steps, memory_means, label=f"{model.upper()}", linestyle=line_styles[model],
                   marker=markers.get(model, 'o'), color=colors[model],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, linewidth=1.5, alpha=1 if 'Ada' in model else 0.5)
    
    # Configure top-left plot (Iteration Time vs N)
    axs[0, 0].set_xlabel('$N$')
    axs[0, 0].set_ylabel('Time per iteration [s]')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].yaxis.grid(True, alpha=0.75, linestyle='--', which='major')
    axs[0, 0].xaxis.grid(True, alpha=0.75, linestyle='--', which='major')
    # axs[0, 0].set_title('Iteration Time vs N', fontsize=12)
    
    # Configure top-right plot (Iteration Time vs T)
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Time per iteration [s]')
    axs[0, 1].set_yscale('log')
    axs[0, 1].yaxis.grid(True, alpha=0.75, linestyle='--', which='major')
    axs[0, 1].xaxis.grid(True, alpha=0.75, linestyle='--', which='major')
    # axs[0, 1].set_title(f'Iteration Time vs Training Steps (N={selected_N})', fontsize=12)
    
    # Configure bottom-left plot (Memory vs N)
    axs[1, 0].set_xlabel('$N$')
    axs[1, 0].set_ylabel('Peak memory usage [MB]')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].yaxis.grid(True, alpha=0.75, linestyle='--', which='major')
    axs[1, 0].xaxis.grid(True, alpha=0.75, linestyle='--', which='major')
    # axs[1, 0].set_title('Memory Usage vs N', fontsize=12)
    
    # Configure bottom-right plot (Memory vs T)
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('Peak memory usage [MB]')
    axs[1, 1].set_yscale('log')
    axs[1, 1].yaxis.grid(True, alpha=0.75, linestyle='--', which='major')
    axs[1, 1].xaxis.grid(True, alpha=0.75, linestyle='--', which='major')
    # axs[1, 1].set_title(f'Memory Usage vs Training Steps (N={selected_N})', fontsize=12)
    
    # Create legend with specified ordering and layout
    handles, labels = axs[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    available_models = results_df['model'].unique()
    model_order = ['AdaCGP_P1', 'AdaCGP_P2', 'TISO', 'TIRSO', 'SDSEM', 'GLasso', 'VAR', 'GrangerVAR']

    ordered_labels = [display_labels[model] for model in model_order if model in available_models]
    ordered_handles = [by_label[label] for label in ordered_labels if label in by_label]
    
    # Add the legend with 3 methods per row, 2 rows
    fig.legend(
        ordered_handles, 
        ordered_labels, 
        loc='lower center', 
        ncol=4, 
        bbox_to_anchor=(0.5, 0.01), 
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=10
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='svg')
    print(f"Figure saved to {output_path}")
    return fig, axs

def print_fit_results(results_df, time_step=9000, break_point=None):
    """
    Fits polynomial regression models to different segments of complexity data to detect scaling changes.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing complexity results with columns for model, N, iteration_time, etc.
    time_step : int, optional
        Time step to use for the analysis, default is 9000
    break_point : float, optional
        Percentile of N values to use as break point for segmentation (default: 50%)
    """
    import numpy as np
    import pandas as pd
    from scipy import stats, optimize
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    
    # Set default break point if not provided
    if break_point is None:
        break_point = 50  # Default to 50th percentile
    
    # Get unique models
    models = results_df['model'].unique()
    
    # Initialize results storage
    fit_results = {
        'model': [],
        'full_range_exponent': [],
        'full_range_r2': [],
        'lower_range_exponent': [],
        'lower_range_r2': [],
        'upper_range_exponent': [],
        'upper_range_r2': [],
        'scaling_change': []
    }
    
    # Prepare figure for visualizing fits
    fig, axs = plt.subplots(len(models), 2, figsize=(15, 5*len(models)))
    if len(models) == 1:
        axs = np.array([axs])  # Ensure axs is always a 2D array
    
    # For each model, fit separate polynomials to different N ranges
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        
        # Group by N and average over seeds
        grouped = model_data.groupby('N')
        
        # Extract time and memory metrics
        time_col = f'iteration_time_alg1_T{time_step}'
        memory_col = f'iteration_memory_alg1_T{time_step}'
        
        if time_col not in model_data.columns or memory_col not in model_data.columns:
            print(f"Warning: Metrics {time_col} or {memory_col} not found for model {model}")
            continue
            
        # Calculate means (excluding NaN values)
        n_values = []
        time_means = []
        memory_means = []
        
        for n, group in grouped:
            time_values = group[time_col].dropna()
            memory_values = group[memory_col].dropna()
            
            if len(time_values) > 0 and len(memory_values) > 0:
                n_values.append(n)
                time_means.append(time_values.mean())
                memory_means.append(memory_values.mean())
        
        # Sort by N for proper fitting
        sort_indices = np.argsort(n_values)
        n_values = np.array(n_values)[sort_indices]
        time_means = np.array(time_means)[sort_indices]
        memory_means = np.array(memory_means)[sort_indices]
        
        # Skip if not enough data points
        if len(n_values) < 5:  # Need at least 5 points for meaningful segmentation
            print(f"Warning: Not enough data points for model {model}")
            continue
        
        # Convert to log space for polynomial fitting
        log_n = np.log(n_values)
        log_time = np.log(time_means)
        log_memory = np.log(memory_means)
        
        # Determine break point for segmentation
        break_idx = int(len(n_values) * break_point / 100)
        if break_idx < 2 or break_idx > len(n_values) - 2:
            break_idx = len(n_values) // 2  # Fallback to middle point
        
        # Separate data into lower and upper ranges
        lower_log_n = log_n[:break_idx]
        lower_log_time = log_time[:break_idx]
        lower_log_memory = log_memory[:break_idx]
        
        upper_log_n = log_n[break_idx:]
        upper_log_time = log_time[break_idx:]
        upper_log_memory = log_memory[break_idx:]
        
        # Fit polynomials to full range
        time_slope_full, time_intercept_full, time_r_full, _, _ = stats.linregress(log_n, log_time)
        mem_slope_full, mem_intercept_full, mem_r_full, _, _ = stats.linregress(log_n, log_memory)
        
        # Fit polynomials to lower range
        time_slope_lower, time_intercept_lower, time_r_lower, _, _ = stats.linregress(lower_log_n, lower_log_time)
        mem_slope_lower, mem_intercept_lower, mem_r_lower, _, _ = stats.linregress(lower_log_n, lower_log_memory)
        
        # Fit polynomials to upper range
        time_slope_upper, time_intercept_upper, time_r_upper, _, _ = stats.linregress(upper_log_n, upper_log_time)
        mem_slope_upper, mem_intercept_upper, mem_r_upper, _, _ = stats.linregress(upper_log_n, upper_log_memory)
        
        # Calculate scaling change
        time_scaling_change = time_slope_upper - time_slope_lower
        mem_scaling_change = mem_slope_upper - mem_slope_lower
        
        # Calculate R² values
        time_r2_full = time_r_full**2
        time_r2_lower = time_r_lower**2
        time_r2_upper = time_r_upper**2
        
        mem_r2_full = mem_r_full**2
        mem_r2_lower = mem_r_lower**2
        mem_r2_upper = mem_r_upper**2
        
        # Store results for time complexity
        fit_results['model'].append(f"{model} (Time)")
        fit_results['full_range_exponent'].append(time_slope_full)
        fit_results['full_range_r2'].append(time_r2_full)
        fit_results['lower_range_exponent'].append(time_slope_lower)
        fit_results['lower_range_r2'].append(time_r2_lower)
        fit_results['upper_range_exponent'].append(time_slope_upper)
        fit_results['upper_range_r2'].append(time_r2_upper)
        fit_results['scaling_change'].append(time_scaling_change)
        
        # Store results for memory complexity
        fit_results['model'].append(f"{model} (Memory)")
        fit_results['full_range_exponent'].append(mem_slope_full)
        fit_results['full_range_r2'].append(mem_r2_full)
        fit_results['lower_range_exponent'].append(mem_slope_lower)
        fit_results['lower_range_r2'].append(mem_r2_lower)
        fit_results['upper_range_exponent'].append(mem_slope_upper)
        fit_results['upper_range_r2'].append(mem_r2_upper)
        fit_results['scaling_change'].append(mem_scaling_change)
        
        # Plot fits for time complexity
        axs[i, 0].scatter(n_values, time_means, color='blue', alpha=0.7, label='Data')
        axs[i, 0].plot(n_values, np.exp(time_intercept_full) * n_values**time_slope_full, 
                 'r-', label=f'Full fit: O(N^{time_slope_full:.2f})')
        axs[i, 0].plot(n_values[:break_idx], np.exp(time_intercept_lower) * n_values[:break_idx]**time_slope_lower, 
                 'g--', label=f'Lower fit: O(N^{time_slope_lower:.2f})')
        axs[i, 0].plot(n_values[break_idx:], np.exp(time_intercept_upper) * n_values[break_idx:]**time_slope_upper, 
                 'm--', label=f'Upper fit: O(N^{time_slope_upper:.2f})')
        axs[i, 0].set_xscale('log')
        axs[i, 0].set_yscale('log')
        axs[i, 0].set_xlabel('N')
        axs[i, 0].set_ylabel('Time per iteration [s]')
        axs[i, 0].set_title(f'{model} - Time Complexity')
        axs[i, 0].legend()
        axs[i, 0].grid(True, alpha=0.3)
        
        # Plot fits for memory complexity
        axs[i, 1].scatter(n_values, memory_means, color='blue', alpha=0.7, label='Data')
        axs[i, 1].plot(n_values, np.exp(mem_intercept_full) * n_values**mem_slope_full, 
                 'r-', label=f'Full fit: O(N^{mem_slope_full:.2f})')
        axs[i, 1].plot(n_values[:break_idx], np.exp(mem_intercept_lower) * n_values[:break_idx]**mem_slope_lower, 
                 'g--', label=f'Lower fit: O(N^{mem_slope_lower:.2f})')
        axs[i, 1].plot(n_values[break_idx:], np.exp(mem_intercept_upper) * n_values[break_idx:]**mem_slope_upper, 
                 'm--', label=f'Upper fit: O(N^{mem_slope_upper:.2f})')
        axs[i, 1].set_xscale('log')
        axs[i, 1].set_yscale('log')
        axs[i, 1].set_xlabel('N')
        axs[i, 1].set_ylabel('Peak memory usage [MB]')
        axs[i, 1].set_title(f'{model} - Memory Complexity')
        axs[i, 1].legend()
        axs[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('segmented_complexity_analysis.png', dpi=300)
    
    # Convert to DataFrame for nice display
    results_table = pd.DataFrame(fit_results)
    
    # Format values for display
    results_table['full_range_exponent'] = results_table['full_range_exponent'].apply(lambda x: f"{x:.3f}")
    results_table['full_range_r2'] = results_table['full_range_r2'].apply(lambda x: f"{x:.3f}")
    results_table['lower_range_exponent'] = results_table['lower_range_exponent'].apply(lambda x: f"{x:.3f}")
    results_table['lower_range_r2'] = results_table['lower_range_r2'].apply(lambda x: f"{x:.3f}")
    results_table['upper_range_exponent'] = results_table['upper_range_exponent'].apply(lambda x: f"{x:.3f}")
    results_table['upper_range_r2'] = results_table['upper_range_r2'].apply(lambda x: f"{x:.3f}")
    results_table['scaling_change'] = results_table['scaling_change'].apply(lambda x: f"{x:.3f}")
    
    # Rename columns for display
    display_table = results_table.rename(columns={
        'model': 'Model',
        'full_range_exponent': 'Full Range Exponent',
        'full_range_r2': 'Full R²',
        'lower_range_exponent': 'Lower Range Exponent',
        'lower_range_r2': 'Lower R²',
        'upper_range_exponent': 'Upper Range Exponent',
        'upper_range_r2': 'Upper R²',
        'scaling_change': 'Scaling Change'
    })
    
    # Print table
    print("\nSegmented Complexity Analysis:")
    print("=============================\n")
    print(tabulate(display_table, headers='keys', tablefmt='fancy_grid', showindex=False))    
    return results_table, fig

def generate_latex_table(results, display_labels=None):
    """
    Generate a single compact LaTeX table with both time and memory complexity.
    Uses formatting consistent with other tables in the document.
    
    Parameters:
    -----------
    results : pandas.DataFrame or tuple
        DataFrame or tuple containing complexity analysis results
    display_labels : dict, optional
        Dictionary mapping original model names to display labels
    
    Returns:
    --------
    str
        A LaTeX formatted table combining time and memory complexity
    """
    import pandas as pd
    
    # Default display labels
    if display_labels is None:
        display_labels = {
            'AdaCGP_P1': 'AdaCGP (P1)',
            'AdaCGP_P2': 'AdaCGP (P2)',
            'SDSEM': 'SD-SEM',
            'TISO': 'TISO',
            'TIRSO': 'TIRSO',
            'VAR': 'VAR',
            'GrangerVAR': 'VAR + Granger',
            'GLasso': 'GLasso',
            'GLSigRep': 'GL-SigRep'
        }
    
    # Handle case where results is a tuple from print_fit_results
    if isinstance(results, tuple):
        results_table = results[0]
    else:
        results_table = results
    
    # Create a dictionary to store time and memory complexities by model
    complexity_data = {}
    
    # Process time complexity data
    time_data = results_table[results_table['model'].str.contains('Time')]
    for _, row in time_data.iterrows():
        base_model = row['model'].replace(" (Time)", "")
        exponent = float(row['upper_range_exponent'])
        r2 = row['upper_range_r2']
        
        if base_model not in complexity_data:
            complexity_data[base_model] = {'time_exp': exponent, 'time_r2': r2}
        else:
            complexity_data[base_model]['time_exp'] = exponent
            complexity_data[base_model]['time_r2'] = r2
    
    # Process memory complexity data
    memory_data = results_table[results_table['model'].str.contains('Memory')]
    for _, row in memory_data.iterrows():
        base_model = row['model'].replace(" (Memory)", "")
        exponent = float(row['upper_range_exponent'])
        r2 = row['upper_range_r2']
        
        if base_model not in complexity_data:
            complexity_data[base_model] = {'memory_exp': exponent, 'memory_r2': r2}
        else:
            complexity_data[base_model]['memory_exp'] = exponent
            complexity_data[base_model]['memory_r2'] = r2
    
    # Define batch algorithms and adaptive algorithms
    batch_algorithms = ['GLasso', 'GLSigRep', 'VAR', 'GrangerVAR']
    adaptive_algorithms = ['SDSEM', 'TIRSO', 'TISO', 'AdaCGP_P2', 'AdaCGP_P1']
    
    # LaTeX table header with improved formatting
    latex_table = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Computational and Memory Complexity Scaling with Problem Size.}",
        "\\label{tab:complexity_scaling_combined}",
        "\\setlength{\\tabcolsep}{6pt}  % Increased horizontal spacing",
        "\\setlength{\\aboverulesep}{0pt}",
        "\\setlength{\\belowrulesep}{0pt}",
        "\\renewcommand{\\arraystretch}{1.15}"
    ]
    
    # Define column structure with better spacing
    column_def = "@{}c|l|c!{\\hspace{.25em}}c@{}}"
    latex_table.append("\\begin{tabular}{" + column_def)
    latex_table.append("\\toprule[1pt]\\midrule[0.3pt]")
    
    # Header row
    latex_table.append("&& \\multicolumn{2}{c}{\\textsc{Complexity}} \\\\")
    latex_table.append("\\cmidrule(lr){3-4}")
    
    # Metric names row
    latex_table.append("& Algorithm & Time & Memory \\\\")
    latex_table.append("\\midrule")
    
    # Write batch algorithms section
    batch_count = sum(1 for alg in complexity_data if alg in batch_algorithms)
    if batch_count > 0:
        latex_table.append("\\multirow{" + str(batch_count) + "}{*}{\\rotatebox{90}{\\textsc{Batch}}} ")
        
        # Add rows for batch algorithms
        for alg in batch_algorithms:
            if alg not in complexity_data:
                continue
                
            # Get display name
            display_name = display_labels.get(alg, alg)
            
            # Get complexity values
            data = complexity_data[alg]
            time_exp = data.get('time_exp', float('nan'))
            memory_exp = data.get('memory_exp', float('nan'))
            
            # Format values
            if not pd.isna(time_exp):
                time_complexity = f"$O(N^{{{time_exp:.2f}}})$"
            else:
                time_complexity = "---"
                
            if not pd.isna(memory_exp):
                memory_complexity = f"$O(N^{{{memory_exp:.2f}}})$"
            else:
                memory_complexity = "---"
            
            # Write row
            latex_table.append(f"& {display_name} & {time_complexity} & {memory_complexity} \\\\")
        
        # Add separator between batch and adaptive
        latex_table.append("\\midrule")
    
    # Write adaptive algorithms section
    adaptive_count = sum(1 for alg in complexity_data if alg in adaptive_algorithms)
    if adaptive_count > 0:
        latex_table.append("\\multirow{" + str(adaptive_count) + "}{*}{\\rotatebox{90}{\\textsc{Adaptive}}} ")
        
        # Add rows for adaptive algorithms
        for alg in adaptive_algorithms:
            if alg not in complexity_data:
                continue
                
            # Get display name
            display_name = display_labels.get(alg, alg)
            
            # Get complexity values
            data = complexity_data[alg]
            time_exp = data.get('time_exp', float('nan'))
            memory_exp = data.get('memory_exp', float('nan'))
            
            # Format values
            if not pd.isna(time_exp):
                time_complexity = f"$O(N^{{{time_exp:.2f}}})$"
            else:
                time_complexity = "---"
                
            if not pd.isna(memory_exp):
                memory_complexity = f"$O(N^{{{memory_exp:.2f}}})$"
            else:
                memory_complexity = "---"
            
            # Write row
            latex_table.append(f"& {display_name} & {time_complexity} & {memory_complexity} \\\\")
    
    # Add any remaining algorithms that aren't categorized
    remaining_algs = [alg for alg in complexity_data.keys() 
                    if alg not in batch_algorithms and alg not in adaptive_algorithms]
    
    if remaining_algs:
        # Add separator if needed
        if batch_count > 0 or adaptive_count > 0:
            latex_table.append("\\midrule")
            
        latex_table.append("\\multirow{" + str(len(remaining_algs)) + "}{*}{\\rotatebox{90}{\\textsc{Other}}} ")
        
        # Add rows for remaining algorithms
        for alg in remaining_algs:
            # Get display name
            display_name = display_labels.get(alg, alg)
            
            # Get complexity values
            data = complexity_data[alg]
            time_exp = data.get('time_exp', float('nan'))
            memory_exp = data.get('memory_exp', float('nan'))
            
            # Format values
            if not pd.isna(time_exp):
                time_complexity = f"$O(N^{{{time_exp:.2f}}})$"
            else:
                time_complexity = "---"
                
            if not pd.isna(memory_exp):
                memory_complexity = f"$O(N^{{{memory_exp:.2f}}})$"
            else:
                memory_complexity = "---"
            
            # Write row
            latex_table.append(f"& {display_name} & {time_complexity} & {memory_complexity} \\\\")
    
    # Close the tabular environment
    latex_table.append("\\midrule[0.3pt]\\toprule[1pt]")
    latex_table.append("\\end{tabular}")
    latex_table.append("\\end{table*}")
    
    return "\n".join(latex_table)

if __name__ == "__main__":
    path_to_complexity_results = "logs/complexity/2025-03-24/15-41-57"
    results_df = read_complexity_results(path_to_complexity_results)
    fig, axs = plot_unified_complexity_analysis(results_df, output_path='figures/unified_complexity_analysis.svg', 
                                            time_step=9000, selected_N=50)
    fit_table = print_fit_results(results_df, break_point=50)
    latex_table = generate_latex_table(fit_table)
    print(latex_table)