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
    fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    
    # Get unique models
    models = results_df['model'].unique()

    # set font
    # plt.rcParams.update({'font.size': 12})
    # plt.rcParams.update({'font.family': 'serif'})

    # cmaps and markers
    cmap_blue = plt.get_cmap('Blues')
    cmap_red = plt.get_cmap('Reds')
    
    markers = {'AdaCGP_P1': 's', 'AdaCGP_P2': '^', 'SDSEM': 'o', 'TISO': 'D', 'TIRSO': 'X', 'VAR': 'P'}
    colors = {
        'AdaCGP_P1': cmap_blue((50+10) / 90), 
        'AdaCGP_P2': cmap_red((50+10) / 90),
        'SDSEM': '#FF8C00',       # Distinct green
        'TISO': '#9370DB',        # Medium purple - changed from the pinkish color
        'TIRSO': '#4B0082',       # Indigo - darker to distinguish from TISO
        'VAR': '#00A36C'          # Dark orange
    }
    display_labels = {
        'AdaCGP_P1': 'AdaCGP (P1)',
        'AdaCGP_P2': 'AdaCGP (P2)',
        'SDSEM': 'SD-SEM',
        'TISO': 'TISO',
        'TIRSO': 'TIRSO',
        'VAR': 'VAR'
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
        axs[0, 0].plot(n_values, time_means, label=display_labels[model],
                   marker=markers.get(model, 'o'), color=colors[model],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, linewidth=1.5)
        
        # Plot iteration memory vs N
        axs[1, 0].plot(n_values, memory_means, label=display_labels[model],
                   marker=markers.get(model, 'o'), color=colors[model],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, linewidth=1.5)
    
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
        axs[0, 1].plot(time_steps, time_means, label=f"{model.upper()}", 
                   marker=markers.get(model, 'o'), color=colors[model],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, linewidth=1.5)
        
        # Plot iteration memory vs time steps
        axs[1, 1].plot(time_steps, memory_means, label=f"{model.upper()}",
                   marker=markers.get(model, 'o'), color=colors[model],
                   markersize=6, markeredgecolor='black', markeredgewidth=0.5, linewidth=1.5)
    
    # Configure top-left plot (Iteration Time vs N)
    axs[0, 0].set_xlabel('N')
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
    axs[1, 0].set_xlabel('N')
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
    model_order = ['AdaCGP_P1', 'AdaCGP_P2', 'VAR', 'SDSEM', 'TISO', 'TIRSO']

    ordered_labels = [display_labels[model] for model in model_order if model in available_models]
    ordered_handles = [by_label[label] for label in ordered_labels if label in by_label]
    
    # Add the legend with 3 methods per row, 2 rows
    fig.legend(
        ordered_handles, 
        ordered_labels, 
        loc='lower center', 
        ncol=3, 
        bbox_to_anchor=(0.5, 0.01), 
        bbox_transform=fig.transFigure,
        frameon=True,
        fontsize=10
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='svg')
    print(f"Figure saved to {output_path}")
    return fig, axs

if __name__ == "__main__":
    path_to_complexity_results = "logs/complexity/2025-03-24/15-41-57"
    results_df = read_complexity_results(path_to_complexity_results)
    fig, axs = plot_unified_complexity_analysis(results_df, output_path='figures/unified_complexity_analysis.svg', 
                                            time_step=9000, selected_N=50)
    plt.show()

