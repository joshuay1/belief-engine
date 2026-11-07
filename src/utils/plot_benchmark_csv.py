#!/usr/bin/env python3
"""
Plot 4-Condition Benchmark results directly from results.csv.
Style based on experiments/summaries/plots_4cond reference figures.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def parse_stance_history(stance_history_str):
    """Parse stance_history from CSV string to list of dicts."""
    if pd.isna(stance_history_str):
        return []
    try:
        return ast.literal_eval(stance_history_str)
    except:
        return []

def extract_condition(config_name):
    """Map config_name to condition group."""
    if config_name.startswith("Subject_Standard"):
        return "Subject_Standard"
    elif config_name.startswith("Subject_Bayesian"):
        return "Subject_Bayesian"
    elif config_name.startswith("Moltbot_Standard"):
        return "Moltbot_Standard"
    elif config_name.startswith("Moltbot_Stance"):
        return "Moltbot_Stance"
    return None

def plot_benchmark(csv_path, title=None):
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return
        
    run_dir = os.path.dirname(csv_path)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trials from {csv_path}")
    
    # Add condition column
    df['condition'] = df['config_name'].apply(extract_condition)
    df = df[df['condition'].notna()]
    
    # Infer topic from first row if no title provided
    topic = title
    if not topic:
        # Try to extract topic from trial directory name
        topic = "AI Agent Debate: 4-Condition Benchmark"
    
    # Group by condition
    conditions = ["Subject_Standard", "Subject_Bayesian", "Moltbot_Standard", "Moltbot_Stance"]
    condition_labels = {
        "Subject_Standard": "Subject (Blind)",
        "Subject_Bayesian": "Subject (Bayesian)", 
        "Moltbot_Standard": "Moltbot (Standard)",
        "Moltbot_Stance": "Moltbot (Stance)"
    }
    
    # Colors matching reference style
    PRO_COLOR = '#3498DB'   # Blue
    CON_COLOR = '#E74C3C'   # Red
    PRO_FILL = '#AED6F1'    # Light blue
    CON_FILL = '#F5B7B1'    # Light red
    
    # Set up figure with font style matching reference plots - LARGER FONTS
    plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    fig.patch.set_facecolor('white')
    
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        ax.set_facecolor('white')
        cond_df = df[df['condition'] == condition]
        
        if cond_df.empty:
            ax.set_title(condition_labels.get(condition, condition), fontsize=13)
            ax.set_xlabel("Round")
            if idx == 0:
                ax.set_ylabel("Agent Stance", fontsize=12)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xlim(-0.5, 15.5)
            continue
        
        # Collect Pro (internal) and Con (evaluated) trajectories
        pro_trajectories = []  # Internal stance (Subject)
        con_trajectories = []  # External stance (Subject_evaluated)
        
        for _, row in cond_df.iterrows():
            history = parse_stance_history(row.get('stance_history', '[]'))
            if history:
                pro_traj = []
                con_traj = []
                for entry in history:
                    if isinstance(entry, dict):
                        if 'Subject' in entry:
                            pro_traj.append(entry['Subject'])
                        if 'Subject_evaluated' in entry:
                            con_traj.append(entry['Subject_evaluated'])
                if pro_traj:
                    pro_trajectories.append(pro_traj)
                if con_traj:
                    con_trajectories.append(con_traj)
        
        # Plot Pro trajectories (Internal - Blue)
        if pro_trajectories:
            max_len = max(len(t) for t in pro_trajectories)
            padded = []
            for t in pro_trajectories:
                if len(t) < max_len:
                    t = t + [t[-1]] * (max_len - len(t))
                padded.append(t[:max_len])
            
            pro_arr = np.array(padded)
            rounds = np.arange(max_len)
            
            pro_mean = np.mean(pro_arr, axis=0)
            pro_std = np.std(pro_arr, axis=0)
            
            # Shaded region
            ax.fill_between(rounds, pro_mean - pro_std, pro_mean + pro_std,
                           alpha=0.25, color=PRO_COLOR, linewidth=0)
            
            # Mean line
            ax.plot(rounds, pro_mean, color=PRO_COLOR, linewidth=2.5, solid_capstyle='round')
            
            # Start label
            ax.annotate(f'{pro_mean[0]:.2f}', xy=(0, pro_mean[0]), 
                       xytext=(-5, 0), textcoords='offset points',
                       fontsize=12, color=PRO_COLOR, fontweight='bold',
                       ha='right', va='center')
            
            # End label (value only)
            ax.annotate(f'{pro_mean[-1]:.2f}', xy=(max_len-1, pro_mean[-1]),
                       xytext=(8, 0), textcoords='offset points',
                       fontsize=12, color=PRO_COLOR, fontweight='bold',
                       ha='left', va='center')
        
        # Plot Con trajectories (External/Evaluated - Red)
        if con_trajectories:
            max_len = max(len(t) for t in con_trajectories)
            padded = []
            for t in con_trajectories:
                if len(t) < max_len:
                    t = t + [t[-1]] * (max_len - len(t))
                padded.append(t[:max_len])
            
            con_arr = np.array(padded)
            rounds = np.arange(max_len)
            
            con_mean = np.mean(con_arr, axis=0)
            con_std = np.std(con_arr, axis=0)
            
            # Shaded region
            ax.fill_between(rounds, con_mean - con_std, con_mean + con_std,
                           alpha=0.25, color=CON_COLOR, linewidth=0)
            
            # Mean line
            ax.plot(rounds, con_mean, color=CON_COLOR, linewidth=2.5, solid_capstyle='round')
            
            # Start label
            ax.annotate(f'{con_mean[0]:.2f}', xy=(0, con_mean[0]), 
                       xytext=(-5, 0), textcoords='offset points',
                       fontsize=12, color=CON_COLOR, fontweight='bold',
                       ha='right', va='center')
            
            # End label (value only)
            ax.annotate(f'{con_mean[-1]:.2f}', xy=(max_len-1, con_mean[-1]),
                       xytext=(8, 0), textcoords='offset points',
                       fontsize=12, color=CON_COLOR, fontweight='bold',
                       ha='left', va='center')
        
        # Style axes
        ax.set_title(condition_labels.get(condition, condition), fontsize=13, fontweight='normal', pad=20)
        ax.set_xlabel("Round", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Agent Stance", fontsize=12)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlim(-0.5, 15.5)
        
        # Zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Tick styling
        ax.tick_params(axis='both', which='major', labelsize=10, colors='#666666')
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Main title - bold italic matching reference style
    plt.suptitle(topic, fontsize=20, fontweight='bold', fontstyle='italic', y=1.02)
    
    # Global Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PRO_COLOR, lw=3, label='Internal Stance'),
        Line2D([0], [0], color=CON_COLOR, lw=3, label='External (Evaluated) Stance')
    ]
    # Vertical legend in lower right
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.9, 0.15), 
              ncol=1, frameon=False, fontsize=12)
    
    plt.subplots_adjust(top=0.85, bottom=0.20, wspace=0.3, right=0.82)
    
    # We use subplots_adjust instead of tight_layout to have more control
    # plt.tight_layout()
    
    output_path = os.path.join(run_dir, "benchmark_4cond.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 4-Condition Benchmark from CSV")
    parser.add_argument("--csv_path", required=True, help="Path to CSV")
    parser.add_argument("--title", default=None, help="Optional custom title")
    args = parser.parse_args()
    
    plot_benchmark(args.csv_path, title=args.title)
