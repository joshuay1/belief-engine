import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import argparse
import os
import sys

def set_style():
    # Use seaborn 'whitegrid' style which is clean and modern
    sns.set_theme(style="whitegrid")
    
    # Update matplotlib parameters for even larger fonts
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 28,
        'lines.linewidth': 5,
        'lines.markersize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

def get_custom_colors():
    return {
        "Bias_0.0_Control": "#E74C3C", # Red
        "Bias_0.4_Moderate": "#F39C12", # Orange
        "Bias_0.5_Mid": "#D35400",     # Darker Orange
        "Bias_0.6_MidHigh": "#2ECC71", # Green (Safe)
        "Bias_0.7_High": "#27AE60",    # Darker Green
        "Bias_0.8_EchoChamber": "#16A085", # Teal
        "Bias_0.9_VeryHigh": "#2980B9", # Blue
        "Bias_0.9_Strong": "#2980B9",
        "Bias_1.0_Absolute": "#8E44AD", # Purple
        "Control": "#E74C3C",
        "Echo Chamber": "#2ECC71",
        # Sensitivity Sweep
        "Sens_0.1_Rigid": "#2C3E50", # Charcoal/Black
        "Sens_0.2_RigidMid": "#8E44AD", # Purple
        "Sens_0.3_RigidMid": "#3498DB", # Bright Blue
        "Sens_0.5_Mid": "#27AE60",   # Green (Reference)
        "Sens_1.0_Standard": "#F39C12", # Orange
        "Sens_2.0_High": "#E74C3C", # Red
        "Sens_3.0_VeryHigh": "#C0392B", # Dark Red
        "Sens_5.0_Hyper": "#922B21", # Maroon
        # Architecture Sweep
        "Arch_1_Baseline": "#7F8C8D",       # Grey (Baseline)
        "Arch_2_LogicFiltered": "#9B59B6",  # Amethyst (Smart)
        "Arch_3_Amem_Vector": "#2980B9",    # Blue (Database)
        "Arch_4_FullAgentic": "#E74C3C"     # Red (Agentic)
    }

def get_display_label(config_name):
    # Improve Labels with specific settings
    if "Arch_" in config_name:
        if "Arch_1" in config_name: return "1. MemoryStore + VectorJudgement"
        if "Arch_2" in config_name: return "2. MemoryStore + AgenticJudgement"
        if "Arch_3" in config_name: return "3. A-Mem + VectorJudgement"
        if "Arch_4" in config_name: return "4. A-Mem + AgenticJudgement"
        return config_name
    elif "Sens_" in config_name:
        # Sensitivity Sweep
        parts = config_name.split("_")
        sens_val = parts[1]
        desc = parts[2] if len(parts) > 2 else ""
        return f"Sensitivity {sens_val}"
    elif "Bias_0.0" in config_name:
        return "Bias 0.0 (Control)"
    elif "Bias_0.4" in config_name:
            return "Bias 0.4 (Moderate)"
    elif "Bias_0.5" in config_name:
            return "Bias 0.5 (Mid)"
    elif "Bias_0.6" in config_name:
            return "Bias 0.6 (Mid-High)"
    elif "Bias_0.7" in config_name:
            return "Bias 0.7 (High)"
    elif "Bias_0.8" in config_name:
        return "Bias 0.8 (Strong)"
    elif "Bias_0.9" in config_name and "0.95" not in config_name:
        return "Bias 0.9 (Very Strong)"
    elif "Bias_1.0" in config_name:
        return "Bias 1.0 (Absolute)"
    elif "Bias_0.95" in config_name:
        return "Bias 0.95 (Extreme)"
    else:
        return config_name.replace("_", " ")

def normalize_stance_history(raw_history):
    """
    Converts stance_history into a list of numeric values.
    Supports raw lists, stringified lists, and list of dicts.
    """
    history = raw_history
    if isinstance(raw_history, str):
        try:
            history = ast.literal_eval(raw_history)
        except Exception:
            return []
    if not isinstance(history, list):
        return []

    normalized = []
    for entry in history:
        if isinstance(entry, (int, float)):
            normalized.append(entry)
        elif isinstance(entry, dict):
            targets = ["Subject", "ProAgent", "SubjectAgent", "Subject_Standard"]
            found = False
            for t in targets:
                if isinstance(entry.get(t), (int, float)):
                    normalized.append(entry.get(t))
                    found = True
                    break
            
            if not found:
                for key, val in entry.items():
                    if str(key).lower() == "round":
                        continue
                    if isinstance(val, (int, float)):
                        normalized.append(val)
                        break
    return normalized


def detect_agent_family(row):
    agent_type = str(row.get("var_debate.agent_type", "")).lower()
    config_name = str(row.get("config_name", "")).lower()

    if "moltbot" in agent_type or config_name.startswith("openclaw_"):
        return "openclaw"
    if "subject" in agent_type or "deterministic" in agent_type or config_name.startswith("corebot_"):
        return "corebot"
    return "other"


def basic_trajectory_plot(df, output_dir, output_basename="stance_trajectories", title="Stance Evolution Trajectories"):
    plt.figure(figsize=(16, 10))
    
    # Import centralized fallback colors
    try:
        from src.utils.plot_theme import COLORS_GENERIC
    except ImportError:
        COLORS_GENERIC = sns.color_palette("husl", 10).as_hex()
        
    custom_colors = get_custom_colors()

    data = []
    max_round = 0
    start_labels = {}
    end_labels = {}

    for index, row in df.iterrows():
        config_name = row.get("config_name", f"Run {index}")
        display_label = get_display_label(config_name)
        history = normalize_stance_history(row.get("stance_history", ""))
        
        if not history:
            continue
            
        full_history = history
        max_round = max(max_round, len(full_history))
        
        # Save start/end values for annotations over the aggregated means later
        if display_label not in start_labels:
            start_labels[display_label] = {'values': []}
            end_labels[display_label] = {'values': []}
            
        start_labels[display_label]['values'].append(full_history[0])
        end_labels[display_label]['values'].append(full_history[-1])
            
        # Start rounds at 1, since we skipped the artificial Round 0 initial_stance
        for r_idx, val in enumerate(full_history, start=1):
            data.append({
                "Variation": display_label,
                "Round": r_idx,
                "Stance": val,
                "OriginalConfig": config_name
            })
            
    if not data:
        print("No valid history data found.")
        return
        
    plot_df = pd.DataFrame(data)
    
    # Build palette mapping safely
    palette_mapping = {}
    variations = plot_df['Variation'].unique()
    for idx, var in enumerate(variations):
        # find original
        cfg = plot_df[plot_df['Variation'] == var]['OriginalConfig'].iloc[0]
        color = custom_colors.get(cfg, COLORS_GENERIC[idx % len(COLORS_GENERIC)])
        palette_mapping[var] = color
        
        if var in start_labels:
            start_labels[var]['color'] = color
            end_labels[var]['color'] = color
    
    # Use seaborn's native errorbar estimator to aggregate trials into smooth bands
    sns.lineplot(
        data=plot_df,
        x="Round",
        y="Stance",
        hue="Variation",
        palette=palette_mapping,
        marker='o',
        linewidth=4,
        errorbar="sd"
    )
    
    # Add textual annotations for means (starting at x=0.9 for Round 1)
    final_start_labels = []
    final_end_labels = []
    for var, info in start_labels.items():
        if not info['values']: continue
        mean_start = sum(info['values']) / len(info['values'])
        final_start_labels.append({'x': 0.9, 'y': mean_start, 'text': f"{mean_start:.2f}", 'color': info['color'], 'align': 'right'})
        
    for var, info in end_labels.items():
        if not info['values']: continue
        mean_end = sum(info['values']) / len(info['values'])
        final_end_labels.append({'x': max_round + 0.1, 'y': mean_end, 'text': f"{mean_end:.2f}", 'color': info['color'], 'align': 'left'})

    def plot_adjusted_labels(labels, min_dist=0.08):
        # Sort by Y descending
        labels.sort(key=lambda x: x['y'], reverse=True)
        
        # Simple separation logic
        for i in range(1, len(labels)):
            diff = labels[i-1]['y'] - labels[i]['y']
            if diff < min_dist:
                # Push the lower one down
                labels[i]['y'] = labels[i-1]['y'] - min_dist
        
        for lbl in labels:
            plt.text(
                x=lbl['x'], 
                y=lbl['y'], 
                s=lbl['text'], 
                color=lbl['color'], 
                fontweight='bold', 
                verticalalignment='center',
                horizontalalignment=lbl['align']
            )

    plot_adjusted_labels(final_start_labels)
    plot_adjusted_labels(final_end_labels)

    plt.suptitle(title, fontsize=28, weight='bold', y=0.98)
    
    plt.xlabel("Debate Round", labelpad=15)
    plt.ylabel("Agent Stance\n(-1.0 Opposing <---> +1.0 Supporting)", labelpad=15)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-0.5, max_round + 0.8) # Give some left margin
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhspan(-0.1, 0.1, color='gray', alpha=0.1) # Neutral Zone

    # Remove standard legend frame for cleaner look
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # make room for suptitle
    
    output_path = os.path.join(output_dir, f"{output_basename}_modern.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Modern Plot saved to: {output_path}")
    
    # Also save as the standard name to ensure backward compatibility/linking
    standard_output_path = os.path.join(output_dir, f"{output_basename}.png")
    plt.savefig(standard_output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_family_splits(df, output_dir):
    family_titles = {
        "corebot": "CoreBot Stance Evolution Trajectories",
        "openclaw": "OpenClaw Stance Evolution Trajectories",
        "other": "Other Stance Evolution Trajectories",
    }

    df = df.copy()
    df["agent_family"] = df.apply(detect_agent_family, axis=1)
    families = [family for family in ["corebot", "openclaw", "other"] if family in set(df["agent_family"])]

    if len(families) <= 1:
        family = families[0] if families else "other"
        if family in family_titles:
            basic_trajectory_plot(
                df,
                output_dir,
                output_basename=f"stance_trajectories_{family}",
                title=family_titles.get(family, "Stance Evolution Trajectories"),
            )
        basic_trajectory_plot(df, output_dir)
        return

    for family in families:
        family_df = df[df["agent_family"] == family]
        if family_df.empty:
            continue
        basic_trajectory_plot(
            family_df,
            output_dir,
            output_basename=f"stance_trajectories_{family}",
            title=family_titles.get(family, "Stance Evolution Trajectories"),
        )


def plot_stance_history(csv_path):
    """
    Main entry point used by run_experiment.py
    """
    print(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if "stance_history" not in df.columns:
        print("Error: 'stance_history' column not found in CSV.")
        return

    output_dir = os.path.dirname(csv_path)
    
    # Ensure config_name configures sorting
    if 'config_name' in df.columns:
        df = df.sort_values(by='config_name')
        
    set_style()
    plot_family_splits(df, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot stance history from batch results CSV.")
    parser.add_argument("csv_path", help="Path to the batch results CSV file")
    args = parser.parse_args()
    
    plot_stance_history(args.csv_path)
