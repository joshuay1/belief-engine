import matplotlib.pyplot as plt
import seaborn as sns

# --- Color Palettes ---
COLORS_GENERIC = [
    "#2C3E50",
    "#2874A6",
    "#3498DB",
    "#1ABC9C",
    "#2ECC71",
    "#F1C40F",
    "#E67E22",
    "#D35400",
    "#E74C3C",
    "#8E44AD",
]

COLORS_SENSITIVITY = {
    'Update_0.1_Glacial': '#2C3E50', 
    'Update_0.2_Conservative': '#2874A6',
    'Update_0.3_Cautious': '#3498DB',
    'Update_0.4_Steady': '#1ABC9C',
    'Update_0.5_Balanced': '#2ECC71',
    'Update_0.6_Adaptive': '#F1C40F',
    'Update_0.7_Resilient': '#D4AC0D',
    'Update_0.8_Responsive': '#E67E22',
    'Update_0.9_Proactive': '#D35400',
    'Update_1.0_Reactive': '#E74C3C'
}

COLORS_BIAS = {
    'Bias_0.0_Objective': '#2ECC71',
    'Bias_0.3_Subtle': '#F1C40F', 
    'Bias_0.6_Strong': '#E67E22',
    'Bias_0.9_Radical': '#C0392B'
}

COLORS_MODELS = {
    "GPT-4o-Mini": "#3498DB",
    "gpt-4o-mini": "#3498DB",
    "GPT-5": "#8E44AD",
    "gpt-5": "#8E44AD"
}

COLORS_SEEDING = {
    "Strategy_Quality_Max": "#2ECC71",
    "Strategy_Quality_Balanced": "#3498DB",
    "Strategy_Random": "#E74C3C"
}

COLORS_ARCHITECTURE = {
    "Arch_1_Baseline": "#95A5A6",
    "Arch_2_LogicFiltered": "#2ECC71",
    "Arch_3_Amem_Vector": "#3498DB",
    "Arch_4_FullAgentic": "#8E44AD"
}

COLORS_ANCHOR = {
    'Anchor 0.1': '#E74C3C', # Red (Weak/Fragile)
    'Anchor 0.3': '#E67E22', # Orange
    'Anchor 0.5': '#F1C40F', # Yellow
    'Anchor 0.7': '#2ECC71', # Green
    'Anchor 0.9': '#8E44AD'  # Purple (Strong/Solid)
}

COLORS_SIMILARITY = {
    "Threshold_0.6": "#2ECC71", # Green (Stable)
    "Threshold_0.7": "#F1C40F", # Yellow/Orange (Tipping)
    "Threshold_0.8": "#E67E22", # Orange (Collapse)
    "Threshold_0.9": "#E74C3C"  # Red (Deep Collapse)
}

COLORS_SEEDING_BINS = {
    'Bin_1_Elite_0.98_1.00': '#2ECC71', # Green
    'Bin_2_High_0.81_0.98': '#F1C40F', # Yellow
    'Bin_3_Broad_0.00_0.81': '#E74C3C', # Red
}

SYMMETRICAL_EXPERIMENT_STYLE = {
    "figure_size": (24, 12),
    "dpi": 200,
    "pro_color": "#2F6B3B",
    "con_color": "#C65A1E",
    "neutral_color": "#9E9E9E",
    "y_limits": (-1.1, 1.1),
    "neutral_band": (-0.1, 0.1),
    "neutral_line_width": 1.6,
    "neutral_line_alpha": 0.35,
    "neutral_band_alpha": 0.05,
    "title_size": 30,
    "label_size": 28,
    "tick_size": 25,
    "line_width": 5.5,
    "trial_line_width": 1.0,
    "trial_line_alpha": 0.15,
    "fill_alpha": 0.12,
    "annotation_size": 24,
    "stats_size": 24,
    "legend_size": 22,
}


def get_symmetrical_style(overrides=None):
    """Return the shared style config for symmetrical debate experiment plots."""
    style = dict(SYMMETRICAL_EXPERIMENT_STYLE)
    if overrides:
        style.update(overrides)
    return style


def apply_modern_style():
    """Applies the standardized 'Modern Heavy' plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 28,
        'lines.linewidth': 4,
        'lines.markersize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def apply_symmetrical_style(overrides=None):
    """Apply standardized rcParams for symmetrical debate plots."""
    style = get_symmetrical_style(overrides=overrides)
    apply_modern_style()
    plt.rcParams.update({
        "axes.titlesize": style["title_size"],
        "axes.labelsize": style["label_size"],
        "xtick.labelsize": style["tick_size"],
        "ytick.labelsize": style["tick_size"],
        "legend.fontsize": style["legend_size"],
        "lines.linewidth": style["line_width"],
    })
    return style


def style_symmetrical_axis(ax, title, style, title_pad=8, show_grid=False):
    """Apply the shared symmetrical style to a single axis."""
    ax.set_title(title, fontsize=style["title_size"], fontweight="bold", pad=title_pad)
    ax.set_ylim(*style["y_limits"])
    ax.axhline(
        y=0,
        color=style["neutral_color"],
        linestyle="--",
        alpha=style["neutral_line_alpha"],
        linewidth=style["neutral_line_width"],
    )
    ax.axhspan(
        style["neutral_band"][0],
        style["neutral_band"][1],
        color=style["neutral_color"],
        alpha=style["neutral_band_alpha"],
    )
    ax.grid(show_grid)
    ax.tick_params(axis="both", which="major", labelsize=style["tick_size"])


def setup_figure(title, xlabel="Debate Round", ylabel="Agent Stance (Pro/Con)", figsize=(14, 9)):
    """Creates a standard figure and axes with labels."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=24, weight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=20, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=15)
    ax.set_ylim(-1.1, 1.1)
    
    # Standard Guidelines
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhspan(-0.1, 0.1, color='gray', alpha=0.1) # Neutral Zone
    
    return fig, ax

def finalize_plot(fig, ax, output_path, legend_loc='lower left'):
    """Finalizes and saves the plot."""
    ax.legend(frameon=False, fontsize=18, loc=legend_loc)
    sns.despine(left=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def plot_trajectory_group(ax, df, group_col, stance_col='stance', round_col='round', color_map=None):
    """
    Plots mean trajectories with error bands for groups in the dataframe.
    """
    groups = df[group_col].unique()
    try:
        groups = sorted(groups)
    except:
        pass
        
    for i, group in enumerate(groups):
        subset = df[df[group_col] == group]
        if subset.empty: continue
        
        # Aggregate stats
        stats = subset.groupby(round_col)[stance_col].agg(['mean', 'std']).reset_index()
        
        # Determine Color
        color = '#333333'
        if color_map:
            color = color_map.get(str(group), COLORS_GENERIC[i % len(COLORS_GENERIC)])
        else:
            color = COLORS_GENERIC[i % len(COLORS_GENERIC)]
            
        label = str(group).replace('_', ' ')
        
        # Plot
        ax.plot(stats[round_col], stats['mean'], label=label, color=color, marker='o')
        ax.fill_between(stats[round_col], 
                        stats['mean'] - stats['std'], 
                        stats['mean'] + stats['std'], 
                        color=color, alpha=0.15)
