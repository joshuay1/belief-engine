import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LivePlotter:
    """
    Watches a growing results.csv file emitted by the agentmem runners
    and dynamically updates a Matplotlib figure without blocking the main event loops.
    """
    def __init__(self, log_dir: str, plot_type: str = "trajectory", interval_ms: int = 2000):
        self.log_dir = log_dir
        self.plot_type = plot_type
        self.interval_ms = interval_ms
        self.csv_path = os.path.join(log_dir, "results.csv")
        self.fig = None
        self.ax = None
        
    def start(self):
        print(f"Starting Live Plotter matching '{self.plot_type}' on:")
        print(f"  -> {self.csv_path}")
        print("Waiting for initial data... (Close the native window to exit)")
        
        # Use our standard style
        try:
            from src.utils.plot_theme import apply_modern_style
            apply_modern_style()
        except ImportError:
            pass # Fallback to default
            
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("AgentMem Live Visualizer")
        
        ani = FuncAnimation(
            self.fig, self.update_plot, interval=self.interval_ms, cache_frame_data=False
        )
        plt.show()

    def update_plot(self, frame):
        if not os.path.exists(self.csv_path):
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Waiting for results.csv to be created...", 
                         ha='center', va='center', fontsize=14)
            self.ax.axis('off')
            return

        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return
        except Exception as e:
            # File might be mid-write, ignore this frame
            return

        self.ax.clear()
        self.ax.axis('on')

        if self.plot_type == "trajectory":
            self._draw_trajectory(df)
        else:
            self.ax.text(0.5, 0.5, f"Unsupported plot_type: {self.plot_type}", 
                         ha='center', va='center', fontsize=14)

    def _draw_trajectory(self, df):
        # We dynamically import the history parsers so we don't duplicate logic
        try:
            from src.utils.plot_history import normalize_stance_history, get_display_label, get_custom_colors
        except ImportError:
            # Fallbacks just in case
            def normalize_stance_history(x): return []
            def get_display_label(x): return x
            def get_custom_colors(): return {}

        import seaborn as sns
        custom_colors = get_custom_colors()
        max_round = 0
        
        # Only plot the last 20 trajectories to avoid massive clutter during deep sweeps
        plot_df = df.tail(20)
        
        for index, row in plot_df.iterrows():
            config_name = row.get("config_name", f"Run {index}")
            display_label = get_display_label(config_name)
            
            history = normalize_stance_history(row.get("stance_history", ""))
            
            if not history:
                continue
                
            initial_stance = row.get("initial_stance", 0.0)
            full_history = [initial_stance] + history
            rounds = range(len(full_history))
            max_round = max(max_round, len(full_history) - 1)
            
            # Select color
            palette = sns.color_palette("husl", n_colors=10)
            color = custom_colors.get(config_name, palette[index % 10])
            
            # Highlight the most recent trial by making older ones slightly faded
            is_latest = (index == df.index[-1])
            alpha = 0.9 if is_latest else 0.4
            linewidth = 4 if is_latest else 2
            
            self.ax.plot(rounds, full_history, marker='o', label=display_label, 
                         color=color, linewidth=linewidth, alpha=alpha)

        self.ax.set_title(f"Live Stance Trajectories (Trials: {len(df)})", fontsize=20, weight='bold', pad=15)
        self.ax.set_xlabel("Debate Round", fontsize=16)
        self.ax.set_ylabel("Agent Stance (-1.0 to +1.0)", fontsize=16)
        self.ax.set_ylim(-1.1, 1.1)
        if max_round > 0:
            self.ax.set_xlim(-0.5, max_round + 0.5)
            
        self.ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        self.ax.axhspan(-0.1, 0.1, color='gray', alpha=0.1) # Neutral Zone
        
        # Deduplicate legend labels (since we might have multiple trials of the same config)
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label and len(by_label) <= 15:
            self.ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
        
        self.fig.tight_layout()
