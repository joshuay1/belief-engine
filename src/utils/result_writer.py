import os
import json
import yaml
import traceback
from datetime import datetime

class ResultWriter:
    """
    Centralized utility for standardized result output.
    Handles directory setup and artifact serialization (JSON, YAML, CSV, Markdown).
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    @classmethod
    def create_from_config(cls, config_name: str, base_dir: str = "experiments/logs") -> 'ResultWriter':
        """
        Creates a new log directory based on config name and timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_dir, config_name, timestamp)
        return cls(log_dir)

    def get_log_dir(self) -> str:
        return self.log_dir

    def _secure_path(self, filename: str) -> str:
        path = os.path.join(self.log_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def save_json(self, filename: str, data: dict, indent: int = 2):
        """Standard JSON dumper."""
        path = self._secure_path(filename)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=indent, default=str)
        except Exception as e:
            print(f"    [WARN] ResultWriter failed to save {filename}: {e}")

    def save_yaml(self, filename: str, data: dict):
        """Standard YAML dumper."""
        path = self._secure_path(filename)
        try:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"    [WARN] ResultWriter failed to save {filename}: {e}")

    def save_csv(self, filename: str, dataframe):
        """Standard CSV dumper using pandas."""
        path = self._secure_path(filename)
        try:
            dataframe.to_csv(path, index=False)
        except Exception as e:
            print(f"    [WARN] ResultWriter failed to save {filename}: {e}")

    def save_text(self, filename: str, content: str):
        """Standard generic text/markdown dumper."""
        path = self._secure_path(filename)
        try:
            with open(path, "w") as f:
                f.write(content)
        except Exception as e:
            print(f"    [WARN] ResultWriter failed to save {filename}: {e}")

    # --- Domain Specific Savers ---

    def save_memory_dump(self, agents: list, filename: str = "memory_dump.json"):
        """Extracts and saves memory from applicable agents."""
        memory_dump = {
            agent.name: agent.memory.get_all_arguments()
            for agent in agents
            if getattr(agent, "memory", None) is not None
        }
        if memory_dump:
            self.save_json(filename, memory_dump)

    def save_settings_summary(self, config_name: str, config_path: str, config: dict, filename: str = "settings_summary.txt"):
        """Saves a basic plaintext settings summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mem_conf = config.get("memory", {})
        
        content = f"Simulation Config: {config_name}\n"
        content += f"Timestamp: {timestamp}\n"
        content += f"Config Path: {config_path}\n"
        content += "-" * 20 + "\n"
        content += f"Worldview Openness: {mem_conf.get('worldview_openness', 'unknown')}\n"
        content += f"Retrieval Strategy: {mem_conf.get('retrieval_strategy', 'unknown')}\n"
        content += f"Update Sensitivity: {mem_conf.get('update_sensitivity', 'unknown')}\n"
        self.save_text(filename, content)

    def save_debate_transcript(self, result: dict, config_name: str, trial: int, filename: str = "transcript.md"):
        """Saves a Markdown transcript of the debate."""
        transcript = result.get("transcript", [])
        if not transcript:
            return

        content = f"# Debate Transcript\n"
        content += f"**Config**: {config_name}\n"
        content += f"**Trial**: {trial}\n"
        content += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += "---\n\n"
        
        role_map = {"user": "Counter Agent", "assistant": "Subject Agent"}
        for msg in transcript:
            role = role_map.get(msg['role'], msg['role'])
            msg_content = msg['content']
            content += f"### {role}\n{msg_content}\n\n"
            
        self.save_text(filename, content)
        print(f"    Transcript saved to: {os.path.join(self.log_dir, filename)}")

    def save_experiment_summary(self, results: list, experiment_name: str, run_id: str, filename: str = "summary_report.md"):
        """Generates a comprehensive Markdown summary table of experiment results."""
        content = f"# Experiment Summary: {experiment_name}\n"
        content += f"**Run ID**: `{run_id}`\n"
        content += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += "## 1. Results Overview\n"
        content += "| Config / Variation | Initial Stance | Final Stance | Change | Outcome |\n"
        content += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        for r in results:
            if "error" in r:
                content += f"| {r.get('config_name', 'Unknown')} | N/A | N/A | N/A | **ERROR**: {r['error']} |\n"
                continue
                
            initial = r.get("initial_stance", 0.0)
            final = r.get("final_stance", 0.0)
            change = final - initial
            
            outcome = "Stable"
            if abs(change) > 0.5: outcome = "Significant Drift"
            if (initial > 0 and final < 0) or (initial < 0 and final > 0):
                outcome = "**FLIPPED**"
                
            content += f"| **{r.get('config_name', 'Unknown')}** | {initial:.4f} | {final:.4f} | {change:+.4f} | {outcome} |\n"
            
        content += "\n## 2. Visualizations\n"
        split_plot_filenames = [
            ("CoreBot Trajectories", "stance_trajectories_corebot.png"),
            ("OpenClaw Trajectories", "stance_trajectories_openclaw.png"),
        ]
        split_found = False
        for label, plot_filename in split_plot_filenames:
            abs_plot_path = os.path.abspath(os.path.join(self.log_dir, plot_filename))
            if os.path.exists(abs_plot_path):
                content += f"### {label}\n"
                content += f"![{label}](file://{abs_plot_path})\n\n"
                split_found = True

        if not split_found:
            plot_filename = "stance_trajectories.png"
            abs_plot_path = os.path.abspath(os.path.join(self.log_dir, plot_filename))
            if os.path.exists(abs_plot_path):
                content += f"![Stance Trajectories](file://{abs_plot_path})\n"
            else:
                content += "*No trajectory plot found.*\n"
            
        content += "\n## 3. Configuration Details\n"
        content += "### Variations\n"
        for r in results:
            if "error" in r: continue
            name = r.get("config_name", "Default")
            content += f"#### {name}\n```yaml\n"
            for k, v in r.items():
                if k.startswith("var_"):
                     content += f"{k[4:]}: {v}\n"
            content += "```\n"

        self.save_text(filename, content)
        print(f"    Summary report saved to: {os.path.join(self.log_dir, filename)}")
