import os
import datetime

class RegistryManager:
    def __init__(self, registry_path="experiments/registry.md"):
        self.registry_path = registry_path
        # Ensure project root is handling paths correctly
        if not os.path.exists(os.path.dirname(self.registry_path)):
             os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

    def log_experiment(self, run_id: str, config_name: str, status: str = "Pending Analysis", insight: str = ""):
        """
        Appends a new experiment run to the registry markdown table.
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Format the table row
        # | Date | Run ID | Config Name | Status | Insight |
        row = f"| {date_str} | `{run_id}` | {config_name} | {status} | {insight} |"
        
        # Append to file
        try:
            with open(self.registry_path, "a") as f:
                f.write(row + "\n")
            print(f"Logged experiment {run_id} to {self.registry_path}")
        except Exception as e:
            print(f"[ERROR] Failed to log to registry: {e}")

    def update_status(self, run_id: str, new_status: str, insight: str = None):
        """
        Updates the status and insight of an existing run.
        (Simple implementation: reads all lines, modifies matching line, writes back)
        """
        try:
            with open(self.registry_path, "r") as f:
                lines = f.readlines()
            
            with open(self.registry_path, "w") as f:
                for line in lines:
                    if f"`{run_id}`" in line:
                        # Reconstruct the line
                        parts = [p.strip() for p in line.split("|")]
                        # parts[0] is empty (before first |)
                        # parts[1] is Date
                        # parts[2] is Run ID
                        # parts[3] is Config Name
                        # parts[4] is Status
                        # parts[5] is Insight
                        
                        current_insight = parts[5] if len(parts) > 5 else ""
                        final_insight = insight if insight is not None else current_insight
                        
                        new_line = f"| {parts[1]} | {parts[2]} | {parts[3]} | {new_status} | {final_insight} |\n"
                        f.write(new_line)
                    else:
                        f.write(line)
                        
            print(f"Updated status for {run_id} in registry.")
            
        except Exception as e:
             print(f"[ERROR] Failed to update registry: {e}")
