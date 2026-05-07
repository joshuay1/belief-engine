
import subprocess
import sys
import os
import argparse
import glob

# Define the Experiment Registry (from Paper)
# Keys map to --experiment arguments
# Values: (config_file, plotter_script)
EXPERIMENTS = {
    "sensitivity": {
        "config": "experiments/definitions/sensitivity_decoupled.yaml",
        "plotter": "sensitivity"
    },
    "anchor": {
        "config": "experiments/definitions/anchor_sensitivity_sweep.yaml",
        "plotter": "anchor"
    },
    "architecture": {
        "config": "experiments/definitions/architecture_sweep.yaml",
        "plotter": "architecture"
    },
    "seeding": {
        "config": "experiments/definitions/seeding_sweep.yaml",
        "plotter": "seeding"
    },
    "identity": {
        "config": "experiments/definitions/similarity_threshold_sweep.yaml",
        "plotter": "identity"
    },
    "echo_chamber": {
        "config": "experiments/definitions/echo_chamber.yaml",
        "plotter": "generic" # Uses generic plotter for drift
    },
    "robustness": {
        "config": "experiments/definitions/topic_robustness_sweep.yaml",
        "plotter": "robustness"
    }
}

def run_command(cmd, dry_run=False):
    print(f"\n[EXEC] {cmd}")
    if dry_run: return
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print(f"Error executing command: {cmd}")
        sys.exit(1)

def find_latest_log_dir(config_name):
    # Assumes logs are named like logs/run_<config_name>_<timestamp> or similar
    # We search for any directory containing the config name created recently
    search_pattern = f"experiments/logs/*{config_name}*"
    candidates = glob.glob(search_pattern)
    candidates = [c for c in candidates if os.path.isdir(c)]
    
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)

def main():
    parser = argparse.ArgumentParser(description="Run a full experiment pipeline: Execute -> Recover -> Plot.")
    
    # Mode 1: New/Custom Experiment
    parser.add_argument("--config", type=str, help="Path to experiment config YAML (e.g., experiments/definitions/my_new_sweep.yaml)")
    
    # Mode 2: Reproduce Paper Experiment
    parser.add_argument("--experiment", type=str, choices=list(EXPERIMENTS.keys()), help="Reproduce a specific figure from the paper.")
    
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per variation (default: 3)")
    parser.add_argument("--plotter", type=str, default="generic", 
                        choices=["generic", "sensitivity", "seeding", "anchor", "identity", "robustness", "architecture"],
                        help="Which standard plotter to use (if running custom config). Default: generic.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")

    args = parser.parse_args()
    
    # Resolve Configuration
    config_path = None
    plotter_alias = args.plotter
    
    if args.experiment:
        # Reproduction Mode
        details = EXPERIMENTS[args.experiment]
        config_path = details['config']
        plotter_alias = details['plotter']
        print(f"--- Reproducing Paper Experiment: {args.experiment} ---")
        print(f"Config: {config_path}")
    elif args.config:
        # Custom Mode
        config_path = args.config
        print(f"--- Running Custom Experiment ---")
    else:
        print("Error: You must specify either --config (for new experiments) or --experiment (for paper reproduction).")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # 1. Run Experiment
    print(f"--- Step 1: Running {args.trials} trials ---")
    run_cmd = f"python src/experiment/run_experiment.py --config {config_path} --trials {args.trials}"
    run_command(run_cmd, args.dry_run)
    
    if args.dry_run:
        print("[DRY-RUN] Pipeline finished.")
        return

    # 2. Identify Log Directory
    config_name = os.path.basename(config_path).replace(".yaml", "")
    log_dir = find_latest_log_dir(config_name)
    
    if not log_dir:
        print(f"Error: Could not find log directory for {config_name}. Did the run fail?")
        sys.exit(1)
        
    print(f"--- Found Standard Log Directory: {log_dir} ---")

    # 3. Recover Results
    print(f"--- Step 2: Recovering Data ---")
    recover_cmd = f"python src/experiment/recover_results.py {log_dir}"
    run_command(recover_cmd)

    # 4. Plot Results
    print(f"--- Step 3: Generating Standard Plot ({plotter_alias}) ---")
    
    # Map plotter alias to script path
    plotter_map = {
        "generic": "src/analysis/plots/plot_generic_sweep.py",
        "sensitivity": "src/analysis/plots/plot_sensitivity_calibration.py",
        "seeding": "src/analysis/plots/plot_seeding_results.py",
        "anchor": "src/analysis/plots/plot_anchor_sensitivity.py",
        "identity": "src/analysis/plots/plot_similarity_threshold.py",
        "robustness": "src/analysis/plots/plot_topic_robustness.py",
        "architecture": "src/analysis/plots/plot_architecture_sweep.py"
    }
    
    script_path = plotter_map.get(plotter_alias, plotter_alias) # Allow custom path too
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"Error: Plotter script not found at {script_path}")
        sys.exit(1)

    # Construct command (some legacy plotters take log_dir, modern take csv_path)
    legacy_log_dir_plotters = ["plot_sensitivity_calibration.py", "plot_seeding_results.py", "plot_anchor_sensitivity.py"]
    
    target_arg = os.path.join(log_dir, "results.csv")
    if any(p in script_path for p in legacy_log_dir_plotters):
        target_arg = log_dir
        
    plot_cmd = f"python {script_path} {target_arg}"
    run_command(plot_cmd)
    
    print(f"\n[SUCCESS] Pipeline Complete. Artifacts located in {log_dir}")
    
if __name__ == "__main__":
    main()
