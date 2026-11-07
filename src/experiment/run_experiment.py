import argparse
import yaml
import os
import sys
import pandas as pd
from datetime import datetime
import json
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.experiment.orchestrator import run_experiment
from src.experiment.registry_manager import RegistryManager
from src.utils.plot_history import plot_stance_history

from src.utils.config import ConfigLoader

def save_transcript(result, config_name, trial):
    """
    Saves a readable markdown transcript of the debate.
    """
    transcript = result.get("transcript", [])
    log_dir = result.get("log_dir")
    
    if not log_dir or not transcript:
        return

    output_path = os.path.join(log_dir, "transcript.md")
    
    with open(output_path, 'w') as f:
        f.write(f"# Debate Transcript\n")
        f.write(f"**Config**: {config_name}\n")
        f.write(f"**Trial**: {trial}\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        
        role_map = {"user": "Counter Agent", "assistant": "Subject Agent"}
        
        for msg in transcript:
            role = role_map.get(msg['role'], msg['role'])
            content = msg['content']
            
            f.write(f"### {role}\n")
            f.write(f"{content}\n\n")
            
    print(f"    Transcript saved to: {output_path}")

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _sweep_self_argument_setting_is_explicit(config_dict):
    """Return True if a sweep config explicitly sets the self-argument flag."""
    memory_config = config_dict.get("memory", {})
    if isinstance(memory_config, dict):
        if "allow_self_arguments" in memory_config:
            return True

    return "memory.allow_self_arguments" in config_dict


def _get_config_value(config_dict, dotted_key, default=None):
    """Read a value from nested and/or dotted-key config dictionaries."""
    if dotted_key in config_dict:
        return config_dict[dotted_key]

    current = config_dict
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current


def _config_sets_value_explicitly(config_dict, dotted_key):
    """Return True if the provided config explicitly contains the given key."""
    if dotted_key in config_dict:
        return True

    current = config_dict
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]

    return True


def _apply_sweep_defaults(config_dict):
    """
    Sweep experiments disable self-produced arguments by default unless the
    sweep definition explicitly opts in.
    """
    if _sweep_self_argument_setting_is_explicit(config_dict):
        return config_dict

    memory_config = config_dict.get("memory")
    if isinstance(memory_config, dict):
        memory_config["allow_self_arguments"] = False
    else:
        config_dict["memory.allow_self_arguments"] = False

    return config_dict


def _warn_on_confounded_anchor_sweep(config_name, variation_overrides, merged_config):
    """
    Warn when a sweep variation explicitly sets anchor_sensitivity while a
    target_initial_stance is active, since runtime will overwrite the anchor.
    """
    if not _config_sets_value_explicitly(variation_overrides, "memory.anchor_sensitivity"):
        return

    target_initial_stance = _get_config_value(merged_config, "debate.target_initial_stance", None)
    if target_initial_stance is None or str(target_initial_stance).lower() == "none":
        return

    anchor_sensitivity = _get_config_value(merged_config, "memory.anchor_sensitivity", "Def")
    print(
        "WARNING: "
        f"{config_name} explicitly sets memory.anchor_sensitivity={anchor_sensitivity}, "
        f"but debate.target_initial_stance={target_initial_stance} is also active. "
        "The runner will auto-calculate anchor_sensitivity from the target stance, "
        "so this variation will not measure the anchor parameter cleanly. "
        "Set debate.target_initial_stance: none for anchor sweeps."
    )

def main():
    parser = argparse.ArgumentParser(description="Run an experiment from a YAML definition.")
    parser.add_argument("--config", required=True, help="Path to the experiment definition YAML file.")
    parser.add_argument("--name", help="Optional name/tag for this run (defaults to config name + timestamp).")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per variation.")
    parser.add_argument("--resume", help="Resume an existing run by providing the run_id. Will skip already completed trials.")
    args = parser.parse_args()

    # 1. Load Definition
    print(f"Loading experiment definition from: {args.config}")
    try:
        exp_def = ConfigLoader.load_config(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return
    
    # 2. Setup Run Identity
    if "name" not in exp_def:
        # Fallback: Use config filename stem
        config_filename = os.path.basename(args.config)
        experiment_name = os.path.splitext(config_filename)[0]
    else:
        experiment_name = exp_def["name"]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # If resuming, use that run_id; else if --name provided, use it; else generate new
    if args.resume:
        run_id = args.resume
        print(f"[RESUME MODE] Continuing run: {run_id}")
    else:
        run_id = args.name if args.name else f"{experiment_name}_{timestamp}"
    
    # Extract optional fields from definition
    description = exp_def.get("description", "")
    variations = exp_def.get("variations", [])
    base_config = exp_def.get("base_config", {})
    
    # Create Log Directory
    log_dir = os.path.join("experiments/logs", run_id)
    os.makedirs(log_dir, exist_ok=True)
    
    # 3. Initialize Registry
    registry = RegistryManager()
    registry.log_experiment(run_id, experiment_name, status="Running", insight=description)
    
    print(f"Starting Experiment: {experiment_name}")
    print(f"Run ID: {run_id}")
    print(f"Logs: {log_dir}")
    
    # Load existing results if resuming
    completed_trials = set()
    results = []
    csv_path = os.path.join(log_dir, "results.csv")
    if args.resume and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        for _, row in existing_df.iterrows():
            completed_trials.add((row['config_name'], int(row['trial'])))
        results = existing_df.to_dict('records')
        print(f"[RESUME] Loaded {len(completed_trials)} completed trials from existing CSV")
    
    # 4. Enumerate Configurations
    # If no variations are defined, run just the base config
    configs_to_run = []
    
    if not variations:
        configs_to_run.append({
            "name": "Default",
            "overrides": {}
        })
    else:
        for i, var in enumerate(variations):
            # var is a dictionary of overrides
            # We can support a "name" key in the variation for better logging, or auto-generate
            var_name = var.pop("name", f"Var_{i+1}")
            
            # Check if there is an explicit "overrides" key (preferred structure)
            if "overrides" in var:
                actual_overrides = var["overrides"]
            else:
                # Otherwise, assume the remaining keys are the overrides
                actual_overrides = var
                
            configs_to_run.append({
                "name": var_name,
                "overrides": actual_overrides
            })
            
    total_steps = len(configs_to_run) * args.trials
    current_step = 0
    
    # 5. Execution Loop
    try:
        for config_item in configs_to_run:
            config_name = config_item["name"]
            variation_overrides = config_item["overrides"]
            
            # Merge base_config and variation_overrides
            # This logic assumes simple key-value merging. 
            # Ideally, base_config should be flattened or handled by the orchestrator.
            # Here we will pass a merged dictionary to the orchestrator.
            
            # Start with base
            # Deep Merge base_config and variation_overrides
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            final_overrides = base_config.copy()
            # We need a deep copy of base_config to avoid mutating it across iterations,
            # but base_config.copy() is shallow.
            # Using deep_update on a fresh dict might be safer?
            # Or just deepcopy base_config.
            import copy
            final_overrides = copy.deepcopy(base_config)
            
            deep_update(final_overrides, variation_overrides)
            _apply_sweep_defaults(final_overrides)
            _warn_on_confounded_anchor_sweep(config_name, variation_overrides, final_overrides)

            # Extract optional plotting hooks (do not pass to orchestrator)
            plot_per_trial = final_overrides.pop("plot_per_trial", False)
            plot_per_trial_script = final_overrides.pop("plot_per_trial_script", None)
            
            # Print Configuration (Before Flattening)
            print(f"\n{'='*60}")
            print(f"RUNNING CONFIGURATION: {config_name}")
            print(f"{'='*60}")
            
            deb_conf = final_overrides.get("debate", {})
            mem_conf = final_overrides.get("memory", {})
            
            print(f"Topic: {deb_conf.get('topic', 'Unknown')}")
            print(f"Agent Type: {deb_conf.get('agent_type', 'Unknown')}")
            print(f"Stance Method: {mem_conf.get('stance_method', 'N/A')}")
            print(f"Storage Mode: {mem_conf.get('storage_mode', 'N/A')}")
            print(f"Allow Self Arguments: {mem_conf.get('allow_self_arguments', False)}")
            print(f"Sensitivity (Upd/Anch): {mem_conf.get('update_sensitivity', 'Def')} / {mem_conf.get('anchor_sensitivity', 'Def')}")
            print(f"Decay Factor: {mem_conf.get('decay_factor', 'Def')}")
            print(f"{'-'*60}")

            # Flatten to ensure dot-notation keys work if they were nested in YAML
            final_overrides = flatten_dict(final_overrides)

            for trial in range(args.trials):
                current_step += 1
                
                # Skip if already completed (resume mode)
                if (config_name, trial + 1) in completed_trials:
                    print(f"    Trial {trial+1}/{args.trials} (Global: {current_step}/{total_steps}) - SKIPPED (already complete)")
                    continue
                    
                print(f"    Trial {trial+1}/{args.trials} (Global: {current_step}/{total_steps})")
                
                try:
                    # Execute
                    result = run_experiment(
                        config_overrides=final_overrides, 
                        return_results=True, 
                        batch_dir=log_dir,
                        run_name_prefix=config_name
                    )
                    
                    # Augment Result
                    result["run_id"] = run_id
                    result["config_name"] = config_name
                    result["trial"] = trial + 1
                    # Add variation params to result for easy CSV analysis
                    for k, v in variation_overrides.items():
                         result[f"var_{k}"] = v

                    # Save per-trial results.json (for plotting/aggregation)
                    try:
                        result_log_dir = result.get("log_dir")
                        if result_log_dir:
                            results_path = os.path.join(result_log_dir, "results.json")
                            with open(results_path, "w") as f:
                                json.dump(result, f, indent=2, default=str)
                    except Exception as e:
                        print(f"    [WARN] Failed to save results.json: {e}")
                         
                    # Save Readable Transcript
                    try:
                        save_transcript(result, config_name, trial + 1)
                    except Exception as e:
                        print(f"    [WARN] Failed to save transcript: {e}")

                    results.append(result)
                    
                except Exception as e:
                    print(f"    [ERROR] Trial failed: {e}")
                    traceback.print_exc()
                    results.append({
                        "run_id": run_id,
                        "config_name": config_name,
                        "trial": trial + 1,
                        "error": str(e)
                    })
                    
                # Incremental Save (Inside Trial Loop)
                if results:
                    df = pd.DataFrame(results)
                    csv_path = os.path.join(log_dir, "results.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"    [INFO] Incremental save to: {csv_path}")

                    # Generate plots after each trial if configured
                    if plot_per_trial and plot_per_trial_script:
                        try:
                            import subprocess
                            print("    [INFO] Generating per-trial plots...")
                            subprocess.run(
                                ["python", plot_per_trial_script, "--run_id", run_id],
                                check=False
                            )
                        except Exception as e:
                            print(f"    [WARN] Failed to generate per-trial plots: {e}")
                    
                    # Generate Preliminary Plots every 4 trials (1 cycle of agents)
                    if (not plot_per_trial) and len(results) % 4 == 0:
                        print(f"    [INFO] Generating interim plots (Cycle {len(results)//4} Complete)...")
                        try:
                            import subprocess
                            subprocess.run(
                                ["python", "src/analysis/plots/plot_benchmark_4cond.py", "--run_id", run_id],
                                check=False
                            )
                        except Exception as e:
                            print(f"    [WARN] Failed to generate interim plots: {e}")

        # 6. Save Final Results
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(log_dir, "results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results saved to: {csv_path}")
            
            # Generate Plots
            # 1. Trajectories
            try:
                plot_stance_history(csv_path)
            except Exception as e:
                print(f"[WARN] Could not plot trajectories: {e}")

            # Generate Markdown Summary
            try:
                summary_path = os.path.join(log_dir, "summary_report.md")
                generate_summary_md(results, experiment_name, run_id, log_dir, summary_path)
                print(f"Summary report saved to: {summary_path}")
            except Exception as e:
                print(f"[WARN] Could not generate summary report: {e}")
                traceback.print_exc()

            # Update Registry
            registry.update_status(run_id, "Complete", insight=f"Saved to {log_dir}")
        else:
            print("[WARN] No results generated.")
            registry.update_status(run_id, "Failed", insight="No results generated")
            
    except Exception as e:
        print(f"[FATAL] Experiment crashed: {e}")
        traceback.print_exc()
        registry.update_status(run_id, "Crashed", insight=str(e))

def generate_summary_md(results, name, run_id, log_dir, output_path):
    """
    Generates a markdown summary of the experiment results.
    """
    with open(output_path, 'w') as f:
        f.write(f"# Experiment Summary: {name}\n")
        f.write(f"**Run ID**: `{run_id}`\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Results Overview\n")
        f.write("| Config / Variation | Initial Stance | Final Stance | Change | Outcome |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for r in results:
            if "error" in r:
                f.write(f"| {r.get('config_name', 'Unknown')} | N/A | N/A | N/A | **ERROR**: {r['error']} |\n")
                continue
                
            # Orchestrator returns: { 'transcript': ..., 'stance_history': ..., 'initial_stance': ..., 'final_stance': ... }
            initial = r.get("initial_stance", 0.0)
            final = r.get("final_stance", 0.0)
            change = final - initial
            
            # Outcome Logic
            outcome = "Stable"
            if abs(change) > 0.5: outcome = "Significant Drift"
            if (initial > 0 and final < 0) or (initial < 0 and final > 0):
                outcome = "**FLIPPED**"
                
            f.write(f"| **{r.get('config_name', 'Unknown')}** | {initial:.4f} | {final:.4f} | {change:+.4f} | {outcome} |\n")
            
        f.write("\n## 2. Visualizations\n")
        # Check for plot files
        split_plot_filenames = [
            ("CoreBot Trajectories", "stance_trajectories_corebot.png"),
            ("OpenClaw Trajectories", "stance_trajectories_openclaw.png"),
        ]
        split_found = False
        for label, plot_filename in split_plot_filenames:
            plot_path = os.path.join(log_dir, plot_filename)
            if os.path.exists(plot_path):
                abs_plot_path = os.path.abspath(plot_path)
                f.write(f"### {label}\n")
                f.write(f"![{label}](file://{abs_plot_path})\n\n")
                split_found = True

        if not split_found:
            plot_filename = "stance_trajectories.png"
            if os.path.exists(os.path.join(log_dir, plot_filename)):
                abs_plot_path = os.path.abspath(os.path.join(log_dir, plot_filename))
                f.write(f"![Stance Trajectories](file://{abs_plot_path})\n")
            else:
                f.write("*No trajectory plot found.*\n")
            
        f.write("\n## 3. Configuration Details\n")
        f.write("### Variations\n")
        
        for r in results:
            if "error" in r: continue
            name = r.get("config_name", "Default")
            f.write(f"#### {name}\n")
            f.write("```yaml\n")
            for k, v in r.items():
                if k.startswith("var_"):
                     f.write(f"{k[4:]}: {v}\n")
            f.write("```\n")

if __name__ == "__main__":
    main()
