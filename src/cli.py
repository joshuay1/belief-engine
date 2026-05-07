"""
Unified CLI for review-artifact experiments.

Usage:
    agentmem debate   --config path/to/config.yaml
    agentmem sweep    --config path/to/sweep.yaml --trials 5
    agentmem compare  path/to/campaign.yaml
    agentmem --help
"""

import argparse
import logging
import os
import sys


def setup_environment():
    """
    One-time environment setup. Called before any subcommand.

    - Loads .env file (API keys)
    - Suppresses noisy library loggers
    - Sets TOKENIZERS_PARALLELISM to avoid warnings
    """
    from dotenv import load_dotenv
    load_dotenv()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Configure root logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Silence noisy libraries
    for name in [
        "httpx",
        "chromadb",
        "chromadb.telemetry.posthog",
        "posthog",
        "sentence_transformers",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def main():
    parser = argparse.ArgumentParser(
        prog="agentmem",
        description="Memory architectures and identity stability in LLM agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  agentmem debate  --config config/defaults.yaml
  agentmem sweep   --config my_sweep.yaml --trials 5
  agentmem compare my_campaign.yaml
  agentmem compare my_campaign.yaml --resume experiments/logs/campaign_20260324
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Experiment type to run")

    # --- debate (Single Debate) ---
    debate_parser = subparsers.add_parser(
        "debate",
        help="Run a single debate between agents",
        description="One debate, one config. Best for exploration, debugging, and verifying a configuration.",
    )
    debate_parser.add_argument(
        "--config", default="config/defaults.yaml",
        help="Path to YAML config (default: config/defaults.yaml)",
    )
    debate_parser.add_argument(
        "--log-dir",
        help="Override the output directory (default: experiments/logs/<config_name>/<timestamp>)",
    )

    # --- sweep (Parameter Sweep) ---
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run a parameter sweep with multiple trials",
        description="Vary parameters across multiple trials. Produces CSV data for statistical analysis.",
    )
    sweep_parser.add_argument(
        "--config", required=True,
        help="Path to sweep definition YAML",
    )
    sweep_parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of trials per variation (default: 1)",
    )
    sweep_parser.add_argument(
        "--name",
        help="Custom name/tag for this run (default: config name + timestamp)",
    )
    sweep_parser.add_argument(
        "--resume",
        help="Resume a previous run by providing the run_id. Skips completed trials.",
    )

    # --- compare (Agent Comparison / Campaign) ---
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run an agent comparison campaign across topics and conditions",
        description="Batch utility — runs many Single Debates across topics × conditions × trials.",
    )
    compare_parser.add_argument(
        "config",
        help="Path to campaign config YAML",
    )
    compare_parser.add_argument(
        "--resume", metavar="LOG_DIR",
        help="Resume a previous campaign. Skips trials where results.json already exists.",
    )

    # --- plot ---
    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate stylized plots from experiment logs",
        description="Generates static plots like trajectory or benchmark charts from a results.csv file.",
    )
    plot_parser.add_argument("log_dir", help="Path to the experiment log directory containing results.csv")
    plot_parser.add_argument(
        "--type", choices=["trajectory", "benchmark"], default="trajectory",
        help="Type of plot to generate (default: trajectory)"
    )
    plot_parser.add_argument("--title", help="Optional title override for the plot")

    # --- live-plot ---
    live_plot_parser = subparsers.add_parser(
        "live-plot",
        help="Watch an active experiment and update a plot in real-time",
        description="Reads results.csv periodically and updates a matplotlib window without blocking.",
    )
    live_plot_parser.add_argument("log_dir", help="Path to the active experiment log directory")
    live_plot_parser.add_argument(
        "--type", choices=["trajectory"], default="trajectory",
        help="Type of plot to trace live"
    )

    # --- Parse ---
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize environment once
    setup_environment()

    # --- Dispatch ---
    if args.command == "debate":
        from src.simulation.run_simulation import run_simulation
        run_simulation(args.config, log_dir_override=args.log_dir)

    elif args.command == "sweep":
        # Re-inject args for run_experiment's main() which uses argparse internally
        # We override sys.argv so it picks up our parsed args
        sweep_args = ["run_experiment", "--config", args.config, "--trials", str(args.trials)]
        if args.name:
            sweep_args.extend(["--name", args.name])
        if args.resume:
            sweep_args.extend(["--resume", args.resume])
        sys.argv = sweep_args
        from src.experiment.run_experiment import main as sweep_main
        sweep_main()

    elif args.command == "compare":
        from scripts.runners.run_campaign import run_campaign
        run_campaign(args.config, resume_dir=args.resume)

    elif args.command == "plot":
        csv_path = os.path.join(args.log_dir, "results.csv")
        if not os.path.exists(csv_path) and os.path.isfile(args.log_dir):
            csv_path = args.log_dir
            
        if args.type == "trajectory":
            from src.utils.plot_history import plot_stance_history
            plot_stance_history(csv_path)
        elif args.type == "benchmark":
            from src.utils.plot_benchmark_csv import plot_benchmark
            plot_benchmark(csv_path, title=args.title)
            
    elif args.command == "live-plot":
        from src.utils.live_plotter import LivePlotter
        plotter = LivePlotter(args.log_dir, plot_type=args.type)
        plotter.start()


if __name__ == "__main__":
    main()
