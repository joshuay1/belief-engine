"""
Centralized output module for simulation print statements.
Controls verbosity levels and formats output consistently.
"""
import os

# Global verbosity level (0=minimal, 1=normal, 2=verbose, 3=debug)
_VERBOSITY = int(os.environ.get("SIM_VERBOSITY", 1))

def set_verbosity(level: int):
    """Set global verbosity level."""
    global _VERBOSITY
    _VERBOSITY = level

def get_verbosity() -> int:
    """Get current verbosity level."""
    return _VERBOSITY

# --- Print Helpers ---

def print_header(text: str, char: str = "=", width: int = 50):
    """Print a decorated header (always shown)."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")

def print_round(round_num: int, width: int = 50):
    """Print round header."""
    print(f"\n{'=' * width}")
    print(f"=== Round {round_num} ===")
    print(f"{'=' * width}\n")

def print_agent_message(agent_name: str, message: str, truncate: int = 0):
    """Print agent's message (always shown)."""
    if truncate > 0 and len(message) > truncate:
        message = message[:truncate] + "..."
    print(f"\n[{agent_name}]:\n\"{message}\"\n")

def print_stance_summary(stances: list, round_num: int):
    """Print stance summary for all agents."""
    print(f"\n   === Stance Summary (Round {round_num}) ===")
    for name, stance, change in stances:
        change_str = f"[{change:+.4f}]" if round_num > 0 else "(initial)"
        print(f"   {name}: {stance:+.4f} {change_str}")

def print_system(message: str, level: int = 1):
    """Print system message if verbosity allows."""
    if _VERBOSITY >= level:
        print(f"   (System) {message}")

def print_memory(agent_id: str, message: str, level: int = 1):
    """Print memory-related message if verbosity allows."""
    if _VERBOSITY >= level:
        print(f"   [{agent_id}] (Memory) {message}")

def print_debug(message: str, prefix: str = "DEBUG"):
    """Print debug message (only at verbosity 3)."""
    if _VERBOSITY >= 3:
        print(f"{prefix}: {message}")

def print_info(message: str, level: int = 1):
    """Print info message if verbosity allows."""
    if _VERBOSITY >= level:
        print(f"   {message}")

def print_argument_update(agent_id: str, status: str, claim: str, reason: str = "", level: int = 2):
    """Print argument update notification."""
    if _VERBOSITY >= level:
        claim_preview = claim[:40] + "..." if len(claim) > 40 else claim
        if reason:
            print(f"   [{agent_id}] {status.upper()}: \"{claim_preview}\" ({reason})")
        else:
            print(f"   [{agent_id}] {status.upper()}: \"{claim_preview}\"")

def print_extraction_summary(agent_name: str, total: int, internalized: int):
    """Print extraction summary."""
    if _VERBOSITY >= 1:
        print(f"   (System) Extracted {total} arguments from {agent_name}.")
        if internalized > 0:
            print(f"      + {agent_name} internalized {internalized} new arguments.")

def print_strategy_init(agent_id: str, strategy: str, details: str = ""):
    """Print strategy initialization."""
    if _VERBOSITY >= 1:
        detail_str = f" ({details})" if details else ""
        print(f"   [{agent_id}] (Memory) Using {strategy}{detail_str}")
