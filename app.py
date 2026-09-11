# ==============================================================================
# APP.PY: THE EXECUTIVE RESUME BULLET & IMPACT OPTIMIZER
# Standalone CLI & Module Entry Point
# ==============================================================================

import sys
from dotenv import load_dotenv
from Day9 import (
    ExecutiveResumeOptimizer,
    ResumeBulletOptimization,
    display_optimization_card,
    setup_git_hygiene,
    run_portfolio_benchmark
)

# Load environment variables cleanly
load_dotenv()

def interactive_cli():
    """
    Provides an interactive terminal interface for students and users to
    optimize and revise their own resume bullets in real time.
    """
    print("\n" + "=" * 80)
    print("💼 WELCOME TO THE EXECUTIVE RESUME BULLET & IMPACT OPTIMIZER")
    print("=" * 80)
    print("Google XYZ Formula: 'Accomplished [X], as measured by [Y], by doing [Z]'\n")

    optimizer = ExecutiveResumeOptimizer()

    user_input = input("Enter a raw/weak resume bullet (or press Enter to run benchmark tests):\n> ").strip()
    if not user_input:
        print("\nNo custom bullet entered. Running benchmark tests...")
        run_portfolio_benchmark()
        return

    # Turn 1: Optimize
    print("\n⏳ Optimizing bullet...")
    result = optimizer.optimize(raw_bullet=user_input)
    display_optimization_card(result, turn_title="Initial Optimization")

    # Multi-turn revision loop
    while True:
        feedback = input("Request revision feedback (e.g. 'Make it more senior', 'Focus on cloud costs') or 'exit':\n> ").strip()
        if not feedback or feedback.lower() in ("exit", "quit", "q"):
            print("\n✅ Session completed. Good luck with your applications!")
            break

        print("\n⏳ Processing revision...")
        result = optimizer.request_revision(user_feedback=feedback)
        display_optimization_card(result, turn_title="Revised Optimization")

if __name__ == "__main__":
    setup_git_hygiene()
    interactive_cli()
