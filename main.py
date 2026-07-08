"""
main.py — Entry point for the benchmark (primary prompt format).

Uses the same prompt structure as Ramaswamy et al. (Nature Medicine 2026),
so open-source models receive identical input to ChatGPT Health.
Output directory: benchmark/output/

Usage (from project root):
  python benchmark/main.py                    # full run
  python benchmark/main.py --phase predict    # predictions only
  python benchmark/main.py --phase judge      # hallucination checks only
  python benchmark/main.py --from-checkpoint  # evaluate existing checkpoint
  python benchmark/main.py --compile-only     # build Excel results only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.triage_system       import OllamaClient, PREDICTOR_MODELS, LLAMA_MODEL, OLLAMA_BASE_URL
from src.experiment_runner   import run_experiment, load_checkpoint
from src.evaluation_pipeline import compute_metrics, save_metrics, print_summary

DATA_V2    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vignettes.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def ensure_dirs():
    for sub in ["results", "reports", "visualizations"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def check_models():
    print("Checking model availability...")
    all_ok = True
    for model in PREDICTOR_MODELS + [LLAMA_MODEL]:
        client = OllamaClient(model=model)
        ok     = client.is_available()
        print(f"  [{'OK ' if ok else 'MISSING'}] {model}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("\n  One or more models missing. Run: ollama pull <model_name>")
        sys.exit(1)
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark: multi-model AI health triage (Ramaswamy prompt format)")
    p.add_argument("--data",            default=DATA_V2)
    p.add_argument("--output",          default=OUTPUT_DIR)
    p.add_argument("--ollama-url",      default=OLLAMA_BASE_URL)
    p.add_argument("--phase",           default="both", choices=["predict", "judge", "both"])
    p.add_argument("--from-checkpoint", action="store_true")
    p.add_argument("--compile-only",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    print("=" * 70)
    print("Multi-Model AI Health Triage Benchmark (Ramaswamy prompt format)")
    print("=" * 70)
    print(f"  Data    : {args.data}")
    print(f"  Output  : {args.output}")
    print(f"  Models  : {PREDICTOR_MODELS} + ChatGPT Health (pre-computed from Ramaswamy et al.)")
    print(f"  Judge   : {LLAMA_MODEL}")
    print(f"  Phase   : {args.phase}")
    print()

    if args.compile_only or args.from_checkpoint:
        print("Loading existing checkpoint...")
        results = list(load_checkpoint(args.output).values())
    else:
        check_models()
        results = run_experiment(
            data_path        = args.data,
            output_dir       = args.output,
            predictor_models = PREDICTOR_MODELS,
            ollama_url       = args.ollama_url,
            llama_url        = args.ollama_url,
            phase            = args.phase,
        )

    if results:
        print(f"\nComputing metrics over {len(results)} records...")
        metrics = compute_metrics(results)
        path    = save_metrics(metrics, args.output)
        print(f"  Metrics: {path}")
        print_summary(metrics)

    # Compile Excel
    ckpt_path = os.path.join(args.output, "results", "checkpoint.json")
    if os.path.exists(ckpt_path):
        print("\nCompiling Excel workbook...")
        compile_dir = os.path.dirname(os.path.abspath(__file__))
        spec_path   = os.path.join(compile_dir, "compile_results.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("compile_results", spec_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.compile_results()


if __name__ == "__main__":
    main()
