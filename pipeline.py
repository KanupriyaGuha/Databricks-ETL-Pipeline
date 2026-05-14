"""
pipeline.py
Pipeline Orchestrator — runs Bronze → Silver → Gold in sequence.
Saves a run log to pipeline_runs.json for the dashboard to display.
"""

import json
import time
import os
from datetime import datetime

from generate_data import save_all
from bronze import run_bronze
from silver import run_silver
from gold import run_gold


LOG_FILE = "pipeline_runs.json"


def load_runs() -> list:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return []


def save_run(run: dict):
    runs = load_runs()
    runs.append(run)
    with open(LOG_FILE, "w") as f:
        json.dump(runs, f, indent=2)


def run_pipeline(generate_fresh: bool = True, data_dir: str = "data", delta_base: str = "delta"):
    run_id    = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_start = time.time()

    print("\n" + "█" * 55)
    print(f"  PIPELINE START  |  {run_id}")
    print("█" * 55)

    run_log = {
        "run_id":     run_id,
        "started_at": datetime.now().isoformat(),
        "status":     "running",
        "stages":     [],
    }

    try:
        # ── Step 0: Generate Data ──────────────────────────────
        if generate_fresh:
            print("\n[Pipeline] Step 0 — Generating synthetic data...")
            t0 = time.time()
            customers, products, transactions, returns = save_all(data_dir)
            gen_time = round(time.time() - t0, 2)
            run_log["data_generation"] = {
                "customers":    len(customers),
                "products":     len(products),
                "transactions": len(transactions),
                "returns":      len(returns),
                "total_rows":   len(customers) + len(products) + len(transactions) + len(returns),
                "elapsed_s":    gen_time,
            }

        # ── Step 1: Bronze ─────────────────────────────────────
        bronze_stats = run_bronze(data_dir, f"{delta_base}/bronze")
        run_log["stages"].append({"layer": "bronze", "tables": bronze_stats})

        # ── Step 2: Silver ─────────────────────────────────────
        silver_stats = run_silver(delta_base)
        run_log["stages"].append({"layer": "silver", "tables": silver_stats})

        # ── Step 3: Gold ───────────────────────────────────────
        gold_stats = run_gold(delta_base)
        run_log["stages"].append({"layer": "gold", "tables": gold_stats})

        # ── Summary ────────────────────────────────────────────
        total_time = round(time.time() - run_start, 2)
        run_log["status"]       = "success"
        run_log["completed_at"] = datetime.now().isoformat()
        run_log["total_elapsed_s"] = total_time

        print("\n" + "█" * 55)
        print(f"  PIPELINE COMPLETE ✅  |  {total_time}s")
        print("█" * 55)

    except Exception as e:
        run_log["status"] = "failed"
        run_log["error"]  = str(e)
        run_log["completed_at"] = datetime.now().isoformat()
        print(f"\n❌ Pipeline failed: {e}")
        raise

    finally:
        save_run(run_log)
        print(f"\n  Run log saved → {LOG_FILE}")

    return run_log


if __name__ == "__main__":
    run_pipeline(generate_fresh=True)
