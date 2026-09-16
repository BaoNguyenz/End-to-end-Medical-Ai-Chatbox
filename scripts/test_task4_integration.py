"""
Task 4 Smoke Test — Integrated Platform & CLI Verification
==========================================================
Verifies:
    TEST 1: evaluate.py CLI execution (Mock mode) -> outputs JSON and Markdown report
    TEST 2: compare.py CLI execution (Mock mode)  -> outputs comparison report and charts
    TEST 3: main.py master CLI execution (Mock)   -> executes end-to-end platform flow
    TEST 4: Deliverables verification             -> ANSWERS.md, chart PNGs, report.md

Usage:
    python scripts/test_task4_integration.py
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PASS = "[PASS]"
FAIL = "[FAIL]"

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

TEST_RESULTS_DIR = BASE_DIR / "test_results"
TEST_OUTPUT_DIR = BASE_DIR / "test_output"



def clean_test_dirs():
    if TEST_RESULTS_DIR.exists():
        shutil.rmtree(TEST_RESULTS_DIR, ignore_errors=True)
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR, ignore_errors=True)
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_evaluate_cli() -> bool:
    print("\n--- TEST 1: evaluate.py CLI (Mock Mode) ---")
    cmd = [
        PYTHON,
        str(BASE_DIR / "evaluate.py"),
        "--architecture", "naive",
        "--mock",
        "--limit", "2",
        "--output", str(TEST_RESULTS_DIR),
    ]
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"{FAIL} evaluate.py failed with exit code {proc.returncode}")
        print("STDERR:\n", proc.stderr)
        return False

    json_out = TEST_RESULTS_DIR / "naive_results.json"
    md_out = TEST_RESULTS_DIR / "naive_report.md"

    if not json_out.exists() or not md_out.exists():
        print(f"{FAIL} Missing expected output files: {json_out} or {md_out}")
        return False

    print(f"{PASS} evaluate.py generated {json_out.name} ({json_out.stat().st_size} bytes) and {md_out.name}")
    return True


def test_compare_cli() -> bool:
    print("\n--- TEST 2: compare.py CLI (Mock Mode) ---")
    report_file = TEST_OUTPUT_DIR / "comparison_report.md"
    cmd = [
        PYTHON,
        str(BASE_DIR / "compare.py"),
        "--mock",
        "--limit", "2",
        "--output", str(report_file),
        "--charts-dir", str(TEST_OUTPUT_DIR),
    ]
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"{FAIL} compare.py failed with exit code {proc.returncode}")
        print("STDERR:\n", proc.stderr)
        return False

    if not report_file.exists():
        print(f"{FAIL} Missing expected comparison report: {report_file}")
        return False

    print(f"{PASS} compare.py generated report ({report_file.stat().st_size} bytes)")
    return True


def test_main_cli() -> bool:
    print("\n--- TEST 3: main.py Master CLI (Mock Mode) ---")
    report_file = TEST_OUTPUT_DIR / "main_report.md"
    cmd = [
        PYTHON,
        str(BASE_DIR / "main.py"),
        "--mock",
        "--limit", "2",
        "--output", str(report_file),
        "--charts-dir", str(TEST_OUTPUT_DIR),
    ]
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"{FAIL} main.py failed with exit code {proc.returncode}")
        print("STDERR:\n", proc.stderr)
        return False

    if not report_file.exists():
        print(f"{FAIL} Missing expected main report: {report_file}")
        return False

    print(f"{PASS} main.py master CLI executed successfully and generated {report_file.name}")
    return True


def test_deliverables_and_charts() -> bool:
    print("\n--- TEST 4: Charts and Deliverables Verification ---")
    expected_charts = [
        TEST_OUTPUT_DIR / "radar_chart.png",
        TEST_OUTPUT_DIR / "bar_faithfulness.png",
        TEST_OUTPUT_DIR / "comparison_metrics.png",
        TEST_OUTPUT_DIR / "tradeoff_quality_cost_latency.png",
    ]

    all_ok = True
    for chart in expected_charts:
        if chart.exists() and chart.stat().st_size > 500:
            print(f"{PASS} Chart generated: {chart.name} ({chart.stat().st_size} bytes)")
        else:
            print(f"{FAIL} Chart missing or empty: {chart.name}")
            all_ok = False

    answers_md = BASE_DIR / "docs" / "ANSWERS.md"
    if not answers_md.exists():
        answers_md = BASE_DIR / "ANSWERS.md"
    if answers_md.exists() and answers_md.stat().st_size > 1000:
        print(f"{PASS} ANSWERS.md verified at {answers_md.relative_to(BASE_DIR)} ({answers_md.stat().st_size} bytes)")
    else:
        print(f"{FAIL} ANSWERS.md is missing or too short!")
        all_ok = False

    return all_ok


def main():
    print("=" * 65)
    print("  GaleMed AI — Task 4 Integration Smoke Test Suite")
    print("=" * 65)

    clean_test_dirs()

    results = [
        ("TEST 1: evaluate.py CLI", test_evaluate_cli()),
        ("TEST 2: compare.py CLI", test_compare_cli()),
        ("TEST 3: main.py Master CLI", test_main_cli()),
        ("TEST 4: Charts & Deliverables", test_deliverables_and_charts()),
    ]

    print("\n" + "=" * 65)
    print("  TASK 4 TEST RESULTS SUMMARY")
    print("=" * 65)
    all_passed = True
    for label, status in results:
        status_str = PASS if status else FAIL
        print(f"  {status_str}  {label}")
        if not status:
            all_passed = False
    print("=" * 65)

    if all_passed:
        print("\n[SUCCESS] ALL TASK 4 TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n[FAILURE] SOME TESTS FAILED. Please review output above.")
        sys.exit(1)



if __name__ == "__main__":
    main()
