"""
Bangla Distill — Master Benchmarking Orchestrator
Executes all benchmark suites sequentially and generates individual PDF reports for each.
Suites:
  - Suite 1: Quantitative Held-Out NLG & Perplexity Benchmark
  - Suite 2: Domain-Specific Qualitative Empirical Benchmark
  - Suite 3: System Performance & Edge Hardware Efficiency
  - Suite 4: Robust Academic Bengali Benchmark (BnMMLU, BLUCK, BanglaMATH, Belebele/QuAD)
"""

import os
import sys
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import run_benchmark_suite1
import run_benchmark_suite2
import run_benchmark_suite3
import run_benchmark_suite4_academic

def main():
    parser = argparse.ArgumentParser(description="Run Bangla Distill Benchmark Suites")
    parser.add_argument("--suite", type=int, choices=[1, 2, 3, 4], default=None, help="Run specific suite (1, 2, 3, or 4). Default: run all.")
    args = parser.parse_args()

    reports = []
    
    if args.suite is None or args.suite == 1:
        print("\n" + "=" * 75)
        print("🚀 STARTING SUITE 1: QUANTITATIVE HELD-OUT NLG BENCHMARK")
        print("=" * 75)
        pdf1 = run_benchmark_suite1.run_suite()
        reports.append(pdf1)

    if args.suite is None or args.suite == 2:
        print("\n" + "=" * 75)
        print("🚀 STARTING SUITE 2: DOMAIN-SPECIFIC EMPIRICAL BENCHMARK")
        print("=" * 75)
        pdf2 = run_benchmark_suite2.run_suite()
        reports.append(pdf2)

    if args.suite is None or args.suite == 3:
        print("\n" + "=" * 75)
        print("🚀 STARTING SUITE 3: SYSTEM EFFICIENCY & HARDWARE BENCHMARK")
        print("=" * 75)
        pdf3 = run_benchmark_suite3.profile_efficiency()
        reports.append(pdf3)

    if args.suite is None or args.suite == 4:
        print("\n" + "=" * 75)
        print("🚀 STARTING SUITE 4: ROBUST ACADEMIC BENGALI BENCHMARK (BnMMLU, BLUCK, BanglaMATH)")
        print("=" * 75)
        pdf4 = run_benchmark_suite4_academic.run_suite()
        reports.append(pdf4)

    print("\n" + "=" * 75)
    print("🏆 ALL BENCHMARK SUITES COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    for idx, rep in enumerate(reports):
        print(f"  [{idx+1}] {rep}")
    print("=" * 75)

if __name__ == "__main__":
    main()
