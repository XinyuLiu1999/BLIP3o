#!/usr/bin/env python3
"""
Run all data pipeline tests and report results.

Usage:
    python scripts/data_pipeline/tests/run_all_tests.py
    python scripts/data_pipeline/tests/run_all_tests.py --verbose
    python scripts/data_pipeline/tests/run_all_tests.py --filter index  # run only test files matching 'index'
"""

import argparse
import subprocess
import sys
import os
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_MODULES = [
    "test_index_tars.py",
    "test_enrich_index.py",
    "test_materialize.py",
    "test_dataset_integration.py",
    "test_determinism.py",
]


def run_test_module(script_path, verbose=False):
    """Run a single test module and return (success, stdout, stderr, elapsed)."""
    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    return result.returncode == 0, result.stdout, result.stderr, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run all data pipeline tests.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full output from each test module.")
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="Only run test files whose name contains this substring.")
    args = parser.parse_args()

    modules = TEST_MODULES
    if args.filter:
        modules = [m for m in modules if args.filter in m]
        if not modules:
            print(f"No test modules match filter '{args.filter}'")
            sys.exit(1)

    print(f"Running {len(modules)} test module(s)...\n")

    results = []
    total_elapsed = 0.0

    for module in modules:
        script_path = os.path.join(TESTS_DIR, module)
        print(f"{'=' * 60}")
        print(f"  {module}")
        print(f"{'=' * 60}")

        success, stdout, stderr, elapsed = run_test_module(script_path, args.verbose)
        total_elapsed += elapsed
        results.append((module, success, elapsed))

        if args.verbose or not success:
            if stdout.strip():
                print(stdout)
            if stderr.strip():
                print(stderr, file=sys.stderr)
        else:
            # In non-verbose mode, show just the summary line from each module
            for line in stdout.strip().splitlines():
                if line.startswith("Results:") or line.startswith("[FAIL]"):
                    print(f"  {line}")

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {elapsed:.1f}s\n")

    # Final summary
    passed = sum(1 for _, s, _ in results if s)
    failed = sum(1 for _, s, _ in results if not s)

    print("=" * 60)
    print(f"  SUMMARY: {passed}/{len(results)} modules passed  ({total_elapsed:.1f}s)")
    print("=" * 60)

    if failed:
        print("\nFailed modules:")
        for module, success, _ in results:
            if not success:
                print(f"  - {module}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    main()
