"""Run all offline (no-GPU, no-corpus) tests for the concept_rebalancing package.

    python concept_rebalancing/tests/run_all.py
"""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

TESTS = [
    "test_multiplicity.py",
    "test_schedule.py",
    "test_pipeline_offline.py",
]


def main():
    failed = []
    for t in TESTS:
        print("\n" + "#" * 66 + f"\n# {t}\n" + "#" * 66)
        r = subprocess.run([sys.executable, os.path.join(_HERE, t)])
        if r.returncode != 0:
            failed.append(t)
    print("\n" + "=" * 66)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
