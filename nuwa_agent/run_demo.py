"""Quick demo: run NUWA-Agent with a short protein on E. coli.

Pipes pre-set inputs to main.py via subprocess.
"""
import subprocess
import sys
import os

# Inputs (simulate user typing)
inputs = "\n".join([
    "MVSKGEELFTGVVPILVELDGDVNGHKFSVS",  # 30aa EGFP fragment
    "Escherichia coli",                   # host
    "",                                    # skip GC constraints
])

cmd = [
    sys.executable,
    "-c", """
import config
# Reduce params for demo speed
config.NUM_CANDIDATES = 20
config.MAX_ITERATION_ROUNDS = 3
config.MIN_ITERATION_ROUNDS = 2
config.PARETO_TOP_K = 5

import main
main.main()
"""
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
proc = subprocess.run(
    cmd,
    input=inputs,
    text=True,
    capture_output=True,
    timeout=1200,  # 20 min max
    cwd=os.getcwd(),
    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
)

print("=" * 60)
print("STDOUT:")
print(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)
if proc.stderr:
    print("=" * 60)
    print("STDERR:")
    print(proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
print(f"\nExit code: {proc.returncode}")
