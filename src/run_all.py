import subprocess
import run_s
import sys


for file in run_s.files:
    print("\n" + "=" * 80)
    print(f"Running {file}")
    print("=" * 80)
    subprocess.check_call([sys.executable, file])
