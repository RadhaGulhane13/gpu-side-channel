import subprocess
import csv
import time

FIELDS = [
    "timestamp",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "utilization.gpu",
    "utilization.memory"
]

cmd = [
    "nvidia-smi",
    f"--query-gpu={','.join(FIELDS)}",
    "--format=csv,noheader,nounits",
    "-lms", "50"   # ~20 Hz
]

with open("gpu_trace.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(FIELDS)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    start = time.time()
    while time.time() - start < 15:  # seconds
        line = proc.stdout.readline().strip()
        if line:
            writer.writerow(line.split(", "))

    proc.terminate()

