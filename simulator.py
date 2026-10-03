"""
Appends synthetic sensor rows to a log file at the given sampling rate, in the
same 2-header-row / tab-delimited format the app expects. Use this to test
app.py before wiring it to the real acquisition process.

Run:
    python simulate_sensor.py --out /path/to/data.txt --rate 1000 --seconds 1200
"""

import argparse
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="sim_data.txt")
parser.add_argument("--rate", type=float, default=1000.0, help="Samples per second")
parser.add_argument("--seconds", type=float, default=None, help="Stop after N seconds (default: run forever)")
parser.add_argument("--flush-every", type=float, default=0.5, help="Seconds of data to buffer before writing")
args = parser.parse_args()

with open(args.out, "w") as f:
    f.write("Header line 1 (placeholder)\n")
    f.write("Header line 2 (placeholder)\n")

t = 0.0
dt = 1.0 / args.rate
rows_per_flush = max(1, int(args.rate * args.flush_every))
rng = np.random.default_rng(0)

print(f"Writing to {args.out} at {args.rate} Hz (Ctrl+C to stop)...")
try:
    while args.seconds is None or t < args.seconds:
        lines = []
        for _ in range(rows_per_flush):
            speed = 1500 + 5 * np.sin(2 * np.pi * 0.05 * t) + rng.normal(0, 1)
            power = 60 + 3 * np.sin(2 * np.pi * 125 * t) + rng.normal(0, 0.5)  # ~125 Hz component
            freq = 50 + rng.normal(0, 0.2)
            voltage = 230 + rng.normal(0, 2)
            current1 = 10 + rng.normal(0, 0.3)
            current2 = 10 + rng.normal(0, 0.3)
            lines.append(f"{t:.3f}\t{speed:.3f}\t{power:.3f}\t{freq:.3f}\t{voltage:.3f}\t"
                         f"{current1:.3f}\t{current2:.3f}\t0\t0")
            t += dt
        with open(args.out, "a") as f:
            f.write("\n".join(lines) + "\n")
        time.sleep(args.flush_every)
except KeyboardInterrupt:
    print("Stopped.")