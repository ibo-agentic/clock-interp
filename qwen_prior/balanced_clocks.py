"""
balanced_clocks.py -- Balanced dataset for the probing experiments.

The 500-clock run had uneven counts per hand position (n=2 for some, n=22 for
others). Here we generate an equal number of clocks for every minute-hand
position (the 12 numbers) crossed with hours, so every cell of the
conversion table has real weight.
"""
import os
import pandas as pd
import numpy as np
from clocks import generate_clock_image

def generate_balanced(out_dir="data_balanced", per_position=40, seed=7):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    rows, i = [], 0

    # minute-hand position 1..12 means minutes 5,10,...,60->0
    for pos in range(1, 13):
        minute = (pos % 12) * 5
        for _ in range(per_position):
            hour = int(rng.randint(1, 13))
            fn = f"bal_{i:04d}.png"
            generate_clock_image(hour, minute, os.path.join(out_dir, fn))
            rows.append({"filename": fn, "hour": hour, "minute": minute,
                         "minute_position": pos})
            i += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "data.csv"), index=False)
    print(f"Wrote {len(df)} clocks to {out_dir}/ ({per_position} per minute position)")
    return df

if __name__ == "__main__":
    generate_balanced()
