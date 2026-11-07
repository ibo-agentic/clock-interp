"""EXPLORATORY control-adjusted E3d comparison (does NOT replace the pre-registered verdict in e3d_prereg_verdict.json).
For one model: per-trial paired contrasts, cluster bootstrap over the (N,H) cell (unit of analysis), 10000 resamples, seed 0:
   adjusted effect = (H_all blocked) - (random_all blocked), for accuracy and for leak rate;  plus each effect vs intact.
usage: python analysis/e3d_control_adjusted.py [model ...]   (default: internvl3-2b)  -> results/mech_e3/<model>/e3d_control_adjusted.csv"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import mech_lib as ML

def run(model):
    d = pd.read_csv(os.path.join("results", "mech_e3", model, "e3d_knockout.csv"))
    d["leak"] = (d.label == "leak").astype(float); d["acc"] = d.minute_correct.astype(float)
    key = ["lead", "N", "H"]
    w = {c: d[d.cell == c].set_index(key) for c in ("intact", "H_all", "random_all")}
    idx = w["intact"].index
    assert all(len(w[c]) == len(idx) for c in w), "cells differ in trial count"
    cl = np.array([f"{n}_{h}" for (_, n, h) in idx])
    rows = []
    for metric in ("acc", "leak"):
        v = {c: w[c].loc[idx, metric].values for c in w}
        for name, x in (("H_all - intact", v["H_all"] - v["intact"]), ("random_all - intact", v["random_all"] - v["intact"]),
                        ("H_all - random_all  [CONTROL-ADJUSTED]", v["H_all"] - v["random_all"])):
            m, lo, hi = ML.cluster_bootstrap_ci(x, cl, n_boot=10000, seed=0)
            rows.append({"model": model, "metric": metric, "contrast": name, "n_trials": len(idx), "mean_diff_points": 100 * m,
                         "ci_lo_points": 100 * lo, "ci_hi_points": 100 * hi, "sign_flip_p": ML.sign_flip_p(x, cl)})
    t = pd.DataFrame(rows)
    leak_i, leak_h, leak_r = (w[c].loc[idx, "leak"].sum() for c in ("intact", "H_all", "random_all"))
    t.attrs["leaks"] = (int(leak_i), int(leak_h), int(leak_r))
    out = os.path.join("results", "mech_e3", model, "e3d_control_adjusted.csv")
    t.to_csv(out, index=False)
    print(f"\n{model}  EXPLORATORY control-adjusted E3d  (leaks: intact {int(leak_i)}, H-blocked {int(leak_h)}, random-span {int(leak_r)})")
    print(t.round(3).to_string(index=False)); print("wrote", out)

if __name__ == "__main__":
    for m in (sys.argv[1:] or ["internvl3-2b"]):
        run(m)
