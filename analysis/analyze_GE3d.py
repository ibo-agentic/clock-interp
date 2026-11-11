"""GE3d (Gemma-3-4B-it, E3d minimal run 2): pre-registered rule, control-adjusted contrasts (all keys and intact-parsed keys), side-by-side with Qwen-3B and InternVL3-2B. CPU only.
usage: python analysis/analyze_GE3d.py -> results/gemma/GE3d_report.txt"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd  # noqa: E402
import mech_lib as ML  # noqa: E402

out = []
P = lambda s="": (print(s), out.append(s))
KEY = ["lead", "N", "H"]
MODELS = ["qwen2.5-vl-3b", "internvl3-2b", "gemma-3-4b-it"]


def load(m):
    d = pd.read_csv(os.path.join(ROOT, "results/mech_e3", m, "e3d_knockout.csv"))
    d["leak"] = (d.label == "leak").astype(float)
    d["acc"] = d.minute_correct.astype(float)
    return d


def contrasts(d, keys=None):
    w = {c: d[d.cell == c].set_index(KEY) for c in ("intact", "H_all", "random_all")}
    idx = w["intact"].index
    if keys is not None:
        idx = idx[idx.isin(keys)]
    cl = np.array([f"{n}_{h}" for (_, n, h) in idx])
    r = {"n": len(idx)}
    for met in ("acc", "leak"):
        v = {c: w[c].loc[idx, met].values for c in w}
        for nm, x in (("H", v["H_all"] - v["intact"]), ("R", v["random_all"] - v["intact"]), ("adj", v["H_all"] - v["random_all"])):
            m, lo, hi = ML.cluster_bootstrap_ci(x, cl, n_boot=10000, seed=0)
            r[f"{met}_{nm}"] = (100 * m, 100 * lo, 100 * hi)
        r[f"{met}_n"] = {c: int(v[c].sum()) for c in v}
    return r


def rule(r):
    li, lh = r["leak_n"]["intact"], r["leak_n"]["H_all"]
    drop = 1 - lh / li if li else float("nan")
    a, lo, hi = r["acc_H"]
    ok_leak = drop >= 0.30
    ok_acc = a >= 4 and lo > 0
    ok_ctrl = abs(r["acc_R"][0]) <= 2
    return drop, ok_leak, ok_acc, ok_ctrl, (ok_leak and ok_acc and ok_ctrl)


P("=== GE3d Gemma-4B: files ===")
st = json.load(open(os.path.join(ROOT, "results/gemma/GE3d_status.json")))
for s in st:
    P(f"  {s['step']:<22} {s['status']:<4} {s['minutes']:>6.1f} min")
P(f"  total {sum(s['minutes'] for s in st) / 60:.2f} h (guess 1.9 h)")
v = json.load(open(os.path.join(ROOT, "results/mech_e3/gemma-3-4b-it/verify.json")))
P(f"  verify pass = {v['pass']}; amended gates: " + "; ".join(f"{k.split('_')[0]} {c['agreement'] if 'agreement' in c else c['activated']} usable {c['usable']}/{c['n']}" for k, c in v["checks"].items() if k.startswith(("G8_", "G8b", "G9"))))

g = load("gemma-3-4b-it")
P("\n=== Gemma-4B label counts per cell (432 trials; all keys) ===")
P(g.groupby("cell").label.value_counts().unstack(fill_value=0).to_string())
intact = g[g.cell == "intact"].set_index(KEY)
par = intact.index[(intact.label != "unparsed").values]
P(f"\nintact-parsed keys: {len(par)} of {len(intact)} (unparsed intact replies: {len(intact) - len(par)})")
for c in ("H_all", "random_all"):
    d = g[g.cell == c].set_index(KEY).loc[intact.index]
    P(f"  {c}: unparsed {int((d.label == 'unparsed').sum())} of 432 (on the 48 intact-unparsed keys: {int((d.loc[intact.index.difference(par)].label == 'unparsed').sum())})")

P("\n=== Pre-registered E3d rule, exactly as pre-registered (leaks down >= 30%, accuracy up >= 4 points with CI excluding 0, random-span |acc change| <= 2 points) ===")
rows, res = [], {}
for m in MODELS:
    d = load(m)
    r = contrasts(d)
    res[m] = r
    drop, okl, oka, okc, ok = rule(r)
    ver = json.load(open(os.path.join(ROOT, "results/mech_e3", m, "e3d_prereg_verdict.json")))
    rows.append({"model": m, "n": r["n"], "leaks intact": r["leak_n"]["intact"], "leaks H-blocked": r["leak_n"]["H_all"], "leaks random": r["leak_n"]["random_all"], "leak drop": f"{drop:.1%}",
                 "acc gain pts [CI]": f"{r['acc_H'][0]:+.1f} [{r['acc_H'][1]:+.1f},{r['acc_H'][2]:+.1f}]", "random acc pts": f"{r['acc_R'][0]:+.1f}",
                 "control-adj acc pts [CI]": f"{r['acc_adj'][0]:+.1f} [{r['acc_adj'][1]:+.1f},{r['acc_adj'][2]:+.1f}]",
                 "control-adj leak pts [CI]": f"{r['leak_adj'][0]:+.1f} [{r['leak_adj'][1]:+.1f},{r['leak_adj'][2]:+.1f}]",
                 "leak>=30%": okl, "acc>=4,CI>0": oka, "ctrl<=2": okc, "rule met (mine)": ok, "M_bind_supported (json)": ver["M_bind_supported"]})
    assert ok == ver["M_bind_supported"], (m, ok, ver)
P(pd.DataFrame(rows).T.to_string(header=False))

P("\n=== Gemma-4B, PRIMARY = keys whose intact reply parsed (n=%d); SENSITIVITY = all keys above ===" % len(par))
rp = contrasts(g, keys=par)
drop, okl, oka, okc, ok = rule(rp)
ver = json.load(open(os.path.join(ROOT, "results/mech_e3/gemma-3-4b-it/e3d_prereg_verdict_intactparsed.json")))
P(f"  leaks intact {rp['leak_n']['intact']}, H-blocked {rp['leak_n']['H_all']}, random {rp['leak_n']['random_all']}; leak drop {drop:.1%}")
P(f"  accuracy: intact {100 * g[(g.cell == 'intact')].set_index(KEY).loc[par].acc.mean():.1f}% -> H-blocked {100 * g[g.cell == 'H_all'].set_index(KEY).loc[par].acc.mean():.1f}%")
P(f"  acc gain {rp['acc_H'][0]:+.1f} [{rp['acc_H'][1]:+.1f},{rp['acc_H'][2]:+.1f}] pts; random-span {rp['acc_R'][0]:+.1f}; control-adjusted acc {rp['acc_adj'][0]:+.1f} [{rp['acc_adj'][1]:+.1f},{rp['acc_adj'][2]:+.1f}]; control-adjusted leak {rp['leak_adj'][0]:+.1f} [{rp['leak_adj'][1]:+.1f},{rp['leak_adj'][2]:+.1f}]")
P(f"  rule: leak>=30% {okl}, acc>=4 & CI>0 {oka}, random within 2 pts {okc} -> met {ok}; json verdict M_bind_supported={ver['M_bind_supported']}")
assert ok == ver["M_bind_supported"]
open(os.path.join(ROOT, "results/gemma/GE3d_report.txt"), "w", encoding="utf8").write("\n".join(out) + "\n")
