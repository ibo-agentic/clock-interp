"""Regression check for the additive --style / --conditions options of conversion_test.py (PREREG_MECHANISM.md ninth entry, item 8).

Runs the smoke design (5 prompts x 4 conditions) with the OLD conversion_test.py (frozen copy tests/fixtures/conversion_test_before_style.py, i.e. the file as it was
before the options existed) and the NEW one on the same tiny random Gemma3 (CPU), in separate empty folders, and requires byte-identical output: the results CSV, the image
manifest, every rendered PNG, and the run_config.json (minus the date). Then checks the new options themselves: styled images differ from the default ones and from each
other, styled runs use their own file/folder names, a --conditions subset reproduces the full run's rows for those conditions, and unknown styles/conditions are rejected.
usage: python tests/test_conversion_style_regression.py"""
import filecmp
import importlib.util
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

import adapters  # noqa: E402
import test_gemma_adapter_offline as T  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")

    old = load_module(os.path.join(ROOT, "tests", "fixtures", "conversion_test_before_style.py"), "conversion_test_old")
    new = load_module(os.path.join(ROOT, "conversion_test.py"), "conversion_test_new")
    with tempfile.TemporaryDirectory() as tmp:
        model, proc = T.build(tmp)
        cls = adapters.Gemma3Adapter
        saved_load, saved_exp = cls.load, cls.expected_image_tokens

        def fake_load(self, model_id=None):
            self.model, self.processor = model, proc
            self.torch_dtype, self.dtype_name = adapters._gemma_dtype()
            return model, proc
        cls.load, cls.expected_image_tokens = fake_load, T.N_TOK
        cwd = os.getcwd()
        os.environ.pop("CLOCK_DTYPE", None)
        try:
            def run_in(sub, mod, **kw):
                d = os.path.join(tmp, sub)
                os.makedirs(d)
                os.chdir(d)
                try:
                    return mod.run(adapter_name="gemma-3-4b-it", smoke=True, results_dir="results", **kw), d
                finally:
                    os.chdir(cwd)

            _, d_old = run_in("old", old)
            _, d_new = run_in("new", new)
            f = "results/gemma-3-4b-it/conversion_test_smoke.csv"
            check("default run: results CSV byte-identical old vs new", filecmp.cmp(os.path.join(d_old, f), os.path.join(d_new, f), shallow=False))
            check("default run: image manifest byte-identical", filecmp.cmp(os.path.join(d_old, "conversion_images_smoke", "manifest.csv"),
                                                                            os.path.join(d_new, "conversion_images_smoke", "manifest.csv"), shallow=False))
            pngs_old = sorted(x for x in os.listdir(os.path.join(d_old, "conversion_images_smoke")) if x.endswith(".png"))
            pngs_new = sorted(x for x in os.listdir(os.path.join(d_new, "conversion_images_smoke")) if x.endswith(".png"))
            same_png = pngs_old == pngs_new and all(filecmp.cmp(os.path.join(d_old, "conversion_images_smoke", x), os.path.join(d_new, "conversion_images_smoke", x), shallow=False) for x in pngs_old)
            check("default run: every rendered PNG byte-identical", same_png, f"({len(pngs_old)} images)")
            cfg = "results/gemma-3-4b-it/conversion_test_run_config_smoke.json"
            jo, jn = json.load(open(os.path.join(d_old, cfg))), json.load(open(os.path.join(d_new, cfg)))
            for j in (jo, jn):
                j.pop("date_utc", None)
            check("default run: run_config identical (minus date)", jo == jn, "" if jo == jn else str({k: (jo.get(k), jn.get(k)) for k in set(jo) | set(jn) if jo.get(k) != jn.get(k)})[:300])
            check("default run: no style/conditions keys added to the config", "style" not in jn and "conditions_run" not in jn)

            # new options
            df_full = pd.read_csv(os.path.join(d_new, f))
            for st in ("s1_thin", "s2_ticks"):
                _, d_st = run_in("st_" + st, new, style=st, conditions=["matching_clock", "mismatched_clock"])
                fs = f"results/gemma-3-4b-it/conversion_test_smoke_{st}.csv"
                check(f"{st}: own CSV name and own image folder", os.path.exists(os.path.join(d_st, fs)) and os.path.isdir(os.path.join(d_st, f"conversion_images_smoke_{st}")))
                dfs = pd.read_csv(os.path.join(d_st, fs))
                check(f"{st}: only the requested conditions", set(dfs.condition) == {"matching_clock", "mismatched_clock"}, str(sorted(set(dfs.condition))))
                m_def = os.path.join(d_new, "conversion_images_smoke", "match_N01_H01.png")
                m_sty = os.path.join(d_st, f"conversion_images_smoke_{st}", "match_N01_H01.png")
                check(f"{st}: styled image differs from the default one", os.path.exists(m_sty) and not filecmp.cmp(m_def, m_sty, shallow=False))
                # styled images of the same keys keep the same hands: compare hand pixels via an independent render with the style kwargs
                cj = json.load(open(os.path.join(d_st, f"results/gemma-3-4b-it/conversion_test_run_config_smoke_{st}.json")))
                check(f"{st}: config records style and conditions", cj.get("style", {}).get("name") == st and cj.get("conditions_run") == ["matching_clock", "mismatched_clock"])
                key = ["condition", "position", "hour", "lead_in_id"]
                mm = dfs.merge(df_full, on=key, suffixes=("_s", "_d"))
                check(f"{st}: prompts and image keys identical to the default run", len(mm) == len(dfs) and (mm.prompt_s == mm.prompt_d).all()
                      and (mm.image_hour_s.fillna(-1) == mm.image_hour_d.fillna(-1)).all() and (mm.image_minute_s.fillna(-1) == mm.image_minute_d.fillna(-1)).all())
                if st == "s1_thin":
                    s1_img = m_sty
                else:
                    check("S1 and S2 images differ from each other", not filecmp.cmp(s1_img, m_sty, shallow=False))
            # a conditions-only subset with the DEFAULT style reproduces the full run's rows
            _, d_sub = run_in("sub", new, conditions=["blank", "matching_clock"])
            sub_csv = [x for x in os.listdir(os.path.join(d_sub, "results/gemma-3-4b-it")) if x.startswith("conversion_test_smoke_cond-")]
            check("default-style subset gets its own CSV name", len(sub_csv) == 1, str(sub_csv))
            dsub = pd.read_csv(os.path.join(d_sub, "results/gemma-3-4b-it", sub_csv[0]))
            mm = dsub.merge(df_full, on=["condition", "position", "hour", "lead_in_id"], suffixes=("_s", "_d"))
            check("default-style subset: raw outputs equal the full run's for the same trials", len(mm) == len(dsub) and (mm.raw_output_s.astype(str) == mm.raw_output_d.astype(str)).all())
            for bad_kw, what in ((dict(style="nope"), "unknown style"), (dict(conditions=["nope"]), "unknown condition")):
                try:
                    run_in("bad_" + what.replace(" ", "_"), new, **bad_kw)
                    check(f"{what} is rejected", False)
                except ValueError:
                    check(f"{what} is rejected", True)
        finally:
            os.chdir(cwd)
            cls.load, cls.expected_image_tokens = saved_load, saved_exp
    return ok


if __name__ == "__main__":
    r = main()
    print("\nALL PASS" if r else "\nSOME FAILED")
    sys.exit(0 if r else 1)
