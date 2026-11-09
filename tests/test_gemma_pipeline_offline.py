"""Offline run of the behaviour-pipeline scripts (text_only_v2, conversion_test, describe_check, eval_behavior, probe extraction) with the Gemma3 adapter on the TINY RANDOM
Gemma3 from test_gemma_adapter_offline.py (CPU, no download).  Checks that every script runs end to end with the Gemma adapter, writes its files under results/gemma-3-4b-it/,
that CLOCK_DTYPE=float32 gives a separate output path for the cross-check, that the markdown-stripping parse step works on Gemma-style replies, and that probe extraction
(hidden states, vision hook, image-token pooling) produces finite arrays.  It does NOT test real-weight behaviour.
usage: python tests/test_gemma_pipeline_offline.py"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import adapters  # noqa: E402
import test_gemma_adapter_offline as T  # noqa: E402
from textnorm import normalize_reply  # noqa: E402


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")

    # parse step on Gemma-style replies (taken from notebook G0b logs)
    import describe_check as DC
    import eval_behavior as EB
    r1 = ("Here's the breakdown of the clock:\n\n*   **Long minute hand:** The long minute hand is pointing at the number **3**.\n"
          "*   **Short hour hand:** The short hour hand is pointing at the number **9**.\n\nTherefore, the time is **9:30**.")
    check("describe parse on markdown reply", DC.parse_reply(r1) == (3, 9, (9, 30)), str(DC.parse_reply(r1)))
    check("tolerant hour parse on markdown reply", DC.parse_hour_tolerant(r1) == 9)
    check("time parse on **9:30**", EB.parse_time_answer("**9:30**") == (9, 30, True))
    check("normalize is identity without markdown", normalize_reply("The time is 9:30.") == "The time is 9:30." and normalize_reply(None) is None)

    data_src = os.path.join(ROOT, "data_balanced")
    if not os.path.exists(os.path.join(data_src, "data.csv")):
        print("SKIP image-based checks: data_balanced/ not present locally")
        return ok
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
        env_saved = os.environ.get("CLOCK_DTYPE")
        os.environ.pop("CLOCK_DTYPE", None)
        try:
            os.chdir(tmp)
            import text_only_v2 as T2
            import conversion_test as CT
            import probe as P
            kw = dict(adapter_name="gemma-3-4b-it")
            T2.run(smoke=True, **kw)
            check("text_only_v2 smoke wrote its smoke csv", os.path.exists("results/gemma-3-4b-it/text_only_v2_smoke.csv"))
            CT.run(smoke=True, **kw)
            check("conversion_test smoke ran", os.path.isdir("results/gemma-3-4b-it") and any("conversion" in f for f in os.listdir("results/gemma-3-4b-it")),
                  str(os.listdir("results/gemma-3-4b-it")))
            csv = pd.read_csv(os.path.join(data_src, "data.csv")).head(4)
            csv.to_csv("d.csv", index=False)
            DC.main(data_csv="d.csv", images_dir=data_src, max_images=3, **kw)
            check("describe_check wrote describe.csv", os.path.exists("results/gemma-3-4b-it/describe.csv"))
            EB.run_eval(data_csv="d.csv", images_dir=data_src, max_images=3, **kw)
            check("eval_behavior wrote results.csv", os.path.exists("results/gemma-3-4b-it/results.csv"))
            os.environ["CLOCK_DTYPE"] = "float32"
            EB.run_eval(data_csv="d.csv", images_dir=data_src, max_images=2, out_csv="results/gemma-3-4b-it_fp32_crosscheck/results.csv", **kw)
            check("fp32 cross-check goes to its own folder", os.path.exists("results/gemma-3-4b-it_fp32_crosscheck/results.csv")
                  and len(pd.read_csv("results/gemma-3-4b-it_fp32_crosscheck/results.csv")) == 2)
            os.environ.pop("CLOCK_DTYPE", None)
            P.extract_activations(data_csv="d.csv", images_dir=data_src, out_dir="acts", max_images=3, **kw)
            hl = np.load("acts/hidden_last.npy")
            vm = np.load("acts/vision_meanpool.npy") if os.path.exists("acts/vision_meanpool.npy") else None
            check("probe extraction: finite hidden states, layers+1 axis", hl.shape[0] == 3 and hl.shape[1] == 4 and np.isfinite(hl).all(), str(hl.shape))
            check("probe extraction: vision hook captured something", vm is not None and np.isfinite(vm).all(), str(None if vm is None else vm.shape))
        finally:
            os.chdir(cwd)
            cls.load, cls.expected_image_tokens = saved_load, saved_exp
            (os.environ.pop("CLOCK_DTYPE", None) if env_saved is None else os.environ.__setitem__("CLOCK_DTYPE", env_saved))
    return ok


if __name__ == "__main__":
    r = main()
    print("\nALL PASS" if r else "\nSOME FAILED")
    sys.exit(0 if r else 1)
