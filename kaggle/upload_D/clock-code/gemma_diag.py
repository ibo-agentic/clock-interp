"""Gemma 3 precision diagnostics and smoke tests (notebook G0b).  Read-only with respect to the model: only forward hooks, nothing is patched.

  python gemma_diag.py --mode diag  --model_id google/gemma-3-4b-it --adapter gemma-3-4b-it            # fp16: where do values first go non-finite? sdpa vs eager
  python gemma_diag.py --mode smoke --model_id google/gemma-3-12b-it --adapter gemma-3-12b-it --dtype bf16   # adapters.run_smoke + timing + load time + memory

Outputs: results/gemma/<short>_<dtype>/diag.json | smoke.json  (+ everything printed to stdout)."""
import argparse
import gc
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------------------------------- diagnostic
def find_first_nonfinite(model, inputs, prefixes=("language_model", "vision_tower", "multi_modal_projector")):
    """Run ONE forward pass with a hook on every submodule whose qualified name contains one of `prefixes`. Returns a dict:
       records: execution-ordered list of {order,name,cls,nan,inf,maxabs,in_bad}   (maxabs = max |x| over the finite entries of the first tensor output)
       first_bad: first record with nan+inf>0;  first_origin: first bad record whose own input was finite (= where the problem is CREATED, not propagated);
       logits_finite, error."""
    recs, handles, counter = [], [], [0]

    def make(name, cls):
        def hook(mod, args, out):
            o = out[0] if isinstance(out, (tuple, list)) and len(out) and torch.is_tensor(out[0]) else out
            if not torch.is_tensor(o) or not o.is_floating_point():
                return
            x = o.detach()
            fin = torch.isfinite(x)
            a0 = args[0] if args and torch.is_tensor(args[0]) and args[0].is_floating_point() else None
            in_bad = None if a0 is None else (~torch.isfinite(a0.detach())).sum()
            recs.append((counter[0], name, cls, torch.isnan(x).sum(), (~fin & ~torch.isnan(x)).sum(),
                         torch.where(fin, x.float().abs(), torch.zeros((), device=x.device)).max(), in_bad))
            counter[0] += 1
        return hook

    for name, mod in model.named_modules():
        if name and any(p in name for p in prefixes):
            handles.append(mod.register_forward_hook(make(name, type(mod).__name__)))
    err, logits_finite = None, None
    try:
        with torch.no_grad():
            out = model(**inputs)
        logits_finite = bool(torch.isfinite(out.logits).all().item())
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:300]
    finally:
        for h in handles:
            h.remove()
    records = [{"order": o, "name": n, "cls": c, "nan": int(a), "inf": int(b), "maxabs": float(m), "in_bad": (None if ib is None else int(ib))}
               for o, n, c, a, b, m, ib in recs]
    bad = [r for r in records if r["nan"] + r["inf"] > 0]
    origin = [r for r in bad if not r["in_bad"]]
    return {"records": records, "first_bad": bad[0] if bad else None, "first_origin": origin[0] if origin else None,
            "n_bad_modules": len(bad), "logits_finite": logits_finite, "error": err}


def summarize_diag(res, label, top=8):
    print(f"\n--- {label}: logits_finite={res['logits_finite']} error={res['error']} bad_modules={res['n_bad_modules']}/{len(res['records'])}")
    fb, fo = res["first_bad"], res["first_origin"]
    print("  first non-finite output:", fb)
    print("  first non-finite whose INPUT was finite (where it is created):", fo)
    if fo is not None:
        kind = "inf (overflow-type)" if fo["inf"] and not fo["nan"] else ("nan" if fo["nan"] and not fo["inf"] else "nan+inf")
        print(f"  -> created as {kind} in {fo['name']} ({fo['cls']})")
    fin = [r for r in res["records"] if r["nan"] + r["inf"] == 0 and r["cls"] in ("Gemma3DecoderLayer", "Gemma3TextDecoderLayer")]
    fin = sorted(fin, key=lambda r: -r["maxabs"])[:top]
    print("  largest finite |activation| at decoder-layer outputs (fp16 max is 65504):")
    for r in fin:
        print(f"     {r['name']:<55} max|x|={r['maxabs']:.1f}")
    allfin = sorted([r for r in res["records"] if r["nan"] + r["inf"] == 0], key=lambda r: -r["maxabs"])[:top]
    print("  largest finite |activation| at any hooked module:")
    for r in allfin:
        print(f"     {r['name']:<55} {r['cls']:<24} max|x|={r['maxabs']:.1f}")
    return {k: v for k, v in res.items() if k != "records"} | {"top_finite": allfin}


def mode_diag(args):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    import adapters
    from clocks import generate_clock_image
    out_dir = os.path.join("results", "gemma", f"{args.adapter}_fp16_diag")
    os.makedirs(out_dir, exist_ok=True)
    img = os.path.join(out_dir, "clock_10_10.png")
    generate_clock_image(10, 10, img)
    proc = AutoProcessor.from_pretrained(args.model_id)
    report = {"model_id": args.model_id, "torch": torch.__version__}
    for impl in args.impls.split(","):
        t0 = time.time()
        print(f"\n{'=' * 90}\nLoading {args.model_id} float16 attn_implementation={impl} ...", flush=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(args.model_id, dtype=torch.float16, device_map="auto", attn_implementation=impl).eval()
        print(f"loaded in {time.time() - t0:.0f}s; attn impl on text config: {getattr(model.config.text_config, '_attn_implementation', None)}")
        ad = adapters.Gemma3Adapter()
        ad.model, ad.processor = model, proc
        for kind, (image, prompt) in {"text_only": (None, "What is 7 times 5? Answer with a number only."),
                                      "with_image": (img, "What time does the clock show? Answer only in HH:MM format.")}.items():
            inputs = ad.build_inputs(image, prompt)
            res = find_first_nonfinite(model, inputs)
            report[f"{impl}/{kind}"] = summarize_diag(res, f"fp16 / {impl} / {kind}")
            # embeddings of the prompt alone, to see whether the very first activations are already large
        del model, ad
        gc.collect()
        torch.cuda.empty_cache()
    json.dump(report, open(os.path.join(out_dir, "diag.json"), "w"), indent=1, default=str)
    print("\nwrote", os.path.join(out_dir, "diag.json"))
    return True


# ---------------------------------------------------------------------------------------------------- smoke + timing
def mode_smoke(args):
    os.environ["CLOCK_DTYPE"] = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}[args.dtype]
    import adapters
    from clocks import generate_clock_image
    from eval_behavior import PROMPT as READ_PROMPT
    short = args.adapter
    out_dir = os.path.join("results", "gemma", f"{short}_{args.dtype}")
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.is_available():
        print("bf16 supported (incl. emulation):", torch.cuda.is_bf16_supported(including_emulation=True),
              "| native:", torch.cuda.is_bf16_supported(including_emulation=False), "| capability", torch.cuda.get_device_capability(0))
    last, load_s = {}, {}
    cls = type(adapters.get_adapter(adapter_name=short))
    orig_load = cls.load

    def timed_load(self, model_id=None):
        t0 = time.time()
        r = orig_load(self, model_id)
        load_s["seconds"] = time.time() - t0
        last["adapter"] = self
        return r
    cls.load = timed_load
    ok = adapters.run_smoke(args.model_id, args.adapter, max_new_tokens=32, tmp_dir=os.path.join(out_dir, "smoke_images"))
    ad = last["adapter"]
    print(f"\nload time: {load_s['seconds']:.0f} s   dtype: {ad.dtype_name}")
    # finite-logit sweep + timing (text-only vs with image), fixed 16 new tokens
    img = os.path.join(out_dir, "timing_clock.png")
    generate_clock_image(4, 35, img)
    timing = {}
    for kind, (image, prompt) in {"text_only": (None, "The long minute hand is pointing at the number 7, and the short hour hand is pointing at the number 4. What time does the clock show? Answer only in HH:MM format."),
                                  "with_image": (img, READ_PROMPT)}.items():
        inputs = ad.build_inputs(image, prompt)
        ad.generate_answer(inputs, max_new_tokens=16)       # warm-up
        times, outs = [], []
        for _ in range(args.n_timing):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.time()
            o = ad.generate_answer(inputs, max_new_tokens=16)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - t0)
            outs.append(o)
        n_tok = max(1, len(ad.processor.tokenizer.encode(outs[0], add_special_tokens=False)))
        with torch.no_grad():
            t0 = time.time()
            ad.model(**inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            prefill = time.time() - t0
        timing[kind] = {"seconds_per_generation_mean": sum(times) / len(times), "seconds_min": min(times), "seconds_max": max(times),
                        "new_tokens_in_first_output": n_tok, "prefill_forward_seconds": prefill, "answer": outs[0], "answers_identical": len(set(outs)) == 1,
                        "prompt_tokens": int(inputs["input_ids"].shape[1])}
        print(f"timing [{kind}]: {timing[kind]}")
    mem = {f"cuda:{d}": torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())} if torch.cuda.is_available() else {}
    rec = {"model_id": args.model_id, "dtype": ad.dtype_name, "smoke_pass": bool(ok), "load_seconds": load_s["seconds"], "peak_memory_gib": mem, "timing": timing,
           "device_map": {str(d): sum(1 for v in getattr(ad.model, "hf_device_map", {}).values() if v == d) for d in set(getattr(ad.model, "hf_device_map", {}).values())}}
    json.dump(rec, open(os.path.join(out_dir, "smoke.json"), "w"), indent=1, default=str)
    print("\nwrote", os.path.join(out_dir, "smoke.json"), "\nSMOKE", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["diag", "smoke"])
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--impls", default="sdpa,eager")
    ap.add_argument("--n_timing", type=int, default=5)
    a = ap.parse_args()
    sys.exit(0 if (mode_diag(a) if a.mode == "diag" else mode_smoke(a)) else 1)
