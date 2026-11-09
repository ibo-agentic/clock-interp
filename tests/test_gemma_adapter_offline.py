"""Offline smoke test of Gemma3Adapter / Gemma3_12BAdapter plumbing on a TINY RANDOM Gemma3 (CPU, no download, no gated repo).
Builds a toy tokenizer + Gemma3Processor + tiny Gemma3ForConditionalGeneration, loads them into the adapter, and checks the adapter interface:
build_inputs (image and text-only), single BOS, token_type_ids, 256-style placeholder expansion (here 4 tokens), image_token_positions, generate_answer
(greedy, runs, deterministic), decoder_layers count, registry/hints, and the run_smoke hook.  It does NOT test real-weight behaviour (fp16 overflow, answer quality):
that is what `python adapters.py --smoke --model_id google/gemma-3-4b-it` on Kaggle is for.
usage: python tests/test_gemma_adapter_offline.py"""
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PIL import Image  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (Gemma3Config, Gemma3ForConditionalGeneration, Gemma3ImageProcessor, Gemma3Processor, Gemma3TextConfig,  # noqa: E402
                          PreTrainedTokenizerFast, SiglipVisionConfig)

import adapters  # noqa: E402

TEMPLATE = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ 'model' if m['role'] == 'assistant' else 'user' }}\n"
            "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for it in m['content'] %}"
            "{% if it['type'] == 'image' %}<start_of_image>{% elif it['type'] == 'text' %}{{ it['text'] }}{% endif %}{% endfor %}{% endif %}"
            "<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>", "<image_soft_token>"]
WORDS = ["what", "time", "does", "the", "clock", "show", "?", "answer", "only", "in", "hh:mm", "format", ".", "user", "model", "7", "x", "5", "is", "times", "\n"]
N_TOK = 4          # tokens per image in the toy model (256 in the real one)


def build(tmp):
    vocab = {t: i for i, t in enumerate(SPECIALS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    fast.add_special_tokens({"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>", "<image_soft_token>"]})   # matched inside text, as in the real tokenizer
    # attributes the real Gemma3 tokenizer carries and the processor reads
    fast.boi_token, fast.eoi_token, fast.image_token = "<start_of_image>", "<end_of_image>", "<image_soft_token>"
    fast.boi_token_id, fast.eoi_token_id, fast.image_token_id = (vocab[t] for t in ("<start_of_image>", "<end_of_image>", "<image_soft_token>"))
    ip = Gemma3ImageProcessor(size={"height": 56, "width": 56}, do_pan_and_scan=False)
    proc = Gemma3Processor(image_processor=ip, tokenizer=fast, chat_template=TEMPLATE, image_seq_length=N_TOK)
    V = len(vocab)
    text = Gemma3TextConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=16, max_position_embeddings=256, sliding_window=32, layer_types=["sliding_attention", "sliding_attention", "full_attention"])
    vis = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=56, patch_size=14)
    cfg = Gemma3Config(text_config=text, vision_config=vis, mm_tokens_per_image=N_TOK, image_token_id=vocab["<image_soft_token>"],
                       boi_token_id=vocab["<start_of_image>"], eoi_token_id=vocab["<end_of_image>"])
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration(cfg).eval()
    model.generation_config.suppress_tokens = [vocab["<image_soft_token>"], vocab["<start_of_image>"]]   # toy-only: a random model could emit the placeholder id, which the no-cache re-feed would then count as an image token
    return model, proc


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")

    with tempfile.TemporaryDirectory() as tmp:
        model, proc = build(tmp)
        for cls in (adapters.Gemma3Adapter, adapters.Gemma3_12BAdapter):
            ad = cls()
            ad.model, ad.processor = model, proc
            print(f"{cls.__name__} short_name={ad.short_name} default_model_id={ad.default_model_id}")
            img_path = os.path.join(tmp, "clock.png")
            Image.new("RGB", (100, 80), (200, 200, 200)).save(img_path)
            a = ad.build_inputs(img_path, "what time does the clock show ?")
            t = ad.build_inputs(None, "what is 7 x 5 ?")
            ids = a["input_ids"][0]
            check("one BOS at the start, not two", int((ids[:2] == proc.tokenizer.bos_token_id).sum()) == 1, str(ids[:4].tolist()))
            check("image tokens == toy mm_tokens_per_image", int(ad.image_token_positions(a).sum()) == N_TOK, f"{int(ad.image_token_positions(a).sum())}")
            check("text-only input has 0 image tokens", int(ad.image_token_positions(t).sum()) == 0)
            check("token_type_ids present and count == image tokens", "token_type_ids" in a and int((a["token_type_ids"][0] == 1).sum()) == N_TOK)
            check("pixel_values cast to model dtype", a["pixel_values"].dtype == model.dtype, str(a["pixel_values"].dtype))
            check("adapter smoke_checks report no problem", ad.smoke_checks(a) == [], str(ad.smoke_checks(a)))
            o1 = ad.generate_answer(a, max_new_tokens=5)
            o2 = ad.generate_answer(a, max_new_tokens=5)
            check("generate_answer greedy: runs, returns str, deterministic", isinstance(o1, str) and o1 == o2, repr(o1))
            o3 = ad.generate_answer(a, max_new_tokens=5, use_cache=False)
            check("generate without cache equals with cache", o3 == o1, f"{o1!r} vs {o3!r}")
            check("generate_answer text-only runs", isinstance(ad.generate_answer(t, max_new_tokens=4), str))
            check("decoder_layers finds the 3 layers", len(ad.decoder_layers()) == 3 and ad.num_layers == 3, str(len(ad.decoder_layers())))
            check("run_config_notes has the expected keys", set(ad.run_config_notes()) >= {"chat_template", "image", "decoding", "dtype"})
        # CLOCK_DTYPE (Gemma only) + the non-finite locator used by notebook G0b
        saved = os.environ.get("CLOCK_DTYPE")
        try:
            os.environ.pop("CLOCK_DTYPE", None)
            check("_gemma_dtype default float16", adapters._gemma_dtype() == (torch.float16, "float16"))
            os.environ["CLOCK_DTYPE"] = "bf16"
            check("_gemma_dtype bf16", adapters._gemma_dtype() == (torch.bfloat16, "bfloat16"))
            os.environ["CLOCK_DTYPE"] = "float32"
            check("_gemma_dtype float32", adapters._gemma_dtype() == (torch.float32, "float32"))
            os.environ["CLOCK_DTYPE"] = "int8"
            try:
                adapters._gemma_dtype(); check("_gemma_dtype rejects unknown", False)
            except ValueError:
                check("_gemma_dtype rejects unknown", True)
        finally:
            if saved is None:
                os.environ.pop("CLOCK_DTYPE", None)
            else:
                os.environ["CLOCK_DTYPE"] = saved
        import gemma_diag
        ad = adapters.Gemma3Adapter()
        ad.model, ad.processor = model, proc
        inp = ad.build_inputs(img_path, "what time does the clock show ?")
        clean = gemma_diag.find_first_nonfinite(model, inp)
        check("locator: clean model has no bad module", clean["first_bad"] is None and clean["logits_finite"] is True and len(clean["records"]) > 20, f"{len(clean['records'])} hooked modules")
        lm = model.model.language_model if hasattr(model.model, "language_model") else model.language_model
        with torch.no_grad():
            lm.layers[1].mlp.down_proj.weight[0, 0] = float("nan")
        bad = gemma_diag.find_first_nonfinite(model, inp)
        check("locator: finds layers.1.mlp.down_proj as the creation point", bad["first_origin"] is not None and bad["first_origin"]["name"].endswith("layers.1.mlp.down_proj")
              and bad["logits_finite"] is False, str(bad["first_origin"] and bad["first_origin"]["name"]))
        # gemma_diag smoke mode end-to-end on the toy model (load stubbed): exercises timing, load-time, json writing
        import types
        def fake_load(self, model_id=None):
            self.model, self.processor = model, proc
            self.torch_dtype, self.dtype_name = adapters._gemma_dtype()
            return model, proc
        cls_ = adapters.Gemma3Adapter
        saved_load, saved_exp = cls_.load, cls_.expected_image_tokens
        cls_.load, cls_.expected_image_tokens = fake_load, N_TOK
        cwd = os.getcwd()
        env_saved = os.environ.get("CLOCK_DTYPE")
        try:
            os.chdir(tmp)
            sys.path.insert(0, ROOT)
            ns = types.SimpleNamespace(model_id="toy", adapter="gemma-3-4b-it", dtype="fp32", n_timing=2)
            r = gemma_diag.mode_smoke(ns)
            rec = __import__("json").load(open(os.path.join("results", "gemma", "gemma-3-4b-it_fp32", "smoke.json")))
            check("gemma_diag smoke mode ran and wrote smoke.json with timing", set(rec["timing"]) == {"text_only", "with_image"} and rec["load_seconds"] >= 0 and rec["dtype"] == "float32",
                  f"verdict={r}")
        finally:
            os.chdir(cwd)
            (os.environ.pop("CLOCK_DTYPE", None) if env_saved is None else os.environ.__setitem__("CLOCK_DTYPE", env_saved))
            cls_.load, cls_.expected_image_tokens = saved_load, saved_exp
        # CLOCK_MAX_MEMORY (Gemma only) and the G0c ladder driver (subprocess mocked)
        saved_mm = os.environ.get("CLOCK_MAX_MEMORY")
        try:
            os.environ.pop("CLOCK_MAX_MEMORY", None)
            check("_gemma_max_memory unset -> None (default load call unchanged)", adapters._gemma_max_memory() is None)
            os.environ["CLOCK_MAX_MEMORY"] = "0:9.5GiB,1:13.5GiB"
            check("_gemma_max_memory parses", adapters._gemma_max_memory() == {0: "9.5GiB", 1: "13.5GiB"}, str(adapters._gemma_max_memory()))
        finally:
            (os.environ.pop("CLOCK_MAX_MEMORY", None) if saved_mm is None else os.environ.__setitem__("CLOCK_MAX_MEMORY", saved_mm))
        import subprocess as _sp
        class FakeProc:
            def __init__(self, rc, text): self.stdout, self._rc = iter(text.splitlines(True)), rc
            def wait(self): return self._rc
        rcs = iter([(1, "x" + chr(10) + "torch.OutOfMemoryError: CUDA out of memory" + chr(10)), (0, "SMOKE PASS" + chr(10))])
        real_popen = _sp.Popen
        _sp.Popen = lambda *a, **k: FakeProc(*next(rcs))
        cwd2 = os.getcwd()
        try:
            os.chdir(tmp)
            r = gemma_diag.mode_ladder(types.SimpleNamespace(model_id="m", adapter="gemma-3-12b-it", dtype="bf16", n_timing=1, caps="0:9GiB,1:13GiB;0:10GiB,1:13GiB;0:11GiB,1:12GiB"))
            lad = __import__("json").load(open(os.path.join("results", "gemma", "gemma-3-12b-it_bf16_ladder.json")))
            check("ladder: stops at first PASS, records the OOM of try 0", r is True and lad["winner_try"] == 1 and lad["tries"][0]["oom"] and len(lad["tries"]) == 2, str(lad["winner_max_memory"]))
        finally:
            _sp.Popen = real_popen
            os.chdir(cwd2)
        check("registry + hints", adapters._ADAPTERS["gemma-3-12b-it"] is adapters.Gemma3_12BAdapter
              and type(adapters.get_adapter(model_id="google/gemma-3-12b-it")).__name__ == "Gemma3_12BAdapter"
              and type(adapters.get_adapter(model_id="google/gemma-3-4b-it")).__name__ == "Gemma3Adapter")
        check("Qwen/InternVL hints unchanged", type(adapters.get_adapter(model_id="Qwen/Qwen2.5-VL-7B-Instruct")).__name__ == "Qwen7BAdapter"
              and type(adapters.get_adapter(model_id="OpenGVLab/InternVL3-2B-hf")).__name__ == "InternVLAdapter")
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
