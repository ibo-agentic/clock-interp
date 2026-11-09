"""Reply normalisation applied before ALL answer parsing (every model).  Removes markdown emphasis only: runs of '*', backticks, runs of 2+ underscores, '~~'.
Identity on any string that contains none of those characters (so, by construction and by analysis/check_normalization_noop.py, on all replies of the Qwen2.5-VL and
InternVL3 models, which contain no markdown).  The raw reply is always stored unchanged; only the parser sees the normalised text."""
import re

_MD = re.compile(r"\*+|`+|_{2,}|~~")


def normalize_reply(text):
    return _MD.sub("", text) if isinstance(text, str) else text
