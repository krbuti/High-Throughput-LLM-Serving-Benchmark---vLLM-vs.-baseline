"""Synthetic prompt generation.

Two workloads:

* ``random``  - every prompt is unique, so the prefix cache can't help.
* ``shared-prefix`` - every prompt starts with the same long "system prompt"
  and ends with a short unique question. This is the RAG / agent pattern that
  prefix caching (vLLM ``--enable-prefix-caching``) is built for.

Lengths are counted in tokens when a Hugging Face tokenizer is available
(``--tokenizer``); otherwise we fall back to a ~0.75 words-per-token estimate.
The server's ``usage.prompt_tokens`` is recorded per request either way, so the
reported numbers are always exact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Small, deterministic vocabulary. Real words keep the tokenizer at roughly
# one token per word, which makes the length estimate decent without a tokenizer.
_WORDS = (
    "the model serves requests from a queue while the scheduler batches tokens "
    "across sequences memory blocks hold keys and values for attention layers "
    "throughput depends on batch size latency depends on queueing and prefill "
    "engineers measure percentiles because averages hide the slow tail of users "
    "cache pages are allocated on demand and freed when a sequence finishes "
    "data network storage compute kernel tensor matrix vector gradient weight "
    "river mountain forest ocean desert city village bridge tower harbor garden "
    "history science music painting theatre poetry novel journal letter archive"
).split()

_QUESTIONS = [
    "Summarise the passage above in three sentences.",
    "List the five most important ideas in the text above.",
    "Write a short story inspired by the text above.",
    "Explain the text above to a ten year old.",
    "Translate the main idea of the text above into French.",
    "What questions would a skeptical reader ask about the text above?",
    "Write a haiku about the text above.",
    "Rewrite the text above as a formal business memo.",
]


@dataclass
class Prompt:
    text: str
    approx_tokens: int


class _Counter:
    def __init__(self, tokenizer_name: str | None):
        self.tok = None
        if tokenizer_name:
            from transformers import AutoTokenizer  # optional dependency

            self.tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def make_text(self, rng: random.Random, n_tokens: int) -> str:
        if n_tokens <= 0:
            return ""
        if self.tok is None:
            n_words = max(1, int(n_tokens * 0.75))
            return " ".join(rng.choice(_WORDS) for _ in range(n_words))
        # Over-generate, then cut to exactly n_tokens.
        words = " ".join(rng.choice(_WORDS) for _ in range(n_tokens * 2))
        ids = self.tok.encode(words, add_special_tokens=False)[:n_tokens]
        return self.tok.decode(ids)

    def count(self, text: str) -> int:
        if self.tok is None:
            return int(round(len(text.split()) / 0.75))
        return len(self.tok.encode(text, add_special_tokens=False))


def build_prompts(
    num_prompts: int,
    input_len: int,
    workload: str = "random",
    prefix_len: int = 0,
    seed: int = 0,
    tokenizer: str | None = None,
) -> list[Prompt]:
    """Return ``num_prompts`` prompts of roughly ``input_len`` tokens each.

    For ``shared-prefix``, ``prefix_len`` of the ``input_len`` tokens are the
    same across all prompts; the rest are unique.
    """
    rng = random.Random(seed)
    counter = _Counter(tokenizer)

    if workload == "random":
        prefix = ""
        unique_len = input_len
    elif workload == "shared-prefix":
        if prefix_len <= 0 or prefix_len >= input_len:
            raise ValueError("shared-prefix needs 0 < prefix_len < input_len")
        prefix = (
            "You are a helpful assistant. Read the reference document carefully.\n\n"
            + counter.make_text(random.Random(seed + 10_000), prefix_len)
            + "\n\n"
        )
        unique_len = input_len - prefix_len
    else:
        raise ValueError(f"unknown workload {workload!r}")

    prompts = []
    for i in range(num_prompts):
        question = _QUESTIONS[i % len(_QUESTIONS)]
        # A request id at the front of the unique part guarantees no two
        # prompts share more than the intended prefix.
        body = f"[doc {i}] " + counter.make_text(rng, max(1, unique_len - 16))
        text = f"{prefix}{body}\n\n{question}"
        prompts.append(Prompt(text=text, approx_tokens=counter.count(text)))
    return prompts
