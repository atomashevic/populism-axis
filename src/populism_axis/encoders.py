"""Frozen sentence encoders for the pipeline, with per-model input templates.

Ported from `scripts/encode_contextual_axis_v3.py::make_encoder`. Two changes:
each model carries an explicit input template (the earlier run encoded Qwen3-Embedding
without the instruction prefix its model card requires), and the encoder is
resource-conservative for the shared GPU host (fp16, two CPU threads, memory
released on close). torch and transformers are imported lazily so the CPU-only
evaluation environment can import this module for templates and specs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable, Sequence

import numpy as np

from .model_revisions import E5_LARGE_V2, PINNED_REVISIONS

# Retrieval framing, not labels: the instruction asks for claim-level similarity.
DEFAULT_TASK = (
    "Retrieve political speech sentences that make the same claim about the people, "
    "the elite, or the moral character of political opponents"
)


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    label: str
    pooling: str  # "mean" | "cls" | "last"
    template: str = "{text}"  # may contain {task} and {text}
    max_length: int = 96
    batch_size: int = 64
    trust_remote_code: bool = False
    dtype: str = "float16"
    revision: str | None = None  # exact Hub commit; None follows the moving main branch

    def as_dict(self) -> dict:
        return asdict(self)


ENCODER_SPECS: dict[str, EncoderSpec] = {
    spec.label: spec
    for spec in (
        # The frozen configuration's encoder loads an exact Hub commit (model_revisions.py).
        # The sweep-only encoders below were never pinned and still follow main.
        EncoderSpec(E5_LARGE_V2, "e5-large-v2", "mean", "query: {text}", revision=PINNED_REVISIONS[E5_LARGE_V2]),
        EncoderSpec("intfloat/e5-small-v2", "e5-small-v2", "mean", "query: {text}"),
        # Qwen3-Embedding model card: f"Instruct: {task}\nQuery:{query}", last-token pooling.
        # Qwen3 models run in bfloat16 (their native precision; fp16 can overflow).
        EncoderSpec(
            "Qwen/Qwen3-Embedding-0.6B", "qwen3-emb-0.6b", "last",
            "Instruct: {task}\nQuery:{text}", batch_size=32, dtype="bfloat16",
        ),
        EncoderSpec(
            "Qwen/Qwen3-Embedding-0.6B", "qwen3-emb-0.6b-noinstr", "last",
            "{text}", batch_size=32, dtype="bfloat16",
        ),
        EncoderSpec(
            "Qwen/Qwen3-Embedding-4B", "qwen3-emb-4b", "last",
            "Instruct: {task}\nQuery:{text}", batch_size=12, dtype="bfloat16",
        ),
        EncoderSpec(
            "Qwen/Qwen3-Embedding-4B", "qwen3-emb-4b-noinstr", "last",
            "{text}", batch_size=12, dtype="bfloat16",
        ),
        EncoderSpec(
            "Qwen/Qwen3-Embedding-8B", "qwen3-emb-8b", "last",
            "Instruct: {task}\nQuery:{text}", batch_size=8, dtype="bfloat16",
        ),
        EncoderSpec(
            "Qwen/Qwen3-Embedding-8B", "qwen3-emb-8b-noinstr", "last",
            "{text}", batch_size=8, dtype="bfloat16",
        ),
        EncoderSpec(
            "Alibaba-NLP/gte-large-en-v1.5", "gte-large-en-v1.5", "cls", "{text}", trust_remote_code=True,
        ),
        # multilingual-e5-large-instruct model card: f"Instruct: {task}\nQuery: {query}", mean pooling.
        EncoderSpec(
            "intfloat/multilingual-e5-large-instruct", "me5-large-instruct", "mean",
            "Instruct: {task}\nQuery: {text}", batch_size=32,
        ),
        # Model cards use the instruction for queries only; documents are encoded plain.
        EncoderSpec(
            "intfloat/multilingual-e5-large-instruct", "me5-large-instruct-noinstr", "mean",
            "{text}", batch_size=32,
        ),
    )
}


def get_spec(label: str, **overrides) -> EncoderSpec:
    if label not in ENCODER_SPECS:
        raise KeyError(f"unknown encoder label {label!r}; known: {sorted(ENCODER_SPECS)}")
    spec = ENCODER_SPECS[label]
    return replace(spec, **{key: value for key, value in overrides.items() if value is not None})


def render_text(spec: EncoderSpec, text: str, task: str = DEFAULT_TASK) -> str:
    return spec.template.format(task=task, text=str(text))


def make_encoder(
    spec: EncoderSpec,
    device: str = "cuda",
    task: str = DEFAULT_TASK,
    num_threads: int = 2,
    log_every: int = 50,
    local_files_only: bool = False,
) -> tuple[Callable[[Sequence[str]], np.ndarray], Callable[[], None]]:
    """Return (encode, close). `encode` yields raw pooled float32 vectors (no centering)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code, revision=spec.revision, local_files_only=local_files_only
        )
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtypes[spec.dtype] if device.startswith("cuda") else torch.float32
        model = AutoModel.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code, dtype=dtype, revision=spec.revision, local_files_only=local_files_only
        ).eval().to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if spec.pooling == "last":
            tokenizer.padding_side = "left"
        # Instruction templates get their own token budget on top of `max_length`, so the
        # instruction never displaces window text. Plain prefixes ("query: ") keep the earlier
        # convention of counting toward `max_length`.
        max_length = spec.max_length
        if "{task}" in spec.template:
            max_length += len(tokenizer(render_text(spec, "", task), add_special_tokens=True)["input_ids"])

    except BaseException:
        model = None
        torch.set_num_threads(previous_threads)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        raise

    @torch.no_grad()
    def encode(texts: Sequence[str]) -> np.ndarray:
        hidden_size = model.config.hidden_size
        out = np.zeros((len(texts), hidden_size), dtype=np.float32)
        for batch_number, start in enumerate(range(0, len(texts), spec.batch_size)):
            chunk = [render_text(spec, item, task) for item in texts[start : start + spec.batch_size]]
            batch = tokenizer(
                chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            hidden = model(**batch).last_hidden_state
            if spec.pooling == "cls":
                pooled = hidden[:, 0]
            elif spec.pooling == "last":
                pooled = hidden[:, -1]  # left padding puts the last real token at -1
            else:
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out[start : start + len(chunk)] = pooled.float().cpu().numpy()
            if log_every and batch_number % log_every == 0:
                print(f"    encoded {start + len(chunk)}/{len(texts)}", flush=True)
        if not np.isfinite(out).all():
            raise FloatingPointError(f"{spec.label}: non-finite embeddings in {int((~np.isfinite(out)).any(axis=1).sum())} rows")
        return out

    encode.effective_max_length = max_length  # type: ignore[attr-defined]
    # The commit transformers actually resolved, for run manifests.
    encode.resolved_revision = getattr(model.config, "_commit_hash", None)  # type: ignore[attr-defined]

    def close() -> None:
        nonlocal model
        model = None
        torch.set_num_threads(previous_threads)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return encode, close
