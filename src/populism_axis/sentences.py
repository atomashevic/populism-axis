"""Load anchor, nuisance and hypothesis sentence files into ordered groups.

A group is a named, ordered list of sentences (for example `E__populist`).
The same loader serves the proposition schema, the minimal-pair schema,
the nuisance schema, and a plain `{"sentences": [...]}` list, so the GPU
encode script needs no knowledge of how an axis will be built from them.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_sentence_groups(path: Path) -> tuple[str, dict[str, list[str]]]:
    """Return (schema, {group_name: [sentences]}) with group order preserved."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(payload.get("schema", "plain"))
    groups: dict[str, list[str]] = {}
    if "sentences" in payload and isinstance(payload["sentences"], list):
        groups["sentences"] = [str(item) for item in payload["sentences"]]
    elif schema.startswith("populism_axis.v3.proposition_anchors"):
        for channel, block in payload["channels"].items():
            groups[f"{channel}__populist"] = list(block["populist"])
            groups[f"{channel}__mainstream"] = list(block["mainstream"])
    elif schema.startswith("populism_axis.v4.pair_anchors"):
        for channel, block in payload["channels"].items():
            pairs = block["pairs"]
            groups[f"{channel}__populist"] = [pair["populist"] for pair in pairs]
            groups[f"{channel}__mainstream"] = [pair["matched_mainstream"] for pair in pairs]
    elif schema.startswith("populism_axis.v4.nuisance_sets"):
        for name, block in payload["sets"].items():
            pairs = block["pairs"]
            groups[f"{name}__nuisance"] = [pair["nuisance"] for pair in pairs]
            groups[f"{name}__neutral"] = [pair["neutral"] for pair in pairs]
    else:
        raise ValueError(f"unsupported sentence file schema: {schema}")
    for name, sentences in groups.items():
        if not sentences or any(not str(item).strip() for item in sentences):
            raise ValueError(f"group {name} is empty or contains an empty sentence")
    return schema, groups
