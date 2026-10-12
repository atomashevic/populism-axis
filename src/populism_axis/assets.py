"""Verified runtime assets, accessible from an installed package."""
from hashlib import sha256
from importlib.resources import files
import io
import json

_ASSETS = {
    "anchors": ("proposition_anchors.json", "ea98e5b2071567a825edef608e45bac02b82b374e87623ebdd8397427cb4bcd6"),
    "hypotheses": ("claim_hypotheses.json", "487a7cc29bb1f0893fd66df877f33e34da38fcc4c9124ecb18b3a949bbe5c047"),
    "reference_stats": ("reference_stats.json", "a0ce613937153caf7b15ec7097a805272ebaed13bfa200391b2d6db3a9d6d8e6"),
    "reference_mean": ("reference_mean_embedding.npy", "c33becef7eb45a3bfeb2c46658106654d38b4b24d75c2c43a64466239d822cfa"),
}


def _verified_bytes():
    root = files("populism_axis").joinpath("resources")
    payload = {}
    for logical, (filename, expected) in _ASSETS.items():
        data = root.joinpath(filename).read_bytes()
        actual = sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Packaged asset '{filename}' failed SHA-256 verification. Restore the matching package; do not regenerate frozen assets.")
        payload[logical] = data
    return payload


def preset_asset_hashes() -> dict:
    """Return verified logical asset names and SHA-256 identities."""
    return {name: sha256(data).hexdigest() for name, data in _verified_bytes().items()}


def load_preset_assets() -> dict:
    import numpy as np
    data = _verified_bytes()
    out = {name: json.loads(value) for name, value in data.items() if name != "reference_mean"}
    out["reference_mean"] = np.load(io.BytesIO(data["reference_mean"]), allow_pickle=False)
    if out["reference_mean"].shape != (1024,) or out["reference_mean"].dtype != np.dtype("float32"):
        raise RuntimeError("Packaged reference mean must be a 1024-element float32 vector.")
    return out


def verify_assets() -> dict:
    """Verify bytes and reference-vector metadata without model inference."""
    assets = load_preset_assets()
    return {"hashes": preset_asset_hashes(), "reference_mean_shape": list(assets["reference_mean"].shape),
            "reference_mean_dtype": str(assets["reference_mean"].dtype)}
