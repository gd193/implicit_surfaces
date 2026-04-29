"""Convert MLP weights between two on-disk formats.

* **NIQ npz** (`assets/niq_models/`, `~/neural-implicit-queries/sample_inputs/`):
  flat numpy archive with keys `NNNN.dense.A`, `NNNN.dense.b`, `NNNN.<act>._`,
  `NNNN.squeeze_last._`. Weight matrix `A` has shape `(in, out)` (forward pass:
  `x @ A + b`). Used by `niImplicitSurface` via NIQ's `implicit_mlp_utils`.

* **eqx stream** (`assets/equinox_models/*.eqx`, e.g. `bunny_softplus.eqx`):
  sequential `.npy` arrays — `(W1, b1, W2, b2, …, W_out, b_out)`. Weight matrix
  `W` has shape `(out, in)` (forward pass: `W @ x + b`). Companion `.json`
  sidecar holds metadata: `{activation, input_dim, output_dim, hidden_dims}`.
  Used by `jnpboundedImplicit` via a manual MLP closure (see
  `scripts/render_neural_implicit.py` or `scripts/test_box_queries.py`).

Both formats describe the same MLP geometry; conversion just renames keys and
transposes weight matrices. Activation name is preserved (NIQ supports `relu`,
`elu`, `softplus`; autobound supports `softplus`, `swish`, `sigmoid` —
`softplus` is the common subset).

CLI:
    python scripts/convert_models.py niq2eqx in.npz out.eqx
    python scripts/convert_models.py eqx2niq in.eqx out.npz [--activation softplus]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# eqx (sequential .npy) reader/writer
# --------------------------------------------------------------------------

def read_eqx_stream(path: str | Path) -> List[np.ndarray]:
    arrs = []
    with open(path, "rb") as f:
        while True:
            try:
                arrs.append(np.lib.format.read_array(f))
            except (EOFError, ValueError):
                break
    return arrs


def write_eqx_stream(path: str | Path, arrs: List[np.ndarray]) -> None:
    with open(path, "wb") as f:
        for a in arrs:
            np.lib.format.write_array(f, np.asarray(a))


def sidecar_path_for(eqx_path: str | Path) -> Path:
    p = Path(eqx_path)
    return p.with_suffix(p.suffix + ".json")


# --------------------------------------------------------------------------
# Layer extraction
# --------------------------------------------------------------------------

# Each layer is (W: (out, in) float, b: (out,) float).  Activation lives on the
# whole network, between layers. The output layer has no following activation.
Layers = List[Tuple[np.ndarray, np.ndarray]]


def eqx_to_layers(eqx_path: str | Path) -> Layers:
    arrs = read_eqx_stream(eqx_path)
    if len(arrs) % 2 != 0:
        raise ValueError(f"{eqx_path}: expected even number of arrays, got {len(arrs)}")
    layers: Layers = []
    for i in range(0, len(arrs), 2):
        W, b = arrs[i], arrs[i + 1]
        if W.ndim != 2 or b.ndim != 1 or W.shape[0] != b.shape[0]:
            raise ValueError(f"{eqx_path}: bad layer shape at index {i}: W={W.shape}, b={b.shape}")
        layers.append((W, b))
    return layers


def niq_npz_to_layers(npz_path: str | Path) -> Tuple[Layers, str]:
    """Returns (layers, activation_name). Activation is extracted from the npz keys."""
    z = np.load(npz_path, allow_pickle=True)

    by_idx: dict[int, dict[str, np.ndarray]] = {}
    activations: list[str] = []
    for k in z.files:
        idx_s, name, _kind = k.split(".", 2)
        idx = int(idx_s)
        if name == "dense":
            by_idx.setdefault(idx, {})[k.rsplit(".", 1)[-1]] = z[k]
        elif name == "squeeze_last":
            continue
        else:
            activations.append(name)

    activation_set = set(activations)
    if len(activation_set) > 1:
        raise ValueError(f"{npz_path}: mixed activations {activation_set}")
    activation = activations[0] if activations else "linear"

    layers: Layers = []
    for idx in sorted(by_idx):
        L = by_idx[idx]
        if "A" not in L or "b" not in L:
            raise ValueError(f"{npz_path}: missing A or b at index {idx}")
        # NIQ stores W as (in, out); transpose to (out, in).
        layers.append((L["A"].T.copy(), L["b"].copy()))
    if not layers:
        raise ValueError(f"{npz_path}: no dense layers found")
    return layers, activation


# --------------------------------------------------------------------------
# Layer writers
# --------------------------------------------------------------------------

def layers_to_eqx(layers: Layers, eqx_path: str | Path,
                  activation: str, dtype: np.dtype = np.float64) -> None:
    arrs: list[np.ndarray] = []
    for W, b in layers:
        arrs.append(np.asarray(W, dtype=dtype))
        arrs.append(np.asarray(b, dtype=dtype))
    write_eqx_stream(eqx_path, arrs)

    in_dim = int(layers[0][0].shape[1])
    out_dim = int(layers[-1][0].shape[0])
    hidden_dims = [int(W.shape[0]) for W, _ in layers[:-1]]
    meta = {
        "activation": activation,
        "input_dim": in_dim,
        "output_dim": out_dim,
        "hidden_dims": hidden_dims,
    }
    sidecar_path_for(eqx_path).write_text(json.dumps(meta, indent=2))


def layers_to_niq_npz(layers: Layers, npz_path: str | Path,
                      activation: str, dtype: np.dtype = np.float32) -> None:
    out: dict[str, np.ndarray] = {}
    n_layers = len(layers)
    idx = 0
    for li, (W, b) in enumerate(layers):
        out[f"{idx:04d}.dense.A"] = np.asarray(W.T, dtype=dtype)  # back to (in, out)
        out[f"{idx:04d}.dense.b"] = np.asarray(b, dtype=dtype)
        idx += 1
        if li < n_layers - 1:
            out[f"{idx:04d}.{activation}._"] = np.zeros((0,), dtype=dtype)
            idx += 1
    out[f"{idx:04d}.squeeze_last._"] = np.zeros((0,), dtype=dtype)
    np.savez(npz_path, **out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_niq2eqx(args: argparse.Namespace) -> None:
    layers, activation = niq_npz_to_layers(args.input)
    if args.activation:
        activation = args.activation
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers_to_eqx(layers, out_path, activation, dtype=_dtype_for(args.dtype))
    print(_summary("niq", "eqx", args.input, out_path, layers, activation))


def cmd_eqx2niq(args: argparse.Namespace) -> None:
    layers = eqx_to_layers(args.input)
    activation = args.activation
    if activation is None:
        side = sidecar_path_for(args.input)
        if side.exists():
            activation = json.loads(side.read_text())["activation"]
        else:
            raise SystemExit(
                "activation unknown: provide --activation or place a sidecar JSON next to the eqx"
            )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layers_to_niq_npz(layers, out_path, activation, dtype=_dtype_for(args.dtype))
    print(_summary("eqx", "niq", args.input, out_path, layers, activation))


def _dtype_for(name: str | None) -> np.dtype:
    return {None: np.float64, "f32": np.float32, "f64": np.float64}[name]


def _summary(src_kind, dst_kind, src_path, dst_path, layers: Layers, activation: str) -> str:
    in_dim = layers[0][0].shape[1]
    arch = " → ".join([str(in_dim)] + [str(W.shape[0]) for W, _ in layers])
    return (
        f"{src_kind} → {dst_kind}: {Path(src_path).name} → {Path(dst_path).name} | "
        f"arch {arch}, activation {activation}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("niq2eqx", help="NIQ npz → eqx stream + JSON sidecar")
    a.add_argument("input")
    a.add_argument("output")
    a.add_argument("--activation", help="Override the activation name written to the sidecar")
    a.add_argument("--dtype", choices=["f32", "f64"], default="f64",
                   help="Output dtype (default f64; eqx files in repo are f64)")
    a.set_defaults(func=cmd_niq2eqx)

    b = sub.add_parser("eqx2niq", help="eqx stream → NIQ npz")
    b.add_argument("input")
    b.add_argument("output")
    b.add_argument("--activation", help="Activation name (defaults to sidecar JSON)")
    b.add_argument("--dtype", choices=["f32", "f64"], default="f32",
                   help="Output dtype (default f32; matches existing NIQ npz files)")
    b.set_defaults(func=cmd_eqx2niq)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
