#!/usr/bin/env python3
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r8
import r7

KEEP = ("OR", "OR_lo", "OR_hi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    P, R = os.path.expanduser(a.private), os.path.expanduser(a.results)
    keys = r7.BASE_KEYS + ["nopdx", "ord", "dose"] + list(r8.NEW)
    E = {k: json.load(open(os.path.join(P, f"{'r8' if k in r8.NEW else 'r7'}_est_{k}.json"))) for k in keys}
    B = {k: r7.load_boot(P, k) for k in keys}
    S = {k: r7.summarise(E, B, k) for k in keys}
    out = {"source": "r7.py and r8.py summaries, unrounded (fig_values.py)", "per_key": {}, "pooled": {}, "ratio": {}}
    for k in keys:
        out["per_key"][k] = {n: {x: e.get(x) for x in KEEP} for n, e in S[k].items()}
    pairs = {f"ppi:{n}": ("c2m", "c2e", n) for n in r7.names_for("c2m") if n in S["c2e"]}
    pairs.update({f"famdose:{n}": ("c1m", "c1e", n) for n in r7.names_for("c1m")})
    for v, other in (("nopdx", "c2e"), ("ord", "c2e"), ("nohome", "c2e"), ("ordnohome", "c2e")):
        pairs.update({f"{v}:{n}": (v, other, n) for n in ("m0_primary", "m0_negctl")})
    pairs.update({f"dose:{n}": ("c1m", "dose", n) for n in ("m0_primary", "m0_negctl")})
    for name, (x, y, n) in pairs.items():
        p = r7.pool(S[x][n], S[y][n])
        out["pooled"][name] = None if p is None else {z: p[z] for z in ("OR", "OR_lo", "OR_hi", "I2")}
    for k in ("c2m", "c2e"):
        rt = r7.ratio(B, k, "px_icu", "px_cam")
        out["ratio"][k] = None if rt is None else {"point": S[k]["px_icu"]["OR"] / S[k]["px_cam"]["OR"], **rt}
    json.dump(out, open(os.path.join(R, "fig_values_r7r8.json"), "w", encoding="utf-8"), indent=1)
    print(f"fig_values_r7r8.json written ({len(keys)} keys, {len(out['pooled'])} pooled)", flush=True)


if __name__ == "__main__":
    main()
