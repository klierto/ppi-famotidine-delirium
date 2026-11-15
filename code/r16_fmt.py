#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r16

INPUTS = ["famdose_estimates.json", "ppi_estimates.json", "r15.json", "r11.json", "r12.json"]


def precise(o, key=""):
    if isinstance(o, dict):
        return {k: precise(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [precise(v, key) for v in o]
    if isinstance(o, float):
        return float(f"{o:.3g}") if key == "p" else round(o, 10)
    return o


def same(a, b, path=""):
    bad = []
    if isinstance(b, dict):
        for k in b:
            bad += same(a.get(k) if isinstance(a, dict) else None, b[k], f"{path}/{k}")
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            return [path]
        for i, (x, y) in enumerate(zip(a, b)):
            bad += same(x, y, f"{path}[{i}]")
    elif isinstance(b, float) and isinstance(a, (int, float)):
        key = path.rsplit("/", 1)[-1].split("[")[0]
        want = float(f"{a:.3g}") if key == "p" else round(a, 3 if key.startswith("risk") else 4)
        if abs(want - b) > 1e-12:
            bad.append(f"{path}: {a} vs {b}")
    elif a != b and not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) < 1e-12):
        bad.append(f"{path}: {a!r} vs {b!r}")
    return bad


RATIO = {"OR", "RR", "HR", "OR_lo", "OR_hi", "RR_lo", "RR_hi", "ratio", "OR_without"}


def fmt(a, b, key="", parent=""):
    if isinstance(b, dict):
        return {k: fmt(a[k], b[k], k, key) for k in b}
    if isinstance(b, list):
        return [fmt(x, y, key, parent) for x, y in zip(a, b)]
    if isinstance(b, float):
        rd = key in ("RD", "RD_lo", "RD_hi") or (parent == "RD" and key in ("RD", "lo", "hi", "est"))
        if rd:
            return round(a, 3)
        if key in RATIO or (key in ("lo", "hi", "est", "point", "limit") and parent != "RD") or key in ("shared", "separate", "range_without"):
            return round(a, 2)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    scratch = os.path.join(a.private, "res_precise")
    os.makedirs(scratch, exist_ok=True)
    for f in INPUTS:
        shutil.copyfile(os.path.join(a.results, f), os.path.join(scratch, f))
    r16.round_tree = precise
    r16.report(argparse.Namespace(private=a.private, results=scratch))
    A = json.load(open(os.path.join(scratch, "r16.json"), encoding="utf-8"))
    B = json.load(open(os.path.join(a.results, "r16.json"), encoding="utf-8"))
    A.pop("software", None)
    soft = B.pop("software", None)
    bad = same(A, B)
    if bad:
        print("\n".join(bad[:20]))
        raise SystemExit(f"{len(bad)} leaves differ from results/r16.json: stop (nothing written)")
    out = fmt(A, B)
    out["software"] = soft
    out["note"] = "r16.json with ratios rounded to 2 decimals and risk differences to 3, each rounded once from the unrounded estimate (code/r16_fmt.py)"
    json.dump(out, open(os.path.join(a.results, "r16_fmt.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("scratch report equals results/r16.json at the stored precision; results/r16_fmt.json written")


if __name__ == "__main__":
    main()
