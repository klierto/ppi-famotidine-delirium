#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc
import r16

NAMES = ["pdx_primary", "pdx_negctl", "pdx_light_primary"]


def add_spec(d):
    groups = [g for g in r16.PDX if g != "circulatory"]
    adm = [a for a in r16.ADM if a != "emergency"]
    cols = [d["pdx"] == g for g in groups] + [d["adm"] == a for a in adm]
    d["X_pdx"], d["names_pdx"], dropped = r16.add_cols(d, cols, [f"pdx_{g}" for g in groups] + [f"adm_{a}" for a in adm])
    return dropped


def est(d, cache):
    res = {}
    alln = np.ones(len(d["A"]), bool)
    s, w = r16.unit_fit(d, alln, cache, "pdx", "pdx")
    res["pdx_primary"], res["pdx_negctl"] = r16.cif(s, w, s["t_del"]), r16.cif(s, w, s["t_hypo"])
    s, w = r16.unit_fit(d, d["rgrp"] == "light", cache, "pdx_light", "pdx")
    res["pdx_light_primary"] = r16.cif(s, w, s["t_del"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    t0 = time.time()
    d, _ = r16.load2(a.private, "m")
    dropped = add_spec(d)
    cache = {}
    pt = est(d, cache)
    s, w = r16.unit_fit(d, np.ones(len(d["A"]), bool), {}, "pdx", "pdx")
    M = np.column_stack([d["pdx"] == g for g in r16.PDX] + [d["adm"] == g for g in r16.ADM]).astype(float)
    smd_w = sc.smd_table(M, list(r16.PDX) + list(r16.ADM), d["A"], w)
    V = np.full((r16.NBOOT, 2 * len(NAMES)), np.nan)
    failed = 0
    for rep in range(r16.NBOOT):
        try:
            b = asd.resample(d, "m", np.random.default_rng(asd.SEED + rep))
            r = est(b, dict(cache))
            V[rep] = [x for nm in NAMES for x in (r[nm]["riskA"], r[nm]["riskB"])]
        except Exception:
            failed += 1
    B = {"cols": [f"{nm}:{k}" for nm in NAMES for k in ("A", "B")], "v": V}
    main_json = json.load(open(os.path.join(a.results, "r16.json"), encoding="utf-8"))["db"]["m"]["est"]
    out = {"plan": "added after the R16 results were seen (post hoc)", "boot": {"reps": r16.NBOOT, "failed": failed},
           "columns_added": [c for c in d["names_pdx"] if c not in d["names"]], "columns_dropped": dropped,
           "smd_after_weighting_with_the_added_model": {k: round(v, 3) for k, v in smd_w.items()}, "est": {}, "reading": {}}
    for nm in NAMES:
        e = r16.risk_entry(pt[nm], B, nm)
        out["est"][nm] = e
        ref = main_json["light_primary" if "light" in nm else ("negctl" if "negctl" in nm else "primary")]
        out["reading"][nm] = r16.consistent(e, ref)
    json.dump(r16.round_tree(asd.to_json(out)), open(os.path.join(a.results, "r16b.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for nm in NAMES:
        e = out["est"][nm]
        print(f"{nm}: n {e['nA']}/{e['nB']}; risks {100 * e['riskA']:.1f}%/{100 * e['riskB']:.1f}%; OR {e['OR']:.2f} ({e['OR_lo']:.2f}-{e['OR_hi']:.2f}); "
              f"RR {e['RR']:.2f} ({e['RR_lo']:.2f}-{e['RR_hi']:.2f}); RD {100 * e['RD']:.1f} ({100 * e['RD_lo']:.1f} to {100 * e['RD_hi']:.1f}); {out['reading'][nm]}")
    print("weighted SMD with the added model:", out["smd_after_weighting_with_the_added_model"])
    asd.log_run(a.results, f"r16b: ok, failed {failed} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
