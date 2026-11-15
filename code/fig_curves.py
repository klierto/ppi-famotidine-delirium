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
import r11
import r15

DBS = ("m", "e")
UNITS = {"all": "all patients (asd.py's overlap weights)",
         "light": "awake or lightly sedated, RASS >= -1 at time zero (E1's stratum; propensity model refitted, as E1 and R15)"}
ARMS = ((1, "PPI"), (0, "Famotidine"))
DAYS = tuple(range(8))


def masked(n):
    s = sc.mask_n(n)
    return s if s.startswith("<") else int(s)


def unit_data(d, u, cache):
    if u == "all":
        return d, r11.ow_of(d, cache, "full")
    s = asd.subset_d(d, r15.unit_mask(d, "light"))
    return s, r11.ow_of(s, cache, "r15_light")


def curves(dd, w):
    A = dd["A"]
    kend, typ, _ = sc.follow_types(dd["t_end"], dd["end_death"], dd["t_del"], disc_competing=True)
    W = np.asarray(w, float)
    out = {}
    for a, arm in ARMS:
        r = A == a
        cif = sc.aj_cif(W[r], kend[r], typ[r])
        at = [int((kend[r] >= 24 * k).sum()) for k in DAYS[:-1]] + [int((typ[r] == 0).sum())]
        out[arm] = {"cif": [0.0] + [float(v) for v in cif], "at_risk": [masked(n) for n in at], "n": masked(int(r.sum()))}
    ref = asd.cif_contrast(dd, w, dd["t_del"])[168]
    return out, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    t0 = time.time()
    P, R = os.path.expanduser(a.private), os.path.expanduser(a.results)
    c2 = json.load(open(os.path.join(R, "ppi_estimates.json"), encoding="utf-8"))["per_db"]
    j1 = json.load(open(os.path.join(R, "ppi_explore_E1.json"), encoding="utf-8"))["db"]
    j15 = json.load(open(os.path.join(R, "r15.json"), encoding="utf-8"))["est"]
    out = {"source": "fig_curves.py (presentation only; no new analysis)", "estimand": "SAP 1.3, primary: weighted "
           "Aalen-Johansen cumulative incidence of the first positive delirium assessment in the ICU, ICU discharge alive "
           "and death competing, overlap weights", "hours": "cif[k] = cumulative incidence at k hours, k = 0..168",
           "at_risk_days": list(DAYS), "units": UNITS, "db": {}, "checks": []}
    good = True
    for db in DBS:
        d, flow = r15.load(P, db)
        cache = {}
        out["db"][db] = {}
        for u in UNITS:
            dd, w = unit_data(d, u, cache)
            cv, ref = curves(dd, w)
            rA, rB = cv["PPI"]["cif"][168], cv["Famotidine"]["cif"][168]
            e = sc.effects(rA, rB)
            checks = [(f"{r15.DBNAME[db]} {u}: 168-h risks equal asd.cif_contrast", max(abs(rA - ref["riskA"]), abs(rB - ref["riskB"])), 0.0, 0.0)]
            if u == "all":
                p = c2[db]["analyses"]["primary"]
                checks += [(f"{r15.DBNAME[db]} all: OR vs ppi_estimates.json", round(e["OR"], 4), p["OR"], 1e-9),
                           (f"{r15.DBNAME[db]} all: PPI risk vs ppi_estimates.json", round(rA, 3), p["riskA"], 1e-9),
                           (f"{r15.DBNAME[db]} all: famotidine risk vs ppi_estimates.json", round(rB, 3), p["riskB"], 1e-9)]
            else:
                p = j15[db]["light_primary"]
                checks += [(f"{r15.DBNAME[db]} light: OR vs E1", round(e["OR"], 4), float(j1[db]["strata"]["rass"]["light"]["OR"]), 5e-4),
                           (f"{r15.DBNAME[db]} light: OR vs r15.json", round(e["OR"], 4), p["OR"], 1e-9),
                           (f"{r15.DBNAME[db]} light: PPI risk vs r15.json", round(rA, 3), p["riskA"], 1e-9),
                           (f"{r15.DBNAME[db]} light: famotidine risk vs r15.json", round(rB, 3), p["riskB"], 1e-9)]
            checks += [(f"{r15.DBNAME[db]} {u}: {arm} at risk at day 0 equals the patients analysed", 0.0 if
                        cv[arm]["at_risk"][0] == cv[arm]["n"] else 1.0, 0.0, 0.0) for arm in cv]
            for name, x, refv, tol in checks:
                ok = abs(x - refv) <= tol + 1e-12
                good &= ok
                line = f"{'PASS' if ok else 'FAIL'} | {name}: {x:.4f} vs {refv:.4f}"
                out["checks"].append(line)
                print(line, flush=True)
            for arm in cv:
                cv[arm]["cif"] = [round(v, 6) for v in cv[arm]["cif"]]
            out["db"][db][u] = {"curves": cv, "risk168": {"PPI": round(rA, 6), "Famotidine": round(rB, 6)},
                                "OR168": round(e["OR"], 6)}
        asd.status(f"{r15.DBNAME[db]} curves done", t0)
    if not good:
        asd.log_run(R, "fig_curves: REPRODUCTION CHECK FAILED, nothing written")
        raise SystemExit("the curves do not reproduce the reported estimates: stop")
    with open(os.path.join(R, "fig_values_curves.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    asd.status("fig_values_curves.json written; reproduction checks passed", t0)
    asd.log_run(R, f"fig_curves: ok, {len(out['checks'])} reproduction checks passed, fig_values_curves.json written "
                   f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
