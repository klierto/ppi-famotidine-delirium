#!/usr/bin/env python3
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r9

KEEP = ("OR", "OR_lo", "OR_hi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    P, R = os.path.expanduser(a.private), os.path.expanduser(a.results)
    S = {}
    for db in ("n", "i"):
        f = os.path.join(P, f"r9_est_{db}.json")
        if os.path.exists(f):
            S[db] = r9.summarise(json.load(open(f))["res"], r9.load_boot(P, db))
    ext = r9.external(R)

    def item(lab, e):
        return (lab, e.get("OR") if e else None, e.get("se_logOR") if e else None)

    icu = [item("MIMIC-IV", ext["MIMIC-IV icu"]), item("eICU", ext["eICU icu"])] + [item(r9.DBNAME[d], S[d]["icu"]) for d in ("n", "i") if d in S]
    hosp = [item("MIMIC-IV", ext["MIMIC-IV hosp"])] + [item(r9.DBNAME[d], S[d]["hosp"]) for d in ("n", "i") if d in S] + \
           [item("MOVER", ext["MOVER hosp"])]
    icu_c = r9.pool_list([item("MIMIC-IV", ext["MIMIC-IV hosp"])] + ([item("NWICU", S["n"]["hosp"])] if "n" in S else []))
    surg = r9.pool_list(([item("INSPIRE", S["i"]["hosp"])] if "i" in S else []) + [item("MOVER", ext["MOVER hosp"])])
    pooled = {"icu": r9.pool_list(icu), "hosp": r9.pool_list(hosp), "hosp_icu_cohorts": icu_c, "hosp_surgical": surg}
    out = {"source": "r9.py report summaries, unrounded (fig_values_r9.py)",
           "per_db": {r9.DBNAME[d]: {n: {x: S[d][n].get(x) for x in KEEP} for n in r9.NAMES} for d in S},
           "pooled": {k: (None if p is None else {x: p[x] for x in ("OR", "OR_lo", "OR_hi", "I2")}) for k, p in pooled.items()},
           "setting_ratio": r9.setting_ratio(icu_c, surg)}
    json.dump(out, open(os.path.join(R, "fig_values_r9.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"fig_values_r9.json written ({len(S)} databases)", flush=True)


if __name__ == "__main__":
    main()
