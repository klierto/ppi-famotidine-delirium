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
import r7
import r16

DEL = {"m": "\n    AND NOT EXISTS (SELECT 1 FROM cam c WHERE c.stay_id = f.stay_id AND c.charttime <= f.t0)",
       "e": "\n    AND NOT EXISTS (SELECT 1 FROM dsc d WHERE d.patientunitstayid = p.patientunitstayid AND d.pos AND d.t <= f.drugstartoffset)"}
DBNAME = {"m": "MIMIC-IV", "e": "eICU"}


def nodel_body(db):
    body = asd.gate_body(2, db)
    if body.count(DEL[db]) != 1:
        raise SystemExit(f"{db}: the delirium criterion was not found exactly once in the gate SQL")
    return body.replace(DEL[db], "")


def connect(root, P, db):
    con = r16.connect_full(root, P)
    con.execute("CREATE SCHEMA IF NOT EXISTS src")
    for tb, src in ((("emar", "hosp.emar"), ("inputevents", "icu.inputevents")) if db == "m" else (("medication", "eicu.medication"),)):
        con.execute(f"CREATE OR REPLACE VIEW src.{tb} AS SELECT * FROM {src}")
    return con


def extract(P, db, root):
    t0 = time.time()
    con = connect(root, P, db)
    pf = r16.pf
    nd = pf(P, db, "nodel_cohort")
    st = [("WITH " + nodel_body(db) + ",\nq_out AS (\n" + asd.COHORT_SELECT[(db, 2)] + "\n)\nSELECT * FROM q_out", nd)]
    files = asd.plist([nd])
    cov, fol, out = (asd.M_COVARS, asd.M_FOLLOW, asd.M_OUTCOME) if db == "m" else (asd.E_COVARS, asd.E_FOLLOW, asd.E_OUTCOME)
    st += [(cov.format(files=files), pf(P, db, "nodel_covars")), (fol.format(files=files), pf(P, db, "nodel_follow")),
           (out.format(files=files), pf(P, db, "nodel_outcome"))]
    for name, cohort in (("main", pf(P, db, "c2_cohort")), ("nodel", nd)):
        f = asd.plist([cohort])
        st += [((r7.M_COV if db == "m" else r7.E_COV).format(files=f), pf(P, db, f"a16_{name}_cov")),
               ((r7.M_OUT if db == "m" else r7.E_OUT).format(files=f), pf(P, db, f"a16_{name}_out"))]
    for sql, path in st:
        asd.write_pq(con, sql, path)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        asd.status(f"{os.path.basename(path)} written: {sc.mask_n(n)} rows (masked)", t0)
    con.close()


def load(P, db, name):
    pf = r16.pf
    if name == "main":
        df, flow = asd.load_key(P, "c2" + db, stage2=True)
    else:
        co = asd.read_pq(pf(P, db, "nodel_cohort"))
        flow = {"eligible": len(co), "A": int((co["arm"] == "A").sum()), "B": int((co["arm"] == "B").sum()), "other": int(co["arm"].isna().sum())}
        df = co[co["arm"].isin(["A", "B"])].copy()
        for part in ("covars", "follow", "outcome"):
            df = df.merge(asd.read_pq(pf(P, db, f"nodel_{part}")), on="stay_id", how="left", validate="one_to_one")
        df["t_dev"] = np.nan
        keep = asd.overlap_keep(df, db == "e")
        flow["outside_overlap"] = int((~keep).sum())
        df = df[keep].sort_values("stay_id", kind="mergesort").reset_index(drop=True)
    n0 = len(df)
    for part in ("cov", "out"):
        df = df.merge(asd.read_pq(pf(P, db, f"a16_{name}_{part}")), on="stay_id", how="left", validate="one_to_one")
    if len(df) != n0:
        raise SystemExit(f"{db} {name}: rows changed in the merge")
    d = asd.build(df, db, 2, stage2=True)
    for c in (r7.PROXY_M if db == "m" else r7.PROXY_E):
        d[c] = asd.num(df, c)
    d["ap_pre"], d["psych"] = r7.nz(df["ap_pre"]) == 1, r7.nz(df["psych"]) == 1
    if db == "m":
        d["end_death_h"] = r7.nz(df["end_death_h"]) == 1
    d["sid"] = df["stay_id"].values
    return d, flow


def est(d, db, cache):
    ds = asd.subset_d(d, ~d["ap_pre"] & ~d["psych"])
    w = sc.overlap_w(asd.ow_fit(ds, cache, "px"), ds["A"])
    res = {"px_icu": asd.cif_contrast(ds, w, ds["t_ap_icu"])[168]}
    if db == "m":
        res["px_hosp"] = r7.hosp_contrast(ds, w, ds["t_ap_hosp"])
    return res


def run(d, db):
    cache = {}
    pt = est(d, db, cache)
    names = list(pt)
    V = np.full((r16.NBOOT, 2 * len(names)), np.nan)
    failed = 0
    for rep in range(r16.NBOOT):
        try:
            r = est(asd.resample(d, db, np.random.default_rng(asd.SEED + rep)), db, dict(cache))
            V[rep] = [x for nm in names for x in (r[nm]["riskA"], r[nm]["riskB"])]
        except Exception:
            failed += 1
    B = {"cols": [f"{nm}:{k}" for nm in names for k in ("A", "B")], "v": V}
    return {nm: r16.risk_entry(pt[nm], B, nm) for nm in names}, failed


def find_key(o, key):
    if isinstance(o, dict):
        if key in o and isinstance(o[key], dict):
            return o[key]
        for v in o.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--eicu", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    t0 = time.time()
    P = a.private
    j7 = json.load(open(os.path.join(a.results, "r7.json"), encoding="utf-8"))
    out = {"plan": "SAP_addendum_R16.md v1.0, section 16 (A16)", "checks": [], "db": {}}
    good = True
    for db, root in (("m", a.mimic), ("e", a.eicu)):
        extract(P, db, root)
        res = {}
        dm, _ = load(P, db, "main")
        main_ids = set(dm["sid"].tolist())
        for name in ("main", "nodel"):
            d, flow = (dm, None) if name == "main" else load(P, db, "nodel")
            e, failed = run(d, db)
            A = d["A"] == 1
            px = ~d["ap_pre"] & ~d["psych"]
            prev = ~np.isin(d["sid"], list(main_ids))
            desc = {}
            for lab, r in (("A", A), ("B", ~A)):
                desc[lab] = {"n": r16.cnt(r.sum()), "positive_screen_before_first_dose": r16.share(prev, r), "earlier_antipsychotic": r16.share(d["ap_pre"], r),
                             "psychotic_or_bipolar_disorder": r16.share(d["psych"], r), "in_antipsychotic_analysis": r16.share(px, r),
                             "positive_before_among_analysed": r16.share(prev & px, r & px)}
            res[name] = {"est": e, "boot_failed": failed, "desc": desc}
            asd.status(f"{DBNAME[db]} {name} done", t0)
        ref = find_key(j7, "c2" + db) or {}
        for nm in res["main"]["est"]:
            x, r = res["main"]["est"][nm], find_key(ref, nm)
            ok = r is not None and abs(round(x["OR"], 4) - r["OR"]) < 1e-9
            lo = r.get("OR_lo", r.get("lo")) if r else None
            hi = r.get("OR_hi", r.get("hi")) if r else None
            ok &= lo is not None and abs(round(x["OR_lo"], 4) - lo) < 1e-9 and abs(round(x["OR_hi"], 4) - hi) < 1e-9
            good &= bool(ok)
            line = f"{'PASS' if ok else 'FAIL'} | {DBNAME[db]} {nm} in the main cohort vs R7: {x['OR']:.4f} ({x['OR_lo']:.4f}-{x['OR_hi']:.4f}) vs " \
                   + (f"{r['OR']:.4f} ({lo}-{hi})" if r else "not found in r7.json")
            out["checks"].append(line)
            print(line, flush=True)
        out["db"][db] = res
    if not good:
        raise SystemExit("R7's antipsychotic estimates were not reproduced: stop (nothing written)")
    json.dump(r16.round_tree(asd.to_json(out)), open(os.path.join(a.results, "r16_a16.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for db in ("m", "e"):
        for name in ("main", "nodel"):
            for nm, e in out["db"][db][name]["est"].items():
                print(f"{DBNAME[db]} {name} {nm}: n {e['nA']}/{e['nB']}; events {e['eventsA']}/{e['eventsB']}; risks {100 * e['riskA']:.1f}%/{100 * e['riskB']:.1f}%; "
                      f"OR {e['OR']:.2f} ({e['OR_lo']:.2f}-{e['OR_hi']:.2f}); RR {e['RR']:.2f} ({e['RR_lo']:.2f}-{e['RR_hi']:.2f}); RD {100 * e['RD']:.1f} ({100 * e['RD_lo']:.1f} to {100 * e['RD_hi']:.1f})")
            print(f"   {name} description:", json.dumps(out["db"][db][name]["desc"], ensure_ascii=False))
    asd.log_run(a.results, f"r16_a16: ok ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
