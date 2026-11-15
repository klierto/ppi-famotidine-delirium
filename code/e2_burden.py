import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc

KEYS = asd.KEYS
OUTS = ("dcfd7", "deldays7", "dcfd7_rass3")
MCID_DAYS = 1.0

DAYS_M = """
WITH c AS (SELECT DISTINCT stay_id, hadm_id, t0, outtime FROM read_parquet('{f}') WHERE arm IS NOT NULL),
g AS (SELECT c.stay_id, c.t0, c.outtime,
             CASE WHEN a.deathtime IS NOT NULL THEN date_diff('minute', c.t0, a.deathtime) / 60.0 END AS h_death
      FROM c LEFT JOIN hosp.admissions a USING (hadm_id)),
del_d AS (SELECT DISTINCT g.stay_id, CAST(ceil(date_diff('minute', g.t0, x.charttime) / 1440.0) AS INTEGER) AS k
          FROM g JOIN icu.chartevents x ON x.stay_id = g.stay_id AND x.itemid = 228332 AND x.value = 'Positive'
           AND x.charttime > g.t0 AND x.charttime <= least(g.outtime, g.t0 + INTERVAL 168 HOUR)),
coma_d AS (SELECT DISTINCT g.stay_id, CAST(ceil(date_diff('minute', g.t0, x.charttime) / 1440.0) AS INTEGER) AS k
           FROM g JOIN icu.chartevents x ON x.stay_id = g.stay_id AND x.itemid = 228096 AND x.valuenum BETWEEN -5 AND -4
            AND x.charttime > g.t0 AND x.charttime <= least(g.outtime, g.t0 + INTERVAL 168 HOUR)),
{tail}
"""
DAYS_E = """
WITH c AS (SELECT DISTINCT stay_id, t0 FROM read_parquet('{f}') WHERE arm IS NOT NULL),
g AS (SELECT c.stay_id, c.t0, p.unitdischargeoffset AS icu_end,
             CASE WHEN p.hospitaldischargestatus = 'Expired' THEN (p.hospitaldischargeoffset - c.t0) / 60.0 END AS h_death
      FROM c JOIN eicu.patient p ON p.patientunitstayid = c.stay_id),
del_d AS (SELECT DISTINCT g.stay_id, CAST(ceil((n.nursingchartoffset - g.t0) / 1440.0) AS INTEGER) AS k
          FROM g JOIN eicu.nursecharting n ON n.patientunitstayid = g.stay_id
           AND n.nursingchartcelltypevallabel = 'Delirium Scale/Score' AND n.nursingchartcelltypevalname = 'Delirium Score'
           AND (upper(n.nursingchartvalue) = 'YES' OR TRY_CAST(n.nursingchartvalue AS INTEGER) >= 4)
           AND n.nursingchartoffset > g.t0 AND n.nursingchartoffset <= least(g.icu_end, g.t0 + 10080)),
coma_d AS (SELECT DISTINCT g.stay_id, CAST(ceil((n.nursingchartoffset - g.t0) / 1440.0) AS INTEGER) AS k
           FROM g JOIN eicu.nursecharting n ON n.patientunitstayid = g.stay_id
            AND n.nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal' AND n.nursingchartcelltypevalname = 'Sedation Score'
            AND TRY_CAST(n.nursingchartvalue AS INTEGER) BETWEEN -5 AND -4
            AND n.nursingchartoffset > g.t0 AND n.nursingchartoffset <= least(g.icu_end, g.t0 + 10080)),
{tail}
"""
TAIL = """
grid AS (SELECT g.stay_id, g.h_death, r.k FROM g CROSS JOIN (SELECT range AS k FROM range(1, 8)) r),
lab AS (SELECT grid.stay_id, grid.k, grid.h_death, d.k IS NOT NULL AS del, m.k IS NOT NULL AS coma
        FROM grid LEFT JOIN del_d d ON d.stay_id = grid.stay_id AND d.k = grid.k
                  LEFT JOIN coma_d m ON m.stay_id = grid.stay_id AND m.k = grid.k)
SELECT stay_id,
       sum(CASE WHEN (h_death IS NULL OR h_death > 24 * k) AND NOT del AND NOT coma THEN 1 ELSE 0 END) AS dcfd7,
       sum(CASE WHEN del THEN 1 ELSE 0 END) AS deldays7
FROM lab GROUP BY stay_id
"""


def extract(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    con = asd.connect_slim(P, args.db)
    for comparison in (1, 2):
        f = os.path.join(P, f"{args.db}_c{comparison}_cohort.parquet")
        sql = (DAYS_M if args.db == "m" else DAYS_E).format(f=f, tail=TAIL)
        asd.write_pq(con, sql, os.path.join(P, f"{args.db}_c{comparison}_dcfd.parquet"))
        asd.status(f"{args.db}_c{comparison}_dcfd.parquet written", t0)
    asd.log_run(args.results, f"e2_burden extract --db {args.db}: ok ({time.time() - t0:.0f} s)")


def load_arrays(P, key):
    comparison, db = int(key[1]), key[2]
    df, flow = asd.load_key(P, key, stage2=True)
    dc = asd.read_pq(os.path.join(P, f"{db}_c{comparison}_dcfd.parquet"))
    df = df.merge(dc, on="stay_id", how="left", validate="one_to_one")
    if df["dcfd7"].isna().any():
        raise SystemExit(f"{key}: stays without DCFD row")
    d = asd.build(df, db, comparison, stage2=True)
    d["dcfd7"] = df["dcfd7"].astype(float).values
    d["deldays7"] = df["deldays7"].astype(float).values
    return d, flow


def wmeans(A, y, w):
    mA = float((w * y)[A == 1].sum() / w[A == 1].sum())
    mB = float((w * y)[A == 0].sum() / w[A == 0].sum())
    return {"mA": mA, "mB": mB, "diff": mA - mB}


def estimates(d, cache):
    A = d["A"]
    ow = sc.overlap_w(asd.ow_fit(d, cache, "full"), A)
    out = {"dcfd7": wmeans(A, d["dcfd7"], ow), "deldays7": wmeans(A, d["deldays7"], ow)}
    m = d["sub"]["rass3"]
    ds = asd.subset_d(d, m)
    ows = sc.overlap_w(asd.ow_fit(ds, cache, "rass3"), ds["A"])
    out["dcfd7_rass3"] = wmeans(ds["A"], ds["dcfd7"], ows)
    out["n"] = {"A": int((A == 1).sum()), "B": int((A == 0).sum()),
                "A_rass3": int((ds["A"] == 1).sum()), "B_rass3": int((ds["A"] == 0).sum())}
    out["crude"] = {o: {"mA": float(d[o][A == 1].mean()), "mB": float(d[o][A == 0].mean())} for o in ("dcfd7", "deldays7")}
    return out


def estimate(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    for key in KEYS:
        d, flow = load_arrays(P, key)
        cache = {}
        est = estimates(d, cache)
        json.dump({"flow": flow, "est": est}, open(os.path.join(P, f"e2_est_{key}.json"), "w"))
        np.savez(os.path.join(P, f"e2_cache_{key}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
        asd.status(f"e2 estimate {key} done", t0)
    asd.log_run(args.results, f"e2_burden estimate: ok ({time.time() - t0:.0f} s)")


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    d, _ = load_arrays(P, args.key)
    z = np.load(os.path.join(P, f"e2_cache_{args.key}.npz"))
    cache0 = {k.replace("__", ":"): z[k] for k in z.files}
    rows = []
    for rep in range(args.start, args.stop):
        rng = np.random.default_rng(asd.SEED + rep)
        b = asd.resample(d, args.key[2], rng)
        try:
            e = estimates(b, dict(cache0))
            rows.append([rep] + [e[o]["diff"] for o in OUTS])
        except Exception:
            rows.append([rep] + [np.nan] * len(OUTS))
    np.save(os.path.join(P, f"e2_boot_{args.key}_{args.start:04d}_{args.stop:04d}.npy"), np.array(rows, dtype=float))
    asd.status(f"e2 boot {args.key} {args.start}-{args.stop} done", t0)
    asd.log_run(args.results, f"e2_burden boot --key {args.key} --start {args.start} --stop {args.stop}: ok ({time.time() - t0:.0f} s)")


def load_boot(P, key):
    parts = sorted(glob.glob(os.path.join(P, f"e2_boot_{key}_*.npy")))
    if not parts:
        return None
    B = np.concatenate([np.load(p) for p in parts])
    _, idx = np.unique(B[:, 0], return_index=True)
    return B[idx]


def evalue(or_, lo, hi):
    def ev(o):
        rr = math.sqrt(o)
        rr = 1 / rr if rr < 1 else rr
        return rr + math.sqrt(rr * (rr - 1))
    lim = 1.0 if lo <= 1 <= hi else ev(hi if or_ < 1 else lo)
    return ev(or_), lim


def evalue_inputs(results):
    f = os.path.join(results, "ppi_estimates.json")
    if not os.path.exists(f):
        return []
    j = json.load(open(f, encoding="utf-8"))
    rows = [("PPI comparison main analysis MIMIC-IV", j["per_db"]["m"]["analyses"]["primary"]), ("PPI comparison main analysis eICU", j["per_db"]["e"]["analyses"]["primary"]),
            ("PPI comparison main analysis pooled", j["pooled"]["primary"]), ("PPI comparison RASS ≥ −3 at time zero, pooled", j["pooled"]["rass3"])]
    return [(name, e["OR"], e["OR_lo"], e["OR_hi"]) for name, e in rows if e.get("OR_lo") is not None]


LABEL = {"dcfd7": "E2a days alive without delirium or coma within 7 days", "deldays7": "E2b ICU days with delirium within 7 days",
         "dcfd7_rass3": "E2c E2a, RASS ≥ −3 at time zero"}
GROUPS = {1: ("≤20 mg/day", "≥40 mg/day"), 2: ("PPI", "famotidine")}


def report(args):
    P = os.path.expanduser(args.private)
    res = {"famdose": {}, "ppi": {}, "evalue": []}
    for key in KEYS:
        comparison = int(key[1])
        j = json.load(open(os.path.join(P, f"e2_est_{key}.json")))
        B = load_boot(P, key)
        row = {"flow": {k: sc.mask_n(v) for k, v in j["flow"].items()},
               "n": {k: sc.mask_n(v) for k, v in j["est"]["n"].items()}, "boot_ok": int(np.isfinite(B[:, 1]).sum()) if B is not None else 0}
        for i, o in enumerate(OUTS):
            e = dict(j["est"][o])
            if B is not None:
                col = B[:, i + 1]
                col = col[np.isfinite(col)]
                if col.size >= asd.MIN_BOOT:
                    e["lo"], e["hi"], e["se"] = float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5)), float(col.std(ddof=1))
            row[o] = e
        res[asd.STEM[comparison]][key[2]] = row
    for comparison in (1, 2):
        pooled = {}
        for o in OUTS:
            m, e = res[asd.STEM[comparison]].get("m", {}).get(o, {}), res[asd.STEM[comparison]].get("e", {}).get(o, {})
            if "se" in m and "se" in e and m["se"] > 0 and e["se"] > 0:
                w = np.array([1 / m["se"] ** 2, 1 / e["se"] ** 2])
                x = np.array([m["diff"], e["diff"]])
                est = float((w * x).sum() / w.sum())
                se = float(1 / math.sqrt(w.sum()))
                q = float((w * (x - est) ** 2).sum())
                i2 = max(0.0, (q - 1) / q) if q > 0 else 0.0
                pooled[o] = {"diff": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se, "I2": i2,
                             "same_sign": bool(np.sign(x[0]) == np.sign(x[1]))}
        res[asd.STEM[comparison]]["pooled"] = pooled
    res["boot_reps"] = min(res[asd.STEM[int(k[1])]][k[2]]["boot_ok"] for k in KEYS)
    for name, o, lo, hi in evalue_inputs(args.results):
        p, lim = evalue(o, lo, hi)
        res["evalue"].append({"analysis": name, "OR": o, "lo": lo, "hi": hi, "E_point": p, "E_limit": lim})
    json.dump(res, open(os.path.join(args.results, "e2_burden.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(args.results, "e2_burden.md"), "w", encoding="utf-8").write(md(res))
    asd.log_run(args.results, "e2_burden report: ok")
    print("report written", flush=True)


def verdict(p, o):
    if not p:
        return "cannot be pooled"
    agree = "databases agree" if (p["I2"] <= 0.5 and p["same_sign"]) else "databases disagree"
    if o == "deldays7":
        return agree
    if p["lo"] >= MCID_DAYS:
        v = "clinically important benefit"
    elif p["lo"] > 0:
        v = "direction of benefit" + (", clinically important benefit excluded" if p["hi"] < MCID_DAYS else "")
    elif p["hi"] < MCID_DAYS:
        v = "clinically important benefit excluded" + (" (interval includes 0)" if p["lo"] <= 0 <= p["hi"] else "")
    else:
        v = "inconclusive"
    return f"{agree}; {v}"


def md(res):
    L = ["# E2 results: delirium and coma burden (MIMIC-IV, eICU) and E-values", "",
         f"Plan: SAP_addendum_E2.md v1.0 (revision R6). Overlap weighting; intervals: percentile intervals of {res.get('boot_reps')} bootstrap replicates (MIMIC-IV by patient, "
         "eICU by hospital); pooling by fixed effect. E-values from the estimates in this ppi_estimates.json. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", ""]
    for comparison in (1, 2):
        ga, gb = GROUPS[comparison]
        L += [f"## {asd.TITLE[comparison]} (A = {ga}, B = {gb})", "",
              "| Outcome | Database | Patients A / B | Weighted mean A / B (days) | Difference A − B (95% CI) |", "|---|---|---|---|---|"]
        for o in OUTS:
            for db, nm in (("m", "MIMIC-IV"), ("e", "eICU")):
                r = res[asd.STEM[comparison]].get(db)
                if not r:
                    continue
                e = r[o]
                n = (r["n"]["A_rass3"], r["n"]["B_rass3"]) if o == "dcfd7_rass3" else (r["n"]["A"], r["n"]["B"])
                ci = f"{e['diff']:+.2f} ({e['lo']:+.2f} to {e['hi']:+.2f})" if "lo" in e else f"{e['diff']:+.2f}"
                L.append(f"| {LABEL[o]} | {nm} | {n[0]} / {n[1]} | {e['mA']:.2f} / {e['mB']:.2f} | {ci} |")
            p = res[asd.STEM[comparison]]["pooled"].get(o)
            if p:
                L.append(f"| {LABEL[o]} | Pooled | | | {p['diff']:+.2f} ({p['lo']:+.2f} to {p['hi']:+.2f}); I² {100 * p['I2']:.0f}% |")
        L += [""] + [f"- {LABEL[o]}: {verdict(res[asd.STEM[comparison]]['pooled'].get(o), o)}" for o in OUTS] + [""]
    L += ["## E-values (PPI comparison)", "", "| Analysis | OR (95% CI) | E-value (point estimate) | E-value (confidence limit) |", "|---|---|---|---|"]
    for r in res["evalue"]:
        L.append(f"| {r['analysis']} | {r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}) | {r['E_point']:.2f} | {r['E_limit']:.2f} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("extract")
    s.add_argument("--db", choices=["m", "e"], required=True)
    for s in [s] + [sp.add_parser(c) for c in ("estimate", "report")]:
        s.add_argument("--private", required=True)
        s.add_argument("--results", required=True)
    s = sp.add_parser("boot")
    s.add_argument("--key", choices=KEYS, required=True)
    s.add_argument("--start", type=int, required=True)
    s.add_argument("--stop", type=int, required=True)
    s.add_argument("--private", required=True)
    s.add_argument("--results", required=True)
    a = ap.parse_args()
    {"extract": extract, "estimate": estimate, "boot": boot, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
