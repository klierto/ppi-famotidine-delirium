#!/usr/bin/env python3
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import glob
import io
import json
import math
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc

DBS = ("m", "e")
NBOOT = 2000
MIN_N = 20

PROC_M = """
WITH c AS (SELECT stay_id, t0, outtime FROM read_parquet('{f}') WHERE arm IS NOT NULL),
fu AS (SELECT stay_id, t0, least(outtime, t0 + INTERVAL 168 HOUR) AS t_end FROM c),
da AS (SELECT stay_id, charttime, value FROM icu.chartevents WHERE itemid = 228332 AND value IS NOT NULL),
dt AS (SELECT f.stay_id, date_diff('minute', f.t0, d.charttime) / 60.0 AS h,
              max(CASE WHEN d.value = 'Positive' THEN 1 ELSE 0 END) AS pos,
              max(CASE WHEN d.value IN ('Positive', 'Negative') THEN 1 ELSE 0 END) AS ok,
              0 AS pos_icdsc
       FROM fu f JOIN da d ON d.stay_id = f.stay_id AND d.charttime > f.t0 AND d.charttime <= f.t_end
       GROUP BY f.stay_id, f.t0, d.charttime),
base AS (SELECT c.stay_id, count(DISTINCT d.charttime) AS n_base FROM c JOIN da d
         ON d.stay_id = c.stay_id AND d.charttime BETWEEN c.t0 - INTERVAL 24 HOUR AND c.t0 GROUP BY 1),
ra AS (SELECT f.stay_id, count(*) AS n_rass72, sum(CASE WHEN r.valuenum <= -4 THEN 1 ELSE 0 END) AS n_deep72,
              max(CASE WHEN r.valuenum <= -4 AND r.charttime <= f.t0 + INTERVAL 24 HOUR THEN 1 ELSE 0 END) AS deep24
       FROM fu f JOIN icu.chartevents r ON r.stay_id = f.stay_id AND r.itemid = 228096 AND r.charttime > f.t0
        AND r.charttime <= least(f.t_end, f.t0 + INTERVAL 72 HOUR)
       WHERE r.valuenum BETWEEN -5 AND 4 GROUP BY 1),
win AS (SELECT stay_id, date_diff('minute', t0, t_end) / 60.0 AS h_window FROM fu)
{tail}
"""

PROC_E = """
WITH c AS (SELECT stay_id, t0 FROM read_parquet('{f}') WHERE arm IS NOT NULL),
fu AS (SELECT c.stay_id, c.t0, least(p.unitdischargeoffset, c.t0 + 10080) AS t_end FROM c JOIN eicu.patient p ON p.patientunitstayid = c.stay_id),
da AS (SELECT patientunitstayid AS stay_id, nursingchartoffset AS t, nursingchartvalue AS v FROM eicu.nursecharting
       WHERE nursingchartcelltypevallabel = 'Delirium Scale/Score' AND nursingchartcelltypevalname = 'Delirium Score'
         AND NULLIF(nursingchartvalue, '') IS NOT NULL),
dt AS (SELECT f.stay_id, (d.t - f.t0) / 60.0 AS h,
              max(CASE WHEN upper(d.v) = 'YES' OR TRY_CAST(d.v AS INTEGER) >= 4 THEN 1 ELSE 0 END) AS pos,
              max(CASE WHEN upper(d.v) IN ('YES', 'NO') OR TRY_CAST(d.v AS INTEGER) IS NOT NULL THEN 1 ELSE 0 END) AS ok,
              max(CASE WHEN TRY_CAST(d.v AS INTEGER) >= 4 THEN 1 ELSE 0 END) AS pos_icdsc
       FROM fu f JOIN da d ON d.stay_id = f.stay_id AND d.t > f.t0 AND d.t <= f.t_end GROUP BY f.stay_id, f.t0, d.t),
base AS (SELECT c.stay_id, count(DISTINCT d.t) AS n_base FROM c JOIN da d ON d.stay_id = c.stay_id AND d.t BETWEEN c.t0 - 1440 AND c.t0 GROUP BY 1),
ra AS (SELECT f.stay_id, count(*) AS n_rass72, sum(CASE WHEN TRY_CAST(n.nursingchartvalue AS INTEGER) <= -4 THEN 1 ELSE 0 END) AS n_deep72,
              max(CASE WHEN TRY_CAST(n.nursingchartvalue AS INTEGER) <= -4 AND n.nursingchartoffset <= f.t0 + 1440 THEN 1 ELSE 0 END) AS deep24
       FROM fu f JOIN eicu.nursecharting n ON n.patientunitstayid = f.stay_id AND n.nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal'
        AND n.nursingchartcelltypevalname = 'Sedation Score' AND n.nursingchartoffset > f.t0 AND n.nursingchartoffset <= least(f.t_end, f.t0 + 4320)
       WHERE TRY_CAST(n.nursingchartvalue AS INTEGER) BETWEEN -5 AND 4 GROUP BY 1),
win AS (SELECT stay_id, (t_end - t0) / 60.0 AS h_window FROM fu)
{tail}
"""

PROC_TAIL = """,
fp AS (SELECT stay_id, min(h) FILTER (WHERE pos = 1) AS h_pos FROM dt GROUP BY 1),
fpi AS (SELECT d.stay_id, max(d.pos_icdsc) AS first_pos_icdsc FROM dt d JOIN fp p USING (stay_id) WHERE d.h = p.h_pos GROUP BY 1),
agg AS (SELECT d.stay_id, count(*) AS n_any, sum(d.ok) AS n_ok, min(d.h) AS h_first,
               count(DISTINCT CAST(ceil(d.h / 24.0) AS INTEGER)) AS blocks_covered,
               sum(CASE WHEN d.ok = 1 AND (p.h_pos IS NULL OR d.h <= p.h_pos) THEN 1 ELSE 0 END) AS n_ok_until_pos
        FROM dt d JOIN fp p USING (stay_id) GROUP BY 1)
SELECT w.stay_id, w.h_window, coalesce(a.n_any, 0) AS n_any, coalesce(a.n_ok, 0) AS n_ok, a.h_first,
       coalesce(a.blocks_covered, 0) AS blocks_covered, coalesce(a.n_ok_until_pos, 0) AS n_ok_until_pos, p.h_pos, fpi.first_pos_icdsc,
       coalesce(b.n_base, 0) AS n_base, coalesce(r.n_rass72, 0) AS n_rass72, coalesce(r.n_deep72, 0) AS n_deep72, coalesce(r.deep24, 0) AS deep24
FROM win w LEFT JOIN agg a USING (stay_id) LEFT JOIN fp p USING (stay_id) LEFT JOIN fpi USING (stay_id) LEFT JOIN base b USING (stay_id)
LEFT JOIN ra r USING (stay_id)
"""

HOSP_E = """
WITH pt AS (
  SELECT p.patientunitstayid, p.hospitalid, p.unitdischargeoffset
  FROM eicu.patient p
  WHERE TRY_CAST(IF(p.age = '> 89', '90', p.age) AS BIGINT) >= 18
  QUALIFY ROW_NUMBER() OVER (PARTITION BY p.uniquepid ORDER BY p.hospitaldischargeyear, p.patienthealthsystemstayid, p.unitvisitnumber, p.patientunitstayid) = 1
),
dsc AS (SELECT patientunitstayid, nursingchartoffset AS t FROM eicu.nursecharting
        WHERE nursingchartcelltypevallabel = 'Delirium Scale/Score' AND nursingchartcelltypevalname = 'Delirium Score'
          AND NULLIF(nursingchartvalue, '') IS NOT NULL),
dl_hosp AS (SELECT hospitalid FROM pt GROUP BY 1
            HAVING count(*) >= 50 AND count(*) FILTER (WHERE patientunitstayid IN (SELECT patientunitstayid FROM dsc)) >= 0.5 * count(*)),
st AS (SELECT patientunitstayid, hospitalid, greatest(unitdischargeoffset, 1) AS los FROM pt WHERE hospitalid IN (SELECT hospitalid FROM dl_hosp)),
cov AS (SELECT s.patientunitstayid, count(DISTINCT CAST(floor(d.t / 1440.0) AS INTEGER)) AS blocks FROM st s JOIN dsc d
        ON d.patientunitstayid = s.patientunitstayid AND d.t >= 0 AND d.t < s.los GROUP BY 1)
SELECT s.hospitalid, sum(coalesce(c.blocks, 0)) AS blocks_covered, sum(CAST(ceil(s.los / 1440.0) AS BIGINT)) AS blocks_total, count(*) AS stays
FROM st s LEFT JOIN cov c USING (patientunitstayid) GROUP BY 1
"""


def proc_sql(db, P):
    f = os.path.join(P, f"{db}_c2_cohort.parquet")
    return (PROC_M if db == "m" else PROC_E).format(f=f, tail=PROC_TAIL)


def extract(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    db = "m" if args.db == "mimic" else "e"
    con = asd.connect_slim(P, db)
    asd.write_pq(con, proc_sql(db, P), os.path.join(P, f"{db}_c2_process.parquet"))
    asd.status(f"{db}_c2_process.parquet written", t0)
    if db == "e":
        asd.write_pq(con, HOSP_E, os.path.join(P, "e_hosp_intensity.parquet"))
        asd.status("e_hosp_intensity.parquet written", t0)
    asd.log_run(args.results, f"explore_ppi extract --db {args.db}: ok ({time.time() - t0:.0f} s)")


STRATA = {
    "rass": ("RASS at time zero", ["light", "moderate", "deep", "missing"], {"light": "awake or light (≥−1)", "moderate": "moderate (−3 to −2)",
                                                                       "deep": "deep (≤−4)", "missing": "not recorded"}),
    "base": ("delirium assessment in the 24 h before time zero", ["yes", "no"], {"yes": "yes", "no": "no"}),
    "surg": ("surgical", ["no", "yes"], {"no": "non-surgical", "yes": "surgical"}),
    "vent": ("invasive ventilation at time zero", ["no", "yes"], {"no": "no", "yes": "yes"}),
    "era": ("calendar period (MIMIC-IV)", ["early", "late"], {"early": "2011–2016", "late": "2017–2022"}),
    "hosp": ("hospital delirium-assessment intensity (eICU)", ["high", "mid", "low"], {"high": "high", "mid": "mid", "low": "low"}),
}
LEVEL_CODE = {s: {lv: i for i, lv in enumerate(v[1])} for s, v in STRATA.items()}


def load(P, db, hosp_cut=None):
    key = "c2" + db
    df, flow = asd.load_key(P, key, stage2=True)
    pr = asd.read_pq(os.path.join(P, f"{db}_c2_process.parquet"))
    df = df.merge(pr, on="stay_id", how="left", validate="one_to_one")
    d = asd.build(df, db, 2, stage2=True)
    num = lambda c: asd.num(df, c)
    for c in ("h_window", "n_any", "n_ok", "h_first", "blocks_covered", "n_ok_until_pos", "first_pos_icdsc", "n_base", "n_rass72", "n_deep72",
              "deep24"):
        d["p_" + c] = num(c)
    r = num("rass")
    rg = np.where(~np.isfinite(r), "missing", np.where(r <= -4, "deep", np.where(r <= -2, "moderate", "light")))
    d["s_rass"] = np.array([LEVEL_CODE["rass"][x] for x in rg])
    d["s_base"] = np.where(num("n_base") > 0, LEVEL_CODE["base"]["yes"], LEVEL_CODE["base"]["no"])
    surg = df["service"].astype(str).isin(asd.M_SURG if db == "m" else asd.E_SURG).values
    d["s_surg"] = np.where(surg, LEVEL_CODE["surg"]["yes"], LEVEL_CODE["surg"]["no"])
    d["s_vent"] = np.where(np.nan_to_num(num("vent")) == 1, LEVEL_CODE["vent"]["yes"], LEVEL_CODE["vent"]["no"])
    if db == "m":
        late = df["era"].astype(str).isin(["2017 - 2019", "2020 - 2022"]).values
        d["s_era"] = np.where(late, LEVEL_CODE["era"]["late"], LEVEL_CODE["era"]["early"])
    else:
        hi = asd.read_pq(os.path.join(P, "e_hosp_intensity.parquet"))
        hi["intensity"] = hi["blocks_covered"] / hi["blocks_total"]
        inten = dict(zip(hi["hospitalid"].astype(float), hi["intensity"]))
        sites = np.unique(d["site"])
        vals = np.array([inten.get(s, np.nan) for s in sites])
        lo, up = np.nanpercentile(vals, [100 / 3, 200 / 3])
        lvl = {s: ("low" if v <= lo else ("high" if v >= up else "mid")) for s, v in zip(sites, vals)}
        d["s_hosp"] = np.array([LEVEL_CODE["hosp"][lvl[s]] for s in d["site"]])
        d["hosp_cuts"] = (float(lo), float(up))
        d["hosp_n"] = {k: int(sum(1 for v in lvl.values() if v == k)) for k in ("high", "mid", "low")}
    return d, flow


def strata_for(db):
    return ["rass", "base", "surg", "vent", "era" if db == "m" else "hosp"]


def ow_of(d):
    ps, _ = sc.ps_fit(d["X"], d["A"])
    return sc.overlap_w(ps, d["A"])


def primary(d):
    return asd.cif_contrast(d, ow_of(d), d["t_del"])[168]


def stratum_effects(d, db):
    out = {}
    A = d["A"]
    for s in strata_for(db):
        out[s] = {}
        for lv, code in LEVEL_CODE[s].items():
            m = d["s_" + s] == code
            nA, nB = int((m & (A == 1)).sum()), int((m & (A == 0)).sum())
            if min(nA, nB) < MIN_N:
                out[s][lv] = {"nA": nA, "nB": nB, "OR": float("nan")}
                continue
            e = primary(asd.subset_d(d, m))
            e["nA"], e["nB"] = nA, nB
            out[s][lv] = e
    return out


def standardised(strat, shares):
    rA = rB = tot = 0.0
    for lv, sh in shares.items():
        e = strat["rass"].get(lv, {})
        if not np.isfinite(e.get("OR", float("nan"))):
            continue
        rA += sh * e["riskA"]
        rB += sh * e["riskB"]
        tot += sh
    if tot <= 0:
        return float("nan")
    rA, rB = rA / tot, rB / tot
    return sc.effects(rA, rB)["OR"]


def rass_shares(d):
    return {lv: float((d["s_rass"] == code).mean()) for lv, code in LEVEL_CODE["rass"].items()}


def discharge(d, w):
    t_ev = np.where(~d["end_death"] & (d["t_end"] < 168), d["t_end"], np.nan)
    kend, typ, _ = sc.follow_types(d["t_end"], d["end_death"], t_ev, disc_competing=False)
    A = d["A"]
    r = {a: sc.aj_cif(w[A == a], kend[A == a], typ[A == a]) for a in (1, 0)}
    e = sc.effects(r[1][167], r[0][167])
    e["riskA72"], e["riskB72"] = float(r[1][71]), float(r[0][71])
    e["eventsA"], e["eventsB"] = int(((typ == 1) & (A == 1)).sum()), int(((typ == 1) & (A == 0)).sum())
    return e


def per_assessment(d, w):
    ev = np.isfinite(d["t_del"]).astype(float)
    den = np.nan_to_num(d["p_n_ok_until_pos"])
    A = d["A"]
    ra = (w * ev)[A == 1].sum() / max((w * den)[A == 1].sum(), 1e-12)
    rb = (w * ev)[A == 0].sum() / max((w * den)[A == 0].sum(), 1e-12)
    return {"rateA": float(ra), "rateB": float(rb), "RR": float(ra / rb) if rb > 0 else float("nan")}


def describe(d, w):
    A = d["A"]
    hw = np.maximum(np.nan_to_num(d["p_h_window"]), 1.0)
    blocks = np.ceil(hw / 24.0)
    n_any, n_ok = np.nan_to_num(d["p_n_any"]), np.nan_to_num(d["p_n_ok"])
    per_day = n_any / (hw / 24.0)
    cov = np.minimum(np.nan_to_num(d["p_blocks_covered"]) / blocks, 1.0)
    out = {}
    for arm, a in (("A", 1), ("B", 0)):
        r = A == a
        res = {"n": int(r.sum())}
        for lab, ww in (("w", w[r]), ("u", np.ones(int(r.sum())))):
            res[lab] = {
                "assess_per_icu_day": float(np.average(per_day[r], weights=ww)),
                "share_day_blocks_assessed": float(np.average(cov[r], weights=ww)),
                "share_with_any_assessment": float(np.average(n_any[r] > 0, weights=ww)),
                "share_unable_to_assess": float((ww * (n_any[r] - n_ok[r])).sum() / max((ww * n_any[r]).sum(), 1e-12)),
                "share_assessed_24h_before_t0": float(np.average(np.nan_to_num(d["p_n_base"])[r] > 0, weights=ww)),
                "share_rass_le_m4_0_72h": float((ww * np.nan_to_num(d["p_n_deep72"])[r]).sum() / max((ww * np.nan_to_num(d["p_n_rass72"])[r]).sum(), 1e-12)),
                "share_any_rass_le_m4_0_24h": float(np.average(np.nan_to_num(d["p_deep24"])[r] > 0, weights=ww)),
            }
        hf = d["p_h_first"][r]
        hf = hf[np.isfinite(hf)]
        res["hours_to_first_assessment_q25_50_75"] = [round(float(x), 1) for x in np.percentile(hf, [25, 50, 75])] if hf.size else []
        res["n_with_any_assessment"] = int((n_any[r] > 0).sum())
        ev = np.isfinite(d["t_del"][r])
        res["events"] = int(ev.sum())
        res["events_first_by_icdsc"] = int(np.nansum(d["p_first_pos_icdsc"][r][ev] == 1))
        out[arm] = res
    return out


def all_estimates(d, db, shares_other):
    w = ow_of(d)
    strat = stratum_effects(d, db)
    res = {"primary": primary(d), "strata": strat,
           "std_own": standardised(strat, rass_shares(d)), "std_other": standardised(strat, shares_other),
           "discharge": discharge(d, w), "per_assessment": per_assessment(d, w)}
    return res, w


def flat(res, db):
    names, vals = [], []

    def add(n, x):
        names.append(n)
        vals.append(math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan)

    add("primary", res["primary"]["OR"])
    for s, levels in res["strata"].items():
        ref = list(LEVEL_CODE[s])[0]
        for lv, e in levels.items():
            add(f"{s}:{lv}", e.get("OR"))
            ro = e.get("OR", np.nan) / levels[ref].get("OR", np.nan) if lv != ref else np.nan
            add(f"{s}:{lv}:ratio", ro if lv != ref else np.nan)
    add("std_own", res["std_own"])
    add("std_other", res["std_other"])
    add("discharge", res["discharge"]["OR"])
    add("per_assessment", res["per_assessment"]["RR"])
    return names, np.array(vals)


def analyze(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    data = {db: load(P, db) for db in DBS}
    shares = {db: rass_shares(data[db][0]) for db in DBS}
    out = {}
    for db in DBS:
        d, flow = data[db]
        res, w = all_estimates(d, db, shares["e" if db == "m" else "m"])
        res["describe"] = describe(d, w)
        res["flow"] = flow
        res["rass_shares"] = shares[db]
        if db == "e":
            res["hosp_n"] = d["hosp_n"]
        out[db] = res
        asd.status(f"analyze {db} done", t0)
    with open(os.path.join(P, "e1_est.json"), "w") as f:
        json.dump(asd.to_json(out), f)
    with open(os.path.join(P, "e1_shares.json"), "w") as f:
        json.dump(shares, f)
    asd.log_run(args.results, f"explore_ppi analyze: ok ({time.time() - t0:.0f} s)")


def _boot_one(t):
    d, db, rep, shares_other = t
    rng = np.random.default_rng(asd.SEED + rep)
    b = asd.resample(d, db, rng)
    try:
        res, _ = all_estimates(b, db, shares_other)
        return rep, flat(res, db)[1]
    except Exception:
        return rep, None


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    db = args.db
    d, _ = load(P, db)
    shares = json.load(open(os.path.join(P, "e1_shares.json")))
    other = shares["e" if db == "m" else "m"]
    est = json.load(open(os.path.join(P, "e1_est.json")))[db]
    names = flat_from_json(est, db)[0]
    reps = list(range(args.start, min(args.stop, NBOOT)))
    out = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for rep, v in pool.imap_unordered(_boot_one, [(d, db, rep, other) for rep in reps]):
                out[rep] = v
    else:
        for rep in reps:
            out[rep] = _boot_one((d, db, rep, other))[1]
    arr = np.full((len(reps), len(names)), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None and len(out[rep]) == len(names):
            arr[i] = out[rep]
    np.savez(os.path.join(P, f"e1boot_{db}_{args.start:03d}_{args.stop:03d}.npz"), reps=np.array(reps), v=arr, names=np.array(names))
    asd.status(f"boot {db} {args.start}-{args.stop} done, failed {sum(1 for r in reps if out.get(r) is None)}", t0)
    asd.log_run(args.results, f"explore_ppi boot --db {db} --start {args.start} --stop {args.stop}: ok ({time.time() - t0:.0f} s)")


def flat_from_json(est, db):
    res = {"primary": est["primary"], "strata": est["strata"], "std_own": est["std_own"], "std_other": est["std_other"],
           "discharge": est["discharge"], "per_assessment": est["per_assessment"]}
    fix = lambda x: float("nan") if x is None else x
    res["primary"] = {"OR": fix(res["primary"].get("OR"))}
    res["strata"] = {s: {lv: {"OR": fix(e.get("OR"))} for lv, e in lv_.items()} for s, lv_ in res["strata"].items()}
    res["std_own"], res["std_other"] = fix(res["std_own"]), fix(res["std_other"])
    res["discharge"] = {"OR": fix(res["discharge"].get("OR"))}
    res["per_assessment"] = {"RR": fix(res["per_assessment"].get("RR"))}
    return flat(res, db)


def load_boot(P, db):
    files = sorted(glob.glob(os.path.join(P, f"e1boot_{db}_*.npz")))
    if not files:
        return None, None
    reps, vs, names = [], [], None
    for f in files:
        z = np.load(f)
        reps.append(z["reps"])
        vs.append(z["v"])
        names = [str(x) for x in z["names"]]
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return names, np.concatenate(vs)[first]


def ci(names, B, name):
    if B is None or name not in names:
        return (None, None)
    v = B[:, names.index(name)]
    v = v[np.isfinite(v)]
    if v.size < asd.MIN_BOOT:
        return (None, None)
    return float(np.exp(np.percentile(v, 2.5))), float(np.exp(np.percentile(v, 97.5)))


def f2(x, nd=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def orci(x, lohi):
    return f"{f2(x)} ({f2(lohi[0])}–{f2(lohi[1])})"


def pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def report(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    est = json.load(open(os.path.join(P, "e1_est.json")))
    boots = {db: load_boot(P, db) for db in DBS}
    J = {"addendum": "SAP_addendum_E1.md v1.0", "exploratory": True, "db": {}}
    for db in DBS:
        e = est[db]
        names, B = boots[db]
        jd = {"primary": {"OR": e["primary"]["OR"], "CI": ci(names, B, "primary")}, "strata": {}, "describe": {}}
        for arm in ("A", "B"):
            x = e["describe"][arm]
            jd["describe"][arm] = {"n": sc.mask_n(x["n"]), "weighted": x["w"], "unweighted": x["u"],
                                   "hours_to_first_assessment_q25_50_75": x["hours_to_first_assessment_q25_50_75"],
                                   "n_with_any_assessment": sc.mask_n(x["n_with_any_assessment"]), "events": sc.mask_n(x["events"]),
                                   "events_first_by_icdsc": sc.mask_n(x["events_first_by_icdsc"])}
        for s, lv_ in e["strata"].items():
            jd["strata"][s] = {}
            for lv, x in lv_.items():
                ent = {"nA": sc.mask_n(x.get("nA", 0)), "nB": sc.mask_n(x.get("nB", 0)), "OR": x.get("OR"), "CI": ci(names, B, f"{s}:{lv}")}
                if lv != list(LEVEL_CODE[s])[0]:
                    ref = lv_[list(LEVEL_CODE[s])[0]].get("OR")
                    ent["ratio_vs_ref"] = (x.get("OR") / ref) if (x.get("OR") and ref) else None
                    ent["ratio_CI"] = ci(names, B, f"{s}:{lv}:ratio")
                if "eventsA" in x:
                    ent["eventsA"], ent["eventsB"] = sc.mask_n(x["eventsA"]), sc.mask_n(x["eventsB"])
                jd["strata"][s][lv] = ent
        jd["std_own"] = {"OR": e["std_own"], "CI": ci(names, B, "std_own")}
        jd["std_other"] = {"OR": e["std_other"], "CI": ci(names, B, "std_other")}
        dc = e["discharge"]
        jd["discharge"] = {"riskA168": dc["riskA"], "riskB168": dc["riskB"], "riskA72": dc["riskA72"], "riskB72": dc["riskB72"],
                           "RD168": dc["RD"], "OR": dc["OR"], "CI": ci(names, B, "discharge")}
        pa = e["per_assessment"]
        jd["per_assessment"] = {"rateA": pa["rateA"], "rateB": pa["rateB"], "RR": pa["RR"], "CI": ci(names, B, "per_assessment")}
        jd["rass_shares"] = e["rass_shares"]
        if db == "e":
            jd["hosp_n"] = e["hosp_n"]
        jd["boot_valid"] = int(np.isfinite(B[:, 0]).sum()) if B is not None else 0
        J["db"][db] = jd
    J = asd.to_json(J)
    with open(os.path.join(args.results, "ppi_explore_E1.json"), "w", encoding="utf-8") as f:
        json.dump(asd.round_floats(J), f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.results, "ppi_explore_E1.md"), "w", encoding="utf-8") as f:
        f.write(md(J))
    asd.log_run(args.results, f"explore_ppi report: ok ({time.time() - t0:.0f} s)")
    asd.status("report written", t0)


def md(J):
    D = J["db"]
    L = ["# PPI comparison exploratory addendum E1 results: where the difference between the databases comes from (exploratory)", "",
         "Plan: SAP_addendum_E1.md v1.0 (written after the main results and before computing). A = PPI, B = famotidine; odds ratio <1: lower in the PPI arm. "
         "Counts masked: 1–9 as <10, otherwise the nearest multiple of 5. All effects as in the main analysis (overlap weighting; ICU discharge alive and death as competing events; 168 h), with the propensity model refitted within strata. "
         f"Intervals: percentile intervals of {NBOOT} bootstrap replicates (MIMIC-IV by patient, eICU by hospital).", ""]
    L += ["## D1, D2 Assessment process and depth of sedation (overlap weighted; unweighted in brackets)", "",
          "| Measure | MIMIC-IV A | MIMIC-IV B | eICU A | eICU B |", "|---|---|---|---|---|"]
    rows = [("Delirium assessments per ICU day", "assess_per_icu_day", lambda x: f2(x, 1)),
            ("Share of 24-h blocks with an assessment", "share_day_blocks_assessed", pct),
            ("Patients with any assessment during follow-up", "share_with_any_assessment", pct),
            ("Share of assessments recorded as “unable to assess”", "share_unable_to_assess", pct),
            ("Patients assessed in the 24 h before time zero", "share_assessed_24h_before_t0", pct),
            ("Share of RASS records ≤−4 in 0–72 h", "share_rass_le_m4_0_72h", pct),
            ("Patients with any RASS ≤−4 in 0–24 h", "share_any_rass_le_m4_0_24h", pct)]
    for lab, k, fn in rows:
        cells = [f"{fn(D[db]['describe'][arm]['weighted'][k])} ({fn(D[db]['describe'][arm]['unweighted'][k])})" for db in DBS for arm in ("A", "B")]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("| Hours from time zero to the first assessment (quartiles, unweighted) | " + " | ".join(
        "/".join(str(x) for x in D[db]["describe"][arm]["hours_to_first_assessment_q25_50_75"]) for db in DBS for arm in ("A", "B")) + " |")
    L.append("| Patients analysed | " + " | ".join(D[db]["describe"][arm]["n"] for db in DBS for arm in ("A", "B")) + " |")
    L.append("| First positive screens determined by ICDSC (eICU) | — | — | " + " | ".join(
        f"{D['e']['describe'][arm]['events_first_by_icdsc']} / {D['e']['describe'][arm]['events']}" for arm in ("A", "B")) + " |")
    L += ["", "## D3 Effects by stratum", "", "| Stratifier | Stratum | MIMIC-IV patients A / B | MIMIC-IV OR (95% CI) | Ratio to reference stratum | eICU patients A / B | eICU OR (95% CI) | Ratio to reference stratum |",
          "|---|---|---|---|---|---|---|---|"]
    for s in ("rass", "base", "surg", "vent", "era", "hosp"):
        title, levels, labs = STRATA[s]
        for i, lv in enumerate(levels):
            cells = []
            for db in DBS:
                x = D[db]["strata"].get(s, {}).get(lv)
                if x is None:
                    cells += ["—", "—", "—"]
                    continue
                cells += [f"{x['nA']} / {x['nB']}", orci(x.get("OR"), x.get("CI") or (None, None)),
                          "reference" if i == 0 else orci(x.get("ratio_vs_ref"), x.get("ratio_CI") or (None, None))]
            L.append(f"| {title} | {labs[lv]} | " + " | ".join(cells) + " |")
    L += ["", "## D4 Standardisation to the RASS mix at time zero of the other database", ""]
    for db, other in (("m", "e"), ("e", "m")):
        x = D[db]
        L.append(f"- {asd.DBNAME[db]}: main analysis {orci(x['primary']['OR'], x['primary']['CI'])}; standardised to its own RASS mix {orci(x['std_own']['OR'], x['std_own']['CI'])}; "
                 f"standardised to the RASS mix of {asd.DBNAME[other]} {orci(x['std_other']['OR'], x['std_other']['CI'])}")
    L.append("- RASS mix (deep / moderate / awake or light / not recorded): " + "; ".join(
        f"{asd.DBNAME[db]} " + " / ".join(pct(D[db]["rass_shares"][lv]) for lv in ("deep", "moderate", "light", "missing")) for db in DBS))
    L += ["", "## D5 ICU discharge (discharge alive as the event, death competing; overlap weighted)", ""]
    for db in DBS:
        x = D[db]["discharge"]
        L.append(f"- {asd.DBNAME[db]}: discharged alive from the ICU within 72 h A {pct(x['riskA72'])} / B {pct(x['riskB72'])}; within 168 h A {pct(x['riskA168'])} / B {pct(x['riskB168'])}; "
                 f"OR {orci(x['OR'], x['CI'])}")
    L += ["", "## D6 Detection rate per assessment (first positive ÷ assessable assessments up to and including the first positive; overlap weighted)", ""]
    for db in DBS:
        x = D[db]["per_assessment"]
        L.append(f"- {asd.DBNAME[db]}: A {f2(100 * x['rateA'], 1)}%, B {f2(100 * x['rateB'], 1)}% per assessment; rate ratio {orci(x['RR'], x['CI'])}; "
                 f"main analysis OR {orci(D[db]['primary']['OR'], D[db]['primary']['CI'])}")
    if "hosp_n" in D["e"]:
        L += ["", f"- eICU hospitals in the three groups of assessment intensity: high {D['e']['hosp_n']['high']}, mid {D['e']['hosp_n']['mid']}, low {D['e']['hosp_n']['low']}"]
    L += ["", f"- Valid bootstrap replicates: MIMIC-IV {D['m']['boot_valid']}, eICU {D['e']['boot_valid']}", ""]
    return "\n".join(L) + "\n"


def selftest(args):
    import pipeline_selftest as pst
    t0 = time.time()
    rng = np.random.default_rng(7)
    tmp = tempfile.mkdtemp(prefix="asd_e1_")
    P, R = os.path.join(tmp, "private"), os.path.join(tmp, "results")
    os.makedirs(P)
    os.makedirs(R)
    ok = []
    try:
        for db, n in (("m", 3000), ("e", 2500)):
            outc, _ = pst.make_db(P, db, n, rng)
            pst.write_df(outc, os.path.join(P, f"{db}_outcome.parquet"))
            co = asd.read_pq(os.path.join(P, f"{db}_c2_cohort.parquet"))
            fu = asd.read_pq(os.path.join(P, f"{db}_follow.parquet"))
            x = co[["stay_id"]].merge(fu, on="stay_id").merge(outc, on="stay_id")
            hw = np.minimum(x["t_end"].values, 168)
            rate = 2.5 if db == "m" else 1.0
            n_any = rng.poisson(rate * hw / 24)
            n_ok = rng.binomial(n_any, 0.8)
            ev = np.isfinite(x["t_del"].values)
            n_any = np.where(ev, np.maximum(n_any, 1), n_any)
            n_ok = np.where(ev, np.maximum(n_ok, 1), n_ok)
            pr = pd.DataFrame({"stay_id": x["stay_id"], "h_window": hw, "n_any": n_any, "n_ok": n_ok,
                               "h_first": np.where(n_any > 0, rng.uniform(0, 12, len(x)), np.nan),
                               "blocks_covered": np.minimum(n_any, np.ceil(hw / 24)), "n_ok_until_pos": np.where(ev, np.maximum(1, n_ok // 2), n_ok),
                               "h_pos": x["t_del"].values, "first_pos_icdsc": np.where(ev, (rng.random(len(x)) < (0.1 if db == "e" else 0)).astype(float), np.nan),
                               "n_base": rng.poisson(1.0, len(x)), "n_rass72": rng.poisson(8, len(x)), "n_deep72": rng.poisson(1.5, len(x)),
                               "deep24": (rng.random(len(x)) < 0.2).astype(int)})
            pst.write_df(pr, os.path.join(P, f"{db}_c2_process.parquet"))
        hs = np.arange(100, 122)
        pst.write_df(pd.DataFrame({"hospitalid": hs, "blocks_covered": rng.integers(200, 900, hs.size), "blocks_total": np.full(hs.size, 1000),
                                   "stays": np.full(hs.size, 500)}), os.path.join(P, "e_hosp_intensity.parquet"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            analyze(argparse.Namespace(private=P, results=R))
            for db in DBS:
                boot(argparse.Namespace(private=P, results=R, db=db, start=0, stop=60, workers=2))
            report(argparse.Namespace(private=P, results=R))
        J = json.load(open(os.path.join(R, "ppi_explore_E1.json"), encoding="utf-8"))
        ok.append(("report written with both databases", set(J["db"]) == {"m", "e"}))
        ok.append(("stratified effects with intervals", all(J["db"][db]["strata"]["rass"]["light"].get("CI", [None])[0] is not None for db in DBS)))
        ok.append(("standardised and per-assessment estimates present", all(J["db"][db]["std_other"]["OR"] and J["db"][db]["per_assessment"]["RR"] for db in DBS)))
        cnt_ok = all((v == "<10" or v == "—" or (isinstance(v, str) and v.isdigit() and int(v) % 5 == 0))
                     for db in DBS for s in J["db"][db]["strata"].values() for lv in s.values() for k, v in lv.items() if k in ("nA", "nB", "eventsA", "eventsB"))
        ok.append(("counts masked", cnt_ok))
        dm = J["db"]["m"]["describe"]["A"]["weighted"]["assess_per_icu_day"]
        de = J["db"]["e"]["describe"]["A"]["weighted"]["assess_per_icu_day"]
        ok.append(("synthetic assessment frequency recovered (MIMIC 2.5/day > eICU 1.0/day)", 2.0 < dm < 3.0 and 0.7 < de < 1.3))
        mdtxt = open(os.path.join(R, "ppi_explore_E1.md"), encoding="utf-8").read()
        ok.append(("markdown report has all sections", all(h in mdtxt for h in ("D1, D2", "D3", "D4", "D5", "D6"))))
    finally:
        if os.environ.get("ASD_KEEP"):
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    for name, v in ok:
        print(f"{'PASS' if v else 'FAIL'} | {name}")
    print(f"{sum(v for _, v in ok)}/{len(ok)} passed ({time.time() - t0:.0f} s)")
    sys.exit(0 if all(v for _, v in ok) else 1)


def bindcheck(args):
    import duckdb
    import sql_bind_check as sbc
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="asd_e1bind_")
    ok = []
    try:
        for db, src in (("mimic", args.mimic), ("eicu", args.eicu)):
            d = "m" if db == "mimic" else "e"
            con = duckdb.connect()
            mods = ("hosp", "icu", "ed", "note", "derived") if db == "mimic" else ("eicu", "eicu_derived")
            sbc.empty_tables(con, os.path.expanduser(src), mods)
            for sql, path in asd.all_sql(tmp, db)["cohort"]:
                con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")
            stmts = [("process", proc_sql(d, tmp))] + ([("hospital intensity", HOSP_E)] if d == "e" else [])
            for name, sql in stmts:
                try:
                    cols = [r[0] for r in con.execute(f"DESCRIBE {sql}").fetchall()]
                    ok.append((f"{db}: {name}", True, ", ".join(cols)))
                except Exception as e:
                    ok.append((f"{db}: {name}", False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"))
            con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, v, det in ok:
        print(f"{'PASS' if v else 'FAIL'} | {name} | {det}")
    print(f"{sum(v for _, v, _ in ok)}/{len(ok)} passed ({time.time() - t0:.0f} s)")
    sys.exit(0 if all(v for _, v, _ in ok) else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("extract")
    s.add_argument("--db", choices=["mimic", "eicu"], required=True)
    s.add_argument("--private", required=True)
    s.add_argument("--results", default="../results")
    for name in ("analyze", "report"):
        s = sub.add_parser(name)
        s.add_argument("--private", required=True)
        s.add_argument("--results", default="../results")
    s = sub.add_parser("boot")
    s.add_argument("--db", choices=list(DBS), required=True)
    s.add_argument("--start", type=int, required=True)
    s.add_argument("--stop", type=int, required=True)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--private", required=True)
    s.add_argument("--results", default="../results")
    sub.add_parser("selftest")
    s = sub.add_parser("bindcheck")
    s.add_argument("mimic")
    s.add_argument("eicu")
    args = ap.parse_args()
    {"extract": extract, "analyze": analyze, "boot": boot, "report": report, "selftest": selftest, "bindcheck": bindcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
