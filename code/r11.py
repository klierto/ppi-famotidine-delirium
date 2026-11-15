#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import math
import re
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
import r7
import r9
import r10

PLAN = "SAP_addendum_R11.md v1.0"
NBOOT = 2000
H = 168
MIN_N = 20
SEG_H = {3: (0, 24, 72, 168), 6: (0, 12, 24, 48, 72, 120, 168)}
PERIODS = [("early", ("2011 - 2013", "2014 - 2016")), ("mid", ("2017 - 2019",)), ("late", ("2020 - 2022",))]
PERIOD_LAB = {"early": "2011–2016", "mid": "2017–2019", "late": "2020–2022"}
FLEX_LIMIT_MIN = 40.0
DBNAME = {"m": "MIMIC-IV", "e": "eICU"}

STER_M_INC = "^(prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone|cortisone|betamethasone sodium|solu-medrol|medrol)"
STER_M_EXC = r"ophth|cream|oint|lot\.|lotion|gel|foam|rectal|supposit|enema|acetic|study|placebo|nasal|inhal|otic|acetate|dental|paste|susp 0\.1|soln 0\.1"
STER_E_INC = "prednisone|prednisolone|methylprednisolone|solu-medrol|solumedrol|medrol|dexamethasone|decadron|hydrocortisone|solu-cortef|cortef|cortisone"
STER_E_EXC = "ophth|eye|otic|cream|oint|lotion|topical|rectal|supp|enema|foam|gel|inhal|nasal|neb|intra-art|intraart|dental|budesonide|fludro"
E_BENZO, E_DEX, E_KET = "midazolam|versed|lorazepam|ativan", "dexmedetomidine|dexmetetomidine|precedex", "ketamine"
NEWCOV = ["steroid_24h", "benzo_inf", "dex_inf", "ketamine_inf"]
PO_NAMES = ["po_primary", "po_negctl", "po_px_cam", "po_px_icu", "po_px_hosp"]
PER_OUT = ["primary", "negctl", "px_icu", "px_hosp"]
BLOCKS = [("demo", "demographics", r"^(age_|female$)"), ("renal", "kidney function", r"^egfr_"),
          ("severity", "severity and organ support", r"^(sofa_|apache_|vent$|vaso$|coag)"), ("sedation", "sedation", r"^(rass_|sed_inf$|opi_inf$)"),
          ("context", "unit and time", r"^(unit_|surgical$|era_|h_ge|route_)"), ("comorb", "comorbidities", r"^(dementia|alcohol|liver)$"),
          ("hosp", "hospital", r"^hosp_")]
NW_OUT = ["icu", "hosp", "negctl"]

TIMELINE_M = """
WITH c AS (SELECT stay_id, t0, outtime FROM read_parquet('{f}') WHERE arm IS NOT NULL),
fu AS (SELECT stay_id, t0, least(outtime, t0 + INTERVAL 168 HOUR) AS t_end FROM c),
da AS (SELECT stay_id, charttime, value FROM icu.chartevents WHERE itemid = 228332 AND value IS NOT NULL)
SELECT f.stay_id, date_diff('minute', f.t0, d.charttime) / 60.0 AS h,
       max(CASE WHEN d.value = 'Positive' THEN 1 ELSE 0 END) AS pos,
       max(CASE WHEN d.value IN ('Positive', 'Negative') THEN 1 ELSE 0 END) AS ok
FROM fu f JOIN da d ON d.stay_id = f.stay_id AND d.charttime > f.t0 AND d.charttime <= f.t_end
GROUP BY f.stay_id, f.t0, d.charttime
"""
TIMELINE_E = """
WITH c AS (SELECT stay_id, t0 FROM read_parquet('{f}') WHERE arm IS NOT NULL),
fu AS (SELECT c.stay_id, c.t0, least(p.unitdischargeoffset, c.t0 + 10080) AS t_end FROM c JOIN eicu.patient p ON p.patientunitstayid = c.stay_id),
da AS (SELECT patientunitstayid AS stay_id, nursingchartoffset AS t, nursingchartvalue AS v FROM eicu.nursecharting
       WHERE nursingchartcelltypevallabel = 'Delirium Scale/Score' AND nursingchartcelltypevalname = 'Delirium Score'
         AND NULLIF(nursingchartvalue, '') IS NOT NULL)
SELECT f.stay_id, (d.t - f.t0) / 60.0 AS h,
       max(CASE WHEN upper(d.v) = 'YES' OR TRY_CAST(d.v AS INTEGER) >= 4 THEN 1 ELSE 0 END) AS pos,
       max(CASE WHEN upper(d.v) IN ('YES', 'NO') OR TRY_CAST(d.v AS INTEGER) IS NOT NULL THEN 1 ELSE 0 END) AS ok
FROM fu f JOIN da d ON d.stay_id = f.stay_id AND d.t > f.t0 AND d.t <= f.t_end GROUP BY f.stay_id, f.t0, d.t
"""
COV_M = f"""
WITH base AS (SELECT DISTINCT hadm_id, stay_id, t0 FROM read_parquet({{files}})),
ster AS (SELECT DISTINCT b.stay_id FROM base b JOIN src.emar e ON e.hadm_id = b.hadm_id
           AND e.charttime BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0
         WHERE regexp_matches(lower(e.medication), '{STER_M_INC}') AND NOT regexp_matches(lower(e.medication), '{STER_M_EXC}')
           AND regexp_matches(lower(coalesce(e.event_txt, '')), 'administ|started')
           AND NOT regexp_matches(lower(coalesce(e.event_txt, '')), 'not |hold|stopp')),
sed AS (SELECT b.stay_id,
          max(CASE WHEN x.itemid IN (221668, 221385) THEN 1 ELSE 0 END) AS benzo_inf,
          max(CASE WHEN x.itemid IN (225150, 229420) THEN 1 ELSE 0 END) AS dex_inf,
          max(CASE WHEN x.itemid = 221712 THEN 1 ELSE 0 END) AS ketamine_inf
        FROM base b JOIN icu.inputevents x ON x.stay_id = b.stay_id AND x.starttime <= b.t0 AND x.endtime > b.t0
        WHERE x.itemid IN (221668, 221385, 225150, 229420, 221712) AND coalesce(x.rate, 0) > 0
          AND coalesce(x.statusdescription, '') <> 'Rewritten' GROUP BY 1),
rx AS (SELECT b.stay_id, r.starttime, TRY_CAST(regexp_extract(r.dose_val_rx, '^[0-9.]+') AS DOUBLE) * r.doses_per_24_hrs AS daily
       FROM base b JOIN hosp.prescriptions r ON r.hadm_id = b.hadm_id AND r.starttime <= b.t0 + INTERVAL 1 HOUR
        AND (r.stoptime IS NULL OR r.stoptime > b.t0)
       WHERE regexp_matches(lower(r.drug), 'famotidine|pepcid')),
fam AS (SELECT stay_id, daily FROM (
          SELECT stay_id, daily, row_number() OVER (PARTITION BY stay_id ORDER BY starttime DESC, daily DESC NULLS LAST) AS rn FROM rx)
        WHERE rn = 1)
SELECT b.stay_id, (s.stay_id IS NOT NULL) AS steroid_24h, coalesce(x.benzo_inf, 0) AS benzo_inf, coalesce(x.dex_inf, 0) AS dex_inf,
       coalesce(x.ketamine_inf, 0) AS ketamine_inf, f.daily AS fam_daily
FROM base b LEFT JOIN ster s ON s.stay_id = b.stay_id LEFT JOIN sed x ON x.stay_id = b.stay_id LEFT JOIN fam f ON f.stay_id = b.stay_id
"""
COV_E = f"""
WITH base AS (SELECT DISTINCT stay_id, t0 FROM read_parquet({{files}})),
ster AS (SELECT DISTINCT b.stay_id FROM base b JOIN src.medication m ON m.patientunitstayid = b.stay_id
         WHERE coalesce(m.drugordercancelled, 'No') <> 'Yes'
           AND regexp_matches(lower(coalesce(m.drugname, '')), '{STER_E_INC}')
           AND NOT regexp_matches(lower(coalesce(m.drugname, '')) || ' ' || lower(coalesce(m.routeadmin, '')), '{STER_E_EXC}')
           AND m.drugstartoffset <= b.t0 AND (m.drugstopoffset >= b.t0 - 1440 OR m.drugstopoffset <= m.drugstartoffset)),
sed AS (SELECT b.stay_id,
          max(CASE WHEN regexp_matches(lower(i.drugname), '{E_BENZO}') THEN 1 ELSE 0 END) AS benzo_inf,
          max(CASE WHEN regexp_matches(lower(i.drugname), '{E_DEX}') THEN 1 ELSE 0 END) AS dex_inf,
          max(CASE WHEN regexp_matches(lower(i.drugname), '{E_KET}') THEN 1 ELSE 0 END) AS ketamine_inf
        FROM base b JOIN eicu.infusiondrug i ON i.patientunitstayid = b.stay_id AND i.infusionoffset BETWEEN b.t0 - 120 AND b.t0
        WHERE TRY_CAST(i.drugrate AS DOUBLE) > 0 GROUP BY 1)
SELECT b.stay_id, (s.stay_id IS NOT NULL) AS steroid_24h, coalesce(x.benzo_inf, 0) AS benzo_inf, coalesce(x.dex_inf, 0) AS dex_inf,
       coalesce(x.ketamine_inf, 0) AS ketamine_inf, CAST(NULL AS DOUBLE) AS fam_daily
FROM base b LEFT JOIN ster s ON s.stay_id = b.stay_id LEFT JOIN sed x ON x.stay_id = b.stay_id
"""
POSTOP_M = f"""
WITH base AS (SELECT DISTINCT subject_id, hadm_id, stay_id, intime, outtime, t0 FROM read_parquet({{files}})),
prev AS (SELECT stay_id, careunit, t_prev FROM (
           SELECT b.stay_id, t.careunit, t.intime AS t_prev,
                  row_number() OVER (PARTITION BY b.stay_id ORDER BY t.intime DESC, t.careunit) AS rn
           FROM base b JOIN hosp.transfers t ON t.hadm_id = b.hadm_id AND t.intime < b.intime) WHERE rn = 1),
ap AS (
  SELECT e.hadm_id, e.charttime AS t, regexp_matches(lower(e.medication), 'haloperidol|haldol') AS hal
  FROM src.emar e
  WHERE regexp_matches(lower(e.medication), '{r7.AP}') AND NOT regexp_matches(lower(e.medication), '{r7.AP_LAI}')
    AND regexp_matches(lower(coalesce(e.event_txt, '')), 'administ|started')
    AND NOT regexp_matches(lower(coalesce(e.event_txt, '')), 'not |hold|stopp')
  UNION ALL
  SELECT s.hadm_id, x.starttime AS t, true AS hal FROM src.inputevents x JOIN icu.icustays s USING (stay_id)
  WHERE x.itemid = {r7.HAL_ITEM} AND coalesce(x.amount, 0) > 0 AND coalesce(x.statusdescription, '') <> 'Rewritten'
),
win AS (SELECT b.stay_id, b.hadm_id, b.t0, least(b.outtime, b.t0 + INTERVAL 168 HOUR) AS t_icu,
               least(coalesce(least(a.deathtime, a.dischtime), a.dischtime), b.t0 + INTERVAL 168 HOUR) AS t_hosp,
               CASE WHEN p.careunit = 'PACU' THEN p.t_prev END AS t_pacu
        FROM base b LEFT JOIN hosp.admissions a ON a.hadm_id = b.hadm_id LEFT JOIN prev p ON p.stay_id = b.stay_id),
o AS (SELECT w.stay_id,
        min(CASE WHEN a.t > w.t0 AND a.t <= w.t_icu THEN a.t END) AS ap_icu,
        min(CASE WHEN a.t > w.t0 AND a.t <= w.t_hosp THEN a.t END) AS ap_hosp
      FROM win w JOIN ap a ON a.hadm_id = w.hadm_id
      WHERE NOT (a.hal AND w.t_pacu IS NOT NULL AND a.t <= w.t_pacu + INTERVAL 6 HOUR)
      GROUP BY 1)
SELECT b.stay_id, p.careunit AS prev_unit, date_diff('minute', p.t_prev, b.intime) / 60.0 AS h_prev_icu,
       date_diff('minute', p.t_prev, b.t0) / 60.0 AS h_prev_t0,
       date_diff('minute', b.t0, o.ap_icu) / 60.0 AS t_ap_icu_po, date_diff('minute', b.t0, o.ap_hosp) / 60.0 AS t_ap_hosp_po
FROM base b LEFT JOIN prev p ON p.stay_id = b.stay_id LEFT JOIN o ON o.stay_id = b.stay_id
"""
EXPECT = {"timeline": "stay_id h pos ok", "cov": "stay_id steroid_24h benzo_inf dex_inf ketamine_inf fam_daily",
          "postop": "stay_id prev_unit h_prev_icu h_prev_t0 t_ap_icu_po t_ap_hosp_po"}
FILE = {"timeline": "r11tl", "cov": "r11cov", "postop": "r11po"}


def statements(P, db):
    c2 = os.path.join(P, f"{db}_c2_cohort.parquet")
    files = asd.plist([c2])
    st = {"timeline": (TIMELINE_M if db == "m" else TIMELINE_E).format(f=c2), "cov": (COV_M if db == "m" else COV_E).format(files=files)}
    if db == "m":
        st["postop"] = POSTOP_M.format(files=files)
    return st


def extract(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    st = statements(P, args.db)
    if args.part not in st:
        raise SystemExit(f"part {args.part} does not exist for {args.db}")
    con = r7.connect(P, args.db)
    path = r7.pf(P, args.db, FILE[args.part])
    asd.write_pq(con, st[args.part], path)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    asd.status(f"{os.path.basename(path)} written, {sc.mask_n(n)} rows (masked)", t0)
    asd.log_run(args.results, f"r11 extract --db {args.db} --part {args.part}: ok ({time.time() - t0:.0f} s)")


def nz(s):
    return pd.to_numeric(pd.Series(s), errors="coerce").fillna(0).astype(float).values


def build_m0(df, db):
    d0 = asd.build(df, db, 2, stage2=True)
    for c in (r7.PROXY_M if db == "m" else r7.PROXY_E):
        d0[c] = asd.num(df, c)
    d0["ap_pre"] = nz(df["ap_pre"]) == 1
    d0["psych"] = nz(df["psych"]) == 1
    if db == "m":
        d0["end_death_h"] = nz(df["end_death_h"]) == 1
    return d0


def timeline_arrays(d, sid, tl):
    n = len(sid)
    T = np.minimum(d["t_end"], float(H))
    tdel = d["t_del"]
    det = np.isfinite(tdel) & (tdel <= T)
    stop = np.where(det, tdel, T)
    row = pd.Series(np.arange(n), index=sid).reindex(tl["stay_id"].values).values
    keep = np.isfinite(row)
    i = row[keep].astype(int)
    h = tl["h"].values.astype(float)[keep]
    ok = tl["ok"].values.astype(float)[keep] == 1
    pos = tl["pos"].values.astype(float)[keep] == 1
    pre = (h > 0) & (h <= stop[i] + 1e-9)
    out = {"ic_det": det}
    out["scr_any"] = np.bincount(i[pre], minlength=n).astype(float)
    out["scr_ok"] = np.bincount(i[pre & ok], minlength=n).astype(float)
    out["scr_days"] = np.maximum(stop, 0.0) / 24.0
    out["blk_risk"] = np.ceil(np.maximum(stop, 0.0) / 12.0)
    bk = np.ceil(h / 12.0).astype(int) - 1
    m = pre & ok
    pairs = np.unique(np.column_stack([i[m], bk[m]]), axis=0) if m.any() else np.zeros((0, 2), int)
    out["blk_ok"] = np.bincount(pairs[:, 0], minlength=n).astype(float) if len(pairs) else np.zeros(n)
    tcut = np.where(det, tdel, np.inf)
    neg = ok & ~pos & (h > 0) & (h <= T[i] + 1e-9) & (h < tcut[i])
    tL = np.zeros(n)
    np.maximum.at(tL, i[neg], h[neg])
    out["ic_tL"], out["ic_tR"] = tL, np.where(det, tdel, np.nan)
    return out


def load_ppi(P, db):
    df, flow = r7.load_base(P, "c2" + db)
    n0 = len(df)
    df = df.merge(asd.read_pq(r7.pf(P, db, "r11cov")), on="stay_id", how="left", validate="one_to_one")
    if db == "m":
        df = df.merge(asd.read_pq(r7.pf(P, db, "r11po")), on="stay_id", how="left", validate="one_to_one")
    if len(df) != n0:
        raise SystemExit(f"{db}: R11 rows missing")
    d = build_m0(df, db)
    d.update(timeline_arrays(d, df["stay_id"].values, asd.read_pq(r7.pf(P, db, "r11tl"))))
    add_r11_fields(d, df, db)
    return d, df, flow


def add_r11_fields(d, df, db):
    extra = np.column_stack([nz(df[c]) == 1 for c in NEWCOV]).astype(float)
    X3, n3, dr3 = asd.prevalence_filter(np.column_stack([d["X"], extra]), d["names"] + NEWCOV)
    d["X3"], d["names3"], d["dropped3"] = X3, n3, dr3
    if db == "m":
        h1, h2 = asd.num(df, "h_prev_icu"), asd.num(df, "h_prev_t0")
        d["postop"] = (df["prev_unit"].astype(str).values == "PACU") & (h1 >= 0) & (h1 <= 24) & (h2 <= 72)
        for c in ("t_ap_icu_po", "t_ap_hosp_po"):
            d[c] = asd.num(df, c)
        era = df["era"].astype(str).values
        d["period"] = np.full(len(df), -1)
        for k, (_, groups) in enumerate(PERIODS):
            d["period"][np.isin(era, groups)] = k


def ow_of(d, cache, key, X=None, names=None):
    dd = d if X is None else dict(d, X=X, names=names)
    return sc.overlap_w(asd.ow_fit(dd, cache, key), d["A"])


def enough(d, m):
    A = d["A"]
    return int((m & (A == 1)).sum()) >= MIN_N and int((m & (A == 0)).sum()) >= MIN_N


def nan_eff():
    return sc.effects(float("nan"), float("nan"))


def screening(d, w):
    A = d["A"]
    out = {}
    for key, num, den in (("scr_any", "scr_any", "scr_days"), ("scr_ok", "scr_ok", "scr_days"), ("blk_ok_share", "blk_ok", "blk_risk")):
        r = {a: float((w * d[num])[A == a].sum() / max((w * d[den])[A == a].sum(), 1e-12)) for a in (1, 0)}
        out[key] = {"A": r[1], "B": r[0], "ratio": r[1] / r[0] if r[0] > 0 else float("nan")}
    return out


def seg_exposure(t, cuts):
    t = np.asarray(t, float)
    return np.column_stack([np.clip(t - a, 0.0, b - a) for a, b in zip(cuts[:-1], cuts[1:])])


def ic_fit(d, w, nseg=3):
    from scipy.optimize import minimize
    cuts = SEG_H[nseg]
    S = len(cuts) - 1
    det, A = d["ic_det"], d["A"]
    tL = np.asarray(d["ic_tL"], float)
    tR = np.where(det, d["ic_tR"], 0.0)
    inf = det | (tL > 0)
    EL, ER = seg_exposure(tL, cuts), seg_exposure(tR, cuts)
    w = np.asarray(w, float)
    groups = []
    for arm in (1, 0):
        r = inf & (A == arm)
        groups.append((arm, EL[r], ER[r], det[r], w[r]))

    def nll(p):
        lam = np.exp(p[:S])
        tot = 0.0
        for arm, el, er, dt, ww in groups:
            mult = math.exp(p[S] * arm)
            hl = (el @ lam) * mult
            ll = -hl
            dh = (er[dt] @ lam) * mult - hl[dt]
            ll[dt] = ll[dt] + np.log(-np.expm1(-np.maximum(dh, 1e-12)))
            tot -= float((ww * ll).sum())
        return tot

    x0 = np.r_[np.full(S, math.log(0.003)), 0.0]
    fit = minimize(nll, x0, method="L-BFGS-B")
    p = fit.x
    H0 = float(sum(math.exp(p[s]) * (cuts[s + 1] - cuts[s]) for s in range(S)))
    risk = {arm: 1.0 - math.exp(-H0 * math.exp(p[S] * arm)) for arm in (1, 0)}
    e = sc.effects(risk[1], risk[0])
    e["HR"] = float(math.exp(p[S]))
    e["n_inf_A"], e["n_inf_B"] = int((inf & (A == 1)).sum()), int((inf & (A == 0)).sum())
    e["eventsA"], e["eventsB"] = int((det & (A == 1)).sum()), int((det & (A == 0)).sum())
    e["converged"] = bool(fit.success)
    return e


def per_screen(d, w):
    A = d["A"]
    ev = d["ic_det"].astype(float)
    r = {a: float((w * ev)[A == a].sum() / max((w * d["scr_ok"])[A == a].sum(), 1e-12)) for a in (1, 0)}
    return r[1] / r[0]


def cif(d, w, t):
    return asd.cif_contrast(d, w, t)[168]


def postop_analyses(d, cache):
    out = {}
    m = d["postop"]
    if enough(d, m):
        ds = asd.subset_d(d, m)
        w = ow_of(ds, cache, "po")
        out["po_primary"], out["po_negctl"] = cif(ds, w, ds["t_del"]), cif(ds, w, ds["t_hypo"])
    mp = m & r7.proxy_mask(d)
    if enough(d, mp):
        dp = asd.subset_d(d, mp)
        w = ow_of(dp, cache, "popx")
        out["po_px_cam"] = cif(dp, w, dp["t_del"])
        out["po_px_icu"] = cif(dp, w, dp["t_ap_icu_po"])
        out["po_px_hosp"] = r7.hosp_contrast(dp, w, dp["t_ap_hosp_po"])
    return out


def period_analyses(d, cache):
    out = {}
    px = r7.proxy_mask(d)
    for k, (p, _) in enumerate(PERIODS):
        m = d["period"] == k
        if enough(d, m):
            ds = asd.subset_d(d, m)
            w = ow_of(ds, cache, f"per{p}")
            out[f"{p}_primary"], out[f"{p}_negctl"] = cif(ds, w, ds["t_del"]), cif(ds, w, ds["t_hypo"])
        if enough(d, m & px):
            dp = asd.subset_d(d, m & px)
            w = ow_of(dp, cache, f"perpx{p}")
            out[f"{p}_px_icu"] = cif(dp, w, dp["t_ap_icu"])
            out[f"{p}_px_hosp"] = r7.hosp_contrast(dp, w, dp["t_ap_hosp"])
    return out


def analyses_all(d, db, cache):
    res = {}
    w = ow_of(d, cache, "m0")
    res["primary"], res["negctl"] = cif(d, w, d["t_del"]), cif(d, w, d["t_hypo"])
    res["screen"] = screening(d, w)
    res["ic"], res["ic6"] = ic_fit(d, w, 3), ic_fit(d, w, 6)
    m = d["sub"]["rass3"]
    if enough(d, m):
        ds = asd.subset_d(d, m)
        res["ic_rass3"] = ic_fit(ds, ow_of(ds, cache, "rass3"), 3)
    else:
        res["ic_rass3"] = dict(nan_eff(), HR=float("nan"))
    w3 = ow_of(d, cache, "m3", d["X3"], d["names3"])
    res["m3_primary"], res["m3_negctl"] = cif(d, w3, d["t_del"]), cif(d, w3, d["t_hypo"])
    if db == "m":
        res.update(postop_analyses(d, cache))
        res.update(period_analyses(d, cache))
    return res


def flat_names(db):
    n = ["primary", "negctl", "m3_primary", "m3_negctl", "ic", "ic6", "ic_rass3", "screen_scr_any", "screen_scr_ok", "screen_blk_ok_share"]
    if db == "m":
        n += PO_NAMES + [f"{p}_{o}" for p, _ in PERIODS for o in PER_OUT]
    return n


def flat(res, db):
    vals = []
    for nm in flat_names(db):
        if nm.startswith("screen_"):
            x = res["screen"][nm[7:]]["ratio"]
        elif nm in ("ic", "ic6", "ic_rass3"):
            x = res[nm].get("HR")
        else:
            x = (res.get(nm) or {}).get("OR")
        vals.append(math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan)
    return np.array(vals)


def or_cols(d, cols):
    if not cols:
        return cif(d, np.ones(len(d["A"])), d["t_del"])["OR"]
    ps, _ = sc.ps_fit(d["X"][:, cols], d["A"])
    return cif(d, sc.overlap_w(ps, d["A"]), d["t_del"])["OR"]


def decomposition(d):
    names = d["names"]
    allc = list(range(len(names)))
    out = {"crude": or_cols(d, []), "full": or_cols(d, allc), "blocks": {}}
    for key, lab, pat in BLOCKS:
        idx = [j for j, nm in enumerate(names) if re.match(pat, nm)]
        if idx:
            out["blocks"][key] = {"label": lab, "columns": len(idx), "minus": or_cols(d, [j for j in allc if j not in idx]),
                                  "only": or_cols(d, idx)}
    return out


def load_nw(P):
    df = r9.load(P, "n", outcome=True)
    d = r9.build(df, "n", outcome=True)
    d["vent0"] = r9.flag(df["vent"])
    d["lac"] = r9.num(df["lac_max"])
    return d


def nw_strata(d):
    res = {"all": r9.analyses(d, {})}
    for lv, m in (("novent", ~d["vent0"]), ("vent", d["vent0"])):
        res[lv] = r9.analyses(asd.subset_d(d, m), {}) if enough(d, m) else {k: nan_eff() for k in r9.NAMES}
    return res


def nw_features(d):
    ex = d["extras"]
    return np.column_stack([d["X"], ex["age_years"], ex["egfr"], ex["h_icu"], d["lac"]])


def nw_flex(d, folds, make):
    F = nw_features(d)
    A = d["A"]
    e = r10.crossfit_ps(F, A, folds, make)
    ow = sc.overlap_w(e, A)
    di = dict(d, t_end=d["t_end_icu"], end_death=d["end_death_icu"])
    dh = dict(d, t_end=d["t_end_hosp"], end_death=d["end_death_hosp"])
    res = {}
    for o, dv, t in (("icu", di, d["t_ap"]), ("hosp", dh, d["t_ap"]), ("negctl", di, d["t_hypo"])):
        res[f"mlps_{o}"] = asd.cif_contrast(dv, ow, t)[168]
        Y = r10.y168(dv, t)
        m1, m0 = r10.crossfit_outcome(F, A, Y, folds, make)
        res[f"dr_{o}"] = r10.aow(e, A, Y, m1, m0)
    return res, e, ow


NW_STRATA_NAMES = [f"{lv}_{o}" for lv in ("all", "novent", "vent") for o in NW_OUT]
NW_FLEX_NAMES = [f"{k}_{o}" for k in ("mlps", "dr") for o in NW_OUT]


def nw_flat(res, names):
    vals = []
    for nm in names:
        k, o = nm.split("_", 1)
        x = (res.get(k, {}).get(o) if k in ("all", "novent", "vent") else res.get(nm)) or {}
        x = x.get("OR")
        vals.append(math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan)
    return np.array(vals)


def r7_check(d0, db, R):
    key = "c2" + db
    res = r7.analyses({"m0": d0}, key, {})
    ref = json.load(open(os.path.join(R, "r7.json"), encoding="utf-8"))["per_key"][key]
    ok = []
    for nm in ["m0_primary", "m0_negctl", "px_cam", "px_icu"] + (["px_hosp"] if db == "m" else []):
        a, b = round(res[nm]["OR"], 4), ref[nm]["OR"]
        good = abs(a - b) < 1e-9
        ok.append(good)
        print(f"{'PASS' if good else 'FAIL'} | {key} {nm} OR {a:.4f} vs R7 {b:.4f}", flush=True)
    return all(ok)


def nw_check(d, R):
    res = r9.analyses(d, {})
    ref = json.load(open(os.path.join(R, "r9.json"), encoding="utf-8"))["per_db"]["NWICU"]
    ok = []
    for nm in NW_OUT:
        a, b = round(res[nm]["OR"], 4), ref[nm]["OR"]
        good = abs(a - b) < 1e-9
        ok.append(good)
        print(f"{'PASS' if good else 'FAIL'} | NWICU {nm} OR {a:.4f} vs R9 {b:.4f}", flush=True)
    return all(ok)


def describe_newcov(d, df, w):
    A = d["A"]
    out = {}
    for c in NEWCOV:
        x = (nz(df[c]) == 1).astype(float)
        out[c] = {"nA": int(x[A == 1].sum()), "nB": int(x[A == 0].sum()), "NA": int((A == 1).sum()), "NB": int((A == 0).sum()),
                  "smd_unweighted": sc.smd_table(x[:, None], [c], A)[c], "smd_ow": sc.smd_table(x[:, None], [c], A, w)[c]}
    return out


def describe_periods(d, df):
    A = d["A"]
    egfr = asd.egfr_of(df)
    ind = {"surgical": df["service"].astype(str).isin(asd.M_SURG).values,
           "cvicu": df["unit"].astype(str).map(asd.M_UNIT).fillna("OTHER").values == "CVICU",
           "postop": d["postop"], "rass_le_m2": asd.num(df, "rass") <= -2, "rass_missing": ~np.isfinite(asd.num(df, "rass")),
           "sed_inf": nz(df["sed_inf"]) == 1, "benzo_inf": nz(df["benzo_inf"]) == 1, "dex_inf": nz(df["dex_inf"]) == 1,
           "vent": nz(df["vent"]) == 1, "route_enteral": df["route"].astype(str).values == "EN", "egfr_lt60": np.isfinite(egfr) & (egfr < 60),
           "age_ge65": asd.num(df, "age") >= 65, "steroid_24h": nz(df["steroid_24h"]) == 1}
    fam = asd.num(df, "fam_daily")
    out = {}
    for k, (p, _) in enumerate(PERIODS):
        for arm, a in (("A", 1), ("B", 0)):
            r = (d["period"] == k) & (A == a)
            ent = {"N": int(r.sum())}
            for nm, v in ind.items():
                ent[nm] = int((v & r).sum())
            ent["screens_per_day"] = float(d["scr_any"][r].sum() / max(d["scr_days"][r].sum(), 1e-12))
            if a == 0:
                known = r & np.isfinite(fam)
                ent["fam_known"], ent["fam_ge40"] = int(known.sum()), int((known & (fam >= 40)).sum())
                k60 = known & np.isfinite(egfr) & (egfr < 60)
                ent["fam_known_egfr60"], ent["fam_ge40_egfr60"] = int(k60.sum()), int((k60 & (fam >= 40)).sum())
            out[f"{p}:{arm}"] = ent
    return out


def est_ppi(d, df, flow, db, cache):
    res = analyses_all(d, db, cache)
    w = ow_of(d, {}, "m0")
    A = d["A"]
    inf = d["ic_det"] | (d["ic_tL"] > 0)
    ent = {"flow": flow, "res": asd.to_json(res), "decomposition": asd.to_json(decomposition(d)),
           "newcov": asd.to_json(describe_newcov(d, df, w)), "m3_columns_dropped": d["dropped3"],
           "screen_unweighted": asd.to_json(screening(d, np.ones(len(A)))),
           "ic_informative": {"A": int((inf & (A == 1)).sum()), "B": int((inf & (A == 0)).sum())}}
    if db == "m":
        po, px = d["postop"], r7.proxy_mask(d)
        ent["postop_n"] = {"A": int((po & (A == 1)).sum()), "B": int((po & (A == 0)).sum()),
                           "proxy_A": int((po & px & (A == 1)).sum()), "proxy_B": int((po & px & (A == 0)).sum())}
        ent["periods"] = asd.to_json(describe_periods(d, df))
        ent["period_n"] = {p: {"A": int(((d["period"] == k) & (A == 1)).sum()), "B": int(((d["period"] == k) & (A == 0)).sum())}
                           for k, (p, _) in enumerate(PERIODS)}
    return ent


def est_nw(dn):
    res_n = nw_strata(dn)
    flex, e, ow = nw_flex(dn, r10.main_folds(len(dn["A"])), r10.gbm)
    A = dn["A"]
    ps_lr = asd.ow_fit(dn, {}, "full")
    smd = sc.smd_table(dn["X"], dn["names"], A, ow)
    return {"strata": asd.to_json(res_n), "flex": asd.to_json(flex),
            "vent_n": {"novent_A": int((~dn["vent0"] & (A == 1)).sum()), "novent_B": int((~dn["vent0"] & (A == 0)).sum()),
                       "vent_A": int((dn["vent0"] & (A == 1)).sum()), "vent_B": int((dn["vent0"] & (A == 0)).sum())},
            "diag": {"auc_logistic_in_sample": r10.auc(ps_lr, A), "auc_gbm_cross_fitted": r10.auc(e, A),
                     "max_abs_smd_gbm_ow": max(abs(v) for v in smd.values() if np.isfinite(v)),
                     "ess_gbm_ow": {"A": sc.ess(ow[A == 1]), "B": sc.ess(ow[A == 0])}, "features": int(nw_features(dn).shape[1])}}


def estimate(args):
    t0 = time.time()
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    out = {"plan": PLAN, "db": {}}
    good = True
    data = {}
    for db in ("m", "e"):
        d, df, flow = load_ppi(P, db)
        good &= r7_check(d, db, R)
        data[db] = (d, df, flow)
    dn = load_nw(P)
    good &= nw_check(dn, R)
    if not good:
        asd.log_run(args.results, "r11 estimate: REPRODUCTION CHECK FAILED")
        raise SystemExit("a rebuilt input differs from R7 or R9: stop")
    for db in ("m", "e"):
        d, df, flow = data[db]
        cache = {}
        out["db"][db] = est_ppi(d, df, flow, db, cache)
        np.savez(os.path.join(P, f"r11_cache_{db}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
        asd.status(f"r11 estimate {db} done", t0)
    out["nw"] = est_nw(dn)
    json.dump(asd.to_json(out), open(os.path.join(P, "r11_est.json"), "w"))
    asd.status("r11 estimate NWICU done", t0)
    asd.log_run(args.results, f"r11 estimate: ok, reproduction checks passed ({time.time() - t0:.0f} s)")


_G = {}


def _one_ppi(rep):
    d, db, cache = _G["d"], _G["db"], _G["cache"]
    try:
        idx, copy = r7.draw(d, db, np.random.default_rng(asd.SEED + rep))
        b = asd.take(d, idx)
        if copy is not None:
            b["site"] = b["site"] * 1000 + copy
        return rep, flat(analyses_all(b, db, dict(cache)), db)
    except Exception:
        return rep, None


def _one_nw(rep):
    d = _G["d"]
    try:
        n = len(d["A"])
        idx = np.random.default_rng(asd.SEED + rep).integers(0, n, n)
        b = asd.take(d, idx)
        if _G["part"] == "nw":
            return rep, nw_flat(nw_strata(b), NW_STRATA_NAMES)
        res, _, _ = nw_flex(b, r10.rep_folds(n, rep, idx), r10.gbm)
        return rep, nw_flat(res, NW_FLEX_NAMES)
    except Exception:
        return rep, None


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    part = args.part
    if part in ("m", "e"):
        d, _, _ = load_ppi(P, part)
        z = np.load(os.path.join(P, f"r11_cache_{part}.npz"))
        _G.update(d=d, db=part, cache={k.replace("__", ":"): z[k] for k in z.files}, part=part)
        fn, names = _one_ppi, flat_names(part)
    else:
        _G.update(d=load_nw(P), part=part)
        fn, names = _one_nw, (NW_STRATA_NAMES if part == "nw" else NW_FLEX_NAMES)
    reps = list(range(args.start, min(args.stop, NBOOT)))
    out = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for rep, v in pool.imap_unordered(fn, reps, chunksize=2 if part == "nwflex" else 5):
                out[rep] = v
    else:
        for rep in reps:
            out[rep] = fn(rep)[1]
    V = np.full((len(reps), len(names)), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None:
            V[i] = out[rep]
    np.savez(os.path.join(P, f"r11boot_{part}_{args.start:04d}_{args.stop:04d}.npz"), reps=np.array(reps), v=V, names=np.array(names))
    failed = sum(1 for r in reps if out.get(r) is None)
    asd.status(f"r11 boot {part} {args.start}-{args.stop} done, failed {failed}", t0)
    asd.log_run(args.results, f"r11 boot --part {part} --start {args.start} --stop {args.stop}: ok, failed {failed} ({time.time() - t0:.0f} s)")


def synth_nw(n, p, rng):
    X = np.column_stack([rng.normal(size=(n, p // 2)), (rng.random((n, p - p // 2)) < 0.3).astype(float)])
    A = (rng.random(n) < 1 / (1 + np.exp(-(0.6 + 0.4 * X[:, 0] - 0.3 * X[:, 1])))).astype(int)
    t_icu = rng.uniform(2, 240, n)
    d = {"X": X, "names": [f"x{j}" for j in range(p)], "dropped": [], "A": A, "site": None, "sub": {}, "t_dev": np.full(n, np.nan),
         "t_end_icu": t_icu, "end_death_icu": rng.random(n) < 0.08, "t_end_hosp": t_icu + rng.uniform(0, 200, n),
         "end_death_hosp": rng.random(n) < 0.1, "t_ap": np.where(rng.random(n) < 0.08, rng.uniform(0, 160, n), np.nan),
         "t_oq": np.full(n, np.nan), "t_hypo": np.where(rng.random(n) < 0.06, rng.uniform(0, 160, n), np.nan),
         "vent0": rng.random(n) < 0.35, "lac": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(0.5, 6, n)),
         "extras": {"age_years": rng.uniform(18, 95, n), "egfr": np.where(rng.random(n) < 0.05, np.nan, rng.uniform(10, 130, n)),
                    "h_icu": rng.uniform(0, 100, n)}}
    d["t_end"], d["end_death"] = d["t_end_icu"], d["end_death_icu"]
    return d


def timing(args):
    R = os.path.expanduser(args.results)
    n = int(json.load(open(os.path.join(R, "r9_design.json"), encoding="utf-8"))["n"]["flow"]["patients"])
    d = synth_nw(n, 45, np.random.default_rng(11))
    t0 = time.time()
    for rep in range(2):
        idx = np.random.default_rng(rep).integers(0, n, n)
        nw_flex(asd.take(d, idx), r10.rep_folds(n, rep, idx), r10.gbm)
    per = (time.time() - t0) / 2
    mins = per / 2 / 60
    reps = next((r for r in (2000, 1000, 500, 200) if r * mins <= FLEX_LIMIT_MIN), 200)
    lines = [f"NWICU: n {n} (masked), {per:.1f} s per replicate on one core (3 outcomes, 20 boosted fits)",
             f"projected minutes for 2000 / 1000 / 500 / 200 replicates (two workers): {2000 * mins:.0f} / {1000 * mins:.0f} / "
             f"{500 * mins:.0f} / {200 * mins:.0f}; limit {FLEX_LIMIT_MIN:.0f}", f"replicates: {reps}"]
    open(os.path.join(R, "r11_timing.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def load_boots(P, part):
    files = sorted(glob.glob(os.path.join(P, f"r11boot_{part}_*.npz")))
    if not files:
        return None
    reps, vs, names = [], [], None
    for f in files:
        z = np.load(f)
        reps.append(z["reps"])
        vs.append(z["v"])
        names = [str(x) for x in z["names"]]
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return {"names": names, "v": np.concatenate(vs)[first], "n": int(first.size)}


def bcol(B, name):
    if B is None or name not in B["names"]:
        return None
    v = B["v"][:, B["names"].index(name)]
    v = v[np.isfinite(v)]
    return v if v.size >= asd.MIN_BOOT else None


def with_ci(point, B, name, key="OR"):
    e = {key: point}
    v = bcol(B, name)
    if v is not None and point is not None:
        e["lo"], e["hi"], e["se_log"], e["boot_ok"] = float(np.exp(np.percentile(v, 2.5))), float(np.exp(np.percentile(v, 97.5))), \
            float(v.std(ddof=1)), int(v.size)
    return e


def paired_ratio(B, num, den, pnum, pden):
    if B is None or pnum is None or pden is None or not (np.isfinite(pnum) and np.isfinite(pden)) or pden <= 0:
        return None
    a, b = B["v"][:, B["names"].index(num)], B["v"][:, B["names"].index(den)]
    v = (a - b)[np.isfinite(a) & np.isfinite(b)]
    if v.size < asd.MIN_BOOT:
        return None
    return {"ratio": pnum / pden, "lo": float(np.exp(np.percentile(v, 2.5))), "hi": float(np.exp(np.percentile(v, 97.5)))}


def indep_ratio(num, den):
    if not num or not den or num.get("se_log") is None or den.get("se_log") is None:
        return None
    lr = math.log(num["HR"]) - math.log(den["HR"])
    se = math.sqrt(num["se_log"] ** 2 + den["se_log"] ** 2)
    return {"ratio": math.exp(lr), "lo": math.exp(lr - 1.96 * se), "hi": math.exp(lr + 1.96 * se)}


def pool2(a, b, key="OR"):
    ok = [x for x in (a, b) if x and x.get("se_log") and x.get(key) and x[key] > 0]
    if len(ok) < 2:
        return None
    m, se, Q, I2, k = sc.pool_fe([math.log(x[key]) for x in ok], [x["se_log"] for x in ok])
    return {key: math.exp(m), "lo": math.exp(m - 1.96 * se), "hi": math.exp(m + 1.96 * se), "I2": I2}


def inside(x, lo, hi):
    return x is not None and lo is not None and hi is not None and lo <= x <= hi


def read_screen(rm, re_, ratio):
    if not ratio or re_.get("hi") is None:
        return "inconclusive"
    if ratio["lo"] > 1 or ratio["hi"] < 1:
        return "the difference between the databases remains when assessment times are taken into account: assessment frequency and time in the ICU do not explain it"
    if re_["hi"] < 1:
        return "the databases agree when assessment times are taken into account: the difference in the main analysis comes from assessment frequency and time in the ICU"
    return "inconclusive"


def read_postop(po, hosp):
    if not po or po.get("hi") is None:
        return "inconclusive"
    if po["hi"] < 1:
        if hosp and hosp.get("hi") is not None and hosp["hi"] < 1:
            return "association in the post-surgical population with both CAM-ICU and the proxy outcome: the negative INSPIRE and MOVER results cannot be explained by the proxy outcome or the post-surgical setting"
        if hosp and hosp.get("lo") is not None and hosp["lo"] <= 1 <= hosp["hi"]:
            return "association in the post-surgical population with CAM-ICU, not reproduced by the proxy outcome: the negative INSPIRE and MOVER results may come from the proxy outcome"
        return "association in the post-surgical population, proxy outcome inconclusive"
    if po["lo"] <= 1 <= po["hi"] and po["lo"] > asd.MCID:
        return "no association in the post-surgical population: supports a difference by setting"
    return "inconclusive"


def read_late(e):
    if not e or e.get("hi") is None:
        return "inconclusive"
    if e["hi"] < 1:
        return "MIMIC-IV in the same period as NWICU still shows an association: the negative NWICU result cannot be explained by calendar period"
    if e["lo"] <= 1 <= e["hi"] and e["lo"] > asd.MCID:
        return "MIMIC-IV in the same period shows no association either: the negative NWICU result may reflect calendar period"
    return "inconclusive"


def mask_tree(o, key=""):
    if isinstance(o, dict):
        return {k: mask_tree(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [mask_tree(v, key) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)) and re.match(r"^(N|NA|NB|nA|nB|eventsA|eventsB|n_inf_A|n_inf_B|A|B|proxy_A|proxy_B|"
                                                     r"eligible|other|outside_overlap|analysis_A|analysis_B|"
                                                     r"novent_A|novent_B|vent_A|vent_B|fam_.*|surgical|cvicu|postop|rass_.*|sed_inf|"
                                                     r"benzo_inf|dex_inf|vent|route_enteral|egfr_lt60|age_ge65|steroid_24h)$", key):
        return sc.mask_n(int(o))
    return o


def report(args):
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    est = json.load(open(os.path.join(P, "r11_est.json")))
    c2 = json.load(open(os.path.join(R, "ppi_estimates.json"), encoding="utf-8"))
    r9j = json.load(open(os.path.join(R, "r9.json"), encoding="utf-8"))
    mover = json.load(open(os.path.join(R, "e2_mover.json"), encoding="utf-8"))["primary"]
    Bs = {p: load_boots(P, p) for p in ("m", "e", "nw", "nwflex")}
    out = {"plan": PLAN, "boot_reps": {p: (B["n"] if B else 0) for p, B in Bs.items()}, "db": {}, "reading": {}}
    for db in ("m", "e"):
        E, B = est["db"][db]["res"], Bs[db]
        o = {"flow": est["db"][db]["flow"], "ic_informative": est["db"][db]["ic_informative"]}
        for nm in ("primary", "negctl", "m3_primary", "m3_negctl"):
            o[nm] = dict(with_ci(E[nm]["OR"], B, nm), riskA=E[nm]["riskA"], riskB=E[nm]["riskB"], eventsA=E[nm]["eventsA"],
                         eventsB=E[nm]["eventsB"], nA=E[nm]["nA"], nB=E[nm]["nB"])
        for nm in ("ic", "ic6", "ic_rass3"):
            o[nm] = dict(with_ci(E[nm].get("HR"), B, nm, "HR"), riskA=E[nm].get("riskA"), riskB=E[nm].get("riskB"),
                         n_inf_A=E[nm].get("n_inf_A"), n_inf_B=E[nm].get("n_inf_B"), converged=E[nm].get("converged"))
        o["screen"] = {k: dict(E["screen"][k], **{x: y for x, y in with_ci(E["screen"][k]["ratio"], B, "screen_" + k, "ratio").items() if x != "ratio"})
                       for k in E["screen"]}
        o["screen_unweighted"] = est["db"][db]["screen_unweighted"]
        o["decomposition"] = est["db"][db]["decomposition"]
        o["newcov"] = est["db"][db]["newcov"]
        o["m3_columns_dropped"] = est["db"][db]["m3_columns_dropped"]
        main = c2["per_db"][db]["analyses"]
        o["main"] = {nm: {k: main[nm].get(k) for k in ("OR", "OR_lo", "OR_hi")} for nm in ("primary", "negctl")}
        if db == "m":
            for nm in PO_NAMES:
                x = E.get(nm)
                o[nm] = dict(with_ci(x["OR"], B, nm), riskA=x["riskA"], riskB=x["riskB"], eventsA=x["eventsA"], eventsB=x["eventsB"],
                             nA=x["nA"], nB=x["nB"]) if x else None
            o["postop_n"] = est["db"][db]["postop_n"]
            o["periods"] = {}
            for p, _ in PERIODS:
                for oc in PER_OUT:
                    x = E.get(f"{p}_{oc}")
                    o["periods"][f"{p}_{oc}"] = dict(with_ci(x["OR"], B, f"{p}_{oc}"), eventsA=x["eventsA"], eventsB=x["eventsB"],
                                                     nA=x["nA"], nB=x["nB"]) if x else None
            o["period_ratio"] = {f"{p}_{oc}": paired_ratio(B, f"{p}_{oc}", f"early_{oc}", (E.get(f"{p}_{oc}") or {}).get("OR"),
                                                           (E.get(f"early_{oc}") or {}).get("OR"))
                                 for p in ("mid", "late") for oc in PER_OUT}
            o["period_desc"] = est["db"][db]["periods"]
            o["period_n"] = est["db"][db]["period_n"]
        out["db"][db] = o
    M, Ee = out["db"]["m"], out["db"]["e"]
    out["ic_ratio_e_over_m"] = indep_ratio(Ee["ic"], M["ic"])
    out["ic_pooled"] = pool2(M["ic"], Ee["ic"], "HR")
    out["m3_pooled"] = {nm: pool2(M[nm], Ee[nm]) for nm in ("m3_primary", "m3_negctl")}
    out["reading"]["screening"] = read_screen(M["ic"], Ee["ic"], out["ic_ratio_e_over_m"])
    dec = {}
    for db in ("m", "e"):
        D = out["db"][db]["decomposition"]
        big = max(D["blocks"].items(), key=lambda kv: abs(math.log(kv[1]["minus"]) - math.log(D["full"])))
        dec[db] = f"{DBNAME[db]}: without “{big[1]['label']}”, odds ratio {D['full']:.2f} → {big[1]['minus']:.2f} (largest change)"
    out["reading"]["decomposition"] = dec
    out["reading"]["postop"] = read_postop(M["po_primary"], M["po_px_hosp"])
    out["external_postop"] = {"INSPIRE_icu": r9j["per_db"]["INSPIRE"]["icu"], "INSPIRE_hosp": r9j["per_db"]["INSPIRE"]["hosp"],
                              "MOVER_hosp": {"OR": mover["OR"], "OR_lo": mover["OR_lo"], "OR_hi": mover["OR_hi"]},
                              "NWICU_icu": r9j["per_db"]["NWICU"]["icu"], "NWICU_hosp": r9j["per_db"]["NWICU"]["hosp"],
                              "NWICU_negctl": r9j["per_db"]["NWICU"]["negctl"]}
    out["reading"]["late_primary"] = read_late(M["periods"]["late_primary"])
    out["reading"]["late_px_icu"] = read_late(M["periods"]["late_px_icu"])
    rt = M["period_ratio"]["late_primary"]
    out["reading"]["trend"] = ("the association changes with calendar period" if rt and (rt["hi"] < 1 or rt["lo"] > 1) else "no evidence that the association changes with calendar period")
    m3 = {}
    for db in ("m", "e"):
        o = out["db"][db]
        m3[db] = ("adding corticosteroids and sedative type does not change the estimate" if inside(o["m3_primary"]["OR"], o["main"]["primary"]["OR_lo"], o["main"]["primary"]["OR_hi"])
                  else "the additional adjustment changes the estimate")
    out["reading"]["m3"] = m3
    nw, BN, BF = est["nw"], Bs["nw"], Bs["nwflex"]
    N = {"strata": {}, "flex": {}, "vent_n": nw["vent_n"], "diag": nw["diag"]}
    for lv in ("all", "novent", "vent"):
        for oc in NW_OUT:
            x = nw["strata"][lv][oc]
            N["strata"][f"{lv}_{oc}"] = dict(with_ci(x["OR"], BN, f"{lv}_{oc}"), eventsA=x.get("eventsA"), eventsB=x.get("eventsB"),
                                             nA=x.get("nA"), nB=x.get("nB"))
    N["vent_ratio_icu"] = paired_ratio(BN, "novent_icu", "vent_icu", nw["strata"]["novent"]["icu"]["OR"], nw["strata"]["vent"]["icu"]["OR"])
    for nm in NW_FLEX_NAMES:
        x = nw["flex"][nm]
        N["flex"][nm] = dict(with_ci(x["OR"], BF, nm), riskA=x.get("riskA"), riskB=x.get("riskB"))
    out["nw"] = N
    r9n = r9j["per_db"]["NWICU"]
    nc_flex = [N["flex"][f"{k}_negctl"] for k in ("mlps", "dr")]
    main_nc_sig = r9n["negctl"].get("OR_lo") is not None and (r9n["negctl"]["OR_lo"] > 1 or r9n["negctl"]["OR_hi"] < 1)
    nc_gone = main_nc_sig and all(x.get("lo") is not None and x["lo"] <= 1 <= x["hi"] for x in nc_flex)
    out["reading"]["nw_negctl"] = ("the NWICU negative-control signal disappears with the form of the model" if nc_gone else
                                   "the NWICU negative-control signal remains under at least one flexible model" if main_nc_sig else "no NWICU negative-control signal in the main analysis")
    icu_same = all(inside(N["flex"][f"{k}_icu"]["OR"], r9n["icu"].get("OR_lo"), r9n["icu"].get("OR_hi")) for k in ("mlps", "dr"))
    out["reading"]["nw_flex_icu"] = "the NWICU proxy outcome does not depend on the form of the model" if icu_same else "the NWICU proxy outcome changes with the form of the model"
    nv = N["strata"]["novent_icu"]
    out["reading"]["nw_novent"] = ("inconclusive" if nv.get("hi") is None else "association in NWICU also in patients not ventilated" if nv["hi"] < 1 else
                                   "opposite direction" if nv["lo"] > 1 else "no association in NWICU in patients not ventilated either")
    masked = mask_tree(asd.to_json(out))
    json.dump(asd.round_floats(masked), open(os.path.join(R, "r11.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(R, "r11.md"), "w", encoding="utf-8").write(md(masked))
    asd.log_run(args.results, "r11 report: ok")
    print("report written", flush=True)


def f2(x, nd=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def ci(e, key="OR"):
    if not e or e.get(key) is None:
        return "—"
    return f2(e[key]) + (f" ({f2(e['lo'])}–{f2(e['hi'])})" if e.get("lo") is not None else "")


def ci3(e, a="OR", lo="OR_lo", hi="OR_hi"):
    if not e or e.get(a) is None:
        return "—"
    return f"{f2(e[a])} ({f2(e.get(lo))}–{f2(e.get(hi))})"


def rci(r):
    return "—" if not r else f"{f2(r['ratio'])} ({f2(r['lo'])}–{f2(r['hi'])})"


def pc(num, den):
    try:
        nn, dd = int(num), int(den)
    except (TypeError, ValueError):
        return f"{num}"
    return f"{sc.mask_n(nn)} ({sc.pct_text(nn, dd)})"


def md(o):
    M, E = o["db"]["m"], o["db"]["e"]
    L = ["# Revision R11 results: assessment frequency, post-surgical population, calendar period, NWICU and added covariates (PPI vs famotidine)", "",
         f"Plan: {o['plan']}. Bootstrap: MIMIC-IV {o['boot_reps']['m']}, eICU {o['boot_reps']['e']} (paired across sections 1, 2, 3 and 5), NWICU strata "
         f"{o['boot_reps']['nw']}, NWICU flexible models {o['boot_reps']['nwflex']}. Odds ratio <1: lower in the PPI arm. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", ""]
    L += ["## 1. Assessment frequency and estimates that take assessment times into account", "", "### 1a Assessments before the first positive (overlap weighted; unweighted in brackets)", "",
          "| Measure | MIMIC-IV PPI | MIMIC-IV famotidine | MIMIC-IV ratio | eICU PPI | eICU famotidine | eICU ratio |", "|---|---|---|---|---|---|---|"]
    for k, lab, nd in (("scr_any", "Assessments per ICU day (any value)", 2), ("scr_ok", "Assessable assessments per ICU day", 2),
                       ("blk_ok_share", "Share of 12-hour blocks at risk with an assessable assessment", 2)):
        cells = []
        for x in (M, E):
            s, u = x["screen"][k], x["screen_unweighted"][k]
            cells += [f"{f2(s['A'], nd)} ({f2(u['A'], nd)})", f"{f2(s['B'], nd)} ({f2(u['B'], nd)})",
                      f"{f2(s['ratio'])} ({f2(s.get('lo'))}–{f2(s.get('hi'))})"]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "### 1b Onset model that takes assessment times into account (interval-censored, continuous time; ICU discharge and death as censoring)", "",
          "| Estimate | MIMIC-IV | eICU | Pooled (I²) |", "|---|---|---|---|",
          f"| Main analysis (odds ratio of cumulative incidence, reference) | {ci3(M['main']['primary'])} | {ci3(E['main']['primary'])} | — |"]
    for nm, lab in (("ic", "Onset hazard ratio (baseline in 3 segments)"), ("ic6", "Onset hazard ratio (baseline in 6 segments)"), ("ic_rass3", "Onset hazard ratio, RASS ≥ −3 at time zero")):
        pool = o["ic_pooled"] if nm == "ic" else None
        L.append(f"| {lab} | {ci(M[nm], 'HR')} | {ci(E[nm], 'HR')} | "
                 f"{(ci(pool, 'HR') + '; I² ' + f2(100 * pool['I2'], 0) + '%') if pool else '—'} |")
    L.append(f"| 168-hour onset risk had everyone stayed in the ICU, PPI / famotidine (baseline in 3 segments) | {f2(100 * M['ic']['riskA'], 1)}% / {f2(100 * M['ic']['riskB'], 1)}% | "
             f"{f2(100 * E['ic']['riskA'], 1)}% / {f2(100 * E['ic']['riskB'], 1)}% | — |")
    L.append(f"| Informative patients PPI / famotidine | {M['ic_informative']['A']} / {M['ic_informative']['B']} | "
             f"{E['ic_informative']['A']} / {E['ic_informative']['B']} | — |")
    L += ["", f"- Ratio of the onset hazard ratios of the databases (eICU ÷ MIMIC-IV): {rci(o['ic_ratio_e_over_m'])}", f"- Reading (written rule): {o['reading']['screening']}", "",
          "### 1c Which covariate group moves the estimate from unweighted to weighted (main-analysis odds ratio, point estimate)", "",
          "| Covariate group | Columns MIMIC-IV / eICU | MIMIC-IV without the group | MIMIC-IV with the group only | eICU without the group | eICU with the group only |", "|---|---|---|---|---|---|"]
    keys = [k for k, _, _ in BLOCKS]
    for k in keys:
        bm, be = M["decomposition"]["blocks"].get(k), E["decomposition"]["blocks"].get(k)
        lab = (bm or be)["label"]
        L.append(f"| {lab} | {bm['columns'] if bm else 0} / {be['columns'] if be else 0} | {f2(bm['minus']) if bm else '—'} | {f2(bm['only']) if bm else '—'} | "
                 f"{f2(be['minus']) if be else '—'} | {f2(be['only']) if be else '—'} |")
    L.append(f"| Unweighted / full model | — | {f2(M['decomposition']['crude'])} / {f2(M['decomposition']['full'])} | | "
             f"{f2(E['decomposition']['crude'])} / {f2(E['decomposition']['full'])} | |")
    L += ["", "- " + "; ".join(o["reading"]["decomposition"].values()), ""]
    pn = M["postop_n"]
    L += ["## 2. MIMIC-IV post-surgical population (defined as in INSPIRE and MOVER)", "",
          f"Post-surgical population PPI {pn['A']} / famotidine {pn['B']}; of whom in the proxy subset {pn['proxy_A']} / {pn['proxy_B']}.", "",
          "| Outcome | Events PPI / famotidine | Weighted risk PPI / famotidine | OR (95% CI) |", "|---|---|---|---|"]
    for nm, lab in (("po_primary", "Delirium (CAM-ICU, as the main analysis)"), ("po_negctl", "negative control: glucose <70 mg/dL"), ("po_px_cam", "Delirium in the proxy subset (CAM-ICU)"),
                    ("po_px_icu", "First antipsychotic in the ICU in the proxy subset (haloperidol within 6 h after the PACU not counted)"),
                    ("po_px_hosp", "First antipsychotic in the hospital in the proxy subset (same rule)")):
        x = M.get(nm)
        if not x:
            L.append(f"| {lab} | — | — | — |")
            continue
        L.append(f"| {lab} | {x['eventsA']} / {x['eventsB']} | {f2(100 * x['riskA'], 1)}% / {f2(100 * x['riskB'], 1)}% | {ci(x)} |")
    ex = o["external_postop"]
    L += ["", f"- For comparison: INSPIRE in the ICU {ci3(ex['INSPIRE_icu'])}, hospital {ci3(ex['INSPIRE_hosp'])}; MOVER hospital {ci3(ex['MOVER_hosp'])}",
          f"- Reading (written rule): {o['reading']['postop']}", ""]
    L += ["## 3. MIMIC-IV by calendar period", "", "| Period | Patients PPI / famotidine | Delirium | Negative control | Proxy ICU | Proxy hospital |", "|---|---|---|---|---|---|"]
    for p, _ in PERIODS:
        n = M["period_n"][p]
        L.append(f"| {PERIOD_LAB[p]} | {n['A']} / {n['B']} | " + " | ".join(ci(M["periods"].get(f"{p}_{oc}")) for oc in PER_OUT) + " |")
    for p in ("mid", "late"):
        L.append(f"| {PERIOD_LAB[p]} ÷ 2011–2016 | — | " + " | ".join(rci(M["period_ratio"].get(f"{p}_{oc}")) for oc in PER_OUT) + " |")
    L += ["", f"- For comparison: NWICU (2020–2022) ICU proxy {ci3(ex['NWICU_icu'])}, hospital {ci3(ex['NWICU_hosp'])}",
          f"- Reading (written rule): delirium {o['reading']['late_primary']}; proxy (ICU) {o['reading']['late_px_icu']}; {o['reading']['trend']}", "",
          "| Description (unweighted) | " + " | ".join(f"{PERIOD_LAB[p]} {a}" for p, _ in PERIODS for a in ("PPI", "famotidine")) + " |",
          "|---|" + "---|" * 6]
    rows = [("surgical", "surgical"), ("CVICU", "cvicu"), ("Post-surgical (as in section 2)", "postop"), ("RASS ≤ −2 at time zero", "rass_le_m2"), ("RASS not recorded", "rass_missing"),
            ("Sedative infusion", "sed_inf"), ("Benzodiazepine infusion", "benzo_inf"), ("Dexmedetomidine infusion", "dex_inf"), ("Invasive ventilation", "vent"), ("First dose enteral", "route_enteral"),
            ("eGFR <60", "egfr_lt60"), ("≥65 years", "age_ge65"), ("Systemic corticosteroids within 24 h", "steroid_24h")]
    PD = M["period_desc"]
    for lab, k in rows:
        L.append(f"| {lab} | " + " | ".join(pc(PD[f"{p}:{a}"][k], PD[f"{p}:{a}"]["N"]) for p, _ in PERIODS for a in ("A", "B")) + " |")
    L.append("| Assessments per ICU day before the first positive | " + " | ".join(f2(PD[f"{p}:{a}"]["screens_per_day"]) for p, _ in PERIODS for a in ("A", "B")) + " |")
    L.append("| Famotidine planned daily dose ≥40 mg (all; eGFR <60) | " + " | ".join(
        ("—" if a == "A" else f"{pc(PD[f'{p}:B']['fam_ge40'], PD[f'{p}:B']['fam_known'])}; {pc(PD[f'{p}:B']['fam_ge40_egfr60'], PD[f'{p}:B']['fam_known_egfr60'])}")
        for p, _ in PERIODS for a in ("A", "B")) + " |")
    L.append("")
    N = o["nw"]
    vn = N["vent_n"]
    L += ["## 4. NWICU", "", "| Estimate | First antipsychotic in the ICU | First antipsychotic in the hospital | Negative control |", "|---|---|---|---|",
          f"| R9 main analysis (logistic propensity score, overlap weighting) | {ci3(ex['NWICU_icu'])} | {ci3(ex['NWICU_hosp'])} | {ci3(ex['NWICU_negctl'])} |",
          "| 4a gradient-boosted propensity score, overlap weighting | " + " | ".join(ci(N["flex"][f"mlps_{oc}"]) for oc in NW_OUT) + " |",
          "| 4a doubly robust (augmented overlap weighting) | " + " | ".join(ci(N["flex"][f"dr_{oc}"]) for oc in NW_OUT) + " |",
          f"| 4b no invasive ventilation at time zero (PPI {vn['novent_A']} / famotidine {vn['novent_B']}) | " + " | ".join(ci(N["strata"][f"novent_{oc}"]) for oc in NW_OUT) + " |",
          f"| 4b invasive ventilation at time zero (PPI {vn['vent_A']} / famotidine {vn['vent_B']}) | " + " | ".join(ci(N["strata"][f"vent_{oc}"]) for oc in NW_OUT) + " |",
          "", f"- Stratum not ventilated ÷ ventilated (ICU proxy outcome): {rci(N['vent_ratio_icu'])}",
          f"- Propensity score AUC: logistic regression (in sample) {f2(N['diag']['auc_logistic_in_sample'], 3)}, gradient boosting (cross-fitted) {f2(N['diag']['auc_gbm_cross_fitted'], 3)}; "
          f"largest |SMD| after gradient-boosted weighting {f2(N['diag']['max_abs_smd_gbm_ow'], 3)}; effective sample size PPI {f2(N['diag']['ess_gbm_ow']['A'], 0)}, famotidine {f2(N['diag']['ess_gbm_ow']['B'], 0)}",
          f"- Reading (written rule): {o['reading']['nw_negctl']}; {o['reading']['nw_flex_icu']}; {o['reading']['nw_novent']}", ""]
    L += ["## 5. Added covariates: systemic corticosteroids and sedative type", "", "| Covariate | MIMIC-IV PPI | MIMIC-IV famotidine | SMD unweighted / weighted | eICU PPI | eICU famotidine | SMD unweighted / weighted |",
          "|---|---|---|---|---|---|---|"]
    for c, lab in (("steroid_24h", "Systemic corticosteroids within 24 h"), ("benzo_inf", "Benzodiazepine infusion"), ("dex_inf", "Dexmedetomidine infusion"), ("ketamine_inf", "Ketamine infusion")):
        cells = []
        for x in (M, E):
            v = x["newcov"][c]
            cells += [pc(v["nA"], v["NA"]), pc(v["nB"], v["NB"]), f"{f2(v['smd_unweighted'], 3)} / {f2(v['smd_ow'], 3)}"]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "| Outcome | Model | MIMIC-IV | eICU | Pooled (I²) |", "|---|---|---|---|---|"]
    for oc, lab in (("primary", "delirium"), ("negctl", "negative control")):
        L.append(f"| {lab} | Main analysis | {ci3(M['main'][oc])} | {ci3(E['main'][oc])} | — |")
        pool = o["m3_pooled"][f"m3_{oc}"]
        L.append(f"| {lab} | M3 (corticosteroids and sedative type added) | {ci(M[f'm3_{oc}'])} | {ci(E[f'm3_{oc}'])} | "
                 f"{(ci(pool) + '; I² ' + f2(100 * pool['I2'], 0) + '%') if pool else '—'} |")
    dropped = "; ".join(f"{DBNAME[db]} " + (", ".join(o["db"][db]["m3_columns_dropped"]) or "no") for db in ("m", "e"))
    L += ["", f"- New columns dropped by the low-prevalence rule: {dropped}", f"- Reading (written rule): MIMIC-IV {o['reading']['m3']['m']}; eICU {o['reading']['m3']['e']}", ""]
    return "\n".join(L) + "\n"


def sim_screening(n, hr, rate, rng, lam=(0.0025, 0.0038, 0.0025)):
    A = (rng.random(n) < 0.5).astype(int)
    cuts = SEG_H[3]
    target = rng.exponential(1.0, n) / np.where(A == 1, hr, 1.0)
    onset = np.full(n, np.nan)
    cum = 0.0
    for s in range(3):
        a, b = cuts[s], cuts[s + 1]
        within = np.isnan(onset) & (target <= cum + lam[s] * (b - a))
        onset[within] = a + (target[within] - cum) / lam[s]
        cum += lam[s] * (b - a)
    t_end = np.minimum(rng.exponential(80, n) + 1, 400)
    T = np.minimum(t_end, 168.0)
    k = rng.poisson(np.where(A == 1, rate[0], rate[1]) / 24.0 * T)
    idx = np.repeat(np.arange(n), k)
    hh = rng.uniform(1e-6, 1.0, idx.size) * T[idx]
    okk = rng.random(idx.size) >= 0.1
    p = okk & np.isfinite(onset[idx]) & (onset[idx] <= hh)
    tl = pd.DataFrame({"stay_id": idx, "h": hh, "pos": p.astype(int), "ok": okk.astype(int)})
    first = tl[tl["pos"] == 1].groupby("stay_id")["h"].min()
    t_del = np.full(n, np.nan)
    t_del[first.index.values] = first.values
    d = {"A": A, "t_end": t_end, "end_death": np.zeros(n, bool), "t_del": t_del, "t_dev": np.full(n, np.nan)}
    return d, np.arange(n), tl


def selftest(args):
    res = []

    def check(name, ok, detail=""):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name}{' | ' + detail if detail else ''}", flush=True)

    rng = np.random.default_rng(5)
    d = {"t_end": np.array([100.0, 200.0, 30.0]), "t_del": np.array([50.0, np.nan, np.nan]), "A": np.array([1, 0, 1])}
    tl = pd.DataFrame({"stay_id": [10, 10, 10, 10, 20, 20, 30], "h": [5, 30, 40, 50, 20, 160, 25],
                       "pos": [0, 0, 0, 1, 0, 0, 0], "ok": [1, 1, 0, 1, 1, 1, 0]})
    a = timeline_arrays(d, np.array([10, 20, 30]), tl)
    good = (list(a["ic_tL"]) == [30.0, 160.0, 0.0] and a["ic_tR"][0] == 50.0 and np.isnan(a["ic_tR"][1:]).all()
            and list(a["ic_det"]) == [True, False, False] and list(a["scr_any"]) == [4, 2, 1] and list(a["scr_ok"]) == [3, 2, 0])
    check("last negative, first positive and screen counts from a hand-made timeline", good,
          json.dumps({"tL": list(map(float, a["ic_tL"])), "scr_any": list(map(int, a["scr_any"])), "scr_ok": list(map(int, a["scr_ok"]))}))
    dd, sid, tl = sim_screening(120000, 0.8, (2.0, 1.4), rng)
    dd.update(timeline_arrays(dd, sid, tl))
    w = np.ones(len(sid))
    e = ic_fit(dd, w, 3)
    ps = per_screen(dd, w)
    check("onset model recovers the hazard ratio 0.80 with unequal screening (PPI 2.0 vs 1.4 per day)", abs(math.log(e["HR"]) - math.log(0.8)) < 0.05,
          f"{e['HR']:.3f}; per-screen rate ratio {ps:.3f}")
    check("the per-screen rate ratio is biased by unequal screening (further from 0.80 than the onset model)",
          abs(math.log(ps) - math.log(0.8)) > abs(math.log(e["HR"]) - math.log(0.8)) + 0.1, f"{ps:.3f} vs {e['HR']:.3f}")
    e6 = ic_fit(dd, w, 6)
    check("6-segment baseline gives a similar estimate", abs(math.log(e6["HR"]) - math.log(e["HR"])) < 0.03, f"{e6['HR']:.3f}")
    dd, sid, tl = sim_screening(120000, 1.0, (1.4, 2.0), rng)
    dd.update(timeline_arrays(dd, sid, tl))
    e = ic_fit(dd, np.ones(len(sid)), 3)
    check("no effect: onset model near 1 with screening unequal the other way", abs(math.log(e["HR"])) < 0.05,
          f"{e['HR']:.3f}; per-screen rate ratio {per_screen(dd, np.ones(len(sid))):.3f}")
    dt = {"A": np.array([1, 1, 0, 0]), "scr_any": np.array([4.0, 2.0, 1.0, 1.0]), "scr_ok": np.array([2.0, 2.0, 1.0, 0.0]),
          "scr_days": np.array([2.0, 1.0, 1.0, 1.0]), "blk_ok": np.array([2.0, 1.0, 1.0, 0.0]), "blk_risk": np.array([4.0, 2.0, 2.0, 2.0])}
    s = screening(dt, np.ones(4))
    check("screening rates (ratio estimators)", abs(s["scr_any"]["A"] - 2.0) < 1e-12 and abs(s["scr_any"]["B"] - 1.0) < 1e-12
          and abs(s["blk_ok_share"]["ratio"] - (3 / 6) / (1 / 4)) < 1e-12)
    S = {}
    for db in ("m", "e"):
        d0 = r10.synth_d(3000, 12, rng, db)
        n = 3000
        d0["names"] = ["age_ge65", "female", "egfr_per10", "sofa_ge3" if db == "m" else "apache_ge60", "vent", "rass_le_m2", "sed_inf",
                       "unit_SICU", "surgical", "dementia", "liver", "hosp_1" if db == "e" else "era_2017_2019"]
        d0["X"] = (rng.random((n, 12)) < 0.3).astype(float)
        d0["sub"] = {"rass3": rng.random(n) < 0.7}
        d0["t_ap_icu"] = np.where(rng.random(n) < 0.08, rng.uniform(0, 150, n), np.nan)
        d0["ap_pre"], d0["psych"] = rng.random(n) < 0.05, rng.random(n) < 0.03
        d0["X3"] = np.column_stack([d0["X"], rng.random(n) < 0.2])
        d0["names3"] = d0["names"] + ["steroid_24h"]
        if db == "m":
            d0["t_end_h"], d0["end_death_h"] = d0["t_end"] + rng.uniform(0, 200, n), rng.random(n) < 0.1
            d0["t_ap_hosp"] = np.where(rng.random(n) < 0.12, rng.uniform(0, 160, n), np.nan)
            d0["t_ap_icu_po"], d0["t_ap_hosp_po"] = d0["t_ap_icu"], d0["t_ap_hosp"]
            d0["postop"] = rng.random(n) < 0.35
            d0["period"] = rng.integers(0, 3, n)
        tl = pd.DataFrame({"stay_id": np.repeat(np.arange(n), 4), "h": rng.uniform(0.5, 160, 4 * n),
                           "pos": np.zeros(4 * n, int), "ok": (rng.random(4 * n) < 0.85).astype(int)})
        d0.update(timeline_arrays(d0, np.arange(n), tl))
        cache = {}
        r = analyses_all(d0, db, cache)
        v = flat(r, db)
        check(f"{db}: every analysis gives a finite estimate on synthetic data", np.isfinite(v).all(), f"{len(v)} values")
        D = decomposition(d0)
        check(f"{db}: decomposition 'full' equals the main odds ratio", abs(D["full"] - r["primary"]["OR"]) < 1e-9)
        rep = 4
        _G.update(d=d0, db=db, cache=cache, part=db)
        mine = _one_ppi(rep)[1][0]
        b = asd.resample(d0, db, np.random.default_rng(asd.SEED + rep))
        ref = math.log(cif(b, sc.overlap_w(asd.ow_fit(b, dict(cache), "m0"), b["A"]), b["t_del"])["OR"])
        check(f"{db}: paired replicate equals asd.resample for the main odds ratio", abs(mine - ref) < 1e-10, f"{mine:.6f} vs {ref:.6f}")
        S[db] = (d0, cache)
    dn = synth_nw(2500, 20, rng)
    st = nw_strata(dn)
    check("NWICU strata: finite odds ratios", all(np.isfinite(st[lv][oc]["OR"]) for lv in ("all", "novent", "vent") for oc in NW_OUT))
    try:
        fx, _, _ = nw_flex(dn, r10.main_folds(2500), r10.gbm)
        check("NWICU flexible models: finite odds ratios", all(np.isfinite(fx[k]["OR"]) for k in NW_FLEX_NAMES))
    except ImportError:
        check("scikit-learn available", False)
    v = nw_flat({"all": st["all"], "novent": st["novent"], "vent": st["vent"]}, NW_STRATA_NAMES)
    check("NWICU strata flattened in the declared order", np.isfinite(v).all() and len(v) == len(NW_STRATA_NAMES))
    d = {"X": np.ones((50, 1)), "names": ["x"]}
    X3, n3, dr = asd.prevalence_filter(np.column_stack([d["X"], np.r_[np.ones(5), np.zeros(45)], np.r_[np.ones(25), np.zeros(25)]]),
                                       ["x", "rare", "common"])
    check("M3: the low-frequency rule drops a rare new column", "rare" in dr and "common" in n3)
    m = mask_tree({"nA": 7, "eventsB": 123, "OR": 0.81, "fam_ge40": 46, "screen": {"A": 1.7}, "flow": {"eligible": 11883, "analysis_B": 7372}})
    check("counts masked, rates kept", m["nA"] == "<10" and m["eventsB"] == "125" and m["fam_ge40"] == "45" and m["OR"] == 0.81
          and m["screen"]["A"] == 1.7 and m["flow"]["eligible"] == "11885" and m["flow"]["analysis_B"] == "7370")
    import contextlib
    import io
    tmp = tempfile.mkdtemp(prefix="r11_self_")
    try:
        P, R = os.path.join(tmp, "p"), os.path.join(tmp, "r")
        os.makedirs(P)
        os.makedirs(R)
        est = {"plan": PLAN, "db": {}}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for db in ("m", "e"):
                d0, cache = S[db]
                n = len(d0["A"])
                d0["dropped3"] = []
                flow = {"eligible": n, "A": int(d0["A"].sum()), "B": int((d0["A"] == 0).sum()), "other": 0, "outside_overlap": 0,
                        "analysis_A": int(d0["A"].sum()), "analysis_B": int((d0["A"] == 0).sum())}
                est["db"][db] = est_ppi(d0, synth_df(n, rng), flow, db, cache)
                _G.update(d=d0, db=db, cache=cache, part=db)
                V = np.array([_one_ppi(rep)[1] for rep in range(60)])
                np.savez(os.path.join(P, f"r11boot_{db}_0000_0060.npz"), reps=np.arange(60), v=V, names=np.array(flat_names(db)))
            small = synth_nw(1200, 12, rng)
            keep_iter = r10.GBM["max_iter"]
            r10.GBM["max_iter"] = 10
            try:
                est["nw"] = est_nw(small)
                for part, names, k in (("nw", NW_STRATA_NAMES, 60), ("nwflex", NW_FLEX_NAMES, 52)):
                    _G.update(d=small, part=part)
                    V = np.array([_one_nw(rep)[1] for rep in range(k)])
                    np.savez(os.path.join(P, f"r11boot_{part}_0000_{k:04d}.npz"), reps=np.arange(k), v=V, names=np.array(names))
            finally:
                r10.GBM["max_iter"] = keep_iter
            json.dump(asd.to_json(est), open(os.path.join(P, "r11_est.json"), "w"))
            ref = {"OR": 0.9, "OR_lo": 0.8, "OR_hi": 1.0}
            json.dump({"per_db": {db: {"analyses": {"primary": ref, "negctl": ref}} for db in ("m", "e")}},
                      open(os.path.join(R, "ppi_estimates.json"), "w"))
            json.dump({"per_db": {k: {o: {"OR": 1.0, "OR_lo": 0.8, "OR_hi": 1.25} for o in NW_OUT} for k in ("NWICU", "INSPIRE")}},
                      open(os.path.join(R, "r9.json"), "w"))
            json.dump({"primary": {"OR": 1.07, "OR_lo": 0.86, "OR_hi": 1.32}}, open(os.path.join(R, "e2_mover.json"), "w"))
            report(argparse.Namespace(private=P, results=R))
        J = json.load(open(os.path.join(R, "r11.json"), encoding="utf-8"))
        mdtxt = open(os.path.join(R, "r11.md"), encoding="utf-8").read()
        check("end to end: report written with every section", all(h in mdtxt for h in ("## 1.", "### 1b", "### 1c", "## 2.", "## 3.", "## 4.", "## 5.")))
        cnt = []

        def walk(o, key=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, k)
            elif isinstance(o, list):
                for v in o:
                    walk(v, key)
            elif key in ("nA", "nB", "eventsA", "eventsB", "N", "eligible", "analysis_A", "fam_ge40", "proxy_A", "novent_B"):
                cnt.append(o)
        walk(J)
        check("end to end: counts in the JSON are masked", bool(cnt) and all(isinstance(v, str) and (v == "<10" or int(v) % 5 == 0) for v in cnt),
              f"{len(cnt)} counts")
        check("end to end: every written reading present", all(k in J["reading"] for k in ("screening", "postop", "late_primary", "trend", "m3",
                                                                                          "nw_negctl", "nw_flex_icu", "nw_novent")))
    finally:
        if os.environ.get("R11_KEEP"):
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(res)}/{len(res)} passed", flush=True)
    sys.exit(0 if all(res) else 1)


def synth_df(n, rng):
    return pd.DataFrame({"steroid_24h": rng.random(n) < 0.1, "benzo_inf": rng.random(n) < 0.05, "dex_inf": rng.random(n) < 0.08,
                         "ketamine_inf": rng.random(n) < 0.01, "service": rng.choice(["MED", "CSURG", "SURG"], n),
                         "unit": rng.choice(["Medical Intensive Care Unit (MICU)", "Cardiac Vascular Intensive Care Unit (CVICU)"], n),
                         "rass": np.where(rng.random(n) < 0.1, np.nan, rng.integers(-5, 3, n)), "sed_inf": rng.random(n) < 0.3,
                         "vent": rng.random(n) < 0.4, "route": rng.choice(["IV", "EN", "UNK"], n), "age": rng.uniform(18, 95, n),
                         "fam_daily": np.where(rng.random(n) < 0.2, np.nan, rng.choice([20.0, 40.0], n)),
                         "scr": rng.uniform(0.5, 3, n), "egfr": rng.uniform(10, 120, n)})


def bindcheck(args):
    import duckdb
    import sql_bind_check as sbc
    t0 = time.time()
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)

    tmp = tempfile.mkdtemp(prefix="r11_bind_")
    try:
        for db, dbn in (("m", "mimic"), ("e", "eicu")):
            con = duckdb.connect()
            con.execute("SET threads = 2")
            mods = ("hosp", "icu", "ed", "note", "derived") if db == "m" else ("eicu", "eicu_derived")
            sbc.empty_tables(con, r7.SRC[db], mods)
            con.execute("CREATE SCHEMA src")
            for m, tb in ([("hosp", "emar"), ("icu", "inputevents")] if db == "m" else [("eicu", "medication")]):
                con.execute(f'CREATE TABLE src."{tb}" AS SELECT * FROM {m}."{tb}" LIMIT 0')
            for sql, path in asd.all_sql(tmp, dbn)["cohort"]:
                con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")
            for part, sql in statements(tmp, db).items():
                path = r7.pf(tmp, db, FILE[part])
                try:
                    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")
                    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
                    good = cols == EXPECT[part].split()
                    check(f"{db}: {part}", good, "binds, columns as the analysis reads them" if good else f"got {cols}")
                except Exception as e:
                    check(f"{db}: {part}", False, f"{type(e).__name__}: {str(e).splitlines()[0][:180]}")
            con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(results)}/{len(results)} passed ({time.time() - t0:.0f} s)")
    sys.exit(0 if all(results) else 1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    sp.add_parser("bindcheck")
    s = sp.add_parser("timing")
    s.add_argument("--results", required=True)
    subs = {c: sp.add_parser(c) for c in ("extract", "estimate", "boot", "report")}
    for s in subs.values():
        s.add_argument("--private", required=True)
        s.add_argument("--results", required=True)
    subs["extract"].add_argument("--db", choices=["m", "e"], required=True)
    subs["extract"].add_argument("--part", choices=list(FILE), required=True)
    subs["boot"].add_argument("--part", choices=["m", "e", "nw", "nwflex"], required=True)
    subs["boot"].add_argument("--start", type=int, required=True)
    subs["boot"].add_argument("--stop", type=int, required=True)
    subs["boot"].add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"selftest": selftest, "bindcheck": bindcheck, "timing": timing, "extract": extract, "estimate": estimate, "boot": boot,
     "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
