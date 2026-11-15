#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
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
import r11
import r14
import r15

PLAN = "SAP_addendum_R16.md v1.0"
NBOOT = 2000
H = 168
MIN_N = 20
STRATA = ("light", "moderate", "deep", "missing")
UNITS = ("all", "rass3") + STRATA
LMS = (6, 12)
TPTS = (24, 48, 72, 168)
DBS = ("m", "e")
DBNAME = {"m": "MIMIC-IV", "e": "eICU"}
HOURS = (1, 6, 12, 24, 48, 72, 168)
HBINS = (0, 1, 2, 3, 6, 12, 24, 48, 72, 168, 1e9)
HBIN_LAB = ("<1", "1-<2", "2-<3", "3-<6", "6-<12", "12-<24", "24-<48", "48-<72", "72-<168", ">=168")
PDX = ("infection", "neoplasm", "endocrine", "nervous", "circulatory", "respiratory", "digestive", "genitourinary", "injury", "other")
ADM = ("emergency", "elective", "observation")
SEV_E = [("hr_ge110", "hr_max", ">=", 110), ("map_lt65", "map_min", "<", 65), ("rr_ge25", "rr_max", ">=", 25),
         ("temp_ge383", "temp_max", ">=", 38.3), ("temp_lt36", "temp_min", "<", 36.0), ("spo2_lt90", "spo2_min", "<", 90),
         ("gcs_le12", "gcs_min", "<=", 12), ("gcs_le8", "gcs_min", "<=", 8), ("wbc_ge12", "wbc_max", ">=", 12),
         ("wbc_lt4", "wbc_min", "<", 4), ("bun_ge20", "bun_max", ">=", 20), ("bun_ge40", "bun_max", ">=", 40),
         ("na_lt135", "na_min", "<", 135), ("na_gt145", "na_max", ">", 145), ("hco3_lt22", "hco3_min", "<", 22),
         ("hco3_lt18", "hco3_min", "<", 18), ("hct_lt30", "hct_min", "<", 30), ("bili_ge2", "bili_max", ">=", 2),
         ("alb_lt3", "alb_min", "<", 3), ("ph_lt730", "ph_min", "<", 7.30)]
SEV_E_NM = [("hr_nm", "hr_max"), ("map_nm", "map_min"), ("temp_nm", "temp_max"), ("gcs_nm", "gcs_min"), ("wbc_nm", "wbc_max"),
            ("bun_nm", "bun_max"), ("na_nm", "na_min"), ("hco3_nm", "hco3_min"), ("hct_nm", "hct_min"), ("bili_nm", "bili_max"),
            ("alb_nm", "alb_min"), ("ph_nm", "ph_min")]
SEV_E_RAW = ["hr_max", "map_min", "rr_max", "temp_max", "temp_min", "spo2_min", "gcs_min", "wbc_max", "wbc_min", "bun_max", "na_min",
             "na_max", "hco3_min", "hct_min", "bili_max", "alb_min", "ph_min", "lac_max"]

TL_M = """
WITH c AS (SELECT stay_id, t0, outtime FROM read_parquet('{f}') WHERE arm IS NOT NULL),
fu AS (SELECT stay_id, t0, least(outtime, t0 + INTERVAL 168 HOUR) AS t_end FROM c),
da AS (SELECT stay_id, charttime, value FROM icu.chartevents WHERE itemid = 228332 AND value IS NOT NULL)
SELECT f.stay_id, date_diff('minute', f.t0, d.charttime) / 60.0 AS h, CAST(d.charttime > f.t0 AS INTEGER) AS post,
       max(CASE WHEN d.value = 'Positive' THEN 1 ELSE 0 END) AS pos,
       max(CASE WHEN d.value IN ('Positive', 'Negative') THEN 1 ELSE 0 END) AS ok,
       max(CASE WHEN d.value = 'Negative' THEN 1 ELSE 0 END) AS neg
FROM fu f JOIN da d ON d.stay_id = f.stay_id AND d.charttime >= f.t0 - INTERVAL 24 HOUR AND d.charttime <= f.t_end
GROUP BY f.stay_id, f.t0, d.charttime
"""
TL_E = """
WITH c AS (SELECT stay_id, t0 FROM read_parquet('{f}') WHERE arm IS NOT NULL),
fu AS (SELECT c.stay_id, c.t0, least(p.unitdischargeoffset, c.t0 + 10080) AS t_end FROM c JOIN eicu.patient p ON p.patientunitstayid = c.stay_id),
da AS (SELECT patientunitstayid AS stay_id, nursingchartoffset AS t, nursingchartvalue AS v FROM eicu.nursecharting
       WHERE nursingchartcelltypevallabel = 'Delirium Scale/Score' AND nursingchartcelltypevalname = 'Delirium Score'
         AND NULLIF(nursingchartvalue, '') IS NOT NULL)
SELECT f.stay_id, (d.t - f.t0) / 60.0 AS h, CAST(d.t > f.t0 AS INTEGER) AS post,
       max(CASE WHEN upper(d.v) = 'YES' OR TRY_CAST(d.v AS INTEGER) >= 4 THEN 1 ELSE 0 END) AS pos,
       max(CASE WHEN upper(d.v) IN ('YES', 'NO') OR TRY_CAST(d.v AS INTEGER) IS NOT NULL THEN 1 ELSE 0 END) AS ok,
       max(CASE WHEN upper(d.v) = 'NO' OR TRY_CAST(d.v AS INTEGER) < 4 THEN 1 ELSE 0 END) AS neg
FROM fu f JOIN da d ON d.stay_id = f.stay_id AND d.t >= f.t0 - 1440 AND d.t <= f.t_end GROUP BY f.stay_id, f.t0, d.t
"""
_CKD = ("max(CASE WHEN (icd_version = 10 AND regexp_matches(icd_code, '^(N18|Z992|Z49)')) "
        "OR (icd_version = 9 AND regexp_matches(icd_code, '^(585|V4511|V56)')) THEN 1 ELSE 0 END)")
_DIAL = ("max(CASE WHEN (icd_version = 10 AND regexp_matches(icd_code, '^(N186|Z992|Z49)')) "
         "OR (icd_version = 9 AND regexp_matches(icd_code, '^(5856|V4511|V56)')) THEN 1 ELSE 0 END)")
_K9 = "TRY_CAST(substr(icd_code, 1, 3) AS INTEGER)"
_PDX = f"""CASE WHEN icd_version = 10 THEN
         CASE WHEN regexp_matches(icd_code, '^([AB]|U07)') THEN 'infection' WHEN regexp_matches(icd_code, '^(C|D[0-4])') THEN 'neoplasm'
              WHEN regexp_matches(icd_code, '^E') THEN 'endocrine' WHEN regexp_matches(icd_code, '^G') THEN 'nervous'
              WHEN regexp_matches(icd_code, '^I') THEN 'circulatory' WHEN regexp_matches(icd_code, '^J') THEN 'respiratory'
              WHEN regexp_matches(icd_code, '^K') THEN 'digestive' WHEN regexp_matches(icd_code, '^N') THEN 'genitourinary'
              WHEN regexp_matches(icd_code, '^[ST]') THEN 'injury' ELSE 'other' END
       ELSE CASE WHEN {_K9} IS NULL THEN 'other' WHEN {_K9} <= 139 THEN 'infection' WHEN {_K9} <= 239 THEN 'neoplasm'
              WHEN {_K9} BETWEEN 240 AND 279 THEN 'endocrine' WHEN {_K9} BETWEEN 320 AND 389 THEN 'nervous'
              WHEN {_K9} BETWEEN 390 AND 459 THEN 'circulatory' WHEN {_K9} BETWEEN 460 AND 519 THEN 'respiratory'
              WHEN {_K9} BETWEEN 520 AND 579 THEN 'digestive' WHEN {_K9} BETWEEN 580 AND 629 THEN 'genitourinary'
              WHEN {_K9} BETWEEN 800 AND 999 THEN 'injury' ELSE 'other' END END"""
COV_M = f"""
WITH base AS (SELECT DISTINCT subject_id, hadm_id, stay_id, t0 FROM read_parquet({{files}})),
sp AS (SELECT stay_id, sofa_prev FROM (
         SELECT b.stay_id, s.sofa_24hours AS sofa_prev, row_number() OVER (PARTITION BY b.stay_id ORDER BY s.endtime DESC) AS rn
         FROM base b JOIN derived.sofa s ON s.stay_id = b.stay_id AND s.endtime <= b.t0) WHERE rn = 1),
prior AS (
  SELECT b.stay_id, d.icd_code, d.icd_version FROM base b JOIN hosp.admissions a0 ON a0.hadm_id = b.hadm_id
  JOIN hosp.admissions a ON a.subject_id = b.subject_id AND a.admittime < a0.admittime
  JOIN hosp.diagnoses_icd d ON d.hadm_id = a.hadm_id
  UNION ALL
  SELECT b.stay_id, d.icd_code, d.icd_version FROM base b JOIN ed.edstays e ON e.hadm_id = b.hadm_id JOIN ed.diagnosis d ON d.stay_id = e.stay_id),
cur AS (SELECT b.stay_id, d.icd_code, d.icd_version, d.seq_num FROM base b JOIN hosp.diagnoses_icd d ON d.hadm_id = b.hadm_id),
kp AS (SELECT stay_id, {_CKD} AS ckd_prior, {_DIAL} AS dial_prior FROM prior GROUP BY 1),
kc AS (SELECT stay_id, {_CKD} AS ckd_cur, {_DIAL} AS dial_cur FROM cur GROUP BY 1),
rr AS (SELECT DISTINCT b.stay_id FROM base b JOIN derived.rrt r ON r.stay_id = b.stay_id AND r.charttime <= b.t0 WHERE r.dialysis_present = 1),
px AS (SELECT stay_id, min({_PDX}) AS pdx FROM cur WHERE seq_num = 1 GROUP BY 1),
ad AS (SELECT b.stay_id, a.admission_type FROM base b JOIN hosp.admissions a ON a.hadm_id = b.hadm_id)
SELECT b.stay_id, sp.sofa_prev, coalesce(kp.ckd_prior, 0) AS ckd_prior, coalesce(kp.dial_prior, 0) AS dial_prior,
       coalesce(kc.ckd_cur, 0) AS ckd_cur, coalesce(kc.dial_cur, 0) AS dial_cur, CAST(rr.stay_id IS NOT NULL AS INTEGER) AS rrt_pre,
       coalesce(px.pdx, 'other') AS pdx,
       CASE WHEN ad.admission_type IN ('EW EMER.', 'DIRECT EMER.', 'URGENT') THEN 'emergency'
            WHEN ad.admission_type IN ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION') THEN 'elective' ELSE 'observation' END AS adm
FROM base b LEFT JOIN sp USING (stay_id) LEFT JOIN kp USING (stay_id) LEFT JOIN kc USING (stay_id) LEFT JOIN rr USING (stay_id)
LEFT JOIN px USING (stay_id) LEFT JOIN ad USING (stay_id)
"""
COV_E = """
WITH base AS (SELECT DISTINCT stay_id, t0 FROM read_parquet({files})),
vit AS (SELECT b.stay_id, max(v.heartrate) AS hr_max, min(coalesce(v.ibp_mean, v.nibp_mean)) AS map_min, max(v.respiratoryrate) AS rr_max,
               max(v.temperature) AS temp_max, min(v.temperature) AS temp_min, min(v.spo2) AS spo2_min
        FROM base b JOIN eicu_derived.pivoted_vital v ON v.patientunitstayid = b.stay_id AND v.chartoffset BETWEEN b.t0 - 1440 AND b.t0 GROUP BY 1),
gcs AS (SELECT b.stay_id, min(g.gcs) AS gcs_min FROM base b JOIN eicu_derived.pivoted_gcs g
        ON g.patientunitstayid = b.stay_id AND g.chartoffset BETWEEN b.t0 - 1440 AND b.t0 WHERE g.gcs IS NOT NULL GROUP BY 1),
lab AS (SELECT b.stay_id, max(l.wbc) AS wbc_max, min(l.wbc) AS wbc_min, max(l.bun) AS bun_max, min(l.sodium) AS na_min, max(l.sodium) AS na_max,
               min(l.bicarbonate) AS hco3_min, min(l.hematocrit) AS hct_min, max(l.bilirubin) AS bili_max, min(l.albumin) AS alb_min
        FROM base b JOIN eicu_derived.pivoted_lab l ON l.patientunitstayid = b.stay_id AND l.chartoffset BETWEEN b.t0 - 1440 AND b.t0 GROUP BY 1),
bg AS (SELECT b.stay_id, min(g.ph) AS ph_min FROM base b JOIN eicu_derived.pivoted_bg g
       ON g.patientunitstayid = b.stay_id AND g.chartoffset BETWEEN b.t0 - 1440 AND b.t0 WHERE g.ph IS NOT NULL GROUP BY 1),
ph AS (SELECT patientunitstayid AS stay_id,
              max(CASE WHEN pasthistorypath LIKE '%/Renal  (R)/Renal Failure/%' OR pasthistorypath LIKE '%/Renal  (R)/Renal Insufficiency/%'
                         OR pasthistorypath LIKE '%/Renal  (R)/s/p Renal Transplant/%' THEN 1 ELSE 0 END) AS ckd,
              max(CASE WHEN pasthistorypath LIKE '%renal failure - hemodialysis%' OR pasthistorypath LIKE '%renal failure - peritoneal dialysis%'
                       THEN 1 ELSE 0 END) AS dial_ph
       FROM eicu.pasthistory GROUP BY 1),
ap AS (SELECT patientunitstayid AS stay_id, max(CASE WHEN dialysis = 1 THEN 1 ELSE 0 END) AS dial_ap FROM eicu.apacheapsvar GROUP BY 1),
tr AS (SELECT DISTINCT b.stay_id FROM base b JOIN eicu.treatment t ON t.patientunitstayid = b.stay_id AND t.treatmentoffset <= b.t0
       WHERE t.treatmentstring LIKE 'renal|dialysis|hemodialysis%' OR t.treatmentstring LIKE 'renal|dialysis|C V V H%'
          OR t.treatmentstring LIKE 'renal|dialysis|C A V H D%' OR t.treatmentstring LIKE 'renal|dialysis|SLED%'
          OR t.treatmentstring LIKE 'renal|dialysis|peritoneal dialysis%' OR t.treatmentstring LIKE 'renal|dialysis|ultrafiltration%'),
io AS (SELECT DISTINCT b.stay_id FROM base b JOIN eicu.intakeoutput i ON i.patientunitstayid = b.stay_id AND i.intakeoutputoffset <= b.t0
       WHERE coalesce(i.dialysistotal, 0) <> 0),
dx AS (SELECT patientunitstayid AS stay_id, max(apachedxgroup) AS dxgroup FROM eicu_derived.apache_groups GROUP BY 1)
SELECT b.stay_id, vit.hr_max, vit.map_min, vit.rr_max, vit.temp_max, vit.temp_min, vit.spo2_min, gcs.gcs_min, lab.wbc_max, lab.wbc_min,
       lab.bun_max, lab.na_min, lab.na_max, lab.hco3_min, lab.hct_min, lab.bili_max, lab.alb_min, bg.ph_min,
       coalesce(ph.ckd, 0) AS ckd, coalesce(ph.dial_ph, 0) AS dial_ph, coalesce(ap.dial_ap, 0) AS dial_ap,
       CAST(tr.stay_id IS NOT NULL OR io.stay_id IS NOT NULL AS INTEGER) AS rrt_pre, coalesce(dx.dxgroup, 'Other') AS dxgroup
FROM base b LEFT JOIN vit USING (stay_id) LEFT JOIN gcs USING (stay_id) LEFT JOIN lab USING (stay_id) LEFT JOIN bg USING (stay_id)
LEFT JOIN ph USING (stay_id) LEFT JOIN ap USING (stay_id) LEFT JOIN tr USING (stay_id) LEFT JOIN io USING (stay_id) LEFT JOIN dx USING (stay_id)
"""


def _tx_cols(kind):
    out = []
    for T in TPTS:
        for v in ("init", "other"):
            cond = f"a.h >= {T - 24} AND a.h <= {T}" if kind == "m" else f"a.hs <= {T} AND a.he > {T}"
            out.append(f"coalesce(max(CASE WHEN a.{v} AND {cond} THEN 1 ELSE 0 END), 0) AS {v}_{T}")
    return ", ".join(out)


TX2_M = ("WITH c AS (SELECT hadm_id, stay_id, t0, arm FROM read_parquet('{f}') WHERE arm IS NOT NULL),\n" + asd.ASADM_M + """,
a AS (SELECT c.stay_id, date_diff('minute', c.t0, x.t) / 60.0 AS h,
             ((c.arm = 'A' AND x.drug = 'PPI') OR (c.arm = 'B' AND x.drug = 'FAM')) AS init,
             ((c.arm = 'A' AND x.drug IN ('FAM', 'H2O')) OR (c.arm = 'B' AND x.drug = 'PPI')) AS other
      FROM c JOIN asadm x ON x.hadm_id = c.hadm_id AND x.t >= c.t0 AND x.t <= c.t0 + INTERVAL 168 HOUR)
SELECT c.stay_id, """ + _tx_cols("m") + """, min(CASE WHEN a.other AND a.h > 0 THEN a.h END) AS t_other, 0 AS stop_unusable
FROM c LEFT JOIN a USING (stay_id) GROUP BY c.stay_id
""")
TX2_E = ("WITH c AS (SELECT stay_id, t0, arm FROM read_parquet('{f}') WHERE arm IS NOT NULL),\n" + asd.MED_E + """,
a AS (SELECT c.stay_id, (md.drugstartoffset - c.t0) / 60.0 AS hs,
             CASE WHEN md.drugstopoffset > md.drugstartoffset THEN (md.drugstopoffset - c.t0) / 60.0 ELSE 1e9 END AS he,
             ((c.arm = 'A' AND md.drug = 'PPI') OR (c.arm = 'B' AND md.drug = 'FAM')) AS init,
             ((c.arm = 'A' AND md.drug IN ('FAM', 'H2O')) OR (c.arm = 'B' AND md.drug = 'PPI')) AS other
      FROM c JOIN med md ON md.patientunitstayid = c.stay_id AND md.drugstartoffset >= c.t0 AND md.drugstartoffset <= c.t0 + 10080)
SELECT c.stay_id, """ + _tx_cols("e") + """, min(CASE WHEN a.other AND a.hs > 0 THEN a.hs END) AS t_other,
       coalesce(max(CASE WHEN a.init AND a.hs = 0 AND a.he >= 1e9 THEN 1 ELSE 0 END), 0) AS stop_unusable
FROM c LEFT JOIN a USING (stay_id) GROUP BY c.stay_id
""")
_TX1_SEL = ", ".join(f"coalesce(max(CASE WHEN rx.tt = {T} THEN 1 ELSE 0 END), 0) AS act_{T}, max(CASE WHEN rx.tt = {T} THEN rx.daily END) AS daily_{T}"
                     for T in (24, 48, 72))
TX1_M = """
WITH c AS (SELECT hadm_id, stay_id, t0 FROM read_parquet('{f}') WHERE arm IS NOT NULL),
tp AS (SELECT * FROM (VALUES (24), (48), (72)) v(tt)),
r0 AS (SELECT c.stay_id, tp.tt, r.starttime, TRY_CAST(regexp_extract(r.dose_val_rx, '^[0-9.]+') AS DOUBLE) * r.doses_per_24_hrs AS daily
       FROM c CROSS JOIN tp JOIN hosp.prescriptions r ON r.hadm_id = c.hadm_id
        AND r.starttime <= c.t0 + to_hours(tp.tt) AND coalesce(r.stoptime, r.starttime + INTERVAL 30 DAY) > c.t0 + to_hours(tp.tt)
       WHERE regexp_matches(lower(r.drug), 'famotidine|pepcid')),
rx AS (SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY stay_id, tt ORDER BY starttime DESC, daily DESC NULLS LAST) AS rn FROM r0) WHERE rn = 1)
SELECT c.stay_id, """ + _TX1_SEL + """
FROM c LEFT JOIN rx USING (stay_id) GROUP BY c.stay_id
"""
TX1_E = ("WITH c AS (SELECT stay_id, t0 FROM read_parquet('{f}') WHERE arm IS NOT NULL),\n" + asd.MED_E + """,
tp AS (SELECT * FROM (VALUES (24), (48), (72)) v(tt)),
r0 AS (SELECT c.stay_id, tp.tt, md.drugstartoffset AS starttime, md.daily
       FROM c CROSS JOIN tp JOIN med md ON md.patientunitstayid = c.stay_id AND md.drug = 'FAM' AND md.drugstartoffset >= c.t0
        AND md.drugstartoffset <= c.t0 + tp.tt * 60 AND (md.drugstopoffset > c.t0 + tp.tt * 60 OR md.drugstopoffset <= md.drugstartoffset)),
rx AS (SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY stay_id, tt ORDER BY starttime DESC, daily DESC NULLS LAST) AS rn FROM r0) WHERE rn = 1)
SELECT c.stay_id, """ + _TX1_SEL + """
FROM c LEFT JOIN rx USING (stay_id) GROUP BY c.stay_id
""")
BLEED_M = """,
c1 AS (SELECT hadm_id, stay_id, t0 FROM read_parquet('{f}') WHERE arm IS NOT NULL)
SELECT c1.stay_id, CAST(c1.hadm_id IN (SELECT hadm_id FROM gib) AS INTEGER) AS bleed_dx,
       CAST(EXISTS (SELECT 1 FROM ppi_inf p WHERE p.hadm_id = c1.hadm_id
                    AND p.starttime BETWEEN c1.t0 - INTERVAL 24 HOUR AND c1.t0 + INTERVAL 1 HOUR) AS INTEGER) AS ppi_infusion
FROM c1
"""
BLEED_E = """,
c1 AS (SELECT stay_id, t0 FROM read_parquet('{f}') WHERE arm IS NOT NULL)
SELECT c1.stay_id,
       CAST(regexp_matches(lower(coalesce(p.apacheadmissiondx, '')), 'bleeding, upper gi|bleeding, gi|gi bleeding|bleeding, lower gi|ulcer disease|varices|peptic')
            OR EXISTS (SELECT 1 FROM gib g WHERE g.patientunitstayid = c1.stay_id AND g.t <= c1.t0) AS INTEGER) AS bleed_dx,
       CAST(EXISTS (SELECT 1 FROM ppi_inf i WHERE i.patientunitstayid = c1.stay_id AND i.t BETWEEN c1.t0 - 1440 AND c1.t0 + 60) AS INTEGER) AS ppi_infusion
FROM c1 JOIN eicu.patient p ON p.patientunitstayid = c1.stay_id
"""


def pf(P, db, name):
    return os.path.join(P, f"{db}_{name}.parquet")


def statements(P, db):
    c1, c2 = pf(P, db, "c1_cohort"), pf(P, db, "c2_cohort")
    files = asd.plist([c1, c2])
    if db == "m":
        return [("r16tl", TL_M.replace("{f}", c2)), ("r16cov", COV_M.replace("{files}", files)), ("r16tx2", TX2_M.replace("{f}", c2)),
                ("r16tx1", TX1_M.replace("{f}", c1)), ("r16bleed", "WITH " + asd.gate_body(2, "m") + BLEED_M.replace("{f}", c1))]
    return [("r16tl", TL_E.replace("{f}", c2)), ("r16cov", COV_E.replace("{files}", files)), ("r16pre", r14.PRE_E.format(f=c2)),
            ("r16tx2", TX2_E.replace("{f}", c2)), ("r16tx1", TX1_E.replace("{f}", c1)),
            ("r16bleed", "WITH " + asd.gate_body(2, "e") + BLEED_E.replace("{f}", c1))]


def connect_full(root, P):
    sys.path.insert(0, asd.TOOLS)
    from mimic_local import connect
    return connect(root, memory_limit="8GB", threads=8, temp_dir=os.path.join(P, "duck_tmp"))


def extract(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    for db, root in (("m", args.mimic), ("e", args.eicu)):
        con = connect_full(root, P)
        for name, sql in statements(P, db):
            path = pf(P, db, name)
            asd.write_pq(con, sql, path)
            n, k = con.execute(f"SELECT count(*), count(DISTINCT stay_id) FROM read_parquet('{path}')").fetchone()
            asd.status(f"{os.path.basename(path)} written: {sc.mask_n(n)} rows, {sc.mask_n(k)} stays (masked)", t0)
        con.close()
    asd.log_run(args.results, f"r16 extract: ok ({time.time() - t0:.0f} s)")


def merge1(df, P, db, name):
    n0 = len(df)
    out = df.merge(asd.read_pq(pf(P, db, name)), on="stay_id", how="left", validate="one_to_one")
    if len(out) != n0:
        raise SystemExit(f"{db} {name}: rows changed in the merge")
    return out


def b1(df, col):
    return np.nan_to_num(asd.num(df, col)) == 1


def timeline_extra(d, sid, tl):
    n = len(sid)
    row = pd.Series(np.arange(n), index=sid).reindex(tl["stay_id"].values).values
    keep = np.isfinite(row)
    i = row[keep].astype(int)
    h = tl["h"].values.astype(float)[keep]
    post = tl["post"].values.astype(float)[keep] == 1
    ok = tl["ok"].values.astype(float)[keep] == 1
    neg = tl["neg"].values.astype(float)[keep] == 1
    pos = tl["pos"].values.astype(float)[keep] == 1
    T = np.minimum(d["t_end"], float(H))
    det = np.isfinite(d["t_del"]) & (d["t_del"] <= T)
    stop = np.where(det, d["t_del"], T)
    fu = post & (h > 0) & (h <= stop[i] + 1e-9) & ok
    tf = np.full(n, np.inf)
    np.minimum.at(tf, i[fu], h[fu])
    b = ~post
    last = np.full(n, -np.inf)
    np.maximum.at(last, i[b], h[b])
    at = b & (h >= last[i] - 1e-9)
    negl, posl = np.zeros(n, bool), np.zeros(n, bool)
    negl[i[at & neg]] = True
    posl[i[at & pos]] = True
    base = np.where(~np.isfinite(last), "none", np.where(negl & ~posl, "neg", "uta"))
    return {"t_first_ok": np.where(np.isfinite(tf), tf, np.nan), "base": base, "base_pos": posl}


def add_cols(d, cols, names, base="X", base_names="names"):
    X = np.column_stack([d[base]] + [np.asarray(c, float) for c in cols])
    return asd.prevalence_filter(X, list(d[base_names]) + list(names))


def spec_sofa_prev(d, sp):
    names = d["names"]
    keep = [j for j, nm in enumerate(names) if not nm.startswith("sofa_")]
    new = np.column_stack([sp >= 3, sp >= 6, sp >= 9, ~np.isfinite(sp)]).astype(float)
    return asd.prevalence_filter(np.column_stack([d["X"][:, keep], new]),
                                 [names[j] for j in keep] + ["sofap_ge3", "sofap_ge6", "sofap_ge9", "sofap_missing"])


def sev_cols_e(raw):
    cols, names = [], []
    for nm, v, op, cut in SEV_E:
        x = raw[v]
        cols.append({">=": x >= cut, "<": x < cut, "<=": x <= cut, ">": x > cut}[op])
        names.append(nm)
    for nm, v in SEV_E_NM:
        cols.append(~np.isfinite(raw[v]))
        names.append(nm)
    return cols, names


def spec_sev_e(d, raw):
    Xp, namesp, _ = r14.pre_matrix(d["X"], d["names"], raw["vent_pre"], raw["lac_max"])
    cols, names = sev_cols_e(raw)
    return asd.prevalence_filter(np.column_stack([Xp] + [np.asarray(c, float) for c in cols]), list(namesp) + names)


def kidney_fields(d, df, db):
    if db == "m":
        d["ckd_prior"], d["ckd_wide"] = b1(df, "ckd_prior"), b1(df, "ckd_prior") | b1(df, "ckd_cur")
        d["dial_prior"], d["dial_wide"] = b1(df, "dial_prior"), b1(df, "dial_prior") | b1(df, "dial_cur")
        d["rrt_pre"] = b1(df, "rrt_pre")
        d["ckd"], d["rrt0"] = d["ckd_wide"], d["rrt_pre"] | d["dial_wide"]
    else:
        d["ckd"] = b1(df, "ckd")
        d["dial_wide"] = b1(df, "dial_ph") | b1(df, "dial_ap")
        d["rrt_pre"] = b1(df, "rrt_pre")
        d["rrt0"] = d["rrt_pre"] | d["dial_wide"]
    d["egfr"] = asd.egfr_of(df)
    d["X_kadj"], d["names_kadj"], _ = add_cols(d, [d["ckd"], d["rrt0"]], ["ckd", "rrt0"])


def load2(P, db):
    df, flow = asd.load_key(P, "c2" + db, stage2=True)
    for name in ("r16cov", "r16tx2") + (("r16pre",) if db == "e" else ()):
        df = merge1(df, P, db, name)
    d = asd.build(df, db, 2, stage2=True)
    d["sid"] = df["stay_id"].values
    d["rgrp"] = r15.rass_group(asd.num(df, "rass"))
    d["h_icu"] = asd.num(df, "h_icu")
    tl = asd.read_pq(pf(P, db, "r16tl"))
    d.update(r11.timeline_arrays(d, d["sid"], tl[tl["post"] == 1]))
    d.update(timeline_extra(d, d["sid"], tl))
    kidney_fields(d, df, db)
    for T in TPTS:
        d[f"init_{T}"], d[f"other_{T}"] = b1(df, f"init_{T}"), b1(df, f"other_{T}")
    d["t_other"] = asd.num(df, "t_other")
    d["stop_unusable"] = b1(df, "stop_unusable")
    if db == "m":
        sp = asd.num(df, "sofa_prev")
        d["sofa_prev"], d["sofa_main"] = sp, asd.num(df, "sofa")
        d["X_sev"], d["names_sev"], _ = spec_sofa_prev(d, sp)
        d["pdx"], d["adm"] = df["pdx"].astype(str).values, df["adm"].astype(str).values
    else:
        raw = {c: asd.num(df, c) for c in SEV_E_RAW}
        raw["vent_pre"] = df["vent_pre"].fillna(False).astype(bool).values
        for c in SEV_E_RAW:
            d["raw_" + c] = raw[c]
        d["vent_pre"] = raw["vent_pre"]
        d["X_sev"], d["names_sev"], _ = spec_sev_e(d, raw)
        d["dxgroup"] = df["dxgroup"].astype(str).values
        groups = sorted(g for g in set(d["dxgroup"]) if g != "Other")
        d["X_dx"], d["names_dx"], _ = add_cols(d, [d["dxgroup"] == g for g in groups], [f"dx_{g}" for g in groups])
    return d, flow


def load1(P, db):
    df, flow = asd.load_key(P, "c1" + db, stage2=True)
    for name in ("r16cov", "r16tx1", "r16bleed"):
        df = merge1(df, P, db, name)
    d = asd.build(df, db, 1, stage2=True)
    d["sid"] = df["stay_id"].values
    kidney_fields(d, df, db)
    d["bleed_dx"], d["ppi_infusion"] = b1(df, "bleed_dx"), b1(df, "ppi_infusion")
    d["bleed"] = d["bleed_dx"] | d["ppi_infusion"]
    for T in (24, 48, 72):
        d[f"act_{T}"], d[f"daily_{T}"] = b1(df, f"act_{T}"), asd.num(df, f"daily_{T}")
    return d, flow


def load_key(P, key):
    return load2(P, key[2]) if key[1] == "2" else load1(P, key[2])


def nan_eff():
    e = sc.effects(float("nan"), float("nan"))
    e.update(eventsA=0, eventsB=0, nA=0, nB=0)
    return e


def cif(d, w, t):
    return asd.cif_contrast(d, w, t)[168]


def unit_mask(d, u):
    if u == "all":
        return np.ones(len(d["A"]), bool)
    return d["sub"]["rass3"] if u == "rass3" else d["rgrp"] == u


def unit_fit(d, m, cache, key, spec=None):
    A = d["A"]
    if min(int((m & (A == 1)).sum()), int((m & (A == 0)).sum())) < MIN_N:
        return None, None
    dd = d if spec is None else dict(d, X=d["X_" + spec], names=d["names_" + spec])
    s = asd.subset_d(dd, m)
    return s, sc.overlap_w(asd.ow_fit(s, cache, key), s["A"])


def cif_lm(s, w, L):
    Hh = H - L
    kend, typ, _ = sc.follow_types(s["t_end"] - L, s["end_death"], s["t_del"] - L, Hh=Hh, disc_competing=True)
    A, w = s["A"], np.asarray(w, float)
    r = {a: float(sc.aj_cif(w[A == a], kend[A == a], typ[A == a], Hh=Hh)[Hh - 1]) for a in (1, 0)}
    e = sc.effects(r[1], r[0])
    e["eventsA"], e["eventsB"] = int(((typ == 1) & (A == 1)).sum()), int(((typ == 1) & (A == 0)).sum())
    e["nA"], e["nB"] = int((A == 1).sum()), int((A == 0).sum())
    return e


def lm_mask(d, L):
    return (d["t_end"] > L) & ~(np.isfinite(d["t_del"]) & (d["t_del"] <= L))


def scr_ratio(s, w):
    r = r11.screening(s, w)["scr_ok"]
    return {"A": r["A"], "B": r["B"], "ratio": r["ratio"]}


def est_c2(d, db, cache):
    res = {}
    A = d["A"]
    ow = sc.overlap_w(asd.ow_fit(d, cache, "full"), A)
    pr = asd.cif_contrast(d, ow, d["t_del"], reads=(72, 168))
    res["primary"], res["del72"] = pr[168], pr[72]
    res["negctl"] = cif(d, ow, d["t_hypo"])
    res["ic_all"] = r11.ic_fit(d, ow, 3)
    res["scr_all"] = scr_ratio(d, ow)

    def both(name, s, w, negctl=True, ic=None, scr=None):
        res[f"{name}_primary"] = cif(s, w, s["t_del"]) if s is not None else nan_eff()
        if negctl:
            res[f"{name}_negctl"] = cif(s, w, s["t_hypo"]) if s is not None else nan_eff()
        if ic:
            res[ic] = r11.ic_fit(s, w, 3) if s is not None else {"HR": float("nan")}
        if scr:
            res[scr] = scr_ratio(s, w) if s is not None else {"ratio": float("nan")}

    for u in UNITS[1:]:
        s, w = unit_fit(d, unit_mask(d, u), cache, "rass3" if u == "rass3" else f"r15_{u}")
        both(u, s, w, ic=f"ic_{u}", scr=f"scr_{u}")
    for k in ("egfr60", "age65"):
        s, w = unit_fit(d, d["sub"][k], cache, k)
        res[k] = cif(s, w, s["t_del"]) if s is not None else nan_eff()
    light = d["rgrp"] == "light"
    s, w = unit_fit(d, d["base"] == "neg", cache, "bneg")
    both("bneg", s, w)
    for k in ("uta", "none"):
        s, w = unit_fit(d, d["base"] == k, cache, "b" + k)
        both("b" + k, s, w, negctl=False)
    s, w = unit_fit(d, (d["base"] == "neg") & light, cache, "bneg_light")
    both("bneg_light", s, w, negctl=False)
    early = d["h_icu"] < 24
    s, w = unit_fit(d, early, cache, "h24")
    both("h24", s, w)
    s, w = unit_fit(d, early & light, cache, "h24_light")
    both("h24_light", s, w, negctl=False)
    for L in LMS:
        for u, m in (("all", lm_mask(d, L)), ("light", lm_mask(d, L) & light)):
            s, w = unit_fit(d, m, cache, f"lm{L}_{u}")
            res[f"lm{L}_{u}"] = cif_lm(s, w, L) if s is not None else nan_eff()
    alln = np.ones(len(A), bool)
    s, w = unit_fit(d, alln, cache, "sev", "sev")
    both("sev", s, w)
    s, w = unit_fit(d, alln, cache, "kadj", "kadj")
    both("kadj", s, w)
    s, w = unit_fit(d, ~d["rrt0"], cache, "kexc")
    both("kexc", s, w)
    if db == "e":
        s, w = unit_fit(d, alln, cache, "dx", "dx")
        both("dx", s, w)
    return res


def est_c1(d, db, cache):
    res = {}
    A = d["A"]
    ow = sc.overlap_w(asd.ow_fit(d, cache, "full"), A)
    res["primary"], res["negctl"] = cif(d, ow, d["t_del"]), cif(d, ow, d["t_hypo"])
    alln = np.ones(len(A), bool)
    for name, m, spec in (("kadj", alln, "kadj"), ("kexc", ~d["rrt0"], None)):
        s, w = unit_fit(d, m, cache, name, spec)
        res[f"{name}_primary"] = cif(s, w, s["t_del"]) if s is not None else nan_eff()
        res[f"{name}_negctl"] = cif(s, w, s["t_hypo"]) if s is not None else nan_eff()
    s, w = unit_fit(d, ~d["bleed"], cache, "bleedexc")
    res["bleedexc_primary"] = cif(s, w, s["t_del"]) if s is not None else nan_eff()
    return res


def names_of(key):
    if key[1] == "1":
        return ["primary", "negctl", "kadj_primary", "kadj_negctl", "kexc_primary", "kexc_negctl", "bleedexc_primary"], [], []
    R = ["primary", "del72", "negctl"]
    for u in UNITS[1:]:
        R += [f"{u}_primary", f"{u}_negctl"]
    R += ["egfr60", "age65", "bneg_primary", "bneg_negctl", "buta_primary", "bnone_primary", "bneg_light_primary",
          "h24_primary", "h24_negctl", "h24_light_primary"]
    R += [f"lm{L}_{u}" for L in LMS for u in ("all", "light")]
    R += ["sev_primary", "sev_negctl", "kadj_primary", "kadj_negctl", "kexc_primary", "kexc_negctl"]
    if key[2] == "e":
        R += ["dx_primary", "dx_negctl"]
    return R, [f"ic_{u}" for u in UNITS], [f"scr_{u}" for u in UNITS]


def flat(res, key):
    R, HR, SC = names_of(key)
    v = []
    for nm in R:
        v += [res[nm].get("riskA", np.nan), res[nm].get("riskB", np.nan)]
    for nm in HR:
        x = res[nm].get("HR")
        v.append(math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan)
    for nm in SC:
        x = res[nm].get("ratio")
        v.append(math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan)
    return np.array(v, float)


def est_key(d, key, cache):
    return est_c2(d, key[2], cache) if key[1] == "2" else est_c1(d, key[2], cache)


def sep_primary(b):
    keep = [j for j, nm in enumerate(b["names"]) if not nm.startswith("hosp_")]
    vals, counts = np.unique(b["site"], return_counts=True)
    ref = vals[int(np.argmax(counts))]
    cols = [(b["site"] == v).astype(float) for v in vals if v != ref]
    X = np.column_stack([b["X"][:, keep]] + cols)
    ps, _ = sc.ps_fit(X, b["A"])
    return cif(b, sc.overlap_w(ps, b["A"]), b["t_del"])


def cnt(k):
    return sc.mask_n(int(k))


def pct(k, n, nd=1):
    return round(100.0 * k / n, nd) if (n > 0 and k >= 10) else None


def share(x, r):
    k, n = int((x & r).sum()), int(r.sum())
    return {"n": cnt(k), "pct": pct(k, n)}


def wshare(x, r, w):
    k = int((x & r).sum())
    sw = float(w[r].sum())
    return round(100.0 * float(w[r & x].sum()) / sw, 1) if (k >= 10 and sw > 0) else None


def quart(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [round(float(q), 1) for q in np.percentile(v, [25, 50, 75])] if v.size >= MIN_N else None


def arms(d):
    return (("A", d["A"] == 1), ("B", d["A"] == 0))


def desc_assess(s, w):
    T = np.minimum(s["t_end"], float(H))
    anyf, okf = s["scr_any"] > 0, s["scr_ok"] > 0
    out = {}
    for lab, r in arms(s):
        e = {"n": cnt(r.sum())}
        for nm, x in (("never", ~anyf), ("uta_only", anyf & ~okf), ("assessable", okf),
                      ("first_ok_12h", s["t_first_ok"] <= 12), ("first_ok_24h", s["t_first_ok"] <= 24),
                      ("base_neg", s["base"] == "neg"), ("base_uta", s["base"] == "uta"), ("base_none", s["base"] == "none")):
            e[nm] = dict(share(x, r), wpct=wshare(x, r, w))
        e["hours_to_first_ok_q"] = quart(s["t_first_ok"][r])
        e["never_icu_hours_q"] = quart(T[r & ~anyf])
        e["ok_per_day"] = round(float((w * s["scr_ok"])[r].sum() / max((w * s["scr_days"])[r].sum(), 1e-12)), 3)
        e["ok_per_day_unw"] = round(float(s["scr_ok"][r].sum() / max(s["scr_days"][r].sum(), 1e-12)), 3)
        out[lab] = e
    return out


def desc_timing(d):
    h = d["h_icu"]
    out = {}
    for lab, r in arms(d):
        e = {"n": cnt(r.sum()), "q": quart(h[r])}
        for nm, x in (("lt6", h < 6), ("6_24", (h >= 6) & (h < 24)), ("24_72", (h >= 24) & (h < 72)), ("ge72", h >= 72), ("lt24", h < 24)):
            e[nm] = share(x, r)
        e["bins"] = [cnt(((h >= a) & (h < b) & r).sum()) for a, b in zip(HBINS[:-1], HBINS[1:])]
        out[lab] = e
    return out


def curves(s, w):
    kend, typ, _ = sc.follow_types(s["t_end"], s["end_death"], s["t_del"], disc_competing=True)
    dth = np.asarray(s["end_death"], bool)
    cause = np.where(typ == 1, 1, np.where((typ == 2) & dth, 3, np.where(typ == 2, 2, 0)))
    w = np.asarray(w, float)
    out = {}
    for lab, r in arms(s):
        e = {}
        for nm, c in (("delirium", 1), ("discharge", 2), ("death", 3)):
            t2 = np.where(cause[r] == c, 1, np.where(cause[r] > 0, 2, 0))
            e[nm] = [0.0] + [round(float(x), 6) for x in sc.aj_cif(w[r], kend[r], t2)]
        out[lab] = e
    return out


def desc_onset(d, ow):
    out = {}
    T = np.minimum(d["t_end"], float(H))
    det = np.isfinite(d["t_del"]) & (d["t_del"] <= T)
    cv = curves(d, ow)
    for lab, r in arms(d):
        out[lab] = {"q_hours_to_first_positive": quart(d["t_del"][r & det]), "n_detected": cnt((r & det).sum()),
                    "cif": {str(h): round(cv[lab]["delirium"][h], 4) for h in HOURS}}
    return out


def desc_sev(d, db):
    out = {}
    for lab, r in arms(d):
        e = {"n": cnt(r.sum())}
        if db == "e":
            for c in SEV_E_RAW:
                e[c] = share(np.isfinite(d["raw_" + c]), r)
            e["vent_pre"] = share(d["vent_pre"], r)
        else:
            for nm, x in (("sofa_prev_recorded", np.isfinite(d["sofa_prev"])), ("sofa_prev_ge3", d["sofa_prev"] >= 3), ("sofa_prev_ge6", d["sofa_prev"] >= 6),
                          ("sofa_prev_ge9", d["sofa_prev"] >= 9), ("sofa_main_ge3", d["sofa_main"] >= 3), ("sofa_main_ge6", d["sofa_main"] >= 6),
                          ("sofa_main_ge9", d["sofa_main"] >= 9), ("sofa_same_band", np.digitize(np.nan_to_num(d["sofa_prev"], nan=-1), [0, 3, 6, 9])
                                                                 == np.digitize(np.nan_to_num(d["sofa_main"], nan=-1), [0, 3, 6, 9]))):
                e[nm] = share(x, r)
        out[lab] = e
    return out


def desc_kidney(d, db, comparison):
    g = d["egfr"]
    cats = (("egfr_ge60", g >= 60), ("egfr_30_59", (g >= 30) & (g < 60)), ("egfr_lt30", g < 30), ("scr_not_measured", ~np.isfinite(g)))
    extra = [("ckd", d["ckd"]), ("dialysis_history", d["dial_wide"]), ("rrt_before_t0", d["rrt_pre"]), ("rrt0", d["rrt0"])]
    if db == "m":
        extra += [("ckd_prior_only", d["ckd_prior"]), ("dialysis_prior_only", d["dial_prior"])]
    out = {"by_arm": {}, "share_A": {}}
    for lab, r in arms(d):
        out["by_arm"][lab] = dict({"n": cnt(r.sum())}, **{nm: share(x, r) for nm, x in list(cats) + extra})
    A = d["A"] == 1
    for nm, x in cats:
        k, n = int((x & A).sum()), int(x.sum())
        out["share_A"][nm] = pct(k, n) if (n - k) >= 10 else None
    return out


def desc_tx2(d):
    out = {}
    for lab, r in arms(d):
        e = {"n": cnt(r.sum()), "stop_unusable": share(d["stop_unusable"], r)}
        for T in TPTS:
            inicu = d["t_end"] > T
            den = r & inicu
            i, o = d[f"init_{T}"], d[f"other_{T}"]
            e[str(T)] = {"in_icu": share(inicu, r), "left_alive": share(~inicu & ~d["end_death"], r), "died": share(~inicu & d["end_death"], r),
                         "init_only": share(i & ~o, den), "both": share(i & o, den), "other_only": share(~i & o, den), "neither": share(~i & ~o, den),
                         "started_other_by": share(np.isfinite(d["t_other"]) & (d["t_other"] <= T), r)}
        out[lab] = e
    kend, typ, ck = sc.follow_types(d["t_end"], d["end_death"], d["t_del"], t_dev=d["t_dev"], disc_competing=True)
    _, info = asd.censor_w(d["X"], d["A"], kend, typ, ck, False, True, {}, "pp")
    for lab, r in arms(d):
        a = 1 if lab == "A" else 0
        out[lab]["pp_censored"] = share(ck == 2, r)
        out[lab]["pp_weight_cut_max"] = [round(float(x), 3) for x in info[a]]
    return out


def dose_cat(act, daily):
    return np.where(~act, "none", np.where(~np.isfinite(daily), "unk", np.where(daily <= 20, "low", np.where(daily >= 40, "high", "mid"))))


def desc_tx1(d):
    out = {}
    start = {"A": "low", "B": "high"}
    for lab, r in arms(d):
        e = {"n": cnt(r.sum())}
        changed = np.zeros(len(d["A"]), bool)
        for T in (24, 48, 72):
            inicu = d["t_end"] > T
            den = r & inicu
            c = dose_cat(d[f"act_{T}"], d[f"daily_{T}"])
            e[str(T)] = dict({"in_icu": share(inicu, r)}, **{k: share(c == k, den) for k in ("low", "mid", "high", "unk", "none")})
            changed |= inicu & np.isin(c, ("low", "mid", "high")) & (c != start[lab])
        e["category_changed_by_72h"] = share(changed, r)
        out[lab] = e
    return out


def desc_groups(d, w, values, levels):
    M = np.column_stack([values == g for g in levels]).astype(float)
    su, sw_ = sc.smd_table(M, list(levels), d["A"]), sc.smd_table(M, list(levels), d["A"], w)
    out = {}
    for j, g in enumerate(levels):
        x = M[:, j] == 1
        out[g] = {"A": share(x, d["A"] == 1), "B": share(x, d["A"] == 0), "smd": round(su[g], 3), "smd_w": round(sw_[g], 3)}
    return out


def desc_ps(d, ps, ow):
    edges = np.linspace(0, 1, 21)
    out = {"hist": {lab: [cnt(k) for k in np.histogram(ps[r], bins=edges)[0]] for lab, r in arms(d)},
           "ess": {lab: round(sc.ess(ow[r]), 1) for lab, r in arms(d)}, "columns": {}}
    A = d["A"] == 1
    n = len(A)
    for j, nm in enumerate(d["names"]):
        x = d["X"][:, j]
        if set(np.unique(x).tolist()) <= {0.0, 1.0}:
            out["columns"][nm] = {"all_pct": pct(int(x.sum()), n), "A_pct": pct(int(x[A].sum()), int(A.sum())), "B_pct": pct(int(x[~A].sum()), int((~A).sum())),
                                  "overlap_pct": round(100.0 * sc.wmean(x[A], ow[A]), 1)}
    for nm, x in d["extras"].items():
        okx = np.isfinite(x)
        out["columns"]["mean_" + nm] = {"all": round(float(np.mean(x[okx])), 1), "A": round(float(np.mean(x[okx & A])), 1), "B": round(float(np.mean(x[okx & ~A])), 1),
                                        "overlap": round(float(sc.wmean(x[okx], ow[okx])), 1)}
    if "surgical" in d["names"]:
        s = d["X"][:, d["names"].index("surgical")] == 1
        out["surgical_share_patients"] = pct(int(s.sum()), n)
        out["surgical_share_weight"] = round(100.0 * float(ow[s].sum() / ow.sum()), 1)
    return out


def loo_hospitals(d):
    site = d["site"]
    vals, counts = np.unique(site, return_counts=True)
    order = np.argsort(-counts, kind="mergesort")
    out = []
    for rank, j in enumerate(order, 1):
        s, w = unit_fit(d, site != vals[j], {}, "loo")
        e = cif(s, w, s["t_del"]) if s is not None else nan_eff()
        a = int(((site == vals[j]) & (d["A"] == 1)).sum())
        out.append({"hospital": f"H{rank:02d}", "n": cnt(counts[j]), "ppi_share_pct": pct(a, int(counts[j])) if (counts[j] - a) >= 10 else None,
                    "OR_without": round(e["OR"], 4)})
    return out


def describe_c2(d, db, cache):
    A = d["A"]
    ps = asd.ow_fit(d, cache, "full")
    ow = sc.overlap_w(ps, A)
    out = {"n_units": {u: {lab: cnt((unit_mask(d, u) & r).sum()) for lab, r in arms(d)} for u in UNITS},
           "base_positive_at_last_time": cnt(d["base_pos"].sum()), "assess": {}, "curves": {}}
    for u in UNITS:
        s, w = (d, ow) if u == "all" else unit_fit(d, unit_mask(d, u), {}, "desc_" + u)
        if s is not None:
            out["assess"][u] = desc_assess(s, w)
        if u in ("all", "light") and s is not None:
            out["curves"][u] = curves(s, w)
    out["timing"] = desc_timing(d)
    out["onset"] = desc_onset(d, ow)
    out["severity"] = desc_sev(d, db)
    out["kidney"] = desc_kidney(d, db, 2)
    out["tx"] = desc_tx2(d)
    out["ps"] = desc_ps(d, ps, ow)
    out["n_lm"] = {str(L): {lab: cnt((lm_mask(d, L) & r).sum()) for lab, r in arms(d)} for L in LMS}
    out["columns_sev"] = [c for c in d["names_sev"] if c not in d["names"]]
    out["columns_kadj"] = [c for c in d["names_kadj"] if c not in d["names"]]
    if db == "e":
        out["dx"] = desc_groups(d, ow, d["dxgroup"], sorted(set(d["dxgroup"])))
        out["columns_dx"] = [c for c in d["names_dx"] if c not in d["names"]]
        out["loo"] = loo_hospitals(d)
        out["n_hospitals"] = int(np.unique(d["site"]).size)
    else:
        out["dx"] = {"pdx": desc_groups(d, ow, d["pdx"], PDX), "adm": desc_groups(d, ow, d["adm"], ADM)}
    return out


def describe_c1(d, db, cache):
    out = {"kidney": desc_kidney(d, db, 1), "tx": desc_tx1(d), "bleed": {}}
    for lab, r in arms(d):
        out["bleed"][lab] = {"n": cnt(r.sum()), "bleed_dx": share(d["bleed_dx"], r), "ppi_infusion": share(d["ppi_infusion"], r), "either": share(d["bleed"], r)}
    out["columns_kadj"] = [c for c in d["names_kadj"] if c not in d["names"]]
    return out


def repro_checks(key, res, R):
    comparison, db = key[1], key[2]
    st = json.load(open(os.path.join(R, f"{asd.STEM[int(comparison)]}_estimates.json"), encoding="utf-8"))["per_db"][db]["analyses"]
    checks = [(f"{key} primary OR", res["primary"]["OR"], st["primary"]["OR"]), (f"{key} primary RR", res["primary"]["RR"], st["primary"]["RR"]),
              (f"{key} negative control OR", res["negctl"]["OR"], st["negctl"]["OR"])]
    if comparison == "2":
        checks += [(f"{key} 72-h OR", res["del72"]["OR"], st["del72"]["OR"]), (f"{key} RASS >= -3 OR", res["rass3_primary"]["OR"], st["rass3"]["OR"]),
                   (f"{key} eGFR < 60 OR", res["egfr60"]["OR"], st["egfr60"]["OR"]), (f"{key} age >= 65 OR", res["age65"]["OR"], st["age65"]["OR"])]
        j15 = json.load(open(os.path.join(R, "r15.json"), encoding="utf-8"))["est"][db]
        for u in STRATA:
            for o in ("primary", "negctl"):
                checks.append((f"{key} stratum {u} {o} OR vs R15", res[f"{u}_{o}"]["OR"], j15[f"{u}_{o}"]["OR"]))
        checks.append((f"{key} RASS >= -3 negative control OR vs R15", res["rass3_negctl"]["OR"], j15["rass3_negctl"]["OR"]))
        j11 = json.load(open(os.path.join(R, "r11.json"), encoding="utf-8"))["db"][db]
        checks += [(f"{key} interval-censored HR vs R11", res["ic_all"]["HR"], j11["ic"]["HR"]),
                   (f"{key} interval-censored HR, RASS >= -3, vs R11", res["ic_rass3"]["HR"], j11["ic_rass3"]["HR"]),
                   (f"{key} assessable screens per ICU day, ratio, vs R11", res["scr_all"]["ratio"], j11["screen"]["scr_ok"]["ratio"])]
    lines, good = [], True
    for name, x, ref in checks:
        tol = 5e-4 if "HR" in name else 1e-9
        ok = x is not None and ref is not None and np.isfinite(x) and abs(round(float(x), 4) - float(ref)) < tol + 1e-12
        good &= bool(ok)
        lines.append(f"{'PASS' if ok else 'FAIL'} | {name}: {x:.4f} vs {ref:.4f}" if (x is not None and ref is not None) else f"FAIL | {name}: no estimate")
    return lines, good


def estimate(args):
    t0 = time.time()
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    allgood = True
    for key in asd.KEYS:
        d, flow = load_key(P, key)
        cache = {}
        res = est_key(d, key, cache)
        lines, good = repro_checks(key, res, R)
        for ln in lines:
            print(ln, flush=True)
        allgood &= good
        if not good:
            continue
        desc = describe_c2(d, key[2], {}) if key[1] == "2" else describe_c1(d, key[2], {})
        json.dump({"flow": flow, "res": asd.to_json(res), "desc": asd.to_json(desc), "checks": lines, "n": int(len(d["A"])),
                   "point": flat(res, key).tolist()}, open(os.path.join(P, f"r16_est_{key}.json"), "w"))
        np.savez(os.path.join(P, f"r16_cache_{key}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
        asd.status(f"{key} estimated", t0)
    if not allgood:
        asd.log_run(args.results, "r16 estimate: REPRODUCTION CHECK FAILED")
        raise SystemExit("the rebuilt data differ from the stored results: stop")
    asd.status("r16 estimate done; reproduction checks passed", t0)
    asd.log_run(args.results, f"r16 estimate: ok, reproduction checks passed ({time.time() - t0:.0f} s)")


def draw(d, db, scheme, rep):
    rng = np.random.default_rng(asd.SEED + rep)
    if scheme == "pat" or db == "m":
        n = len(d["A"])
        return asd.take(d, rng.integers(0, n, n))
    return asd.resample(d, db, rng)


def one_rep(d, key, scheme, rep, cache):
    try:
        if scheme == "sep":
            b = asd.resample(d, "e", np.random.default_rng(asd.SEED + rep))
            shared = cif(b, sc.overlap_w(asd.ow_fit(b, dict(cache), "full"), b["A"]), b["t_del"])
            sep = sep_primary(b)
            return np.array([shared["riskA"], shared["riskB"], sep["riskA"], sep["riskB"]], float)
        return flat(est_key(draw(d, key[2], scheme, rep), key, dict(cache)), key)
    except Exception:
        return None


def boot_cols(key, scheme):
    if scheme == "sep":
        return ["shared_A", "shared_B", "sep_A", "sep_B"]
    R, HR, SC = names_of(key)
    return [f"{nm}:{a}" for nm in R for a in ("A", "B")] + HR + SC


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    key, scheme = args.key, args.scheme
    if scheme == "sep" and key != "c2e":
        raise SystemExit("scheme sep is for c2e only")
    d, _ = load_key(P, key)
    z = np.load(os.path.join(P, f"r16_cache_{key}.npz"))
    cache = {k.replace("__", ":"): z[k] for k in z.files}
    reps = list(range(args.start, min(args.stop, NBOOT)))
    cols = boot_cols(key, scheme)
    V = np.full((len(reps), len(cols)), np.nan)
    failed = 0
    for i, rep in enumerate(reps):
        v = one_rep(d, key, scheme, rep, cache)
        if v is None:
            failed += 1
        else:
            V[i] = v
    np.savez(os.path.join(P, f"r16boot_{key}_{scheme}_{args.start:04d}_{args.stop:04d}.npz"), reps=np.array(reps), v=V, cols=np.array(cols))
    asd.status(f"r16 boot {key} {scheme} {args.start}-{args.stop} done, failed {failed}", t0)
    asd.log_run(args.results, f"r16 boot --key {key} --scheme {scheme} --start {args.start} --stop {args.stop}: ok, failed {failed} ({time.time() - t0:.0f} s)")


def load_boots(P, key, scheme):
    files = sorted(glob.glob(os.path.join(P, f"r16boot_{key}_{scheme}_*.npz")))
    if not files:
        return None
    reps, vs, cols = [], [], None
    for f in files:
        zz = np.load(f)
        reps.append(zz["reps"])
        vs.append(zz["v"])
        cols = [str(x) for x in zz["cols"]]
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    V = np.concatenate(vs)[first]
    return {"cols": cols, "v": V, "n": int(first.size), "failed": int(np.isnan(V).all(axis=1).sum())}


def col(B, name):
    return B["v"][:, B["cols"].index(name)] if (B is not None and name in B["cols"]) else None


def scales(rA, rB):
    rA, rB = np.asarray(rA, float), np.asarray(rB, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        okk = (rA > 0) & (rA < 1) & (rB > 0) & (rB < 1)
        lor = np.where(okk, np.log(rA / (1 - rA)) - np.log(rB / (1 - rB)), np.nan)
        lrr = np.where(okk, np.log(rA / rB), np.nan)
    return {"OR": lor, "RR": lrr, "RD": np.where(okk, rA - rB, np.nan)}


def pctl(v):
    v = v[np.isfinite(v)]
    if v.size < asd.MIN_BOOT:
        return None
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(v.std(ddof=1)), int(v.size)


def risk_entry(point, B, name):
    e = {k: point.get(k) for k in ("riskA", "riskB", "OR", "RR", "RD")}
    for k in ("eventsA", "eventsB", "nA", "nB"):
        if k in point:
            e[k] = cnt(point[k])
    rA, rB = col(B, name + ":A"), col(B, name + ":B")
    if rA is None or e["OR"] is None or not np.isfinite(e["OR"]):
        return e
    sca = scales(rA, rB)
    for k in ("OR", "RR", "RD"):
        q = pctl(sca[k])
        if q is not None:
            lo, hi, se, n = q
            e[k + "_lo"], e[k + "_hi"] = (math.exp(lo), math.exp(hi)) if k != "RD" else (lo, hi)
            e["se_" + ("log" if k != "RD" else "") + k] = se
            e["boot_ok"] = n
    return e


def log_entry(point, B, name, key="HR"):
    e = {key: point}
    v = col(B, name)
    q = pctl(v) if v is not None else None
    if q is not None and point is not None and np.isfinite(point):
        e["lo"], e["hi"], e["se_log"], e["boot_ok"] = math.exp(q[0]), math.exp(q[1]), q[2], q[3]
    return e


def paired(B, num, den):
    a, b = scales(col(B, num + ":A"), col(B, num + ":B")), scales(col(B, den + ":A"), col(B, den + ":B"))
    out = {}
    for k in ("OR", "RR", "RD"):
        q = pctl(a[k] - b[k])
        if q is not None:
            out[k] = {"lo": math.exp(q[0]) if k != "RD" else q[0], "hi": math.exp(q[1]) if k != "RD" else q[1]}
    return out


def wald_strata(points, B, names):
    from scipy.stats import chi2
    out = {}
    reps = [scales(col(B, nm + ":A"), col(B, nm + ":B")) for nm in names]
    for k in ("OR", "RR", "RD"):
        th = np.array([(math.log(p[k]) if k != "RD" else p[k]) if (p.get(k) is not None and np.isfinite(p[k])) else np.nan for p in points])
        M = np.column_stack([r[k] for r in reps])
        M = M[np.isfinite(M).all(axis=1)]
        if not np.isfinite(th).all() or M.shape[0] < asd.MIN_BOOT:
            out[k] = None
            continue
        C = np.column_stack([-np.ones(len(names) - 1), np.eye(len(names) - 1)])
        V = C @ np.cov(M, rowvar=False) @ C.T
        cth = C @ th
        Q = float(cth @ np.linalg.solve(V, cth))
        out[k] = {"Q": Q, "df": len(names) - 1, "p": float(chi2.sf(Q, len(names) - 1)), "reps": int(M.shape[0])}
    return out


def read_wald(p):
    if p is None:
        return "cannot be tested"
    if p >= 0.05:
        return "no evidence that the stratum estimates differ"
    return "nominally different, not significant after correction for six stratifying dimensions" if p >= 0.05 / 6 else "still different after correction for six stratifying dimensions"


def pool(a, b, k="OR"):
    se = "se_log" + k if k in ("OR", "RR") else ("se_RD" if k == "RD" else "se_log")
    ok = [x for x in (a, b) if x and x.get(se) and x.get(k) is not None and np.isfinite(x[k]) and (k == "RD" or x[k] > 0)]
    if len(ok) < 2:
        return None
    est = [x[k] if k == "RD" else math.log(x[k]) for x in ok]
    m, s, Q, I2, _ = sc.pool_fe(est, [x[se] for x in ok])
    f = (lambda v: v) if k == "RD" else math.exp
    return {k: f(m), "lo": f(m - 1.96 * s), "hi": f(m + 1.96 * s), "I2": I2}


def between(a, b, k):
    se = "se_log" + k if k != "RD" else "se_RD"
    if not (a and b and a.get(se) and b.get(se)):
        return None
    dlt = (b[k] - a[k]) if k == "RD" else math.log(b[k]) - math.log(a[k])
    s = math.sqrt(a[se] ** 2 + b[se] ** 2)
    f = (lambda v: v) if k == "RD" else math.exp
    null = 0.0 if k == "RD" else 1.0
    lo, hi = f(dlt - 1.96 * s), f(dlt + 1.96 * s)
    return {"est": f(dlt), "lo": lo, "hi": hi, "excludes_null": bool(lo > null or hi < null)}


def evalue(rr, lo, hi):
    def ev(x):
        x = 1 / x if x < 1 else x
        return x + math.sqrt(x * (x - 1))
    if rr is None or not np.isfinite(rr):
        return None
    lim = None
    if lo is not None and hi is not None:
        lim = 1.0 if lo <= 1 <= hi else ev(hi if rr < 1 else lo)
    return {"point": ev(rr), "limit": lim}


def mcid_abs(p0, orr=asd.MCID):
    if p0 is None or not (0 < p0 < 1):
        return None
    o = orr * p0 / (1 - p0)
    return o / (1 + o) - p0


def reading(e):
    return asd.interpret(e["OR"], e.get("OR_lo", float("nan")), e.get("OR_hi", float("nan"))) if e and e.get("OR") is not None else "no interval"


def consistent(e, ref):
    if not e or e.get("OR") is None or e.get("OR_lo") is None or not ref or ref.get("OR_lo") is None:
        return "cannot be judged (no interval)"
    inside = ref["OR_lo"] <= e["OR"] <= ref["OR_hi"]
    same = reading(e) == reading(ref)
    if inside and same:
        return "estimate consistent with the main analysis"
    why = []
    if not inside:
        why.append("point estimate outside the main-analysis interval")
    if not same:
        why.append(f"reading changed from “{reading(ref)}” to “{reading(e)}”")
    return "estimate changed (" + "; ".join(why) + ")"


def read_ic(e):
    if not e or e.get("lo") is None:
        return "no interval"
    return "taking assessment times into account, the stratum still shows an association" if e["hi"] < 1 else "taking assessment times into account, the association in the stratum is no longer clear"


def read_assess(a, ratio):
    na, nb = a["A"]["never"]["pct"], a["B"]["never"]["pct"]
    if na is None or nb is None or not ratio or ratio.get("lo") is None:
        return "cannot be summarised"
    if abs(na - nb) < 5 and ratio["lo"] <= 1 <= ratio["hi"]:
        return "similar assessment process in the two arms"
    bits = []
    if abs(na - nb) >= 5:
        bits.append(f"share never assessed {'higher in the PPI arm' if na > nb else 'lower in the PPI arm'}")
    if not (ratio["lo"] <= 1 <= ratio["hi"]):
        bits.append(f"assessable screens per ICU day {'more in the PPI arm' if ratio['ratio'] > 1 else 'fewer in the PPI arm'}")
    return "assessment process differs between arms (" + "; ".join(bits) + ")"


def build_db(P, key):
    E = json.load(open(os.path.join(P, f"r16_est_{key}.json")))
    B = load_boots(P, key, "main")
    R, HR, SC = names_of(key)
    out = {"flow": {k: cnt(v) for k, v in E["flow"].items()}, "boot": {"main": {"reps": B["n"], "failed": B["failed"]} if B else None},
           "est": {nm: risk_entry(E["res"][nm], B, nm) for nm in R}, "ic": {}, "scr": {}, "desc": E["desc"], "checks": E["checks"]}
    for nm in HR:
        x = E["res"][nm]
        out["ic"][nm] = dict(log_entry(x.get("HR"), B, nm), **{k: cnt(x[k]) for k in ("n_inf_A", "n_inf_B", "eventsA", "eventsB") if k in x})
    for nm in SC:
        x = E["res"][nm]
        out["scr"][nm] = dict(log_entry(x.get("ratio"), B, nm, "ratio"), A=x.get("A"), B=x.get("B"))
    if key == "c2e":
        Bp = load_boots(P, key, "pat")
        out["boot"]["pat"] = {"reps": Bp["n"], "failed": Bp["failed"]} if Bp else None
        out["est_pat"] = {nm: risk_entry(E["res"][nm], Bp, nm) for nm in R} if Bp else {}
        out["ic_pat"] = {nm: log_entry(E["res"][nm].get("HR"), Bp, nm) for nm in HR} if Bp else {}
    return out, B, E


def interval_checks(o, key, R):
    lines, good = [], True
    comparison, db = key[1], key[2]
    st = json.load(open(os.path.join(R, f"{asd.STEM[int(comparison)]}_estimates.json"), encoding="utf-8"))["per_db"][db]["analyses"]
    pairs = [("primary", "primary", st["primary"]), ("negctl", "negctl", st["negctl"])]
    if comparison == "2":
        pairs += [("del72", "del72", st["del72"]), ("rass3_primary", "rass3", st["rass3"]), ("egfr60", "egfr60", st["egfr60"]), ("age65", "age65", st["age65"])]
    for nm, lab, ref in pairs:
        e = o["est"][nm]
        okk = e.get("OR_lo") is not None and abs(round(e["OR_lo"], 4) - ref["OR_lo"]) < 1e-9 and abs(round(e["OR_hi"], 4) - ref["OR_hi"]) < 1e-9 \
            and abs(round(e["RD_lo"], 4) - ref["RD_lo"]) < 1e-9 and abs(round(e["RD_hi"], 4) - ref["RD_hi"]) < 1e-9
        good &= bool(okk)
        lines.append(f"{'PASS' if okk else 'FAIL'} | {key} {lab}: OR interval {e.get('OR_lo', float('nan')):.4f}-{e.get('OR_hi', float('nan')):.4f} "
                     f"vs {ref['OR_lo']:.4f}-{ref['OR_hi']:.4f}; RD interval {e.get('RD_lo', float('nan')):.4f} to {e.get('RD_hi', float('nan')):.4f} "
                     f"vs {ref['RD_lo']:.4f} to {ref['RD_hi']:.4f}")
    if comparison == "2":
        j15 = json.load(open(os.path.join(R, "r15.json"), encoding="utf-8"))["est"][db]
        for u in STRATA:
            for oc in ("primary", "negctl"):
                e, ref = o["est"][f"{u}_{oc}"], j15[f"{u}_{oc}"]
                okk = e.get("OR_lo") is not None and abs(round(e["OR_lo"], 4) - ref["lo"]) < 1e-9 and abs(round(e["OR_hi"], 4) - ref["hi"]) < 1e-9
                good &= bool(okk)
                lines.append(f"{'PASS' if okk else 'FAIL'} | {key} stratum {u} {oc} vs R15: {e.get('OR_lo', float('nan')):.4f}-{e.get('OR_hi', float('nan')):.4f} "
                             f"vs {ref['lo']:.4f}-{ref['hi']:.4f}")
        j11 = json.load(open(os.path.join(R, "r11.json"), encoding="utf-8"))["db"][db]
        for nm, ref in (("ic_all", j11["ic"]), ("ic_rass3", j11["ic_rass3"])):
            e = o["ic"][nm]
            okk = e.get("lo") is not None and abs(e["lo"] - ref["lo"]) < 2e-3 and abs(e["hi"] - ref["hi"]) < 2e-3
            lines.append(f"{'PASS' if okk else 'NOTE'} | {key} {nm} vs R11 (within 0.002; not a gate): {e.get('lo', float('nan')):.4f}-{e.get('hi', float('nan')):.4f} "
                         f"vs {ref['lo']:.4f}-{ref['hi']:.4f}")
    if key == "c2e" and o.get("est_pat"):
        j12 = json.load(open(os.path.join(R, "r12.json"), encoding="utf-8"))["est"]
        for nm, ref in (("primary", j12["hosp_primary"]["pat"]), ("negctl", j12["hosp_negctl"]["pat"])):
            e = o["est_pat"][nm]
            okk = e.get("OR_lo") is not None and abs(round(e["OR_lo"], 4) - ref["lo"]) < 1e-9 and abs(round(e["OR_hi"], 4) - ref["hi"]) < 1e-9
            good &= bool(okk)
            lines.append(f"{'PASS' if okk else 'FAIL'} | c2e {nm}, patient-level bootstrap vs R12: {e.get('OR_lo', float('nan')):.4f}-{e.get('OR_hi', float('nan')):.4f} "
                         f"vs {ref['lo']:.4f}-{ref['hi']:.4f}")
    return lines, good


def dist_summary(v, edges):
    v = v[np.isfinite(v)]
    m, s = float(v.mean()), float(v.std(ddof=1))
    return {"n": int(v.size), "mean": m, "sd": s, "skew": float(((v - m) ** 3).mean() / s ** 3), "min": float(v.min()), "max": float(v.max()),
            "q": [float(x) for x in np.percentile(v, [2.5, 25, 50, 75, 97.5])], "hist": [int(k) for k in np.histogram(v, bins=edges)[0]]}


def section10(P, o, E):
    from scipy.stats import t as tdist
    out = {}
    Bs = load_boots(P, "c2e", "sep")
    if Bs:
        sh, sp = scales(col(Bs, "shared_A"), col(Bs, "shared_B"))["OR"], scales(col(Bs, "sep_A"), col(Bs, "sep_B"))["OR"]
        okk = np.isfinite(sh) & np.isfinite(sp)
        out["separate_indicators"] = {"reps": int(okk.sum()), "max_abs_diff_logOR": float(np.max(np.abs(sh[okk] - sp[okk]))),
                                      "shared": [math.exp(x) for x in np.percentile(sh[okk], [2.5, 97.5])],
                                      "separate": [math.exp(x) for x in np.percentile(sp[okk], [2.5, 97.5])]}
    edges = np.round(np.arange(-0.40, 0.4001, 0.02), 2)
    out["hist_edges_logOR"] = [float(x) for x in edges]
    for sch in ("main", "pat"):
        B = load_boots(P, "c2e", sch)
        if B:
            out["dist_" + ("hospital" if sch == "main" else "patient")] = dist_summary(scales(col(B, "primary:A"), col(B, "primary:B"))["OR"], edges)
    loo = E["desc"]["loo"]
    th = np.array([math.log(x["OR_without"]) for x in loo])
    G = len(th)
    sej = math.sqrt((G - 1) / G * float(((th - th.mean()) ** 2).sum()))
    pt = math.log(o["est"]["primary"]["OR"])
    q = float(tdist.ppf(0.975, G - 1))
    out["loo"] = loo
    out["jackknife"] = {"G": G, "se_logOR": sej, "lo": math.exp(pt - q * sej), "hi": math.exp(pt + q * sej),
                        "range_without": [float(math.exp(th.min())), float(math.exp(th.max()))]}
    both = {}
    for nm, e in o["est"].items():
        p = o.get("est_pat", {}).get(nm)
        if p and e.get("OR_lo") is not None and p.get("OR_lo") is not None:
            both[nm] = {"hospital": reading(e), "patient": reading(p), "same": reading(e) == reading(p)}
    out["both_intervals"] = both
    return out


def report(args):
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    out = {"plan": PLAN, "software": versions(), "checks": [], "db": {}, "pooled": {}, "between": {}, "reading": {}, "interaction": {},
           "evalue": {}, "mcid_abs": {}, "famdose": {}}
    Bm, Em, good = {}, {}, True
    for key in asd.KEYS:
        o, B, E = build_db(P, key)
        lines, g = interval_checks(o, key, R)
        out["checks"] += o.pop("checks") + lines
        good &= g
        Bm[key], Em[key] = B, E
        (out["db"] if key[1] == "2" else out["famdose"])[key[2]] = o
    for ln in out["checks"]:
        if ln.startswith("FAIL"):
            print(ln, flush=True)
    if not good:
        asd.log_run(args.results, "r16 report: INTERVAL REPRODUCTION CHECK FAILED")
        raise SystemExit("bootstrap intervals differ from the stored results: stop")
    m, e = out["db"]["m"], out["db"]["e"]
    R2, HR2, _ = names_of("c2m")
    for nm in R2:
        out["pooled"][nm] = {k: pool(m["est"][nm], e["est"][nm], k) for k in ("OR", "RR", "RD")}
    for nm in HR2:
        out["pooled"][nm] = {"HR": pool(m["ic"][nm], e["ic"][nm], "HR")}
    for nm in ("primary", "light_primary", "rass3_primary"):
        out["between"][nm] = {k: between(m["est"][nm], e["est"][nm], k) for k in ("OR", "RR", "RD")}
    bt = out["between"]["primary"]
    flags = {k: bt[k]["excludes_null"] for k in bt if bt[k]}
    out["reading"]["scale_dependence"] = "the reading of the difference between the databases does not depend on the effect scale" if len(set(flags.values())) == 1 else "the reading of the difference between the databases depends on the effect scale"
    for lab, x in (("primary_m", m["est"]["primary"]), ("primary_e", e["est"]["primary"])):
        out["evalue"][lab] = evalue(x["RR"], x.get("RR_lo"), x.get("RR_hi"))
    for lab, nm in (("primary_pooled", "primary"), ("rass3_pooled", "rass3_primary"), ("light_pooled", "light_primary")):
        p = out["pooled"][nm]["RR"]
        out["evalue"][lab] = dict(evalue(p["RR"], p["lo"], p["hi"]), I2=p["I2"]) if p else None
    for db, o in (("m", m), ("e", e)):
        out["mcid_abs"][db] = {nm: mcid_abs(o["est"][nm]["riskB"]) for nm in ("primary", "light_primary", "rass3_primary")}
        out["mcid_abs"]["famdose_" + db] = {"primary": mcid_abs(out["famdose"][db]["est"]["primary"]["riskB"])}
    for db, o in (("m", m), ("e", e)):
        B = Bm["c2" + db]
        names = [f"{u}_primary" for u in STRATA]
        w = wald_strata([o["est"][nm] for nm in names], B, names)
        out["interaction"][db] = {"vs_light": {u: paired(B, f"{u}_primary", "light_primary") for u in STRATA[1:]}, "wald": w,
                                  "reading": {k: read_wald(w[k]["p"] if w[k] else None) for k in w}}
        for u in STRATA[1:]:
            a, b = o["est"][f"{u}_primary"], o["est"]["light_primary"]
            for k in ("OR", "RR", "RD"):
                if k in out["interaction"][db]["vs_light"][u] and a.get(k) is not None and b.get(k) is not None:
                    out["interaction"][db]["vs_light"][u][k]["est"] = (a[k] - b[k]) if k == "RD" else a[k] / b[k]
    rd = out["reading"]
    for db, o in (("m", m), ("e", e)):
        r = rd.setdefault(db, {})
        for nm, x in o["est"].items():
            r[nm] = reading(x)
        main, light = o["est"]["primary"], o["est"]["light_primary"]
        r["consistency"] = {nm: consistent(o["est"][nm], light if "light" in nm else main)
                            for nm in ("bneg_primary", "bneg_light_primary", "h24_primary", "h24_light_primary", "lm6_all", "lm12_all", "lm6_light",
                                       "lm12_light", "sev_primary", "kadj_primary", "kexc_primary") + (("dx_primary",) if db == "e" else ())}
        r["ic"] = {nm: read_ic(o["ic"][nm]) for nm in o["ic"]}
        r["assess"] = {u: read_assess(o["desc"]["assess"][u], o["scr"][f"scr_{u}"]) for u in UNITS if u in o["desc"]["assess"]}
        c1 = out["famdose"][db]
        r["famdose"] = {nm: reading(x) for nm, x in c1["est"].items()}
        r["famdose_consistency"] = {nm: consistent(c1["est"][nm], c1["est"]["primary"]) for nm in ("kadj_primary", "kexc_primary", "bleedexc_primary")}
    rd["ic_light_pooled"] = read_ic(out["pooled"]["ic_light"]["HR"])
    out["pooled_famdose"] = {nm: {k: pool(out["famdose"]["m"]["est"][nm], out["famdose"]["e"]["est"][nm], k) for k in ("OR", "RR", "RD")}
                           for nm in names_of("c1m")[0]}
    out["section10"] = section10(P, e, Em["c2e"])
    fig = {"plan": PLAN, "curves": {db: out["db"][db]["desc"].pop("curves") for db in DBS},
           "ps_hist": {db: out["db"][db]["desc"]["ps"]["hist"] for db in DBS}, "ps_hist_edges": [round(x, 2) for x in np.linspace(0, 1, 21)],
           "timing_bins": {db: {a: out["db"][db]["desc"]["timing"][a]["bins"] for a in ("A", "B")} for db in DBS}, "timing_bin_labels": list(HBIN_LAB),
           "boot": {k: out["section10"][k] for k in ("hist_edges_logOR", "dist_hospital", "dist_patient") if k in out["section10"]},
           "loo": out["section10"]["loo"]}
    json.dump(asd.to_json(fig), open(os.path.join(R, "fig_values_r16.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(round_tree(asd.to_json(out)), open(os.path.join(R, "r16.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(R, "r16.md"), "w", encoding="utf-8").write(md(out))
    asd.log_run(args.results, "r16 report: ok, interval reproduction checks passed")
    print("report written", flush=True)


def versions():
    import duckdb
    import scipy
    return {"python": sys.version.split()[0], "duckdb": duckdb.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}


def round_tree(o, key=""):
    if isinstance(o, dict):
        return {k: round_tree(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [round_tree(v, key) for v in o]
    if isinstance(o, float):
        if key == "p":
            return float(f"{o:.3g}")
        return round(o, 3 if key.startswith("risk") else 4)
    return o


def f2(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def ci3(e, k="OR", nd=2):
    if not e or e.get(k) is None:
        return "—"
    lo, hi = e.get(k + "_lo", e.get("lo")), e.get(k + "_hi", e.get("hi"))
    sc_ = 100.0 if k == "RD" else 1.0
    nd = 1 if k == "RD" else nd
    return f"{f2(e[k] * sc_, nd)} ({f2(lo * sc_, nd)}–{f2(hi * sc_, nd)})" if lo is not None else f2(e[k] * sc_, nd)


def pc(x):
    return "—" if x is None else f"{x:.1f}%"


def sh(e):
    return "—" if not e else f"{e['n']} ({pc(e['pct'])})"


def risks(e):
    return f"{f2(100 * e['riskA'], 1)}% / {f2(100 * e['riskB'], 1)}%" if e and e.get("riskA") is not None else "—"


def row3(lab, e, extra=""):
    return f"| {lab} | {e.get('nA', '—')} / {e.get('nB', '—')} | {risks(e)} | {ci3(e, 'OR')} | {ci3(e, 'RR')} | {ci3(e, 'RD')} |{extra}"


HEAD3 = "| Analysis | Patients PPI / famotidine | Weighted 7-day risk | OR (95% CI) | RR (95% CI) | RD, percentage points (95% CI) |"


def md(o):
    L = [f"# Revision R16: analyses requested in the first round of peer review ({o['plan']})", "",
         "Aggregate results only: counts 1–9 as <10, otherwise the nearest multiple of 5; no percentage when the numerator is below 10. Arm A = PPI (dose comparison: ≤20 mg/day), arm B = famotidine (dose comparison: ≥40 mg/day). "
         "OR, RR <1 and RD <0: fewer in arm A. Intervals: bootstrap percentile intervals (MIMIC-IV by patient, eICU by hospital, unless stated otherwise).", "",
         f"Software: {json.dumps(o['software'], ensure_ascii=False)}", "", "## 0. Reproduction checks", ""]
    L += [f"- {x}" for x in o["checks"]]
    LAB = {"primary": "Main analysis", "del72": "Delirium within 72 h", "negctl": "Negative control (glucose <70 mg/dL)", "rass3_primary": "RASS ≥ −3 (planned)", "rass3_negctl": "RASS ≥ −3, negative control",
           "light_primary": "Awake or lightly sedated", "light_negctl": "Awake or lightly sedated, negative control", "moderate_primary": "Moderately sedated", "moderate_negctl": "Moderately sedated, negative control",
           "deep_primary": "Deeply sedated", "deep_negctl": "Deeply sedated, negative control", "missing_primary": "RASS not recorded", "missing_negctl": "RASS not recorded, negative control",
           "egfr60": "eGFR <60", "age65": "Age ≥65", "bneg_primary": "Negative screen before time zero", "bneg_negctl": "Negative screen before time zero, negative control",
           "buta_primary": "Only “unable to assess” before time zero", "bnone_primary": "No screen before time zero", "bneg_light_primary": "Negative before time zero and awake or lightly sedated",
           "h24_primary": "Started within 24 h of ICU admission", "h24_negctl": "Started within 24 h of ICU admission, negative control", "h24_light_primary": "Started within 24 h and awake or lightly sedated",
           "lm6_all": "Landmark 6 h, all", "lm12_all": "Landmark 12 h, all", "lm6_light": "Landmark 6 h, awake or lightly sedated", "lm12_light": "Landmark 12 h, awake or lightly sedated",
           "sev_primary": "Model with severity measures before time zero", "sev_negctl": "Model with severity measures before time zero, negative control", "kadj_primary": "Chronic kidney disease and kidney replacement added",
           "kadj_negctl": "Chronic kidney disease and kidney replacement added, negative control", "kexc_primary": "Patients on kidney replacement removed", "kexc_negctl": "Patients on kidney replacement removed, negative control",
           "dx_primary": "Admission diagnosis groups added", "dx_negctl": "Admission diagnosis groups added, negative control", "bleedexc_primary": "Patients with a bleeding indication or PPI infusion removed"}
    for db in DBS:
        d = o["db"][db]
        L += ["", f"## {DBNAME[db]} (PPI comparison)", "", f"Bootstrap: {json.dumps(d['boot'], ensure_ascii=False)}", "", "### Effect sizes (sections 1, 4, 6, 7, 8, 9 and 15)", "", HEAD3 + " Reading (SAP section 8) |",
              "|---|---|---|---|---|---|---|"]
        for nm in names_of("c2" + db)[0]:
            L.append(row3(LAB.get(nm, nm), d["est"][nm], f" {o['reading'][db].get(nm, '')} |"))
        L += ["", "Written “consistent / changed” reading:", ""] + [f"- {LAB.get(k, k)}: {v}" for k, v in o["reading"][db]["consistency"].items()]
        if db == "e" and d.get("est_pat"):
            L += ["", "### Intervals from resampling patients (section 10)", "", "| Analysis | OR (by hospital) | OR (by patient) | Same reading with both intervals |", "|---|---|---|---|"]
            for nm, v in o["section10"]["both_intervals"].items():
                L.append(f"| {LAB.get(nm, nm)} | {ci3(d['est'][nm])} | {ci3(d['est_pat'][nm])} | {'yes' if v['same'] else 'no: ' + v['hospital'] + ' / ' + v['patient']} |")
        it = o["interaction"][db]
        L += ["", "### Interaction with sedation strata (section 2)", "", "| Compared with the awake or light stratum | Ratio of ORs | Ratio of RRs | Difference of RDs, percentage points |", "|---|---|---|---|"]
        for u in STRATA[1:]:
            v = it["vs_light"][u]
            L.append(f"| {LAB[u + '_primary']} | " + " | ".join(
                (f"{f2(v[k]['est'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)} ({f2(v[k]['lo'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)}–"
                 f"{f2(v[k]['hi'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)})") if (k in v and 'est' in v[k]) else "—" for k in ("OR", "RR", "RD")) + " |")
        L += ["", "Wald test of equality across the four strata: " + "; ".join(f"{k} scale Q={f2(w['Q'])}, p={w['p']:.4f} ({it['reading'][k]})" if w else f"{k} scale cannot be tested"
                                               for k, w in it["wald"].items())]
        L += ["", "### Interval-censored model (section 5)", "", "| Unit | HR (95% CI) | Patients in the likelihood PPI / famotidine | Reading |", "|---|---|---|---|"]
        for nm, x in d["ic"].items():
            L.append(f"| {nm} | {ci3(x, 'HR')} | {x.get('n_inf_A', '—')} / {x.get('n_inf_B', '—')} | {o['reading'][db]['ic'][nm]} |")
        L += ["", "### Assessment process (section 3; unweighted shares, overlap weighted within the unit in brackets)", "",
              "| Unit | Arm | Patients | Never assessed | Only unable to assess | At least one assessable | Within 12 h | Within 24 h | Hours to the first assessable Q1/median/Q3 | ICU hours of those never assessed | Assessable per ICU day | Before time zero: negative / unable to assess / none |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for u, a in d["desc"]["assess"].items():
            for arm in ("A", "B"):
                x = a[arm]
                w_ = lambda k: f"{sh(x[k])} ({pc(x[k]['wpct'])})"
                L.append(f"| {u} | {arm} | {x['n']} | {w_('never')} | {w_('uta_only')} | {w_('assessable')} | {pc(x['first_ok_12h']['pct'])} | {pc(x['first_ok_24h']['pct'])} | "
                         f"{x['hours_to_first_ok_q']} | {x['never_icu_hours_q']} | {x['ok_per_day']} (unweighted {x['ok_per_day_unw']}) | "
                         f"{pc(x['base_neg']['pct'])} / {pc(x['base_uta']['pct'])} / {pc(x['base_none']['pct'])} |")
        L += ["", "Ratio of assessable screens per ICU day (PPI vs famotidine) and the written summary:"] + [
            f"- {u}: {ci3(d['scr']['scr_' + u], 'ratio')}; {o['reading'][db]['assess'].get(u, '—')}" for u in UNITS if u in d["desc"]["assess"]]
        t = d["desc"]["timing"]
        L += ["", "### ICU admission to time zero (section 6)", "", "| Arm | Patients | Hours Q1/median/Q3 | <6 h | 6–<24 h | 24–<72 h | ≥72 h | <24 h |", "|---|---|---|---|---|---|---|---|"]
        for arm in ("A", "B"):
            x = t[arm]
            L.append(f"| {arm} | {x['n']} | {x['q']} | {sh(x['lt6'])} | {sh(x['6_24'])} | {sh(x['24_72'])} | {sh(x['ge72'])} | {sh(x['lt24'])} |")
        on = d["desc"]["onset"]
        L += ["", "### Time to the first positive screen (section 7)", "", "| Arm | First positive within 168 h | Hours Q1/median/Q3 | " + " | ".join(f"{h} h" for h in HOURS) + " |",
              "|---|---|---|" + "---|" * len(HOURS)]
        for arm in ("A", "B"):
            x = on[arm]
            L.append(f"| {arm} | {x['n_detected']} | {x['q_hours_to_first_positive']} | " + " | ".join(f"{100 * x['cif'][str(h)]:.1f}%" for h in HOURS) + " |")
        L += ["", f"Landmark populations (A / B): " + "; ".join(f"{k} h {v['A']} / {v['B']}" for k, v in d["desc"]["n_lm"].items())]
        sv = d["desc"]["severity"]
        L += ["", "### Severity measures before time zero (section 8)", "", f"Columns added to the model: {', '.join(d['desc']['columns_sev'])}", "", "| Measure | A | B |", "|---|---|---|"]
        for k in sv["A"]:
            if k != "n":
                L.append(f"| {k} | {sh(sv['A'][k])} | {sh(sv['B'][k])} |")
        kd = d["desc"]["kidney"]
        L += ["", "### Kidney function (section 9)", "", "| Measure | A | B | Share of arm A in the category |", "|---|---|---|---|"]
        for k in kd["by_arm"]["A"]:
            if k != "n":
                L.append(f"| {k} | {sh(kd['by_arm']['A'][k])} | {sh(kd['by_arm']['B'][k])} | {pc(kd['share_A'].get(k))} |")
        tx = d["desc"]["tx"]
        L += ["", "### Treatment after the first dose (section 11)", "", "| Time | Arm | Still in the ICU | Discharged alive | Died | Initial class only | Both classes | Other class only | Neither | Started the other class (cumulative) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for T in TPTS:
            for arm in ("A", "B"):
                x = tx[arm][str(T)]
                L.append(f"| {T} h | {arm} | {sh(x['in_icu'])} | {sh(x['left_alive'])} | {sh(x['died'])} | {sh(x['init_only'])} | {sh(x['both'])} | {sh(x['other_only'])} | "
                         f"{sh(x['neither'])} | {sh(x['started_other_by'])} |")
        L += ["", "Censored for deviation in the per-protocol analysis (before the outcome): " + "; ".join(f"{arm} {sh(tx[arm]['pp_censored'])}, weight truncation point and maximum {tx[arm]['pp_weight_cut_max']}" for arm in ("A", "B")),
              "(MIMIC-IV: administration in the 24 h before each time; eICU: orders active at that time, orders without a usable stop time counted as active; time-zero orders without a usable stop time: "
              + "; ".join(f"{arm} {sh(tx[arm]['stop_unusable'])}" for arm in ("A", "B")) + ")"]
        ps = d["desc"]["ps"]
        L += ["", "### Overlap-weighted population (section 13)", "", f"Effective sample size: {ps['ess']}; surgical patients, share of patients {pc(ps.get('surgical_share_patients'))}, share of the overlap weight {pc(ps.get('surgical_share_weight'))}", "",
              "| Characteristic | Whole cohort | A | B | Overlap-weighted population |", "|---|---|---|---|---|"]
        for k, x in ps["columns"].items():
            if k.startswith("mean_"):
                L.append(f"| {k} | {x['all']} | {x['A']} | {x['B']} | {x['overlap']} |")
            elif not k.startswith("hosp_"):
                L.append(f"| {k} | {pc(x['all_pct'])} | {pc(x['A_pct'])} | {pc(x['B_pct'])} | {pc(x['overlap_pct'])} |")
        dx = d["desc"]["dx"]
        L += ["", "### Reasons for ICU admission (section 15)", "", "| Group | A | B | SMD | Weighted SMD |", "|---|---|---|---|---|"]
        for blk in ((dx,) if db == "e" else (dx["pdx"], dx["adm"])):
            for g, x in blk.items():
                L.append(f"| {g} | {sh(x['A'])} | {sh(x['B'])} | {x['smd']} | {x['smd_w']} |")
    L += ["", "## Pooling of the two databases (fixed effect) and comparison of the databases", "", "| Analysis | OR; I² | RR; I² | RD, percentage points; I² |", "|---|---|---|---|"]
    for nm, p in o["pooled"].items():
        if "OR" in p:
            L.append(f"| {LAB.get(nm, nm)} | " + " | ".join(
                (f"{f2(p[k][k] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)} ({f2(p[k]['lo'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)}–"
                 f"{f2(p[k]['hi'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)}); {100 * p[k]['I2']:.0f}%") if p[k] else "—" for k in ("OR", "RR", "RD")) + " |")
        else:
            h = p["HR"]
            L.append(f"| {nm} (HR) | {f2(h['HR'])} ({f2(h['lo'])}–{f2(h['hi'])}); {100 * h['I2']:.0f}% | — | — |" if h else f"| {nm} (HR) | — | — | — |")
    L += ["", "Comparison of the databases (eICU vs MIMIC-IV; ratios for OR and RR, difference for RD):", ""]
    for nm, b in o["between"].items():
        L.append(f"- {LAB.get(nm, nm)}: " + "; ".join(
            f"{k} {f2(b[k]['est'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)} ({f2(b[k]['lo'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)}–"
            f"{f2(b[k]['hi'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)})" for k in ("OR", "RR", "RD") if b[k]))
    L += [f"- Written reading: {o['reading']['scale_dependence']}", "", f"Interval-censored model, awake or light stratum pooled: {o['reading']['ic_light_pooled']}", "",
          "## E-values (from RR) and 0.75 on the absolute scale", ""]
    for k, v in o["evalue"].items():
        L.append(f"- {k}: point estimate {f2(v['point'])}, confidence limit {f2(v['limit'])}" + (f" (I² {100 * v['I2']:.0f}%)" if v and "I2" in v else "") if v else f"- {k}: —")
    for k, v in o["mcid_abs"].items():
        L.append(f"- Risk difference corresponding to OR 0.75, {k}: " + "; ".join(f"{LAB.get(n, n)} {f2(100 * x, 1) if x is not None else '—'} percentage points" for n, x in v.items()))
    s10 = o["section10"]
    L += ["", "## Bootstrap in eICU (section 10)", ""]
    if "separate_indicators" in s10:
        x = s10["separate_indicators"]
        L.append(f"- Separate hospital indicator columns for each copy: {x['reps']} replicates, largest difference in log OR from the original method {x['max_abs_diff_logOR']:.6f}; interval, original method {f2(x['shared'][0], 4)}–{f2(x['shared'][1], 4)}, "
                 f"separate indicators {f2(x['separate'][0], 4)}–{f2(x['separate'][1], 4)}")
    for k, lab in (("dist_hospital", "by hospital"), ("dist_patient", "by patient")):
        if k in s10:
            x = s10[k]
            L.append(f"- Bootstrap distribution of the main-analysis log OR, {lab}: {x['n']} replicates, mean {x['mean']:.4f}, SD {x['sd']:.4f}, skewness {x['skew']:.2f}, range {x['min']:.3f} to {x['max']:.3f}, "
                     f"2.5/25/50/75/97.5 percentiles {[round(v, 4) for v in x['q']]}")
    jk = s10["jackknife"]
    L += [f"- Leaving out one hospital at a time ({jk['G']} hospitals): OR range {f2(jk['range_without'][0])}–{f2(jk['range_without'][1])}; jackknife standard error (log OR) {jk['se_logOR']:.4f}, "
          f"t({jk['G'] - 1}) interval {f2(jk['lo'])}–{f2(jk['hi'])}", "", "| Hospital (by number of patients) | Patients | PPI share | OR without it |", "|---|---|---|---|"]
    L += [f"| {x['hospital']} | {x['n']} | {pc(x['ppi_share_pct'])} | {f2(x['OR_without'])} |" for x in s10["loo"]]
    L += ["", "## Dose comparison (famotidine dose, eGFR <60)", ""]
    for db in DBS:
        c = o["famdose"][db]
        L += [f"### {DBNAME[db]}", "", f"Bootstrap: {json.dumps(c['boot'], ensure_ascii=False)}", "", HEAD3 + " Reading |", "|---|---|---|---|---|---|---|"]
        for nm in names_of("c1" + db)[0]:
            L.append(row3(LAB.get(nm, nm), c["est"][nm], f" {o['reading'][db]['famdose'].get(nm, '')} |"))
        L += ["", "“Consistent / changed” reading: " + "; ".join(f"{LAB.get(k, k)}: {v}" for k, v in o["reading"][db]["famdose_consistency"].items()), ""]
        kd = c["desc"]["kidney"]
        L += ["| Measure | A (≤20 mg) | B (≥40 mg) |", "|---|---|---|"]
        for k in kd["by_arm"]["A"]:
            if k != "n":
                L.append(f"| {k} | {sh(kd['by_arm']['A'][k])} | {sh(kd['by_arm']['B'][k])} |")
        for k, lab in (("bleed_dx", "Bleeding or ulcer indication"), ("ppi_infusion", "Continuous PPI infusion"), ("either", "Either")):
            L.append(f"| {lab} | {sh(c['desc']['bleed']['A'][k])} | {sh(c['desc']['bleed']['B'][k])} |")
        tx = c["desc"]["tx"]
        L += ["", "| Time | Arm | Still in the ICU | ≤20 mg | In between | ≥40 mg | Not parsable | No active order |", "|---|---|---|---|---|---|---|---|"]
        for T in (24, 48, 72):
            for arm in ("A", "B"):
                x = tx[arm][str(T)]
                L.append(f"| {T} h | {arm} | {sh(x['in_icu'])} | {sh(x['low'])} | {sh(x['mid'])} | {sh(x['high'])} | {sh(x['unk'])} | {sh(x['none'])} |")
        L += ["", "Dose category changed within 72 h: " + "; ".join(f"{arm} {sh(tx[arm]['category_changed_by_72h'])}" for arm in ("A", "B")), ""]
    L += ["### Dose comparison, pooling of the two databases", "", "| Analysis | OR; I² | RR; I² | RD, percentage points; I² |", "|---|---|---|---|"]
    for nm, p in o["pooled_famdose"].items():
        L.append(f"| {LAB.get(nm, nm)} | " + " | ".join(
            (f"{f2(p[k][k] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)} ({f2(p[k]['lo'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)}–"
             f"{f2(p[k]['hi'] * (100 if k == 'RD' else 1), 1 if k == 'RD' else 2)}); {100 * p[k]['I2']:.0f}%") if p[k] else "—" for k in ("OR", "RR", "RD")) + " |")
    return "\n".join(L) + "\n"


def synth(n, rng, db="e", effect=0.0, early=0.0):
    n_hosp = 12
    site = rng.integers(0, n_hosp, n)
    u = rng.normal(size=n_hosp)[site]
    x = rng.normal(size=n)
    rass = rng.choice([np.nan, -5.0, -3.0, 0.0], size=n, p=[0.2, 0.15, 0.15, 0.5])
    A = (rng.random(n) < 1 / (1 + np.exp(-(0.2 + 0.8 * u + 0.5 * x)))).astype(int)
    t_end = rng.uniform(8, 240, n)
    base = 0.004 * np.exp(0.4 * x - 0.5 * u)
    e1_ = rng.exponential(1.0, n) / (base * np.exp((effect + early) * A))
    e2_ = 12.0 + rng.exponential(1.0, n) / (base * np.exp(effect * A))
    onset = np.where(e1_ <= 12.0, e1_, e2_)
    hypo = rng.exponential(1.0, n) / (0.0015 * np.exp(-0.5 * u))
    T = np.minimum(t_end, 168.0)
    k = np.maximum(np.floor((T - 4.0) / 8.0).astype(int) + 1, 0)
    sid = np.repeat(np.arange(n), k)
    h = 4.0 + 8.0 * (np.arange(sid.size) - np.repeat(np.cumsum(k) - k, k))
    okk = rng.random(sid.size) > 0.1
    pos = (onset[sid] <= h) & okk
    tl = pd.DataFrame({"stay_id": sid, "h": h, "post": 1, "pos": pos.astype(int), "ok": okk.astype(int), "neg": (okk & ~pos).astype(int)})
    kind = rng.choice(3, size=n, p=[0.5, 0.2, 0.3])
    pre = pd.DataFrame({"stay_id": np.arange(n)[kind < 2], "h": -3.0, "post": 0, "pos": 0, "ok": (kind[kind < 2] == 0).astype(int),
                        "neg": (kind[kind < 2] == 0).astype(int)})
    tl = pd.concat([tl, pre], ignore_index=True)
    first = tl[(tl["pos"] == 1)].groupby("stay_id")["h"].min()
    t_del = np.full(n, np.nan)
    t_del[first.index.values] = first.values
    ref = np.bincount(site).argmax()
    cols = [x, rass <= -2, rass <= -4, ~np.isfinite(rass), rng.random(n) < 0.3] + [(site == s).astype(float) for s in range(n_hosp) if s != ref]
    names = ["x", "rass_le_m2", "rass_le_m4", "rass_missing", "surgical"] + [f"hosp_{s}" for s in range(n_hosp) if s != ref]
    X, names, _ = asd.prevalence_filter(np.column_stack(cols).astype(float), names)
    d = {"X": X, "names": names, "dropped": [], "extras": {"age_years": 60 + 10 * x}, "A": A, "site": site.astype(float), "t_end": t_end,
         "end_death": rng.random(n) < 0.1, "t_dev": np.where(rng.random(n) < 0.1, rng.uniform(1, 100, n), np.nan), "t_del": t_del,
         "t_hypo": np.where(hypo <= t_end, hypo, np.nan), "hosp_death": np.zeros(n, int),
         "sub": {"rass3": np.isfinite(rass) & (rass >= -3), "egfr60": rng.random(n) < 0.3, "age65": x > 0},
         "sid": np.arange(n), "rgrp": r15.rass_group(rass), "h_icu": rng.exponential(8.0, n)}
    post = tl[tl["post"] == 1]
    d.update(r11.timeline_arrays(d, d["sid"], post))
    d.update(timeline_extra(d, d["sid"], tl))
    d["ckd"], d["rrt_pre"], d["dial_wide"] = rng.random(n) < 0.15, rng.random(n) < 0.04, rng.random(n) < 0.03
    d["rrt0"] = d["rrt_pre"] | d["dial_wide"]
    d["egfr"] = np.where(rng.random(n) < 0.1, np.nan, rng.uniform(5, 120, n))
    d["X_kadj"], d["names_kadj"], _ = add_cols(d, [d["ckd"], d["rrt0"]], ["ckd", "rrt0"])
    d["X_sev"], d["names_sev"], _ = add_cols(d, [rng.random(n) < 0.3], ["sevx"])
    for T in TPTS:
        d[f"init_{T}"], d[f"other_{T}"] = rng.random(n) < 0.8, rng.random(n) < 0.1
    d["t_other"], d["stop_unusable"] = d["t_dev"], rng.random(n) < 0.05
    if db == "e":
        d["dxgroup"] = rng.choice(np.array(["Other", "Sepsis", "CVA"]), size=n)
        d["X_dx"], d["names_dx"], _ = add_cols(d, [d["dxgroup"] == g for g in ("CVA", "Sepsis")], ["dx_CVA", "dx_Sepsis"])
        for c in SEV_E_RAW:
            d["raw_" + c] = np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n))
        d["vent_pre"] = rng.random(n) < 0.3
    else:
        d["site"] = None
        d["sofa_prev"], d["sofa_main"] = rng.integers(0, 12, n).astype(float), rng.integers(0, 12, n).astype(float)
        d["pdx"], d["adm"] = rng.choice(np.array(PDX), size=n), rng.choice(np.array(ADM), size=n)
    return d, tl


def selftest(args):
    passed, total = 0, 0

    def check(name, cond, note=""):
        nonlocal passed, total
        total += 1
        passed += bool(cond)
        print(f"{'PASS' if cond else 'FAIL'} | {name}" + (f" | {note}" if note else ""), flush=True)

    rng = np.random.default_rng(16)
    d = {"t_end": np.array([100.0, 100.0, 100.0, 5.0]), "t_del": np.array([np.nan, 20.0, np.nan, np.nan])}
    tl = pd.DataFrame({"stay_id": [0, 0, 0, 1, 1, 1, 2, 2], "h": [-10.0, -2.0, 6.0, -5.0, 12.0, 20.0, -1.0, 30.0], "post": [0, 0, 1, 0, 1, 1, 0, 1],
                       "pos": [0, 0, 0, 0, 0, 1, 0, 0], "ok": [0, 1, 1, 1, 0, 1, 0, 1], "neg": [0, 1, 1, 1, 0, 0, 0, 1]})
    te = timeline_extra(d, np.arange(4), tl)
    check("status before time zero: last charted time decides (negative, negative, unable to assess, none)",
          list(te["base"]) == ["neg", "neg", "uta", "none"] and not te["base_pos"].any(), " ".join(te["base"]))
    check("first assessable screen in follow-up: 6 h, 20 h (the positive), 30 h, none",
          np.allclose(te["t_first_ok"][:3], [6.0, 20.0, 30.0]) and np.isnan(te["t_first_ok"][3]), str(te["t_first_ok"]))
    s = scales(np.array([0.237]), np.array([0.277]))
    ev = evalue(0.237 / 0.277, 0.80, 0.92)
    check("scales and E-value: OR 0.811, RR 0.856, RD -0.040; E-value 1.61 (limit from 0.92: 1.39)",
          abs(math.exp(s["OR"][0]) - 0.8108) < 5e-4 and abs(math.exp(s["RR"][0]) - 0.8556) < 5e-4 and abs(s["RD"][0] + 0.04) < 1e-9
          and abs(ev["point"] - 1.611) < 5e-3 and abs(ev["limit"] - (1 / 0.92 + math.sqrt(1 / 0.92 * (1 / 0.92 - 1)))) < 1e-9
          and evalue(0.97, 0.9, 1.1)["limit"] == 1.0, f"{ev['point']:.3f} {ev['limit']:.3f}")
    check("OR 0.75 on the absolute scale: -5.4 points at 27.7%, -2.9 at 12.9%",
          abs(100 * mcid_abs(0.277) + 5.4) < 0.06 and abs(100 * mcid_abs(0.129) + 2.9) < 0.06, f"{100 * mcid_abs(0.277):.2f} {100 * mcid_abs(0.129):.2f}")
    d0, _ = synth(30000, rng)
    ow = sc.overlap_w(asd.ow_fit(d0, {}, "full"), d0["A"])
    a, b = cif_lm(d0, ow, 0), cif(d0, ow, d0["t_del"])
    check("landmark at 0 h equals the main estimator", abs(a["riskA"] - b["riskA"]) < 1e-12 and abs(a["riskB"] - b["riskB"]) < 1e-12)
    de, _ = synth(60000, rng, early=-1.2)
    full = est_c2(de, "e", {})
    check("an effect confined to the first 12 h: main OR below 1, landmark 12 h near 1",
          full["primary"]["OR"] < 0.92 and abs(math.log(full["lm12_all"]["OR"])) < 0.08,
          f"main {full['primary']['OR']:.3f}, lm6 {full['lm6_all']['OR']:.3f}, lm12 {full['lm12_all']['OR']:.3f}")
    res = est_c2(d0, "e", {})
    R, HR, SC = names_of("c2e")
    bad = [nm for nm in R if not (np.isfinite(res[nm]["OR"]) and abs(math.log(res[nm]["OR"])) < 0.35)]
    check("no drug effect: all risk-type estimates finite and near 1", not bad, " ".join(bad))
    check("no drug effect: interval-censored hazard ratios finite and near 1", all(np.isfinite(res[nm]["HR"]) and abs(math.log(res[nm]["HR"])) < 0.35 for nm in HR),
          " ".join(f"{res[nm]['HR']:.2f}" for nm in HR))
    ref15 = r15.unit_est(d0, "light", {})
    check("strata follow R15's recipe exactly", abs(res["light_primary"]["OR"] - ref15["primary"]["OR"]) < 1e-9
          and abs(res["light_negctl"]["OR"] - ref15["negctl"]["OR"]) < 1e-9)
    cache = {}
    est_c2(d0, "e", cache)
    v1, v2 = one_rep(d0, "c2e", "main", 3, cache), one_rep(d0, "c2e", "main", 3, cache)
    vp = one_rep(d0, "c2e", "pat", 3, cache)
    refv = flat(est_c2(asd.resample(d0, "e", np.random.default_rng(asd.SEED + 3)), "e", dict(cache)), "c2e")
    check("replicates reproducible and equal to asd.resample with the same seed; patient scheme differs",
          v1 is not None and len(v1) == len(boot_cols("c2e", "main")) and np.allclose(v1, v2, equal_nan=True) and np.allclose(v1, refv, equal_nan=True)
          and vp is not None and not np.allclose(vp, v1, equal_nan=True))
    vs = one_rep(d0, "c2e", "sep", 5, cache)
    dlog = abs(scales(vs[0:1], vs[1:2])["OR"][0] - scales(vs[2:3], vs[3:4])["OR"][0])
    check("separate vs shared hospital indicators: log OR differs by < 0.002", vs is not None and dlog < 2e-3, f"{dlog:.6f}")
    cv = curves(d0, ow)
    kend, typ, _ = sc.follow_types(d0["t_end"], d0["end_death"], d0["t_del"], disc_competing=True)
    okc = True
    for lab, r in arms(d0):
        free = float(ow[r & (typ == 0)].sum() / ow[r].sum())
        okc &= abs(cv[lab]["delirium"][168] + cv[lab]["discharge"][168] + cv[lab]["death"][168] + free - 1.0) < 5e-6
    check("competing-event curves: three cumulative incidences and the event-free share add to 1; delirium curve ends at the main risk",
          okc and abs(cv["A"]["delirium"][168] - b["riskA"]) < 1e-6 and abs(cv["B"]["delirium"][168] - b["riskB"]) < 1e-6)
    names = [f"{u}_primary" for u in STRATA]
    pts = [{"OR": 0.8, "RR": 0.85, "RD": -0.03}] * 4
    rr = np.random.default_rng(1)
    colsB = [f"{nm}:{a_}" for nm in names for a_ in ("A", "B")]

    def fake(shift):
        V = np.empty((1500, 8))
        for j in range(4):
            rb = np.clip(0.20 + rr.normal(0, 0.01, 1500), 0.01, 0.99)
            lor = math.log(0.8) + (shift if j == 3 else 0.0) + rr.normal(0, 0.08, 1500)
            oa = np.exp(lor) * rb / (1 - rb)
            V[:, 2 * j], V[:, 2 * j + 1] = oa / (1 + oa), rb
        return {"cols": colsB, "v": V}
    w0 = wald_strata(pts, fake(0.0), names)
    pts2 = pts[:3] + [{"OR": 0.8 * math.exp(0.5), "RR": 1.2, "RD": 0.04}]
    w1 = wald_strata(pts2, fake(0.5), names)
    check("Wald test across strata: no signal when equal, a signal when one stratum differs",
          w0["OR"]["p"] > 0.05 and w1["OR"]["p"] < 0.001 and read_wald(0.5) == "no evidence that the stratum estimates differ"
          and read_wald(0.02).startswith("nominally different") and read_wald(0.001).startswith("still different after correction for six stratifying dimensions"),
          f"p0 {w0['OR']['p']:.3f}, p1 {w1['OR']['p']:.2e}")
    main = {"OR": 0.81, "OR_lo": 0.74, "OR_hi": 0.88}
    check("consistency rule: inside the main interval with the same reading, or changed with the reason",
          consistent({"OR": 0.80, "OR_lo": 0.70, "OR_hi": 0.92}, main) == "estimate consistent with the main analysis"
          and consistent({"OR": 0.95, "OR_lo": 0.80, "OR_hi": 1.10}, main).startswith("estimate changed")
          and "reading changed from" in consistent({"OR": 0.86, "OR_lo": 0.72, "OR_hi": 1.03}, main)
          and read_ic({"HR": 0.8, "lo": 0.7, "hi": 0.95}).endswith("still shows an association") and read_ic({"HR": 0.9, "lo": 0.7, "hi": 1.05}).endswith("no longer clear"))
    tmp = tempfile.mkdtemp(prefix="r16_self_")
    try:
        P, Rr = os.path.join(tmp, "p"), os.path.join(tmp, "r")
        os.makedirs(P)
        os.makedirs(Rr)
        data = {}
        for key in asd.KEYS:
            dk, _ = synth(4000, rng, db=key[2])
            if key[1] == "1":
                dk["bleed_dx"], dk["ppi_infusion"] = rng.random(4000) < 0.05, rng.random(4000) < 0.03
                dk["bleed"] = dk["bleed_dx"] | dk["ppi_infusion"]
                for T in (24, 48, 72):
                    dk[f"act_{T}"], dk[f"daily_{T}"] = rng.random(4000) < 0.7, rng.choice([20.0, 40.0, np.nan], size=4000)
                if key[2] == "m":
                    dk["ckd_prior"], dk["dial_prior"] = dk["ckd"], dk["dial_wide"]
            elif key[2] == "m":
                dk["ckd_prior"], dk["dial_prior"] = dk["ckd"], dk["dial_wide"]
            data[key] = dk
            cache = {}
            res = est_key(dk, key, cache)
            nA = int(dk["A"].sum())
            flow = {"eligible": 4000, "A": nA, "B": 4000 - nA, "other": 0, "outside_overlap": 0, "analysis_A": nA, "analysis_B": 4000 - nA}
            desc = describe_c2(dk, key[2], {}) if key[1] == "2" else describe_c1(dk, key[2], {})
            json.dump({"flow": flow, "res": asd.to_json(res), "desc": asd.to_json(desc), "checks": ["PASS | synthetic"], "n": 4000,
                       "point": flat(res, key).tolist()}, open(os.path.join(P, f"r16_est_{key}.json"), "w"))
            for scheme in ("main",) + (("pat", "sep") if key == "c2e" else ()):
                cols = boot_cols(key, scheme)
                V = np.full((60, len(cols)), np.nan)
                for rep in range(60):
                    v = one_rep(dk, key, scheme, rep, cache)
                    if v is not None:
                        V[rep] = v
                np.savez(os.path.join(P, f"r16boot_{key}_{scheme}_0000_0060.npz"), reps=np.arange(60), v=V, cols=np.array(cols))
        global interval_checks
        saved = interval_checks
        interval_checks = lambda o, key, R: (["PASS | synthetic (no stored intervals)"], True)
        try:
            report(argparse.Namespace(private=P, results=Rr))
        finally:
            interval_checks = saved
        J = json.load(open(os.path.join(Rr, "r16.json"), encoding="utf-8"))
        txt = open(os.path.join(Rr, "r16.md"), encoding="utf-8").read()
        F = json.load(open(os.path.join(Rr, "fig_values_r16.json"), encoding="utf-8"))
        check("end to end: report, JSON and figure values written; every PPI-comparison estimate has intervals on three scales",
              all(J["db"][db]["est"][nm].get(k + "_lo") is not None for db in DBS for nm in names_of("c2" + db)[0] for k in ("OR", "RR", "RD"))
              and "## Bootstrap in eICU" in txt and len(F["curves"]["m"]["all"]["A"]["delirium"]) == 169 and J["section10"]["jackknife"]["G"] == 12)
        counts = []

        def walk(o, k=""):
            if isinstance(o, dict):
                for kk, v in o.items():
                    walk(v, kk)
            elif isinstance(o, list):
                for v in o:
                    walk(v, k)
            elif k in ("n", "nA", "nB", "eventsA", "eventsB", "n_inf_A", "n_inf_B", "n_detected") or k in ("eligible", "A", "B", "other", "analysis_A", "analysis_B"):
                if isinstance(o, (str, int)) and not isinstance(o, bool):
                    counts.append(o)
        walk(J["db"])
        walk(J["famdose"])
        badc = [c for c in counts if not (isinstance(c, str) and (c == "<10" or (c.isdigit() and int(c) % 5 == 0)))]
        check("end to end: every count in the JSON is masked", len(counts) > 200 and not badc, f"{len(counts)} counts, {len(badc)} unmasked: {badc[:5]}")
        check("end to end: interaction, pooled estimates, E-values and written readings present",
              all(J["interaction"][db]["wald"]["OR"] for db in DBS) and J["pooled"]["light_primary"]["RR"] and J["evalue"]["primary_m"]["point"] >= 1
              and all(nm in J["reading"][db]["consistency"] for db in DBS for nm in ("bneg_primary", "h24_primary", "lm12_all", "sev_primary", "kexc_primary"))
              and J["reading"]["scale_dependence"].startswith("the reading of the difference between the databases"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{passed}/{total} checks passed", flush=True)
    if passed != total:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("extract")
    s.add_argument("--private", required=True)
    s.add_argument("--mimic", required=True)
    s.add_argument("--eicu", required=True)
    s.add_argument("--results", default="../results")
    for name in ("estimate", "report"):
        s = sub.add_parser(name)
        s.add_argument("--private", required=True)
        s.add_argument("--results", default="../results")
    s = sub.add_parser("boot")
    s.add_argument("--key", choices=asd.KEYS, required=True)
    s.add_argument("--scheme", choices=["main", "pat", "sep"], default="main")
    s.add_argument("--start", type=int, required=True)
    s.add_argument("--stop", type=int, required=True)
    s.add_argument("--private", required=True)
    s.add_argument("--results", default="../results")
    args = ap.parse_args()
    {"selftest": selftest, "extract": extract, "estimate": estimate, "boot": boot, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
