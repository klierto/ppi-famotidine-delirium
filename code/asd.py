#!/usr/bin/env python3
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime
import glob
import json
import math
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stats_core as sc

SQL_DIR = os.path.join(HERE, "sql")
TOOLS = os.path.normpath(os.path.join(HERE, "tools", "helpers"))
SEED = 20261013
NBOOT = 2000
MIN_BOOT = 50
MCID = 0.75
PRIOR = {1: (0.75, 0.63, 0.89), 2: None}
STEM = {1: "famdose", 2: "ppi"}
TITLE = {1: "Dose comparison", 2: "PPI comparison"}
NAME = {1: "dose comparison", 2: "PPI comparison"}
SHORT = {1: "dose", 2: "PPI"}


def _hook(t, e, tb):
    msg = str(e).splitlines()[0] if str(e) else ""
    print(f"ERROR {t.__name__}: {msg[:200]}", file=sys.stderr)


sys.excepthook = _hook

AS_RX = ("famotidine|pepcid|pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid|ranitidine|"
         "zantac|cimetidine|nizatidine|rabeprazole|dexlansoprazole")
AS_RX_E = ("famotidine|pepcid|pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid|ranitidine|"
           "zantac|cimetidine|rabeprazole")
HICL_AS = "4521, 35085, 35107, 4851, 22008, 4673, 25968"
M_INPUT = [225907, 225910, 227694, 229615, 225911, 222190,
           222168, 221668, 221385, 225150, 229420, 221712,
           221744, 225942, 221833, 225154]
M_CHART = [228332, 228096, 225664, 220621, 226537]
M_LAB = [50912, 50931, 50809, 51237, 51265]
M_SLIMMED = {"emar", "emar_detail", "pharmacy", "prescriptions", "inputevents", "poe", "poe_detail", "labevents", "chartevents"}
E_SLIMMED = {"medication", "lab", "nursecharting"}
GATE = {
    "c1m": {"eligible": "1580", "A": "815", "B": "675", "other": "90"},
    "c1e": {"eligible": "975", "A": "325", "B": "375", "other": "275"},
    "c2m": {"eligible": "11885", "A": "4515", "B": "7370", "other": "0"},
    "c2e": {"eligible": "8035", "A": "4605", "B": "3430", "other": "0"},
}
KEYS = ["c1m", "c1e", "c2m", "c2e"]
DBNAME = {"m": "MIMIC-IV", "e": "eICU"}


def status(msg, t0=None):
    extra = f" ({time.time() - t0:.0f} s)" if t0 is not None else ""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}{extra}", flush=True)


def log_run(results, line):
    if not results:
        return
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "run_log.md"), "a", encoding="utf-8") as f:
        f.write(f"- {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} {line}\n")


def duck(tmp):
    import duckdb
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '2GB'; SET threads = 2; SET temp_directory = '{tmp}'; SET preserve_insertion_order = false")
    return con


def connect_slim(P, db):
    sys.path.insert(0, TOOLS)
    from mimic_local import connect
    return connect(os.path.join(P, "slim_m" if db == "m" else "slim_e"), temp_dir=os.path.join(P, "duck_tmp"))


def read_pq(path):
    import duckdb
    con = duckdb.connect()
    df = con.execute("SELECT * FROM read_parquet(?)", [path]).df()
    con.close()
    return df


def write_pq(con, sql, path):
    tmp = path + ".tmp"
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")
    os.replace(tmp, path)


def gate_body(comparison, db):
    f = os.path.join(SQL_DIR, f"{STEM[comparison]}_{'mimiciv' if db == 'm' else 'eicu'}.sql")
    body = open(f, encoding="utf-8").read()
    body = re.sub(r"^\s*--.*$", "", body, flags=re.M).strip().rstrip(";")
    return re.sub(r"^\s*WITH\s+", "", body, flags=re.I)


def plist(paths):
    return "[" + ", ".join("'" + p + "'" for p in paths) + "]"


def slim_specs(db, step, ref):
    if db == "mimic":
        subj = f"(SELECT DISTINCT subject_id FROM {ref('icu', 'icustays')})"
        specs = {
            "small": [("hosp", "emar", f"subject_id IN {subj} AND regexp_matches(lower(medication), '{AS_RX}')", False),
                      ("hosp", "prescriptions", f"subject_id IN {subj} AND regexp_matches(lower(drug), '{AS_RX}')", False),
                      ("icu", "inputevents", f"itemid IN ({', '.join(map(str, M_INPUT))})", False)],
            "chart": [("icu", "chartevents", f"itemid IN ({', '.join(map(str, M_CHART))})", True)],
            "lab": [("hosp", "labevents", f"itemid IN ({', '.join(map(str, M_LAB))}) AND subject_id IN {subj}", True)],
            "note": [("note", "discharge", f"hadm_id IN (SELECT hadm_id FROM {ref('icu', 'icustays')}) AND regexp_matches(lower(text), '{AS_RX}')", False)],
        }
    else:
        specs = {
            "med": [("eicu", "medication", f"drughiclseqno IN ({HICL_AS}) OR regexp_matches(lower(coalesce(drugname, '')), '{AS_RX_E}')", False)],
            "lab": [("eicu", "lab", "labname IN ('creatinine', 'glucose', 'bedside glucose', 'PT - INR', 'platelets x 1000')", False)],
            "nc": [("eicu", "nursecharting", "nursingchartcelltypevallabel IN ('Delirium Scale/Score', 'Sedation Scale/Score/Goal')", True)],
        }
    return specs if step is None else specs.get(step)


def slim(args):
    t0 = time.time()
    P, src = os.path.expanduser(args.private), os.path.expanduser(args.src)
    dst = os.path.join(P, "slim_m" if args.db == "mimic" else "slim_e")
    con = duck(os.path.join(P, "duck_tmp"))

    def pq(m, tb):
        f = os.path.join(src, m, tb + ".parquet")
        return f"read_parquet('{f}')" if os.path.exists(f) else f"read_parquet('{os.path.join(src, m, tb)}/*.parquet')"

    if args.step == "link":
        mods, skip = (("hosp", "icu", "ed", "derived"), M_SLIMMED) if args.db == "mimic" else (("eicu", "eicu_derived"), E_SLIMMED)
        for m in mods:
            os.makedirs(os.path.join(dst, m), exist_ok=True)
            for f in os.listdir(os.path.join(src, m)):
                name = f[:-8] if f.endswith(".parquet") else f
                if name in skip or not (f.endswith(".parquet") or os.path.isdir(os.path.join(src, m, f))):
                    continue
                d = os.path.join(dst, m, f)
                if not os.path.lexists(d):
                    os.symlink(os.path.join(src, m, f), d)
        if args.db == "mimic":
            os.makedirs(os.path.join(dst, "note"), exist_ok=True)
        status("slim links made", t0)
    else:
        specs = slim_specs(args.db, args.step, pq)
        if specs is None:
            raise SystemExit("unknown step")
        for m, tb, where, folder in specs:
            out = os.path.join(dst, m, tb)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if folder:
                os.makedirs(out, exist_ok=True)
                out = os.path.join(out, "part-0.parquet")
            else:
                out += ".parquet"
            write_pq(con, f"SELECT * FROM {pq(m, tb)} WHERE {where}", out)
            if folder:
                open(os.path.join(dst, m, tb, "_done.txt"), "w").write("asd slim\n")
            status(f"slim {m}.{tb} written", t0)
    log_run(args.results, f"slim --db {args.db} --step {args.step}: ok ({time.time() - t0:.0f} s)")


COHORT_SELECT = {
    ("m", 1): """SELECT c.subject_id, c.hadm_id, c.stay_id, c.intime, c.outtime, c.gender, c.anchor_year_group AS era,
       CAST(NULL AS BIGINT) AS site, c.age, c.t0, c.scr, c.egfr, o.daily, x.arm
FROM elig l JOIN cohort c USING (subject_id) JOIN ord o USING (subject_id) JOIN expo x USING (subject_id)""",
    ("m", 2): """SELECT c.subject_id, c.hadm_id, c.stay_id, c.intime, c.outtime, c.gender, c.anchor_year_group AS era,
       CAST(NULL AS BIGINT) AS site, c.age, c.t0, c.scr, c.egfr, CAST(NULL AS DOUBLE) AS daily, x.arm
FROM elig l JOIN cohort c USING (subject_id) JOIN expo x USING (subject_id)""",
    ("e", 1): """SELECT c.patientunitstayid AS subject_id, c.patientunitstayid AS stay_id, c.gender, CAST(c.hospitaldischargeyear AS VARCHAR) AS era,
       c.hospitalid AS site, c.age, c.t0, c.scr, c.egfr, c.daily, x.arm
FROM elig l JOIN cohort c ON c.patientunitstayid = l.subject_id JOIN expo x ON x.subject_id = l.subject_id""",
}
COHORT_SELECT[("e", 2)] = COHORT_SELECT[("e", 1)]

ASADM_M = f"""asadm AS (
  SELECT e.hadm_id, e.charttime AS t,
         CASE WHEN regexp_matches(lower(e.medication), 'famotidine|pepcid') THEN 'FAM'
              WHEN regexp_matches(lower(e.medication), 'ranitidine|zantac|cimetidine|nizatidine') THEN 'H2O' ELSE 'PPI' END AS drug
  FROM hosp.emar e
  WHERE regexp_matches(lower(e.medication), '{AS_RX}')
    AND regexp_matches(lower(coalesce(e.event_txt, '')), 'administ|started')
    AND NOT regexp_matches(lower(coalesce(e.event_txt, '')), 'not |hold|stopp')
  UNION ALL
  SELECT s.hadm_id, x.starttime,
         CASE WHEN x.itemid = 225907 THEN 'FAM' WHEN x.itemid IN (225911, 222190) THEN 'H2O' ELSE 'PPI' END
  FROM icu.inputevents x JOIN icu.icustays s USING (stay_id)
  WHERE x.itemid IN (225907, 225910, 227694, 229615, 225911, 222190)
    AND coalesce(x.statusdescription, '') <> 'Rewritten' AND coalesce(x.amount, 0) > 0
)"""

MED_E = f"""med0 AS (
  SELECT m.patientunitstayid, m.drugstartoffset, m.drugstopoffset, lower(coalesce(m.dosage, '')) AS dosage, lower(coalesce(m.frequency, '')) AS freq,
         CASE WHEN m.drughiclseqno IN (4521, 35085, 35107, 4851) OR regexp_matches(lower(coalesce(m.drugname, '')), 'famotidine|pepcid') THEN 'FAM'
              WHEN regexp_matches(lower(coalesce(m.drugname, '')), 'ranitidine|zantac|cimetidine') THEN 'H2O' ELSE 'PPI' END AS drug
  FROM eicu.medication m
  WHERE coalesce(m.drugordercancelled, 'No') <> 'Yes'
    AND (m.drughiclseqno IN ({HICL_AS}) OR regexp_matches(lower(coalesce(m.drugname, '')), '{AS_RX_E}'))
),
med AS (
  SELECT *, CASE
           WHEN TRY_CAST(regexp_extract(dosage, '([0-9.]+)', 1) AS DOUBLE) IS NULL THEN NULL
           WHEN regexp_matches(freq, 'q12|bid|twice|every 12|2 times') THEN TRY_CAST(regexp_extract(dosage, '([0-9.]+)', 1) AS DOUBLE) * 2
           WHEN regexp_matches(freq, 'q24|daily|qd|once|every 24|qhs|bedtime|nightly|1 time|1xdaily') THEN TRY_CAST(regexp_extract(dosage, '([0-9.]+)', 1) AS DOUBLE) END AS daily
  FROM med0
)"""

M_COVARS = f"""
WITH base AS (SELECT DISTINCT subject_id, hadm_id, stay_id, intime, t0 FROM read_parquet({{files}})),
sofa AS (SELECT b.stay_id, max(s.sofa_24hours) AS sofa FROM base b JOIN derived.sofa s
         ON s.stay_id = b.stay_id AND b.t0 >= s.starttime AND b.t0 < s.endtime GROUP BY 1),
rt AS (SELECT b.stay_id, c.charttime, c.valuenum FROM base b JOIN icu.chartevents c ON c.stay_id = b.stay_id AND c.itemid = 228096
       AND c.charttime BETWEEN b.t0 - INTERVAL 12 HOUR AND b.t0 WHERE c.valuenum BETWEEN -5 AND 4),
rass AS (SELECT stay_id, min(valuenum) AS rass FROM (SELECT stay_id, charttime, valuenum, max(charttime) OVER (PARTITION BY stay_id) AS tm FROM rt)
         WHERE charttime = tm GROUP BY 1),
vent AS (SELECT DISTINCT b.stay_id FROM base b JOIN derived.ventilation v ON v.stay_id = b.stay_id AND b.t0 >= v.starttime AND b.t0 < v.endtime
         WHERE v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')),
vaso AS (SELECT DISTINCT b.stay_id FROM base b JOIN derived.vasoactive_agent v ON v.stay_id = b.stay_id AND b.t0 >= v.starttime AND b.t0 < v.endtime
         WHERE coalesce(v.norepinephrine, 0) > 0 OR coalesce(v.epinephrine, 0) > 0 OR coalesce(v.phenylephrine, 0) > 0
            OR coalesce(v.vasopressin, 0) > 0 OR coalesce(v.dopamine, 0) > 0),
inf AS (SELECT b.stay_id,
               max(CASE WHEN x.itemid IN (222168, 221668, 221385, 225150, 229420, 221712) THEN 1 ELSE 0 END) AS sed_inf,
               max(CASE WHEN x.itemid IN (221744, 225942, 221833, 225154) THEN 1 ELSE 0 END) AS opi_inf
        FROM base b JOIN icu.inputevents x ON x.stay_id = b.stay_id AND x.starttime <= b.t0 AND x.endtime > b.t0
        WHERE x.itemid IN (222168, 221668, 221385, 225150, 229420, 221712, 221744, 225942, 221833, 225154)
          AND coalesce(x.rate, 0) > 0 AND coalesce(x.statusdescription, '') <> 'Rewritten' GROUP BY 1),
lab AS (SELECT b.stay_id, max(CASE WHEN l.itemid = 51237 THEN l.valuenum END) AS inr_max,
               min(CASE WHEN l.itemid = 51265 THEN l.valuenum END) AS plt_min
        FROM base b JOIN hosp.labevents l ON l.subject_id = b.subject_id AND l.charttime BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0
        WHERE l.itemid IN (51237, 51265) AND l.valuenum IS NOT NULL GROUP BY 1),
unit AS (SELECT stay_id, first_careunit FROM icu.icustays),
svc AS (SELECT stay_id, curr_service FROM (
          SELECT b.stay_id, s.curr_service, row_number() OVER (PARTITION BY b.stay_id ORDER BY s.transfertime DESC, s.curr_service) AS rn
          FROM base b JOIN hosp.services s ON s.hadm_id = b.hadm_id AND s.transfertime <= b.t0) WHERE rn = 1),
t0adm AS (
  SELECT b.stay_id, 'IV' AS route FROM base b JOIN icu.inputevents x ON x.stay_id = b.stay_id AND x.starttime = b.t0
  WHERE x.itemid IN (225907, 225910, 227694, 229615, 225911, 222190) AND coalesce(x.statusdescription, '') <> 'Rewritten' AND coalesce(x.amount, 0) > 0
  UNION ALL
  SELECT b.stay_id, CASE WHEN regexp_matches(upper(coalesce(r.route, '')), '^IV') THEN 'IV'
                         WHEN regexp_matches(upper(coalesce(r.route, '')), 'PO|NG|OG|ORAL|TUBE|ENTERAL') THEN 'EN' ELSE 'UNK' END
  FROM base b JOIN hosp.emar e ON e.hadm_id = b.hadm_id AND e.charttime = b.t0
  LEFT JOIN hosp.prescriptions r ON r.pharmacy_id = e.pharmacy_id
  WHERE regexp_matches(lower(e.medication), '{AS_RX}') AND regexp_matches(lower(coalesce(e.event_txt, '')), 'administ|started')
    AND NOT regexp_matches(lower(coalesce(e.event_txt, '')), 'not |hold|stopp')),
route AS (SELECT stay_id, CASE WHEN bool_or(route = 'IV') THEN 'IV' WHEN bool_or(route = 'EN') THEN 'EN' ELSE 'UNK' END AS route
          FROM t0adm GROUP BY 1),
prior AS (
  SELECT b.stay_id, d.icd_code, d.icd_version FROM base b JOIN hosp.admissions a0 ON a0.hadm_id = b.hadm_id
  JOIN hosp.admissions a ON a.subject_id = b.subject_id AND a.admittime < a0.admittime
  JOIN hosp.diagnoses_icd d ON d.hadm_id = a.hadm_id
  UNION ALL
  SELECT b.stay_id, d.icd_code, d.icd_version FROM base b JOIN ed.edstays e ON e.hadm_id = b.hadm_id JOIN ed.diagnosis d ON d.stay_id = e.stay_id),
como AS (SELECT stay_id,
  max(CASE WHEN (icd_version = 10 AND regexp_matches(icd_code, '^(F01|F02|F03|G30|G310|G311|G3183)'))
             OR (icd_version = 9 AND regexp_matches(icd_code, '^(290|2941|2942|3310|3311|3312|33182)')) THEN 1 ELSE 0 END) AS dementia,
  max(CASE WHEN (icd_version = 10 AND regexp_matches(icd_code, '^F10'))
             OR (icd_version = 9 AND regexp_matches(icd_code, '^(291|303|3050)')) THEN 1 ELSE 0 END) AS alcohol,
  max(CASE WHEN (icd_version = 10 AND regexp_matches(icd_code, '^(K70|K73|K74|K766|I85)'))
             OR (icd_version = 9 AND regexp_matches(icd_code, '^(571|5723|4560|4561|4562)')) THEN 1 ELSE 0 END) AS liver
  FROM prior GROUP BY 1)
SELECT b.stay_id, s.sofa, r.rass, (v.stay_id IS NOT NULL) AS vent, (va.stay_id IS NOT NULL) AS vaso,
       coalesce(i.sed_inf, 0) AS sed_inf, coalesce(i.opi_inf, 0) AS opi_inf, lb.inr_max, lb.plt_min,
       u.first_careunit AS unit, sv.curr_service AS service, coalesce(ro.route, 'UNK') AS route,
       date_diff('minute', b.intime, b.t0) / 60.0 AS h_icu,
       coalesce(cm.dementia, 0) AS dementia, coalesce(cm.alcohol, 0) AS alcohol, coalesce(cm.liver, 0) AS liver
FROM base b LEFT JOIN sofa s USING (stay_id) LEFT JOIN rass r USING (stay_id) LEFT JOIN vent v USING (stay_id)
LEFT JOIN vaso va USING (stay_id) LEFT JOIN inf i USING (stay_id) LEFT JOIN lab lb USING (stay_id) LEFT JOIN unit u USING (stay_id)
LEFT JOIN svc sv USING (stay_id) LEFT JOIN route ro USING (stay_id) LEFT JOIN como cm USING (stay_id)
"""

E_COVARS = f"""
WITH base AS (SELECT DISTINCT stay_id, t0 FROM read_parquet({{files}})),
pt AS (SELECT patientunitstayid AS stay_id, unittype, unitadmitsource, lower(coalesce(apacheadmissiondx, '')) AS adx FROM eicu.patient),
ap AS (SELECT patientunitstayid AS stay_id, max(apachescore) AS apache FROM eicu.apachepatientresult
       WHERE apacheversion = 'IV' AND apachescore > 0 GROUP BY 1),
av AS (SELECT patientunitstayid AS stay_id, max(CASE WHEN intubated = 1 OR vent = 1 THEN 1 ELSE 0 END) AS vent1 FROM eicu.apacheapsvar GROUP BY 1),
rt AS (SELECT b.stay_id, n.nursingchartoffset AS t, TRY_CAST(n.nursingchartvalue AS INTEGER) AS v FROM base b JOIN eicu.nursecharting n
       ON n.patientunitstayid = b.stay_id AND n.nursingchartoffset BETWEEN b.t0 - 720 AND b.t0
       WHERE n.nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal' AND n.nursingchartcelltypevalname = 'Sedation Score'
         AND TRY_CAST(n.nursingchartvalue AS INTEGER) BETWEEN -5 AND 4),
rass AS (SELECT stay_id, min(v) AS rass FROM (SELECT stay_id, t, v, max(t) OVER (PARTITION BY stay_id) AS tm FROM rt) WHERE t = tm GROUP BY 1),
inf AS (SELECT b.stay_id,
          max(CASE WHEN regexp_matches(lower(i.drugname), 'propofol|diprivan|midazolam|versed|lorazepam|ativan|dexmedetomidine|precedex|ketamine') THEN 1 ELSE 0 END) AS sed_inf,
          max(CASE WHEN regexp_matches(lower(i.drugname), 'fentanyl|hydromorphone|dilaudid|morphine|remifentanil') THEN 1 ELSE 0 END) AS opi_inf,
          max(CASE WHEN regexp_matches(lower(i.drugname), 'norepinephrine|levophed|epinephrine|phenylephrine|neo-synephrine|neosynephrine|vasopressin|dopamine') THEN 1 ELSE 0 END) AS vaso
        FROM base b JOIN eicu.infusiondrug i ON i.patientunitstayid = b.stay_id AND i.infusionoffset BETWEEN b.t0 - 120 AND b.t0
        WHERE TRY_CAST(i.drugrate AS DOUBLE) > 0 GROUP BY 1),
lab AS (SELECT b.stay_id, max(CASE WHEN l.labname = 'PT - INR' THEN l.labresult END) AS inr_max,
               min(CASE WHEN l.labname = 'platelets x 1000' THEN l.labresult END) AS plt_min
        FROM base b JOIN eicu.lab l ON l.patientunitstayid = b.stay_id AND l.labresultoffset BETWEEN b.t0 - 1440 AND b.t0
        WHERE l.labname IN ('PT - INR', 'platelets x 1000') AND l.labresult IS NOT NULL GROUP BY 1),
t0o AS (SELECT b.stay_id, lower(coalesce(m.routeadmin, '')) AS r FROM base b JOIN eicu.medication m
        ON m.patientunitstayid = b.stay_id AND m.drugstartoffset = b.t0
        WHERE coalesce(m.drugordercancelled, 'No') <> 'Yes'
          AND (m.drughiclseqno IN ({HICL_AS}) OR regexp_matches(lower(coalesce(m.drugname, '')), '{AS_RX_E}'))),
route AS (SELECT stay_id, CASE WHEN bool_or(regexp_matches(r, '^iv|intraven')) THEN 'IV'
                               WHEN bool_or(regexp_matches(r, 'po|oral|ng|tube|peg|og|enteral')) THEN 'EN' ELSE 'UNK' END AS route
          FROM t0o GROUP BY 1),
ph AS (SELECT patientunitstayid AS stay_id, max(CASE WHEN pasthistorypath LIKE '%/Dementia/%' THEN 1 ELSE 0 END) AS dem_ph,
              max(CASE WHEN pasthistorypath LIKE '%/Cirrhosis/%' THEN 1 ELSE 0 END) AS liv_ph FROM eicu.pasthistory GROUP BY 1),
dx AS (SELECT b.stay_id, max(CASE WHEN regexp_matches(lower(d.diagnosisstring), 'dementia') THEN 1 ELSE 0 END) AS dem_dx,
              max(CASE WHEN regexp_matches(lower(d.diagnosisstring), 'alcohol|etoh') THEN 1 ELSE 0 END) AS alc_dx,
              max(CASE WHEN regexp_matches(lower(d.diagnosisstring), 'cirrhosis|chronic liver|end stage liver|end-stage liver|portal hypertension')
                       THEN 1 ELSE 0 END) AS liv_dx
       FROM base b JOIN eicu.diagnosis d ON d.patientunitstayid = b.stay_id AND d.diagnosisoffset <= b.t0 GROUP BY 1)
SELECT b.stay_id, a.apache, v.vent1 AS vent, r.rass, coalesce(i.sed_inf, 0) AS sed_inf, coalesce(i.opi_inf, 0) AS opi_inf,
       coalesce(i.vaso, 0) AS vaso, lb.inr_max, lb.plt_min, p.unittype AS unit, p.unitadmitsource AS service,
       coalesce(ro.route, 'UNK') AS route, b.t0 / 60.0 AS h_icu,
       greatest(coalesce(ph.dem_ph, 0), coalesce(dx.dem_dx, 0)) AS dementia,
       greatest(coalesce(dx.alc_dx, 0), CASE WHEN regexp_matches(coalesce(p.adx, ''), 'alcohol|etoh') THEN 1 ELSE 0 END) AS alcohol,
       greatest(coalesce(ph.liv_ph, 0), coalesce(dx.liv_dx, 0)) AS liver
FROM base b LEFT JOIN pt p USING (stay_id) LEFT JOIN ap a USING (stay_id) LEFT JOIN av v USING (stay_id) LEFT JOIN rass r USING (stay_id)
LEFT JOIN inf i USING (stay_id) LEFT JOIN lab lb USING (stay_id) LEFT JOIN route ro USING (stay_id) LEFT JOIN ph USING (stay_id)
LEFT JOIN dx USING (stay_id)
"""

M_FOLLOW = """
WITH base AS (SELECT DISTINCT hadm_id, stay_id, outtime, t0 FROM read_parquet({files})),
adm AS (SELECT hadm_id, deathtime FROM hosp.admissions),
x AS (SELECT b.stay_id, b.t0, b.outtime, (a.deathtime IS NOT NULL AND a.deathtime <= b.outtime + INTERVAL 24 HOUR) AS dth, a.deathtime
      FROM base b LEFT JOIN adm a USING (hadm_id))
SELECT stay_id, greatest(date_diff('minute', t0, CASE WHEN coalesce(dth, false) THEN least(outtime, deathtime) ELSE outtime END), 0) / 60.0 AS t_end,
       coalesce(dth, false) AS end_death
FROM x
"""

M_DEV = {
    1: f"""
WITH c AS (SELECT hadm_id, stay_id, t0, arm FROM read_parquet('{{f}}') WHERE arm IS NOT NULL),
{ASADM_M},
rx AS (SELECT c.stay_id, c.arm, r.starttime,
              TRY_CAST(regexp_extract(r.dose_val_rx, '^[0-9.]+') AS DOUBLE) * r.doses_per_24_hrs AS daily
       FROM c JOIN hosp.prescriptions r ON r.hadm_id = c.hadm_id AND r.starttime > c.t0 + INTERVAL 1 HOUR
        AND r.starttime <= c.t0 + INTERVAL 168 HOUR
       WHERE regexp_matches(lower(r.drug), 'famotidine|pepcid')),
d1 AS (SELECT stay_id, min(starttime) AS t FROM rx WHERE (arm = 'A' AND daily > 20) OR (arm = 'B' AND daily < 40) GROUP BY 1),
d2 AS (SELECT c.stay_id, min(a.t) AS t FROM c JOIN asadm a ON a.hadm_id = c.hadm_id AND a.t > c.t0 AND a.t <= c.t0 + INTERVAL 168 HOUR
       WHERE a.drug <> 'FAM' GROUP BY 1)
SELECT c.stay_id, date_diff('minute', c.t0, least(coalesce(d1.t, d2.t), coalesce(d2.t, d1.t))) / 60.0 AS t_dev
FROM c LEFT JOIN d1 USING (stay_id) LEFT JOIN d2 USING (stay_id)
""",
    2: f"""
WITH c AS (SELECT hadm_id, stay_id, t0, arm FROM read_parquet('{{f}}') WHERE arm IS NOT NULL),
{ASADM_M},
d AS (SELECT c.stay_id, min(a.t) AS t FROM c JOIN asadm a ON a.hadm_id = c.hadm_id AND a.t > c.t0 AND a.t <= c.t0 + INTERVAL 168 HOUR
      WHERE (c.arm = 'A' AND a.drug IN ('FAM', 'H2O')) OR (c.arm = 'B' AND a.drug = 'PPI') GROUP BY 1)
SELECT c.stay_id, date_diff('minute', c.t0, d.t) / 60.0 AS t_dev FROM c LEFT JOIN d USING (stay_id)
""",
}

E_FOLLOW = """
WITH base AS (SELECT DISTINCT stay_id, t0 FROM read_parquet({files}))
SELECT b.stay_id, greatest(p.unitdischargeoffset - b.t0, 0) / 60.0 AS t_end,
       coalesce(p.unitdischargestatus = 'Expired'
                OR (p.hospitaldischargestatus = 'Expired' AND p.hospitaldischargeoffset <= p.unitdischargeoffset + 1440), false) AS end_death
FROM base b JOIN eicu.patient p ON p.patientunitstayid = b.stay_id
"""

E_DEV = {
    1: f"""
WITH c AS (SELECT stay_id, t0, arm FROM read_parquet('{{f}}') WHERE arm IS NOT NULL),
{MED_E},
d AS (SELECT c.stay_id, min(m.drugstartoffset) AS t FROM c JOIN med m ON m.patientunitstayid = c.stay_id
        AND m.drugstartoffset > c.t0 AND m.drugstartoffset <= c.t0 + 10080
      WHERE m.drug <> 'FAM' OR (c.arm = 'A' AND m.daily > 20) OR (c.arm = 'B' AND m.daily < 40) GROUP BY 1)
SELECT c.stay_id, (d.t - c.t0) / 60.0 AS t_dev FROM c LEFT JOIN d USING (stay_id)
""",
    2: f"""
WITH c AS (SELECT stay_id, t0, arm FROM read_parquet('{{f}}') WHERE arm IS NOT NULL),
{MED_E},
d AS (SELECT c.stay_id, min(m.drugstartoffset) AS t FROM c JOIN med m ON m.patientunitstayid = c.stay_id
        AND m.drugstartoffset > c.t0 AND m.drugstartoffset <= c.t0 + 10080
      WHERE (c.arm = 'A' AND m.drug IN ('FAM', 'H2O')) OR (c.arm = 'B' AND m.drug = 'PPI') GROUP BY 1)
SELECT c.stay_id, (d.t - c.t0) / 60.0 AS t_dev FROM c LEFT JOIN d USING (stay_id)
""",
}

M_OUTCOME = """
WITH base AS (SELECT DISTINCT subject_id, hadm_id, stay_id, outtime, t0 FROM read_parquet({files})),
fu AS (SELECT *, least(outtime, t0 + INTERVAL 168 HOUR) AS t_end FROM base),
cam AS (SELECT stay_id, charttime FROM icu.chartevents WHERE itemid = 228332 AND value = 'Positive'),
glu AS (SELECT subject_id, charttime FROM hosp.labevents WHERE itemid IN (50931, 50809) AND valuenum > 0 AND valuenum < 70
        UNION ALL
        SELECT subject_id, charttime FROM icu.chartevents WHERE itemid IN (225664, 220621, 226537) AND valuenum > 0 AND valuenum < 70),
dl AS (SELECT f.stay_id, date_diff('minute', f.t0, min(c.charttime)) / 60.0 AS t_del FROM fu f JOIN cam c
       ON c.stay_id = f.stay_id AND c.charttime > f.t0 AND c.charttime <= f.t_end GROUP BY f.stay_id, f.t0),
hy AS (SELECT f.stay_id, date_diff('minute', f.t0, min(g.charttime)) / 60.0 AS t_hypo FROM fu f JOIN glu g
       ON g.subject_id = f.subject_id AND g.charttime > f.t0 AND g.charttime <= f.t_end GROUP BY f.stay_id, f.t0),
hd AS (SELECT b.stay_id, (a.deathtime IS NOT NULL OR coalesce(a.hospital_expire_flag, 0) = 1) AS hosp_death
       FROM base b JOIN hosp.admissions a USING (hadm_id))
SELECT b.stay_id, dl.t_del, hy.t_hypo, coalesce(hd.hosp_death, false) AS hosp_death
FROM base b LEFT JOIN dl USING (stay_id) LEFT JOIN hy USING (stay_id) LEFT JOIN hd USING (stay_id)
"""

E_OUTCOME = """
WITH base AS (SELECT DISTINCT stay_id, t0 FROM read_parquet({files})),
fu AS (SELECT b.stay_id, b.t0, least(p.unitdischargeoffset, b.t0 + 10080) AS t_end, p.hospitaldischargestatus
       FROM base b JOIN eicu.patient p ON p.patientunitstayid = b.stay_id),
dsc AS (SELECT patientunitstayid, nursingchartoffset AS t, (upper(nursingchartvalue) = 'YES' OR TRY_CAST(nursingchartvalue AS INTEGER) >= 4) AS pos
        FROM eicu.nursecharting WHERE nursingchartcelltypevallabel = 'Delirium Scale/Score' AND nursingchartcelltypevalname = 'Delirium Score'
          AND NULLIF(nursingchartvalue, '') IS NOT NULL),
glu AS (SELECT patientunitstayid, labresultoffset AS t FROM eicu.lab WHERE labname IN ('glucose', 'bedside glucose') AND labresult > 0 AND labresult < 70),
dl AS (SELECT f.stay_id, (min(d.t) - f.t0) / 60.0 AS t_del FROM fu f JOIN dsc d
       ON d.patientunitstayid = f.stay_id AND d.pos AND d.t > f.t0 AND d.t <= f.t_end GROUP BY f.stay_id, f.t0),
hy AS (SELECT f.stay_id, (min(g.t) - f.t0) / 60.0 AS t_hypo FROM fu f JOIN glu g
       ON g.patientunitstayid = f.stay_id AND g.t > f.t0 AND g.t <= f.t_end GROUP BY f.stay_id, f.t0)
SELECT f.stay_id, dl.t_del, hy.t_hypo, coalesce(f.hospitaldischargestatus = 'Expired', false) AS hosp_death
FROM fu f LEFT JOIN dl USING (stay_id) LEFT JOIN hy USING (stay_id)
"""


def cohort_sql(comparison, db):
    return "WITH " + gate_body(comparison, db) + ",\nq_out AS (\n" + COHORT_SELECT[(db, comparison)] + "\n)\nSELECT * FROM q_out"


def all_sql(P, db):
    d = "m" if db == "mimic" else "e"
    pf = lambda name: os.path.join(P, f"{d}_{name}.parquet")
    files = plist([pf("c1_cohort"), pf("c2_cohort")])
    out = {"cohort": [(cohort_sql(c, d), pf(f"c{c}_cohort")) for c in (1, 2)]}
    if d == "m":
        out["covars"] = [(M_COVARS.format(files=files), pf("covars"))]
        out["follow"] = [(M_FOLLOW.format(files=files), pf("follow"))] + [(M_DEV[c].format(f=pf(f"c{c}_cohort")), pf(f"c{c}_dev")) for c in (1, 2)]
        out["outcome"] = [(M_OUTCOME.format(files=files), pf("outcome"))]
    else:
        out["covars"] = [(E_COVARS.format(files=files), pf("covars"))]
        out["follow"] = [(E_FOLLOW.format(files=files), pf("follow"))] + [(E_DEV[c].format(f=pf(f"c{c}_cohort")), pf(f"c{c}_dev")) for c in (1, 2)]
        out["outcome"] = [(E_OUTCOME.format(files=files), pf("outcome"))]
    return out


def extract(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    d = "m" if args.db == "mimic" else "e"
    if args.part == "outcome":
        if not args.confirmed or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.confirmed):
            raise SystemExit("outcome extraction needs --confirmed YYYY-MM-DD (the date stage 1 was confirmed)")
        if not args.results or not os.path.exists(os.path.join(args.results, "stage1_design.json")):
            raise SystemExit("stage 1 design diagnostics (results/stage1_design.json) must exist before outcomes are extracted")
    con = connect_slim(P, d)
    for sql, path in all_sql(P, args.db)[args.part]:
        write_pq(con, sql, path)
        status(f"{os.path.basename(path)} written", t0)
    note = f" (stage 1 confirmed on {args.confirmed})" if args.part == "outcome" else ""
    log_run(args.results, f"extract --db {args.db} --part {args.part}: ok ({time.time() - t0:.0f} s){note}")


M_UNIT = {"Medical Intensive Care Unit (MICU)": "MICU", "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MSICU",
          "Surgical Intensive Care Unit (SICU)": "SICU", "Trauma SICU (TSICU)": "SICU",
          "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU", "Coronary Care Unit (CCU)": "CCU",
          "Neuro Surgical Intensive Care Unit (Neuro SICU)": "NEURO", "Neuro Intermediate": "NEURO", "Neuro Stepdown": "NEURO"}
E_UNIT = {"Med-Surg ICU": "MEDSURG", "MICU": "MICU", "SICU": "SICU", "Neuro ICU": "NEURO", "Cardiac ICU": "CARDIAC",
          "CCU-CTICU": "CARDIAC", "CTICU": "CARDIAC", "CSICU": "CARDIAC"}
M_SURG = {"SURG", "CSURG", "NSURG", "TSURG", "VSURG", "ORTHO", "PSURG", "TRAUM", "GU", "GYN", "ENT", "DENT", "EYE"}
E_SURG = {"Operating Room", "Recovery Room", "PACU"}
M_ERA_REF, E_ERA_REF = "2011 - 2013", "2014"


def overlap_keep(df, use_site):
    keep = np.ones(len(df), bool)
    a_all = (df["arm"] == "A").sum()
    b_all = (df["arm"] == "B").sum()
    for col in (["era"] + (["site"] if use_site else [])):
        key = df[col].astype(str).fillna("~")
        g = df.assign(_k=key).groupby("_k")["arm"]
        ok = set()
        for k, s in g:
            a, b = (s == "A").sum(), (s == "B").sum()
            if 100 * a * (a_all + b_all) >= 10 * a_all * (a + b) and 100 * b * (a_all + b_all) >= 10 * b_all * (a + b):
                ok.add(k)
        keep &= key.isin(ok).values
    return keep


def load_key(P, key, stage2=False):
    comparison, db = int(key[1]), key[2]
    co = read_pq(os.path.join(P, f"{db}_c{comparison}_cohort.parquet"))
    flow = {"eligible": len(co), "A": int((co["arm"] == "A").sum()), "B": int((co["arm"] == "B").sum()),
            "other": int(co["arm"].isna().sum())}
    df = co[co["arm"].isin(["A", "B"])].copy()
    for name in ("covars", "follow"):
        df = df.merge(read_pq(os.path.join(P, f"{db}_{name}.parquet")), on="stay_id", how="left", validate="one_to_one")
    df = df.merge(read_pq(os.path.join(P, f"{db}_c{comparison}_dev.parquet")), on="stay_id", how="left", validate="one_to_one")
    if stage2:
        df = df.merge(read_pq(os.path.join(P, f"{db}_outcome.parquet")), on="stay_id", how="left", validate="one_to_one")
    keep = overlap_keep(df, db == "e")
    flow["outside_overlap"] = int((~keep).sum())
    df = df[keep].sort_values("stay_id", kind="mergesort").reset_index(drop=True)
    flow["analysis_A"] = int((df["arm"] == "A").sum())
    flow["analysis_B"] = int((df["arm"] == "B").sum())
    return df, flow


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce").astype(float).values


def egfr_of(df):
    return np.where(np.isfinite(num(df, "scr")), num(df, "egfr"), np.nan)


def design_matrix(df, db, comparison):
    cols, names = [], []

    def add(name, v):
        cols.append(np.asarray(v, float))
        names.append(name)

    age = num(df, "age")
    add("age_lt50", age < 50)
    add("age_ge65", age >= 65)
    add("age_ge80", age >= 80)
    add("female", df["gender"].astype(str).str.upper().str[:1].values == "F")
    egfr = egfr_of(df)
    if comparison == 1:
        e, knots = np.where(np.isfinite(egfr), egfr, 45.0), (15, 30, 45)
    else:
        miss = ~np.isfinite(egfr)
        e, knots = np.where(miss, 90.0, egfr), (30, 60, 90)
        add("egfr_missing", miss)
    add("egfr_per10", e / 10)
    for k in knots:
        add(f"egfr_over{k}_per10", np.maximum(e - k, 0) / 10)
    if db == "m":
        s = num(df, "sofa")
        add("sofa_ge3", s >= 3)
        add("sofa_ge6", s >= 6)
        add("sofa_ge9", s >= 9)
        add("sofa_missing", ~np.isfinite(s))
    else:
        a = num(df, "apache")
        add("apache_ge40", a >= 40)
        add("apache_ge60", a >= 60)
        add("apache_ge80", a >= 80)
        add("apache_ge100", a >= 100)
        add("apache_missing", ~np.isfinite(a))
    r = num(df, "rass")
    add("rass_le_m2", r <= -2)
    add("rass_le_m4", r <= -4)
    add("rass_ge1", r >= 1)
    add("rass_missing", ~np.isfinite(r))
    add("vent", np.nan_to_num(num(df, "vent")) == 1)
    for c in ("sed_inf", "opi_inf", "vaso"):
        add(c, np.nan_to_num(num(df, c)) == 1)
    inr, plt = num(df, "inr_max"), num(df, "plt_min")
    add("coag", (inr >= 1.5) | (plt < 50))
    add("coag_not_measured", ~np.isfinite(inr) & ~np.isfinite(plt))
    route = df["route"].astype(str).values
    add("route_enteral", route == "EN")
    add("route_unknown", route == "UNK")
    unit = df["unit"].astype(str)
    if db == "m":
        u = unit.map(M_UNIT).fillna("OTHER").values
        for g in ("MSICU", "SICU", "CVICU", "CCU", "NEURO", "OTHER"):
            add(f"unit_{g}", u == g)
        add("surgical", df["service"].astype(str).isin(M_SURG).values)
        era = df["era"].astype(str).values
        for g in ("2014 - 2016", "2017 - 2019", "2020 - 2022"):
            add("era_" + g.replace(" - ", "_"), era == g)
    else:
        u = unit.map(E_UNIT).fillna("MEDSURG").values
        for g in ("MICU", "SICU", "NEURO", "CARDIAC"):
            add(f"unit_{g}", u == g)
        add("surgical", df["service"].astype(str).isin(E_SURG).values)
        add("era_2015", df["era"].astype(str).values == "2015")
    h = num(df, "h_icu")
    add("h_ge6", h >= 6)
    add("h_ge24", h >= 24)
    add("h_ge72", h >= 72)
    for c in ("dementia", "alcohol", "liver"):
        add(c, np.nan_to_num(num(df, c)) == 1)
    if db == "e":
        site = df["site"].values
        vc = pd.Series(site).value_counts()
        big = [s for s in vc.index if vc[s] >= 10]
        ref = vc.index[0]
        for s in sorted(big):
            if s != ref:
                add(f"hosp_{int(s)}", site == s)
        small = ~np.isin(site, big)
        if small.any():
            add("hosp_small", small)
    X = np.column_stack(cols)
    extras = {"age_years": age, "egfr": egfr, "h_icu": h}
    if db == "m":
        extras["sofa"] = num(df, "sofa")
    else:
        extras["apache"] = num(df, "apache")
    return X, names, extras


def prevalence_filter(X, names):
    keep = []
    for j in range(X.shape[1]):
        v = X[:, j]
        u = np.unique(v)
        if u.size <= 2 and set(u.tolist()) <= {0.0, 1.0}:
            ones = int(v.sum())
            keep.append(min(ones, v.size - ones) >= 10)
        else:
            keep.append(np.var(v) > 0)
    keep = np.array(keep, bool)
    return X[:, keep], [n for n, k in zip(names, keep) if k], [n for n, k in zip(names, keep) if not k]


def build(df, db, comparison, stage2=False):
    X, names, extras = design_matrix(df, db, comparison)
    X, names, dropped = prevalence_filter(X, names)
    egfr, age, rass = egfr_of(df), num(df, "age"), num(df, "rass")
    sev = num(df, "sofa") if db == "m" else num(df, "apache")
    d = {"X": X, "names": names, "dropped": dropped, "extras": extras, "A": (df["arm"].values == "A").astype(int),
         "site": df["site"].values.astype(float) if db == "e" else None,
         "t_end": num(df, "t_end"), "end_death": df["end_death"].fillna(False).astype(bool).values, "t_dev": num(df, "t_dev"),
         "sub": {"cc": np.isfinite(rass) & np.isfinite(egfr) & np.isfinite(sev), "rass3": np.isfinite(rass) & (rass >= -3)}}
    if comparison == 1:
        d["sub"]["egfr50"] = np.isfinite(egfr) & (egfr < 50)
    else:
        d["sub"]["egfr60"] = np.isfinite(egfr) & (egfr < 60)
        d["sub"]["age65"] = age >= 65
    if stage2:
        d["t_del"] = num(df, "t_del")
        d["t_hypo"] = num(df, "t_hypo")
        d["hosp_death"] = df["hosp_death"].fillna(False).astype(bool).values.astype(int)
    return d


def take(d, idx):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == d["A"].shape[0] and k != "names":
            out[k] = v[idx]
        elif k == "sub":
            out[k] = {s: m[idx] for s, m in v.items()}
        elif k == "extras":
            out[k] = {s: m[idx] for s, m in v.items()}
        else:
            out[k] = v
    return out


def subset_d(d, mask):
    s = take(d, np.nonzero(mask)[0])
    X, names, dropped = prevalence_filter(s["X"], s["names"])
    s["X"], s["names"], s["dropped"] = X, names, dropped
    return s


def censor_w(X, A, kend, typ, ck, hyp, pp, cache, key):
    W = None
    for kind, on, ev, by_arm in (("dis", hyp, ck == 1, False), ("dev", pp, ck == 2, True)):
        if not on:
            continue
        b0 = {g: cache[f"{key}:{kind}:{g}"] for g in ("all", 1, 0) if f"{key}:{kind}:{g}" in cache}
        Wk, b = sc.ipc_weights(X, A, kend, typ, ev.astype(int), by_arm=by_arm, beta0=b0)
        for g, v in b.items():
            cache[f"{key}:{kind}:{g}"] = v
        W = Wk if W is None else W * Wk
    return sc.truncate_arm(W, kend, A)


def cif_contrast(d, w, t_ev, reads=(168,), pp=False, hyp=False, cache=None, key=""):
    cache = {} if cache is None else cache
    A = d["A"]
    kend, typ, ck = sc.follow_types(d["t_end"], d["end_death"], t_ev, t_dev=d["t_dev"] if pp else None, disc_competing=not hyp)
    if hyp or pp:
        Wc, _ = censor_w(d["X"], A, kend, typ, ck, hyp, pp, cache, key)
        W = np.asarray(w, float)[:, None] * Wc
    else:
        W = np.asarray(w, float)
    out = {}
    risks = {}
    for a in (1, 0):
        r = A == a
        risks[a] = sc.aj_cif(W[r], kend[r], typ[r])
    for h in reads:
        e = sc.effects(risks[1][h - 1], risks[0][h - 1])
        e["eventsA"] = int(((typ == 1) & (A == 1) & (kend <= h - 1)).sum())
        e["eventsB"] = int(((typ == 1) & (A == 0) & (kend <= h - 1)).sum())
        e["nA"], e["nB"] = int((A == 1).sum()), int((A == 0).sum())
        out[h] = e
    return out


def ow_fit(d, cache, key):
    ps, beta = sc.ps_fit(d["X"], d["A"], beta0=cache.get(key + ":ps"))
    cache[key + ":ps"] = beta
    return ps


def binary_contrast(A, Y, w):
    rA = float((w * Y)[A == 1].sum() / w[A == 1].sum())
    rB = float((w * Y)[A == 0].sum() / w[A == 0].sum())
    e = sc.effects(rA, rB)
    e["eventsA"], e["eventsB"] = int(Y[A == 1].sum()), int(Y[A == 0].sum())
    e["nA"], e["nB"] = int((A == 1).sum()), int((A == 0).sum())
    return e


def iv_instrument(A, site):
    s = pd.Series(A).groupby(site)
    n, p = s.size(), s.mean()
    p = p[n >= 10]
    Z = np.full(len(A), np.nan)
    if p.size < 3:
        return Z, p, (float("nan"), float("nan"))
    lo, hi = np.percentile(p.values, [100 / 3, 200 / 3])
    zh = {h: (1.0 if v >= hi else (0.0 if v <= lo else np.nan)) for h, v in p.items()}
    Z = np.array([zh.get(h, np.nan) for h in site])
    return Z, p, (float(lo), float(hi))


def iv_columns(names):
    return [j for j, nm in enumerate(names) if not nm.startswith("hosp_")]


def iv_estimate(d, Y):
    Z, _, _ = iv_instrument(d["A"], d["site"])
    r = np.isfinite(Z)
    if r.sum() < 50 or len(np.unique(Z[r])) < 2:
        return {"rd": float("nan"), "F": float("nan"), "pi": float("nan")}
    cols = iv_columns(d["names"])
    return sc.tsls(Y[r], d["A"][r], Z[r], d["X"][r][:, cols], d["site"][r])


ANALYSES = {1: ["primary", "hyp", "negctl", "death", "iptw", "pp", "cc", "rass3", "egfr50", "crude"],
            2: ["primary", "hyp", "negctl", "death", "iptw", "pp", "cc", "rass3", "egfr60", "age65", "crude"]}
LABELS = {
    "primary": "Main: overlap weighting; ICU discharge (alive) and death as competing events; 168 h",
    "del72": "Secondary: delirium within 72 h (as the main analysis)",
    "hyp": "Secondary: estimand of the plan (ICU discharge censored with inverse probability of censoring weights, death competing)",
    "death": "Secondary: in-hospital death (binary, overlap weighting)",
    "negctl": "Negative control: first glucose <70 mg/dL within 168 h (as the main analysis)",
    "iptw": "Sensitivity: inverse probability of treatment weighting (average treatment effect, truncated at the 1st/99th percentiles)",
    "pp": "Secondary: per protocol (censored at deviation, inverse probability weighted)",
    "cc": "Sensitivity: complete cases (RASS, creatinine, SOFA or APACHE all recorded)",
    "rass3": "Sensitivity: RASS ≥ −3 at time zero",
    "egfr50": "Sensitivity: eGFR < 50 (former label threshold)",
    "egfr60": "Stratum: eGFR < 60",
    "age65": "Stratum: age ≥ 65 years",
    "crude": "Descriptive: unweighted (as the main analysis)",
    "iv": "Instrumental variable (eICU hospital preference, 2SLS risk difference)",
}


def run_stage2(d, comparison, db, cache=None):
    cache = {} if cache is None else cache
    res = {}
    A = d["A"]
    ps = ow_fit(d, cache, "full")
    ow = sc.overlap_w(ps, A)
    pr = cif_contrast(d, ow, d["t_del"], reads=(72, 168))
    res["primary"], res["del72"] = pr[168], pr[72]
    res["hyp"] = cif_contrast(d, ow, d["t_del"], hyp=True, cache=cache, key="hyp")[168]
    res["negctl"] = cif_contrast(d, ow, d["t_hypo"])[168]
    res["death"] = binary_contrast(A, d["hosp_death"], ow)
    iw, _, _ = sc.iptw_w(ps, A)
    res["iptw"] = cif_contrast(d, iw, d["t_del"])[168]
    res["pp"] = cif_contrast(d, ow, d["t_del"], pp=True, cache=cache, key="pp")[168]
    for s in ANALYSES[comparison]:
        if s in d["sub"]:
            m = d["sub"][s]
            if m.sum() < 20 or len(np.unique(A[m])) < 2:
                res[s] = sc.effects(float("nan"), float("nan"))
                continue
            ds = subset_d(d, m)
            ows = sc.overlap_w(ow_fit(ds, cache, s), ds["A"])
            res[s] = cif_contrast(ds, ows, ds["t_del"])[168]
    res["crude"] = cif_contrast(d, np.ones(len(A)), d["t_del"])[168]
    if db == "e":
        y = np.isfinite(d["t_del"]).astype(float)
        res["iv"] = iv_estimate(d, y)
    return res


def q(v, ps=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    return [round(float(x), 3) for x in np.percentile(v, ps)] if len(v) else []


def design_key(d, flow, key):
    comparison, db = int(key[1]), key[2]
    X, names, A = d["X"], d["names"], d["A"]
    out = {"flow": {k: sc.mask_n(v) for k, v in flow.items()}}
    out["gate_check"] = {k: (sc.mask_n(flow[k]) == GATE[key][k]) for k in ("eligible", "A", "B", "other")}
    out["columns"] = names
    out["columns_dropped_low_prevalence"] = d["dropped"]
    ps, beta = sc.ps_fit(X, A)
    ow = sc.overlap_w(ps, A)
    iw, cuts, wmax = sc.iptw_w(ps, A)
    out["ps_quantiles"] = {"A": q(ps[A == 1]), "B": q(ps[A == 0])}
    lo, hi = max(ps[A == 1].min(), ps[A == 0].min()), min(ps[A == 1].max(), ps[A == 0].max())
    outside = (ps < lo) | (ps > hi)
    out["outside_common_support"] = {"A": sc.mask_n(int((outside & (A == 1)).sum())), "B": sc.mask_n(int((outside & (A == 0)).sum()))}
    out["ess"] = {"overlap": {"A": round(sc.ess(ow[A == 1]), 1), "B": round(sc.ess(ow[A == 0]), 1)},
                  "iptw": {"A": round(sc.ess(iw[A == 1]), 1), "B": round(sc.ess(iw[A == 0]), 1)}}
    out["iptw_truncation"] = {"cut_low": round(cuts[0], 3), "cut_high": round(cuts[1], 3), "max_untruncated": round(wmax, 2)}
    Xe = np.column_stack([X] + [np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 0) for v in d["extras"].values()])
    ne = names + [f"{k} (continuous)" for k in d["extras"]]
    un, so, si = sc.smd_table(Xe, ne, A), sc.smd_table(Xe, ne, A, ow), sc.smd_table(Xe, ne, A, iw)
    out["smd"] = {nm: [round(un[nm], 3), round(so[nm], 3), round(si[nm], 3)] for nm in ne}
    out["max_abs_smd_model_columns"] = {"overlap": round(max(abs(so[nm]) for nm in names), 4), "iptw": round(max(abs(si[nm]) for nm in names), 3)}
    out["max_abs_smd"] = {"unweighted": round(max(abs(v) for v in un.values()), 3),
                          "overlap": round(max(abs(v) for v in so.values()), 3), "iptw": round(max(abs(v) for v in si.values()), 3)}
    prev = {}
    for j, nm in enumerate(names):
        v = X[:, j]
        if set(np.unique(v).tolist()) <= {0.0, 1.0}:
            prev[nm] = {arm: f"{sc.mask_n(int(v[A == a].sum()))} ({sc.pct_text(int(v[A == a].sum()), int((A == a).sum()))})"
                        for arm, a in (("A", 1), ("B", 0))}
    out["prevalence"] = prev
    te, dth = d["t_end"], d["end_death"]
    out["followup_both_arms"] = {f"in_icu_alive_at_{h}h": sc.mask_n(int((te > h).sum())) for h in (24, 72, 168)}
    out["followup_both_arms"]["discharged_alive_before_168h"] = sc.mask_n(int(((te < 168) & ~dth).sum()))
    kend, typ, ck = sc.follow_types(te, dth, np.full(len(A), np.nan))
    _, tinfo = censor_w(X, A, kend, typ, ck, True, False, {}, "s1")
    out["ipcw_discharge"] = {("A" if a == 1 else "B"): {"cut99": round(v[0], 3), "max": round(v[1], 3)} for a, v in tinfo.items()}
    kend2, typ2, ck2 = sc.follow_types(te, dth, np.full(len(A), np.nan), t_dev=d["t_dev"], disc_competing=True)
    dev = ck2 == 2
    out["pp_deviation_in_followup"] = {"A": sc.mask_n(int((dev & (A == 1)).sum())), "B": sc.mask_n(int((dev & (A == 0)).sum()))}
    _, tinfo2 = censor_w(X, A, kend2, typ2, ck2, False, True, {}, "s1pp")
    out["ipcw_pp"] = {("A" if a == 1 else "B"): {"cut99": round(v[0], 3), "max": round(v[1], 3)} for a, v in tinfo2.items()}
    subs = {}
    for s, m in d["sub"].items():
        ent = {"A": sc.mask_n(int((m & (A == 1)).sum())), "B": sc.mask_n(int((m & (A == 0)).sum()))}
        if m.sum() >= 20 and len(np.unique(A[m])) == 2:
            ds = subset_d(d, m)
            pss, _ = sc.ps_fit(ds["X"], ds["A"])
            ows = sc.overlap_w(pss, ds["A"])
            ent["max_abs_smd_overlap"] = round(max(abs(v) for v in sc.smd_table(ds["X"], ds["names"], ds["A"], ows).values()), 3)
            ent["ess_overlap"] = {"A": round(sc.ess(ows[ds["A"] == 1]), 1), "B": round(sc.ess(ows[ds["A"] == 0]), 1)}
        subs[s] = ent
    out["subsets"] = subs
    if db == "e":
        Z, p, (clo, chi) = iv_instrument(A, d["site"])
        r = np.isfinite(Z)
        iv = {"hospitals_with_10plus": int(p.size), "hospitals_high": int(sum(1 for v in p.values if v >= chi)),
              "hospitals_low": int(sum(1 for v in p.values if v <= clo)),
              "patients_Z1": sc.mask_n(int((Z == 1).sum())), "patients_Z0": sc.mask_n(int((Z == 0).sum()))}
        if r.sum() >= 50 and len(np.unique(Z[r])) == 2:
            nz1, nz0 = int((A[Z == 1] == 1).sum()), int((A[Z == 0] == 1).sum())
            iv["share_A_Z1"] = sc.pct_text(nz1, int((Z == 1).sum()))
            iv["share_A_Z0"] = sc.pct_text(nz0, int((Z == 0).sum()))
            cols = iv_columns(names)
            fs = sc.tsls(np.zeros(int(r.sum())), A[r], Z[r], X[r][:, cols], d["site"][r])
            iv["first_stage_pi"] = round(fs["pi"], 3)
            iv["first_stage_F_cluster"] = round(fs["F"], 1)
            zsmd = sc.smd_table(X[r][:, cols], [names[j] for j in cols], Z[r].astype(int))
            iv["max_abs_smd_by_Z"] = round(max(abs(v) for v in zsmd.values()), 3)
            iv["smd_by_Z_over_0.1"] = {k: round(v, 3) for k, v in zsmd.items() if abs(v) > 0.1}
        out["iv"] = iv
    return out


def design(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    if glob.glob(os.path.join(P, "*_outcome.parquet")):
        raise SystemExit("an outcome table exists in the private folder: stage 1 must run before outcomes are extracted")
    res = {"sap": "SAP.md v1.0", "stage": 1, "created": datetime.datetime.now().isoformat(timespec="seconds"), "keys": {}}
    for key in KEYS:
        df, flow = load_key(P, key)
        d = build(df, key[2], int(key[1]))
        res["keys"][key] = design_key(d, flow, key)
        status(f"design {key} done", t0)
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "stage1_design.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.results, "stage1_design.md"), "w", encoding="utf-8") as f:
        f.write(stage1_md(res))
    log_run(args.results, f"design: ok ({time.time() - t0:.0f} s)")


def stage1_md(res):
    L = ["# Stage 1 design diagnostics (no outcomes): the dose and PPI comparisons", "",
         f"Plan: {res['sap']}; generated: {res['created']}. Counts masked (1–9 as <10, otherwise the nearest multiple of 5); percentages only when the numerator is ≥10.", ""]
    for key in KEYS:
        r = res["keys"][key]
        comparison, db = int(key[1]), key[2]
        arms = "A = planned daily dose ≤20 mg, B = ≥40 mg" if comparison == 1 else "A = PPI, B = famotidine"
        L += [f"## {TITLE[comparison]} · {DBNAME[db]} ({arms})", ""]
        fl = r["flow"]
        L.append(f"- Flow: eligible {fl['eligible']}; A {fl['A']}; B {fl['B']}; other strategies {fl['other']}; outside overlap {fl['outside_overlap']}; "
                 f"analysis set A {fl['analysis_A']} / B {fl['analysis_B']}. Matches the gate: {'yes' if all(r['gate_check'].values()) else 'no ' + str(r['gate_check'])}")
        L.append(f"- Propensity score quantiles (0/1/5/25/50/75/95/99/100): A {r['ps_quantiles']['A']}; B {r['ps_quantiles']['B']}; "
                 f"outside common support A {r['outside_common_support']['A']}, B {r['outside_common_support']['B']}")
        L.append(f"- Effective sample size: overlap weights A {r['ess']['overlap']['A']} / B {r['ess']['overlap']['B']}; IPT weights A {r['ess']['iptw']['A']} / B {r['ess']['iptw']['B']}"
                 f" (truncation {r['iptw_truncation']['cut_low']}–{r['iptw_truncation']['cut_high']}, largest untruncated {r['iptw_truncation']['max_untruncated']})")
        m = r["max_abs_smd"]
        L.append(f"- Largest |SMD|: unweighted {m['unweighted']}; overlap weighted {m['overlap']}; IPT weighted {m['iptw']}")
        big = {k: v for k, v in r["smd"].items() if abs(v[0]) >= 0.1}
        if big:
            L.append("- Variables with unweighted |SMD| ≥0.1 (unweighted / overlap / IPTW): " + "; ".join(f"{k} {v[0]}/{v[1]}/{v[2]}" for k, v in big.items()))
        if r["columns_dropped_low_prevalence"]:
            L.append("- Variables dropped because one level had fewer than 10 patients: " + ", ".join(r["columns_dropped_low_prevalence"]))
        fu = r["followup_both_arms"]
        L.append("- Follow-up (both arms): " + "; ".join(f"{k} {v}" for k, v in fu.items()))
        L.append(f"- ICU-discharge censoring weights for the estimand of the plan (secondary analysis; fitted in stage 1 on outcome-free ICU follow-up, stabilised): A truncation point {r['ipcw_discharge']['A']['cut99']}, largest {r['ipcw_discharge']['A']['max']}; "
                 f"B truncation point {r['ipcw_discharge']['B']['cut99']}, largest {r['ipcw_discharge']['B']['max']}")
        L.append(f"- Per protocol (secondary analysis): deviations during follow-up A {r['pp_deviation_in_followup']['A']}, B {r['pp_deviation_in_followup']['B']}; censoring weights A truncation point {r['ipcw_pp']['A']['cut99']}, largest {r['ipcw_pp']['A']['max']}; "
                 f"B truncation point {r['ipcw_pp']['B']['cut99']}, largest {r['ipcw_pp']['B']['max']}")
        for s, e in r["subsets"].items():
            extra = f", largest |SMD| after overlap weighting {e['max_abs_smd_overlap']}, effective sample size A {e['ess_overlap']['A']} / B {e['ess_overlap']['B']}" if "max_abs_smd_overlap" in e else ""
            L.append(f"- Subset {s}: A {e['A']} / B {e['B']}{extra}")
        if "iv" in r:
            iv = r["iv"]
            L.append(f"- Instrumental variable (thirds of hospital preference): hospitals with ≥10 patients {iv['hospitals_with_10plus']}, high {iv['hospitals_high']}, low {iv['hospitals_low']}; "
                     f"Z=1 {iv['patients_Z1']} patients, Z=0 {iv['patients_Z0']} patients" +
                     (f"; share of A Z=1 {iv['share_A_Z1']}, Z=0 {iv['share_A_Z0']}; first-stage coefficient {iv['first_stage_pi']}, F clustered by hospital {iv['first_stage_F_cluster']}; "
                      f"largest |SMD| by Z {iv['max_abs_smd_by_Z']}" if "first_stage_pi" in iv else ""))
        L += ["", "| Variable | A n (%) | B n (%) | SMD unweighted | SMD overlap | SMD IPTW |", "|---|---|---|---|---|---|"]
        for nm, v in r["smd"].items():
            pa = r["prevalence"].get(nm, {"A": "", "B": ""})
            L.append(f"| {nm} | {pa['A']} | {pa['B']} | {v[0]} | {v[1]} | {v[2]} |")
        L.append("")
    return "\n".join(L) + "\n"


def to_json(o):
    if isinstance(o, dict):
        return {str(k): to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def estimate(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    for key in KEYS:
        df, flow = load_key(P, key, stage2=True)
        d = build(df, key[2], int(key[1]), stage2=True)
        cache = {}
        res = run_stage2(d, int(key[1]), key[2], cache)
        with open(os.path.join(P, f"est_{key}.json"), "w") as f:
            json.dump({"flow": flow, "res": to_json(res)}, f)
        np.savez(os.path.join(P, f"cache_{key}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
        status(f"estimate {key} done", t0)
    log_run(args.results, f"estimate: ok ({time.time() - t0:.0f} s)")


def resample(d, db, rng):
    n = len(d["A"])
    if db == "m":
        return take(d, rng.integers(0, n, n))
    site = d["site"]
    hs = np.unique(site)
    draw = rng.choice(hs, size=hs.size, replace=True)
    rows = {h: np.nonzero(site == h)[0] for h in hs}
    idx = np.concatenate([rows[h] for h in draw])
    copy = np.concatenate([np.full(rows[h].size, i) for i, h in enumerate(draw)])
    b = take(d, idx)
    b["site"] = b["site"] * 1000 + copy
    return b


def boot_names(comparison, db):
    names = ["primary", "del72"] + [a for a in ANALYSES[comparison] if a != "primary"]
    return names


def _boot_one(args_tuple):
    d, comparison, db, rep, cache = args_tuple
    rng = np.random.default_rng(SEED + rep)
    b = resample(d, db, rng)
    try:
        res = run_stage2(b, comparison, db, dict(cache))
    except Exception:
        return rep, None
    names = boot_names(comparison, db)
    lor = [math.log(res[n]["OR"]) if res[n].get("OR") and np.isfinite(res[n]["OR"]) and res[n]["OR"] > 0 else np.nan for n in names]
    rd = [res[n]["RD"] if np.isfinite(res[n].get("RD", np.nan)) else np.nan for n in names]
    iv = [res["iv"]["rd"], res["iv"]["F"]] if db == "e" else [np.nan, np.nan]
    return rep, (lor, rd, iv)


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    key = args.key
    comparison, db = int(key[1]), key[2]
    df, _ = load_key(P, key, stage2=True)
    d = build(df, db, comparison, stage2=True)
    cache = {}
    cf = os.path.join(P, f"cache_{key}.npz")
    if os.path.exists(cf):
        z = np.load(cf)
        cache = {k.replace("__", ":"): z[k] for k in z.files}
    reps = list(range(args.start, min(args.stop, NBOOT)))
    out = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for rep, r in pool.imap_unordered(_boot_one, [(d, comparison, db, rep, cache) for rep in reps]):
                out[rep] = r
    else:
        for rep in reps:
            out[rep] = _boot_one((d, comparison, db, rep, cache))[1]
    names = boot_names(comparison, db)
    lor = np.full((len(reps), len(names)), np.nan)
    rd = np.full((len(reps), len(names)), np.nan)
    iv = np.full((len(reps), 2), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None:
            lor[i], rd[i], iv[i] = out[rep]
    np.savez(os.path.join(P, f"boot_{key}_{args.start:03d}_{args.stop:03d}.npz"), reps=np.array(reps), lor=lor, rd=rd, iv=iv,
             names=np.array(names))
    status(f"boot {key} {args.start}-{args.stop} done, failed {sum(1 for r in reps if out.get(r) is None)}", t0)
    log_run(args.results, f"boot --key {key} --start {args.start} --stop {args.stop}: ok ({time.time() - t0:.0f} s)")


def load_boot(P, key):
    files = sorted(glob.glob(os.path.join(P, f"boot_{key}_*.npz")))
    if not files:
        return None
    reps, lor, rd, iv, names = [], [], [], [], None
    for f in files:
        z = np.load(f)
        reps.append(z["reps"])
        lor.append(z["lor"])
        rd.append(z["rd"])
        iv.append(z["iv"])
        names = [str(x) for x in z["names"]]
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return {"names": names, "lor": np.concatenate(lor)[first], "rd": np.concatenate(rd)[first], "iv": np.concatenate(iv)[first]}


def rnd(x, nd):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def round_floats(o, key=""):
    if isinstance(o, dict):
        return {k: round_floats(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [round_floats(v, key) for v in o]
    if isinstance(o, float):
        return round(o, 3 if key.startswith("risk") else (1 if key == "F" else 4))
    return o


def fmt(x, nd=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def interpret(orr, lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "no interval"
    if hi < 1:
        return "positive (direction of benefit)" + (", effect smaller than the MCID" if lo > MCID else "")
    if lo > 1:
        return "direction opposite to the hypothesis"
    if lo > MCID:
        return "clinically important benefit excluded (precise negative result)"
    return "inconclusive"


def report(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    out = {}
    for comparison in (1, 2):
        cres = {"per_db": {}, "pooled": {}}
        for db in ("m", "e"):
            key = f"c{comparison}{db}"
            est = json.load(open(os.path.join(P, f"est_{key}.json")))
            bt = load_boot(P, key)
            per = {"flow": {k: sc.mask_n(v) for k, v in est["flow"].items()}, "analyses": {}}
            names = boot_names(comparison, db)
            for j, nm in enumerate(names):
                e = est["res"][nm]
                ent = {"OR": e.get("OR"), "RD": e.get("RD"), "RR": e.get("RR"),
                       "_lor": math.log(e["OR"]) if e.get("OR") else np.nan}
                ran = e.get("OR") is not None and "eventsA" in e
                evA, evB = e.get("eventsA", 0), e.get("eventsB", 0)
                ent["eventsA"], ent["eventsB"] = (sc.mask_n(evA), sc.mask_n(evB)) if ran else ("—", "—")
                ent["nA"], ent["nB"] = (sc.mask_n(e.get("nA", 0)), sc.mask_n(e.get("nB", 0))) if ran else ("—", "—")
                unweighted = nm == "crude"
                ent["riskA"] = e.get("riskA") if (evA >= 10 and not unweighted) else None
                ent["riskB"] = e.get("riskB") if (evB >= 10 and not unweighted) else None
                if unweighted:
                    ent["RD"] = ent["RR"] = None
                    ent["OR"] = rnd(e.get("OR"), 2)
                if bt is not None:
                    v = bt["lor"][:, j]
                    v = v[np.isfinite(v)]
                    w = bt["rd"][:, j]
                    w = w[np.isfinite(w)]
                    ent["boot_valid"] = int(v.size)
                    if v.size >= MIN_BOOT:
                        ent["OR_lo"], ent["OR_hi"] = float(np.exp(np.percentile(v, 2.5))), float(np.exp(np.percentile(v, 97.5)))
                        if unweighted:
                            ent["OR_lo"], ent["OR_hi"] = rnd(ent["OR_lo"], 2), rnd(ent["OR_hi"], 2)
                        ent["se_logOR"] = float(np.std(v, ddof=1))
                    if w.size >= MIN_BOOT and ent["RD"] is not None:
                        ent["RD_lo"], ent["RD_hi"] = float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))
                per["analyses"][nm] = ent
            if db == "e":
                e = est["res"]["iv"]
                ent = {"rd": e.get("rd"), "F": e.get("F"), "pi": e.get("pi")}
                if bt is not None:
                    v = bt["iv"][:, 0]
                    v = v[np.isfinite(v)]
                    if v.size >= MIN_BOOT:
                        ent["rd_lo"], ent["rd_hi"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
                ent["weak"] = not (e.get("F") is not None and e.get("F") >= 10)
                per["iv"] = ent
            cres["per_db"][db] = per
        for nm in boot_names(comparison, "m"):
            pm, pe = cres["per_db"]["m"]["analyses"].get(nm, {}), cres["per_db"]["e"]["analyses"].get(nm, {})
            lors = [x.get("_lor", np.nan) for x in (pm, pe)]
            ses = [x.get("se_logOR", np.nan) for x in (pm, pe)]
            m, se, Q, I2, k = sc.pool_fe(lors, ses)
            ent = {"k": k, "_m": m, "_se": se}
            if k > 0:
                ent.update({"OR": math.exp(m), "OR_lo": math.exp(m - 1.959964 * se), "OR_hi": math.exp(m + 1.959964 * se), "se_logOR": se,
                            "Q": Q, "I2": I2})
                if k == 2:
                    mr, ser, tau2 = sc.pool_dl(lors, ses)
                    ent.update({"OR_re": math.exp(mr), "OR_re_lo": math.exp(mr - 1.959964 * ser), "OR_re_hi": math.exp(mr + 1.959964 * ser),
                                "tau2": tau2})
            cres["pooled"][nm] = ent
        p = cres["pooled"]["primary"]
        m_, se_ = p.get("_m", np.nan), p.get("_se", np.nan)
        cres["conclusion"] = interpret(math.exp(m_), math.exp(m_ - 1.959964 * se_), math.exp(m_ + 1.959964 * se_)) if np.isfinite(se_) else "no interval"
        nc = cres["pooled"]["negctl"]
        nc_sig = nc.get("OR_lo") is not None and (nc["OR_lo"] > 1 or nc["OR_hi"] < 1)
        same = nc_sig and p.get("OR_hi") is not None and ((nc["OR_hi"] < 1 and p["OR_hi"] < 1) or (nc["OR_lo"] > 1 and p["OR_lo"] > 1))
        cres["negative_control"] = ("negative control in the same direction as the main analysis with a 95% interval excluding 1: main conclusion does not hold (residual confounding)" if same else
                                    "negative control 95% interval excludes 1: residual confounding cannot be ruled out" if nc_sig else "negative control 95% interval includes 1")
        het = p.get("I2") is not None and p["I2"] > 0.5
        cres["heterogeneity"] = "the two databases disagree (I² > 50%): pooled conclusion to be read with caution" if het else "no clear disagreement between the two databases (I² ≤ 50%)"
        cres["per_db_reading"] = {db: interpret(x.get("OR"), x.get("OR_lo") or np.nan, x.get("OR_hi") or np.nan)
                                  for db, x in ((db, cres["per_db"][db]["analyses"]["primary"]) for db in ("m", "e"))}
        if het:
            cres["conclusion"] = ("the two databases disagree; described by the estimate of each database: " + "; ".join(f"{DBNAME[db]} {cres['per_db_reading'][db]}" for db in ("m", "e"))
                                  + f" (pooled estimate for reference only: {cres['conclusion']})")
        if np.isfinite(se_):
            cmd = [sys.executable, os.path.join(TOOLS, "bayes_reanalysis.py"), "--or", f"{math.exp(m_):.4f}", "--ci",
                   f"{math.exp(m_ - 1.959964 * se_):.4f}", f"{math.exp(m_ + 1.959964 * se_):.4f}", "--mcid", str(MCID)]
            if PRIOR[comparison]:
                cmd += ["--rct"] + [str(x) for x in PRIOR[comparison]]
            try:
                cres["bayes_cmd"] = " ".join(cmd[1:]).replace(TOOLS + os.sep, "tools/helpers/")
                cres["bayes"] = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
            except Exception as e:
                cres["bayes"] = f"bayes_reanalysis.py failed: {type(e).__name__}"
            cres["bayes_reading"] = bayes_reading(cres.get("bayes", ""), comparison)
        out[comparison] = cres
    for comparison in (1, 2):
        for db in ("m", "e"):
            for ent in out[comparison]["per_db"][db]["analyses"].values():
                ent.pop("_lor", None)
        for ent in out[comparison]["pooled"].values():
            ent.pop("_m", None)
            ent.pop("_se", None)
    os.makedirs(args.results, exist_ok=True)
    for comparison in (1, 2):
        with open(os.path.join(args.results, f"{STEM[comparison]}_estimates.json"), "w", encoding="utf-8") as f:
            json.dump(round_floats(to_json(out[comparison])), f, ensure_ascii=False, indent=1)
        with open(os.path.join(args.results, f"{STEM[comparison]}_estimates.md"), "w", encoding="utf-8") as f:
            f.write(estimates_md(comparison, out[comparison]))
    with open(os.path.join(args.results, "bayes.txt"), "w", encoding="utf-8") as f:
        for comparison in (1, 2):
            f.write(f"=== {NAME[comparison]}: {out[comparison].get('bayes_cmd', '')}\n{out[comparison].get('bayes', '')}\n")
    log_run(args.results, f"report: ok ({time.time() - t0:.0f} s)")
    status("report written", t0)


def bayes_reading(text, comparison):
    lab = "evidence-based" if PRIOR[comparison] else "skeptical"
    for line in text.splitlines():
        if line.startswith(lab):
            m1 = re.search(r"P\(OR<1\) ([0-9.]+)", line)
            m2 = re.search(r"P\(OR<MCID\) ([0-9.]+)", line)
            if not m1:
                break
            p1 = float(m1.group(1))
            word = "probably in the direction of benefit" if p1 >= 0.95 else ("leaning towards benefit" if p1 >= 0.80 else "inconclusive")
            prior = "evidence-based prior (Tawadrous 2014, half weight)" if PRIOR[comparison] else "sceptical prior (no evidence-based prior for the PPI comparison)"
            return f"{prior}: P(OR<1) {p1:.3f}, P(OR<MCID) {float(m2.group(1)):.3f}: {word}"
    return "Bayesian reanalysis not run"


def estimates_md(comparison, c):
    arms = "A = planned daily dose ≤20 mg, B = ≥40 mg (new famotidine users with eGFR <60)" if comparison == 1 else "A = first acid suppressant a PPI, B = famotidine"
    L = [f"# {TITLE[comparison]} results ({arms})", "",
         "Odds ratio <1: lower risk in arm A (the direction of benefit hypothesised in the plan). Counts masked: 1–9 as <10, otherwise the nearest multiple of 5; the risk of an arm with <10 events is not reported.", "",
         f"- Main conclusion (pooled main analysis; reading rules in SAP section 8): {c['conclusion']}", f"- {c['negative_control']}", f"- {c['heterogeneity']}",
         f"- Bayesian: {c.get('bayes_reading', 'not run')}", ""]
    for db in ("m", "e"):
        fl = c["per_db"][db]["flow"]
        L.append(f"- {DBNAME[db]} flow: eligible {fl['eligible']}; A {fl['A']}; B {fl['B']}; other strategies {fl['other']}; analysis set A {fl['analysis_A']} / B {fl['analysis_B']}")
    L += ["", "| Analysis | Database | Events A / B | Risk A / B | RD (95% CI) | OR (95% CI) |", "|---|---|---|---|---|---|"]
    for nm in boot_names(comparison, "m"):
        for db in ("m", "e"):
            e = c["per_db"][db]["analyses"].get(nm, {})
            risk = f"{fmt(e.get('riskA'), 3)} / {fmt(e.get('riskB'), 3)}" if nm != "death" else f"{fmt(e.get('riskA'), 3)} / {fmt(e.get('riskB'), 3)}"
            L.append(f"| {LABELS[nm]} | {DBNAME[db]} | {e.get('eventsA', '')} / {e.get('eventsB', '')} | {risk} | "
                     f"{fmt(e.get('RD'), 3)} ({fmt(e.get('RD_lo'), 3)} to {fmt(e.get('RD_hi'), 3)}) | "
                     f"{fmt(e.get('OR'))} ({fmt(e.get('OR_lo'))}–{fmt(e.get('OR_hi'))}) |")
        p = c["pooled"].get(nm, {})
        L.append(f"| {LABELS[nm]} | Pooled (fixed effect) | | | | {fmt(p.get('OR'))} ({fmt(p.get('OR_lo'))}–{fmt(p.get('OR_hi'))})"
                 f"{'; I² ' + fmt(100 * p['I2'], 0) + '%' if p.get('I2') is not None else ''} |")
    p = c["pooled"].get("primary", {})
    if p.get("OR_re") is not None:
        L += ["", f"- Main analysis, random-effects pooling (DerSimonian–Laird): OR {fmt(p.get('OR_re'))} ({fmt(p.get('OR_re_lo'))}–{fmt(p.get('OR_re_hi'))}), tau² {fmt(p.get('tau2'), 4)}"]
    iv = c["per_db"]["e"].get("iv", {})
    L += ["", f"- {LABELS['iv']}: RD {fmt(iv.get('rd'), 3)} ({fmt(iv.get('rd_lo'), 3)} to {fmt(iv.get('rd_hi'), 3)}), first-stage coefficient {fmt(iv.get('pi'), 3)}, "
              f"F clustered by hospital {fmt(iv.get('F'), 1)}" + ("; F < 10, weak instrument, not interpreted" if iv.get("weak") else ""), ""]
    if c.get("bayes"):
        L += ["## Bayesian reanalysis (pooled main analysis)", "", "```", c.get("bayes_cmd", ""), c["bayes"].rstrip(), "```", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slim")
    s.add_argument("--db", choices=["mimic", "eicu"], required=True)
    s.add_argument("--step", required=True)
    s.add_argument("--src", required=True)
    s.add_argument("--private", required=True)
    s.add_argument("--results", default="../results")
    s = sub.add_parser("extract")
    s.add_argument("--db", choices=["mimic", "eicu"], required=True)
    s.add_argument("--part", choices=["cohort", "covars", "follow", "outcome"], required=True)
    s.add_argument("--private", required=True)
    s.add_argument("--results", default="../results")
    s.add_argument("--confirmed", help="date stage 1 was confirmed (needed for --part outcome)")
    for name in ("design", "estimate", "report"):
        s = sub.add_parser(name)
        s.add_argument("--private", required=True)
        s.add_argument("--results", default="../results")
    s = sub.add_parser("boot")
    s.add_argument("--key", choices=KEYS, required=True)
    s.add_argument("--start", type=int, required=True)
    s.add_argument("--stop", type=int, required=True)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--private", required=True)
    s.add_argument("--results", default="../results")
    args = ap.parse_args()
    {"slim": slim, "extract": extract, "design": design, "estimate": estimate, "boot": boot, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
