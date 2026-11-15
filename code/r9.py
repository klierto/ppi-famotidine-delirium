#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import gzip
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


def _hook(t, e, tb):
    msg = str(e).split("Original Line")[0].strip().splitlines()
    print(f"ERROR {t.__name__}: {msg[0][:160] if msg else ''}", file=sys.stderr)


sys.excepthook = _hook

SRC = {"n": os.path.join(os.environ.get("DATA_DIR", "data"), "northwestern-icu-nwicu-database-0.1.0/data"),
       "i": os.path.join(os.environ.get("DATA_DIR", "data"), "inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.4.2/"
                               "inspire-a-publicly-available-research-dataset-for-perioperative-medicine-1.4.2")}
FILES = {"n": {"icu": "nw_icu/icustays.csv.gz", "adm": "nw_hosp/admissions.csv.gz", "pat": "nw_hosp/patients.csv.gz",
               "dx": "nw_hosp/diagnoses_icd.csv.gz", "proc": "nw_icu/procedureevents.csv.gz", "emar": "nw_hosp/emar.csv.gz",
               "lab": "nw_hosp/labevents.csv.gz"},
         "i": {"op": "operations.csv.gz", "dx": "diagnosis.csv.gz", "med": "medications.csv.gz", "lab": "labs.csv.gz",
               "wv": "ward_vitals.csv.gz"}}
PARTS = {"n": {"small": ["icu", "adm", "pat", "dx", "proc"], "emar": ["emar"], "lab": ["lab"]},
         "i": {"small": ["op", "dx"], "med": ["med"], "lab": ["lab", "wv"]}}
DBNAME = {"n": "NWICU", "i": "INSPIRE"}
IDCOL = {"n": "stay_id", "i": "op_id"}
NAMES = ["icu", "icu_oq", "hosp", "hosp_oq", "negctl"]
ICU_UNITS = "'ICU', 'MICU', 'NSICU', 'CICU', 'CCU', 'SICU', 'CTICU', 'ICU OVERFLOW'"

PPI_N = "pantopra|protonix|omepra|prilosec|esomepra|nexium|lansopra|prevacid|rabepra|aciphex|dexilant"
PPI_I = "pantopra|lansopra|esomepra|omepra|rabepra|ilapra"
FAM = "famotid|pepcid"
H2O_N = "ranitid|zantac|cimetid|nizatid"
H2O_I = "ranitid|nizatid|cimetid|lafutid|roxatid|revapraz|tegopraz|fexupraz"
HAL = "haloperid|haldol"
OQ = "olanzap|zyprexa|quetiap|seroquel"
APO = "risperid|risperdal|ziprasid|geodon|aripipraz|abilify"
LAI = "decanoate|pamoate|lauroxil|consta|maintena|aristada|sustenna|relprevv|long.acting|im sers"
SED_N = "propofol|diprivan|midazolam|versed|lorazepam|ativan|dexmedetomidine|precedex|ketamine"
SED_I = "midazolam|lorazepam|diazepam|propofol|dexmedetomidine"
OPI = ("fentanyl|sufentanil|remifentanil|alfentanil|hydromorphone|dilaudid|morphine|oxycodone|hydrocodone|meperidine|pethidine|"
       "methadone|tramadol|tapentadol|buprenorphine|codeine|nalbuphine|butorphanol")
PRES = "norepinephrine|levophed|epinephrine|phenylephrine|vasopressin|dopamine"
PRES_NOT = ("lidocaine|bupivacaine|ropivacaine|mepivacaine|articaine|prilocaine|chloroprocaine|tetracaine|racepinephrine|nebu|"
            "dobutamine|ophth|nasal|topical")
N_ALL = "|".join([PPI_N, FAM, H2O_N, HAL, OQ, APO, SED_N, OPI, PRES])
I_ALL = "|".join([PPI_I, FAM, H2O_I, HAL, OQ, APO, "clozap|lithium", SED_I, OPI, PRES])


def ap_case(col, lai):
    return (f"WHEN regexp_matches({col}, '{HAL}') THEN CASE WHEN regexp_matches({col}, '{lai}') THEN NULL ELSE 'HAL' END "
            f"WHEN regexp_matches({col}, '{OQ}') THEN CASE WHEN regexp_matches({col}, '{lai}') THEN NULL ELSE 'OQ' END "
            f"WHEN regexp_matches({col}, '{APO}') THEN CASE WHEN regexp_matches({col}, '{lai}') THEN NULL ELSE 'APO' END ")


N_CLS = (f"CASE WHEN regexp_matches(med, '{PPI_N}') THEN 'PPI' WHEN regexp_matches(med, '{FAM}') THEN 'FAM' "
         f"WHEN regexp_matches(med, '{H2O_N}') THEN 'OTH' " + ap_case("med", LAI)
         + f"WHEN regexp_matches(med, '{SED_N}') THEN 'SED' WHEN regexp_matches(med, '{OPI}') THEN 'OPI' "
         f"WHEN regexp_matches(med, '{PRES}') AND NOT regexp_matches(med, '{PRES_NOT}') THEN 'PRES' END")
I_CLS = (f"CASE WHEN regexp_matches(drug, '{PPI_I}') THEN 'PPI' WHEN regexp_matches(drug, '{FAM}') THEN 'FAM' "
         f"WHEN regexp_matches(drug, '{H2O_I}') OR regexp_matches(atc, '^A02B[AC]') THEN 'OTH' " + ap_case("drug", LAI)
         + f"WHEN regexp_matches(drug, 'clozap|lithium') THEN 'PSYRX' WHEN regexp_matches(drug, '{SED_I}') THEN 'SED' "
         f"WHEN regexp_matches(drug, '{OPI}') THEN 'OPI' "
         f"WHEN regexp_matches(drug, '{PRES}') AND route = 'iv' AND NOT regexp_matches(drug, '{PRES_NOT}') THEN 'PRES' END")
GI_N = ("(regexp_matches(d.code, '^K2[5-8]') OR d.code IN ('K920', 'K921', 'K922', 'I8501', 'I8511', 'K2211', 'K2271') "
        "OR regexp_matches(d.code, '^K29[0-9]1') OR regexp_matches(d.code, '^K31811'))")
GI_I = "regexp_matches(d.code, '^(K2[5-8]|K92)')"
ROUTE_IV = r"\biv\b|\binj\b"
ROUTE_EN = r"\boral\b"


def src(db, key, root=None):
    q = ", quote='\"', escape='\"'" if db == "i" else ""
    return f"read_csv('{os.path.join(root or SRC[db], FILES[db][key])}', header=true, all_varchar=true{q})"


def B(x, db="n"):
    return f"TRY_CAST({x} AS BIGINT)" if db == "n" else f"TRY_CAST(TRY_CAST({x} AS DOUBLE) AS BIGINT)"


def T(x):
    return f"TRY_CAST({x} AS TIMESTAMP)"


def extract_sql(db, key, root=None):
    s = src(db, key, root)
    if db == "n":
        q = {
            "icu": f"SELECT {B('subject_id')} AS subject_id, {B('hadm_id')} AS hadm_id, {B('stay_id')} AS stay_id, first_careunit, "
                   f"{T('intime')} AS intime, {T('outtime')} AS outtime FROM {s}",
            "adm": f"SELECT {B('subject_id')} AS subject_id, {B('hadm_id')} AS hadm_id, {T('admittime')} AS admittime, "
                   f"{T('dischtime')} AS dischtime, {T('deathtime')} AS deathtime, admission_type, admission_location, "
                   f"discharge_location, hospital_expire_flag FROM {s}",
            "pat": f"SELECT {B('subject_id')} AS subject_id, gender, TRY_CAST(anchor_age AS INTEGER) AS anchor_age, "
                   f"TRY_CAST(anchor_year AS INTEGER) AS anchor_year, anchor_year_group FROM {s}",
            "dx": f"SELECT {B('subject_id')} AS subject_id, {B('hadm_id')} AS hadm_id, TRY_CAST(seq_num AS INTEGER) AS seq_num, "
                  f"upper(replace(trim(icd_code), '.', '')) AS code FROM {s} WHERE icd_code IS NOT NULL",
            "proc": f"SELECT {B('subject_id')} AS subject_id, {B('hadm_id')} AS hadm_id, {B('stay_id')} AS stay_id, {T('starttime')} AS t, "
                    f"TRY_CAST(itemid AS INTEGER) AS itemid FROM {s} WHERE itemid IN ('787541', '753461', '763189', '792843')",
            "emar": f"SELECT * FROM (SELECT subject_id, hadm_id, t, med, {N_CLS} AS cls FROM ("
                    f"SELECT {B('subject_id')} AS subject_id, {B('hadm_id')} AS hadm_id, {T('charttime')} AS t, lower(medication) AS med "
                    f"FROM {s} WHERE event_txt IN ('Confirmed', 'Applied') AND regexp_matches(lower(coalesce(medication, '')), '{N_ALL}'))) "
                    f"WHERE cls IS NOT NULL AND t IS NOT NULL AND hadm_id IS NOT NULL",
            "lab": f"SELECT {B('subject_id')} AS subject_id, {B('hadm_id')} AS hadm_id, TRY_CAST(itemid AS INTEGER) AS itemid, "
                   f"{T('charttime')} AS t, TRY_CAST(valuenum AS DOUBLE) AS v FROM {s} "
                   f"WHERE itemid IN ('100001', '100045', '100002', '100030', '100014', '100031') AND TRY_CAST(valuenum AS DOUBLE) IS NOT NULL",
        }
    else:
        cols = ["op_id", "subject_id", "hadm_id", "age", "asa", "emop", "orin_time", "orout_time", "cpbon_time", "icuin_time",
                "icuout_time", "admission_time", "discharge_time", "inhosp_death_time"]
        q = {
            "op": "SELECT " + ", ".join(f"{B(c, 'i')} AS {c}" for c in cols) + f", sex, department, antype FROM {s}",
            "dx": f"SELECT {B('subject_id', 'i')} AS subject_id, {B('chart_time', 'i')} AS t, upper(replace(trim(icd10_cm), '.', '')) AS code "
                  f"FROM {s} WHERE regexp_matches(upper(coalesce(icd10_cm, '')), '^(K2[5-8]|K92|G30|K70|K73|I85)')",
            "med": f"SELECT * FROM (SELECT subject_id, t, route, drug, atc, {I_CLS} AS cls FROM ("
                   f"SELECT {B('subject_id', 'i')} AS subject_id, {B('chart_time', 'i')} AS t, lower(coalesce(route, '')) AS route, "
                   f"lower(coalesce(drug_name, '')) AS drug, coalesce(atc_code, '') AS atc FROM {s} "
                   f"WHERE regexp_matches(lower(coalesce(drug_name, '')), '{I_ALL}') OR regexp_matches(coalesce(atc_code, ''), '^A02B[AC]'))) "
                   f"WHERE cls IS NOT NULL AND t IS NOT NULL",
            "lab": f"SELECT {B('subject_id', 'i')} AS subject_id, {B('chart_time', 'i')} AS t, item_name AS item, TRY_CAST(value AS DOUBLE) AS v "
                   f"FROM {s} WHERE item_name IN ('creatinine', 'glucose', 'ptinr', 'platelet', 'lacate') AND TRY_CAST(value AS DOUBLE) IS NOT NULL",
            "wv": f"SELECT {B('subject_id', 'i')} AS subject_id, {B('chart_time', 'i')} AS t, item_name AS item, TRY_CAST(value AS DOUBLE) AS v "
                  f"FROM {s} WHERE item_name IN ('vent', 'gcs_m') AND TRY_CAST(value AS DOUBLE) IS NOT NULL",
        }
    return q[key]


N_FL = ("""
WITH icu AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.first_careunit AS unit, i.intime, i.outtime, p.gender, p.anchor_year_group AS era,
         p.anchor_age + (year(i.intime) - p.anchor_year) AS age
  FROM n_icu i JOIN n_pat p USING (subject_id)
  WHERE i.first_careunit IN (""" + ICU_UNITS + """) AND i.intime IS NOT NULL AND i.outtime > i.intime),
icu1 AS (SELECT * FROM icu QUALIFY row_number() OVER (PARTITION BY subject_id ORDER BY intime, stay_id) = 1),
adm AS (SELECT a.hadm_id, a.admittime, a.dischtime, a.admission_type, a.admission_location,
               (upper(coalesce(a.discharge_location, '')) LIKE 'EXPIRED%' OR coalesce(a.hospital_expire_flag, '') = '1') AS died,
               CASE WHEN upper(coalesce(a.discharge_location, '')) LIKE 'EXPIRED%' OR coalesce(a.hospital_expire_flag, '') = '1'
                    THEN least(coalesce(a.deathtime, a.dischtime), a.dischtime) END AS t_death
        FROM n_adm a),
asx AS (SELECT hadm_id, t, cls, med FROM n_emar WHERE cls IN ('PPI', 'FAM', 'OTH')),
first_t AS (SELECT s.stay_id, min(x.t) AS t0 FROM icu1 s JOIN asx x ON x.hadm_id = s.hadm_id AND x.t BETWEEN s.intime AND s.outtime
            GROUP BY 1),
first_as AS (
  SELECT f.stay_id, f.t0, CASE WHEN count(DISTINCT x.cls) = 1 THEN min(x.cls) ELSE 'MIX' END AS first_cls,
         CASE WHEN bool_or(regexp_matches(x.med, '""" + ROUTE_IV + """')) THEN 'IV'
              WHEN bool_or(regexp_matches(x.med, '""" + ROUTE_EN + """')) THEN 'EN' ELSE 'UNK' END AS route0
  FROM first_t f JOIN icu1 s USING (stay_id) JOIN asx x ON x.hadm_id = s.hadm_id AND x.t = f.t0 GROUP BY 1, 2)
SELECT s.*, a.admittime, a.dischtime, a.admission_type, a.admission_location, a.died, a.t_death, f.t0, f.first_cls, f.route0,
  coalesce(s.age >= 18, false) AS adult, f.t0 IS NOT NULL AS has_as,
  EXISTS (SELECT 1 FROM asx x WHERE x.hadm_id = s.hadm_id AND x.t < f.t0 AND x.t >= f.t0 - INTERVAL 24 HOUR) AS x_prior24,
  EXISTS (SELECT 1 FROM n_emar x WHERE x.hadm_id = s.hadm_id AND x.cls IN ('HAL', 'OQ', 'APO') AND x.t <= f.t0) AS x_ap,
  EXISTS (SELECT 1 FROM n_dx d JOIN n_adm a2 ON a2.hadm_id = d.hadm_id
          WHERE d.subject_id = s.subject_id AND a2.admittime <= a.admittime AND regexp_matches(d.code, '^(F2[0-9]|F31)')) AS x_psy,
  EXISTS (SELECT 1 FROM n_dx d WHERE d.hadm_id = s.hadm_id AND d.seq_num = 1 AND """ + GI_N + """) AS x_gib,
  coalesce(f.t0 >= coalesce(a.t_death, a.dischtime), false) AS x_end
FROM icu1 s JOIN adm a USING (hadm_id) LEFT JOIN first_as f USING (stay_id)
""")
N_COHORT = """
SELECT *, CASE WHEN first_cls = 'PPI' THEN 'A' ELSE 'B' END AS arm FROM n_fl
WHERE adult AND has_as AND NOT x_prior24 AND NOT (x_ap OR x_psy) AND NOT x_gib AND first_cls IN ('PPI', 'FAM') AND NOT x_end
"""
N_COV = """
WITH b AS (SELECT * FROM n_cohort),
lab AS (
  SELECT b.stay_id,
    (SELECT l.v FROM n_lab l WHERE l.subject_id = b.subject_id AND l.itemid = 100002 AND l.t BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0
       AND l.v BETWEEN 0.1 AND 25 ORDER BY l.t DESC, l.v DESC LIMIT 1) AS scr,
    (SELECT max(l.v) FROM n_lab l WHERE l.subject_id = b.subject_id AND l.itemid = 100030 AND l.t BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0) AS inr_max,
    (SELECT min(l.v) FROM n_lab l WHERE l.subject_id = b.subject_id AND l.itemid = 100014 AND l.t BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0) AS plt_min,
    (SELECT max(l.v) FROM n_lab l WHERE l.subject_id = b.subject_id AND l.itemid = 100031 AND l.t BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0) AS lac_max
  FROM b),
cm AS (
  SELECT b.stay_id,
    max(CASE WHEN regexp_matches(d.code, '^(F01|F02|F03|G30|G310|G311|G3183)') THEN 1 ELSE 0 END) AS dementia,
    max(CASE WHEN regexp_matches(d.code, '^F10') THEN 1 ELSE 0 END) AS alcohol,
    max(CASE WHEN regexp_matches(d.code, '^(K70|K73|K74|K766|I85)') THEN 1 ELSE 0 END) AS liver
  FROM b JOIN n_dx d ON d.subject_id = b.subject_id JOIN n_adm a2 ON a2.hadm_id = d.hadm_id AND a2.admittime <= b.admittime
  GROUP BY 1)
SELECT b.stay_id, lab.scr, lab.inr_max, lab.plt_min, lab.lac_max,
  EXISTS (SELECT 1 FROM n_proc p WHERE p.hadm_id = b.hadm_id AND p.itemid IN (787541, 753461) AND p.t BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0
          AND NOT EXISTS (SELECT 1 FROM n_proc q WHERE q.hadm_id = b.hadm_id AND q.itemid = 763189 AND q.t > p.t AND q.t <= b.t0)) AS vent,
  EXISTS (SELECT 1 FROM n_proc p WHERE p.hadm_id = b.hadm_id AND p.itemid = 792843 AND p.t BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0) AS niv,
  EXISTS (SELECT 1 FROM n_emar x WHERE x.hadm_id = b.hadm_id AND x.cls = 'SED' AND x.t BETWEEN b.t0 - INTERVAL 6 HOUR AND b.t0) AS sed6,
  EXISTS (SELECT 1 FROM n_emar x WHERE x.hadm_id = b.hadm_id AND x.cls = 'OPI' AND x.t BETWEEN b.t0 - INTERVAL 6 HOUR AND b.t0) AS opi6,
  EXISTS (SELECT 1 FROM n_emar x WHERE x.hadm_id = b.hadm_id AND x.cls = 'PRES' AND x.t BETWEEN b.t0 - INTERVAL 6 HOUR AND b.t0) AS pres6,
  coalesce(cm.dementia, 0) AS dementia, coalesce(cm.alcohol, 0) AS alcohol, coalesce(cm.liver, 0) AS liver,
  date_diff('minute', b.intime, b.t0) / 60.0 AS h_icu,
  date_diff('minute', b.t0, least(b.outtime, coalesce(b.t_death, b.outtime))) / 60.0 AS t_end_icu,
  coalesce(b.died AND b.t_death <= b.outtime, false) AS end_death_icu,
  date_diff('minute', b.t0, coalesce(b.t_death, b.dischtime)) / 60.0 AS t_end_hosp,
  coalesce(b.died, false) AS end_death_hosp
FROM b LEFT JOIN lab USING (stay_id) LEFT JOIN cm USING (stay_id)
"""
I_FL = """
WITH o AS (
  SELECT * FROM i_op
  WHERE icuin_time IS NOT NULL AND icuout_time > icuin_time AND orout_time IS NOT NULL
    AND icuin_time BETWEEN orout_time - 60 AND orout_time + 1440 AND admission_time IS NOT NULL AND discharge_time IS NOT NULL),
asx AS (SELECT subject_id, t, cls, route FROM i_med WHERE cls IN ('PPI', 'FAM', 'OTH')),
first_t AS (SELECT o.op_id, min(x.t) AS t0 FROM o JOIN asx x ON x.subject_id = o.subject_id AND x.t BETWEEN o.icuin_time AND o.icuout_time
            GROUP BY 1),
first_as AS (
  SELECT f.op_id, f.t0, CASE WHEN count(DISTINCT x.cls) = 1 THEN min(x.cls) ELSE 'MIX' END AS first_cls,
         CASE WHEN bool_or(x.route = 'iv') THEN 'IV' WHEN bool_or(x.route = 'po') THEN 'EN' ELSE 'UNK' END AS route0
  FROM first_t f JOIN o USING (op_id) JOIN asx x ON x.subject_id = o.subject_id AND x.t = f.t0 GROUP BY 1, 2)
SELECT o.*, f.t0, f.first_cls, f.route0, coalesce(o.age >= 18, false) AS adult, f.t0 IS NOT NULL AS has_as,
  EXISTS (SELECT 1 FROM asx x WHERE x.subject_id = o.subject_id AND x.t < f.t0 AND x.t >= f.t0 - 1440) AS x_prior24,
  EXISTS (SELECT 1 FROM i_med m WHERE m.subject_id = o.subject_id AND m.cls IN ('HAL', 'OQ', 'APO', 'PSYRX') AND m.t <= f.t0) AS x_ap,
  false AS x_psy,
  EXISTS (SELECT 1 FROM i_dx d WHERE d.subject_id = o.subject_id AND d.t BETWEEN o.admission_time AND o.discharge_time AND """ + GI_I + """) AS x_gib,
  coalesce(f.t0 >= least(o.discharge_time, coalesce(o.inhosp_death_time, o.discharge_time)), false) AS x_end
FROM o LEFT JOIN first_as f USING (op_id)
"""
I_COHORT = """
SELECT *, CASE WHEN first_cls = 'PPI' THEN 'A' ELSE 'B' END AS arm FROM i_fl
WHERE adult AND has_as AND NOT x_prior24 AND NOT x_ap AND NOT x_gib AND first_cls IN ('PPI', 'FAM') AND NOT x_end
QUALIFY row_number() OVER (PARTITION BY subject_id ORDER BY t0, op_id) = 1
"""
I_COV = """
WITH b AS (SELECT * FROM i_cohort)
SELECT b.op_id,
  (SELECT l.v FROM i_lab l WHERE l.subject_id = b.subject_id AND l.item = 'creatinine' AND l.t BETWEEN b.t0 - 1440 AND b.t0
     AND l.v BETWEEN 0.1 AND 25 ORDER BY l.t DESC, l.v DESC LIMIT 1) AS scr,
  (SELECT max(l.v) FROM i_lab l WHERE l.subject_id = b.subject_id AND l.item = 'ptinr' AND l.t BETWEEN b.t0 - 1440 AND b.t0) AS inr_max,
  (SELECT min(l.v) FROM i_lab l WHERE l.subject_id = b.subject_id AND l.item = 'platelet' AND l.t BETWEEN b.t0 - 1440 AND b.t0) AS plt_min,
  (SELECT max(l.v) FROM i_lab l WHERE l.subject_id = b.subject_id AND l.item = 'lacate' AND l.t BETWEEN b.t0 - 1440 AND b.t0) AS lac_max,
  (SELECT w.v FROM i_wv w WHERE w.subject_id = b.subject_id AND w.item = 'vent' AND w.t BETWEEN b.t0 - 360 AND b.t0
     ORDER BY w.t DESC, w.v DESC LIMIT 1) AS vent_last,
  (SELECT w.v FROM i_wv w WHERE w.subject_id = b.subject_id AND w.item = 'gcs_m' AND w.t BETWEEN b.t0 - 360 AND b.t0
     ORDER BY w.t DESC, w.v ASC LIMIT 1) AS gcsm_last,
  EXISTS (SELECT 1 FROM i_med m WHERE m.subject_id = b.subject_id AND m.cls = 'SED' AND m.t BETWEEN b.t0 - 360 AND b.t0) AS sed6,
  EXISTS (SELECT 1 FROM i_med m WHERE m.subject_id = b.subject_id AND m.cls = 'OPI' AND m.t BETWEEN b.t0 - 360 AND b.t0) AS opi6,
  EXISTS (SELECT 1 FROM i_med m WHERE m.subject_id = b.subject_id AND m.cls = 'PRES' AND m.t BETWEEN b.t0 - 360 AND b.t0) AS pres6,
  EXISTS (SELECT 1 FROM i_dx d WHERE d.subject_id = b.subject_id AND d.t <= b.t0 AND regexp_matches(d.code, '^G30')) AS dementia,
  EXISTS (SELECT 1 FROM i_dx d WHERE d.subject_id = b.subject_id AND d.t <= b.t0 AND regexp_matches(d.code, '^(K70|K73|I85)')) AS liver,
  (b.orout_time - b.orin_time) / 60.0 AS or_h, b.cpbon_time IS NOT NULL AS cpb, (b.t0 - b.orout_time) / 60.0 AS h_or,
  (least(b.icuout_time, coalesce(b.inhosp_death_time, b.icuout_time)) - b.t0) / 60.0 AS t_end_icu,
  coalesce(b.inhosp_death_time <= b.icuout_time, false) AS end_death_icu,
  (least(b.discharge_time, coalesce(b.inhosp_death_time, b.discharge_time)) - b.t0) / 60.0 AS t_end_hosp,
  b.inhosp_death_time IS NOT NULL AS end_death_hosp
FROM b
"""
N_OUT = """
WITH b AS (SELECT * FROM n_cohort),
ap AS (SELECT b.stay_id, min(date_diff('minute', b.t0, x.t)) / 60.0 AS t_ap,
              min(CASE WHEN x.cls = 'OQ' THEN date_diff('minute', b.t0, x.t) END) / 60.0 AS t_oq
       FROM b JOIN n_emar x ON x.hadm_id = b.hadm_id AND x.cls IN ('HAL', 'OQ', 'APO') AND x.t > b.t0 AND x.t <= b.t0 + INTERVAL 168 HOUR
       GROUP BY 1),
hy AS (SELECT b.stay_id, min(date_diff('minute', b.t0, l.t)) / 60.0 AS t_hypo
       FROM b JOIN n_lab l ON l.subject_id = b.subject_id AND l.itemid IN (100001, 100045) AND l.v > 0 AND l.v < 70
        AND l.t > b.t0 AND l.t <= b.t0 + INTERVAL 168 HOUR
       GROUP BY 1)
SELECT b.stay_id, ap.t_ap, ap.t_oq, hy.t_hypo FROM b LEFT JOIN ap USING (stay_id) LEFT JOIN hy USING (stay_id)
"""
I_OUT = """
WITH b AS (SELECT * FROM i_cohort),
ap AS (SELECT b.op_id, (m.t - b.t0) / 60.0 AS dt, m.cls
       FROM b JOIN i_med m ON m.subject_id = b.subject_id AND m.cls IN ('HAL', 'OQ', 'APO') AND m.t > b.t0 AND m.t <= b.t0 + 10080
       WHERE NOT (m.cls = 'HAL' AND EXISTS (SELECT 1 FROM i_op o WHERE o.subject_id = b.subject_id
                                            AND m.t >= o.orin_time AND m.t <= o.orout_time + 360))),
apm AS (SELECT op_id, min(dt) AS t_ap, min(CASE WHEN cls = 'OQ' THEN dt END) AS t_oq FROM ap GROUP BY 1),
hy AS (SELECT b.op_id, min(l.t - b.t0) / 60.0 AS t_hypo
       FROM b JOIN i_lab l ON l.subject_id = b.subject_id AND l.item = 'glucose' AND l.v > 0 AND l.v < 70 AND l.t > b.t0 AND l.t <= b.t0 + 10080
       GROUP BY 1)
SELECT b.op_id, apm.t_ap, apm.t_oq, hy.t_hypo FROM b LEFT JOIN apm USING (op_id) LEFT JOIN hy USING (op_id)
"""
DESIGN_SQL = {"n": [("fl", N_FL), ("cohort", N_COHORT), ("cov", N_COV)], "i": [("fl", I_FL), ("cohort", I_COHORT), ("cov", I_COV)]}
OUT_SQL = {"n": N_OUT, "i": I_OUT}
EXPECT = {"n_cov": "stay_id scr inr_max plt_min lac_max vent niv sed6 opi6 pres6 dementia alcohol liver h_icu t_end_icu end_death_icu "
                   "t_end_hosp end_death_hosp",
          "i_cov": "op_id scr inr_max plt_min lac_max vent_last gcsm_last sed6 opi6 pres6 dementia liver or_h cpb h_or t_end_icu "
                   "end_death_icu t_end_hosp end_death_hosp",
          "n_out": "stay_id t_ap t_oq t_hypo", "i_out": "op_id t_ap t_oq t_hypo"}


def priv(args):
    P = os.path.abspath(os.path.expanduser(args.private))
    os.makedirs(P, exist_ok=True)
    return P


def pq(P, name):
    return os.path.join(P, f"{name}.parquet")


def con_for(P):
    import duckdb
    tmp = os.path.join(P, "duck_tmp")
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = 2; SET memory_limit = '2GB'; SET temp_directory = '{tmp}'; SET preserve_insertion_order = false")
    return con


def views(con, P, db):
    for name in list(FILES[db]) + ["fl", "cohort", "cov", "out"]:
        f = pq(P, f"{db}_{name}")
        if os.path.exists(f):
            con.execute(f"CREATE OR REPLACE VIEW {db}_{name} AS SELECT * FROM read_parquet('{f}')")


def num(s):
    return pd.to_numeric(pd.Series(s), errors="coerce").astype(float).values


def flag(s):
    v = pd.Series(s)
    if v.dtype == bool or v.dtype == object or str(v.dtype) == "boolean":
        return v.fillna(False).astype(bool).values
    return num(v) == 1


def ckd_epi(scr, age, female):
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = scr / k
        e = 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.2 * 0.9938 ** age * np.where(female, 1.012, 1.0)
    return np.where(np.isfinite(scr), e, np.nan)


def design_matrix(df, db):
    cols, names = [], []

    def add(name, v):
        cols.append(np.asarray(v, float))
        names.append(name)

    age = num(df["age"])
    add("age_lt50", age < 50)
    add("age_ge65", age >= 65)
    add("age_ge80", age >= 80)
    sexcol = "gender" if db == "n" else "sex"
    female = df[sexcol].astype(str).str.upper().str[:1].values == "F"
    add("female", female)
    egfr = ckd_epi(num(df["scr"]), age, female)
    miss = ~np.isfinite(egfr)
    e = np.where(miss, 90.0, egfr)
    add("egfr_missing", miss)
    add("egfr_per10", e / 10)
    for k in (30, 60, 90):
        add(f"egfr_over{k}_per10", np.maximum(e - k, 0) / 10)
    inr, plt, lac = num(df["inr_max"]), num(df["plt_min"]), num(df["lac_max"])
    add("coag", (inr >= 1.5) | (plt < 50))
    add("coag_not_measured", ~np.isfinite(inr) & ~np.isfinite(plt))
    add("lactate_ge2", lac >= 2)
    add("lactate_ge4", lac >= 4)
    add("lactate_not_measured", ~np.isfinite(lac))
    r = df["route0"].astype(str).values
    add("route_enteral", r == "EN")
    add("route_unknown", r == "UNK")
    if db == "n":
        for c in ("vent", "niv", "sed6", "opi6", "pres6"):
            add(c, flag(df[c]))
        u = df["unit"].astype(str).values
        for g, us in (("MICU", ["MICU"]), ("NSICU", ["NSICU"]), ("CICU", ["CICU", "CCU"]), ("SICU", ["SICU"]), ("CTICU", ["CTICU"])):
            add(f"unit_{g}", np.isin(u, us))
        add("elective", np.isin(df["admission_type"].astype(str).str.upper().values, ["ELECTIVE", "ELECTIVE-ROUTINE"]))
        add("from_surgery", df["admission_location"].astype(str).str.upper().values == "SURGERY")
        add("era_2017_2019", df["era"].astype(str).values == "2017 - 2019")
        h = num(df["h_icu"])
        add("h_ge6", h >= 6)
        add("h_ge24", h >= 24)
        add("h_ge72", h >= 72)
        for c in ("dementia", "alcohol", "liver"):
            add(c, flag(df[c]))
        extras = {"age_years": age, "egfr": egfr, "h_icu": h}
    else:
        asa = num(df["asa"])
        add("asa_ge3", asa >= 3)
        add("asa_ge4", asa >= 4)
        add("asa_ge5", asa >= 5)
        add("asa_missing", ~np.isfinite(asa))
        add("emergency", num(df["emop"]) == 1)
        dep = df["department"].astype(str).values
        for g in ("GS", "NS", "OS"):
            add(f"dept_{g}", dep == g)
        add("dept_other", ~np.isin(dep, ["CTS", "GS", "NS", "OS"]))
        add("non_general", df["antype"].astype(str).values != "General")
        add("cpb", flag(df["cpb"]))
        orh = num(df["or_h"])
        add("or_ge4h", orh >= 4)
        add("or_ge8h", orh >= 8)
        h = num(df["h_or"])
        add("h_or_ge6", h >= 6)
        add("h_or_ge24", h >= 24)
        add("h_or_ge48", h >= 48)
        add("h_or_per6", h / 6)
        for k in (6, 24):
            add(f"h_or_over{k}_per6", np.maximum(h - k, 0) / 6)
        v, g = num(df["vent_last"]), num(df["gcsm_last"])
        add("vent", v == 1)
        add("vent_unknown", ~np.isfinite(v))
        add("gcs_motor_lt6", g < 6)
        add("gcs_unknown", ~np.isfinite(g))
        for c in ("sed6", "opi6", "pres6", "dementia", "liver"):
            add(c, flag(df[c]))
        extras = {"age_years": age, "egfr": egfr, "h_or": h}
    return np.column_stack(cols), names, extras


def build(df, db, outcome=False):
    X, names, extras = design_matrix(df, db)
    X, names, dropped = asd.prevalence_filter(X, names)
    n = len(df)
    d = {"X": X, "names": names, "dropped": dropped, "extras": extras, "A": (df["arm"].values == "A").astype(int), "site": None,
         "t_end_icu": num(df["t_end_icu"]), "end_death_icu": flag(df["end_death_icu"]),
         "t_end_hosp": num(df["t_end_hosp"]), "end_death_hosp": flag(df["end_death_hosp"]), "t_dev": np.full(n, np.nan), "sub": {}}
    d["t_end"], d["end_death"] = d["t_end_icu"], d["end_death_icu"]
    if outcome:
        for c in ("t_ap", "t_oq", "t_hypo"):
            d[c] = num(df[c])
    return d


def load(P, db, outcome=False):
    con = con_for(P)
    try:
        views(con, P, db)
        idc = IDCOL[db]
        sel = "c.*, v.* EXCLUDE (" + idc + ")" + (", o.t_ap, o.t_oq, o.t_hypo" if outcome else "")
        frm = db + "_cohort c JOIN " + db + "_cov v USING (" + idc + ")" + (" JOIN " + db + "_out o USING (" + idc + ")" if outcome else "")
        return con.execute("SELECT " + sel + " FROM " + frm + " ORDER BY c." + idc).df()
    finally:
        con.close()


def flow_counts(fl, db, n_patients, nA, nB):
    def b(c):
        return fl[c].astype("boolean").fillna(False).astype(bool)
    adult, has = b("adult"), b("has_as")
    top = fl[adult & has]
    x1 = top["x_prior24"].astype("boolean").fillna(False).astype(bool)
    x2 = ~x1 & (top["x_ap"].astype("boolean").fillna(False).astype(bool) | top["x_psy"].astype("boolean").fillna(False).astype(bool))
    x3 = ~x1 & ~x2 & top["x_gib"].astype("boolean").fillna(False).astype(bool)
    x4 = ~x1 & ~x2 & ~x3 & ~top["first_cls"].isin(["PPI", "FAM"])
    x5 = ~x1 & ~x2 & ~x3 & ~x4 & top["x_end"].astype("boolean").fillna(False).astype(bool)
    el = ~(x1 | x2 | x3 | x4 | x5)
    return {"before_top": int(adult.sum()), "top": int(len(top)), "x1_prior24": int(x1.sum()), "x2_antipsychotic_or_psych": int(x2.sum()),
            "x3_bleeding": int(x3.sum()), "x4_other_first_drug": int(x4.sum()), "x5_time_zero_at_end": int(x5.sum()),
            "eligible": int(el.sum()), "patients": int(n_patients), "A": int(nA), "B": int(nB)}


def extract(args):
    t0 = time.time()
    P = priv(args)
    con = con_for(P)
    try:
        for key in PARTS[args.db][args.part]:
            path = pq(P, f"{args.db}_{key}")
            asd.write_pq(con, extract_sql(args.db, key), path)
            n = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
            asd.status(f"{args.db}_{key}: {sc.mask_n(n)} rows (masked)", t0)
    finally:
        con.close()
    asd.log_run(args.results, f"r9 extract --db {args.db} --part {args.part}: ok ({time.time() - t0:.0f} s)")


def design(args):
    t0 = time.time()
    P, db = priv(args), args.db
    if os.path.exists(pq(P, f"{db}_out")):
        raise SystemExit("outcome table exists: stage 1 is outcome-free and runs before it")
    con = con_for(P)
    try:
        views(con, P, db)
        for name, sql in DESIGN_SQL[db]:
            asd.write_pq(con, sql, pq(P, f"{db}_{name}"))
            views(con, P, db)
        fl = con.execute(f"SELECT adult, has_as, x_prior24, x_ap, x_psy, x_gib, x_end, first_cls FROM {db}_fl").df()
        nA, nB = con.execute(f"SELECT count(*) FILTER (WHERE arm = 'A'), count(*) FILTER (WHERE arm = 'B') FROM {db}_cohort").fetchone()
    finally:
        con.close()
    flow = flow_counts(fl, db, nA + nB, nA, nB)
    df = load(P, db)
    d = build(df, db)
    X, names, A = d["X"], d["names"], d["A"]
    ps, _ = sc.ps_fit(X, A)
    ow = sc.overlap_w(ps, A)
    essA, essB = sc.ess(ow[A == 1]), sc.ess(ow[A == 0])
    gate = {"nA": int(A.sum()), "nB": int((A == 0).sum()), "weights_finite": bool(np.isfinite(ow).all()),
            "essA": round(essA, 1), "essB": round(essB, 1)}
    gate["pass"] = bool(gate["nA"] >= 100 and gate["nB"] >= 100 and gate["weights_finite"] and essA >= 50 and essB >= 50)
    Xe = np.column_stack([X] + [np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 0) for v in d["extras"].values()])
    ne = names + [f"{k} (continuous)" for k in d["extras"]]
    un, so = sc.smd_table(Xe, ne, A), sc.smd_table(Xe, ne, A, ow)
    prev = {}
    for j, nm in enumerate(names):
        v = X[:, j]
        if set(np.unique(v).tolist()) <= {0.0, 1.0}:
            prev[nm] = {arm: f"{sc.mask_n(int(v[A == a].sum()))} ({sc.pct_text(int(v[A == a].sum()), int((A == a).sum()))})"
                        for arm, a in (("A", 1), ("B", 0))}
    fu = {}
    for kind in ("icu", "hosp"):
        te, dth = d[f"t_end_{kind}"], d[f"end_death_{kind}"]
        for h in (24, 72, 168):
            fu[f"{kind}: still followed at {h} h"] = sc.mask_n(int((te > h).sum()))
        fu[f"{kind}: died before 168 h"] = sc.mask_n(int(((te <= 168) & dth).sum()))
        fu[f"{kind}: discharged alive before 168 h"] = sc.mask_n(int(((te <= 168) & ~dth).sum()))
    out = {"flow": {k: sc.mask_n(v) for k, v in flow.items()}, "columns": names, "columns_dropped_low_prevalence": d["dropped"],
           "ps_quantiles": {"A": asd.q(ps[A == 1]), "B": asd.q(ps[A == 0])}, "ess_overlap": {"A": round(essA, 1), "B": round(essB, 1)},
           "smd": {nm: [round(un[nm], 3), round(so[nm], 3)] for nm in ne},
           "max_abs_smd": {"unweighted": round(max(abs(v) for v in un.values()), 3), "overlap": round(max(abs(v) for v in so.values()), 3)},
           "prevalence": prev, "followup_both_arms": fu,
           "gate": {"nA": sc.mask_n(gate["nA"]), "nB": sc.mask_n(gate["nB"]), "weights_finite": gate["weights_finite"],
                    "essA": gate["essA"], "essB": gate["essB"], "pass": gate["pass"]}}
    R = os.path.expanduser(args.results)
    jf = os.path.join(R, "r9_design.json")
    allj = json.load(open(jf, encoding="utf-8")) if os.path.exists(jf) else {}
    allj[db] = out
    json.dump(allj, open(jf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(R, "r9_design.md"), "w", encoding="utf-8").write(design_md(allj))
    asd.log_run(args.results, f"r9 design --db {db}: ok, gate {'passed' if gate['pass'] else 'NOT passed'} ({time.time() - t0:.0f} s)")
    asd.status(f"design {db}: gate {'passed' if gate['pass'] else 'NOT passed'}", t0)


FLOW_LAB = {"n": [("before_top", "Adults at their first ICU stay (ICU units)"), ("top", "of whom started an acid suppressant in the ICU"), ("x1_prior24", "Excluded: acid suppressant in the 24 h before time zero"),
                  ("x2_antipsychotic_or_psych", "Excluded: antipsychotic before time zero, or schizophrenia-spectrum or bipolar disorder diagnosis"),
                  ("x3_bleeding", "Excluded: principal diagnosis upper GI bleeding or peptic ulcer"), ("x4_other_first_drug", "Excluded: first was another H2 blocker, or both classes at once"),
                  ("x5_time_zero_at_end", "Excluded: time zero not before discharge or death (inconsistent records)"), ("eligible", "Eligible (ICU stays)"), ("patients", "Patients analysed"),
                  ("A", "A (PPI)"), ("B", "B (famotidine)")],
            "i": [("before_top", "Adult surgical cases admitted to the ICU within 24 h after leaving the operating room"), ("top", "of which with an acid suppressant recorded in the ICU"),
                  ("x1_prior24", "Excluded: acid suppressant in the 24 h before time zero (including during surgery)"), ("x2_antipsychotic_or_psych", "Excluded: antipsychotic, clozapine or lithium recorded before time zero"),
                  ("x3_bleeding", "Excluded: peptic ulcer or GI bleeding diagnosis in the admission"), ("x4_other_first_drug", "Excluded: first was another acid suppressant, or both classes at once"),
                  ("x5_time_zero_at_end", "Excluded: time zero not before discharge or death (inconsistent records)"), ("eligible", "Eligible (cases)"),
                  ("patients", "Patients analysed (first eligible case of each)"), ("A", "A (PPI)"), ("B", "B (famotidine)")]}


def design_md(allj):
    L = ["# Revision R9 stage 1 (no outcomes): NWICU and INSPIRE", "", "Plan: SAP_addendum_R9.md v1.0. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", ""]
    for db in ("n", "i"):
        o = allj.get(db)
        if not o:
            continue
        g = o["gate"]
        L += [f"## {DBNAME[db]}", "", "| Step | Number |", "|---|---|"] + [f"| {t} | {o['flow'][k]} |" for k, t in FLOW_LAB[db]]
        L += ["", f"- Gate: A {g['nA']}, B {g['nB']}; weights finite {g['weights_finite']}; effective sample size with overlap weights A {g['essA']}, B {g['essB']}; "
              f"{'passed, outcomes extracted' if g['pass'] else 'not passed, stage 1 reported only'}",
              f"- Model columns: {', '.join(o['columns'])}", f"- Columns dropped for low prevalence: {', '.join(o['columns_dropped_low_prevalence']) or 'no'}",
              f"- Propensity score quantiles (0, 1, 5, 25, 50, 75, 95, 99, 100) A: {o['ps_quantiles']['A']}; B: {o['ps_quantiles']['B']}",
              f"- Largest |SMD|: unweighted {o['max_abs_smd']['unweighted']}, overlap weighted {o['max_abs_smd']['overlap']}", "",
              "| Variable | A number (%) | B number (%) | SMD unweighted | SMD overlap weighted |", "|---|---|---|---|---|"]
        for nm, (u, w) in o["smd"].items():
            p = o["prevalence"].get(nm, {"A": "", "B": ""})
            L.append(f"| {nm} | {p['A']} | {p['B']} | {u:+.3f} | {w:+.3f} |")
        L += ["", "Follow-up (both arms):", ""] + [f"- {k}: {v}" for k, v in o["followup_both_arms"].items()] + [""]
    return "\n".join(L)


def outcome(args):
    t0 = time.time()
    P, db = priv(args), args.db
    jf = os.path.join(os.path.expanduser(args.results), "r9_design.json")
    g = json.load(open(jf, encoding="utf-8")).get(db, {}).get("gate", {}) if os.path.exists(jf) else {}
    if not g.get("pass"):
        raise SystemExit(f"{db}: stage 1 missing or gate not passed: no outcome is extracted (addendum section 6)")
    con = con_for(P)
    try:
        views(con, P, db)
        asd.write_pq(con, OUT_SQL[db], pq(P, f"{db}_out"))
    finally:
        con.close()
    asd.log_run(args.results, f"r9 outcome --db {db}: ok ({time.time() - t0:.0f} s)")
    asd.status(f"outcome {db}: done", t0)


def analyses(d, cache):
    A = d["A"]
    ow = sc.overlap_w(asd.ow_fit(d, cache, "full"), A)
    di = dict(d, t_end=d["t_end_icu"], end_death=d["end_death_icu"])
    dh = dict(d, t_end=d["t_end_hosp"], end_death=d["end_death_hosp"])
    return {"icu": asd.cif_contrast(di, ow, d["t_ap"])[168], "icu_oq": asd.cif_contrast(di, ow, d["t_oq"])[168],
            "hosp": asd.cif_contrast(dh, ow, d["t_ap"])[168], "hosp_oq": asd.cif_contrast(dh, ow, d["t_oq"])[168],
            "negctl": asd.cif_contrast(di, ow, d["t_hypo"])[168]}


def estimate(args):
    t0 = time.time()
    P = priv(args)
    for db in ("n", "i"):
        if not os.path.exists(pq(P, f"{db}_out")):
            asd.status(f"{db}: no outcome table (gate not passed or not run)", t0)
            continue
        d = build(load(P, db, outcome=True), db, outcome=True)
        cache = {}
        res = analyses(d, cache)
        one = np.ones(len(d["A"]))
        crude = {"icu": asd.cif_contrast(dict(d, t_end=d["t_end_icu"], end_death=d["end_death_icu"]), one, d["t_ap"])[168]["OR"],
                 "hosp": asd.cif_contrast(dict(d, t_end=d["t_end_hosp"], end_death=d["end_death_hosp"]), one, d["t_ap"])[168]["OR"]}
        json.dump({"res": asd.to_json(res), "crude": asd.to_json(crude)}, open(os.path.join(P, f"r9_est_{db}.json"), "w"))
        np.savez(os.path.join(P, f"r9_cache_{db}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
        asd.status(f"estimate {db} done", t0)
    asd.log_run(args.results, f"r9 estimate: ok ({time.time() - t0:.0f} s)")


_G = {}


def _boot_one(rep):
    try:
        b = asd.resample(_G["d"], "m", np.random.default_rng(asd.SEED + rep))
        r = analyses(b, dict(_G["cache"]))
    except Exception:
        return rep, None
    lor = [math.log(r[n]["OR"]) if np.isfinite(r[n].get("OR", np.nan)) and r[n]["OR"] > 0 else np.nan for n in NAMES]
    rd = [r[n]["RD"] if np.isfinite(r[n].get("RD", np.nan)) else np.nan for n in NAMES]
    return rep, (lor, rd)


def boot(args):
    t0 = time.time()
    P, db = priv(args), args.db
    d = build(load(P, db, outcome=True), db, outcome=True)
    z = np.load(os.path.join(P, f"r9_cache_{db}.npz"))
    _G.update(d=d, cache={k.replace("__", ":"): z[k] for k in z.files})
    reps = list(range(args.start, min(args.stop, asd.NBOOT)))
    out = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for rep, r in pool.imap_unordered(_boot_one, reps, chunksize=10):
                out[rep] = r
    else:
        for rep in reps:
            out[rep] = _boot_one(rep)[1]
    lor = np.full((len(reps), len(NAMES)), np.nan)
    rd = np.full((len(reps), len(NAMES)), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None:
            lor[i], rd[i] = out[rep]
    np.savez(os.path.join(P, f"r9_boot_{db}_{args.start:04d}_{args.stop:04d}.npz"), reps=np.array(reps), lor=lor, rd=rd)
    failed = sum(1 for r in reps if out.get(r) is None)
    asd.status(f"boot {db} {args.start}-{args.stop} done, failed {failed}", t0)
    asd.log_run(args.results, f"r9 boot --db {db} --start {args.start} --stop {args.stop}: ok, failed {failed} ({time.time() - t0:.0f} s)")


def load_boot(P, db):
    files = sorted(glob.glob(os.path.join(P, f"r9_boot_{db}_*.npz")))
    if not files:
        return None
    reps, lor, rd = [], [], []
    for f in files:
        z = np.load(f)
        reps.append(z["reps"])
        lor.append(z["lor"])
        rd.append(z["rd"])
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return {"lor": np.concatenate(lor)[first], "rd": np.concatenate(rd)[first], "n": int(first.size)}


def summarise(res, bt):
    out = {}
    for i, n in enumerate(NAMES):
        e = dict(res[n])
        if bt is not None:
            lor, rd = bt["lor"][:, i], bt["rd"][:, i]
            lor, rd = lor[np.isfinite(lor)], rd[np.isfinite(rd)]
            if lor.size >= asd.MIN_BOOT:
                e["OR_lo"], e["OR_hi"] = math.exp(np.percentile(lor, 2.5)), math.exp(np.percentile(lor, 97.5))
                e["se_logOR"], e["boot_ok"] = float(lor.std(ddof=1)), int(lor.size)
            if rd.size >= asd.MIN_BOOT:
                e["RD_lo"], e["RD_hi"] = float(np.percentile(rd, 2.5)), float(np.percentile(rd, 97.5))
        out[n] = e
    return out


def verdict(e):
    if not e or e.get("OR_lo") is None:
        return "no interval"
    if e["OR_hi"] < 1:
        return "proxy outcome supports fewer events in the PPI arm"
    if e["OR_lo"] > 1:
        return "proxy outcome in the opposite direction"
    return "proxy outcome inconclusive"


def nc_reading(e):
    if not e or e.get("OR_lo") is None:
        return "no interval"
    return "negative control suggests residual confounding" if (e["OR_hi"] < 1 or e["OR_lo"] > 1) else "negative control does not suggest residual confounding"


def pool_list(items):
    ok = [(lab, o, s) for lab, o, s in items if o is not None and s is not None and o > 0 and s > 0 and np.isfinite(s)]
    if len(ok) < 2:
        return None
    m, se, Q, I2, k = sc.pool_fe([math.log(o) for _, o, _ in ok], [s for _, _, s in ok])
    return {"members": [lab for lab, _, _ in ok], "OR": math.exp(m), "OR_lo": math.exp(m - 1.96 * se), "OR_hi": math.exp(m + 1.96 * se),
            "I2": I2, "Q": Q, "k": k, "log": m, "se_log": se}


def setting_ratio(icu, surg):
    if not icu or not surg:
        return None
    lr, se = surg["log"] - icu["log"], math.sqrt(icu["se_log"] ** 2 + surg["se_log"] ** 2)
    r = {"ratio": math.exp(lr), "lo": math.exp(lr - 1.96 * se), "hi": math.exp(lr + 1.96 * se)}
    r["reading"] = "the proxy-outcome difference differs between the post-surgical and ICU cohorts" if (r["hi"] < 1 or r["lo"] > 1) else "no evidence that the post-surgical cohorts differ"
    return r


def external(R):
    r7 = json.load(open(os.path.join(R, "r7.json"), encoding="utf-8"))["per_key"]
    mv = json.load(open(os.path.join(R, "e2_mover.json"), encoding="utf-8"))["primary"]
    ext = {"MIMIC-IV icu": r7["c2m"]["px_icu"], "eICU icu": r7["c2e"]["px_icu"], "MIMIC-IV hosp": r7["c2m"]["px_hosp"],
           "MOVER hosp": {"OR": mv["OR"], "OR_lo": mv["OR_lo"], "OR_hi": mv["OR_hi"],
                          "se_logOR": (math.log(mv["OR_hi"]) - math.log(mv["OR_lo"])) / 3.92}}
    return {k: {x: v.get(x) for x in ("OR", "OR_lo", "OR_hi", "se_logOR")} for k, v in ext.items()}


def report(args):
    P, R = priv(args), os.path.expanduser(args.results)
    out = {"plan": "SAP_addendum_R9.md v1.0", "boot_reps": {}, "per_db": {}, "crude_OR": {}, "reading": {}, "pooled": {}}
    S = {}
    for db in ("n", "i"):
        f = os.path.join(P, f"r9_est_{db}.json")
        if not os.path.exists(f):
            continue
        est = json.load(open(f))
        bt = load_boot(P, db)
        out["boot_reps"][DBNAME[db]] = bt["n"] if bt else 0
        S[db] = summarise(est["res"], bt)
        out["per_db"][DBNAME[db]] = S[db]
        out["crude_OR"][DBNAME[db]] = est["crude"]
        out["reading"][DBNAME[db]] = {n: (nc_reading(S[db][n]) if n == "negctl" else verdict(S[db][n])) for n in NAMES}
    ext = external(R)
    out["external"] = ext

    def item(lab, e):
        return (lab, e.get("OR") if e else None, e.get("se_logOR") if e else None)

    icu = [item("MIMIC-IV", ext["MIMIC-IV icu"]), item("eICU", ext["eICU icu"])] + \
          [item(DBNAME[db], S[db]["icu"]) for db in ("n", "i") if db in S]
    hosp = [item("MIMIC-IV", ext["MIMIC-IV hosp"])] + [item(DBNAME[db], S[db]["hosp"]) for db in ("n", "i") if db in S] + \
           [item("MOVER", ext["MOVER hosp"])]
    out["pooled"]["icu"] = pool_list(icu)
    out["pooled"]["hosp"] = pool_list(hosp)
    icu_c = pool_list([item("MIMIC-IV", ext["MIMIC-IV hosp"])] + ([item("NWICU", S["n"]["hosp"])] if "n" in S else []))
    surg = pool_list(([item("INSPIRE", S["i"]["hosp"])] if "i" in S else []) + [item("MOVER", ext["MOVER hosp"])])
    out["pooled"]["hosp_icu_cohorts"], out["pooled"]["hosp_surgical"] = icu_c, surg
    out["setting_ratio"] = setting_ratio(icu_c, surg)
    for k, p in out["pooled"].items():
        if p:
            out["reading"][f"pooled:{k}"] = asd.interpret(p["OR"], p["OR_lo"], p["OR_hi"])
    masked = mask(out)
    json.dump(asd.round_floats(masked), open(os.path.join(R, "r9.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(R, "r9.md"), "w", encoding="utf-8").write(md(masked))
    asd.log_run(args.results, "r9 report: ok")
    print("report written", flush=True)


def mask(o, key=""):
    if isinstance(o, dict):
        return {k: mask(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [mask(v, key) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)) and key in ("eventsA", "eventsB", "nA", "nB"):
        return sc.mask_n(int(o))
    return o


def orci(e):
    if not e or e.get("OR") is None:
        return "—"
    s = asd.fmt(e["OR"])
    return s + (f" ({asd.fmt(e['OR_lo'])}–{asd.fmt(e['OR_hi'])})" if e.get("OR_lo") is not None else "")


def md(o):
    lab = {"icu": "First antipsychotic in the ICU (primary)", "icu_oq": "First olanzapine or quetiapine in the ICU", "hosp": "First antipsychotic in the hospital", "hosp_oq": "First olanzapine or quetiapine in the hospital",
           "negctl": "Negative control: first glucose <70 mg/dL in the ICU"}
    L = ["# Revision R9 results: antipsychotic proxy outcome in NWICU and INSPIRE (PPI vs famotidine)", "",
         f"Plan: {o['plan']}. Bootstrap replicates: {', '.join(f'{k} {v}' for k, v in o['boot_reps'].items())} (by patient). Overlap weighting; "
         "168-hour cumulative incidence; the ICU version with ICU discharge alive and death as competing events, the hospital version with hospital discharge alive and death as competing events. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", ""]
    for name, r in o["per_db"].items():
        L += [f"## {name}", "", "| Outcome | Events A / B | Patients A / B | Weighted 7-day cumulative incidence A / B | OR (95% CI) | Risk difference, percentage points (95% CI) | Reading |",
              "|---|---|---|---|---|---|---|"]
        for n in NAMES:
            e = r[n]
            rd = (f"{100 * e['RD']:+.1f} ({100 * e['RD_lo']:+.1f} to {100 * e['RD_hi']:+.1f})" if e.get("RD_lo") is not None
                  else f"{100 * e['RD']:+.1f}")
            L.append(f"| {lab[n]} | {e['eventsA']} / {e['eventsB']} | {e['nA']} / {e['nB']} | {100 * e['riskA']:.1f}% / {100 * e['riskB']:.1f}% | "
                     f"{orci(e)} | {rd} | {o['reading'][name][n]} |")
        c = o["crude_OR"][name]
        L += ["", f"- Unweighted (descriptive) OR: ICU {asd.fmt(c['icu'])}, hospital {asd.fmt(c['hosp'])}", ""]
    L += ["## Proxy outcome in each database (R7, E2 and this revision)", "", "| Database | Cohort type | Follow-up | OR (95% CI) |", "|---|---|---|---|"]
    e = o["external"]
    rows = [("MIMIC-IV", "ICU", "ICU", e["MIMIC-IV icu"]), ("eICU", "ICU", "ICU", e["eICU icu"])]
    rows += [(k, "ICU" if k == "NWICU" else "post-surgical", "ICU", v["icu"]) for k, v in o["per_db"].items()]
    rows += [("MIMIC-IV", "ICU", "hospital", e["MIMIC-IV hosp"])] + [(k, "ICU" if k == "NWICU" else "post-surgical", "hospital", v["hosp"]) for k, v in o["per_db"].items()]
    rows += [("MOVER", "post-surgical", "hospital", e["MOVER hosp"])]
    L += [f"| {a} | {b} | {c} | {orci(v)} |" for a, b, c, v in rows]
    L += ["", "## Pooled (fixed effect)", ""]
    plab = {"icu": "ICU proxy outcome", "hosp": "hospital proxy outcome", "hosp_icu_cohorts": "hospital proxy outcome, ICU cohorts", "hosp_surgical": "hospital proxy outcome, post-surgical cohorts"}
    for k, t in plab.items():
        p = o["pooled"].get(k)
        if p:
            L.append(f"- {t} ({', '.join(p['members'])}): {orci(p)}; I² {100 * p['I2']:.0f}%; reading: {o['reading'].get('pooled:' + k, '')}")
    r = o.get("setting_ratio")
    if r:
        L.append(f"- Ratio of post-surgical to ICU cohorts: {asd.fmt(r['ratio'])} ({asd.fmt(r['lo'])}–{asd.fmt(r['hi'])}); reading: {r['reading']}")
    L += ["", "- MOVER standard error derived from its 95% interval (bootstrap files deleted); MIMIC-IV and eICU use the bootstrap standard errors of R7.", ""]
    return "\n".join(L)


def synth_df(db, n, rng):
    df = pd.DataFrame({"arm": np.where(rng.random(n) < 0.5, "A", "B"), "age": rng.integers(18, 95, n), "scr": rng.uniform(0.5, 4, n),
                       "inr_max": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(0.9, 3, n)),
                       "plt_min": np.where(rng.random(n) < 0.3, np.nan, rng.uniform(20, 400, n)),
                       "lac_max": np.where(rng.random(n) < 0.4, np.nan, rng.uniform(0.5, 8, n)),
                       "route0": rng.choice(["IV", "EN", "UNK"], n, p=[0.6, 0.38, 0.02]),
                       "sed6": rng.random(n) < 0.3, "opi6": rng.random(n) < 0.4, "pres6": rng.random(n) < 0.2,
                       "dementia": (rng.random(n) < 0.05).astype(int), "liver": (rng.random(n) < 0.08).astype(int),
                       "t_end_icu": rng.uniform(1, 300, n), "end_death_icu": rng.random(n) < 0.08})
    df["t_end_hosp"] = df["t_end_icu"] + rng.uniform(0, 300, n)
    df["end_death_hosp"] = df["end_death_icu"] | (rng.random(n) < 0.03)
    if db == "n":
        df["gender"] = rng.choice(["F", "M"], n)
        df["vent"], df["niv"] = rng.random(n) < 0.3, rng.random(n) < 0.1
        df["unit"] = rng.choice(["ICU", "MICU", "NSICU", "CICU", "CCU", "SICU", "CTICU"], n)
        df["admission_type"] = rng.choice(["EMERGENCY", "ELECTIVE", "URGENT"], n)
        df["admission_location"] = rng.choice(["EMERGENCY MEDICINE", "SURGERY"], n)
        df["era"] = rng.choice(["2020 - 2022", "2017 - 2019"], n, p=[0.97, 0.03])
        df["h_icu"] = rng.uniform(0, 120, n)
        df["alcohol"] = (rng.random(n) < 0.06).astype(int)
    else:
        df["sex"] = rng.choice(["F", "M"], n)
        df["asa"] = rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[0.05, 0.4, 0.4, 0.1, 0.03, 0.02])
        df["emop"] = (rng.random(n) < 0.2).astype(int)
        df["department"] = rng.choice(["CTS", "GS", "NS", "OS", "UR"], n)
        df["antype"] = rng.choice(["General", "MAC"], n, p=[0.93, 0.07])
        df["cpb"] = rng.random(n) < 0.2
        df["or_h"], df["h_or"] = rng.uniform(1, 12, n), rng.uniform(0, 72, n)
        df["vent_last"] = rng.choice([1.0, 0.0, np.nan], n)
        df["gcsm_last"] = rng.choice([6.0, 5.0, 3.0, np.nan], n)
    for c in ("t_ap", "t_oq", "t_hypo"):
        df[c] = np.where(rng.random(n) < 0.1, rng.uniform(0.5, 160, n), np.nan)
    return df


def selftest(args):
    res = []

    def check(name, ok, detail=""):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name}{' | ' + detail if detail else ''}", flush=True)

    rng = np.random.default_rng(11)
    for db in ("n", "i"):
        df = synth_df(db, 1500, rng)
        d = build(df, db, outcome=True)
        check(f"{db}: design matrix finite, names unique", np.isfinite(d["X"]).all() and len(set(d["names"])) == len(d["names"]),
              f"{len(d['names'])} columns")
        r = analyses(d, {})
        check(f"{db}: every analysis gives a finite odds ratio on synthetic data", all(np.isfinite(r[n]["OR"]) for n in NAMES))
        rep = 5
        b = asd.resample(d, "m", np.random.default_rng(asd.SEED + rep))
        _G.update(d=d, cache={})
        _, (lor, _) = _boot_one(rep)
        check(f"{db}: bootstrap replicate equals asd.resample draws", abs(lor[0] - math.log(analyses(b, {})["icu"]["OR"])) < 1e-10)
    n = 400
    d = {"A": np.r_[np.ones(200), np.zeros(200)].astype(int), "X": np.ones((n, 1)), "t_dev": np.full(n, np.nan),
         "t_end_icu": np.full(n, 50.0), "end_death_icu": np.zeros(n, bool), "t_end_hosp": np.full(n, 150.0),
         "end_death_hosp": np.zeros(n, bool), "t_ap": np.full(n, 80.0)}
    di = dict(d, t_end=d["t_end_icu"], end_death=d["end_death_icu"])
    dh = dict(d, t_end=d["t_end_hosp"], end_death=d["end_death_hosp"])
    ei, eh = asd.cif_contrast(di, np.ones(n), d["t_ap"])[168], asd.cif_contrast(dh, np.ones(n), d["t_ap"])[168]
    check("a dose after ICU discharge counts hospital-wide only", ei["eventsA"] + ei["eventsB"] == 0 and eh["eventsA"] + eh["eventsB"] == n)
    fl = pd.DataFrame({"adult": [True] * 7 + [False], "has_as": [True] * 6 + [False, True],
                       "x_prior24": [True, False, False, False, False, False, False, False],
                       "x_ap": [False, True, False, False, False, False, False, False], "x_psy": [False] * 8,
                       "x_gib": [True, False, True, False, False, False, False, False], "x_end": [False] * 5 + [True, False, False],
                       "first_cls": ["PPI", "PPI", "FAM", "OTH", "FAM", "PPI", None, "PPI"]})
    f = flow_counts(fl, "n", 2, 1, 1)
    want = {"before_top": 7, "top": 6, "x1_prior24": 1, "x2_antipsychotic_or_psych": 1, "x3_bleeding": 1, "x4_other_first_drug": 1,
            "x5_time_zero_at_end": 1, "eligible": 1}
    check("flow: hierarchical attribution on a toy table", all(f[k] == v for k, v in want.items()), json.dumps(f))
    p = pool_list([("a", 0.8, 0.1), ("b", 0.8, 0.2)])
    check("pooling: identical estimates give I2 0 and the same OR", abs(p["OR"] - 0.8) < 1e-12 and p["I2"] == 0)
    sr = setting_ratio(pool_list([("a", 0.8, 0.1), ("b", 0.8, 0.1)]), pool_list([("c", 0.8, 0.1), ("d", 0.8, 0.1)]))
    check("setting ratio: equal groups give 1 and 'no evidence'", abs(sr["ratio"] - 1) < 1e-12 and sr["reading"] == "no evidence that the post-surgical cohorts differ")
    check("verdicts", verdict({"OR_lo": 0.6, "OR_hi": 0.9}) == "proxy outcome supports fewer events in the PPI arm" and verdict({"OR_lo": 1.1, "OR_hi": 1.5}) == "proxy outcome in the opposite direction"
          and nc_reading({"OR_lo": 1.02, "OR_hi": 1.3}) == "negative control suggests residual confounding")
    txt = {"n_fl": N_FL, "n_cohort": N_COHORT, "n_cov": N_COV, "n_out": N_OUT, "i_fl": I_FL, "i_cohort": I_COHORT, "i_cov": I_COV, "i_out": I_OUT,
           "n_emar": extract_sql("n", "emar"), "i_med": extract_sql("i", "med")}
    frags = [("n_fl", "x.t >= f.t0 - INTERVAL 24 HOUR"), ("n_fl", "d.seq_num = 1"), ("n_fl", "'^(F2[0-9]|F31)'"),
             ("n_fl", "PARTITION BY subject_id ORDER BY intime, stay_id"), ("n_emar", "event_txt IN ('Confirmed', 'Applied')"),
             ("i_fl", "icuin_time BETWEEN orout_time - 60 AND orout_time + 1440"), ("i_fl", "x.t >= f.t0 - 1440"),
             ("i_fl", "'HAL', 'OQ', 'APO', 'PSYRX'"), ("i_cohort", "PARTITION BY subject_id ORDER BY t0, op_id"),
             ("i_out", "m.t <= o.orout_time + 360"), ("n_out", "x.t <= b.t0 + INTERVAL 168 HOUR"), ("i_out", "l.v < 70"),
             ("n_cov", "x.t BETWEEN b.t0 - INTERVAL 6 HOUR AND b.t0"), ("i_cov", "w.t BETWEEN b.t0 - 360 AND b.t0"),
             ("i_med", "^A02B[AC]")]
    check("SQL text carries every written condition", all(s in txt[k] for k, s in frags),
          ", ".join(f"{k}:{s[:25]}" for k, s in frags if s not in txt[k]))
    print(f"{sum(res)}/{len(res)} passed")
    sys.exit(0 if all(res) else 1)


def bindcheck(args):
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="r9_bind_")
    res = []

    def check(name, ok, detail=""):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)

    try:
        for db in ("n", "i"):
            root = os.path.join(tmp, "src_" + db)
            for key, rel in FILES[db].items():
                f = os.path.join(root, rel)
                os.makedirs(os.path.dirname(f), exist_ok=True)
                with gzip.open(os.path.join(SRC[db], rel), "rt", encoding="utf-8-sig", newline="") as fi:
                    header = fi.readline()
                with gzip.open(f, "wt", encoding="utf-8", newline="") as fo:
                    fo.write(header)
            P = os.path.join(tmp, "p_" + db)
            os.makedirs(P)
            con = con_for(P)
            steps = [(f"extract {key}", extract_sql(db, key, root), pq(P, f"{db}_{key}"), None) for key in FILES[db]]
            steps += [(f"design {name}", sql, pq(P, f"{db}_{name}"), EXPECT.get(f"{db}_{name}")) for name, sql in DESIGN_SQL[db]]
            steps += [("outcome", OUT_SQL[db], pq(P, f"{db}_out"), EXPECT[f"{db}_out"])]
            for name, sql, path, want in steps:
                try:
                    views(con, P, db)
                    asd.write_pq(con, sql, path)
                    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
                    good = want is None or cols == want.split()
                    check(f"{db}: {name}", good, "binds" + ("" if good else f"; got {cols}"))
                except Exception as e:
                    msg = str(e).split("Original Line")[0].strip().splitlines()
                    check(f"{db}: {name}", False, f"{type(e).__name__}: {msg[0][:160] if msg else ''}")
            con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(res)}/{len(res)} passed ({time.time() - t0:.0f} s)")
    sys.exit(0 if all(res) else 1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "bindcheck"):
        sp.add_parser(name)
    subs = {c: sp.add_parser(c) for c in ("extract", "design", "outcome", "estimate", "boot", "report")}
    for s in subs.values():
        s.add_argument("--private", required=True)
        s.add_argument("--results", required=True)
    for c in ("extract", "design", "outcome", "boot"):
        subs[c].add_argument("--db", choices=["n", "i"], required=True)
    subs["extract"].add_argument("--part", choices=["small", "emar", "med", "lab"], required=True)
    subs["boot"].add_argument("--start", type=int, required=True)
    subs["boot"].add_argument("--stop", type=int, required=True)
    subs["boot"].add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "extract" and a.part not in PARTS[a.db]:
        raise SystemExit(f"part {a.part} does not exist for --db {a.db}: {', '.join(PARTS[a.db])}")
    {"selftest": selftest, "bindcheck": bindcheck, "extract": extract, "design": design, "outcome": outcome, "estimate": estimate,
     "boot": boot, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
