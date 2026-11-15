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

SRC = {"m": os.path.join(os.environ.get("DATA_DIR", "data"), "mimic-iv-3.1-parquet"), "e": os.path.join(os.environ.get("DATA_DIR", "data"), "eicu-crd-2.0-parquet")}
NBOOT = asd.NBOOT
VARIANT_FILES = {"nopdx": "r7_ppi_mimiciv_nopdx.sql", "ord": "r7_ppi_mimiciv_ord.sql", "dose": "r7_famdose_eicu_dose.sql"}
VARIANTS = {"nopdx": ("m", 2), "ord": ("m", 2), "dose": ("e", 1)}
BASE_KEYS = ["c1m", "c1e", "c2m", "c2e"]
ALL_KEYS = BASE_KEYS + list(VARIANTS)

AP = "haloperidol|haldol|olanzapine|zyprexa|quetiapine|seroquel|risperidone|risperdal|ziprasidone|geodon|aripiprazole|abilify"
AP_LAI = "decanoate|long acting|pamoate|lauroxil|aripiprazole extended"
OQ = "olanzapine|zyprexa|quetiapine|seroquel"
INS_ITEMS = "223257, 223258, 223259, 223260, 223261, 223262, 229299, 229619"
HAL_ITEM = 221824
AP_HICL = "1661, 1662, 18084, 14015, 4137"
OQ_HICL = "14015, 4137"
DEM = ("^(F01|F02|F03|G30|G310|G311|G3183)", "^(290|2941|2942|3310|3311|3312|33182)")
ALC = ("^F10", "^(291|303|3050)")
LIV = ("^(K70|K73|K74|K766|I85)", "^(571|5723|4560|4561|4562)")
DM = ("^E(08|09|10|11|13)", "^250")
PSY = ("^(F2[0-9]|F31)", "^(295|297|2960|2961|2964|2965|2966|2967|2968)")


def ws(s):
    return re.sub(r"\s+", " ", s).strip()


def cte_blocks(body):
    pos = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^(\w+) AS \(", body)]
    out = {}
    for i, (s, name) in enumerate(pos):
        e = pos[i + 1][0] if i + 1 < len(pos) else len(body)
        txt = body[s:e].rstrip()
        out[name] = txt[:-1] if txt.endswith(",") else txt
    return out


def sql_body(path):
    body = open(path, encoding="utf-8").read()
    body = re.sub(r"^\s*--.*$", "", body, flags=re.M).strip().rstrip(";")
    return re.sub(r"^\s*WITH\s+", "", body, flags=re.I)


def must_replace(text, old, new):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"R7 build: expected one occurrence, found {n}: {old[:70]}")
    return text.replace(old, new)


def split_gib(gib):
    parts = gib.split("\n  UNION\n")
    if len(parts) != 2 or not parts[0].startswith("gib AS ("):
        raise SystemExit("R7 build: the gate's gib CTE does not have the expected two parts")
    return parts[0].replace("gib AS (", "gib_ed AS (", 1) + "\n)", "gib_pdx AS (\n" + parts[1]


def pf(P, db, name):
    return os.path.join(P, f"{db}_{name}.parquet")


def nz(s):
    return pd.to_numeric(pd.Series(s), errors="coerce").fillna(0).astype(float).values


def connect(P, db):
    con = asd.connect_slim(P, db)
    con.execute("CREATE SCHEMA IF NOT EXISTS src")
    tabs = [("hosp", "emar"), ("icu", "inputevents")] if db == "m" else [("eicu", "medication")]
    base = SRC[db] if db == "m" else os.path.join(SRC[db], "eicu")
    for m, tb in tabs:
        d = base if db == "e" else os.path.join(base, m)
        f = os.path.join(d, tb + ".parquet")
        ref = f if os.path.exists(f) else os.path.join(d, tb, "*.parquet")
        con.execute(f"CREATE OR REPLACE VIEW src.{tb} AS SELECT * FROM read_parquet('{ref}')")
    return con


ASADM_ORD = f"""asadm AS (
  SELECT r.hadm_id, r.starttime AS t, r.stoptime AS t_stop,
         CASE WHEN regexp_matches(lower(r.drug), 'famotidine|pepcid') THEN 'FAM'
              WHEN regexp_matches(lower(r.drug), 'ranitidine|zantac|cimetidine|nizatidine') THEN 'H2O' ELSE 'PPI' END AS drug
  FROM hosp.prescriptions r
  WHERE regexp_matches(lower(r.drug), '{asd.AS_RX}') AND r.starttime IS NOT NULL AND r.hadm_id IS NOT NULL
)"""
WASH_ADM = "AND NOT EXISTS (SELECT 1 FROM asadm a WHERE a.hadm_id = f.hadm_id AND a.t < f.t0 AND a.t >= f.t0 - INTERVAL 24 HOUR)"
WASH_ORD = ("AND NOT EXISTS (SELECT 1 FROM asadm a WHERE a.hadm_id = f.hadm_id AND a.t < f.t0 AND a.t > f.t0 - INTERVAL 7 DAY\n"
            "                    AND (a.t_stop IS NULL OR a.t_stop > f.t0 - INTERVAL 24 HOUR OR a.t_stop <= a.t))")
DOSE_MED = r"""med AS (
  SELECT *, CASE fcls WHEN 'x0.5' THEN mg * 0.5 WHEN 'x1' THEN mg WHEN 'x2' THEN mg * 2 WHEN 'x3' THEN mg * 3
                      WHEN 'x4' THEN mg * 4 WHEN 'one' THEN mg END AS daily,
         (fcls IS NOT NULL AND fcls <> 'one' AND mg IS NOT NULL) AS sched
  FROM (
    SELECT *,
           CASE WHEN regexp_matches(freq, 'q12|q 12|bid|twice|every 12|2 times') THEN 'x2'
                WHEN regexp_matches(freq, 'q8|q 8|tid|every 8|3 times') THEN 'x3'
                WHEN regexp_matches(freq, 'q6|q 6|qid|every 6|4 times') THEN 'x4'
                WHEN regexp_matches(freq, 'q48|q 48|every 48|every other day|qod') THEN 'x0.5'
                WHEN regexp_matches(freq, 'q24|q 24|daily|qd|every 24|qhs|bedtime|nightly|1 time|1xdaily|qam|qpm') THEN 'x1'
                WHEN regexp_matches(freq, 'once|now|stat|x1|one time|preop|on call') THEN 'one' END AS fcls,
           CASE WHEN regexp_matches(dosage, '^\s*[0-9.]+\s*mg') OR regexp_matches(dosage, '^\s*[0-9.]+\s+[0-9]+\s*$')
                     OR regexp_matches(dosage, '^\s*[0-9.]+\s*$')
                THEN TRY_CAST(regexp_extract(dosage, '([0-9.]+)', 1) AS DOUBLE) END AS mg
    FROM med0
  )
)"""
DOSE_FIRST = """first_as0 AS (
  SELECT f.patientunitstayid, f.t0 AS drugstartoffset,
         CASE WHEN count(DISTINCT m.drug) = 1 THEN min(m.drug) END AS drug,
         count(*) FILTER (WHERE m.sched) AS n_sched,
         CASE WHEN min(m.daily) FILTER (WHERE m.sched) = max(m.daily) FILTER (WHERE m.sched)
              THEN min(m.daily) FILTER (WHERE m.sched) END AS daily_sched,
         CASE WHEN min(m.daily) FILTER (WHERE m.fcls = 'one') = max(m.daily) FILTER (WHERE m.fcls = 'one')
              THEN min(m.daily) FILTER (WHERE m.fcls = 'one') END AS daily_one
  FROM first_off f JOIN med m ON m.patientunitstayid = f.patientunitstayid AND m.drugstartoffset = f.t0
  GROUP BY 1, 2
),
nxt AS (
  SELECT f.patientunitstayid,
         (SELECT m.daily FROM med m WHERE m.patientunitstayid = f.patientunitstayid AND m.drug = 'FAM' AND m.sched
            AND m.drugstartoffset > f.drugstartoffset AND m.drugstartoffset <= f.drugstartoffset + 1440
          ORDER BY m.drugstartoffset, m.daily LIMIT 1) AS daily_next
  FROM first_as0 f
),
first_as AS (
  SELECT f.patientunitstayid, f.drugstartoffset, f.drug,
         CASE WHEN f.n_sched > 0 THEN f.daily_sched ELSE coalesce(n.daily_next, f.daily_one) END AS daily
  FROM first_as0 f LEFT JOIN nxt n USING (patientunitstayid)
)"""
TITLES = {"nopdx": "R7 variant 4a: MIMIC-IV PPI comparison, bleeding exclusion without the principal discharge diagnosis",
          "ord": "R7 variant 4b: MIMIC-IV PPI comparison, time zero and arm from orders (hosp.prescriptions), as in eICU",
          "dose": "R7 variant 4c: eICU dose comparison, dose classification recovered"}


def variant_sql(name):
    if name in ("nopdx", "ord"):
        b = cte_blocks(asd.gate_body(2, "m"))
        if name == "nopdx":
            ed, _ = split_gib(b["gib"])
            b["gib"] = ed.replace("gib_ed AS (", "gib AS (", 1)
        else:
            b["asadm"] = ASADM_ORD
            b["cand"] = must_replace(b["cand"], WASH_ADM, WASH_ORD)
        src = "ppi_mimiciv.sql"
    else:
        b = cte_blocks(asd.gate_body(1, "e"))
        for k in ("med", "first_as"):
            if k not in b:
                raise SystemExit(f"R7 build: CTE {k} missing in famdose_eicu.sql")
        b["med"], b["first_as"] = DOSE_MED, DOSE_FIRST
        src = "famdose_eicu.sql"
    head = (f"-- {TITLES[name]}\n-- Generated by r7.py make-variants from the locked {src} (revision R7, "
            "SAP_addendum_R7.md v1.0); do not edit by hand.\n")
    return head + "WITH " + ",\n".join(b.values()) + "\n"


def make_variants(args):
    for name, f in VARIANT_FILES.items():
        txt = variant_sql(name)
        path = os.path.join(asd.SQL_DIR, f)
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if old != txt:
            open(path, "w", encoding="utf-8").write(txt)
        print(f"{f}: {'unchanged' if old == txt else 'written'}", flush=True)


M_FRAGS = ["f.drug IN ('FAM', 'PPI') AND f.anchor_year_group <> '2008 - 2010'",
           "NOT EXISTS (SELECT 1 FROM asadm a WHERE a.hadm_id = f.hadm_id AND a.t < f.t0 AND a.t >= f.t0 - INTERVAL 24 HOUR)",
           "f.hadm_id NOT IN (SELECT hadm_id FROM home)",
           "NOT EXISTS (SELECT 1 FROM cam c WHERE c.stay_id = f.stay_id AND c.charttime <= f.t0)",
           "f.hadm_id NOT IN (SELECT hadm_id FROM gib)",
           "NOT EXISTS (SELECT 1 FROM ppi_inf p WHERE p.hadm_id = f.hadm_id AND p.starttime BETWEEN f.t0 - INTERVAL 24 HOUR AND f.t0 + INTERVAL 1 HOUR)",
           "(SELECT c.valuenum FROM cr c WHERE c.subject_id = f.subject_id AND c.charttime BETWEEN f.t0 - INTERVAL 24 HOUR AND f.t0 "
           "ORDER BY c.charttime DESC LIMIT 1) AS scr"]
M_FL = """fl AS (
  SELECT f.subject_id, f.hadm_id, f.stay_id, f.t0, f.drug, f.anchor_year_group AS era, f.gender, f.age,
         EXISTS (SELECT 1 FROM asadm a WHERE a.hadm_id = f.hadm_id AND a.t < f.t0 AND a.t >= f.t0 - INTERVAL 24 HOUR) AS x_prior24,
         f.hadm_id IN (SELECT hadm_id FROM home) AS x_home,
         EXISTS (SELECT 1 FROM cam c WHERE c.stay_id = f.stay_id AND c.charttime <= f.t0) AS x_del,
         f.hadm_id IN (SELECT hadm_id FROM gib_ed) AS x_gib_ed,
         f.hadm_id IN (SELECT hadm_id FROM gib_pdx) AS x_gib_pdx,
         EXISTS (SELECT 1 FROM ppi_inf p WHERE p.hadm_id = f.hadm_id AND p.starttime BETWEEN f.t0 - INTERVAL 24 HOUR AND f.t0 + INTERVAL 1 HOUR) AS x_inf,
         (SELECT c.valuenum FROM cr c WHERE c.subject_id = f.subject_id AND c.charttime BETWEEN f.t0 - INTERVAL 24 HOUR AND f.t0
          ORDER BY c.charttime DESC LIMIT 1) AS scr
  FROM first_as f
)"""
E_FRAGS = ["p.hospitalid IN (SELECT hospitalid FROM dl_hosp) AND f.drug IN ('FAM', 'PPI') AND f.drugstartoffset < p.unitdischargeoffset",
           "NOT regexp_matches(lower(coalesce(p.apacheadmissiondx, '')), 'bleeding, upper gi|bleeding, gi|gi bleeding|bleeding, lower gi|ulcer disease|varices|peptic')",
           "NOT EXISTS (SELECT 1 FROM gib g WHERE g.patientunitstayid = p.patientunitstayid AND g.t <= f.drugstartoffset)",
           "NOT EXISTS (SELECT 1 FROM ppi_inf i WHERE i.patientunitstayid = p.patientunitstayid AND i.t BETWEEN f.drugstartoffset - 1440 AND f.drugstartoffset + 60)",
           "NOT EXISTS (SELECT 1 FROM med a WHERE a.patientunitstayid = p.patientunitstayid AND a.drugstartoffset < f.drugstartoffset "
           "AND a.drugstartoffset > f.drugstartoffset - 10080 AND (a.drugstopoffset > f.drugstartoffset - 1440 OR a.drugstopoffset <= a.drugstartoffset))",
           "p.patientunitstayid NOT IN (SELECT patientunitstayid FROM home)",
           "NOT EXISTS (SELECT 1 FROM dsc d WHERE d.patientunitstayid = p.patientunitstayid AND d.pos AND d.t <= f.drugstartoffset)",
           "(SELECT c.labresult FROM cr c WHERE c.patientunitstayid = p.patientunitstayid AND c.labresultoffset BETWEEN f.drugstartoffset - 1440 "
           "AND f.drugstartoffset ORDER BY c.labresultoffset DESC LIMIT 1) AS scr"]
E_FL = """fl AS (
  SELECT p.patientunitstayid, p.hospitalid, p.gender, p.age, f.drugstartoffset AS t0, f.drug,
         p.hospitalid IN (SELECT hospitalid FROM dl_hosp) AS in_hosp,
         f.drugstartoffset < p.unitdischargeoffset AS before_disc,
         regexp_matches(lower(coalesce(p.apacheadmissiondx, '')), 'bleeding, upper gi|bleeding, gi|gi bleeding|bleeding, lower gi|ulcer disease|varices|peptic') AS x_gib_apache,
         EXISTS (SELECT 1 FROM gib g WHERE g.patientunitstayid = p.patientunitstayid AND g.t <= f.drugstartoffset) AS x_gib_dx,
         EXISTS (SELECT 1 FROM ppi_inf i WHERE i.patientunitstayid = p.patientunitstayid AND i.t BETWEEN f.drugstartoffset - 1440 AND f.drugstartoffset + 60) AS x_inf,
         EXISTS (SELECT 1 FROM med a WHERE a.patientunitstayid = p.patientunitstayid AND a.drugstartoffset < f.drugstartoffset
                 AND a.drugstartoffset > f.drugstartoffset - 10080
                 AND (a.drugstopoffset > f.drugstartoffset - 1440 OR a.drugstopoffset <= a.drugstartoffset)) AS x_prior24,
         p.patientunitstayid IN (SELECT patientunitstayid FROM home) AS x_home,
         EXISTS (SELECT 1 FROM dsc d WHERE d.patientunitstayid = p.patientunitstayid AND d.pos AND d.t <= f.drugstartoffset) AS x_del,
         (SELECT c.labresult FROM cr c WHERE c.patientunitstayid = p.patientunitstayid AND c.labresultoffset BETWEEN f.drugstartoffset - 1440 AND f.drugstartoffset
          ORDER BY c.labresultoffset DESC LIMIT 1) AS scr
  FROM pt p JOIN first_as f USING (patientunitstayid)
)"""


def flow_sql(db):
    b2 = cte_blocks(asd.gate_body(2, db))
    b1 = cte_blocks(asd.gate_body(1, db))
    shared = ["icu1", "asadm", "first_t", "first_as", "home", "cr", "cam"] if db == "m" else \
             ["pt", "dsc", "dl_hosp", "med0", "med", "first_off", "first_as", "home", "cr"]
    for k in shared:
        t2 = ws(b2[k])
        if db == "e" and k == "pt":
            t2 = t2.replace(", p.apacheadmissiondx", "")
        if ws(b1[k]) != t2:
            raise SystemExit(f"R7 flow: CTE {k} differs between the dose and PPI comparisons ({db})")
    cand = ws(b2["cand"])
    for frag in (M_FRAGS if db == "m" else E_FRAGS):
        if ws(frag) not in cand:
            raise SystemExit(f"R7 flow: gate condition not found verbatim ({db}): {frag[:70]}")
    egfr = re.search(r"SELECT \*, (.*?) AS egfr", ws(b2["cohort"])).group(1)
    if db == "m":
        ed, pdx = split_gib(b2["gib"])
        parts = [b2[k] for k in ("icu1", "asadm", "first_t", "first_as", "home", "ppi_inf", "cr", "cam")] + [ed, pdx, M_FL]
        pre = "WITH " + b2["icu1"] + "\nSELECT count(*) AS n FROM icu1 WHERE age >= 18 AND anchor_year_group <> '2008 - 2010'"
    else:
        parts = [b2[k] for k in ("pt", "dsc", "dl_hosp", "med0", "med", "first_off", "first_as", "home", "gib", "ppi_inf", "cr")] + [E_FL]
        pre = ("WITH " + ",\n".join(b2[k] for k in ("pt", "dsc", "dl_hosp"))
               + "\nSELECT count(*) AS n FROM pt WHERE hospitalid IN (SELECT hospitalid FROM dl_hosp)")
    main = "WITH " + ",\n".join(parts) + f"\nSELECT *, CASE WHEN scr IS NULL THEN NULL ELSE {egfr} END AS egfr FROM fl"
    return main, pre


def flow_counts(fl, db, pre_n, arms1):
    if db == "m":
        top = fl[fl["era"] != "2008 - 2010"].copy()
        bleed = top["x_gib_ed"] | top["x_gib_pdx"] | top["x_inf"]
        src = {"ed_dx": top["x_gib_ed"], "principal_dx": top["x_gib_pdx"], "ppi_infusion": top["x_inf"]}
    else:
        top = fl[fl["in_hosp"] & fl["before_disc"]].copy()
        bleed = top["x_gib_apache"] | top["x_gib_dx"] | top["x_inf"]
        src = {"apache_dx": top["x_gib_apache"], "dx_before_t0": top["x_gib_dx"], "ppi_infusion": top["x_inf"]}
    for c in ("x_prior24", "x_home", "x_del"):
        top[c] = top[c].astype(bool)
    drug = top["drug"].astype(object)
    ok_drug = drug.isin(["FAM", "PPI"])
    e1 = top["x_prior24"] | top["x_home"]
    e2 = ~e1 & top["x_del"]
    e3 = ~e1 & ~e2 & bleed
    e4 = ~e1 & ~e2 & ~e3 & ~ok_drug
    el = ~(e1 | e2 | e3 | e4)
    c2 = {"before_top": int(pre_n), "top": int(len(top)), "x1_prior_or_home": int(e1.sum()), "x2_delirium_before": int(e2.sum()),
          "x3_bleeding": int(e3.sum()), "x4_other_first_drug": int(e4.sum()), "excluded": int((~el).sum()),
          "eligible": int(el.sum()), "A": int((el & (drug == "PPI")).sum()), "B": int((el & (drug == "FAM")).sum()),
          "any_prior24": int(top["x_prior24"].sum()), "any_home": int(top["x_home"].sum()), "any_delirium_before": int(top["x_del"].sum()),
          "any_bleeding": int(bleed.sum()), "any_other_first_drug": int((~ok_drug).sum())}
    for k, v in src.items():
        c2[f"bleeding_source_{k}"] = int((e3 & v.astype(bool)).sum())
    if db == "m":
        only_pdx = ~e1 & ~e2 & ok_drug & top["x_gib_pdx"] & ~top["x_gib_ed"] & ~top["x_inf"]
        c2["only_principal_dx_A"] = int((only_pdx & (drug == "PPI")).sum())
        c2["only_principal_dx_B"] = int((only_pdx & (drug == "FAM")).sum())
    nf = drug != "FAM"
    f1 = ~nf & e1
    f2 = ~nf & ~f1 & top["x_del"]
    egfr = pd.to_numeric(top["egfr"], errors="coerce")
    f3 = ~nf & ~f1 & ~f2 & top["scr"].isna()
    f4 = ~nf & ~f1 & ~f2 & ~f3 & ~(egfr < 60)
    el1 = ~(nf | f1 | f2 | f3 | f4)
    c1 = {"top": int(len(top)), "x0_first_not_famotidine": int(nf.sum()), "x1_prior_or_home": int(f1.sum()),
          "x2_delirium_before": int(f2.sum()), "x3_no_creatinine": int(f3.sum()), "x4_egfr_ge60": int(f4.sum()),
          "eligible": int(el1.sum()), **arms1}
    return {"ppi": c2, "famdose": c1}


def flow(args):
    t0 = time.time()
    P, db = os.path.expanduser(args.private), args.db
    con = connect(P, db)
    main, pre = flow_sql(db)
    asd.write_pq(con, main, pf(P, db, "r7flow"))
    pre_n = con.execute(pre).fetchone()[0]
    fl = asd.read_pq(pf(P, db, "r7flow"))
    co1 = asd.read_pq(pf(P, db, "c1_cohort"))
    arms1 = {"cohort_eligible": int(len(co1)), "A": int((co1["arm"] == "A").sum()), "B": int((co1["arm"] == "B").sum()),
             "other": int(co1["arm"].isna().sum())}
    res = flow_counts(fl, db, pre_n, arms1)
    key2, key1 = f"c2{db}", f"c1{db}"
    ok = {"ppi": {k: sc.mask_n(res["ppi"][k]) == asd.GATE[key2][g] for k, g in (("eligible", "eligible"), ("A", "A"), ("B", "B"))},
          "famdose": {"eligible": sc.mask_n(res["famdose"]["eligible"]) == asd.GATE[key1]["eligible"],
                    "cohort": sc.mask_n(res["famdose"]["cohort_eligible"]) == asd.GATE[key1]["eligible"]}}
    res["gate_check"] = ok
    json.dump(res, open(os.path.join(P, f"{db}_r7flow.json"), "w"))
    good = all(v for g in ok.values() for v in g.values())
    asd.status(f"flow {db}: eligible PPI comparison {sc.mask_n(res['ppi']['eligible'])}, dose comparison {sc.mask_n(res['famdose']['eligible'])}; "
               f"gate check {'PASS' if good else 'FAIL'}", t0)
    asd.log_run(args.results, f"r7 flow --db {db}: {'ok' if good else 'GATE CHECK FAILED'} ({time.time() - t0:.0f} s)")
    if not good:
        raise SystemExit("flow counts differ from the gate: stop (addendum section 1)")


def ccase(pat, name):
    return (f"max(CASE WHEN (icd_version = 10 AND regexp_matches(icd_code, '{pat[0]}')) OR "
            f"(icd_version = 9 AND regexp_matches(icd_code, '{pat[1]}')) THEN 1 ELSE 0 END) AS {name}")


M_COV = f"""
WITH base AS (SELECT DISTINCT subject_id, hadm_id, stay_id, t0 FROM read_parquet({{files}})),
prior AS (
  SELECT b.stay_id, d.icd_code, d.icd_version FROM base b JOIN hosp.admissions a0 ON a0.hadm_id = b.hadm_id
  JOIN hosp.admissions a ON a.subject_id = b.subject_id AND a.admittime < a0.admittime
  JOIN hosp.diagnoses_icd d ON d.hadm_id = a.hadm_id
  UNION ALL
  SELECT b.stay_id, d.icd_code, d.icd_version FROM base b JOIN ed.edstays e ON e.hadm_id = b.hadm_id JOIN ed.diagnosis d ON d.stay_id = e.stay_id),
codes AS (SELECT stay_id, 0 AS cur, icd_code, icd_version FROM prior
          UNION ALL
          SELECT b.stay_id, 1 AS cur, d.icd_code, d.icd_version FROM base b JOIN hosp.diagnoses_icd d ON d.hadm_id = b.hadm_id),
fl AS (SELECT stay_id, cur, {ccase(DM, 'dm')}, {ccase(DEM, 'dem')}, {ccase(ALC, 'alc')}, {ccase(LIV, 'liv')}, {ccase(PSY, 'psy')}
       FROM codes GROUP BY 1, 2),
ii AS (SELECT DISTINCT b.stay_id FROM base b JOIN src.inputevents x ON x.stay_id = b.stay_id AND x.starttime <= b.t0 AND x.endtime > b.t0
       WHERE x.itemid IN ({INS_ITEMS}) AND coalesce(x.rate, 0) > 0 AND coalesce(x.statusdescription, '') <> 'Rewritten'),
i24 AS (SELECT DISTINCT b.stay_id FROM base b JOIN src.inputevents x ON x.stay_id = b.stay_id AND x.starttime <= b.t0
          AND coalesce(x.endtime, x.starttime) >= b.t0 - INTERVAL 24 HOUR
        WHERE x.itemid IN ({INS_ITEMS}) AND (coalesce(x.amount, 0) > 0 OR coalesce(x.rate, 0) > 0)
          AND coalesce(x.statusdescription, '') <> 'Rewritten'
        UNION
        SELECT DISTINCT b.stay_id FROM base b JOIN src.emar e ON e.hadm_id = b.hadm_id
          AND e.charttime BETWEEN b.t0 - INTERVAL 24 HOUR AND b.t0
        WHERE regexp_matches(lower(e.medication), 'insulin')
          AND regexp_matches(lower(coalesce(e.event_txt, '')), 'administ|started')
          AND NOT regexp_matches(lower(coalesce(e.event_txt, '')), 'not |hold|stopp'))
SELECT b.stay_id, coalesce(p.dm, 0) AS diabetes, coalesce(c.dm, 0) AS diabetes_cur, coalesce(c.dem, 0) AS dementia_cur,
       coalesce(c.alc, 0) AS alcohol_cur, coalesce(c.liv, 0) AS liver_cur, greatest(coalesce(p.psy, 0), coalesce(c.psy, 0)) AS psych,
       (ii.stay_id IS NOT NULL) AS insulin_inf, (i24.stay_id IS NOT NULL) AS insulin_24h
FROM base b LEFT JOIN fl p ON p.stay_id = b.stay_id AND p.cur = 0 LEFT JOIN fl c ON c.stay_id = b.stay_id AND c.cur = 1
LEFT JOIN ii ON ii.stay_id = b.stay_id LEFT JOIN i24 ON i24.stay_id = b.stay_id
"""

M_OUT = f"""
WITH base AS (SELECT DISTINCT subject_id, hadm_id, stay_id, intime, outtime, t0 FROM read_parquet({{files}})),
ap AS (
  SELECT e.hadm_id, e.charttime AS t, regexp_matches(lower(e.medication), '{OQ}') AS oq
  FROM src.emar e
  WHERE regexp_matches(lower(e.medication), '{AP}') AND NOT regexp_matches(lower(e.medication), '{AP_LAI}')
    AND regexp_matches(lower(coalesce(e.event_txt, '')), 'administ|started')
    AND NOT regexp_matches(lower(coalesce(e.event_txt, '')), 'not |hold|stopp')
  UNION ALL
  SELECT s.hadm_id, x.starttime AS t, false AS oq FROM src.inputevents x JOIN icu.icustays s USING (stay_id)
  WHERE x.itemid = {HAL_ITEM} AND coalesce(x.amount, 0) > 0 AND coalesce(x.statusdescription, '') <> 'Rewritten'
),
edap AS (SELECT DISTINCT e.hadm_id, y.charttime AS t FROM ed.pyxis y JOIN ed.edstays e USING (stay_id)
         WHERE e.hadm_id IS NOT NULL AND regexp_matches(lower(y.name), '{AP}')),
pre AS (SELECT DISTINCT b.stay_id FROM base b JOIN ap a ON a.hadm_id = b.hadm_id AND a.t <= b.t0
        UNION
        SELECT DISTINCT b.stay_id FROM base b JOIN edap a ON a.hadm_id = b.hadm_id AND a.t <= b.t0),
win AS (SELECT b.stay_id, b.hadm_id, b.t0, least(b.outtime, b.t0 + INTERVAL 168 HOUR) AS t_icu,
               least(coalesce(least(a.deathtime, a.dischtime), a.dischtime), b.t0 + INTERVAL 168 HOUR) AS t_hosp,
               coalesce(least(a.deathtime, a.dischtime), a.dischtime) AS t_end_hosp, a.deathtime
        FROM base b LEFT JOIN hosp.admissions a ON a.hadm_id = b.hadm_id),
o AS (SELECT w.stay_id,
        min(CASE WHEN a.t > w.t0 AND a.t <= w.t_icu THEN a.t END) AS ap_icu,
        min(CASE WHEN a.oq AND a.t > w.t0 AND a.t <= w.t_icu THEN a.t END) AS oq_icu,
        min(CASE WHEN a.t > w.t0 AND a.t <= w.t_hosp THEN a.t END) AS ap_hosp,
        min(CASE WHEN a.oq AND a.t > w.t0 AND a.t <= w.t_hosp THEN a.t END) AS oq_hosp
      FROM win w JOIN ap a ON a.hadm_id = w.hadm_id GROUP BY 1),
ca AS (SELECT w.stay_id, count(*) AS n_cam FROM win w JOIN icu.chartevents c ON c.stay_id = w.stay_id AND c.itemid = 228332
         AND c.value IN ('Positive', 'Negative') AND c.charttime > w.t0 AND c.charttime <= w.t_icu GROUP BY 1)
SELECT w.stay_id,
       date_diff('minute', w.t0, o.ap_icu) / 60.0 AS t_ap_icu, date_diff('minute', w.t0, o.oq_icu) / 60.0 AS t_oq_icu,
       date_diff('minute', w.t0, o.ap_hosp) / 60.0 AS t_ap_hosp, date_diff('minute', w.t0, o.oq_hosp) / 60.0 AS t_oq_hosp,
       greatest(date_diff('minute', w.t0, w.t_end_hosp), 0) / 60.0 AS t_end_h, (w.deathtime IS NOT NULL) AS end_death_h,
       (p.stay_id IS NOT NULL) AS ap_pre, coalesce(ca.n_cam, 0) AS n_cam
FROM win w LEFT JOIN o ON o.stay_id = w.stay_id LEFT JOIN pre p ON p.stay_id = w.stay_id LEFT JOIN ca ON ca.stay_id = w.stay_id
"""

E_COV = f"""
WITH base AS (SELECT DISTINCT stay_id, t0 FROM read_parquet({{files}})),
ph AS (SELECT patientunitstayid AS stay_id, max(CASE WHEN pasthistorypath LIKE '%Diabetes%' THEN 1 ELSE 0 END) AS dm_ph
       FROM eicu.pasthistory GROUP BY 1),
apv AS (SELECT patientunitstayid AS stay_id, max(CASE WHEN TRY_CAST(diabetes AS INTEGER) = 1 THEN 1 ELSE 0 END) AS dm_ap
        FROM eicu.apachepredvar GROUP BY 1),
dx AS (SELECT b.stay_id,
         max(CASE WHEN d.diagnosisoffset <= b.t0 AND regexp_matches(lower(d.diagnosisstring), 'diabet') THEN 1 ELSE 0 END) AS dm_dx0,
         max(CASE WHEN regexp_matches(lower(d.diagnosisstring), 'diabet') THEN 1 ELSE 0 END) AS dm_cur,
         max(CASE WHEN regexp_matches(lower(d.diagnosisstring), 'dementia') THEN 1 ELSE 0 END) AS dem_cur,
         max(CASE WHEN regexp_matches(lower(d.diagnosisstring), 'alcohol|etoh') THEN 1 ELSE 0 END) AS alc_cur,
         max(CASE WHEN regexp_matches(lower(d.diagnosisstring), 'cirrhosis|chronic liver|end stage liver|end-stage liver|portal hypertension')
                  THEN 1 ELSE 0 END) AS liv_cur,
         max(CASE WHEN regexp_matches(lower(d.diagnosisstring), 'schizo|bipolar|psychos') THEN 1 ELSE 0 END) AS psy
       FROM base b JOIN eicu.diagnosis d ON d.patientunitstayid = b.stay_id GROUP BY 1),
ii AS (SELECT DISTINCT b.stay_id FROM base b JOIN eicu.infusiondrug i ON i.patientunitstayid = b.stay_id
         AND i.infusionoffset BETWEEN b.t0 - 120 AND b.t0
       WHERE regexp_matches(lower(coalesce(i.drugname, '')), 'insulin') AND TRY_CAST(i.drugrate AS DOUBLE) > 0),
i24 AS (SELECT DISTINCT b.stay_id FROM base b JOIN eicu.infusiondrug i ON i.patientunitstayid = b.stay_id
          AND i.infusionoffset BETWEEN b.t0 - 1440 AND b.t0
        WHERE regexp_matches(lower(coalesce(i.drugname, '')), 'insulin') AND TRY_CAST(i.drugrate AS DOUBLE) > 0
        UNION
        SELECT DISTINCT b.stay_id FROM base b JOIN src.medication m ON m.patientunitstayid = b.stay_id
        WHERE coalesce(m.drugordercancelled, 'No') <> 'Yes' AND regexp_matches(lower(coalesce(m.drugname, '')), 'insulin')
          AND m.drugstartoffset <= b.t0 AND (m.drugstopoffset >= b.t0 - 1440 OR m.drugstopoffset <= m.drugstartoffset))
SELECT b.stay_id, greatest(coalesce(ph.dm_ph, 0), coalesce(apv.dm_ap, 0), coalesce(dx.dm_dx0, 0)) AS diabetes,
       coalesce(dx.dm_cur, 0) AS diabetes_cur, coalesce(dx.dem_cur, 0) AS dementia_cur, coalesce(dx.alc_cur, 0) AS alcohol_cur,
       coalesce(dx.liv_cur, 0) AS liver_cur, coalesce(dx.psy, 0) AS psych,
       (ii.stay_id IS NOT NULL) AS insulin_inf, (i24.stay_id IS NOT NULL) AS insulin_24h
FROM base b LEFT JOIN ph ON ph.stay_id = b.stay_id LEFT JOIN apv ON apv.stay_id = b.stay_id LEFT JOIN dx ON dx.stay_id = b.stay_id
LEFT JOIN ii ON ii.stay_id = b.stay_id LEFT JOIN i24 ON i24.stay_id = b.stay_id
"""

E_OUT = f"""
WITH base AS (SELECT DISTINCT stay_id, t0 FROM read_parquet({{files}})),
fu AS (SELECT b.stay_id, b.t0, least(p.unitdischargeoffset, b.t0 + 10080) AS t_icu
       FROM base b JOIN eicu.patient p ON p.patientunitstayid = b.stay_id),
ap AS (SELECT m.patientunitstayid, m.drugstartoffset AS t,
              (m.drughiclseqno IN ({OQ_HICL}) OR regexp_matches(lower(coalesce(m.drugname, '')), '{OQ}')) AS oq
       FROM src.medication m
       WHERE coalesce(m.drugordercancelled, 'No') <> 'Yes'
         AND (m.drughiclseqno IN ({AP_HICL}) OR regexp_matches(lower(coalesce(m.drugname, '')), '{AP}'))
         AND NOT regexp_matches(lower(coalesce(m.drugname, '')), 'decanoate')),
pre AS (SELECT DISTINCT b.stay_id FROM base b JOIN ap a ON a.patientunitstayid = b.stay_id AND a.t <= b.t0
        UNION
        SELECT DISTINCT b.stay_id FROM base b JOIN eicu.admissiondrug d ON d.patientunitstayid = b.stay_id
        WHERE regexp_matches(lower(coalesce(d.drugname, '')), '{AP}')),
o AS (SELECT f.stay_id, min(CASE WHEN a.t > f.t0 AND a.t <= f.t_icu THEN a.t END) AS ap_icu,
             min(CASE WHEN a.oq AND a.t > f.t0 AND a.t <= f.t_icu THEN a.t END) AS oq_icu
      FROM fu f JOIN ap a ON a.patientunitstayid = f.stay_id GROUP BY 1),
da AS (SELECT f.stay_id, count(*) AS n_cam FROM fu f JOIN eicu.nursecharting n ON n.patientunitstayid = f.stay_id
         AND n.nursingchartcelltypevallabel = 'Delirium Scale/Score' AND n.nursingchartcelltypevalname = 'Delirium Score'
         AND NULLIF(n.nursingchartvalue, '') IS NOT NULL AND n.nursingchartoffset > f.t0 AND n.nursingchartoffset <= f.t_icu
       GROUP BY 1)
SELECT f.stay_id, (o.ap_icu - f.t0) / 60.0 AS t_ap_icu, (o.oq_icu - f.t0) / 60.0 AS t_oq_icu,
       (p.stay_id IS NOT NULL) AS ap_pre, coalesce(da.n_cam, 0) AS n_cam
FROM fu f LEFT JOIN o ON o.stay_id = f.stay_id LEFT JOIN pre p ON p.stay_id = f.stay_id LEFT JOIN da ON da.stay_id = f.stay_id
"""

ROUTE_ORD = f"""
WITH base AS (SELECT DISTINCT hadm_id, stay_id, t0 FROM read_parquet({{files}}))
SELECT b.stay_id, CASE WHEN bool_or(regexp_matches(upper(coalesce(r.route, '')), '^IV')) THEN 'IV'
                       WHEN bool_or(regexp_matches(upper(coalesce(r.route, '')), 'PO|NG|OG|ORAL|TUBE|ENTERAL')) THEN 'EN'
                       ELSE 'UNK' END AS route_ord
FROM base b JOIN hosp.prescriptions r ON r.hadm_id = b.hadm_id AND r.starttime = b.t0
WHERE regexp_matches(lower(r.drug), '{asd.AS_RX}')
GROUP BY 1
"""
EXPECT = {"cov": "stay_id diabetes diabetes_cur dementia_cur alcohol_cur liver_cur psych insulin_inf insulin_24h",
          "out_m": "stay_id t_ap_icu t_oq_icu t_ap_hosp t_oq_hosp t_end_h end_death_h ap_pre n_cam",
          "out_e": "stay_id t_ap_icu t_oq_icu ap_pre n_cam", "route": "stay_id route_ord"}


def extra_statements(P, db):
    both = asd.plist([pf(P, db, "c1_cohort"), pf(P, db, "c2_cohort")])
    c2 = asd.plist([pf(P, db, "c2_cohort")])
    return {"cov": ((M_COV if db == "m" else E_COV).format(files=both), pf(P, db, "r7cov")),
            "out": ((M_OUT if db == "m" else E_OUT).format(files=c2), pf(P, db, "r7out"))}


def extra(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    con = connect(P, args.db)
    sql, path = extra_statements(P, args.db)[args.part]
    asd.write_pq(con, sql, path)
    asd.status(f"{os.path.basename(path)} written", t0)
    asd.log_run(args.results, f"r7 extra --db {args.db} --part {args.part}: ok ({time.time() - t0:.0f} s)")


def variant_statements(P, name):
    db, comparison = VARIANTS[name]
    body = sql_body(os.path.join(asd.SQL_DIR, VARIANT_FILES[name]))
    cohort = pf(P, db, f"{name}_cohort")
    files = asd.plist([cohort])
    st = [("cohort", "WITH " + body + ",\nq_out AS (\n" + asd.COHORT_SELECT[(db, comparison)] + "\n)\nSELECT * FROM q_out", cohort)]
    if db == "m":
        st += [("covars", asd.M_COVARS.format(files=files), pf(P, db, f"{name}_covars")),
               ("follow", asd.M_FOLLOW.format(files=files), pf(P, db, f"{name}_follow")),
               ("outcome", asd.M_OUTCOME.format(files=files), pf(P, db, f"{name}_outcome"))]
        if name == "ord":
            st.append(("route", ROUTE_ORD.format(files=files), pf(P, db, f"{name}_route")))
    else:
        st += [("covars", asd.E_COVARS.format(files=files), pf(P, db, f"{name}_covars")),
               ("follow", asd.E_FOLLOW.format(files=files), pf(P, db, f"{name}_follow")),
               ("outcome", asd.E_OUTCOME.format(files=files), pf(P, db, f"{name}_outcome"))]
    return st


def variant(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    con = asd.connect_slim(P, VARIANTS[args.name][0])
    for part, sql, path in variant_statements(P, args.name):
        asd.write_pq(con, sql, path)
        asd.status(f"{os.path.basename(path)} written", t0)
    asd.log_run(args.results, f"r7 variant --name {args.name}: ok ({time.time() - t0:.0f} s)")


def basecheck(args):
    P, R, RB = os.path.expanduser(args.private), os.path.expanduser(args.results), os.path.expanduser(args.base)
    a = json.load(open(os.path.join(RB, "stage1_design.json"), encoding="utf-8"))
    b = json.load(open(os.path.join(R, "stage1_design.json"), encoding="utf-8"))
    same1 = a["keys"] == b["keys"]
    print(f"{'PASS' if same1 else 'FAIL'} | stage 1 (all keys, all fields except the creation time) equals results/stage1_design.json")
    ok = [same1]
    for key in BASE_KEYS:
        comparison, db = int(key[1]), key[2]
        est = json.load(open(os.path.join(P, f"est_{key}.json")))["res"]["primary"]["OR"]
        r6 = json.load(open(os.path.join(R, f"{asd.STEM[comparison]}_estimates.json"), encoding="utf-8"))["per_db"][db]["analyses"]["primary"]["OR"]
        good = abs(round(est, 4) - r6) < 1e-9
        ok.append(good)
        print(f"{'PASS' if good else 'FAIL'} | {key} primary OR {est:.4f} vs R6 {r6:.4f}")
    asd.log_run(args.results, f"r7 basecheck: {'ok' if all(ok) else 'FAILED'}")
    if not all(ok):
        raise SystemExit("rebuilt base differs from R6: stop")


PROXY_M = ["t_ap_icu", "t_oq_icu", "t_ap_hosp", "t_oq_hosp", "t_end_h", "n_cam"]
PROXY_E = ["t_ap_icu", "t_oq_icu", "n_cam"]
NEWCOV = ["diabetes", "insulin_inf", "insulin_24h", "diabetes_cur", "dementia_cur", "alcohol_cur", "liver_cur", "dementia", "alcohol", "liver"]


def load_base(P, key):
    comparison, db = int(key[1]), key[2]
    df, flow = asd.load_key(P, key, stage2=True)
    n0 = len(df)
    df = df.merge(asd.read_pq(pf(P, db, "r7cov")), on="stay_id", how="left", validate="one_to_one")
    if comparison == 2:
        df = df.merge(asd.read_pq(pf(P, db, "r7out")), on="stay_id", how="left", validate="one_to_one")
    if len(df) != n0 or df["diabetes"].isna().any():
        raise SystemExit(f"{key}: R7 rows missing")
    return df, flow


def load_variant(P, name):
    db, comparison = VARIANTS[name]
    co = asd.read_pq(pf(P, db, f"{name}_cohort"))
    flow = {"eligible": len(co), "A": int((co["arm"] == "A").sum()), "B": int((co["arm"] == "B").sum()), "other": int(co["arm"].isna().sum())}
    df = co[co["arm"].isin(["A", "B"])].copy()
    for part in ("covars", "follow", "outcome"):
        df = df.merge(asd.read_pq(pf(P, db, f"{name}_{part}")), on="stay_id", how="left", validate="one_to_one")
    if name == "ord":
        df = df.merge(asd.read_pq(pf(P, db, f"{name}_route")), on="stay_id", how="left", validate="one_to_one")
        df["route"] = df["route_ord"].fillna("UNK")
    df["t_dev"] = np.nan
    keep = asd.overlap_keep(df, db == "e")
    flow["outside_overlap"] = int((~keep).sum())
    df = df[keep].sort_values("stay_id", kind="mergesort").reset_index(drop=True)
    flow["analysis_A"], flow["analysis_B"] = int((df["arm"] == "A").sum()), int((df["arm"] == "B").sum())
    return df, flow


def d_model(df, db, comparison, model):
    dfm = df.copy()
    if model == "m2":
        for c in ("dementia", "alcohol", "liver"):
            dfm[c] = np.maximum(nz(dfm[c]), nz(dfm[c + "_cur"]))
    d = asd.build(dfm, db, comparison, stage2=True)
    if model in ("m1", "m2"):
        dm = nz(dfm["diabetes"]) if model == "m1" else np.maximum(nz(dfm["diabetes"]), nz(dfm["diabetes_cur"]))
        extra_cols = np.column_stack([dm == 1, nz(dfm["insulin_inf"]) == 1, nz(dfm["insulin_24h"]) == 1]).astype(float)
        X, names, dropped = asd.prevalence_filter(np.column_stack([d["X"], extra_cols]), d["names"] + ["diabetes", "insulin_inf_t0", "insulin_24h"])
        d["X"], d["names"], d["dropped"] = X, names, d["dropped"] + dropped
    return d


def make_dd(df, key):
    if key in VARIANTS:
        db, comparison = VARIANTS[key]
        return {"m0": asd.build(df, db, comparison, stage2=True)}
    comparison, db = int(key[1]), key[2]
    dd = {m: d_model(df, db, comparison, m) for m in ("m0", "m1", "m2")}
    if comparison == 2:
        d0 = dd["m0"]
        for c in (PROXY_M if db == "m" else PROXY_E):
            d0[c] = asd.num(df, c)
        d0["ap_pre"] = nz(df["ap_pre"]) == 1
        d0["psych"] = nz(df["psych"]) == 1
        if db == "m":
            d0["end_death_h"] = nz(df["end_death_h"]) == 1
    return dd


def hosp_contrast(d, w, t_ev):
    A = d["A"]
    kend, typ, _ = sc.follow_types(d["t_end_h"], d["end_death_h"], t_ev, disc_competing=True)
    W = np.asarray(w, float)
    r = {a: sc.aj_cif(W[A == a], kend[A == a], typ[A == a]) for a in (1, 0)}
    e = sc.effects(r[1][sc.H - 1], r[0][sc.H - 1])
    e["eventsA"], e["eventsB"] = int(((typ == 1) & (A == 1)).sum()), int(((typ == 1) & (A == 0)).sum())
    e["nA"], e["nB"] = int((A == 1).sum()), int((A == 0).sum())
    return e


def names_for(key):
    if key in VARIANTS:
        return ["m0_primary", "m0_negctl"]
    n = [f"{m}_{o}" for m in ("m0", "m1", "m2") for o in ("primary", "negctl")]
    if key[1] == "2":
        n += ["m2_rass3", "px_cam", "px_icu", "px_icu_oq"] + (["px_hosp", "px_hosp_oq"] if key == "c2m" else [])
    return n


def proxy_mask(d0):
    return ~d0["ap_pre"] & ~d0["psych"]


def analyses(dd, key, cache):
    res = {}
    for m, d in dd.items():
        ow = sc.overlap_w(asd.ow_fit(d, cache, f"{m}:full"), d["A"])
        res[f"{m}_primary"] = asd.cif_contrast(d, ow, d["t_del"])[168]
        res[f"{m}_negctl"] = asd.cif_contrast(d, ow, d["t_hypo"])[168]
        if m == "m2" and key in ("c2m", "c2e"):
            ds = asd.subset_d(d, d["sub"]["rass3"])
            res["m2_rass3"] = asd.cif_contrast(ds, sc.overlap_w(asd.ow_fit(ds, cache, "m2:rass3"), ds["A"]), ds["t_del"])[168]
        if m == "m0" and key in ("c2m", "c2e"):
            ds = asd.subset_d(d, proxy_mask(d))
            ows = sc.overlap_w(asd.ow_fit(ds, cache, "m0:px"), ds["A"])
            res["px_cam"] = asd.cif_contrast(ds, ows, ds["t_del"])[168]
            res["px_icu"] = asd.cif_contrast(ds, ows, ds["t_ap_icu"])[168]
            res["px_icu_oq"] = asd.cif_contrast(ds, ows, ds["t_oq_icu"])[168]
            if key == "c2m":
                res["px_hosp"] = hosp_contrast(ds, ows, ds["t_ap_hosp"])
                res["px_hosp_oq"] = hosp_contrast(ds, ows, ds["t_oq_hosp"])
    return res


def balance(d, cache, m):
    ow = sc.overlap_w(asd.ow_fit(d, cache, f"{m}:full"), d["A"])
    s = sc.smd_table(d["X"], d["names"], d["A"], ow)
    v = [abs(x) for x in s.values() if np.isfinite(x)]
    return {"max_abs_smd_ow": max(v) if v else float("nan"), "columns": len(d["names"]), "dropped": d["dropped"]}


def describe_new(df, dd, cache):
    d2 = dd["m2"]
    A = d2["A"]
    ow = sc.overlap_w(asd.ow_fit(d2, cache, "m2:full"), A)
    out = {}
    for c in NEWCOV:
        x = (nz(df[c]) == 1).astype(float)
        if c in ("dementia", "alcohol", "liver"):
            x = np.maximum(x, (nz(df[c + "_cur"]) == 1).astype(float))
            c = c + "_any"
        un = sc.smd_table(x[:, None], [c], A)[c]
        wt = sc.smd_table(x[:, None], [c], A, ow)[c]
        out[c] = {"nA": int(x[A == 1].sum()), "nB": int(x[A == 0].sum()), "NA": int((A == 1).sum()), "NB": int((A == 0).sum()),
                  "smd_unweighted": un, "smd_ow_m2": wt}
    return out


def proxy_desc(d0):
    m = proxy_mask(d0)
    A = d0["A"]
    cam = np.isfinite(d0["t_del"]) & m
    ap = np.isfinite(d0["t_ap_icu"]) & m
    assessed = (d0["n_cam"] > 0) & m
    neg = assessed & ~cam
    un = m & ~assessed
    return {"excluded_ap_before": int(d0["ap_pre"].sum()), "excluded_psych": int((d0["psych"] & ~d0["ap_pre"]).sum()),
            "nA": int((m & (A == 1)).sum()), "nB": int((m & (A == 0)).sum()),
            "cam_pos": int(cam.sum()), "cam_pos_ap": int((cam & ap).sum()),
            "assessed_neg": int(neg.sum()), "assessed_neg_ap": int((neg & ap).sum()),
            "unassessed": int(un.sum()), "unassessed_ap": int((un & ap).sum()),
            "ap_total": int(ap.sum())}


def dose_crosstab(P):
    new = asd.read_pq(pf(P, "e", "dose_cohort"))[["stay_id", "arm"]].rename(columns={"arm": "new"})
    old = asd.read_pq(pf(P, "e", "c1_cohort"))[["stay_id", "arm"]].rename(columns={"arm": "old"})
    x = old.merge(new, on="stay_id", how="outer", validate="one_to_one")
    x = x.fillna({"old": "other", "new": "other"})
    return {f"{o}->{n}": int(((x["old"] == o) & (x["new"] == n)).sum()) for o in ("A", "B", "other") for n in ("A", "B", "other")}


def load_dd(P, key):
    df, flow = (load_variant(P, key) if key in VARIANTS else load_base(P, key))
    return df, flow, make_dd(df, key)


def estimate(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    for key in ALL_KEYS:
        df, flow, dd = load_dd(P, key)
        cache = {}
        res = analyses(dd, key, cache)
        desc = {"balance": {m: balance(d, cache, m) for m, d in dd.items()}}
        if key in BASE_KEYS:
            desc["newcov"] = describe_new(df, dd, cache)
            if key[1] == "2":
                desc["proxy"] = proxy_desc(dd["m0"])
        if key == "dose":
            desc["crosstab"] = dose_crosstab(P)
        json.dump({"flow": flow, "res": asd.to_json(res), "desc": asd.to_json(desc)}, open(os.path.join(P, f"r7_est_{key}.json"), "w"))
        np.savez(os.path.join(P, f"r7_cache_{key}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
        asd.status(f"r7 estimate {key} done", t0)
    asd.log_run(args.results, f"r7 estimate: ok ({time.time() - t0:.0f} s)")


_G = {}


def draw(d0, db, rng):
    n = len(d0["A"])
    if db == "m":
        return rng.integers(0, n, n), None
    site = d0["site"]
    hs = np.unique(site)
    pick = rng.choice(hs, size=hs.size, replace=True)
    rows = {h: np.nonzero(site == h)[0] for h in hs}
    idx = np.concatenate([rows[h] for h in pick])
    copy = np.concatenate([np.full(rows[h].size, i) for i, h in enumerate(pick)])
    return idx, copy


def replicate(dd, key, db, rep, cache):
    rng = np.random.default_rng(asd.SEED + rep)
    idx, copy = draw(dd["m0"], db, rng)
    bd = {}
    for m, d in dd.items():
        b = asd.take(d, idx)
        if copy is not None:
            b["site"] = b["site"] * 1000 + copy
        bd[m] = b
    return analyses(bd, key, dict(cache))


def _boot_one(rep):
    dd, key, db, cache = _G["dd"], _G["key"], _G["db"], _G["cache"]
    try:
        res = replicate(dd, key, db, rep, cache)
    except Exception:
        return rep, None
    names = names_for(key)
    lor = [math.log(res[n]["OR"]) if np.isfinite(res[n].get("OR", np.nan)) and res[n]["OR"] > 0 else np.nan for n in names]
    rd = [res[n]["RD"] if np.isfinite(res[n].get("RD", np.nan)) else np.nan for n in names]
    return rep, (lor, rd)


def key_db(key):
    return VARIANTS[key][0] if key in VARIANTS else key[2]


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    key = args.key
    _, _, dd = load_dd(P, key)
    z = np.load(os.path.join(P, f"r7_cache_{key}.npz"))
    _G.update(dd=dd, key=key, db=key_db(key), cache={k.replace("__", ":"): z[k] for k in z.files})
    reps = list(range(args.start, min(args.stop, NBOOT)))
    out = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for rep, r in pool.imap_unordered(_boot_one, reps, chunksize=10):
                out[rep] = r
    else:
        for rep in reps:
            out[rep] = _boot_one(rep)[1]
    names = names_for(key)
    lor = np.full((len(reps), len(names)), np.nan)
    rd = np.full((len(reps), len(names)), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None:
            lor[i], rd[i] = out[rep]
    np.savez(os.path.join(P, f"r7_boot_{key}_{args.start:04d}_{args.stop:04d}.npz"), reps=np.array(reps), lor=lor, rd=rd, names=np.array(names))
    failed = sum(1 for r in reps if out.get(r) is None)
    asd.status(f"r7 boot {key} {args.start}-{args.stop} done, failed {failed}", t0)
    asd.log_run(args.results, f"r7 boot --key {key} --start {args.start} --stop {args.stop}: ok, failed {failed} ({time.time() - t0:.0f} s)")


def load_boot(P, key):
    files = sorted(glob.glob(os.path.join(P, f"r7_boot_{key}_*.npz")))
    if not files:
        return None
    reps, lor, rd, names = [], [], [], None
    for f in files:
        z = np.load(f)
        reps.append(z["reps"])
        lor.append(z["lor"])
        rd.append(z["rd"])
        names = [str(x) for x in z["names"]]
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return {"names": names, "lor": np.concatenate(lor)[first], "rd": np.concatenate(rd)[first], "n": int(first.size)}


def ci(col):
    col = col[np.isfinite(col)]
    if col.size < asd.MIN_BOOT:
        return None
    return float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5)), float(col.std(ddof=1)), int(col.size)


def summarise(E, B, key):
    out = {}
    res = E[key]["res"]
    b = B.get(key)
    for i, n in enumerate(names_for(key)):
        e = dict(res[n])
        if b is not None:
            c = ci(b["lor"][:, i])
            r = ci(b["rd"][:, i])
            if c:
                e["OR_lo"], e["OR_hi"], e["se_logOR"], e["boot_ok"] = math.exp(c[0]), math.exp(c[1]), c[2], c[3]
            if r:
                e["RD_lo"], e["RD_hi"] = r[0], r[1]
        out[n] = e
    return out


def pool(a, b):
    def ok(e):
        return bool(e) and isinstance(e.get("OR"), float) and e["OR"] > 0 and bool(e.get("se_logOR"))
    if not (ok(a) and ok(b)):
        return None
    m, se, Q, I2, k = sc.pool_fe([math.log(a["OR"]), math.log(b["OR"])], [a["se_logOR"], b["se_logOR"]])
    return {"OR": math.exp(m), "OR_lo": math.exp(m - 1.96 * se), "OR_hi": math.exp(m + 1.96 * se), "I2": I2, "Q": Q}


def ratio(B, key, num, den):
    b = B.get(key)
    if b is None:
        return None
    i, j = b["names"].index(num), b["names"].index(den)
    c = ci(b["lor"][:, i] - b["lor"][:, j])
    return None if c is None else {"lo": math.exp(c[0]), "hi": math.exp(c[1])}


def read_adjust(m2m, pooled_m2):
    p, nc = m2m.get("m2_primary", {}), m2m.get("m2_negctl", {})
    if "OR_lo" not in p or "OR_lo" not in nc:
        return "no interval"
    if p["OR_lo"] <= 1 <= p["OR_hi"]:
        return "the MIMIC-IV association no longer holds after additional adjustment"
    if nc["OR_lo"] <= 1 <= nc["OR_hi"] and p["OR_hi"] < 1:
        return "the negative-control signal comes mainly from glucose-related imbalance; the MIMIC-IV association does not disappear with it"
    if not (nc["OR_lo"] <= 1 <= nc["OR_hi"]):
        return "residual confounding still cannot be ruled out"
    return "not applicable"


def read_bridge(px, rt):
    if not px or "OR_hi" not in px or not rt:
        return "no interval"
    if rt["lo"] <= 1 <= rt["hi"]:
        return "the proxy outcome reproduces the CAM-ICU difference" if px["OR_hi"] < 1 else "same direction, but the proxy outcome lacks precision"
    return "the proxy outcome does not reproduce it"


def report(args):
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    E = {k: json.load(open(os.path.join(P, f"r7_est_{k}.json"))) for k in ALL_KEYS}
    B = {k: load_boot(P, k) for k in ALL_KEYS}
    S = {k: summarise(E, B, k) for k in ALL_KEYS}
    out = {"plan": "SAP_addendum_R7.md v1.0", "boot_reps": {k: (B[k]["n"] if B[k] else 0) for k in ALL_KEYS},
           "flow": {}, "per_key": S, "pooled": {}, "ratio": {}, "reading": {}, "desc": {k: E[k]["desc"] for k in ALL_KEYS},
           "flow_variant": {k: E[k]["flow"] for k in VARIANTS}, "r6_check": {}}
    for db in ("m", "e"):
        f = os.path.join(P, f"{db}_r7flow.json")
        if os.path.exists(f):
            out["flow"][db] = json.load(open(f))
    for comparison, (km, ke) in ((2, ("c2m", "c2e")), (1, ("c1m", "c1e"))):
        for n in names_for(km):
            if n in S[ke]:
                out["pooled"][f"{asd.STEM[comparison]}:{n}"] = pool(S[km][n], S[ke][n])
    for v, (a, b) in {"nopdx": ("nopdx", "c2e"), "ord": ("ord", "c2e"), "dose": ("c1m", "dose")}.items():
        for n in ("m0_primary", "m0_negctl"):
            out["pooled"][f"{v}:{n}"] = pool(S[a][n], S[b][n])
    for k in ("c2m", "c2e"):
        out["ratio"][k] = ratio(B, k, "px_icu", "px_cam")
        out["reading"][f"bridge_{k}"] = read_bridge(S[k].get("px_icu"), out["ratio"][k])
    out["reading"]["adjust_mimic"] = read_adjust(S["c2m"], out["pooled"].get("ppi:m2_primary"))
    for name, p in out["pooled"].items():
        if p and name.endswith("primary") or (p and name.endswith("rass3")):
            out["reading"][f"pooled:{name}"] = asd.interpret(p["OR"], p["OR_lo"], p["OR_hi"])
    for key in BASE_KEYS:
        comparison, db = int(key[1]), key[2]
        r6 = json.load(open(os.path.join(R, f"{asd.STEM[comparison]}_estimates.json"), encoding="utf-8"))["per_db"][db]["analyses"]["primary"]
        m0 = S[key]["m0_primary"]
        same = all(isinstance(m0.get(x), float) and abs(round(m0[x], 4) - r6[x]) < 1e-9 for x in ("OR", "OR_lo", "OR_hi"))
        out["r6_check"][key] = {"same": bool(same), "r7": [m0.get("OR"), m0.get("OR_lo"), m0.get("OR_hi")], "r6": [r6["OR"], r6["OR_lo"], r6["OR_hi"]]}
    out["proxy_pct"] = {}
    for k in ("c2m", "c2e"):
        d = E[k]["desc"]["proxy"]
        out["proxy_pct"][k] = {a: sc.pct_text(d[a], d[b]) for a, b in (("cam_pos_ap", "cam_pos"), ("assessed_neg_ap", "assessed_neg"),
                                                                        ("unassessed_ap", "unassessed"))}
    masked = mask_tree(out)
    json.dump(asd.round_floats(masked), open(os.path.join(R, "r7.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(R, "r7.md"), "w", encoding="utf-8").write(md(masked))
    asd.log_run(args.results, "r7 report: ok")
    print("report written", flush=True)


COUNT_KEYS = re.compile(r"^(n|N|nA|nB|NA|NB|eventsA|eventsB|eligible|A|B|other|top|before_top|excluded|cohort_eligible|analysis_A|"
                        r"analysis_B|outside_overlap|x\d.*|any_.*|bleeding_source_.*|only_principal_dx_.*|cam_pos.*|assessed_neg.*|"
                        r"unassessed.*|ap_total|excluded_.*|.*->.*)$")


def mask_tree(o, key=""):
    if isinstance(o, dict):
        return {k: mask_tree(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [mask_tree(v, key) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)) and COUNT_KEYS.match(key):
        return sc.mask_n(int(o))
    return o


def mn(x):
    return str(x)


def i2(p):
    return "" if not p else "; I² {:.0f}%".format(100 * p["I2"])


def orci(e):
    if not e:
        return "—"
    lo, hi = e.get("OR_lo"), e.get("OR_hi")
    s = asd.fmt(e.get("OR"))
    return s + (f" ({asd.fmt(lo)}–{asd.fmt(hi)})" if lo is not None and hi is not None else "")


def md(o):
    L = ["# Revision R7 results: exclusion counts, additional adjustment, proxy-outcome bridge, three variants", "",
         f"Plan: {o['plan']}. Bootstrap replicates: {', '.join(f'{k} {v}' for k, v in o['boot_reps'].items())}. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5; "
         "percentages only when the numerator is ≥10. Odds ratios A vs B (PPI comparison: A = PPI, B = famotidine; dose comparison: A = ≤20 mg/day, B = ≥40 mg/day).", ""]
    L += ["## 0. Reproduction check (M0 and the R6 main analysis)", ""]
    for k, v in o["r6_check"].items():
        L.append(f"- {k}: {'same' if v['same'] else 'different'} (R7 {asd.fmt(v['r7'][0])} ({asd.fmt(v['r7'][1])}–{asd.fmt(v['r7'][2])}), "
                 f"R6 {asd.fmt(v['r6'][0])} ({asd.fmt(v['r6'][1])}–{asd.fmt(v['r6'][2])}))")
    L += ["", "## 1. Exclusion counts (Figure 1)", ""]
    lab2 = [("before_top", "Adults at a first ICU stay within the same scope"), ("top", "of whom started an acid suppressant in the ICU"), ("x1_prior_or_home", "Excluded: used in the 24 h before time zero or taken before admission"),
            ("x2_delirium_before", "Excluded: positive delirium screen before the first dose"), ("x3_bleeding", "Excluded: bleeding or ulcer indication, PPI infusion"),
            ("x4_other_first_drug", "Excluded: first was ranitidine or cimetidine, or both classes at once"), ("eligible", "Eligible"), ("A", "PPI"), ("B", "famotidine")]
    lab1 = [("top", "Started an acid suppressant in the ICU"), ("x0_first_not_famotidine", "Excluded: first was not famotidine"), ("x1_prior_or_home", "Excluded: used in the 24 h before time zero or taken before admission"),
            ("x2_delirium_before", "Excluded: positive delirium screen before the first dose"), ("x3_no_creatinine", "Excluded: no creatinine in the 24 h before time zero"), ("x4_egfr_ge60", "Excluded: eGFR ≥60"),
            ("eligible", "Eligible"), ("A", "≤20 mg/day"), ("B", "≥40 mg/day"), ("other", "Other strategies")]
    for db, nm in (("m", "MIMIC-IV"), ("e", "eICU")):
        f = o["flow"].get(db)
        if not f:
            continue
        L += [f"### {nm}", "", "| Step | PPI comparison | | Step | Dose comparison |", "|---|---|---|---|---|"]
        for i in range(max(len(lab2), len(lab1))):
            a = lab2[i] if i < len(lab2) else ("", "")
            b = lab1[i] if i < len(lab1) else ("", "")
            L.append(f"| {a[1]} | {mn(f['ppi'].get(a[0], '')) if a[0] else ''} | | {b[1]} | {mn(f['famdose'].get(b[0], '')) if b[0] else ''} |")
        extra_l = [f"{k.replace('bleeding_source_', 'bleeding source ')} {mn(v)}" for k, v in f["ppi"].items() if k.startswith("bleeding_source_")]
        L += ["", "- PPI comparison bleeding exclusions by source (patients in the sequential count): " + "; ".join(extra_l)]
        L.append("- Patients meeting each criterion on its own (regardless of order): " + "; ".join(f"{k[4:]} {mn(v)}" for k, v in f["ppi"].items() if k.startswith("any_")))
        if db == "m":
            L.append(f"- Excluded only because of the principal discharge diagnosis, all other criteria met (the increment of variant 4a): PPI {mn(f['ppi']['only_principal_dx_A'])}, "
                     f"famotidine {mn(f['ppi']['only_principal_dx_B'])}")
        L.append(f"- Gate reproduced: {json.dumps(f['gate_check'], ensure_ascii=False)}")
        L.append("")
    L += ["## 2. Additional adjustment", "", "| Model and outcome | Comparison | MIMIC-IV | eICU | Pooled (I²) |", "|---|---|---|---|---|"]
    lab = {"m0_primary": "M0 primary outcome (R6)", "m0_negctl": "M0 negative control (R6)", "m1_primary": "M1 primary outcome", "m1_negctl": "M1 negative control",
           "m2_primary": "M2 primary outcome", "m2_negctl": "M2 negative control", "m2_rass3": "M2 RASS ≥ −3 at time zero"}
    for comparison, (km, ke) in ((2, ("c2m", "c2e")), (1, ("c1m", "c1e"))):
        for n, t in lab.items():
            if n not in o["per_key"][km]:
                continue
            p = o["pooled"].get(f"{asd.STEM[comparison]}:{n}")
            L.append(f"| {t} | {asd.SHORT[comparison]} | {orci(o['per_key'][km][n])} | {orci(o['per_key'][ke][n])} | "
                     f"{orci(p)}{i2(p)} |")
    L += ["", f"- Reading (MIMIC-IV negative control, written rule): {o['reading']['adjust_mimic']}"]
    for k in ("ppi:m2_primary", "ppi:m2_rass3", "famdose:m2_primary"):
        if f"pooled:{k}" in o["reading"]:
            L.append(f"- Pooled {k}: {o['reading'][f'pooled:{k}']}")
    L += ["", "New variables (as in M2; dementia, alcohol and liver disease with current-admission codes added are labelled _any): counts by arm and standardised mean differences:", "",
          "| Variable | Database-comparison | Arm A patients | Arm B patients | SMD before weighting | SMD after overlap weighting (M2) |", "|---|---|---|---|---|---|"]
    for k in BASE_KEYS:
        for c, v in o["desc"][k]["newcov"].items():
            L.append(f"| {c} | {k} | {mn(v['nA'])} | {mn(v['nB'])} | {v['smd_unweighted']:+.3f} | {v['smd_ow_m2']:+.3f} |")
    L += ["", "## 3. Proxy-outcome bridge (PPI comparison)", "", "| Analysis | MIMIC-IV | eICU | Pooled (I²) |", "|---|---|---|---|"]
    lab3 = {"px_cam": "CAM-ICU primary outcome in the proxy subset", "px_icu": "First antipsychotic in the ICU", "px_icu_oq": "First olanzapine or quetiapine in the ICU (eICU: quetiapine only)",
            "px_hosp": "First antipsychotic in the hospital (as MOVER)", "px_hosp_oq": "First olanzapine or quetiapine in the hospital"}
    for n, t in lab3.items():
        p = o["pooled"].get(f"ppi:{n}")
        L.append(f"| {t} | {orci(o['per_key']['c2m'].get(n))} | {orci(o['per_key']['c2e'].get(n))} | "
                 f"{orci(p)}{i2(p)} |")
    for k, nm in (("c2m", "MIMIC-IV"), ("c2e", "eICU")):
        rt = o["ratio"].get(k)
        e = o["per_key"][k]
        a, b = (e.get("px_icu") or {}).get("OR"), (e.get("px_cam") or {}).get("OR")
        if rt and isinstance(a, float) and isinstance(b, float):
            L.append(f"- {nm}: ratio of the odds ratios, proxy outcome in the ICU to CAM-ICU, {asd.fmt(a / b)} "
                     f"({asd.fmt(rt['lo'])}–{asd.fmt(rt['hi'])}); reading: {o['reading'][f'bridge_{k}']}")
    L += ["", "Description of the proxy outcome (proxy subset, arms combined, unweighted):", ""]
    for k, nm in (("c2m", "MIMIC-IV"), ("c2e", "eICU")):
        d, pc = o["desc"][k]["proxy"], o["proxy_pct"][k]
        L.append(f"- {nm}: subset A {mn(d['nA'])} / B {mn(d['nB'])} (excluded for an antipsychotic before time zero {mn(d['excluded_ap_before'])}, schizophrenia-spectrum or bipolar disorder "
                 f"{mn(d['excluded_psych'])}); CAM-ICU positive {mn(d['cam_pos'])}, of whom given an antipsychotic in the ICU {mn(d['cam_pos_ap'])}"
                 f" ({pc['cam_pos_ap']}); assessed and always negative {mn(d['assessed_neg'])}, of whom given one {mn(d['assessed_neg_ap'])}"
                 f" ({pc['assessed_neg_ap']}); not assessed {mn(d['unassessed'])}, of whom given one {mn(d['unassessed_ap'])}"
                 f" ({pc['unassessed_ap']})")
    L += ["", "## 4. Three variants", "", "| Variant | Outcome | Database with the variant | Other database (M0) | Pooled (I²) |", "|---|---|---|---|---|"]
    for v, (a, b, t) in {"nopdx": ("nopdx", "c2e", "4a MIMIC-IV bleeding exclusion without the principal discharge diagnosis"), "ord": ("ord", "c2e", "4b MIMIC-IV time zero and arm from orders"),
                         "dose": ("dose", "c1m", "4c eICU dose comparison, dose classification recovered")}.items():
        for n, tn in (("m0_primary", "primary outcome"), ("m0_negctl", "negative control")):
            p = o["pooled"].get(f"{v}:{n}")
            L.append(f"| {t} | {tn} | {orci(o['per_key'][a][n])} | {orci(o['per_key'][b][n])} | "
                     f"{orci(p)}{i2(p)} |")
    for v in VARIANTS:
        f = o["flow_variant"][v]
        L.append(f"- {v}: eligible {mn(f['eligible'])}, A {mn(f['A'])}, B {mn(f['B'])}, other strategies {mn(f['other'])}; "
                 f"largest |SMD| after overlap weighting {o['desc'][v]['balance']['m0']['max_abs_smd_ow']:.3f}")
    ct = o["desc"]["dose"].get("crosstab")
    if ct:
        L.append("- 4c cross-tabulation with the original arms (original→new): " + "; ".join(f"{k} {mn(v)}" for k, v in ct.items()))
    for name in ("nopdx:m0_primary", "ord:m0_primary", "dose:m0_primary"):
        if f"pooled:{name}" in o["reading"]:
            L.append(f"- Pooled {name}: {o['reading'][f'pooled:{name}']}")
    L.append("")
    return "\n".join(L)


def selftest(args):
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)

    rng = np.random.default_rng(7)
    n = 2500

    def synth(db, extra=False):
        X = rng.normal(size=(n, 5))
        A = (rng.random(n) < 1 / (1 + np.exp(-0.5 * X[:, 0]))).astype(int)
        t_end = rng.uniform(2, 240, n)
        d = {"X": X, "names": [f"x{j}" for j in range(5)], "dropped": [], "extras": {}, "A": A,
             "site": (rng.integers(0, 15, n).astype(float) if db == "e" else None), "t_end": t_end,
             "end_death": rng.random(n) < 0.1, "t_dev": np.full(n, np.nan), "sub": {"rass3": rng.random(n) < 0.7},
             "t_del": np.where(rng.random(n) < 0.3 - 0.05 * A, rng.uniform(0, 150, n), np.nan),
             "t_hypo": np.where(rng.random(n) < 0.1, rng.uniform(0, 150, n), np.nan),
             "hosp_death": (rng.random(n) < 0.12).astype(int)}
        if extra:
            d["X"] = np.column_stack([X, rng.random(n) < 0.3])
            d["names"] = d["names"] + ["diabetes"]
        return d

    for db, key in (("m", "c2m"), ("e", "c2e")):
        d0 = synth(db)
        rows = np.arange(n)
        d0["t_ap_icu"] = np.where(rng.random(n) < 0.08, rng.uniform(0, 150, n), np.nan)
        d0["t_oq_icu"] = np.where(np.isfinite(d0["t_ap_icu"]) & (rng.random(n) < 0.5), d0["t_ap_icu"], np.nan)
        d0["n_cam"] = rng.integers(0, 6, n).astype(float)
        d0["ap_pre"] = rng.random(n) < 0.05
        d0["psych"] = rng.random(n) < 0.03
        if db == "m":
            d0["t_end_h"] = d0["t_end"] + rng.uniform(0, 200, n)
            d0["end_death_h"] = rng.random(n) < 0.1
            d0["t_ap_hosp"] = np.where(rng.random(n) < 0.12, rng.uniform(0, 160, n), np.nan)
            d0["t_oq_hosp"] = np.where(np.isfinite(d0["t_ap_hosp"]) & (rng.random(n) < 0.5), d0["t_ap_hosp"], np.nan)
        dd = {"m0": d0, "m1": synth(db, True), "m2": synth(db, True)}
        for m in ("m1", "m2"):
            for k in d0:
                if k not in ("X", "names", "dropped"):
                    dd[m][k] = d0[k]
        cache = {}
        res = analyses(dd, key, cache)
        okn = all(np.isfinite(res[x]["OR"]) for x in names_for(key))
        check(f"{key}: every R7 analysis gives a finite odds ratio on synthetic data", okn, ", ".join(names_for(key)))
        rep = 3
        mine = replicate(dd, key, db, rep, cache)["m0_primary"]["OR"]
        b = asd.resample(d0, db, np.random.default_rng(asd.SEED + rep))
        ref = asd.cif_contrast(b, sc.overlap_w(asd.ow_fit(b, dict(cache), "m0:full"), b["A"]), b["t_del"])[168]["OR"]
        check(f"{key}: paired replicate equals asd.resample for model M0", abs(mine - ref) < 1e-10, f"{mine:.6f} vs {ref:.6f}")
        del rows
    d = synth("m")
    d["t_end_h"] = np.full(n, 50.0)
    d["end_death_h"] = np.zeros(n, bool)
    late = np.full(n, 80.0)
    e = hosp_contrast(d, np.ones(n), late)
    check("hospital-wide proxy: doses after the end of follow-up do not count", e["eventsA"] + e["eventsB"] == 0)
    early = np.full(n, 10.0)
    e = hosp_contrast(d, np.ones(n), early)
    check("hospital-wide proxy: doses before the end of follow-up count", e["eventsA"] + e["eventsB"] == n)
    for name in VARIANTS:
        a, b = variant_sql(name), variant_sql(name)
        check(f"variant {name}: generation is deterministic", a == b)
    check("variant nopdx: principal-diagnosis source removed, ED source kept",
          "seq_num = 1" not in variant_sql("nopdx") and "FROM ed.diagnosis d" in variant_sql("nopdx"))
    check("variant ord: exposure from orders with the eICU-style washout",
          "FROM hosp.prescriptions r" in variant_sql("ord") and "INTERVAL 7 DAY" in variant_sql("ord")
          and "FROM hosp.emar e" not in cte_blocks(sql_body_text(variant_sql("ord")))["asadm"])
    check("variant dose: recovered parser and the three-step rule", all(s in variant_sql("dose") for s in ("first_as0 AS (", "nxt AS (", "'x3'", "daily_next")))
    for db in ("m", "e"):
        try:
            main, pre = flow_sql(db)
            check(f"flow SQL ({db}): built, gate conditions found verbatim", "fl AS (" in main and "SELECT count(*)" in pre)
        except SystemExit as ex:
            check(f"flow SQL ({db})", False, str(ex))
    fl = pd.DataFrame({"era": ["2011 - 2013"] * 6 + ["2008 - 2010"], "drug": ["PPI", "FAM", "FAM", None, "PPI", "H2O", "PPI"],
                       "x_prior24": [False, True, False, False, False, False, False], "x_home": [False] * 7,
                       "x_del": [False, False, True, False, False, False, False], "x_gib_ed": [False] * 7,
                       "x_gib_pdx": [False, False, False, False, True, False, False], "x_inf": [False] * 7,
                       "scr": [1.0, 2.0, 1.0, None, 1.0, 1.0, 1.0], "egfr": [80.0, 30.0, 90.0, None, 80.0, 80.0, 80.0]})
    r = flow_counts(fl, "m", 10, {"cohort_eligible": 0, "A": 0, "B": 0, "other": 0})
    want = {"top": 6, "x1_prior_or_home": 1, "x2_delirium_before": 1, "x3_bleeding": 1, "x4_other_first_drug": 2, "eligible": 1, "A": 1, "B": 0}
    check("flow counts: hierarchical attribution on a toy table", all(r["ppi"][k] == v for k, v in want.items()), json.dumps(r["ppi"])[:120])
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


def sql_body_text(txt):
    body = re.sub(r"^\s*--.*$", "", txt, flags=re.M).strip().rstrip(";")
    return re.sub(r"^\s*WITH\s+", "", body, flags=re.I)


def bindcheck(args):
    import duckdb
    import sql_bind_check as sbc
    t0 = time.time()
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)

    tmp = tempfile.mkdtemp(prefix="r7_bind_")
    try:
        for db, dbn in (("m", "mimic"), ("e", "eicu")):
            con = duckdb.connect()
            con.execute("SET threads = 2")
            mods = ("hosp", "icu", "ed", "note", "derived") if db == "m" else ("eicu", "eicu_derived")
            sbc.empty_tables(con, SRC[db], mods)
            con.execute("CREATE SCHEMA src")
            for m, tb in ([("hosp", "emar"), ("icu", "inputevents")] if db == "m" else [("eicu", "medication")]):
                con.execute(f'CREATE TABLE src."{tb}" AS SELECT * FROM {m}."{tb}" LIMIT 0')
            for sql, path in asd.all_sql(tmp, dbn)["cohort"]:
                con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")
            st = []
            main, pre = flow_sql(db)
            st += [("flow", main, os.path.join(tmp, f"{db}_r7flow.parquet"), None), ("flow_pre", pre, os.path.join(tmp, f"{db}_pre.parquet"), "n")]
            for part, (sql, path) in extra_statements(tmp, db).items():
                st.append((part, sql, path, EXPECT["cov"] if part == "cov" else EXPECT[f"out_{db}"]))
            for name, (vdb, _) in VARIANTS.items():
                if vdb == db:
                    for part, sql, path in variant_statements(tmp, name):
                        want = {"cohort": None, "route": EXPECT["route"]}.get(part)
                        st.append((f"{name}:{part}", sql, path, want))
            for name, sql, path, want in st:
                try:
                    con.execute(f"COPY ({sql}) TO '{path}' (FORMAT parquet)")
                    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
                    good = want is None or cols == want.split()
                    check(f"{db}: {name}", good, "binds" + ("" if good else f"; got {cols}"))
                except Exception as e:
                    check(f"{db}: {name}", False, f"{type(e).__name__}: {str(e).splitlines()[0][:180]}")
            con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(results)}/{len(results)} passed ({time.time() - t0:.0f} s)")
    sys.exit(0 if all(results) else 1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make-variants", "selftest", "bindcheck"):
        sp.add_parser(name)
    s = sp.add_parser("basecheck")
    s.add_argument("--base", required=True)
    s = [s, sp.add_parser("flow"), sp.add_parser("extra"), sp.add_parser("variant"), sp.add_parser("estimate"),
         sp.add_parser("boot"), sp.add_parser("report")]
    for x in s:
        x.add_argument("--private", required=True)
        x.add_argument("--results", required=True)
    s[1].add_argument("--db", choices=["m", "e"], required=True)
    s[2].add_argument("--db", choices=["m", "e"], required=True)
    s[2].add_argument("--part", choices=["cov", "out"], required=True)
    s[3].add_argument("--name", choices=list(VARIANTS), required=True)
    s[5].add_argument("--key", choices=ALL_KEYS, required=True)
    s[5].add_argument("--start", type=int, required=True)
    s[5].add_argument("--stop", type=int, required=True)
    s[5].add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"make-variants": make_variants, "selftest": selftest, "bindcheck": bindcheck, "basecheck": basecheck, "flow": flow,
     "extra": extra, "variant": variant, "estimate": estimate, "boot": boot, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
