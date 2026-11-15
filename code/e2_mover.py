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
import tarfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc

sys.path.insert(0, asd.TOOLS)
import extract_mover as XM

sys.excepthook = asd._hook
EMR = "EPIC_EMR.tar.gz"
SMALL_RE = r"patient_(information|visit|history|lda|post_op_complications)\.csv$"
AS_RE = ("famotidine|pepcid|pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid|ranitidine|zantac|"
         "cimetidine|nizatidine|rabeprazole|aciphex")
AP_RE = "haloperidol|haldol|olanzapine|zyprexa|quetiapine|seroquel|risperidone|risperdal|ziprasidone|geodon|aripiprazole|abilify"
SED_RE = "propofol|diprivan|midazolam|versed|lorazepam|ativan|dexmedetomidine|precedex"
OPI_RE = ("fentanyl|sufentanil|remifentanil|alfentanil|hydromorphone|dilaudid|morphine|oxycodone|hydrocodone|meperidine|methadone|"
          "tramadol|tapentadol|buprenorphine|codeine")
PRES_RE = "norepinephrine|levophed|epinephrine|phenylephrine|vasopressin|dopamine"
MEDS_RE = "|".join([AS_RE, AP_RE, SED_RE, OPI_RE, PRES_RE])
MEDS_KEEP = ["MRN", "ORDER_CLASS_NM", "DISPLAY_NAME", "MEDICATION_NM", "START_DATE", "RECORD_TYPE", "MAR_ACTION_NM",
             "MED_ACTION_TIME", "ADMIN_SIG", "DOSE_UNIT_NM", "MED_ROUTE_NM"]
LABS_KEEP = ["MRN", "Lab Code", "Lab Name", "Observation Value", "Measurement Units", "Collection Datetime"]
STREAMS = {"meds": (r"patient_medications\.csv$", MEDS_KEEP, ("MEDICATION_NM", "DISPLAY_NAME"), MEDS_RE),
           "labs": (r"patient_labs\.csv$", LABS_KEEP, ("Lab Name",), "creatinine")}
TS = ("coalesce(try_strptime({x}, ['%m/%d/%y %H:%M', '%m/%d/%Y %H:%M', '%m/%d/%y %H:%M:%S', '%m/%d/%Y %H:%M:%S', "
      "'%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %I:%M %p', '%m/%d/%y', '%m/%d/%Y']), "
      "CASE WHEN regexp_matches({x}, '^[0-9]{{4}}-[0-9]{{1,2}}-[0-9]{{1,2}}') THEN TRY_CAST({x} AS TIMESTAMP) END)")


def ts(x):
    return TS.format(x=x)


def paths(args):
    P = os.path.abspath(os.path.expanduser(args.private))
    M = os.path.join(P, "mover")
    for d in ("stage", "parts", "raw"):
        os.makedirs(os.path.join(M, d), exist_ok=True)
    return P, M, os.path.expanduser(args.mover_dir)


def dbcon(M, read_only=False):
    import duckdb
    con = duckdb.connect(os.path.join(M, "mover.duckdb"), read_only=read_only)
    tmp = os.path.join(M, "duck_tmp")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET threads = 2; SET memory_limit = '1500MB'; SET temp_directory = '{tmp}'; SET preserve_insertion_order = false")
    return con


def small(args):
    t0 = time.time()
    P, M, root = paths(args)
    raw = os.path.join(M, "raw")
    shutil.rmtree(raw, ignore_errors=True)
    os.makedirs(raw)
    with tarfile.open(os.path.join(root, EMR), "r|gz") as tf:
        for m in tf:
            if m.isfile() and re.search(SMALL_RE, m.name):
                with tf.extractfile(m) as src, open(os.path.join(raw, os.path.basename(m.name)), "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 << 20)
    con = dbcon(M)
    try:
        for f in sorted(glob.glob(os.path.join(raw, "*.csv"))):
            name = "epic_" + os.path.basename(f)[:-4].replace("patient_", "")
            XM.read_csv_varchar(con, name, f)
            n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            print(f"  {name}: {sc.mask_n(n)} rows (masked)", flush=True)
            os.remove(f)
        con.execute(f"""CREATE OR REPLACE TABLE mv_mrn AS SELECT DISTINCT trim(MRN) AS mrn FROM epic_information
                        WHERE upper(trim(ICU_ADMIN_FLAG)) IN ('YES', 'Y', '1', 'TRUE')""")
        print(f"  patients with an ICU-flagged case: {sc.mask_n(con.execute('SELECT count(*) FROM mv_mrn').fetchone()[0])}", flush=True)
    finally:
        con.close()
    asd.status("small: done", t0)
    asd.log_run(args.results, f"e2_mover small: ok ({time.time() - t0:.0f} s)")


def make_filter(prefix, keep, rcols, regex, mrns, stats):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    mrn_set = pa.array(sorted(mrns), type=pa.string())

    def skip(row):
        stats["bad_rows"] += 1
        return "skip"

    def on_chunk(body, cols, member):
        use = [c for c in keep if c in cols]
        ro = pcsv.ReadOptions(column_names=cols, block_size=8 << 20)
        po = pcsv.ParseOptions(newlines_in_values=False, invalid_row_handler=skip)
        co = pcsv.ConvertOptions(column_types={c: pa.string() for c in cols}, include_columns=use, strings_can_be_null=True)
        tbl = pcsv.read_csv(pa.py_buffer(body), read_options=ro, parse_options=po, convert_options=co)
        stats["rows_in"] += tbl.num_rows
        m = None
        for c in rcols:
            if c in use:
                x = pc.match_substring_regex(tbl[c], regex, ignore_case=True)
                m = x if m is None else pc.or_kleene(m, x)
        mm = pc.is_in(pc.utf8_trim_whitespace(tbl["MRN"]), value_set=mrn_set)
        out = tbl.filter(pc.fill_null(pc.and_kleene(m, mm), False))
        if out.num_rows:
            for c in keep:
                if c not in out.column_names:
                    out = out.append_column(c, pa.nulls(out.num_rows, pa.string()))
            pq.write_table(out.select(keep), f"{prefix}_{stats['parts']:06d}.parquet", compression="zstd")
            stats["parts"] += 1
            stats["rows_kept"] += out.num_rows

    return on_chunk


def stream(args):
    t0 = time.time()
    P, M, root = paths(args)
    member_re, keep, rcols, regex = STREAMS[args.cmd]
    stage, parts = os.path.join(M, "stage"), os.path.join(M, "parts", args.cmd)
    os.makedirs(parts, exist_ok=True)
    if os.path.exists(os.path.join(parts, "_done.txt")):
        print(f"{args.cmd}: done (already)")
        return
    con = dbcon(M, read_only=True)
    mrns = {r[0] for r in con.execute("SELECT mrn FROM mv_mrn WHERE mrn IS NOT NULL").fetchall()}
    con.close()
    st_path = os.path.join(stage, f"{args.cmd}.state.json")
    stats_path = st_path + ".stats"
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {"rows_in": 0, "rows_kept": 0, "bad_rows": 0, "parts": 0}
    s = XM.TarCsvStream(os.path.join(root, EMR), member_re, st_path, os.path.join(stage, EMR + ".idx"))
    st = s.run(make_filter(os.path.join(parts, "part"), keep, rcols, regex, mrns, stats), args.budget, 96 << 20)
    json.dump(stats, open(stats_path, "w"))
    if st["done"]:
        open(os.path.join(parts, "_done.txt"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        print(f"{args.cmd}: done; kept {sc.mask_n(stats['rows_kept'])} of {sc.mask_n(stats['rows_in'])} rows "
              f"(unparsable {sc.mask_n(stats['bad_rows'])}; masked)", flush=True)
        asd.log_run(args.results, f"e2_mover {args.cmd}: done")
    else:
        print(f"{args.cmd}: not done ({st['bytes_done'] / 1e9:.1f} GB read); run again ({time.time() - t0:.0f} s)", flush=True)


def vc(con, sql, top=40):
    rows = con.execute(sql).fetchall()
    return "; ".join(f"{r[0]} [{sc.mask_n(r[1])}]" for r in rows[:top]) + (f"; ... ({len(rows)} values)" if len(rows) > top else "")


def dictionary(args):
    P, M, root = paths(args)
    con = dbcon(M, read_only=True)
    pat = "regexp_replace(regexp_replace({c}, '[0-9]', '9', 'g'), '[A-Za-z]', 'a', 'g')"
    meds = f"read_parquet('{M}/parts/meds/*.parquet')"
    labs = f"read_parquet('{M}/parts/labs/*.parquet')"
    cdt = pat.format(c='"Collection Datetime"')
    q = {
        "info.ICU_ADMIN_FLAG": "SELECT ICU_ADMIN_FLAG, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.DISCH_DISP": "SELECT DISCH_DISP, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.PATIENT_CLASS_NM": "SELECT PATIENT_CLASS_NM, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.PATIENT_CLASS_GROUP": "SELECT PATIENT_CLASS_GROUP, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.PRIMARY_ANES_TYPE_NM": "SELECT PRIMARY_ANES_TYPE_NM, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.ASA_RATING_C/ASA_RATING": "SELECT ASA_RATING_C || ' = ' || coalesce(ASA_RATING, ''), count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.SEX": "SELECT SEX, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.BIRTH_DATE pattern": f"SELECT {pat.format(c='BIRTH_DATE')}, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.HOSP_ADMSN_TIME pattern": f"SELECT {pat.format(c='HOSP_ADMSN_TIME')}, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.OUT_OR_DTTM pattern": f"SELECT {pat.format(c='OUT_OR_DTTM')}, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info.SURGERY_DATE pattern": f"SELECT {pat.format(c='SURGERY_DATE')}, count(*) n FROM epic_information GROUP BY 1 ORDER BY 2 DESC",
        "info cases per MRN (ICU-flagged)": "SELECT k, count(*) FROM (SELECT trim(MRN), count(*) k FROM epic_information WHERE ICU_ADMIN_FLAG = 'Yes' GROUP BY 1) GROUP BY 1 ORDER BY 1",
        "visit.diagnosis_code pattern": f"SELECT {pat.format(c='diagnosis_code')}, count(*) n FROM epic_visit GROUP BY 1 ORDER BY 2 DESC",
        "history.diagnosis_code pattern": f"SELECT {pat.format(c='diagnosis_code')}, count(*) n FROM epic_history GROUP BY 1 ORDER BY 2 DESC",
        "lda.Line_Group_Name": "SELECT Line_Group_Name, count(*) n FROM epic_lda GROUP BY 1 ORDER BY 2 DESC",
        "lda.flo_meas_name (airway)": "SELECT flo_meas_name, count(*) n FROM epic_lda WHERE regexp_matches(lower(coalesce(flo_meas_name,'') || ' ' || coalesce(Line_Group_Name,'') || ' ' || coalesce(description,'')), 'airway|ett|endotrach|trach') GROUP BY 1 ORDER BY 2 DESC",
        "lda.description (airway) pattern": f"SELECT {pat.format(c='description')}, count(*) n FROM epic_lda WHERE regexp_matches(lower(coalesce(Line_Group_Name,'') || ' ' || coalesce(description,'')), 'airway|ett|endotrach') GROUP BY 1 ORDER BY 2 DESC",
        "lda.placement_instant pattern": f"SELECT {pat.format(c='placement_instant')}, count(*) n FROM epic_lda GROUP BY 1 ORDER BY 2 DESC",
        "postop.Element_Name": "SELECT Element_Name, count(*) n FROM epic_post_op_complications GROUP BY 1 ORDER BY 2 DESC",
        "postop.CONTEXT_NAME": "SELECT CONTEXT_NAME, count(*) n FROM epic_post_op_complications GROUP BY 1 ORDER BY 2 DESC",
        "meds.RECORD_TYPE": f"SELECT RECORD_TYPE, count(*) n FROM {meds} GROUP BY 1 ORDER BY 2 DESC",
        "meds.MAR_ACTION_NM": f"SELECT MAR_ACTION_NM, count(*) n FROM {meds} GROUP BY 1 ORDER BY 2 DESC",
        "meds.ORDER_CLASS_NM": f"SELECT ORDER_CLASS_NM, count(*) n FROM {meds} GROUP BY 1 ORDER BY 2 DESC",
        "meds.MED_ACTION_TIME pattern": f"SELECT {pat.format(c='MED_ACTION_TIME')}, count(*) n FROM {meds} GROUP BY 1 ORDER BY 2 DESC",
        "meds.MED_ROUTE_NM (acid suppressants)": f"SELECT MED_ROUTE_NM, count(*) n FROM {meds} WHERE regexp_matches(lower(MEDICATION_NM), '{AS_RE}') GROUP BY 1 ORDER BY 2 DESC",
        "meds.MEDICATION_NM (acid suppressants)": f"SELECT MEDICATION_NM, count(*) n FROM {meds} WHERE regexp_matches(lower(MEDICATION_NM), '{AS_RE}') GROUP BY 1 ORDER BY 2 DESC",
        "meds.MEDICATION_NM (antipsychotics)": f"SELECT MEDICATION_NM, count(*) n FROM {meds} WHERE regexp_matches(lower(MEDICATION_NM), '{AP_RE}') GROUP BY 1 ORDER BY 2 DESC",
        "meds.MED_ROUTE_NM (antipsychotics)": f"SELECT MED_ROUTE_NM, count(*) n FROM {meds} WHERE regexp_matches(lower(MEDICATION_NM), '{AP_RE}') GROUP BY 1 ORDER BY 2 DESC",
        "meds.MEDICATION_NM (pressors)": f"SELECT MEDICATION_NM, count(*) n FROM {meds} WHERE regexp_matches(lower(MEDICATION_NM), '{PRES_RE}') GROUP BY 1 ORDER BY 2 DESC",
        "meds.DISPLAY_NAME (famotidine)": f"SELECT DISPLAY_NAME, count(*) n FROM {meds} WHERE regexp_matches(lower(MEDICATION_NM), 'famotidine|pepcid') GROUP BY 1 ORDER BY 2 DESC",
        "meds.DOSE_UNIT_NM (acid suppressants)": f"SELECT DOSE_UNIT_NM, count(*) n FROM {meds} WHERE regexp_matches(lower(MEDICATION_NM), '{AS_RE}') GROUP BY 1 ORDER BY 2 DESC",
        "meds.MEDICATION_NM not matching any class": f"SELECT MEDICATION_NM, count(*) n FROM {meds} WHERE NOT regexp_matches(lower(coalesce(MEDICATION_NM, '')), '{MEDS_RE}') GROUP BY 1 ORDER BY 2 DESC",
        "labs.Lab Name": f"SELECT \"Lab Name\" || ' | ' || coalesce(\"Measurement Units\", ''), count(*) n FROM {labs} GROUP BY 1 ORDER BY 2 DESC",
        "labs.Collection Datetime pattern": f"SELECT {cdt}, count(*) n FROM {labs} GROUP BY 1 ORDER BY 2 DESC",
    }
    for k, s in q.items():
        try:
            print(f"{k}: {vc(con, s)}", flush=True)
        except Exception as e:
            print(f"{k}: not available ({str(e).splitlines()[0][:120]})", flush=True)
    con.close()


ADMIN = ("Given", "Given by Other", "Given in incremental doses", "Bolus", "Bolus From Bag", "New Bag", "Investigational New Bag",
         "New Syringe/Cartridge", "Restarted", "Restarted Dual Sign", "IV Resume", "Rate Verify", "Rate Verify Dual Sign",
         "Rate Change", "Rate Change Dual Sign", "IV Rate Change", "Controlled Patch Applied", "ED Continuing Infusion")
PPI_RE = "pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid"
LOCAL_RE = "lidocaine|bupivacaine|ropivacaine|mepivacaine|articaine|prilocaine|chloroprocaine|tetracaine|tumescent"
DEATH = "('Expired', 'Coroner')"
GI = ("(regexp_matches(code, '^K2[5-8]') OR code IN ('K920', 'K921', 'K922', 'I8501', 'I8511', 'K2211', 'K2271') "
      "OR regexp_matches(code, '^K29[0-9]1') OR regexp_matches(code, '^K31811') OR regexp_matches(code, '^53[1-4]') "
      "OR regexp_matches(code, '^578') OR code IN ('4560', '45620', '53082') OR regexp_matches(code, '^535[0-6]1'))")
PSY = "regexp_matches(code, '^(F2[0-9]|F31|295|297|298|2960|2961|2964|2965|2966|2967|2968)')"
DEM = "regexp_matches(code, '^(F01|F02|F03|G30|G310|G311|G3183|290|2941|2942|3310|3311|3312|33182)')"
ALC = "regexp_matches(code, '^(F10|291|303|3050)')"
FLOW = [("Surgical cases flagged ICU", "TRUE"), ("Age ≥18 years", "f_adult"), ("Admission, discharge and operating-room exit times recorded", "f_times"),
        ("Acid suppressant given within 72 h after leaving the operating room", "f_as72"), ("No acid suppressant in the previous 24 h", "f_new"), ("No antipsychotic from admission to time zero", "f_noap"),
        ("No upper GI bleeding or peptic ulcer diagnosis in the admission", "f_nogib"), ("No schizophrenia-spectrum or bipolar disorder in the history", "f_nopsy"),
        ("No continuous PPI infusion from 24 h before to 1 h after time zero", "f_noinf"), ("First acid suppressant a PPI or famotidine", "f_arm")]
FAMDOSE_NOTE = "Not done: patient_medications has no order frequency field (single doses only), so the planned daily dose of the first famotidine order cannot be derived (dose comparison rule of SAP E2 section 2)"


def design_sql(M):
    acts = ", ".join("'" + a + "'" for a in ADMIN)
    return [
        f"""CREATE OR REPLACE TABLE mv_case AS
SELECT * FROM (SELECT DISTINCT trim(LOG_ID) AS log_id, trim(MRN) AS mrn, {ts('HOSP_ADMSN_TIME')} AS adm, {ts('HOSP_DISCH_TIME')} AS dis,
  DISCH_DISP AS disp, upper(trim(ICU_ADMIN_FLAG)) = 'YES' AS icu, TRY_CAST(BIRTH_DATE AS INTEGER) AS age, SEX AS sex,
  trim(PRIMARY_ANES_TYPE_NM) AS anes, CAST(TRY_CAST(ASA_RATING_C AS DOUBLE) AS INTEGER) AS asa, PATIENT_CLASS_NM AS pclass,
  {ts('IN_OR_DTTM')} AS t_in, {ts('OUT_OR_DTTM')} AS t_out,
  least({ts('HOSP_ADMSN_TIME')}, coalesce({ts('IN_OR_DTTM')}, {ts('OUT_OR_DTTM')})) AS h0
FROM epic_information) QUALIFY count(*) OVER (PARTITION BY log_id) = 1""",
        f"""CREATE OR REPLACE TABLE mv_med AS
WITH r AS (
  SELECT DISTINCT trim(MRN) AS mrn, CAST(MED_ACTION_TIME AS TIMESTAMP) AS t,
    lower(CASE WHEN regexp_matches(lower(coalesce(MEDICATION_NM, '')), '{MEDS_RE}') THEN MEDICATION_NM ELSE DISPLAY_NAME END) AS nm,
    MAR_ACTION_NM AS action, TRY_CAST(ADMIN_SIG AS DOUBLE) AS dose, DOSE_UNIT_NM AS unit, MED_ROUTE_NM AS route
  FROM read_parquet('{M}/parts/meds/*.parquet')
  WHERE MAR_ACTION_NM IN ({acts}) AND MED_ACTION_TIME IS NOT NULL)
SELECT mrn, t, nm, action, dose, unit, route,
  CASE WHEN regexp_matches(nm, '{PPI_RE}') THEN 'PPI'
       WHEN regexp_matches(nm, 'famotidine|pepcid') THEN 'FAM'
       WHEN regexp_matches(nm, '{AS_RE}') THEN 'OTH'
       WHEN regexp_matches(nm, 'haloperidol|haldol') THEN 'HAL'
       WHEN regexp_matches(nm, 'olanzapine|zyprexa|quetiapine|seroquel') THEN 'OQ'
       WHEN regexp_matches(nm, '{AP_RE}') THEN 'APO'
       WHEN regexp_matches(nm, '{SED_RE}') THEN 'SED'
       WHEN regexp_matches(nm, '{OPI_RE}') THEN 'OPI'
       WHEN regexp_matches(nm, '{PRES_RE}') AND regexp_matches(lower(coalesce(route, '')), 'intravenous')
            AND NOT regexp_matches(nm, '{LOCAL_RE}') THEN 'PRES' END AS cls,
  (regexp_matches(nm, 'infusion') OR coalesce(unit, '') = 'mg/hr') AS infusion
FROM r WHERE nm IS NOT NULL""",
        f"""CREATE OR REPLACE TABLE mv_cr AS
SELECT DISTINCT trim(MRN) AS mrn, CAST("Collection Datetime" AS TIMESTAMP) AS t, TRY_CAST("Observation Value" AS DOUBLE) AS v
FROM read_parquet('{M}/parts/labs/*.parquet')
WHERE "Lab Name" = 'Creatinine' AND upper(coalesce("Measurement Units", '')) = 'MG/DL'""",
        """CREATE OR REPLACE TABLE mv_hist AS
SELECT DISTINCT trim(mrn) AS mrn, upper(replace(trim(diagnosis_code), '.', '')) AS code FROM epic_history WHERE diagnosis_code IS NOT NULL""",
        """CREATE OR REPLACE TABLE mv_visit AS
SELECT DISTINCT trim(LOG_ID) AS log_id, upper(replace(trim(diagnosis_code), '.', '')) AS code FROM epic_visit WHERE diagnosis_code IS NOT NULL""",
        f"""CREATE OR REPLACE TABLE mv_ett AS
SELECT DISTINCT trim(MRN) AS mrn, {ts('placement_instant')} AS t_on, {ts('removal_instant')} AS t_off
FROM epic_lda WHERE flo_meas_name = 'NON-SURGICAL AIRWAY'""",
        "CREATE OR REPLACE TABLE mv_or AS SELECT DISTINCT mrn, coalesce(t_in, t_out) AS t_in, t_out FROM mv_case WHERE t_out IS NOT NULL",
        "CREATE OR REPLACE TABLE mv_base AS SELECT * FROM mv_case WHERE icu",
        """CREATE OR REPLACE TABLE mv_t0 AS
WITH a AS (
  SELECT b.log_id, m.t, m.cls, m.route
  FROM mv_base b JOIN mv_med m ON m.mrn = b.mrn AND m.cls IN ('PPI', 'FAM', 'OTH')
   AND m.t > b.t_out AND m.t <= b.t_out + INTERVAL 72 HOUR AND m.t < b.dis),
f AS (SELECT log_id, min(t) AS t0 FROM a GROUP BY 1)
SELECT f.log_id, f.t0, CASE WHEN count(DISTINCT a.cls) > 1 THEN 'MIX' ELSE min(a.cls) END AS first_cls,
  CASE WHEN bool_or(regexp_matches(lower(coalesce(a.route, '')), 'intravenous')) THEN 'IV'
       WHEN bool_or(a.route IS NOT NULL) THEN 'EN' ELSE 'UNK' END AS route0
FROM f JOIN a ON a.log_id = f.log_id AND a.t = f.t0 GROUP BY f.log_id, f.t0""",
        f"""CREATE OR REPLACE TABLE mv_elig AS
SELECT b.*, t.t0, t.first_cls, t.route0,
  coalesce(b.age >= 18, false) AS f_adult,
  (b.t_out IS NOT NULL AND b.adm IS NOT NULL AND b.dis IS NOT NULL) AS f_times,
  t.t0 IS NOT NULL AS f_as72,
  t.t0 IS NOT NULL AND NOT EXISTS (SELECT 1 FROM mv_med m WHERE m.mrn = b.mrn AND m.cls IN ('PPI', 'FAM', 'OTH')
        AND m.t >= t.t0 - INTERVAL 24 HOUR AND m.t < t.t0) AS f_new,
  t.t0 IS NOT NULL AND NOT EXISTS (SELECT 1 FROM mv_med m WHERE m.mrn = b.mrn AND m.cls IN ('HAL', 'OQ', 'APO')
        AND m.t >= b.h0 AND m.t <= t.t0) AS f_noap,
  NOT EXISTS (SELECT 1 FROM mv_visit v WHERE v.log_id = b.log_id AND {GI}) AS f_nogib,
  NOT EXISTS (SELECT 1 FROM mv_hist h WHERE h.mrn = b.mrn AND {PSY}) AS f_nopsy,
  t.t0 IS NOT NULL AND NOT EXISTS (SELECT 1 FROM mv_med m WHERE m.mrn = b.mrn AND m.cls = 'PPI' AND m.infusion
        AND m.t >= t.t0 - INTERVAL 24 HOUR AND m.t <= t.t0 + INTERVAL 1 HOUR) AS f_noinf,
  coalesce(t.first_cls IN ('PPI', 'FAM'), false) AS f_arm
FROM mv_base b LEFT JOIN mv_t0 t USING (log_id)""",
        """CREATE OR REPLACE TABLE mv_cohort AS
SELECT *, CASE WHEN first_cls = 'PPI' THEN 'A' ELSE 'B' END AS arm FROM mv_elig
WHERE f_adult AND f_times AND f_as72 AND f_new AND f_noap AND f_nogib AND f_nopsy AND f_noinf AND f_arm
QUALIFY row_number() OVER (PARTITION BY mrn ORDER BY t0, t_out DESC, log_id) = 1""",
        f"""CREATE OR REPLACE TABLE mv_cov AS
SELECT c.log_id,
  (SELECT r.v FROM mv_cr r WHERE r.mrn = c.mrn AND r.t >= c.t0 - INTERVAL 24 HOUR AND r.t <= c.t0
     AND r.v BETWEEN 0.1 AND 25 ORDER BY r.t DESC, r.v DESC LIMIT 1) AS scr,
  EXISTS (SELECT 1 FROM mv_med m WHERE m.mrn = c.mrn AND m.cls = 'SED' AND m.t >= c.t0 - INTERVAL 6 HOUR AND m.t <= c.t0) AS sed6,
  EXISTS (SELECT 1 FROM mv_med m WHERE m.mrn = c.mrn AND m.cls = 'OPI' AND m.t >= c.t0 - INTERVAL 6 HOUR AND m.t <= c.t0) AS opi6,
  EXISTS (SELECT 1 FROM mv_med m WHERE m.mrn = c.mrn AND m.cls = 'PRES' AND m.t >= c.t0 - INTERVAL 6 HOUR AND m.t <= c.t0) AS pres6,
  EXISTS (SELECT 1 FROM mv_ett e WHERE e.mrn = c.mrn AND e.t_on <= c.t0 AND (e.t_off IS NULL OR e.t_off > c.t0)) AS ett,
  EXISTS (SELECT 1 FROM mv_hist h WHERE h.mrn = c.mrn AND {DEM}) AS dementia,
  EXISTS (SELECT 1 FROM mv_hist h WHERE h.mrn = c.mrn AND {ALC}) AS alcohol
FROM mv_cohort c""",
    ]


OUTCOME_SQL = """CREATE OR REPLACE TABLE mv_outcome AS
WITH ap AS (
  SELECT c.log_id, m.t, m.cls FROM mv_cohort c JOIN mv_med m ON m.mrn = c.mrn AND m.cls IN ('HAL', 'OQ', 'APO')
    AND m.t > c.t0 AND m.t <= c.t0 + INTERVAL 168 HOUR AND m.t <= c.dis
  WHERE NOT (m.cls = 'HAL' AND EXISTS (SELECT 1 FROM mv_or o WHERE o.mrn = c.mrn AND m.t >= o.t_in AND m.t <= o.t_out + INTERVAL 6 HOUR)))
SELECT c.log_id, min(date_diff('minute', c.t0, a.t)) / 60.0 AS t_ap,
  min(CASE WHEN a.cls = 'OQ' THEN date_diff('minute', c.t0, a.t) END) / 60.0 AS t_oq
FROM mv_cohort c LEFT JOIN ap a USING (log_id) GROUP BY c.log_id"""


def has_table(con, name):
    return con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0] > 0


def load(M, outcome=False):
    con = dbcon(M, read_only=True)
    try:
        sql = f"""SELECT c.log_id, c.mrn, c.arm, c.age, c.sex, c.asa, c.pclass, c.anes, c.route0, year(c.t0) AS yr,
  date_diff('minute', c.t_out, c.t0) / 60.0 AS h_or, date_diff('minute', c.t0, c.dis) / 60.0 AS t_end,
  c.disp IN {DEATH} AS end_death, v.scr, v.sed6, v.opi6, v.pres6, v.ett, v.dementia, v.alcohol
  {', o.t_ap, o.t_oq' if outcome else ''}
FROM mv_cohort c JOIN mv_cov v USING (log_id) {'JOIN mv_outcome o USING (log_id)' if outcome else ''} ORDER BY c.mrn"""
        return con.execute(sql).df()
    finally:
        con.close()


def ckd_epi(scr, age, female):
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    with np.errstate(invalid="ignore"):
        r = scr / k
        e = 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.2 * 0.9938 ** age * np.where(female, 1.012, 1.0)
    return np.where(np.isfinite(scr), e, np.nan)


def anes_group(s):
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
    if s == "General":
        return "GENERAL"
    if s.startswith("Monitored Anesthesia Care") or s.startswith("Moderate Sedation"):
        return "MAC"
    if s in ("Regional", "Epidural", "Spinal"):
        return "REGIONAL"
    return "OTHER"


def design_matrix(df):
    cols, names = [], []

    def add(name, v):
        cols.append(np.asarray(v, float))
        names.append(name)

    age = df["age"].astype(float).values
    add("age_lt50", age < 50)
    add("age_ge65", age >= 65)
    add("age_ge80", age >= 80)
    female = df["sex"].astype(str).values == "Female"
    add("female", female)
    egfr = ckd_epi(df["scr"].astype(float).values, age, female)
    miss = ~np.isfinite(egfr)
    e = np.where(miss, 90.0, egfr)
    add("egfr_missing", miss)
    add("egfr_per10", e / 10)
    for k in (30, 60, 90):
        add(f"egfr_over{k}_per10", np.maximum(e - k, 0) / 10)
    asa = df["asa"].astype(float).values
    add("asa_ge3", asa >= 3)
    add("asa_ge4", asa >= 4)
    add("asa_ge5", asa >= 5)
    add("asa_missing", ~np.isfinite(asa))
    h = df["h_or"].astype(float).values
    add("h_or_ge6", h >= 6)
    add("h_or_ge24", h >= 24)
    add("h_or_ge48", h >= 48)
    add("h_or_per6", h / 6)
    for k in (6, 24):
        add(f"h_or_over{k}_per6", np.maximum(h - k, 0) / 6)
    pcl = df["pclass"].astype(str).values
    add("class_outpatient_surgery", pcl == "Hospital Outpatient Surgery")
    add("class_inpatient_surgery", pcl == "Hospital Inpatient Surgery")
    an = np.array([anes_group(s) for s in df["anes"].values])
    for g in ("MAC", "REGIONAL", "OTHER"):
        add(f"anes_{g.lower()}", an == g)
    for c in ("sed6", "opi6", "pres6", "ett", "dementia", "alcohol"):
        add(c, df[c].fillna(False).astype(bool).values)
    r = df["route0"].astype(str).values
    add("route_enteral", r == "EN")
    add("route_unknown", r == "UNK")
    yr = df["yr"].astype(float).values
    for y in sorted(set(int(v) for v in yr[np.isfinite(yr)]))[1:]:
        add(f"year_{y}", yr == y)
    return np.column_stack(cols), names, {"age_years": age, "egfr": egfr, "h_or": h}


def build(df, outcome=False):
    X, names, extras = design_matrix(df)
    X, names, dropped = asd.prevalence_filter(X, names)
    n = len(df)
    d = {"X": X, "names": names, "dropped": dropped, "extras": extras, "A": (df["arm"].values == "A").astype(int), "site": None,
         "t_end": df["t_end"].astype(float).values, "end_death": df["end_death"].fillna(False).astype(bool).values,
         "t_dev": np.full(n, np.nan), "sub": {}}
    if outcome:
        d["t_ap"] = df["t_ap"].astype(float).values
        d["t_oq"] = df["t_oq"].astype(float).values
    return d


def flow_counts(M):
    con = dbcon(M, read_only=True)
    try:
        out, conds = {}, []
        for label, f in FLOW:
            conds.append(f)
            out[label] = int(con.execute(f"SELECT count(*) FROM mv_elig WHERE {' AND '.join(conds)}").fetchone()[0])
        out["Patients (first eligible case of each)"] = int(con.execute("SELECT count(*) FROM mv_cohort").fetchone()[0])
        a, b = con.execute("SELECT count(*) FILTER (WHERE arm = 'A'), count(*) FILTER (WHERE arm = 'B') FROM mv_cohort").fetchone()
        out["A (PPI)"], out["B (famotidine)"] = int(a), int(b)
        return out
    finally:
        con.close()


def design(args):
    t0 = time.time()
    P, M, root = paths(args)
    con = dbcon(M)
    try:
        if has_table(con, "mv_outcome"):
            raise SystemExit("outcome table exists: design is outcome-free and runs before it")
        for s in design_sql(M):
            con.execute(s)
    finally:
        con.close()
    asd.status("design tables built", t0)
    flow = flow_counts(M)
    df = load(M)
    d = build(df)
    X, names, A = d["X"], d["names"], d["A"]
    ps, _ = sc.ps_fit(X, A)
    ow = sc.overlap_w(ps, A)
    out = {"flow": {k: sc.mask_n(v) for k, v in flow.items()}, "famdose": FAMDOSE_NOTE, "columns": names,
           "columns_dropped_low_prevalence": d["dropped"], "ps_quantiles": {"A": asd.q(ps[A == 1]), "B": asd.q(ps[A == 0])},
           "ess_overlap": {"A": round(sc.ess(ow[A == 1]), 1), "B": round(sc.ess(ow[A == 0]), 1)}}
    Xe = np.column_stack([X] + [np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 0) for v in d["extras"].values()])
    ne = names + [f"{k} (continuous)" for k in d["extras"]]
    un, so = sc.smd_table(Xe, ne, A), sc.smd_table(Xe, ne, A, ow)
    out["smd"] = {nm: [round(un[nm], 3), round(so[nm], 3)] for nm in ne}
    out["max_abs_smd"] = {"unweighted": round(max(abs(v) for v in un.values()), 3), "overlap": round(max(abs(v) for v in so.values()), 3)}
    prev = {}
    for j, nm in enumerate(names):
        v = X[:, j]
        if set(np.unique(v).tolist()) <= {0.0, 1.0}:
            prev[nm] = {arm: f"{sc.mask_n(int(v[A == a].sum()))} ({sc.pct_text(int(v[A == a].sum()), int((A == a).sum()))})"
                        for arm, a in (("A", 1), ("B", 0))}
    out["prevalence"] = prev
    te, dth = d["t_end"], d["end_death"]
    fu = {f"in_hospital_alive_at_{h}h": sc.mask_n(int((te > h).sum())) for h in (24, 72, 168)}
    fu["died_before_168h"] = sc.mask_n(int(((te <= 168) & dth).sum()))
    fu["discharged_alive_before_168h"] = sc.mask_n(int(((te <= 168) & ~dth).sum()))
    out["followup_both_arms"] = fu
    json.dump(out, open(os.path.join(args.results, "e2_mover_design.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(args.results, "e2_mover_design.md"), "w", encoding="utf-8").write(design_md(out))
    asd.log_run(args.results, f"e2_mover design: ok ({time.time() - t0:.0f} s)")
    asd.status("design: done", t0)


def design_md(o):
    L = ["# E2 MOVER stage 1 (no outcomes)", "", "Plan: SAP_addendum_E2.md section 2; revision R5. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", "",
         "## Flow", "", "| Step | Number (cases) |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in o["flow"].items()]
    L += ["", f"Dose comparison: {o['famdose']}", "", "## Propensity score and balance", "",
          f"- Model columns: {', '.join(o['columns'])}", f"- Columns dropped for low prevalence: {', '.join(o['columns_dropped_low_prevalence']) or 'no'}",
          f"- Propensity score quantiles (0, 1, 5, 25, 50, 75, 95, 99, 100) A: {o['ps_quantiles']['A']}; B: {o['ps_quantiles']['B']}",
          f"- Effective sample size with overlap weights A {o['ess_overlap']['A']}, B {o['ess_overlap']['B']}",
          f"- Largest |SMD|: unweighted {o['max_abs_smd']['unweighted']}, overlap weighted {o['max_abs_smd']['overlap']}", "",
          "| Variable | A number (%) | B number (%) | SMD unweighted | SMD overlap weighted |", "|---|---|---|---|---|"]
    for nm, (u, w) in o["smd"].items():
        p = o["prevalence"].get(nm, {"A": "", "B": ""})
        L.append(f"| {nm} | {p['A']} | {p['B']} | {u:+.3f} | {w:+.3f} |")
    L += ["", "## Follow-up (both arms)", ""] + [f"- {k}: {v}" for k, v in o["followup_both_arms"].items()] + [""]
    return "\n".join(L)


def outcome(args):
    t0 = time.time()
    P, M, root = paths(args)
    if not os.path.exists(os.path.join(args.results, "e2_mover_design.json")):
        raise SystemExit("run design first")
    con = dbcon(M)
    try:
        con.execute(OUTCOME_SQL)
    finally:
        con.close()
    asd.log_run(args.results, f"e2_mover outcome: ok ({time.time() - t0:.0f} s)")
    asd.status("outcome: done", t0)


def m_estimates(d, cache):
    A = d["A"]
    ow = sc.overlap_w(asd.ow_fit(d, cache, "full"), A)
    return {"primary": asd.cif_contrast(d, ow, d["t_ap"])[168], "oq": asd.cif_contrast(d, ow, d["t_oq"])[168],
            "crude": asd.cif_contrast(d, np.ones(len(A)), d["t_ap"])[168]}


def estimate(args):
    t0 = time.time()
    P, M, root = paths(args)
    d = build(load(M, outcome=True), outcome=True)
    cache = {}
    est = m_estimates(d, cache)
    json.dump(est, open(os.path.join(M, "est.json"), "w"))
    np.savez(os.path.join(M, "cache.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
    asd.log_run(args.results, f"e2_mover estimate: ok ({time.time() - t0:.0f} s)")
    asd.status("estimate: done", t0)


def boot(args):
    t0 = time.time()
    P, M, root = paths(args)
    d = build(load(M, outcome=True), outcome=True)
    z = np.load(os.path.join(M, "cache.npz"))
    cache0 = {k.replace("__", ":"): z[k] for k in z.files}
    rows = []
    for rep in range(args.start, args.stop):
        b = asd.resample(d, "m", np.random.default_rng(asd.SEED + rep))
        try:
            e = m_estimates(b, dict(cache0))
            rows.append([rep] + [math.log(e[o]["OR"]) if e[o]["OR"] > 0 else np.nan for o in ("primary", "oq")]
                        + [e[o]["RD"] for o in ("primary", "oq")])
        except Exception:
            rows.append([rep] + [np.nan] * 4)
    np.save(os.path.join(M, f"boot_{args.start:04d}_{args.stop:04d}.npy"), np.array(rows, dtype=float))
    asd.log_run(args.results, f"e2_mover boot --start {args.start} --stop {args.stop}: ok ({time.time() - t0:.0f} s)")
    asd.status(f"boot {args.start}-{args.stop} done", t0)


def verdict_or(lo, hi):
    if hi < 1:
        return "proxy outcome supports fewer events in arm A"
    if lo > 1:
        return "proxy outcome in the opposite direction"
    return "proxy outcome inconclusive"


def report(args):
    P, M, root = paths(args)
    est = json.load(open(os.path.join(M, "est.json")))
    parts = sorted(glob.glob(os.path.join(M, "boot_*.npy")))
    B = np.concatenate([np.load(p) for p in parts]) if parts else np.zeros((0, 5))
    if B.size:
        _, idx = np.unique(B[:, 0], return_index=True)
        B = B[idx]
    res = {"famdose": FAMDOSE_NOTE, "boot_reps": int(B.shape[0])}
    for i, o in enumerate(("primary", "oq")):
        e = est[o]
        r = {"riskA": e["riskA"], "riskB": e["riskB"], "OR": e["OR"], "RD": e["RD"], "RR": e["RR"],
             "eventsA": sc.mask_n(e["eventsA"]), "eventsB": sc.mask_n(e["eventsB"]), "nA": sc.mask_n(e["nA"]), "nB": sc.mask_n(e["nB"])}
        lor, rd = B[:, 1 + i], B[:, 3 + i]
        lor, rd = lor[np.isfinite(lor)], rd[np.isfinite(rd)]
        r["boot_ok"] = int(lor.size)
        if lor.size >= asd.MIN_BOOT:
            r["OR_lo"], r["OR_hi"] = float(math.exp(np.percentile(lor, 2.5))), float(math.exp(np.percentile(lor, 97.5)))
            r["RD_lo"], r["RD_hi"] = float(np.percentile(rd, 2.5)), float(np.percentile(rd, 97.5))
            r["verdict"] = verdict_or(r["OR_lo"], r["OR_hi"])
        res[o] = r
    res["crude_OR"] = round(est["crude"]["OR"], 2)
    json.dump(res, open(os.path.join(args.results, "e2_mover.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lab = {"primary": "First antipsychotic (6 drugs; haloperidol within 6 h after leaving the operating room not counted)", "oq": "Sensitivity: olanzapine or quetiapine only"}
    L = ["# E2 MOVER proxy outcome: PPI (A) vs famotidine (B)", "",
         "Plan: SAP_addendum_E2.md section 2; revision R5. Overlap weighting; death and discharge alive as competing events, cumulative incidence read at 168 hours; "
         f"intervals: percentile intervals of {res['boot_reps']} bootstrap replicates by patient. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5. Not pooled with MIMIC-IV or eICU.", "",
         "| Outcome | Events A / B | Patients A / B | Weighted 7-day cumulative incidence A / B | OR (95% CI) | Risk difference (95% CI) | Reading |", "|---|---|---|---|---|---|---|"]
    for o in ("primary", "oq"):
        r = res[o]
        ci = f"{r['OR']:.2f} ({r['OR_lo']:.2f}–{r['OR_hi']:.2f})" if "OR_lo" in r else f"{r['OR']:.2f}"
        rdc = f"{100 * r['RD']:+.1f} percentage points ({100 * r['RD_lo']:+.1f} to {100 * r['RD_hi']:+.1f})" if "RD_lo" in r else f"{100 * r['RD']:+.1f}"
        L.append(f"| {lab[o]} | {r['eventsA']} / {r['eventsB']} | {r['nA']} / {r['nB']} | {100 * r['riskA']:.1f}% / {100 * r['riskB']:.1f}% "
                 f"| {ci} | {rdc} | {r.get('verdict', '')} |")
    L += ["", f"- Unweighted (descriptive) OR {res['crude_OR']:.2f}", f"- Dose comparison: {FAMDOSE_NOTE}", ""]
    open(os.path.join(args.results, "e2_mover.md"), "w", encoding="utf-8").write("\n".join(L))
    asd.log_run(args.results, "e2_mover report: ok")
    print("report written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["small", "meds", "labs", "dict", "design", "outcome", "estimate", "boot", "report"])
    ap.add_argument("--private", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--mover-dir", default=os.path.join(os.environ.get("DATA_DIR", "data"), "MOVER"))
    ap.add_argument("--budget", type=float, default=120)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=asd.NBOOT)
    a = ap.parse_args()
    {"small": small, "meds": stream, "labs": stream, "dict": dictionary, "design": design, "outcome": outcome,
     "estimate": estimate, "boot": boot, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
