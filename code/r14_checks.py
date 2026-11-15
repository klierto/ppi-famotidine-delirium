#!/usr/bin/env python3
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc

sys.path.insert(0, asd.TOOLS)
from mimic_local import connect

E_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "eicu-crd-2.0-parquet")
TMP = os.path.join(os.environ["WORK_DIR"], "r14_tmp")
VENT_LABELS = ("'Vent Rate', 'Tidal Volume (set)', 'TV/kg IBW', 'Plateau Pressure', 'Peak Insp. Pressure', 'Mean Airway Pressure', "
               "'Exhaled MV', 'Exhaled TV (machine)', 'Pressure Control', 'RT Vent On/Off', 'Adult Con Setting Set RR', "
               "'Adult Con Setting Set Vt', 'Set Vt (Servo,LTV)'")
RASS_CAT = ("CASE WHEN {x} IS NULL THEN '5 not recorded' WHEN {x} <= -4 THEN '1 <= -4' WHEN {x} <= -2 THEN '2 -3 to -2' "
            "WHEN {x} <= 0 THEN '3 -1 to 0' ELSE '4 >= +1' END")


def m(n):
    return sc.mask_n(n)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b > 0 and a >= 10 else "-"


def table(con, title, sql, header, note=None):
    rows = con.execute(sql).fetchall()
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        *keys, n = r
        print("| " + " | ".join("" if k is None else str(k) for k in keys) + f" | {m(n)} |")
    if note:
        print(f"\n{note}")
    return rows


def main():
    con = connect(E_ROOT, temp_dir=TMP)
    print("# R14 verification checks: eICU-CRD delirium and sedation scales; timing of covariates (PPI comparison)\n")
    print("Plan: revision R14. Masked counts: 1–9 as <10, otherwise the nearest multiple of 5. No outcome by arm.")

    con.execute(f"CREATE TEMP TABLE coh AS {asd.cohort_sql(2, 'e')}")
    n, na, nb = con.execute("SELECT count(*), count(*) FILTER (WHERE arm = 'A'), count(*) FILTER (WHERE arm = 'B') FROM coh").fetchone()
    g = asd.GATE["c2e"]
    print(f"\n## 0. Cohort\n\nRebuilt cohort: {m(n)} (PPI {m(na)}, famotidine {m(nb)}); gate and main analysis: {g['eligible']} "
          f"({g['A']}, {g['B']}); {'identical' if (m(n), m(na), m(nb)) == (g['eligible'], g['A'], g['B']) else 'DIFFERENT'}.")
    con.execute("""CREATE TEMP TABLE fu AS SELECT c.stay_id AS s, c.t0, least(p.unitdischargeoffset, c.t0 + 10080) AS t_end,
                   p.hospitalid AS hosp FROM coh c JOIN eicu.patient p ON p.patientunitstayid = c.stay_id""")
    con.execute("""CREATE TEMP TABLE nc AS SELECT patientunitstayid AS s, nursingchartoffset AS t, nursingchartcelltypevalname AS nm,
                   nursingchartvalue AS v FROM eicu.nursecharting
                   WHERE nursingchartcelltypevallabel IN ('Delirium Scale/Score', 'Sedation Scale/Score/Goal')
                     AND patientunitstayid IN (SELECT s FROM fu)""")

    con.execute("""CREATE TEMP TABLE dl AS
SELECT a.s, a.t, a.v, coalesce(b.v, '(no scale row)') AS scale,
  CASE WHEN upper(a.v) = 'YES' THEN 'Yes' WHEN upper(a.v) = 'NO' THEN 'No' WHEN a.v = 'N/A' THEN 'N/A'
       WHEN TRY_CAST(a.v AS INTEGER) BETWEEN 0 AND 3 THEN 'number 0-3' WHEN TRY_CAST(a.v AS INTEGER) BETWEEN 4 AND 8 THEN 'number 4-8'
       ELSE 'other' END AS vcls,
  coalesce(upper(a.v) = 'YES' OR TRY_CAST(a.v AS INTEGER) >= 4, false) AS pos_main,
  coalesce((b.v = 'CAM-ICU' AND upper(a.v) = 'YES') OR (b.v = 'ICDSC' AND TRY_CAST(a.v AS INTEGER) >= 4), false) AS pos_conc
FROM nc a LEFT JOIN nc b ON b.s = a.s AND b.t = a.t AND b.nm = 'Delirium Scale'
WHERE a.nm = 'Delirium Score' AND NULLIF(a.v, '') IS NOT NULL""")
    print("\n## 1. Delirium Score (nursecharting: label 'Delirium Scale/Score', name 'Delirium Score')\n")
    print("The primary outcome is the first row with this label and name whose value is 'Yes' (any case) or a number of 4 or "
          "more, in (time zero, min(unit discharge, time zero + 7 days)]. The scale is the 'Delirium Scale' row of the same stay "
          "and time.")
    table(con, "1a. All Delirium Score rows in the cohort's stays: scale charted with the score × value",
          "SELECT scale, vcls, count(*) FROM dl GROUP BY 1, 2 ORDER BY 1, 2", ["Scale", "Value", "Rows"])
    con.execute("""CREATE TEMP TABLE fp AS
WITH a AS (SELECT f.s, min(d.t) AS t FROM fu f JOIN dl d ON d.s = f.s AND d.pos_main AND d.t > f.t0 AND d.t <= f.t_end GROUP BY 1),
     c AS (SELECT f.s, min(d.t) AS t FROM fu f JOIN dl d ON d.s = f.s AND d.pos_conc AND d.t > f.t0 AND d.t <= f.t_end GROUP BY 1)
SELECT f.s, a.t AS t_main, c.t AS t_conc FROM fu f LEFT JOIN a USING (s) LEFT JOIN c USING (s)""")
    npos = con.execute("SELECT count(*) FROM fp WHERE t_main IS NOT NULL").fetchone()[0]
    print(f"\nPatients with a first positive screen within 7 days (main definition, both arms together): {m(npos)}")
    table(con, "1b. The record that defines each patient's first positive screen: scale × value",
          """SELECT d.scale, d.vcls, count(*) FROM fp JOIN dl d ON d.s = fp.s AND d.t = fp.t_main GROUP BY 1, 2 ORDER BY 3 DESC""",
          ["Scale", "Value", "Patients"])
    table(con, "1c. Scale-concordant definition (CAM-ICU 'Yes' or ICDSC ≥4) against the main definition",
          """SELECT CASE WHEN t_main IS NULL THEN 'no positive screen under either definition'
                         WHEN t_conc = t_main THEN 'same first positive screen'
                         WHEN t_conc > t_main THEN 'positive under both, later under the concordant definition'
                         ELSE 'positive only under the main definition' END, count(*)
             FROM fp GROUP BY 1 ORDER BY 2 DESC""", ["Patients", "n"])

    con.execute("""CREATE TEMP TABLE sd AS
SELECT a.s, a.t, TRY_CAST(a.v AS INTEGER) AS v, coalesce(b.v, '(no scale row)') AS scale
FROM nc a LEFT JOIN nc b ON b.s = a.s AND b.t = a.t AND b.nm = 'Sedation Scale'
WHERE a.nm = 'Sedation Score' AND NULLIF(a.v, '') IS NOT NULL""")
    print("\n## 2. Sedation Score (nursecharting: label 'Sedation Scale/Score/Goal', name 'Sedation Score')\n")
    print("asd.py takes as RASS at time zero the last Sedation Score in [time zero − 12 h, time zero] with an integer value from "
          "−5 to 4 (the lowest if several at that time), without reading the 'Sedation Scale' row charted with it. SAS (1–7), "
          "MAAS (0–6) and Ramsay (1–6) scores of 0–4 pass that filter and would be read as RASS 0 to +4.")
    table(con, "2a. All Sedation Score rows in the cohort's stays, by the scale charted with them",
          "SELECT scale, count(*) FROM sd GROUP BY 1 ORDER BY 2 DESC", ["Scale", "Rows"])
    con.execute(f"""CREATE TEMP TABLE r0 AS
WITH w AS (SELECT f.s, d.t, d.v, d.scale FROM fu f JOIN sd d ON d.s = f.s AND d.t BETWEEN f.t0 - 720 AND f.t0 WHERE d.v BETWEEN -5 AND 4),
     l AS (SELECT s, max(t) AS tm FROM w GROUP BY 1)
SELECT w.s, min(w.v) AS rass, min(w.scale) AS scale FROM w JOIN l ON l.s = w.s AND w.t = l.tm GROUP BY 1""")
    con.execute(f"""CREATE TEMP TABLE r1 AS
WITH w AS (SELECT f.s, d.t, d.v FROM fu f JOIN sd d ON d.s = f.s AND d.t BETWEEN f.t0 - 720 AND f.t0
           WHERE d.v BETWEEN -5 AND 4 AND d.scale = 'RASS'),
     l AS (SELECT s, max(t) AS tm FROM w GROUP BY 1)
SELECT w.s, min(w.v) AS rass FROM w JOIN l ON l.s = w.s AND w.t = l.tm GROUP BY 1""")
    con.execute(f"""CREATE TEMP TABLE rr AS SELECT f.s, f.hosp, r0.rass AS rass_main, r0.scale AS scale_main, r1.rass AS rass_conf,
                    {RASS_CAT.format(x='r0.rass')} AS cat_main, {RASS_CAT.format(x='r1.rass')} AS cat_conf
                    FROM fu f LEFT JOIN r0 USING (s) LEFT JOIN r1 USING (s)""")
    table(con, "2b. The value used as RASS at time zero: scale charted with it × category in the model",
          "SELECT coalesce(scale_main, '(no value)'), cat_main, count(*) FROM rr GROUP BY 1, 2 ORDER BY 1, 2",
          ["Scale", "Category used", "Patients"])
    table(con, "2c. Category used × category from scores charted as RASS only",
          "SELECT cat_main, cat_conf, count(*) FROM rr GROUP BY 1, 2 ORDER BY 1, 2",
          ["Category used", "RASS-only category", "Patients"])
    a = con.execute("""SELECT count(*) FILTER (WHERE cat_main <> cat_conf),
                              count(*) FILTER (WHERE (rass_main >= -3) <> (rass_conf >= -3) OR (rass_main IS NULL) <> (rass_conf IS NULL)),
                              count(*) FILTER (WHERE rass_main >= -3), count(*) FILTER (WHERE rass_conf >= -3),
                              count(*) FILTER (WHERE rass_main >= -1), count(*) FILTER (WHERE rass_conf >= -1),
                              count(*) FILTER (WHERE scale_main IS NOT NULL AND scale_main <> 'RASS'), count(*) FILTER (WHERE scale_main IS NOT NULL)
                       FROM rr""").fetchone()
    print(f"\n- Patients whose time-zero category changes: {m(a[0])} of {m(n)} ({pct(a[0], n)})")
    print(f"- Value at time zero from a non-RASS scale: {m(a[6])} of {m(a[7])} patients with a value ({pct(a[6], a[7])})")
    print(f"- RASS ≥ −3 subgroup: {m(a[2])} as analysed; {m(a[3])} with RASS-only scores; membership differs for {m(a[1])}")
    print(f"- RASS ≥ −1 stratum: {m(a[4])} as analysed; {m(a[5])} with RASS-only scores")

    rows = con.execute("""
WITH h AS (SELECT hosp, count(*) AS n, count(scale_main) AS nv, count(*) FILTER (WHERE scale_main <> 'RASS') AS nonr,
                  count(*) FILTER (WHERE cat_main <> cat_conf) AS changed FROM rr GROUP BY 1),
     d AS (SELECT f.hosp, count(*) AS nd, count(*) FILTER (WHERE d.scale = 'ICDSC') AS icdsc FROM fu f JOIN dl d ON d.s = f.s GROUP BY 1),
     sc AS (SELECT f.hosp, count(*) AS ns, count(*) FILTER (WHERE s.scale <> 'RASS') AS nons FROM fu f JOIN sd s ON s.s = f.s GROUP BY 1)
SELECT h.*, coalesce(d.nd, 0), coalesce(d.icdsc, 0), coalesce(sc.ns, 0), coalesce(sc.nons, 0)
FROM h LEFT JOIN d USING (hosp) LEFT JOIN sc USING (hosp) ORDER BY h.n DESC, h.hosp""").fetchall()
    print("\n### 2d. By hospital (labelled by cohort size; hospital identifiers not shown)\n")
    print("| Hospital | Patients | With a time-zero value | Time-zero value not RASS | Category changed | Sedation Score rows not RASS | "
          "Delirium Score rows charted as ICDSC |")
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        _, nn, nv, nonr, ch, nd, icd, ns, nons = r
        print(f"| H{i:02d} | {m(nn)} | {m(nv)} | {m(nonr)} ({pct(nonr, nv)}) | {m(ch)} | {pct(nons, ns)} of {m(ns)} | "
              f"{pct(icd, nd)} of {m(nd)} |")

    print("\n## 3. Timing of the eICU-CRD covariates\n")
    print("APACHE IV and the APACHE ventilation flag (apacheapsvar: intubated or vent) summarise the first ICU day, which "
          "starts at unit admission; time zero is the offset of the first acid-suppressant order from unit admission.")
    table(con, "3a. Time zero from ICU admission",
          """SELECT CASE WHEN t0 < 60 THEN '1 <1 h' WHEN t0 < 360 THEN '2 1-6 h' WHEN t0 < 1440 THEN '3 6-24 h' ELSE '4 >=24 h' END, count(*)
             FROM fu GROUP BY 1 ORDER BY 1""", ["Band", "Patients"])
    con.execute("""CREATE TEMP TABLE av AS SELECT patientunitstayid AS s, max(CASE WHEN intubated = 1 OR vent = 1 THEN 1 ELSE 0 END) AS vent1
                   FROM eicu.apacheapsvar WHERE patientunitstayid IN (SELECT s FROM fu) GROUP BY 1""")
    con.execute(f"""CREATE TEMP TABLE vt AS
SELECT patientunitstayid AS s, respchartoffset AS t, 'chart' AS src FROM eicu.respiratorycharting
WHERE respchartvaluelabel IN ({VENT_LABELS}) AND patientunitstayid IN (SELECT s FROM fu)
UNION ALL
SELECT patientunitstayid, treatmentoffset, 'treatment' FROM eicu.treatment
WHERE (treatmentstring LIKE 'pulmonary|ventilation and oxygenation|mechanical ventilation%' AND treatmentstring NOT LIKE '%non-invasive%'
       OR treatmentstring = 'surgery|pulmonary therapies|mechanical ventilation') AND patientunitstayid IN (SELECT s FROM fu)
UNION ALL
SELECT patientunitstayid, ventstartoffset, 'respcare' FROM eicu.respiratorycare
WHERE ventstartoffset IS NOT NULL AND ventstartoffset <> 0 AND patientunitstayid IN (SELECT s FROM fu)""")
    con.execute("""CREATE TEMP TABLE vc AS
SELECT f.s, coalesce(a.vent1, 0) AS vent1,
  EXISTS (SELECT 1 FROM vt v WHERE v.s = f.s AND ((v.src = 'chart' AND v.t BETWEEN f.t0 - 360 AND f.t0)
                                              OR (v.src = 'treatment' AND v.t BETWEEN f.t0 - 1440 AND f.t0)
                                              OR (v.src = 'respcare' AND v.t <= f.t0 AND v.t >= f.t0 - 1440))) AS pre,
  EXISTS (SELECT 1 FROM vt v WHERE v.s = f.s AND v.t > f.t0 AND v.t <= f.t0 + 1440) AS post
FROM fu f LEFT JOIN av a USING (s)""")
    table(con, "3b. First-day ventilation flag × time-stamped ventilator records",
          """SELECT CASE vent1 WHEN 1 THEN 'flag: ventilated' ELSE 'flag: not ventilated' END,
                    CASE WHEN pre THEN 'record before or at time zero' WHEN post THEN 'record only in the 24 h after time zero'
                         ELSE 'no ventilator record in either window' END, count(*)
             FROM vc GROUP BY 1, 2 ORDER BY 1, 2""", ["APACHE flag", "Time-stamped records", "Patients"],
          note="Ventilator records: respiratorycharting ventilator settings or measurements in the 6 h before time zero, a "
               "'mechanical ventilation' treatment entry (not non-invasive) in the 24 h before, or a respiratorycare ventilation start "
               "in the 24 h before; 'after' covers the same sources in the 24 h after time zero.")
    con.close()


if __name__ == "__main__":
    main()
