#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc

SQL = f"""
WITH c AS (SELECT hadm_id, stay_id, intime, outtime, t0, arm FROM read_parquet('{{f}}') WHERE arm IS NOT NULL),
o AS (SELECT r.hadm_id, r.starttime AS t,
             CASE WHEN regexp_matches(lower(r.drug), 'famotidine|pepcid') THEN 'FAM'
                  WHEN regexp_matches(lower(r.drug), 'ranitidine|zantac|cimetidine|nizatidine') THEN 'H2O' ELSE 'PPI' END AS drug
      FROM hosp.prescriptions r
      WHERE regexp_matches(lower(r.drug), '{asd.AS_RX}') AND r.starttime IS NOT NULL AND r.hadm_id IS NOT NULL),
f AS (SELECT c.stay_id, min(o.t) AS t_ord FROM c JOIN o ON o.hadm_id = c.hadm_id AND o.t BETWEEN c.intime AND c.outtime GROUP BY 1),
k AS (SELECT f.stay_id, f.t_ord, CASE WHEN count(DISTINCT o.drug) = 1 THEN min(o.drug) END AS drug
      FROM f JOIN c USING (stay_id) JOIN o ON o.hadm_id = c.hadm_id AND o.t = f.t_ord GROUP BY 1, 2)
SELECT c.stay_id, c.arm, date_diff('minute', k.t_ord, c.t0) / 60.0 AS h_order_to_dose, k.drug
FROM c LEFT JOIN k USING (stay_id)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, asd.TOOLS)
    from mimic_local import connect
    con = connect(a.mimic, memory_limit="8GB", threads=8, temp_dir=os.path.join(a.private, "duck_tmp"))
    df = con.execute(SQL.replace("{f}", os.path.join(a.private, "m_c2_cohort.parquet"))).df()
    con.close()
    h = df["h_order_to_dose"].astype(float).values
    has = np.isfinite(h)
    drug = df["drug"].astype(str).values
    arm = df["arm"].astype(str).values
    agree = has & (((arm == "A") & (drug == "PPI")) | ((arm == "B") & (drug == "FAM")))
    other = has & (((arm == "A") & np.isin(drug, ("FAM", "H2O"))) | ((arm == "B") & (drug == "PPI")))

    def sh(x, r):
        k, n = int((x & r).sum()), int(r.sum())
        return {"n": sc.mask_n(k), "pct": round(100.0 * k / n, 1) if (n > 0 and k >= 10) else None}

    def q(v):
        v = v[np.isfinite(v)]
        return [round(float(x), 1) for x in np.percentile(v, [25, 50, 75])] if v.size >= 20 else None

    out = {"plan": "SAP_addendum_R16.md v1.0, section 17 (A17)", "by_arm": {}}
    for lab, r in (("all", np.ones(len(df), bool)), ("A", arm == "A"), ("B", arm == "B")):
        before = has & (h >= 0)
        out["by_arm"][lab] = {"n": sc.mask_n(int(r.sum())), "order_in_icu": sh(has, r), "class_agrees": sh(agree, r & has),
                              "other_class_ordered_first": sh(other, r & has), "two_classes_at_first_order": sh(has & ~agree & ~other, r & has),
                              "order_starts_at_or_before_first_dose": sh(before, r & has), "order_starts_after_first_dose": sh(has & (h < 0), r & has),
                              "hours_order_to_first_dose_q": q(h[r & before]), "within_1h": sh(before & (h <= 1), r & before),
                              "within_6h": sh(before & (h <= 6), r & before)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for lab, e in out["by_arm"].items():
        print(lab, json.dumps(e, ensure_ascii=False))


if __name__ == "__main__":
    main()
