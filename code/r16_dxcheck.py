#!/usr/bin/env python3
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc
import r16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--eicu", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    df, _ = asd.load_key(a.private, "c2e", stage2=True)
    ids = [int(x) for x in df["stay_id"]]
    con = r16.connect_full(a.eicu, a.private)
    con.execute("CREATE TEMP TABLE coh AS SELECT unnest(?) AS stay_id", [ids])
    rows = con.execute("""
        SELECT coalesce(p.apacheadmissiondx, '(none)') AS dx, count(*) AS n
        FROM coh c JOIN eicu_derived.apache_groups g ON g.patientunitstayid = c.stay_id
        JOIN eicu.patient p ON p.patientunitstayid = c.stay_id
        WHERE g.apachedxgroup = 'GIBleed' GROUP BY 1 ORDER BY 2 DESC, 1""").fetchall()
    out = {"note": "eICU-CRD analysed cohort (both arms together), APACHE admission diagnoses of the patients in the derived group GIBleed; masked counts",
           "n_cohort": sc.mask_n(len(ids)), "n_group": sc.mask_n(sum(r[1] for r in rows)), "diagnoses": [{"apacheadmissiondx": r[0], "n": sc.mask_n(r[1])} for r in rows]}
    json.dump(out, open(os.path.join(a.results, "r16_dxcheck.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
