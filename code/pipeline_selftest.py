import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import glob
import io
import json
import math
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asd
import stats_core as sc

RESULTS = []
BETA = {1: -0.45, 2: -0.30}


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)


def write_df(df, path):
    import duckdb
    con = duckdb.connect()
    con.register("df", df)
    con.execute(f"COPY df TO '{path}' (FORMAT parquet)")
    con.close()


def hours_sim(h, rng, H=168):
    g = rng.geometric(np.clip(h, 1e-9, 1)) - 1
    return np.where(g < H, g + 0.5, np.nan)


def make_db(P, db, n, rng):
    stay = np.arange(1, n + 1) + (10 ** 6 if db == "e" else 3 * 10 ** 7)
    age = np.clip(rng.normal(64, 15, n), 18, 91).round()
    female = rng.random(n) < 0.42
    scr = np.exp(rng.normal(0.1, 0.5, n))
    egfr = 142 * np.minimum(scr / np.where(female, 0.7, 0.9), 1) ** np.where(female, -0.241, -0.302) \
        * np.maximum(scr / np.where(female, 0.7, 0.9), 1) ** -1.2 * 0.9938 ** age * np.where(female, 1.012, 1)
    miss_egfr = rng.random(n) < (0.05 if db == "m" else 0.25)
    egfr_obs = np.where(miss_egfr, np.nan, egfr)
    vent = rng.random(n) < 0.35
    vaso = rng.random(n) < 0.2
    sed = rng.random(n) < np.where(vent, 0.6, 0.05)
    opi = rng.random(n) < np.where(vent, 0.5, 0.05)
    rass = np.where(sed, rng.integers(-5, -1, n), rng.integers(-2, 2, n)).astype(float)
    rass[rng.random(n) < (0.1 if db == "m" else 0.35)] = np.nan
    sev = np.clip(rng.normal(5, 3, n) + 2 * vent + 2 * vaso, 0, 20).round() if db == "m" else np.clip(rng.normal(55, 20, n) + 15 * vent, 10, 160).round()
    sev[rng.random(n) < (0.01 if db == "m" else 0.1)] = np.nan
    inr = np.where(rng.random(n) < 0.6, np.exp(rng.normal(0.2, 0.25, n)), np.nan)
    plt_ = np.where(rng.random(n) < 0.95, np.clip(rng.normal(200, 90, n), 5, 900), np.nan)
    liver = rng.random(n) < 0.06
    dementia = rng.random(n) < 0.05
    alcohol = rng.random(n) < 0.08
    h_icu = rng.exponential(14, n)
    if db == "m":
        era = rng.choice(["2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"], n)
        unit = rng.choice(list(asd.M_UNIT.keys()) + ["PACU"], n)
        service = rng.choice(["MED", "SURG", "CMED", "NSURG", "TRAUM"], n)
        site = np.full(n, np.nan)
    else:
        era = rng.choice(["2014", "2015"], n, p=[0.4, 0.6])
        unit = rng.choice(list(asd.E_UNIT.keys()), n)
        service = rng.choice(["Emergency Department", "Floor", "Operating Room", "Recovery Room"], n)
        site = rng.choice(np.arange(100, 122), n).astype(float)
    route = rng.choice(["IV", "EN", "UNK"], n, p=[0.6, 0.35, 0.05])
    site_eff = (np.nan_to_num(site) % 7 - 3) * 0.15
    coag = (np.nan_to_num(inr) >= 1.5) | (np.nan_to_num(plt_, nan=200) < 50)
    lin2 = -0.3 + 0.5 * vent + 0.4 * coag + 0.6 * liver + 0.3 * dementia + site_eff
    A2 = (rng.random(n) < sc.expit(lin2)).astype(int)
    c1 = (A2 == 0) & (egfr < 60)
    p1 = sc.expit(1.2 - 0.08 * (egfr - 30) + 0.4 * (age >= 80))
    A1 = (rng.random(n) < p1).astype(int)
    other1 = c1 & (rng.random(n) < 0.08)
    daily = np.where(A1 == 1, 20.0, 40.0)
    daily[other1] = np.nan
    lin_d = -5.1 + 0.02 * (age - 65) + 0.5 * sed + 0.6 * (np.nan_to_num(rass, nan=0) <= -4) + 0.3 * vent + 0.5 * dementia + 0.3 * alcohol
    lin_d = lin_d + BETA[2] * A2 + np.where(c1, BETA[1] * A1, 0.0)
    t_del_all = hours_sim(sc.expit(lin_d), rng)
    t_death = hours_sim(sc.expit(-7.6 + 0.03 * (age - 65) + 0.6 * vaso), rng, H=400)
    t_dis = hours_sim(sc.expit(-4.1 - 0.4 * vent - 0.3 * vaso), rng, H=400)
    t_end = np.fmin(np.nan_to_num(t_death, nan=999), np.nan_to_num(t_dis, nan=999))
    end_death = np.nan_to_num(t_death, nan=999) <= np.nan_to_num(t_dis, nan=999)
    lim = np.minimum(t_end, 168)
    t_del = np.where(t_del_all <= lim, t_del_all, np.nan)
    t_h_all = hours_sim(sc.expit(-6.4 + 0.5 * liver + 0.3 * vaso + 0.2 * (np.nan_to_num(egfr_obs, nan=90) < 30)), rng)
    t_hypo = np.where(t_h_all <= lim, t_h_all, np.nan)
    hosp_death = end_death | (rng.random(n) < sc.expit(-2.5 + 0.03 * (age - 65)))
    t_dev = hours_sim(sc.expit(-6.0 + 0.4 * vent), rng)
    t0 = pd.Timestamp("2150-01-01") + pd.to_timedelta(rng.integers(0, 10 ** 6, n), unit="m")
    egfr_gate = np.where(miss_egfr, 142 * 0.9938 ** age * np.where(female, 1.012, 1), egfr)
    base = {"subject_id": stay, "stay_id": stay, "gender": np.where(female, "F" if db == "m" else "Female", "M" if db == "m" else "Male"),
            "era": era, "site": site, "age": age, "scr": np.where(miss_egfr, np.nan, scr), "egfr": egfr_gate}
    if db == "m":
        base.update({"hadm_id": stay + 10 ** 6, "intime": t0 - pd.to_timedelta(h_icu, unit="h"), "outtime": t0 + pd.to_timedelta(t_end, unit="h"),
                     "t0": t0})
    else:
        base.update({"t0": (h_icu * 60).round().astype(np.int64)})
    co = pd.DataFrame(base)
    c2 = co.assign(daily=np.nan, arm=np.where(A2 == 1, "A", "B"))
    c1d = co[c1].assign(daily=daily[c1], arm=np.where(other1[c1], None, np.where(A1[c1] == 1, "A", "B")))
    for comparison, df in ((1, c1d), (2, c2)):
        cols = (["subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender", "era", "site", "age", "t0", "scr", "egfr", "daily", "arm"]
                if db == "m" else ["subject_id", "stay_id", "gender", "era", "site", "age", "t0", "scr", "egfr", "daily", "arm"])
        write_df(df[cols].reset_index(drop=True), os.path.join(P, f"{db}_c{comparison}_cohort.parquet"))
    cv = pd.DataFrame({"stay_id": stay, "rass": rass, "vent": vent.astype(float), "vaso": vaso.astype(int), "sed_inf": sed.astype(int),
                       "opi_inf": opi.astype(int), "inr_max": inr, "plt_min": plt_, "unit": unit, "service": service, "route": route,
                       "h_icu": h_icu, "dementia": dementia.astype(int), "alcohol": alcohol.astype(int), "liver": liver.astype(int)})
    cv["sofa" if db == "m" else "apache"] = sev
    write_df(cv, os.path.join(P, f"{db}_covars.parquet"))
    write_df(pd.DataFrame({"stay_id": stay, "t_end": t_end, "end_death": end_death}), os.path.join(P, f"{db}_follow.parquet"))
    for comparison, m in ((1, c1 & ~other1), (2, np.ones(n, bool))):
        write_df(pd.DataFrame({"stay_id": stay[m], "t_dev": t_dev[m]}), os.path.join(P, f"{db}_c{comparison}_dev.parquet"))
    out = pd.DataFrame({"stay_id": stay, "t_del": t_del, "t_hypo": t_hypo, "hosp_death": hosp_death})
    return out, {"c1": int(c1.sum()), "c2": n}


def run(fn, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(argparse.Namespace(**kw))
    return buf.getvalue()


def masked_counts_ok(obj, path=""):
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("eventsA", "eventsB", "nA", "nB", "eligible", "A", "B", "other", "outside_overlap", "analysis_A", "analysis_B",
                     "patients_Z1", "patients_Z0") and isinstance(v, str):
                n0 = v.split(" ")[0]
                if not (n0 == "<10" or (n0.isdigit() and int(n0) % 5 == 0)):
                    bad.append(path + "/" + k)
                m = re.search(r"\((\d+)%\)", v)
                if m and n0 == "<10":
                    bad.append(path + "/" + k + " (percentage with <10)")
            bad += masked_counts_ok(v, path + "/" + str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += masked_counts_ok(v, f"{path}[{i}]")
    return bad


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261013)
    tmp = tempfile.mkdtemp(prefix="asd_pipe_")
    P, R = os.path.join(tmp, "private"), os.path.join(tmp, "results")
    os.makedirs(P)
    os.makedirs(R)
    try:
        outs, sizes = {}, {}
        for db, n in (("m", 6000), ("e", 5000)):
            outs[db], sizes[db] = make_db(P, db, n, rng)
        check("synthetic private tables written", len(os.listdir(P)) == 12, f"{len(os.listdir(P))} files; dose comparison sizes {sizes['m']['c1']}/{sizes['e']['c1']}")

        run(asd.design, private=P, results=R)
        s1 = json.load(open(os.path.join(R, "stage1_design.json"), encoding="utf-8"))
        check("design writes stage 1 diagnostics for the four database-comparison keys", sorted(s1["keys"]) == sorted(asd.KEYS)
              and os.path.exists(os.path.join(R, "stage1_design.md")))
        k = s1["keys"]["c2e"]
        check("stage 1 includes balance, weights, follow-up, per-protocol, subsets and the eICU instrument",
              all(x in k for x in ("smd", "ess", "ipcw_discharge", "pp_deviation_in_followup", "subsets", "iv")) and "first_stage_F_cluster" in k["iv"],
              f"c2e max |SMD| {k['max_abs_smd']}; IV F {k['iv'].get('first_stage_F_cluster')}")
        check("overlap weights balance every model covariate in stage 1", all(s1["keys"][x]["max_abs_smd_model_columns"]["overlap"] < 0.005 for x in asd.KEYS),
              "; ".join(f"{x} {s1['keys'][x]['max_abs_smd_model_columns']['overlap']}" for x in asd.KEYS))
        bad = masked_counts_ok(s1)
        check("stage 1 counts are masked", not bad, ", ".join(bad[:5]))
        check("PPI comparison: eGFR counts as missing where creatinine is missing (revision R1)",
              all("egfr_missing" in s1["keys"][x]["columns"] for x in ("c2m", "c2e")))
        md = open(os.path.join(R, "stage1_design.md"), encoding="utf-8").read()
        check("stage 1 report names no outcome", not re.search(r"delirium|Delirium|hypoglycaemia|t_del|t_hypo|hosp_death", md))

        try:
            run(asd.extract, db="mimic", part="outcome", private=P, results=R, confirmed=None)
            ok = False
        except SystemExit:
            ok = True
        check("outcome extraction refuses to run without --confirmed", ok)
        for db in ("m", "e"):
            write_df(outs[db], os.path.join(P, f"{db}_outcome.parquet"))
        try:
            run(asd.design, private=P, results=R)
            ok = False
        except SystemExit:
            ok = True
        check("design refuses to run once an outcome table exists", ok)

        run(asd.estimate, private=P, results=R)
        est = {key: json.load(open(os.path.join(P, f"est_{key}.json"))) for key in asd.KEYS}
        check("estimate writes point estimates for every analysis", all(set(asd.boot_names(int(key[1]), key[2])) <= set(est[key]["res"]) for key in asd.KEYS))
        tb = time.time()
        for key in asd.KEYS:
            run(asd.boot, key=key, start=0, stop=30, workers=2, private=P, results=R)
            run(asd.boot, key=key, start=30, stop=60, workers=1, private=P, results=R)
        bt = asd.load_boot(P, "c2m")
        check("bootstrap replicates run in chunks (2 workers and 1 worker give the same format)", bt is not None and bt["lor"].shape[0] == 60
              and np.isfinite(bt["lor"][:, 0]).all(), f"{time.time() - tb:.0f} s for 4 keys x 60 replicates")
        b1 = asd.load_boot(P, "c1m")
        run(asd.boot, key="c1m", start=0, stop=30, workers=1, private=P, results=R)
        b2 = asd.load_boot(P, "c1m")
        check("bootstrap replicates are reproducible (seed 20261013 + replicate)", np.allclose(b1["lor"], b2["lor"], equal_nan=True))
        for name in ("m_c1_cohort", "m_covars", "m_follow", "m_c1_dev", "m_outcome"):
            f = os.path.join(P, name + ".parquet")
            write_df(asd.read_pq(f).sample(frac=1.0, random_state=7).reset_index(drop=True), f)
        for f in glob.glob(os.path.join(P, "boot_c1m_*.npz")):
            os.remove(f)
        run(asd.boot, key="c1m", start=0, stop=30, workers=1, private=P, results=R)
        b3 = asd.load_boot(P, "c1m")
        check("bootstrap replicates do not depend on the row order of the private tables (revision R6)",
              b3 is not None and np.allclose(b2["lor"][:30], b3["lor"], equal_nan=True))
        run(asd.boot, key="c1m", start=30, stop=60, workers=1, private=P, results=R)

        run(asd.report, private=P, results=R)
        c = {comparison: json.load(open(os.path.join(R, f"{asd.STEM[comparison]}_estimates.json"), encoding="utf-8")) for comparison in (1, 2)}
        check("report writes per-comparison estimates and the Bayesian reanalysis", all(os.path.exists(os.path.join(R, f)) for f in
              ("famdose_estimates.md", "ppi_estimates.md", "bayes.txt")) and "evidence-based" in c[1].get("bayes", "") and "skeptical" in c[2].get("bayes", ""))
        bad = masked_counts_ok(c)
        check("report counts are masked", not bad, ", ".join(bad[:5]))
        for comparison in (1, 2):
            p = c[comparison]["pooled"]["primary"]
            ok = p.get("OR") is not None and p["OR_lo"] < p["OR"] < p["OR_hi"] and p["k"] == 2
            check(f"{asd.NAME[comparison]}: pooled primary OR with a Wald interval from the bootstrap SEs", ok,
                  f"OR {p.get('OR', float('nan')):.3f} ({p.get('OR_lo', float('nan')):.3f}-{p.get('OR_hi', float('nan')):.3f}); synthetic hourly logit effect {BETA[comparison]}"
                  f"; conclusion: {c[comparison]['conclusion']}")
        p1 = c[1]["pooled"]["primary"]
        p2 = c[2]["pooled"]["primary"]
        check("synthetic benefit is recovered in direction (OR < 1 for both comparisons)", p1["OR"] < 1 and p2["OR"] < 1)
        nc = c[2]["pooled"]["negctl"]
        check("negative-control outcome (no synthetic effect) is near 1 for the PPI comparison", 0.85 < nc["OR"] < 1.18,
              f"OR {nc['OR']:.3f} ({nc['OR_lo']:.3f}-{nc['OR_hi']:.3f}); {c[2]['negative_control']}")
        risky = [(comparison, db, nm) for comparison in (1, 2) for db in ("m", "e") for nm, e in c[comparison]["per_db"][db]["analyses"].items()
                 if e.get("eventsA") == "<10" and e.get("riskA") is not None]
        check("an arm's risk is withheld when its events are < 10", not risky)

        Pt = os.path.join(tmp, "timing")
        os.makedirs(Pt)
        outs_t, _ = make_db(Pt, "m", 12000, rng)
        write_df(outs_t, os.path.join(Pt, "m_outcome.parquet"))
        df, _ = asd.load_key(Pt, "c2m", stage2=True)
        d = asd.build(df, "m", 2, stage2=True)
        cache = {}
        ts = time.time()
        asd.run_stage2(d, 2, "m", cache)
        t_point = time.time() - ts
        ts = time.time()
        for rep in range(3):
            asd._boot_one((d, 2, "m", rep, cache))
        check("timing at 12,000 patients (PPI comparison, MIMIC size)", True, f"point estimate {t_point:.1f} s; {(time.time() - ts) / 3:.1f} s per replicate (one worker)")
    finally:
        if os.environ.get("ASD_KEEP"):
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed ({time.time() - t0:.0f} s)")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
