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
import r7
import r11
import r12

PLAN = "SAP_addendum_R14.md v1.0"
NBOOT = 2000
E_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "eicu-crd-2.0-parquet")
SPECS = ("main", "pre")
OUTS = ("primary", "ic", "negctl")
STRATA = (("flag", "vent_flag", "main"), ("pre", "vent_pre", "pre"))
NAMES = [f"{s}_{o}" for s in SPECS for o in OUTS] + [f"strat_{k}_{lv}" for k, _, _ in STRATA for lv in (0, 1)]
NEW_COLS = ["vent_pre", "lactate_ge2", "lactate_ge4", "lactate_not_measured"]
SPEC_LAB = {"main": "Main-analysis model (APACHE IV, APACHE first-day ventilation flag)",
            "pre": "Pre-time-zero covariate model (APACHE IV removed; ventilation from time-stamped records before time zero; highest lactate in the 24 h before time zero added)"}
OUT_LAB = {"primary": "Delirium (as the main analysis, 168-h odds ratio)", "ic": "Onset hazard ratio taking assessment times into account (interval-censored, baseline in 3 segments)",
           "negctl": "Negative control: first glucose <70 mg/dL (168-h odds ratio)"}
STRAT_LAB = {"flag": "APACHE first-day ventilation flag (E1 strata, main-analysis model)", "pre": "Time-stamped invasive ventilation records before time zero (pre-time-zero covariate model)"}
VENT_LABELS = ("'Vent Rate', 'Tidal Volume (set)', 'TV/kg IBW', 'Plateau Pressure', 'Peak Insp. Pressure', 'Mean Airway Pressure', "
               "'Exhaled MV', 'Exhaled TV (machine)', 'Pressure Control', 'RT Vent On/Off', 'Adult Con Setting Set RR', "
               "'Adult Con Setting Set Vt', 'Set Vt (Servo,LTV)'")
E1_VENT = {0: 0.8405, 1: 1.1674}

PRE_E = f"""
WITH c AS (SELECT stay_id, t0 FROM read_parquet('{{f}}') WHERE arm IS NOT NULL),
vt AS (
  SELECT patientunitstayid AS s, respchartoffset AS t, 'chart' AS src FROM eicu.respiratorycharting
  WHERE respchartvaluelabel IN ({VENT_LABELS}) AND patientunitstayid IN (SELECT stay_id FROM c)
  UNION ALL
  SELECT patientunitstayid, treatmentoffset, 'treatment' FROM eicu.treatment
  WHERE (treatmentstring LIKE 'pulmonary|ventilation and oxygenation|mechanical ventilation%' AND treatmentstring NOT LIKE '%non-invasive%'
         OR treatmentstring = 'surgery|pulmonary therapies|mechanical ventilation') AND patientunitstayid IN (SELECT stay_id FROM c)
  UNION ALL
  SELECT patientunitstayid, ventstartoffset, 'respcare' FROM eicu.respiratorycare
  WHERE ventstartoffset IS NOT NULL AND ventstartoffset <> 0 AND patientunitstayid IN (SELECT stay_id FROM c)),
lac AS (SELECT c.stay_id, max(l.labresult) AS lac_max FROM c JOIN eicu.lab l ON l.patientunitstayid = c.stay_id
          AND l.labresultoffset BETWEEN c.t0 - 1440 AND c.t0
        WHERE l.labname = 'lactate' AND l.labresult > 0 GROUP BY 1)
SELECT c.stay_id,
  EXISTS (SELECT 1 FROM vt v WHERE v.s = c.stay_id AND ((v.src = 'chart' AND v.t BETWEEN c.t0 - 360 AND c.t0)
          OR (v.src IN ('treatment', 'respcare') AND v.t BETWEEN c.t0 - 1440 AND c.t0))) AS vent_pre,
  lac.lac_max
FROM c LEFT JOIN lac ON lac.stay_id = c.stay_id
"""
EXPECT = "stay_id vent_pre lac_max"


def connect_full(P):
    sys.path.insert(0, asd.TOOLS)
    from mimic_local import connect
    return connect(E_ROOT, temp_dir=os.path.join(P, "duck_tmp"))


def extract(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    con = connect_full(P)
    path = r7.pf(P, "e", "r14pre")
    asd.write_pq(con, PRE_E.format(f=r7.pf(P, "e", "c2_cohort")), path)
    n, nv, nl = con.execute(f"SELECT count(*), count(*) FILTER (WHERE vent_pre), count(lac_max) FROM read_parquet('{path}')").fetchone()
    con.close()
    asd.status(f"{os.path.basename(path)} written: {sc.mask_n(n)} rows, ventilated before time zero {sc.mask_n(nv)}, "
               f"lactate measured {sc.mask_n(nl)} (masked, both arms together)", t0)
    asd.log_run(args.results, f"r14 extract: ok ({time.time() - t0:.0f} s)")


def pre_matrix(X, names, vent_pre, lac):
    keep = [j for j, nm in enumerate(names) if not (nm.startswith("apache_") or nm == "vent")]
    lac = np.asarray(lac, float)
    new = np.column_stack([np.asarray(vent_pre, bool), lac >= 2, lac >= 4, ~np.isfinite(lac)]).astype(float)
    Xp = np.column_stack([X[:, keep], new])
    return asd.prevalence_filter(Xp, [names[j] for j in keep] + NEW_COLS)


def add_pre(d, vent_flag, vent_pre, lac):
    d["vent_flag"] = np.asarray(vent_flag, bool)
    d["vent_pre"] = np.asarray(vent_pre, bool)
    d["lac"] = np.asarray(lac, float)
    d["X_pre"], d["names_pre"], d["dropped_pre"] = pre_matrix(d["X"], d["names"], d["vent_pre"], d["lac"])
    return d


def load_e(P):
    df, flow = asd.load_key(P, "c2e", stage2=True)
    n0 = len(df)
    df = df.merge(asd.read_pq(r7.pf(P, "e", "r14pre")), on="stay_id", how="left", validate="one_to_one")
    if len(df) != n0 or df["vent_pre"].isna().any():
        raise SystemExit("r14 rows missing")
    d = asd.build(df, "e", 2, stage2=True)
    d.update(r11.timeline_arrays(d, df["stay_id"].values, asd.read_pq(r7.pf(P, "e", "r11tl"))))
    add_pre(d, np.nan_to_num(asd.num(df, "vent")) == 1, df["vent_pre"].astype(bool).values, asd.num(df, "lac_max"))
    return d, flow


def spec_d(d, spec):
    return d if spec == "main" else dict(d, X=d["X_pre"], names=d["names_pre"])


def stratum(d, m, spec, cache, key):
    A = d["A"]
    nA, nB = int((m & (A == 1)).sum()), int((m & (A == 0)).sum())
    if min(nA, nB) < r11.MIN_N:
        return {"OR": float("nan"), "nA": nA, "nB": nB}
    s = asd.subset_d(spec_d(d, spec), m)
    e = r11.cif(s, r11.ow_of(s, cache, key), s["t_del"])
    e["nA"], e["nB"] = nA, nB
    return e


def estimates(d, cache):
    out = {}
    for s in SPECS:
        dd = spec_d(d, s)
        w = r11.ow_of(dd, cache, s)
        out[f"{s}_primary"] = r11.cif(dd, w, dd["t_del"])
        out[f"{s}_ic"] = r11.ic_fit(dd, w, 3)
        out[f"{s}_negctl"] = r11.cif(dd, w, dd["t_hypo"])
    for k, var, spec in STRATA:
        for lv in (0, 1):
            out[f"strat_{k}_{lv}"] = stratum(d, d[var] == bool(lv), spec, cache, f"strat_{k}_{lv}")
    return out


def flat(res):
    vals = []
    for nm in NAMES:
        x = res[nm].get("HR" if nm.endswith("_ic") else "OR")
        vals.append(math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan)
    return np.array(vals)


def describe(d):
    f, p = d["vent_flag"], d["vent_pre"]
    return {"flag1_pre1": int((f & p).sum()), "flag1_pre0": int((f & ~p).sum()), "flag0_pre1": int((~f & p).sum()),
            "flag0_pre0": int((~f & ~p).sum()), "lac_measured": int(np.isfinite(d["lac"]).sum()), "n": int(len(d["A"])),
            "dropped_pre": [c for c in d["dropped_pre"] if c in NEW_COLS],
            "removed": [c for c in d["names"] if c.startswith("apache_") or c == "vent"]}


def core_estimate(d, flow):
    cache = {}
    res = estimates(d, cache)
    w = r11.ow_of(spec_d(d, "pre"), dict(cache), "pre")
    smd = sc.smd_table(d["X_pre"], d["names_pre"], d["A"], w)
    vals = [abs(v) for v in smd.values() if np.isfinite(v)]
    return {"plan": PLAN, "flow": flow, "res": asd.to_json(res), "desc": describe(d),
            "smd_pre_max": max(vals) if vals else None}, cache


def estimate(args):
    t0 = time.time()
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    d, flow = load_e(P)
    est, cache = core_estimate(d, flow)
    res = est["res"]
    c2 = json.load(open(os.path.join(R, "ppi_estimates.json"), encoding="utf-8"))["per_db"]["e"]["analyses"]
    j11 = json.load(open(os.path.join(R, "r11.json"), encoding="utf-8"))["db"]["e"]
    checks = [("primary OR, main model, vs ppi_estimates.json", res["main_primary"]["OR"], c2["primary"]["OR"], 1e-9),
              ("negative-control OR, main model, vs ppi_estimates.json", res["main_negctl"]["OR"], c2["negctl"]["OR"], 1e-9),
              ("onset HR, main model, vs r11.json", res["main_ic"]["HR"], j11["ic"]["HR"], 1e-9),
              ("not ventilated (APACHE flag) vs E1", res["strat_flag_0"]["OR"], E1_VENT[0], 5e-4),
              ("ventilated (APACHE flag) vs E1", res["strat_flag_1"]["OR"], E1_VENT[1], 5e-4)]
    good = True
    for name, a, b, tol in checks:
        ok = abs(round(a, 4) - b) < tol + 1e-12
        good &= ok
        print(f"{'PASS' if ok else 'FAIL'} | {name}: {a:.4f} vs {b:.4f}", flush=True)
    if not good:
        asd.log_run(args.results, "r14 estimate: REPRODUCTION CHECK FAILED")
        raise SystemExit("the rebuilt eICU data differ from R6/R11/E1: stop")
    json.dump(est, open(os.path.join(P, "r14_est.json"), "w"))
    np.savez(os.path.join(P, "r14_cache.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
    asd.status(f"r14 estimate done; new-model columns {len(d['names_pre'])} (main {len(d['names'])})", t0)
    asd.log_run(args.results, f"r14 estimate: ok, reproduction checks passed ({time.time() - t0:.0f} s)")


_G = {}


def _one(rep):
    d, cache = _G["d"], _G["cache"]
    try:
        b = r12.draw(d, "hosp", rep)
        return rep, flat(estimates(b, dict(cache)))
    except Exception:
        return rep, None


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    d, _ = load_e(P)
    z = np.load(os.path.join(P, "r14_cache.npz"))
    _G.update(d=d, cache={k.replace("__", ":"): z[k] for k in z.files})
    reps = list(range(args.start, min(args.stop, NBOOT)))
    out = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for rep, v in pool.imap_unordered(_one, reps, chunksize=5):
                out[rep] = v
    else:
        for rep in reps:
            out[rep] = _one(rep)[1]
    V = np.full((len(reps), len(NAMES)), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None:
            V[i] = out[rep]
    np.savez(os.path.join(P, f"r14boot_{args.start:04d}_{args.stop:04d}.npz"), reps=np.array(reps), v=V, names=np.array(NAMES))
    failed = sum(1 for r in reps if out.get(r) is None)
    asd.status(f"r14 boot {args.start}-{args.stop} done, failed {failed}", t0)
    asd.log_run(args.results, f"r14 boot --start {args.start} --stop {args.stop}: ok, failed {failed} ({time.time() - t0:.0f} s)")


def load_boots(P):
    files = sorted(glob.glob(os.path.join(P, "r14boot_*.npz")))
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


MAIN_CI = (0.8977, 1.1306)


def read_pre(e, main_reading):
    if e.get("lo") is None:
        return "inconclusive"
    same = r11.inside(e["OR"], *MAIN_CI) and asd.interpret(e["OR"], e["lo"], e["hi"]) == main_reading
    return ("the eICU estimate does not depend on covariates that may hold information from after time zero (APACHE IV, first-day ventilation flag)" if same
            else "with the pre-time-zero covariate model the eICU estimate changed")


def read_negctl(e):
    if e.get("lo") is None:
        return "inconclusive"
    return "negative control suggests residual confounding" if (e["lo"] > 1 or e["hi"] < 1) else "negative control does not suggest residual confounding"


def read_strat(r):
    if not r:
        return "inconclusive"
    if r["lo"] > 1:
        return "higher odds ratio in ventilated patients: effect modification supported by this definition"
    if r["hi"] < 1:
        return "lower odds ratio in ventilated patients: effect modification supported by this definition (direction opposite to E1)"
    return "effect modification by this definition inconclusive"


def report(args):
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    est = json.load(open(os.path.join(P, "r14_est.json")))
    B = load_boots(P)
    E = est["res"]
    out = {"plan": PLAN, "boot_reps": B["n"] if B else 0, "flow": {k: sc.mask_n(v) for k, v in est["flow"].items()},
           "est": {}, "ratio": {}, "reading": {}}
    for nm in NAMES:
        x = E[nm]
        key = "HR" if nm.endswith("_ic") else "OR"
        ent = r11.with_ci(x.get(key), B, nm, key)
        for k in ("riskA", "riskB"):
            if x.get(k) is not None:
                ent[k] = x[k]
        for k in ("eventsA", "eventsB", "nA", "nB", "n_inf_A", "n_inf_B"):
            if k in x:
                ent[k] = sc.mask_n(x[k])
        out["est"][nm] = ent
    for o in OUTS:
        key = "HR" if o == "ic" else "OR"
        out["ratio"][f"{o}_pre_vs_main"] = r11.paired_ratio(B, f"pre_{o}", f"main_{o}", E[f"pre_{o}"][key], E[f"main_{o}"][key])
    for k, _, _ in STRATA:
        out["ratio"][f"strat_{k}"] = r11.paired_ratio(B, f"strat_{k}_1", f"strat_{k}_0", E[f"strat_{k}_1"]["OR"], E[f"strat_{k}_0"]["OR"])
    dsc = est["desc"]
    out["desc"] = {k: (sc.mask_n(v) if isinstance(v, int) else v) for k, v in dsc.items()}
    out["smd_pre_max"] = est.get("smd_pre_max")
    m = out["est"]["main_primary"]
    main_reading = asd.interpret(m["OR"], m["lo"], m["hi"]) if m.get("lo") is not None else "no interval"
    rd = out["reading"]
    rd["main_primary"] = main_reading
    rd["pre_primary"] = read_pre(out["est"]["pre_primary"], main_reading)
    e = out["est"]["pre_primary"]
    rd["pre_primary_rule"] = asd.interpret(e["OR"], e["lo"], e["hi"]) if e.get("lo") is not None else "no interval"
    rd["pre_negctl"] = read_negctl(out["est"]["pre_negctl"])
    for k, _, _ in STRATA:
        rd[f"strat_{k}"] = read_strat(out["ratio"][f"strat_{k}"])
    json.dump(asd.round_floats(asd.to_json(out)), open(os.path.join(R, "r14.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(R, "r14.md"), "w", encoding="utf-8").write(md(out))
    asd.log_run(args.results, "r14 report: ok")
    print("report written", flush=True)


def rci(e, key):
    f2 = r11.f2
    if not e or e.get(key) is None:
        return "—"
    return f"{f2(e[key])} ({f2(e['lo'])}–{f2(e['hi'])})" if e.get("lo") is not None else f2(e[key])


def md(o):
    f2 = r11.f2
    L = ["# Revision R14 results: eICU pre-time-zero covariate model and ventilation strata defined by time stamps (PPI vs famotidine)", "",
         f"Plan: {o['plan']}. Bootstrap: by hospital {o['boot_reps']} replicates (seeds as asd.py; the two models and four strata paired). "
         "Odds ratio <1: lower in the PPI arm. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", "",
         f"Analysis set: PPI {o['flow']['analysis_A']} / famotidine {o['flow']['analysis_B']} (outside overlap {o['flow']['outside_overlap']}).", "",
         "## 1. Pre-time-zero covariate model", "",
         "| Outcome | Model | Estimate (95% CI) | Weighted risk PPI / famotidine |", "|---|---|---|---|"]
    for oc in OUTS:
        key = "HR" if oc == "ic" else "OR"
        for s in SPECS:
            e = o["est"][f"{s}_{oc}"]
            risk = f"{f2(100 * e['riskA'], 1)}% / {f2(100 * e['riskB'], 1)}%" if e.get("riskA") is not None else "—"
            L.append(f"| {OUT_LAB[oc]} | {SPEC_LAB[s]} | {rci(e, key)} | {risk} |")
        r = o["ratio"].get(f"{oc}_pre_vs_main")
        L.append(f"| {OUT_LAB[oc]} | Pre-time-zero ÷ main analysis (paired) | {(f2(r['ratio']) + ' (' + f2(r['lo']) + '–' + f2(r['hi']) + ')') if r else '—'} | |")
    ds = o["desc"]
    L += ["", f"- Columns removed: {', '.join(ds['removed'])}; new columns dropped by the low-prevalence rule in the new model: {', '.join(ds['dropped_pre']) or 'no'}; "
              f"largest |SMD| after overlap weighting with the pre-time-zero model: {f2(o['smd_pre_max'], 3) if o.get('smd_pre_max') is not None else '—'}",
          f"- Lactate measured (24 h before time zero): {ds['lac_measured']} / {ds['n']}",
          f"- Reading (written rule): main analysis {o['reading']['main_primary']}; pre-time-zero model {o['reading']['pre_primary_rule']}; {o['reading']['pre_primary']}",
          f"- Reading (written rule): {o['reading']['pre_negctl']}", "",
          "## 2. Strata of invasive ventilation at time zero (primary outcome)", "",
          "| Definition | Not ventilated: patients PPI / famotidine; OR (95% CI) | Ventilated: patients PPI / famotidine; OR (95% CI) | Ventilated ÷ not ventilated (95% CI) | Reading (written rule) |",
          "|---|---|---|---|---|"]
    for k, _, _ in STRATA:
        a, b, r = o["est"][f"strat_{k}_0"], o["est"][f"strat_{k}_1"], o["ratio"].get(f"strat_{k}")
        L.append(f"| {STRAT_LAB[k]} | {a.get('nA', '—')} / {a.get('nB', '—')}; {rci(a, 'OR')} | {b.get('nA', '—')} / {b.get('nB', '—')}; {rci(b, 'OR')} | "
                 f"{(f2(r['ratio']) + ' (' + f2(r['lo']) + '–' + f2(r['hi']) + ')') if r else '—'} | {o['reading'][f'strat_{k}']} |")
    L += ["", "Cross-tabulation of the two ventilation definitions (both arms together):", "",
          "| | Time-stamped record before time zero | No time-stamped record before time zero |", "|---|---|---|",
          f"| APACHE flag: ventilated | {ds['flag1_pre1']} | {ds['flag1_pre0']} |",
          f"| APACHE flag: not ventilated | {ds['flag0_pre1']} | {ds['flag0_pre0']} |", ""]
    return "\n".join(L)


def synth(n, rng):
    d = r12.synth(n, rng)
    k = len(d["A"])
    sev = rng.normal(size=k)
    apache = 50 + 20 * sev
    flag = rng.random(k) < 0.3
    pre = np.where(rng.random(k) < 0.85, flag, ~flag)
    lac = np.where(rng.random(k) < 0.4, np.nan, np.exp(0.4 * sev + rng.normal(0, 0.4, k)))
    add = np.column_stack([apache >= 40, apache >= 60, apache >= 80, flag]).astype(float)
    d["X"] = np.column_stack([d["X"], add])
    d["names"] = d["names"] + ["apache_ge40", "apache_ge60", "apache_ge80", "vent"]
    return add_pre(d, flag, pre, lac)


def selftest(args):
    res = []

    def check(name, ok, detail=""):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name}{' | ' + detail if detail else ''}", flush=True)

    rng = np.random.default_rng(14)
    X = np.column_stack([np.arange(40) % 2, (np.arange(40) // 2) % 2, np.arange(40) % 3 == 0, np.arange(40) % 3 == 1]).astype(float)
    names = ["age_ge65", "apache_ge40", "vent", "hosp_7"]
    vp = (np.arange(40) % 4) < 2
    lac = np.where(np.arange(40) % 3 == 0, np.nan, np.where(np.arange(40) % 3 == 1, 1.5, 4.5))
    Xp, nm, dr = pre_matrix(X, names, vp, lac)
    check("time-zero model: apache_* and vent removed, four columns added", nm == ["age_ge65", "hosp_7"] + NEW_COLS
          and np.array_equal(Xp[:, 0], X[:, 0]) and np.array_equal(Xp[:, 2], vp.astype(float))
          and np.array_equal(Xp[:, 3], (lac >= 2).astype(float)) and np.array_equal(Xp[:, 5], (~np.isfinite(lac)).astype(float)), str(nm))
    d = synth(20000, rng)
    cache = {}
    r = estimates(d, cache)
    check("no drug effect: main and time-zero models near 1 (primary)", all(abs(math.log(r[f"{s}_primary"]["OR"])) < 0.12 for s in SPECS),
          f"{r['main_primary']['OR']:.3f} / {r['pre_primary']['OR']:.3f}")
    check("no drug effect: four strata finite and near 1", all(np.isfinite(r[f"strat_{k}_{lv}"]["OR"]) and abs(math.log(r[f"strat_{k}_{lv}"]["OR"])) < 0.2
                                                             for k, _, _ in STRATA for lv in (0, 1)),
          " / ".join(f"{r[f'strat_{k}_{lv}']['OR']:.3f}" for k, _, _ in STRATA for lv in (0, 1)))
    check("onset model finite under both models", all(np.isfinite(r[f"{s}_ic"]["HR"]) and r[f"{s}_ic"]["converged"] for s in SPECS))
    m = d["vent_pre"]
    s = asd.subset_d(spec_d(d, "pre"), m)
    s2 = asd.subset_d(d, d["vent_flag"])
    check("stratifier dropped within strata; the time-stamped strata use the time-zero model",
          "vent_pre" not in s["names"] and "vent" not in s2["names"] and "lactate_ge2" in s["names"] and "apache_ge40" in s2["names"]
          and "apache_ge40" not in s["names"])
    e1 = asd.cif_contrast(s2, explore_ow(s2), s2["t_del"])[168]["OR"]
    check("stratum estimate equals E1's recipe", abs(r["strat_flag_1"]["OR"] - e1) < 1e-6, f"{r['strat_flag_1']['OR']:.6f} vs {e1:.6f}")
    b = r12.draw(d, "hosp", 7)
    ref = asd.resample(d, "e", np.random.default_rng(asd.SEED + 7))
    check("bootstrap draws equal asd.resample (by hospital), including the new arrays", np.array_equal(b["A"], ref["A"])
          and np.array_equal(b["X_pre"], ref["X_pre"]) and np.array_equal(b["vent_pre"], ref["vent_pre"]) and np.array_equal(b["site"], ref["site"]))
    _G.update(d=d, cache=cache)
    v = _one(3)[1]
    check("replicate values in the declared order and reproducible", v is not None and len(v) == len(NAMES)
          and np.allclose(v, flat(estimates(r12.draw(d, "hosp", 3), dict(cache))), equal_nan=True))
    check("readings follow the written rules",
          read_pre({"OR": 0.98, "lo": 0.90, "hi": 1.15}, "clinically important benefit excluded (precise negative result)").startswith("the eICU estimate does not depend")
          and read_pre({"OR": 0.85, "lo": 0.76, "hi": 0.95}, "clinically important benefit excluded (precise negative result)").endswith("changed")
          and read_pre({"OR": 1.00, "lo": 0.70, "hi": 1.30}, "clinically important benefit excluded (precise negative result)").endswith("changed")
          and read_negctl({"OR": 1.2, "lo": 1.05, "hi": 1.4}) == "negative control suggests residual confounding" and read_negctl({"OR": 1.0, "lo": 0.9, "hi": 1.1}).startswith("negative control does not")
          and read_strat({"ratio": 1.4, "lo": 1.05, "hi": 2.0}).startswith("higher odds ratio in ventilated patients") and read_strat({"ratio": 1.2, "lo": 0.9, "hi": 1.6}).endswith("inconclusive"))
    import contextlib
    import io
    tmp = tempfile.mkdtemp(prefix="r14_self_")
    try:
        P, R = os.path.join(tmp, "p"), os.path.join(tmp, "r")
        os.makedirs(P)
        os.makedirs(R)
        small = synth(4000, rng)
        nn = len(small["A"])
        flow = {"eligible": nn, "A": int(small["A"].sum()), "B": int((small["A"] == 0).sum()), "other": 0, "outside_overlap": 0,
                "analysis_A": int(small["A"].sum()), "analysis_B": int((small["A"] == 0).sum())}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            est, cache = core_estimate(small, flow)
            json.dump(est, open(os.path.join(P, "r14_est.json"), "w"))
            _G.update(d=small, cache=cache)
            V = np.array([_one(rep)[1] for rep in range(60)])
            np.savez(os.path.join(P, "r14boot_0000_0060.npz"), reps=np.arange(60), v=V, names=np.array(NAMES))
            report(argparse.Namespace(private=P, results=R))
        J = json.load(open(os.path.join(R, "r14.json"), encoding="utf-8"))
        mdtxt = open(os.path.join(R, "r14.md"), encoding="utf-8").read()
        check("end to end: report written with both sections and every estimate", "## 1." in mdtxt and "## 2." in mdtxt
              and all(J["est"][nm].get("lo") is not None for nm in NAMES))
        cnt = [J["flow"][k] for k in J["flow"]] + [J["desc"][k] for k in ("flag1_pre1", "flag1_pre0", "flag0_pre1", "flag0_pre0", "lac_measured", "n")] \
            + [J["est"][nm][k] for nm in NAMES for k in ("eventsA", "eventsB", "nA", "nB") if k in J["est"][nm]]
        check("end to end: counts in the JSON are masked", bool(cnt) and all(isinstance(v, str) and (v == "<10" or int(v) % 5 == 0) for v in cnt),
              f"{len(cnt)} counts")
        check("end to end: every written reading present", all(k in J["reading"] for k in ("main_primary", "pre_primary", "pre_primary_rule",
                                                                                          "pre_negctl", "strat_flag", "strat_pre")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(res)}/{len(res)} passed", flush=True)
    sys.exit(0 if all(res) else 1)


def explore_ow(d):
    ps, _ = sc.ps_fit(d["X"], d["A"])
    return sc.overlap_w(ps, d["A"])


def bindcheck(args):
    import duckdb
    res = []
    con = duckdb.connect()
    con.execute("CREATE SCHEMA eicu")
    src = connect_full(tempfile.mkdtemp(prefix="r14_bind_"))
    for tb in ("respiratorycharting", "treatment", "respiratorycare", "lab"):
        cols = src.execute(f"DESCRIBE SELECT * FROM eicu.{tb}").fetchall()
        con.execute(f"CREATE TABLE eicu.{tb} AS SELECT * FROM (SELECT {', '.join(f'CAST(NULL AS {c[1]}) AS {c[0]}' for c in cols)}) WHERE false")
    src.close()
    tmp = tempfile.mkdtemp(prefix="r14_bind_")
    f = os.path.join(tmp, "c.parquet")
    con.execute(f"COPY (SELECT CAST(1 AS BIGINT) AS stay_id, CAST(100 AS BIGINT) AS t0, 'A' AS arm) TO '{f}' (FORMAT parquet)")
    try:
        cur = con.execute(f"SELECT * FROM ({PRE_E.format(f=f)}) q LIMIT 0")
        cols = [c[0] for c in cur.description]
        rows = con.execute(PRE_E.format(f=f)).fetchall()
        ok = " ".join(cols) == EXPECT and rows == [(1, False, None)]
        res.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} | extraction statement binds; columns {' '.join(cols)}; one cohort row, no records -> {rows}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(res)}/{len(res)} passed", flush=True)
    sys.exit(0 if all(res) else 1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    sp.add_parser("bindcheck")
    subs = {c: sp.add_parser(c) for c in ("extract", "estimate", "boot", "report")}
    for s in subs.values():
        s.add_argument("--private", required=True)
        s.add_argument("--results", required=True)
    subs["boot"].add_argument("--start", type=int, required=True)
    subs["boot"].add_argument("--stop", type=int, required=True)
    subs["boot"].add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"selftest": selftest, "bindcheck": bindcheck, "extract": extract, "estimate": estimate, "boot": boot, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
