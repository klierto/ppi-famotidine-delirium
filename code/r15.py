#!/usr/bin/env python3
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import glob
import io
import json
import math
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asd
import stats_core as sc
import r11
import r12
import explore_ppi as e1

PLAN = "SAP_addendum_R15.md v1.0"
NBOOT = 2000
DBS = ("m", "e")
DBNAME = {"m": "MIMIC-IV", "e": "eICU"}
STRATA = ("light", "moderate", "deep", "missing")
UNITS = ("rass3",) + STRATA
OUTS = ("primary", "negctl")
NAMES = [f"{u}_{o}" for u in UNITS for o in OUTS]
POOLED = ("light_negctl", "light_primary", "rass3_negctl", "rass3_primary")
UPPER = 1.0 / asd.MCID
UNIT_LAB = {"rass3": "Prespecified subgroup: RASS ≥ −3 at time zero (patients with a record)", "light": "Awake or lightly sedated (RASS ≥ −1)",
            "moderate": "Moderately sedated (RASS −3 to −2)", "deep": "Deeply sedated (RASS ≤ −4)", "missing": "RASS not recorded at time zero"}
OUT_LAB = {"primary": "Delirium (as the main analysis, 168 h)", "negctl": "Negative control: first glucose <70 mg/dL (168 h)"}


def rass_group(r):
    r = np.asarray(r, float)
    return np.where(~np.isfinite(r), "missing", np.where(r <= -4, "deep", np.where(r <= -2, "moderate", "light")))


def load(P, db):
    df, flow = asd.load_key(P, "c2" + db, stage2=True)
    d = asd.build(df, db, 2, stage2=True)
    d["rgrp"] = rass_group(asd.num(df, "rass"))
    return d, flow


def unit_mask(d, u):
    return d["sub"]["rass3"] if u == "rass3" else d["rgrp"] == u


def unit_est(d, u, cache):
    m = unit_mask(d, u)
    A = d["A"]
    nA, nB = int((m & (A == 1)).sum()), int((m & (A == 0)).sum())
    if min(nA, nB) < e1.MIN_N:
        return {o: {"OR": float("nan"), "nA": nA, "nB": nB} for o in OUTS}
    s = asd.subset_d(d, m)
    w = r11.ow_of(s, cache, "rass3" if u == "rass3" else f"r15_{u}")
    out = {}
    for o, t in (("primary", s["t_del"]), ("negctl", s["t_hypo"])):
        e = r11.cif(s, w, t)
        e["nA"], e["nB"] = nA, nB
        out[o] = e
    return out


def estimates(d, cache):
    res = {}
    for u in UNITS:
        for o, e in unit_est(d, u, cache).items():
            res[f"{u}_{o}"] = e
    return res


def flat(res):
    vals = []
    for nm in NAMES:
        x = res[nm].get("OR")
        vals.append(math.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan)
    return np.array(vals)


def core_estimate(d, flow):
    cache = {}
    w = r11.ow_of(d, cache, "full")
    main = {"primary": r11.cif(d, w, d["t_del"]), "negctl": r11.cif(d, w, d["t_hypo"])}
    res = estimates(d, cache)
    shares = {u: int((d["rgrp"] == u).sum()) for u in STRATA}
    return {"flow": flow, "main": asd.to_json(main), "res": asd.to_json(res), "n_by_stratum": shares, "n": int(len(d["A"]))}, cache


def estimate(args):
    t0 = time.time()
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    c2 = json.load(open(os.path.join(R, "ppi_estimates.json"), encoding="utf-8"))["per_db"]
    j1 = json.load(open(os.path.join(R, "ppi_explore_E1.json"), encoding="utf-8"))["db"]
    out = {"plan": PLAN, "db": {}, "checks": []}
    caches = {}
    good = True
    for db in DBS:
        d, flow = load(P, db)
        est, cache = core_estimate(d, flow)
        a = c2[db]["analyses"]
        checks = [(f"{DBNAME[db]} primary OR, main model, vs ppi_estimates.json", est["main"]["primary"]["OR"], a["primary"]["OR"], 1e-9),
                  (f"{DBNAME[db]} negative-control OR, main model, vs ppi_estimates.json", est["main"]["negctl"]["OR"], a["negctl"]["OR"], 1e-9),
                  (f"{DBNAME[db]} RASS >= -3 subgroup, delirium OR, vs ppi_estimates.json", est["res"]["rass3_primary"]["OR"], a["rass3"]["OR"], 1e-9)]
        for u in STRATA:
            checks.append((f"{DBNAME[db]} RASS stratum {u}, delirium OR, vs E1", est["res"][f"{u}_primary"]["OR"],
                           float(j1[db]["strata"]["rass"][u]["OR"]), 5e-4))
        for name, x, ref, tol in checks:
            ok = x is not None and abs(round(x, 4) - ref) < tol + 1e-12
            good &= ok
            line = f"{'PASS' if ok else 'FAIL'} | {name}: {x:.4f} vs {ref:.4f}" if x is not None else f"FAIL | {name}: no estimate"
            out["checks"].append(line)
            print(line, flush=True)
        out["db"][db] = est
        caches[db] = cache
        asd.status(f"{DBNAME[db]} estimated", t0)
    if not good:
        asd.log_run(args.results, "r15 estimate: REPRODUCTION CHECK FAILED")
        raise SystemExit("the rebuilt data differ from R6/E1: stop")
    json.dump(out, open(os.path.join(P, "r15_est.json"), "w"))
    for db, cache in caches.items():
        np.savez(os.path.join(P, f"r15_cache_{db}.npz"), **{k.replace(":", "__"): v for k, v in cache.items()})
    asd.status("r15 estimate done; reproduction checks passed", t0)
    asd.log_run(args.results, f"r15 estimate: ok, reproduction checks passed ({time.time() - t0:.0f} s)")


_G = {}


def _one(rep):
    d, db, cache = _G["d"], _G["db"], _G["cache"]
    try:
        b = asd.resample(d, db, np.random.default_rng(asd.SEED + rep))
        return rep, flat(estimates(b, dict(cache)))
    except Exception:
        return rep, None


def run_reps(reps, workers):
    out = {}
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for rep, v in pool.imap_unordered(_one, reps, chunksize=5):
                out[rep] = v
    else:
        for rep in reps:
            out[rep] = _one(rep)[1]
    V = np.full((len(reps), len(NAMES)), np.nan)
    for i, rep in enumerate(reps):
        if out.get(rep) is not None:
            V[i] = out[rep]
    return V, sum(1 for r in reps if out.get(r) is None)


def boot(args):
    t0 = time.time()
    P = os.path.expanduser(args.private)
    db = args.db
    d, _ = load(P, db)
    z = np.load(os.path.join(P, f"r15_cache_{db}.npz"))
    _G.update(d=d, db=db, cache={k.replace("__", ":"): z[k] for k in z.files})
    reps = list(range(args.start, min(args.stop, NBOOT)))
    V, failed = run_reps(reps, args.workers)
    np.savez(os.path.join(P, f"r15boot_{db}_{args.start:04d}_{args.stop:04d}.npz"), reps=np.array(reps), v=V, names=np.array(NAMES))
    asd.status(f"r15 boot {db} {args.start}-{args.stop} done, failed {failed}", t0)
    asd.log_run(args.results, f"r15 boot --db {db} --start {args.start} --stop {args.stop}: ok, failed {failed} ({time.time() - t0:.0f} s)")


def load_boots(P, db):
    files = sorted(glob.glob(os.path.join(P, f"r15boot_{db}_*.npz")))
    if not files:
        return None
    reps, vs, names = [], [], None
    for f in files:
        zz = np.load(f)
        reps.append(zz["reps"])
        vs.append(zz["v"])
        names = [str(x) for x in zz["names"]]
    reps = np.concatenate(reps)
    _, first = np.unique(reps, return_index=True)
    return {"names": names, "v": np.concatenate(vs)[first], "n": int(first.size)}


def has_ci(e):
    return bool(e) and e.get("lo") is not None and e.get("hi") is not None


def nc_flag(e):
    return e["lo"] > 1 or e["hi"] < 1


def nc_precise(e):
    return e["lo"] > asd.MCID and e["hi"] < UPPER


def read_nc(e):
    if not has_ci(e):
        return "no interval"
    if nc_flag(e):
        return "negative control suggests residual confounding"
    if nc_precise(e):
        return "negative control does not suggest residual confounding, and excludes a bias as large as the minimal clinically important difference"
    return "negative control does not suggest residual confounding, but lacks the precision to exclude a bias as large as the minimal clinically important difference"


def read_question(em, ee):
    if not (has_ci(em) and has_ci(ee)):
        return "inconclusive (no interval)"
    pair = (("MIMIC-IV", em), ("eICU", ee))
    flagged = [f"{n} ({'higher in the PPI arm' if e['lo'] > 1 else 'lower in the PPI arm'})" for n, e in pair if nc_flag(e)]
    if flagged:
        return "the agreement in the awake or lightly sedated stratum may come partly from residual confounding: " + ", ".join(flagged) + " negative control suggests residual confounding"
    txt = "the agreement of the delirium estimates of the two databases in the awake or lightly sedated stratum is not accompanied by a negative-control signal"
    loose = [n for n, e in pair if not nc_precise(e)]
    if loose:
        txt += " (" + ", ".join(loose) + " negative control lacks the precision to exclude a bias as large as the minimal clinically important difference)"
    return txt


def read_pooled_nc(p):
    if not p:
        return "no pooled estimate"
    if p["I2"] > 50:
        return "the databases disagree; pooled estimate for reference only"
    return read_nc(p)


def read_pooled_del(p):
    if not p:
        return "no pooled estimate"
    if p["I2"] > 50:
        return "the databases disagree; pooled estimate for reference only"
    return asd.interpret(p["OR"], p["lo"], p["hi"])


def report(args):
    P, R = os.path.expanduser(args.private), os.path.expanduser(args.results)
    est = json.load(open(os.path.join(P, "r15_est.json")))
    j1 = json.load(open(os.path.join(R, "ppi_explore_E1.json"), encoding="utf-8"))["db"]
    out = {"plan": PLAN, "checks": est["checks"], "boot_reps": {}, "flow": {}, "n_by_stratum": {}, "est": {}, "pooled": {},
           "reading": {}, "e1_light_ci": {}}
    for db in DBS:
        B = load_boots(P, db)
        E = est["db"][db]
        out["boot_reps"][db] = B["n"] if B else 0
        out["flow"][db] = {k: sc.mask_n(v) for k, v in E["flow"].items()}
        out["n_by_stratum"][db] = {k: sc.mask_n(v) for k, v in E["n_by_stratum"].items()}
        out["est"][db] = {}
        for nm in NAMES:
            x = E["res"][nm]
            ent = r11.with_ci(x.get("OR"), B, nm, "OR")
            for k in ("riskA", "riskB"):
                if x.get(k) is not None:
                    ent[k] = x[k]
            for k in ("eventsA", "eventsB", "nA", "nB"):
                if k in x:
                    ent[k] = sc.mask_n(x[k])
            out["est"][db][nm] = ent
        out["est"][db]["main_negctl"] = {"OR": E["main"]["negctl"]["OR"]}
        rd = out["reading"].setdefault(db, {})
        for u in UNITS:
            rd[f"{u}_negctl"] = read_nc(out["est"][db][f"{u}_negctl"])
            p = out["est"][db][f"{u}_primary"]
            rd[f"{u}_primary"] = asd.interpret(p["OR"], p["lo"], p["hi"]) if has_ci(p) else "no interval"
        e1l = j1[db]["strata"]["rass"]["light"]
        out["e1_light_ci"][db] = {"E1": [float(e1l["OR"])] + [float(v) for v in e1l["CI"]],
                                  "R15": [out["est"][db]["light_primary"].get(k) for k in ("OR", "lo", "hi")]}
    for nm in POOLED:
        out["pooled"][nm] = r11.pool2(out["est"]["m"][nm], out["est"]["e"][nm])
    rq = out["reading"]
    rq["question"] = read_question(out["est"]["m"]["light_negctl"], out["est"]["e"]["light_negctl"])
    rq["pooled_light_negctl"] = read_pooled_nc(out["pooled"]["light_negctl"])
    rq["pooled_rass3_negctl"] = read_pooled_nc(out["pooled"]["rass3_negctl"])
    rq["pooled_light_primary"] = read_pooled_del(out["pooled"]["light_primary"])
    rq["pooled_rass3_primary"] = read_pooled_del(out["pooled"]["rass3_primary"])
    json.dump(asd.round_floats(asd.to_json(r11.mask_tree(out))), open(os.path.join(R, "r15.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    open(os.path.join(R, "r15.md"), "w", encoding="utf-8").write(md(out))
    asd.log_run(args.results, "r15 report: ok")
    print("report written", flush=True)


def rci(e, key="OR"):
    f2 = r11.f2
    if not e or e.get(key) is None:
        return "—"
    return f"{f2(e[key])} ({f2(e['lo'])}–{f2(e['hi'])})" if e.get("lo") is not None else f2(e[key])


def risks(e):
    f2 = r11.f2
    if e.get("riskA") is None:
        return "—"
    return f"{f2(100 * e['riskA'], 1)}% / {f2(100 * e['riskB'], 1)}%"


def pooled_txt(p, key="OR"):
    if not p:
        return "—"
    f2 = r11.f2
    return f"{f2(p[key])} ({f2(p['lo'])}–{f2(p['hi'])}); I² {p['I2']:.0f}%"


def md(o):
    E, rd = o["est"], o["reading"]
    L = ["# Revision R15 results: the negative control by RASS stratum at time zero (PPI vs famotidine; MIMIC-IV, eICU)", "",
         f"Plan: {o['plan']}. Bootstrap: MIMIC-IV by patient {o['boot_reps']['m']} replicates, eICU by hospital {o['boot_reps']['e']} replicates (seeds as asd.py; "
         "in every replicate the five units estimated in the same sample). Odds ratio <1: lower in the PPI arm. Propensity model refitted within strata, overlap weighting, ICU discharge alive and death as "
         "competing events, 168 h. Counts masked: 1–9 as <10, otherwise the nearest multiple of 5.", "",
         "## 1. Main question: awake or lightly sedated at time zero (RASS ≥ −1)", "",
         "| Database | Patients PPI / famotidine | Delirium OR (95% CI) | Negative control OR (95% CI) | Negative control weighted risk PPI / famotidine | Negative control reading (written rule) |",
         "|---|---|---|---|---|---|"]
    for db in DBS:
        p, n = E[db]["light_primary"], E[db]["light_negctl"]
        L.append(f"| {DBNAME[db]} | {p.get('nA', '—')} / {p.get('nB', '—')} | {rci(p)} | {rci(n)} | {risks(n)} | {rd[db]['light_negctl']} |")
    L.append(f"| Pooled (fixed effect) | — | {pooled_txt(o['pooled']['light_primary'])} | {pooled_txt(o['pooled']['light_negctl'])} | — | "
             f"{rd['pooled_light_negctl']} |")
    L += ["", f"- Reading (written rule), main question: {rd['question']}",
          f"- Reading (written rule), pooled delirium: {rd['pooled_light_primary']}",
          f"- For comparison: negative control in the whole cohort MIMIC-IV {r11.f2(E['m']['main_negctl']['OR'])}, eICU {r11.f2(E['e']['main_negctl']['OR'])} (R6).",
          "- Reproduction: delirium in the awake or light stratum against E1 " + "; ".join(
              f"{DBNAME[db]} E1 {r11.f2(o['e1_light_ci'][db]['E1'][0])} ({r11.f2(o['e1_light_ci'][db]['E1'][1])}–{r11.f2(o['e1_light_ci'][db]['E1'][2])}), "
              f"R15 {rci(E[db]['light_primary'])}" for db in DBS), "",
          "## 2. Other strata and the prespecified RASS ≥ −3 subgroup", "",
          "| Database | Unit | Patients PPI / famotidine | Delirium OR (95% CI) | Delirium reading (SAP section 8) | Negative control OR (95% CI) | Negative control reading (written rule) |",
          "|---|---|---|---|---|---|---|"]
    for db in DBS:
        for u in UNITS:
            if u == "light":
                continue
            p, n = E[db][f"{u}_primary"], E[db][f"{u}_negctl"]
            L.append(f"| {DBNAME[db]} | {UNIT_LAB[u]} | {p.get('nA', '—')} / {p.get('nB', '—')} | {rci(p)} | {rd[db][f'{u}_primary']} | "
                     f"{rci(n)} | {rd[db][f'{u}_negctl']} |")
    L += ["", f"- RASS ≥ −3 subgroup pooled: delirium {pooled_txt(o['pooled']['rass3_primary'])} ({rd['pooled_rass3_primary']}); "
              f"negative control {pooled_txt(o['pooled']['rass3_negctl'])} ({rd['pooled_rass3_negctl']})", "",
          "## 3. Reproduction checks (estimate, before any new estimate)", ""] + [f"- {c}" for c in o["checks"]] + [""]
    L += ["Analysis set: " + "; ".join(f"{DBNAME[db]} PPI {o['flow'][db]['analysis_A']} / famotidine {o['flow'][db]['analysis_B']}" for db in DBS)
          + ". Patients in each RASS stratum (both arms): " + "; ".join(
              f"{DBNAME[db]} " + ", ".join(f"{u} {o['n_by_stratum'][db][u]}" for u in STRATA) for db in DBS) + ".", ""]
    return "\n".join(L)


RASS_P = (0.08, 0.12, 0.06, 0.06, 0.28, 0.32, 0.06, 0.02)


def synth(n, rng, plant=0.0):
    d = r12.synth(n, rng)
    k = len(d["A"])
    rass = rng.choice(np.arange(-5, 3), size=k, p=RASS_P).astype(float)
    rass[rng.random(k) < 0.15] = np.nan
    d["rgrp"] = rass_group(rass)
    d["sub"] = {"rass3": np.isfinite(rass) & (rass >= -3)}
    ind = np.column_stack([rass <= -4, (rass >= -3) & (rass <= -2), rass >= 1, ~np.isfinite(rass)]).astype(float)
    d["X"] = np.column_stack([d["X"], ind])
    d["names"] = d["names"] + ["rass_le_m4", "rass_m3_m2", "rass_ge1", "rass_na"]
    if plant > 0:
        deep = d["rgrp"] == "deep"
        z = rng.normal(size=k)
        a_new = (rng.random(k) < 1 / (1 + np.exp(-(0.2 + plant * z)))).astype(int)
        d["A"] = np.where(deep, a_new, d["A"])
        hypo = rng.exponential(1.0, k) / (0.003 * np.exp(plant * z))
        d["t_hypo"] = np.where(deep, np.where(hypo <= d["t_end"], hypo, np.nan), d["t_hypo"])
    return d


def e1_recipe(d, m):
    s = asd.subset_d(d, m)
    ps, _ = sc.ps_fit(s["X"], s["A"])
    w = sc.overlap_w(ps, s["A"])
    return [asd.cif_contrast(s, w, s[t])[168]["OR"] for t in ("t_del", "t_hypo")]


def selftest(args):
    res = []

    def check(name, ok, detail=""):
        res.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'} | {name}{' | ' + detail if detail else ''}", flush=True)

    rng = np.random.default_rng(15)
    r = np.array([np.nan, -5, -4, -3, -2, -1, 0, 1, 4], float)
    g = rass_group(r)
    check("RASS groups follow E1 (<= -4 deep, -3 to -2 moderate, >= -1 light, missing)",
          list(g) == ["missing", "deep", "deep", "moderate", "moderate", "light", "light", "light", "light"]
          and list(e1.LEVEL_CODE["rass"]) == list(STRATA) and e1.MIN_N == r11.MIN_N == 20, " ".join(g))
    d = synth(40000, rng)
    cache = {}
    est = estimates(d, cache)
    check("no drug effect: delirium and negative control near 1 in every unit",
          all(np.isfinite(est[nm]["OR"]) and abs(math.log(est[nm]["OR"])) < 0.3 for nm in NAMES),
          " / ".join(f"{nm} {est[nm]['OR']:.2f}" for nm in NAMES))
    ok3 = all(abs(est[f"{u}_{o}"]["OR"] - ref) < 1e-6 for u in STRATA
              for o, ref in zip(OUTS, e1_recipe(d, d["rgrp"] == u)))
    ds = asd.subset_d(d, d["sub"]["rass3"])
    ref3 = asd.cif_contrast(ds, sc.overlap_w(asd.ow_fit(ds, {}, "rass3"), ds["A"]), ds["t_del"])[168]["OR"]
    check("strata follow E1's recipe and the RASS >= -3 subgroup asd.py's (same weights for both outcomes)",
          ok3 and abs(est["rass3_primary"]["OR"] - ref3) < 1e-9, f"rass3 {est['rass3_primary']['OR']:.6f} vs {ref3:.6f}")
    sl = asd.subset_d(d, d["rgrp"] == "light")
    sdp = asd.subset_d(d, d["rgrp"] == "deep")
    check("constant RASS columns dropped within strata", not any(c in sl["names"] for c in ("rass_le_m4", "rass_m3_m2", "rass_na"))
          and "rass_ge1" in sl["names"] and not any(c.startswith("rass_") for c in sdp["names"]), f"{sl['names'][-3:]}")
    dp = synth(24000, rng, plant=1.0)
    ep = estimates(dp, {})
    check("planted bias: negative control raised in the deep stratum only",
          ep["deep_negctl"]["OR"] > 1.25 and abs(math.log(ep["light_negctl"]["OR"])) < 0.15,
          f"deep {ep['deep_negctl']['OR']:.2f}, light {ep['light_negctl']['OR']:.2f}")
    _G.update(d=d, db="e", cache=cache)
    v1, v2 = _one(4)[1], _one(4)[1]
    ref = flat(estimates(asd.resample(d, "e", np.random.default_rng(asd.SEED + 4)), dict(cache)))
    _G.update(db="m")
    vm = _one(4)[1]
    refm = flat(estimates(asd.resample(d, "m", np.random.default_rng(asd.SEED + 4)), dict(cache)))
    check("replicates reproducible, by hospital (e) and by patient (m), in the declared order",
          v1 is not None and len(v1) == len(NAMES) and np.allclose(v1, v2, equal_nan=True) and np.allclose(v1, ref, equal_nan=True)
          and vm is not None and np.allclose(vm, refm, equal_nan=True) and not np.allclose(vm, v1, equal_nan=True))
    flag_hi, flag_lo = {"OR": 1.3, "lo": 1.05, "hi": 1.6}, {"OR": 0.8, "lo": 0.66, "hi": 0.97}
    precise, loose = {"OR": 1.0, "lo": 0.85, "hi": 1.20}, {"OR": 1.05, "lo": 0.70, "hi": 1.55}
    check("readings follow the written rules",
          read_nc(flag_hi) == "negative control suggests residual confounding" and read_nc(flag_lo) == "negative control suggests residual confounding"
          and read_nc(precise).endswith("and excludes a bias as large as the minimal clinically important difference") and read_nc(loose).endswith("but lacks the precision to exclude a bias as large as the minimal clinically important difference")
          and read_nc({"OR": 1.0, "lo": 0.76, "hi": 1.34}).endswith("lacks the precision to exclude a bias as large as the minimal clinically important difference")
          and read_question(precise, precise) == "the agreement of the delirium estimates of the two databases in the awake or lightly sedated stratum is not accompanied by a negative-control signal"
          and read_question(precise, loose).endswith(" (eICU negative control lacks the precision to exclude a bias as large as the minimal clinically important difference)")
          and read_question(flag_hi, precise).startswith("the agreement in the awake or lightly sedated stratum may come partly from residual confounding: MIMIC-IV (higher in the PPI arm)")
          and read_pooled_nc({"OR": 1.0, "lo": 0.9, "hi": 1.1, "I2": 60.0}) == "the databases disagree; pooled estimate for reference only"
          and read_pooled_nc({"OR": 1.0, "lo": 0.9, "hi": 1.1, "I2": 10.0}).startswith("negative control does not suggest residual confounding, and excludes")
          and read_pooled_del({"OR": 0.70, "lo": 0.63, "hi": 0.78, "I2": 0.0}) == asd.interpret(0.70, 0.63, 0.78))
    tmp = tempfile.mkdtemp(prefix="r15_self_")
    try:
        P, R = os.path.join(tmp, "p"), os.path.join(tmp, "r")
        os.makedirs(P)
        os.makedirs(R)
        e1json = {"db": {}}
        out = {"plan": PLAN, "db": {}, "checks": ["PASS | synthetic"]}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for db in DBS:
                small = synth(5000, rng)
                n = len(small["A"])
                flow = {"eligible": n, "A": int(small["A"].sum()), "B": int((small["A"] == 0).sum()), "other": 0, "outside_overlap": 0,
                        "analysis_A": int(small["A"].sum()), "analysis_B": int((small["A"] == 0).sum())}
                est_db, cache_db = core_estimate(small, flow)
                out["db"][db] = est_db
                _G.update(d=small, db=db, cache=cache_db)
                V, _ = run_reps(list(range(60)), 1)
                np.savez(os.path.join(P, f"r15boot_{db}_0000_0060.npz"), reps=np.arange(60), v=V, names=np.array(NAMES))
                e1json["db"][db] = {"strata": {"rass": {"light": {"OR": est_db["res"]["light_primary"]["OR"], "CI": [0.5, 1.5]}}}}
            json.dump(out, open(os.path.join(P, "r15_est.json"), "w"))
            json.dump(e1json, open(os.path.join(R, "ppi_explore_E1.json"), "w"))
            report(argparse.Namespace(private=P, results=R))
        J = json.load(open(os.path.join(R, "r15.json"), encoding="utf-8"))
        mdtxt = open(os.path.join(R, "r15.md"), encoding="utf-8").read()
        check("end to end: report written with all sections and an interval for every estimate",
              all(s in mdtxt for s in ("## 1.", "## 2.", "## 3.")) and all(J["est"][db][nm].get("lo") is not None for db in DBS for nm in NAMES)
              and all(J["pooled"][nm] for nm in POOLED))
        cnt = [J["flow"][db][k] for db in DBS for k in J["flow"][db]] + [J["n_by_stratum"][db][u] for db in DBS for u in STRATA] \
            + [J["est"][db][nm][k] for db in DBS for nm in NAMES for k in ("eventsA", "eventsB", "nA", "nB") if k in J["est"][db][nm]]
        check("end to end: counts in the JSON are masked", bool(cnt) and all(isinstance(v, str) and (v == "<10" or int(v) % 5 == 0) for v in cnt),
              f"{len(cnt)} counts")
        check("end to end: every written reading present",
              all(f"{u}_{o}" in J["reading"][db] for db in DBS for u in UNITS for o in OUTS)
              and all(k in J["reading"] for k in ("question", "pooled_light_negctl", "pooled_rass3_negctl", "pooled_light_primary",
                                                   "pooled_rass3_primary")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(res)}/{len(res)} passed", flush=True)
    sys.exit(0 if all(res) else 1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    subs = {c: sp.add_parser(c) for c in ("estimate", "boot", "report")}
    for s in subs.values():
        s.add_argument("--private", required=True)
        s.add_argument("--results", required=True)
    subs["boot"].add_argument("--db", choices=DBS, required=True)
    subs["boot"].add_argument("--start", type=int, required=True)
    subs["boot"].add_argument("--stop", type=int, required=True)
    subs["boot"].add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"selftest": selftest, "estimate": estimate, "boot": boot, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
